"""Single-process HTTP caching proxy for fixed-origin package downloads.

FastAPI (HTTP surface) + aiohttp (origin client). One uvicorn worker, all
state in this process; completed packages live on disk under CACHE_DIR.

Design notes
------------
* Single-flight: one fill task per path generation; concurrent misses attach
  to the same fill and read the growing ``.part-<gen>`` file from offset 0.
* Slow clients never block the fill or other clients: the fill only writes to
  disk; every reader tails the file at its own pace. Per-client memory is one
  read buffer, independent of package size.
* Resume via single-range ``Range`` (``bytes=start-end`` / ``bytes=start-``).
  The origin ``X-Package-Sha256`` digest is exposed on every 200/206 as a
  strong ``ETag``; clients echo it in ``If-Range``. A matching validator gets
  an exact 206 with ``Content-Range``; a stale (or missing) validator gets a
  full 200 so the client restarts from the new version. Unsatisfiable ranges
  answer 416 with ``Content-Range: bytes */<total>``.
* Range requests are only served from a finished generation: on a miss the
  request waits for the single-flight fill to complete before answering, and
  a failed fill produces an error response with zero package bytes.
* Purge unregisters the current generation. In-flight fills keep feeding
  their existing readers (including resumes that attached before the purge)
  but are never re-registered; new requests fill and read a fresh generation.
  Old files are unlinked once the last reader detaches.
* Origin failure fails every attached download and never leaves a cached
  half-package. Startup deletes leftover ``.part-*`` files and keeps complete
  packages.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import aiohttp
from fastapi import FastAPI, Header
from fastapi.responses import PlainTextResponse, StreamingResponse

ORIGIN_BASE_URL = os.environ.get("ORIGIN_BASE_URL", "http://127.0.0.1:9000").rstrip("/")
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "./cache")).resolve()
CHUNK = 256 * 1024  # bytes per origin read / client write / disk read

_SEGMENT_RE = re.compile(r"\A[A-Za-z0-9._~!$&'()*+,;=:@-]+\Z")


def validate_key(raw: str) -> str:
    """Map a URL path to a safe relative cache key; reject traversal."""
    if not raw or raw.endswith("/"):
        raise ValueError("empty package path")
    parts = raw.split("/")
    for seg in parts:
        if seg in ("", ".", ".."):
            raise ValueError(f"illegal path segment: {seg!r}")
        if not _SEGMENT_RE.match(seg):
            raise ValueError(f"illegal characters in segment: {seg!r}")
    return "/".join(parts)


def _unlink_quiet(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def strong_etag(sha256: str | None) -> str | None:
    """Build a strong validator (quoted string) from the origin digest."""
    return f'"{sha256}"' if sha256 else None


_RANGE_RE = re.compile(r"bytes=(\d+)-(\d*)")


class RangeUnsatisfiable(Exception):
    """The requested first byte lies past the end of the object."""

    def __init__(self, size: int) -> None:
        self.size = size
        super().__init__(f"range not satisfiable; total size {size}")


def parse_single_range(header: str, size: int) -> tuple[int, int] | None:
    """Parse one inclusive byte range: ``bytes=start-end`` or ``bytes=start-``.

    Returns ``(start, end)`` clipped to the object. Anything else (missing,
    non-bytes unit, multiple ranges, non-numeric, inverted) is unsupported and
    yields ``None`` so the caller answers with the full representation.
    """
    m = _RANGE_RE.fullmatch(header.strip())
    if not m:
        return None
    start_s, end_s = m.group(1), m.group(2)
    try:
        start = int(start_s)
        end = int(end_s) if end_s else size - 1  # "" => open-ended (to last byte)
    except ValueError:
        return None
    if start < 0:
        return None
    if start >= size:
        raise RangeUnsatisfiable(size)
    if end < start:
        return None
    return start, min(end, size - 1)


class FillFailed(Exception):
    """Raised inside a client stream when the origin fill failed."""


class Entry:
    """One generation of a cached object.

    state: filling -> complete          (registered, file renamed into place)
                 \-> closed             (purged mid-fill; fill finished, file
                                          kept only until readers drain)
                 \-> failed             (origin error; part file removed)
    """

    def __init__(self, key: str) -> None:
        self.key = key
        self.gen = uuid.uuid4().hex[:12]
        self.state = "filling"
        self.purged = False
        self.readers = 0
        self.written = 0          # bytes flushed to the part file so far
        self.size = -1            # expected total bytes (-1 = not known yet)
        self.error: str | None = None
        self.origin_status: int | None = None
        self.headers_known = False
        self.meta: dict = {}
        self.cond = asyncio.Condition()
        self.fill_task: asyncio.Task | None = None

    @property
    def final_path(self) -> Path:
        return CACHE_DIR / self.key

    @property
    def part_path(self) -> Path:
        return CACHE_DIR / f"{self.key}.part-{self.gen}"

    @property
    def meta_path(self) -> Path:
        return CACHE_DIR / f"{self.key}.meta"

    def info(self) -> dict:
        return {
            "state": self.state,
            "purged": self.purged,
            "readers": self.readers,
            "written": self.written,
            "size": self.size,
            "gen": self.gen,
            "sha256": self.meta.get("sha256"),
            "error": self.error,
        }


class Cache:
    def __init__(self) -> None:
        self.entries: dict[str, Entry] = {}
        self.lock = asyncio.Lock()
        self.session: aiohttp.ClientSession | None = None

    # ------------------------------------------------------------------ #
    # registry                                                            #
    # ------------------------------------------------------------------ #

    def _entry_from_disk(self, key: str) -> Entry | None:
        """Adopt a complete on-disk package (e.g. after a restart)."""
        final = CACHE_DIR / key
        try:
            st = final.stat()
        except OSError:
            return None
        if not final.is_file():
            return None
        e = Entry(key)
        e.state = "complete"
        e.size = st.st_size
        e.written = st.st_size
        e.headers_known = True
        try:
            e.meta = json.loads(e.meta_path.read_text())
        except (OSError, ValueError):
            e.meta = {}
        e.meta.setdefault("content_type", "application/octet-stream")
        self.entries[key] = e
        return e

    async def attach(self, key: str) -> Entry:
        """Return the current generation for key, starting a fill if needed."""
        async with self.lock:
            e = self.entries.get(key)
            if e is None:
                e = self._entry_from_disk(key)
            if e is None:
                e = Entry(key)
                self.entries[key] = e
                e.fill_task = asyncio.create_task(self._fill(e))
            e.readers += 1
            return e

    async def detach(self, e: Entry) -> None:
        async with self.lock:
            e.readers -= 1
            self._maybe_reap_locked(e)

    async def purge(self, key: str) -> bool:
        """Unregister the current generation; files go away with last reader."""
        async with self.lock:
            e = self.entries.pop(key, None)
            if e is not None:
                e.purged = True
                self._maybe_reap_locked(e)
                return True
            # Not in memory (e.g. never requested since restart): clean disk.
            existed = False
            final = CACHE_DIR / key
            for p in (final, CACHE_DIR / f"{key}.meta"):
                if p.exists():
                    _unlink_quiet(p)
                    existed = True
            try:
                for p in final.parent.glob(final.name + ".part-*"):
                    _unlink_quiet(p)
                    existed = True
            except OSError:
                pass
            return existed

    def _maybe_reap_locked(self, e: Entry) -> None:
        """Delete files once no readers remain and the fill is over."""
        if e.readers > 0 or e.state == "filling":
            return
        if e.state == "failed":
            _unlink_quiet(e.part_path)  # never keep a half package
        elif e.purged:  # complete or closed
            _unlink_quiet(e.part_path)
            _unlink_quiet(e.final_path)
            _unlink_quiet(e.meta_path)

    # ------------------------------------------------------------------ #
    # origin fill                                                         #
    # ------------------------------------------------------------------ #

    async def _fill(self, e: Entry) -> None:
        assert self.session is not None
        url = f"{ORIGIN_BASE_URL}/packages/{e.key}"
        try:
            async with self.session.get(url) as resp:
                e.origin_status = resp.status
                if resp.status != 200:
                    e.error = f"origin responded with status {resp.status}"
                    await self._finish(e, "failed")
                    return
                cl = resp.content_length
                e.size = cl if cl is not None else -1
                e.meta = {
                    "content_type": resp.headers.get("Content-Type")
                    or "application/octet-stream",
                    "sha256": resp.headers.get("X-Package-Sha256"),
                    "fetched_at": time.time(),
                }
                e.part_path.parent.mkdir(parents=True, exist_ok=True)
                f = open(e.part_path, "wb")
                try:
                    # Part file exists before readers are allowed to open it.
                    e.headers_known = True
                    async with e.cond:
                        e.cond.notify_all()
                    written = 0
                    async for chunk in resp.content.iter_chunked(CHUNK):
                        await asyncio.to_thread(f.write, chunk)
                        await asyncio.to_thread(f.flush)
                        written += len(chunk)
                        e.written = written
                        async with e.cond:
                            e.cond.notify_all()
                finally:
                    await asyncio.to_thread(f.close)
                if e.size != -1 and written != e.size:
                    raise aiohttp.ClientPayloadError(
                        f"short read from origin: {written}/{e.size} bytes"
                    )
            async with self.lock:
                e.size = written if e.size == -1 else e.size
                e.meta["size"] = written
                if e.purged:
                    # Purged mid-fill: keep the part file for current readers
                    # only; never register this generation.
                    e.state = "closed"
                else:
                    os.replace(e.part_path, e.final_path)  # atomic publish
                    try:
                        e.meta_path.write_text(json.dumps(e.meta))
                    except OSError:
                        pass
                    e.state = "complete"
                async with e.cond:
                    e.cond.notify_all()
                self._maybe_reap_locked(e)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - any origin/IO error fails the fill
            e.error = f"{type(exc).__name__}: {exc}"
            await self._finish(e, "failed")

    async def _finish(self, e: Entry, state: str) -> None:
        async with self.lock:
            e.state = state
            if state == "failed" and self.entries.get(e.key) is e:
                del self.entries[e.key]  # next GET retries the origin
            async with e.cond:
                e.cond.notify_all()
            self._maybe_reap_locked(e)

    # ------------------------------------------------------------------ #
    # client streaming                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _open_current(e: Entry):
        try:
            return open(e.part_path, "rb")
        except FileNotFoundError:
            return open(e.final_path, "rb")

    async def stream(self, e: Entry):
        """Yield the package from byte 0, tailing the fill file if needed."""
        offset = 0
        f = await asyncio.to_thread(self._open_current, e)
        try:
            while True:
                async with e.cond:
                    await e.cond.wait_for(
                        lambda: e.written > offset or e.state != "filling"
                    )
                if e.state == "failed":
                    raise FillFailed(e.error or "origin fill failed")
                if e.state == "filling":
                    limit = CHUNK
                else:  # complete | closed: drain exactly to the final size
                    limit = min(CHUNK, e.size - offset)
                    if limit <= 0:
                        break
                data = await asyncio.to_thread(f.read, limit)
                if not data:
                    if e.state == "filling":
                        continue  # defensive; written>offset implies data
                    break
                offset += len(data)
                yield data
        finally:
            try:
                f.close()
            finally:
                await self.detach(e)

    async def stream_range(self, e: Entry, start: int, end: int):
        """Yield exactly bytes ``start..end`` (inclusive) of a finished entry.

        Only called once the fill is over, so the file size is final; the
        caller decides 206 vs 416. Detaches the reader when streaming ends.
        """
        f = await asyncio.to_thread(self._open_current, e)
        try:
            await asyncio.to_thread(f.seek, start)
            remaining = end - start + 1
            while remaining > 0:
                data = await asyncio.to_thread(f.read, min(CHUNK, remaining))
                if not data:
                    break  # defensive: entry size is supposed to be final
                remaining -= len(data)
                yield data
        finally:
            try:
                f.close()
            finally:
                await self.detach(e)


cache = Cache()


def cleanup_cache_dir() -> None:
    """On boot: drop partial fills and orphan meta, keep complete packages."""
    for p in CACHE_DIR.rglob("*"):
        if not p.is_file():
            continue
        name = p.name
        if ".part-" in name:
            _unlink_quiet(p)
        elif name.endswith(".meta"):
            final = p.with_name(name[: -len(".meta")])
            if not final.exists():
                _unlink_quiet(p)


@asynccontextmanager
async def lifespan(app: FastAPI):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_cache_dir()
    timeout = aiohttp.ClientTimeout(total=None, connect=10, sock_connect=10, sock_read=300)
    cache.session = aiohttp.ClientSession(timeout=timeout)
    try:
        yield
    finally:
        await cache.session.close()


app = FastAPI(title="pkg-cache-proxy", version="1.0.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return {"ok": True}


def _version_headers(entry: Entry) -> dict:
    """Strong ETag from the origin digest plus the digest header itself."""
    headers: dict = {}
    sha256 = entry.meta.get("sha256")
    if sha256:
        headers["X-Package-Sha256"] = sha256
    etag = strong_etag(sha256)
    if etag:
        headers["ETag"] = etag
    return headers


def _full_response(entry: Entry, hit: bool) -> StreamingResponse:
    headers = {
        "X-Cache": "HIT" if hit else "MISS",
        "Accept-Ranges": "bytes",
        **_version_headers(entry),
    }
    if entry.size >= 0:
        headers["Content-Length"] = str(entry.size)
    return StreamingResponse(
        cache.stream(entry),
        media_type=entry.meta.get("content_type", "application/octet-stream"),
        headers=headers,
    )


def _range_response(entry: Entry, hit: bool, range_header: str,
                    if_range: str | None) -> PlainTextResponse | StreamingResponse:
    size = entry.size
    etag = strong_etag(entry.meta.get("sha256"))

    # If-Range: only honour the range when the client's validator still
    # matches this generation; otherwise restart with a full 200 so the
    # client replaces the stale bytes it already holds.
    if if_range is not None and (etag is None or if_range.strip() != etag):
        return _full_response(entry, hit)

    try:
        parsed = parse_single_range(range_header, size)
    except RangeUnsatisfiable as exc:
        return PlainTextResponse(
            f"requested range not satisfiable; total length {exc.size}",
            status_code=416,
            headers={
                "Content-Range": f"bytes */{exc.size}",
                "Accept-Ranges": "bytes",
                **_version_headers(entry),
            },
        )
    # Unsupported range syntax (multipart, non-bytes unit, inverted, ...):
    # ignore it and serve the full representation per RFC 9110.
    if parsed is None:
        return _full_response(entry, hit)

    start, end = parsed
    return StreamingResponse(
        cache.stream_range(entry, start, end),
        status_code=206,
        media_type=entry.meta.get("content_type", "application/octet-stream"),
        headers={
            "X-Cache": "HIT" if hit else "MISS",
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Content-Length": str(end - start + 1),
            **_version_headers(entry),
        },
    )


@app.get(
    "/packages/{path:path}",
    summary="Download a package, or resume with a single byte Range",
)
async def get_package(
    path: str,
    range_header: str | None = Header(
        default=None,
        alias="Range",
        description="Single byte range, e.g. `bytes=1048576-` or "
        "`bytes=1048576-2097151`. Multipart/other units are ignored (full 200).",
    ),
    if_range: str | None = Header(
        default=None,
        alias="If-Range",
        description="Strong ETag from a previous response (the quoted "
        "X-Package-Sha256). Matching -> 206; stale/missing -> full 200.",
    ),
):
    """Stream a package from byte 0, or serve a resume range.

    \b
    Resume flow:
    1. `GET /packages/x`            -> 200 with `ETag`/`X-Package-Sha256`
    2. interrupted, keep byte count N and the ETag
    3. `GET /packages/x` with `Range: bytes=N-`, `If-Range: "<etag>"`
       - same version: 206 + `Content-Range: bytes N-<last>/<total>`
       - new version: full 200 (restart the download from byte 0)
       - N past the end: 416 with `Content-Range: bytes */<total>`

    On a cache miss a range request waits for the single-flight fill to
    finish before answering; a failed fill returns an error and no bytes.
    """
    try:
        key = validate_key(path)
    except ValueError as exc:
        return PlainTextResponse(str(exc), status_code=400)

    entry = await cache.attach(key)
    hit = entry.state == "complete"
    try:
        if not entry.headers_known:
            async with entry.cond:
                await entry.cond.wait_for(
                    lambda: entry.headers_known or entry.state == "failed"
                )
    except asyncio.CancelledError:
        await cache.detach(entry)
        raise

    if entry.state == "failed":
        await cache.detach(entry)
        status = (
            entry.origin_status
            if entry.origin_status and entry.origin_status != 200
            else 502
        )
        return PlainTextResponse(f"origin fetch failed: {entry.error}", status_code=status)

    # Full downloads (and ignored/unsupported ranges) tail the live fill and
    # deliver the first chunk as soon as it lands on disk.
    if range_header is None:
        return _full_response(entry, hit)

    # Resumes are answered from a finished generation only. On a miss, wait
    # for this single-flight fill to complete (or fail) before responding.
    if entry.state == "filling":
        try:
            async with entry.cond:
                await entry.cond.wait_for(lambda: entry.state != "filling")
        except asyncio.CancelledError:
            await cache.detach(entry)
            raise
        if entry.state == "failed":
            await cache.detach(entry)
            status = (
                entry.origin_status
                if entry.origin_status and entry.origin_status != 200
                else 502
            )
            return PlainTextResponse(
                f"origin fetch failed: {entry.error}", status_code=status
            )

    return _range_response(entry, hit, range_header, if_range)


@app.post("/purge/{path:path}")
async def purge_package(path: str):
    """Purge one path. The next download re-fills from the origin; downloads
    already in progress keep reading their old generation to completion."""
    try:
        key = validate_key(path)
    except ValueError as exc:
        return PlainTextResponse(str(exc), status_code=400)
    was_cached = await cache.purge(key)
    return {"path": key, "purged": True, "was_cached": was_cached}


@app.get("/cache/status")
async def cache_status():
    async with cache.lock:
        return {k: e.info() for k, e in cache.entries.items()}

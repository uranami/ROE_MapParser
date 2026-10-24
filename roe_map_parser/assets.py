"""Read-only SFC v3 and original game VFS runtime access.

SFC path hash / container format: client 1416CA420 / 1416C98E0.
VFS ABI: original x64 game runtime, validated on Windows.
Only VFS_x64.dll from the supplied game root is loaded; no server DLLs.
"""
from __future__ import annotations

import ctypes as c
import hashlib
import os
from pathlib import Path
import re
import struct

import lz4.block


def path_hash(path):
    result = 0x1712E27F4D2960DB
    for byte in re.sub(r"[/\\]+", r"\\", path).upper().encode("ascii"):
        result = (67 * result + byte) & 0xFFFFFFFFFFFFFFFF
    return result


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Sfc:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.file = self.path.open("rb")
        try:
            self.size = self.path.stat().st_size
            header = self.file.read(12)
            if len(header) != 12 or header[:8] != b"SFC\x1a\x03\0\0\0":
                raise ValueError(f"{self.path}: unsupported SFC header")
            count = struct.unpack_from("<I", header, 8)[0]
            if count > 2000000 or 12 + count * 12 > self.size:
                raise ValueError(f"{self.path}: invalid SFC index count")
            self.index = dict(struct.iter_unpack("<QI", self.file.read(count * 12)))
            if len(self.index) != count or any(o < 12 + count * 12 or o + 8 > self.size
                                               for o in self.index.values()):
                raise ValueError(f"{self.path}: invalid/duplicate SFC index")
        except Exception:
            self.close()
            raise

    def read(self, name, maximum=128 * 1024 * 1024):
        offset = self.index[path_hash(name)]
        self.file.seek(offset)
        packed_size = struct.unpack("<I", self.file.read(4))[0]
        if packed_size > maximum or offset + 8 + packed_size > self.size:
            raise ValueError(f"{self.path}:{name}: invalid compressed length")
        packed = self.file.read(packed_size)
        size = struct.unpack("<I", self.file.read(4))[0]
        if size > maximum:
            raise ValueError(f"{self.path}:{name}: decompression budget exceeded")
        raw = packed if packed_size == size else lz4.block.decompress(packed, uncompressed_size=size)
        if len(raw) != size:
            raise ValueError(f"{self.path}:{name}: incorrect uncompressed length")
        return raw

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class GameSfc:
    """Indexes containers, decompresses only explicitly requested resources."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.archives = []
        self.lookup = {}
        try:
            for path in sorted((self.root / "SFC").glob("*.sfc")):
                archive = Sfc(path)
                self.archives.append(archive)
                for key in archive.index:
                    self.lookup.setdefault(key, archive)
            if not self.archives:
                raise ValueError(f"{self.root}: no SFC archives")
        except Exception:
            self.close()
            raise

    def read(self, path, maximum=128 * 1024 * 1024):
        key = path_hash(path)
        if key not in self.lookup:
            raise FileNotFoundError(f"resource absent from installation SFCs: {path}")
        archive = self.lookup[key]
        raw = archive.read(path, maximum)
        return raw, {"resource": path, "archive": str(archive.path),
                     "record_offset": archive.index[key], "bytes": len(raw),
                     "sha256": hashlib.sha256(raw).hexdigest()}

    def close(self):
        for archive in self.archives:
            archive.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class _FindData(c.Structure):
    _fields_ = [("path", c.c_char * 512), ("size", c.c_uint64),
                ("index", c.c_uint64), ("flags", c.c_uint32),
                ("padding", c.c_uint32), ("aux", c.c_uint64),
                ("extra", c.c_uint8), ("tail", c.c_uint8 * 7)]


class Vfs:
    def __init__(self, root, volume):
        if os.name != "nt" or c.sizeof(c.c_void_p) != 8:
            raise RuntimeError("original game VFS runtime requires x64 Windows")
        root = Path(root).resolve()
        self.path = root / volume
        self.directory = os.add_dll_directory(str(root))
        self.dll = c.CDLL(str(root / "VFS_x64.dll"))
        self.dll.CreateVFS.restype = c.c_void_p
        self.dll.ReleaseVFS.argtypes = [c.c_void_p]
        self.dll.ReleaseVFS.restype = c.c_uint8
        self.object = self.dll.CreateVFS()
        try:
            if not self.object or not self.method(1, c.c_uint8, c.c_uint64)(self.object, 50 * 1024 * 1024):
                raise ValueError("VFS Init failed")
            mount = self.method(10, c.c_uint32, c.c_char_p, c.c_char_p,
                                c.c_uint64, c.c_uint64, c.c_uint8)
            if not mount(self.object, str(self.path).encode("mbcs"), b"", 0, 0, 1):
                raise ValueError(f"VFS Mount failed: {self.path}")
        except Exception:
            self.close()
            raise

    def method(self, slot, result, *arguments):
        table = c.cast(self.object, c.POINTER(c.POINTER(c.c_void_p))).contents
        return c.CFUNCTYPE(result, c.c_void_p, *arguments)(table[slot])

    def entries(self, parent="."):
        first = self.method(21, c.c_void_p, c.c_char_p, c.POINTER(_FindData))
        next_ = self.method(22, c.c_uint8, c.c_void_p, c.POINTER(_FindData))
        close = self.method(23, c.c_uint8, c.c_void_p)
        pending, seen = [parent], set()
        while pending:
            directory = pending.pop()
            if directory in seen:
                raise ValueError(f"cyclic VFS directory: {directory}")
            seen.add(directory)
            data = _FindData()
            iterator = first(self.object, directory.encode("mbcs"), c.byref(data))
            if not iterator:
                continue
            try:
                while True:
                    name = data.path.decode("mbcs")
                    if not name or name in (".", "..") or "\\" in name or "/" in name:
                        raise ValueError(f"invalid VFS child name: {name!r}")
                    path = name if directory == "." else directory + "\\" + name
                    if data.flags & 0x10:
                        pending.append(path)
                    else:
                        yield {"path": path, "size": int(data.size), "volume": self.path.name}
                    data = _FindData()
                    if not next_(self.object, iterator, c.byref(data)):
                        break
            finally:
                close(self.object, iterator)

    def read(self, path, maximum=128 * 1024 * 1024):
        name = path.encode("mbcs")
        size = self.method(18, c.c_int64, c.c_char_p)(self.object, name)
        if size <= 0:
            raise FileNotFoundError(f"{self.path}:{path}: no VFS data")
        if size > maximum:
            raise ValueError(f"{self.path}:{path}: resource budget exceeded")
        handle = self.method(13, c.c_uint32, c.c_char_p)(self.object, name)
        if handle == 0 or handle == 0xFFFFFFFF:
            raise FileNotFoundError(f"{self.path}:{path}: VFS OpenFile failed")
        try:
            buffer = c.create_string_buffer(size)
            read = c.c_uint64()
            ok = self.method(15, c.c_uint8, c.c_uint32, c.c_void_p,
                             c.c_uint64, c.POINTER(c.c_uint64))(
                                 self.object, handle, buffer, size, c.byref(read))
            if not ok or read.value != size:
                raise ValueError(f"{self.path}:{path}: short VFS read")
            return buffer.raw
        finally:
            self.method(14, c.c_uint8, c.c_uint32)(self.object, handle)

    def close(self):
        if self.object:
            self.dll.ReleaseVFS(self.object)
            self.object = None
        self.directory.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class GameAssets(GameSfc):
    """SFC is preferred; use actual VFS data for resources absent from SFC."""
    def __init__(self, root):
        self.volumes = {}
        super().__init__(root)

    def volume(self, path):
        path = Path(path)
        if path.name not in self.volumes:
            self.volumes[path.name] = Vfs(self.root, path.name)
        return self.volumes[path.name]

    def read(self, path, maximum=128 * 1024 * 1024):
        try:
            return super().read(path, maximum)
        except FileNotFoundError:
            for volume in sorted(self.root.glob("data_bas_*.vfs")):
                try:
                    raw = self.volume(volume).read(path, maximum)
                except FileNotFoundError:
                    continue
                return raw, {"resource": path, "archive": str(volume.resolve()),
                             "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            raise FileNotFoundError(f"resource absent from installation SFC/VFS: {path}")

    def close(self):
        for volume in self.volumes.values():
            volume.close()
        super().close()

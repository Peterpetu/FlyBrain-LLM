"""Pre-registration enforcement.

`PREREG.md` in an experiment directory is hashed the first time any run starts
(`.prereg.sha256`). Every later run re-checks the hash; if the file changed after results
exist, the harness refuses to run. Edit -> delete results -> re-freeze is the only path.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


class PreregError(RuntimeError):
    pass


class Prereg:
    def __init__(self, exp_dir: Path):
        self.exp_dir = Path(exp_dir); self.path = self.exp_dir / "PREREG.md"; self.lock = self.exp_dir / ".prereg.sha256"

    def digest(self) -> str:
        if not self.path.exists():
            raise PreregError(f"missing {self.path}: write the pre-registration before running")
        return hashlib.sha256(self.path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()   # line endings normalised: a checkout that converts LF/CRLF must not break a lock

    def freeze_or_check(self) -> str:
        d = self.digest()
        if self.lock.exists():
            frozen = self.lock.read_text().strip()
            if frozen != d:
                raise PreregError(f"PREREG.md was edited after results existed (frozen {frozen[:12]}, now {d[:12]}). "
                                  "Delete runs/ and .prereg.sha256 to re-register.")
        else:
            self.lock.write_text(d + "\n")
        return d

"""Checks shared by the scripts of the works: placeholders, committed inputs, the pinned generator and hashes.

The conditions of a work are frozen in two stages: the template (method, committed before the texts are chosen) and the
conditions with the inputs filled in (committed after the choice and before any value of the chosen texts is
computed). A value "FILL" (or a string that starts with "FILL") marks an input still to be written; a script stops if
one remains.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIB_FILES = ("__init__.py", "field.py", "lexicon.py", "mapping.py", "transform.py")


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def placeholders(c, path: str = "") -> list[str]:
    """Paths of the values that are still "FILL..."."""
    if isinstance(c, dict):
        return [x for k, v in c.items() for x in placeholders(v, f"{path}.{k}")]
    if isinstance(c, list):
        return [x for i, v in enumerate(c) for x in placeholders(v, f"{path}[{i}]")]
    return [path] if isinstance(c, str) and c.startswith("FILL") else []


def load_conditions(path: Path) -> tuple[bytes, dict]:
    if not path.exists():
        raise SystemExit(f"{path} does not exist; copy the template next to it and fill in the inputs")
    raw = path.read_bytes()  # read once: the conditions used and the hash recorded come from the same bytes
    c = json.loads(raw)
    left = placeholders(c)
    if left:
        raise SystemExit(f"inputs still to be filled in: {', '.join(left)}")
    return raw, c


def need(c: dict, key: str, kind, where: str = ""):
    """c[key], checked to be of the type kind (bool is never a number) and, for numbers, finite."""
    if key not in c:
        raise SystemExit(f"missing {where}{key}")
    v = c[key]
    if kind in (int, float) and (isinstance(v, bool) or not isinstance(v, (int, float) if kind is float else int)
                                 or not math.isfinite(v)):
        raise SystemExit(f"{where}{key} = {v!r} is not a finite {kind.__name__}")
    if kind not in (int, float) and not isinstance(v, kind):
        raise SystemExit(f"{where}{key} = {v!r} is not a {kind.__name__}")
    return v


def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def git_state(paths: list[str]) -> dict:
    """HEAD and whether any of the paths differs from it. Every path must exist: git status is silent on a path that
    does not, which would stop guarding it without a sign."""
    missing = [p for p in paths if not (ROOT / p).exists()]
    if missing:
        raise SystemExit(f"paths to guard do not exist: {missing}")
    commit = git("rev-parse", "HEAD")
    if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise SystemExit(f"unexpected commit from git: {commit!r}")
    return {"git_commit": commit, "git_dirty": bool(git("status", "--porcelain", "--", *paths))}


def same_as_head(paths: list[Path]) -> None:
    """The files that are run or read must be the committed bytes (the status check alone would pass an ignored
    file)."""
    for p in paths:
        rel = Path(p).resolve().relative_to(ROOT).as_posix()
        r = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True)
        if r.returncode != 0 or r.stdout != Path(p).read_bytes():
            raise SystemExit(f"{rel} differs from its committed version (or is not committed)")


def installed_commit() -> str:
    return json.loads(metadata.distribution("atact").read_text("direct_url.json"))["vcs_info"]["commit_id"]


def library_sha256() -> dict[str, str]:
    import atact

    d = Path(atact.__file__).parent
    return {n: sha256(d / n) for n in LIB_FILES}


def check_generator(g: dict) -> None:
    if installed_commit() != g["commit"]:
        raise SystemExit(f"atact is installed from {installed_commit()}, not {g['commit']}")
    if library_sha256() != g["library_sha256"]:
        raise SystemExit("the installed library files differ from those named in the conditions")


def check_file(p: Path, expected: str, what: str) -> bytes:
    raw = Path(p).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise SystemExit(f"{what} ({p}) does not match the SHA-256 of the conditions")
    return raw


def check_versions(c: dict, packages: tuple[str, ...]) -> None:
    for p in packages:
        if metadata.version(p) != c[p]:
            raise SystemExit(f"{p} is {metadata.version(p)}, not {c[p]} as in the conditions")


def matches_template(template, filled, path: str = "") -> list[str]:
    """Where the filled conditions depart from the committed template: every value of the template that is not a FILL
    mark must be unchanged, and no key may be added or removed. A FILL mark may be replaced by anything."""
    if isinstance(template, str) and template.startswith("FILL"):
        return []
    if isinstance(template, dict):
        if not isinstance(filled, dict) or set(template) != set(filled):
            return [f"{path}: keys differ"]
        return [x for k in template for x in matches_template(template[k], filled[k], f"{path}.{k}")]
    if isinstance(template, list):
        if not isinstance(filled, list) or len(template) != len(filled):
            return [f"{path}: list differs"]
        return [x for i, (a, b) in enumerate(zip(template, filled)) for x in matches_template(a, b, f"{path}[{i}]")]
    return [] if type(template) is type(filled) and template == filled else [f"{path}: {template!r} -> {filled!r}"]


def check_template(template: Path, c: dict) -> None:
    diff = matches_template(json.loads(Path(template).read_bytes()), c)
    if diff:
        raise SystemExit("the conditions depart from the committed template: " + "; ".join(diff))


def versions(*packages: str) -> dict:
    return {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
            "atact_commit": installed_commit(), **{p: metadata.version(p) for p in packages}}

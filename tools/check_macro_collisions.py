#!/usr/bin/env python3
"""check_macro_collisions.py -- find identifiers that this repo declares and
that the PrismML fork #defines as macros.

Dropping the Bonsai-P40 files into ggml/src/ggml-cuda/ puts them in the same
translation unit as ggml-common.h and common.cuh.  Any name that ggml #defines
as a macro and that this repo declares as a constant/type/function will be
textually replaced by the preprocessor and fail to compile (or, worse, silently
change meaning).

Usage:
    python3 tools/check_macro_collisions.py /path/to/prism-fork
Exit code 1 if a collision is found.
"""
import os
import re
import sys

# identifiers this repo introduces (declarations in src/ and include/)
DECL = re.compile(
    r"^\s*(?:static\s+)?(?:constexpr|const|inline|BONSAI_HD)\s+"
    r"(?:[\w:<>,\s\*&]+?)\s+"
    r"([A-Za-z_]\w*)\s*(?:\(|=|;|\{)")


def declared_names(root):
    names = set()
    for sub in ("src", "include"):
        for dirpath, _, files in os.walk(os.path.join(root, sub)):
            for f in files:
                if not f.endswith((".h", ".cuh", ".cu", ".cpp")):
                    continue
                for line in open(os.path.join(dirpath, f), errors="replace"):
                    m = DECL.match(line)
                    if m:
                        names.add(m.group(1))
                # struct / enum names
                for m in re.finditer(r"\b(?:struct|enum|class)\s+([A-Za-z_]\w*)",
                                     open(os.path.join(dirpath, f), errors="replace").read()):
                    names.add(m.group(1))
    return names


def fork_macros(fork):
    macros = {}
    roots = [os.path.join(fork, "ggml", "src"),
             os.path.join(fork, "ggml", "include")]
    for root in roots:
        for dirpath, _, files in os.walk(root):
            for f in files:
                if not f.endswith((".h", ".cuh", ".cu", ".c", ".cpp")):
                    continue
                p = os.path.join(dirpath, f)
                try:
                    txt = open(p, errors="replace").read()
                except OSError:
                    continue
                for m in re.finditer(r"^#define\s+([A-Za-z_]\w*)", txt, re.M):
                    macros.setdefault(m.group(1), p)
    return macros


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    fork = sys.argv[1]
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    mine = declared_names(here)
    macros = fork_macros(fork)
    print(f"declared names in this repo : {len(mine)}")
    print(f"#define macros in the fork  : {len(macros)}")

    hits = sorted(mine & set(macros))
    if hits:
        print(f"\nCOLLISIONS ({len(hits)}):")
        for h in hits:
            print(f"  {h:24s} <- {os.path.relpath(macros[h], fork)}")
        print("\nThese must be renamed or derived from the fork's macro.")
        return 1
    print("\nno macro collisions")
    return 0


if __name__ == "__main__":
    sys.exit(main())

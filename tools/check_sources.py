#!/usr/bin/env python3
"""check_sources.py -- static sanity checks that would have caught the last
three real bugs in this repo.  No compiler needed.

  1. Un-prefixed ggml macro names.  The PrismML fork #defines QK_PQ2_0 / QK8_1 /
     QK8_0 / QI_PQ2_0 as macros, so declaring or using them bare makes the
     preprocessor eat the identifier.  Everything here must be BONSAI_-prefixed
     (only #if/#ifdef guards and comments may use the bare name).
  2. Dangling `#pragma unroll`.  A pragma that is not immediately followed by a
     loop is silently ignored (nvcc warning #607) -- and it is the fingerprint of
     a function that got cut in half.
  3. Code after the final top-level `}`.  Same fingerprint: a chunk that landed
     at the end of the file instead of inside its function.
  4. Per-file brace balance.

Exit code 1 on any finding.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTS = (".h", ".cuh", ".cu", ".cpp", ".c")
SKIP_DIRS = {".git", "build"}

BARE = re.compile(r"(?<!BONSAI_)\b(QK_PQ2_0|QK8_1|QK8_0|QI_PQ2_0)\b")


def sources():
    for dirpath, dirnames, files in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in files:
            if f.endswith(EXTS):
                yield os.path.join(dirpath, f)


def main():
    problems = []
    n = 0
    for path in sources():
        rel = os.path.relpath(path, ROOT)
        try:
            text = open(path, errors="replace").read()
        except OSError:
            continue
        n += 1
        lines = text.split("\n")

        # 1. bare ggml macro names, outside guards, comments and the intentional
        #    inheritance lines (`static constexpr int BONSAI_X = X;`)
        for i, line in enumerate(lines, 1):
            st = line.strip()
            if st.startswith("#") or st.startswith("//") or st.startswith("*"):
                continue
            m = BARE.search(line)
            if m and f"BONSAI_{m.group(1)}" not in line:
                problems.append(f"{rel}:{i}: bare ggml macro name '{m.group(1)}' "
                                f"(must be BONSAI_{m.group(1)})")

        # 2. dangling #pragma unroll
        for i, line in enumerate(lines):
            if line.strip() != "#pragma unroll":
                continue
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if not (nxt.startswith("for") or nxt.startswith("while") or nxt == "{"):
                problems.append(f"{rel}:{i+1}: '#pragma unroll' not followed by a loop "
                                f"(next: '{nxt[:40]}')")

        # 3. code after the final top-level '}' (a '}' in column 0 closes a
        #    namespace / extern-C block).  Blank lines, comments and further
        #    preprocessor lines are legitimate there.
        idx = [i for i, l in enumerate(lines) if l.startswith("}")]
        if idx:
            for l in lines[idx[-1] + 1:]:
                st = l.strip()
                if not st or st.startswith("//") or st.startswith("/*") or st.startswith("#"):
                    continue
                problems.append(f"{rel}: code after the final top-level '}}': '{st[:40]}'")
                break

        # 4. brace balance
        if text.count("{") != text.count("}"):
            problems.append(f"{rel}: unbalanced braces ({text.count('{')} open, "
                            f"{text.count('}')} close)")

    print(f"checked {n} source files")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems:
            print("  " + p)
        return 1
    print("all static checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

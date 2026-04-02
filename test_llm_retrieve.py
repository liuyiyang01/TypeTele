#!/usr/bin/env python3
"""
Test LLM-based gesture retrieval only: no LEAP hand, no camera, no ASR.

Loads the type catalog from TypeLibrary/<category>/_type_info.json (preferred).
If JSON is missing, falls back to *.txt names in that folder.

Usage:
  export BIGMODEL_API_KEY="your_key"
  python test_llm_retrieve.py
  python test_llm_retrieve.py -q "grasp a thin sheet from the edge"
"""

from __future__ import annotations

import argparse
import os
import sys

from retrieve.retrieve import Retrieve


def main() -> None:
    p = argparse.ArgumentParser(description="Dry-run LLM gesture type retrieval")
    p.add_argument(
        "--category",
        default=os.getenv("TYPE_LIBRARY_CATEGORY", "leap"),
        help="Subfolder under TypeLibrary/ (default: leap)",
    )
    p.add_argument(
        "-q",
        "--query",
        help="Single query then exit (otherwise interactive REPL)",
    )
    p.add_argument(
        "--api-key",
        default=os.getenv("BIGMODEL_API_KEY", ""),
        help="Defaults to env BIGMODEL_API_KEY",
    )
    p.add_argument(
        "--base-url",
        default=os.getenv(
            "BIGMODEL_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/"
        ),
    )
    p.add_argument(
        "--model",
        default=os.getenv("BIGMODEL_CHAT_MODEL", "glm-4-flash"),
    )
    args = p.parse_args()

    if not args.api_key.strip():
        print(
            "Missing API key. Set BIGMODEL_API_KEY or pass --api-key.\n"
            "Example: export BIGMODEL_API_KEY='...'",
            file=sys.stderr,
        )
        sys.exit(1)

    retriever = Retrieve(
        api_key=args.api_key,
        base_url=args.base_url,
        category=args.category,
        model=args.model,
    )
    retriever.load_type_library()

    if not retriever.type_files:
        print(
            "No gesture types loaded. Add TypeLibrary/<category>/_type_info.json "
            "or at least one .txt file with a matching name.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Loaded {len(retriever.type_files)} type ids "
        f"(e.g. {retriever.type_files[:min(5, len(retriever.type_files))]})"
    )

    def run_one(q: str) -> None:
        out = retriever.retrieve_sync(q)
        print(f"  -> {out!r}")

    if args.query is not None:
        print(f"query: {args.query!r}")
        run_one(args.query)
        return

    print("Enter natural language queries (empty line to quit).")
    while True:
        try:
            line = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        print(f"query: {line!r}")
        run_one(line)


if __name__ == "__main__":
    main()

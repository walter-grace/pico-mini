#!/usr/bin/env python3
"""Web search helper for mac-code UI. Usage: python3 search.py <query>"""
import sys
try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        print("ERROR: pip install ddgs")
        sys.exit(1)

query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
if not query:
    print("ERROR: no query")
    sys.exit(1)

results = []
try:
    with DDGS() as d:
        for r in d.text(query, max_results=5):
            title = r.get("title", "")
            body = r.get("body", "")
            results.append(f"- {title}: {body}")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

print("\n".join(results) if results else "No results found.")

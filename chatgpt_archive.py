#!/usr/bin/env python3
"""Pull my ChatGPT threads and stash them in sqlite."""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from openai_client import OpenAIClient, OpenAIError

DB_PATH = Path.home() / ".chatgpt_archive" / "archive.db"

def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS threads (
            id TEXT PRIMARY KEY,
            created_at TEXT,
            title TEXT,
            raw_json TEXT
        );
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            thread_id TEXT,
            created_at TEXT,
            role TEXT,
            content TEXT,
            raw_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id);
        CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);
    """)

def _db_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _store_thread(conn: sqlite3.Connection, thread: dict) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO threads (id, created_at, title, raw_json)
        VALUES (?, ?, ?, ?)
        """,
        (
            thread["id"],
            thread.get("created_at"),
            thread.get("metadata", {}).get("title", "untitled"),
            json.dumps(thread),
        ),
    )

def _store_message(conn: sqlite3.Connection, thread_id: str, msg: dict) -> None:
    content = ""
    for part in msg.get("content", []):
        if part.get("type") == "text":
            content += part["text"].get("value", "")

    conn.execute(
        """
        INSERT OR REPLACE INTO messages (id, thread_id, created_at, role, content, raw_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            msg["id"],
            thread_id,
            msg.get("created_at"),
            msg.get("role"),
            content,
            json.dumps(msg),
        ),
    )

def _already_fetched(conn: sqlite3.Connection, thread_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM threads WHERE id = ?", (thread_id,)
    ).fetchone()
    return row is not None

def _list_local_threads(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, created_at, title FROM threads ORDER BY created_at DESC"
    ).fetchall()

def _search(conn: sqlite3.Connection, query: str) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT m.thread_id, t.title, m.role, m.content
        FROM messages m
        JOIN threads t ON t.id = m.thread_id
        WHERE m.content LIKE ?
        ORDER BY m.created_at DESC
        """,
        (f"%{query}%",),
    ).fetchall()

def _show_thread(conn: sqlite3.Connection, thread_id: str) -> None:
    thread = conn.execute(
        "SELECT * FROM threads WHERE id = ?", (thread_id,)
    ).fetchone()
    if not thread:
        print(f"thread {thread_id} not found locally", file=sys.stderr)
        sys.exit(1)

    print(f"Thread: {thread['title'] or 'untitled'} ({thread['id']})")
    print("-" * 40)

    msgs = conn.execute(
        "SELECT role, content FROM messages WHERE thread_id = ? ORDER BY created_at",
        (thread_id,),
    ).fetchall()
    for msg in msgs:
        label = msg["role"].upper()
        print(f"\n{label}:\n{msg['content']}\n")

def _prune_before(conn: sqlite3.Connection, cutoff: str) -> int:
    cur = conn.execute("DELETE FROM messages WHERE thread_id IN (SELECT id FROM threads WHERE created_at < ?)", (cutoff,))
    conn.execute("DELETE FROM threads WHERE created_at < ?", (cutoff,))
    return cur.rowcount

def _fetch_all(args: argparse.Namespace) -> int:
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("set OPENAI_API_KEY or pass --api-key", file=sys.stderr)
        return 2

    client = OpenAIClient(api_key)
    conn = _db_conn()
    _init_db(conn)

    try:
        threads = client.list_threads(limit=args.limit, since=args.since)
    except OpenAIError as e:
        print(f"api error: {e}", file=sys.stderr)
        return 1

    new_count = 0
    for thread in threads:
        tid = thread["id"]
        if _already_fetched(conn, tid) and not args.force:
            continue

        try:
            messages = client.list_messages(tid)
        except OpenAIError as e:
            print(f"skipping {tid}: {e}", file=sys.stderr)
            continue

        _store_thread(conn, thread)
        for msg in messages:
            _store_message(conn, tid, msg)
        new_count += 1

    if args.prune:
        n = _prune_before(conn, args.prune)
        print(f"pruned {n} old messages")

    conn.commit()
    print(f"fetched {new_count} threads")
    return 0

def _list_cmd(args: argparse.Namespace) -> int:
    conn = _db_conn()
    _init_db(conn)
    rows = _list_local_threads(conn)
    if not rows:
        print("no threads archived yet. run: python chatgpt_archive.py fetch")
        return 0
    for r in rows:
        ts = r["created_at"] or "unknown"
        print(f"{r['id']} | {ts} | {r['title']}")
    return 0

def _search_cmd(args: argparse.Namespace) -> int:
    conn = _db_conn()
    _init_db(conn)
    results = _search(conn, args.query)
    if not results:
        print("no matches")
        return 0
    for r in results:
        snippet = r["content"].replace("\n", " ")[:120]
        print(f"{r['thread_id']} | {r['title']} | [{r['role']}] {snippet}")
    return 0

def _show_cmd(args: argparse.Namespace) -> int:
    conn = _db_conn()
    _init_db(conn)
    _show_thread(conn, args.thread_id)
    return 0

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Archive my ChatGPT threads locally.",
        usage="python chatgpt_archive.py <command> [options]",
    )
    sub = parser.add_subparsers(dest="command")

    fetch = sub.add_parser("fetch", help="pull threads from OpenAI")
    fetch.add_argument("--api-key", default="")
    fetch.add_argument("--limit", type=int, default=100)
    fetch.add_argument("--force", action="store_true")
    fetch.add_argument("--since", default=None)
    fetch.add_argument("--prune", default=None, metavar="DATE",
                        help="delete threads older than DATE (ISO format)")

    sub.add_parser("list", help="show archived thread IDs")

    search = sub.add_parser("search", help="search message content")
    search.add_argument("query")

    show = sub.add_parser("show", help="display a thread")
    show.add_argument("thread_id")

    args = parser.parse_args()

    if args.command == "fetch":
        return _fetch_all(args)
    if args.command == "list":
        return _list_cmd(args)
    if args.command == "search":
        return _search_cmd(args)
    if args.command == "show":
        return _show_cmd(args)

    parser.print_usage()
    return 2

if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except KeyboardInterrupt:
        sys.exit(130)

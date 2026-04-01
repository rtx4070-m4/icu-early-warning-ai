#!/usr/bin/env python3
"""
Database Migration Manager
Applies versioned SQL migrations to the Hospital OS PostgreSQL database.
Tracks applied migrations in a _migrations table.
Supports: apply, rollback, status, create.
"""

import os
import sys
import re
import hashlib
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent.parent / "database" / "migrations"
SCHEMA_TABLE = "_migrations"

MIGRATIONS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────
# Migration file format
# ─────────────────────────────────────────
# Files: database/migrations/V001__description.sql
#        database/migrations/V001__description.down.sql (rollback)
# Content:
#   -- Migration: V001 Add patient flags column
#   ALTER TABLE patients ADD COLUMN flags JSONB DEFAULT '{}';

def _parse_version(filename: str) -> Optional[Tuple[int, str]]:
    """Extract version number and description from filename."""
    m = re.match(r"V(\d+)__(.+?)(?:\.down)?\.sql$", filename)
    if not m:
        return None
    return int(m.group(1)), m.group(2).replace("_", " ")


def _file_checksum(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# ─────────────────────────────────────────
# DB connection
# ─────────────────────────────────────────

def _get_conn():
    try:
        import psycopg2
        return psycopg2.connect(
            host=os.environ.get("DB_HOST", "localhost"),
            port=int(os.environ.get("DB_PORT", 5432)),
            dbname=os.environ.get("DB_NAME", "hospital_os"),
            user=os.environ.get("DB_USER", "hospital"),
            password=os.environ.get("DB_PASSWORD", ""),
            connect_timeout=10,
        )
    except ImportError:
        logger.error("psycopg2 not installed. Run: pip install psycopg2-binary")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        sys.exit(1)


def _ensure_migrations_table(conn):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA_TABLE} (
                id            SERIAL PRIMARY KEY,
                version       INTEGER NOT NULL UNIQUE,
                description   TEXT NOT NULL,
                filename      TEXT NOT NULL,
                checksum      TEXT NOT NULL,
                applied_at    TIMESTAMP DEFAULT NOW(),
                applied_by    TEXT DEFAULT current_user,
                execution_ms  INTEGER
            )
        """)
        conn.commit()
        logger.debug(f"Migrations table ensured: {SCHEMA_TABLE}")


# ─────────────────────────────────────────
# Core commands
# ─────────────────────────────────────────

def cmd_status():
    """Show which migrations have been applied."""
    conn = _get_conn()
    _ensure_migrations_table(conn)

    # Applied
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT version, description, applied_at, execution_ms
            FROM {SCHEMA_TABLE}
            ORDER BY version
        """)
        applied = {row[0]: row for row in cur.fetchall()}

    # Available
    available = sorted(
        [f for f in MIGRATIONS_DIR.glob("V*.sql") if ".down." not in f.name],
        key=lambda f: f.name
    )

    print(f"\n{'='*60}")
    print(f"Migration Status — {os.environ.get('DB_NAME','hospital_os')}")
    print(f"{'='*60}")
    print(f"{'Ver':>4}  {'Status':8}  {'Applied At':19}  Description")
    print(f"{'─'*58}")

    for path in available:
        parsed = _parse_version(path.name)
        if not parsed:
            continue
        ver, desc = parsed
        if ver in applied:
            _, _, applied_at, exec_ms = applied[ver]
            status = "✓ APPLIED"
            ts = applied_at.strftime("%Y-%m-%d %H:%M:%S") if applied_at else ""
        else:
            status = "○ PENDING"
            ts = ""
        print(f"{ver:>4}  {status:8}  {ts:19}  {desc}")

    pending = sum(1 for f in available
                   if _parse_version(f.name) and _parse_version(f.name)[0] not in applied)
    print(f"\n  Applied: {len(applied)}  |  Pending: {pending}")
    conn.close()


def cmd_apply(target_version: Optional[int] = None, dry_run: bool = False):
    """Apply all pending migrations up to target_version."""
    conn = _get_conn()
    _ensure_migrations_table(conn)

    with conn.cursor() as cur:
        cur.execute(f"SELECT version FROM {SCHEMA_TABLE}")
        applied_versions = {row[0] for row in cur.fetchall()}

    migrations = sorted(
        [f for f in MIGRATIONS_DIR.glob("V*.sql") if ".down." not in f.name],
        key=lambda f: f.name
    )
    pending = [
        f for f in migrations
        if _parse_version(f.name) and _parse_version(f.name)[0] not in applied_versions
        and (target_version is None or _parse_version(f.name)[0] <= target_version)
    ]

    if not pending:
        print("No pending migrations.")
        conn.close()
        return

    print(f"{'[DRY RUN] ' if dry_run else ''}Applying {len(pending)} migration(s)...")

    for path in pending:
        ver, desc = _parse_version(path.name)
        sql = path.read_text()
        checksum = _file_checksum(path)
        print(f"  {'[DRY RUN] ' if dry_run else ''}V{ver:03d}: {desc}...", end=" ")

        if dry_run:
            print("(skipped)")
            continue

        t0 = datetime.utcnow()
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                elapsed_ms = int((datetime.utcnow() - t0).total_seconds() * 1000)
                cur.execute(
                    f"INSERT INTO {SCHEMA_TABLE} "
                    f"(version, description, filename, checksum, execution_ms) "
                    f"VALUES (%s, %s, %s, %s, %s)",
                    (ver, desc, path.name, checksum, elapsed_ms)
                )
            conn.commit()
            print(f"✓ ({elapsed_ms}ms)")
        except Exception as e:
            conn.rollback()
            print(f"✗ FAILED: {e}")
            logger.error(f"Migration V{ver} failed: {e}")
            conn.close()
            sys.exit(1)

    print(f"\n✓ Applied {len(pending)} migration(s) successfully.")
    conn.close()


def cmd_rollback(steps: int = 1):
    """Roll back the last N migrations."""
    conn = _get_conn()
    _ensure_migrations_table(conn)

    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT version, description, filename
            FROM {SCHEMA_TABLE}
            ORDER BY version DESC
            LIMIT %s
        """, (steps,))
        to_rollback = cur.fetchall()

    if not to_rollback:
        print("Nothing to roll back.")
        conn.close()
        return

    for ver, desc, filename in to_rollback:
        down_file = MIGRATIONS_DIR / filename.replace(".sql", ".down.sql")
        if not down_file.exists():
            print(f"  ✗ No rollback script for V{ver}: {desc}")
            print(f"    Expected: {down_file}")
            continue

        print(f"  Rolling back V{ver:03d}: {desc}...", end=" ")
        sql = down_file.read_text()
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                cur.execute(f"DELETE FROM {SCHEMA_TABLE} WHERE version = %s", (ver,))
            conn.commit()
            print("✓")
        except Exception as e:
            conn.rollback()
            print(f"✗ FAILED: {e}")
            conn.close()
            sys.exit(1)

    conn.close()


def cmd_create(description: str):
    """Create a new migration file."""
    existing = sorted(
        [f for f in MIGRATIONS_DIR.glob("V*.sql") if ".down." not in f.name]
    )
    next_ver = 1
    if existing:
        versions = [_parse_version(f.name)[0] for f in existing if _parse_version(f.name)]
        next_ver = max(versions) + 1

    slug = description.lower().replace(" ", "_").replace("-", "_")
    filename = f"V{next_ver:03d}__{slug}.sql"
    down_filename = f"V{next_ver:03d}__{slug}.down.sql"

    up_path = MIGRATIONS_DIR / filename
    down_path = MIGRATIONS_DIR / down_filename

    up_path.write_text(f"""-- Migration: V{next_ver:03d} {description}
-- Created: {datetime.utcnow().isoformat()}
-- Apply: ALTER TABLE ... / CREATE TABLE ... / etc.

""")
    down_path.write_text(f"""-- Rollback: V{next_ver:03d} {description}
-- Created: {datetime.utcnow().isoformat()}
-- Undo the changes from the up migration

""")
    print(f"Created migration files:")
    print(f"  UP:   {up_path}")
    print(f"  DOWN: {down_path}")


def cmd_validate():
    """Validate that applied migrations match files on disk (checksum check)."""
    conn = _get_conn()
    _ensure_migrations_table(conn)

    with conn.cursor() as cur:
        cur.execute(f"SELECT version, filename, checksum FROM {SCHEMA_TABLE}")
        applied = {row[0]: (row[1], row[2]) for row in cur.fetchall()}

    issues = []
    for ver, (filename, stored_checksum) in applied.items():
        path = MIGRATIONS_DIR / filename
        if not path.exists():
            issues.append(f"V{ver}: migration file missing ({filename})")
            continue
        actual_checksum = _file_checksum(path)
        if actual_checksum != stored_checksum:
            issues.append(f"V{ver}: checksum mismatch — file was modified after applying")

    if issues:
        print(f"✗ Validation FAILED ({len(issues)} issues):")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
    else:
        print(f"✓ All {len(applied)} applied migrations validated successfully.")

    conn.close()


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
    parser = argparse.ArgumentParser(description="Hospital OS Database Migration Manager")
    sub = parser.add_subparsers(dest="command", metavar="command")

    sub.add_parser("status", help="Show migration status")

    p_apply = sub.add_parser("apply", help="Apply pending migrations")
    p_apply.add_argument("--target", type=int, help="Apply up to this version")
    p_apply.add_argument("--dry-run", action="store_true")

    p_roll = sub.add_parser("rollback", help="Roll back last N migrations")
    p_roll.add_argument("--steps", type=int, default=1)

    p_create = sub.add_parser("create", help="Create a new migration file")
    p_create.add_argument("description", help="Migration description")

    sub.add_parser("validate", help="Validate applied migration checksums")

    args = parser.parse_args()

    if args.command == "status":
        cmd_status()
    elif args.command == "apply":
        cmd_apply(target_version=getattr(args, "target", None),
                   dry_run=getattr(args, "dry_run", False))
    elif args.command == "rollback":
        cmd_rollback(steps=args.steps)
    elif args.command == "create":
        cmd_create(args.description)
    elif args.command == "validate":
        cmd_validate()
    else:
        parser.print_help()

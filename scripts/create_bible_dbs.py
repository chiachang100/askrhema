import json
import sqlite3
from pathlib import Path
import argparse

def create_db_from_json(json_path: Path, db_path: Path) -> None:
    """Load a JSON file and write its contents into a SQLite database."""
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        verses = json.load(f)

    if not verses:
        raise ValueError("JSON file is empty")

    # Remove existing DB if it exists
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Create table
    cursor.execute("""
        CREATE TABLE verses (
            id INTEGER PRIMARY KEY,
            book TEXT NOT NULL,
            chapter INTEGER NOT NULL,
            verse INTEGER NOT NULL,
            text TEXT NOT NULL,
            testament TEXT NOT NULL,
            category TEXT NOT NULL
        )
    """)

    # Create indexes
    cursor.execute("CREATE INDEX idx_book ON verses (book)")
    cursor.execute("CREATE INDEX idx_testament ON verses (testament)")

    # Insert all verses
    for v in verses:
        cursor.execute(
            """
            INSERT INTO verses (id, book, chapter, verse, text, testament, category)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                v["id"],
                v["book"],
                v["chapter"],
                v["verse"],
                v["text"],
                v["testament"],
                v["category"],
            ),
        )

    conn.commit()
    conn.close()
    print(f"✅ Created {db_path} with {len(verses)} verses")

def main():
    parser = argparse.ArgumentParser(
        description="Convert a Bible JSON file to a SQLite database."
    )
    parser.add_argument(
        "input_json",
        type=Path,
        help="Path to the input JSON file (must match AskRhema schema)."
    )
    parser.add_argument(
        "output_db",
        type=Path,
        help="Path where the output SQLite database will be created."
    )
    args = parser.parse_args()

    try:
        create_db_from_json(args.input_json, args.output_db)
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
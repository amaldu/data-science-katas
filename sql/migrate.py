import psycopg2
from pathlib import Path

DB_NAME = "postgres"
DB_USER = "user"
DB_PASSWORD = "password"
DB_HOST = "localhost"   
DB_PORT = 5432

def run_migrations():
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cur = conn.cursor()

    migrations_path = Path("sql/migrations")
    sql_files = sorted(migrations_path.glob("*.sql"))

    if not sql_files:
        print("No SQL migration files found in sql/migrations/")
        return

    for file in sql_files:
        print(f"Applying migration: {file.name}")
        with open(file, "r") as f:
            sql = f.read()
            try:
                cur.execute(sql)
                conn.commit()
                print(f" Migration {file.name} applied successfully")
            except Exception as e:
                conn.rollback()
                print(f" Failed to apply {file.name}: {e}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    run_migrations()

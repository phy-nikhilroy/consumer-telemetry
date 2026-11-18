"""
DuckDB connection and schema management for consumer telemetry warehouse.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional, Union
import duckdb

from src.config import DEFAULT_DB_PATH, SCHEMA_SQL_PATH


class DatabaseManager:
    """
    Manages DuckDB connections, schema initialization, and transactional queries.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        self.db_path = Path(db_path) if db_path and str(db_path) != ":memory:" else db_path
        self._shared_connection: Optional[duckdb.DuckDBPyConnection] = None
        if self.db_path and str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        elif str(self.db_path) == ":memory:" or self.db_path is None:
            # Maintain persistent in-memory connection across manager lifetime
            self._shared_connection = duckdb.connect(":memory:")

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """
        Returns an active DuckDB connection.
        For in-memory instances, returns a cursor to the shared session.
        """
        if self._shared_connection is not None:
            return self._shared_connection.cursor()
        target = str(self.db_path)
        return duckdb.connect(target)

    def init_schema(self, schema_file: Optional[Union[str, Path]] = None) -> None:
        """
        Executes schema DDL to create star schema tables, then migrates the warehouse to
        the current schema version, so an older warehouse gains the new tables and columns.
        """
        from src.features.migrations import migrate

        sql_file = Path(schema_file) if schema_file else SCHEMA_SQL_PATH
        if not sql_file.exists():
            raise FileNotFoundError(f"Schema file not found at {sql_file}")

        with open(sql_file, "r", encoding="utf-8") as f:
            ddl_statements = f.read()

        con = self.get_connection()
        try:
            con.execute(ddl_statements)
            migrate(con)
        finally:
            if self._shared_connection is None:
                con.close()

    def execute_query(self, query: str, params: Optional[list] = None) -> list:
        """
        Executes an arbitrary query and returns fetched rows.
        """
        con = self.get_connection()
        try:
            if params:
                return con.execute(query, params).fetchall()
            return con.execute(query).fetchall()
        finally:
            con.close()


def get_connection(db_path: Optional[Union[str, Path]] = None) -> duckdb.DuckDBPyConnection:
    """
    Utility function to obtain a DuckDB connection.
    """
    mgr = DatabaseManager(db_path=db_path or DEFAULT_DB_PATH)
    return mgr.get_connection()


def init_db(db_path: Optional[Union[str, Path]] = None) -> None:
    """
    Initializes database schema at the target path.
    """
    mgr = DatabaseManager(db_path=db_path or DEFAULT_DB_PATH)
    mgr.init_schema()


@contextmanager
def get_db_session(db_path: Optional[Union[str, Path]] = None) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """
    Context manager providing safe connection lifecycle management.
    Ensures connection is cleanly closed upon exit if not persistent in-memory.
    """
    mgr = DatabaseManager(db_path=db_path or DEFAULT_DB_PATH)
    con = mgr.get_connection()
    try:
        yield con
    finally:
        if mgr._shared_connection is None:
            con.close()

import os
import psycopg2
from psycopg2 import OperationalError

# Add this import for reading the .env file
from dotenv import load_dotenv


import os
import psycopg2
from psycopg2 import OperationalError


class PostgresConnector:
    """
    A Singleton class for managing the PostgreSQL database connection.
    Connection parameters are read from environment variables.
    """

    _instance = None
    _connection = None

    def __new__(cls):
        """
        Implementation of the Singleton pattern.
        Ensures only one instance of the class is created.
        """
        if cls._instance is None:
            cls._instance = super(PostgresConnector, cls).__new__(cls)
            # Initialization happens only once here
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Reads environment variables (from ENV or .env file) and connects."""
        load_dotenv()
        # 1. Read Credentials from Environment Variables (os.environ)
        try:
            # We are now reading DIRECTLY from os.environ, which holds
            # both system env vars AND those loaded from .env
            self.db_name = os.environ["PG_DB_NAME"]
            self.db_user = os.environ["PG_USER"]
            self.db_password = os.environ["PG_PASSWORD"]
            self.db_host = os.environ.get("PG_HOST", "localhost")
            self.db_port = os.environ.get("PG_PORT", "5432")

        except KeyError as e:
            raise EnvironmentError(
                f"Missing required environment variable: {e}. Check your shell environment or .env file."
            ) from e
        # 2. Establish Connection
        self.connect()

    def connect(self):
        """Establishes or re-establishes the database connection."""
        if self._connection is not None:
            # If already connected, close the old connection first (optional, but clean)
            try:
                self._connection.close()
                self._connection = None
            except OperationalError:
                pass  # Connection might already be closed/broken

        try:
            self._connection = psycopg2.connect(
                database=self.db_name,
                user=self.db_user,
                password=self.db_password,
                host=self.db_host,
                port=self.db_port,
            )
            print(
                f"✅ Database connected successfully to {self.db_host}:{self.db_port}/{self.db_name}"
            )
        except OperationalError as e:
            print(f"❌ Connection attempt failed: {e}")
            self._connection = None
            # Re-raise to stop execution if connection is critical
            raise

    def close(self):
        """Closes the active database connection."""
        if self._connection:
            try:
                self._connection.close()
                self._connection = None
                print("👋 Database connection closed.")
            except OperationalError as e:
                print(f"Error closing connection: {e}")

    def get_distance_between_two_coordinates_within_route(
        self, route_id, start_lat, start_lon, end_lat, end_lon
    ):
        """Return the length (meters) of the route segment between two coordinates along a route.

        This executes a PostGIS query that:
        - Projects the input coordinates into metric SRID 22992,
        - Locates the positions along the stored route geometry line,
        - Extracts the line substring between those two positions and returns its length.

        Parameters are passed safely via psycopg2 parameter substitution.
        """

        # Convert the query to use psycopg2 placeholders (%s)
        template_query = """SELECT
                    ST_Length(
                        ST_LineSubstring(
                        line_geom,
                        LEAST(ST_LineLocatePoint(line_geom, p.g1), ST_LineLocatePoint(line_geom, p.g2)),
                        GREATEST(ST_LineLocatePoint(line_geom, p.g1), ST_LineLocatePoint(line_geom, p.g2))
                        )
                    ) AS distance_m
                    FROM route_geometry rg
                    CROSS JOIN LATERAL (
                    SELECT
                        ST_Transform(ST_SetSRID(ST_MakePoint(%s, %s), 4326), 22992) AS g1,
                        ST_Transform(ST_SetSRID(ST_MakePoint(%s, %s), 4326), 22992) AS g2
                    ) AS p
                    CROSS JOIN LATERAL (
                    SELECT COALESCE(rg.geom_22992, ST_Transform(rg.geom_4326, 22992)) AS line_geom
                    ) AS lg
                    WHERE rg.route_geom_id = %s;
                    """

        # Ensure connection exists
        if self._connection is None:
            # Try to reconnect once
            try:
                self.connect()
            except Exception as e:
                raise RuntimeError(f"Database connection not available: {e}") from e

        cur = None
        try:
            cur = self._connection.cursor()
            params = (start_lon, start_lat, end_lon, end_lat, route_id)
            cur.execute(template_query, params)
            row = cur.fetchone()
            # row[0] will be the length in the SRID units (meters for 22992)
            if row and row[0] is not None:
                return float(row[0])
            return None
        except Exception as e:
            # If the error appears to be a connection issue, try reconnecting once
            try:
                self.connect()
                cur = self._connection.cursor()
                params = (start_lon, start_lat, end_lon, end_lat, route_id)
                cur.execute(template_query, params)
                row = cur.fetchone()
                if row and row[0] is not None:
                    return float(row[0])
                return None
            except Exception:
                raise
        finally:
            if cur:
                try:
                    cur.close()
                except Exception:
                    pass

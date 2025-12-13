import os
import psycopg2
from psycopg2 import OperationalError
import logging

# Add this import for reading the .env file
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            logger.info(f"Executing query with params: {params}")
            cur.execute(template_query, params)
            row = cur.fetchone()
            # row[0] will be the length in the SRID units (meters for 22992)
            if row and row[0] is not None:
                return float(row[0])
            return None
        except Exception as e:
            logger.error(
                f"Database error in get_distance_between_two_coordinates_within_route: {e}"
            )
            logger.error(f"Query: {template_query}")
            logger.error(f"Params: {params}")
            logger.error(f"Full exception details:", exc_info=True)
            # If the error appears to be a connection issue, try reconnecting once
            try:
                logger.info("Attempting to reconnect...")
                self.connect()
                cur = self._connection.cursor()
                params = (start_lon, start_lat, end_lon, end_lat, route_id)
                cur.execute(template_query, params)
                row = cur.fetchone()
                if row and row[0] is not None:
                    return float(row[0])
                return None
            except Exception as retry_e:
                logger.error(f"Retry failed: {retry_e}", exc_info=True)
                raise
        finally:
            if cur:
                try:
                    cur.close()
                except Exception:
                    pass

    def get_distance_between_two_stops_within_route(
        self, trip_id, start_stop, end_stop
    ):
        """Return the distance (meters) between two stops along a route.

        This queries the database for the actual stop coordinates, then uses the
        coordinate-based distance calculation.

        Args:
            route_id: The route geometry ID
            start_stop: The starting stop ID
            end_stop: The ending stop ID

        Returns:
            float: Distance in meters, or None if stops/route not found
        """
        # First, get the coordinates of both stops
        stops_query = """
                        SELECT 
                        t.gtfs_trip_id,
                        start_rs.stop_sequence AS start_sequence,
                        end_rs.stop_sequence AS end_sequence,
                        start_stop.name AS start_stop_name,
                        end_stop.name AS end_stop_name,
                        ST_Distance(
                            start_stop.geom_22992,
                            end_stop.geom_22992
                        ) AS straight_line_distance_meters,
                        -- Distance along the route geometry (if available)
                        CASE 
                            WHEN rg.geom_22992 IS NOT NULL THEN
                                ST_Length(
                                    ST_LineSubstring(
                                        rg.geom_22992,
                                        LEAST(
                                            ST_LineLocatePoint(rg.geom_22992, start_stop.geom_22992),
                                            ST_LineLocatePoint(rg.geom_22992, end_stop.geom_22992)
                                        ),
                                        GREATEST(
                                            ST_LineLocatePoint(rg.geom_22992, start_stop.geom_22992),
                                            ST_LineLocatePoint(rg.geom_22992, end_stop.geom_22992)
                                        )
                                    )
                                )
                            ELSE NULL
                        END AS route_distance_meters
                    FROM trip t
                    JOIN route_stop start_rs ON t.trip_id = start_rs.trip_id
                    JOIN route_stop end_rs ON t.trip_id = end_rs.trip_id
                    JOIN "stop" start_stop ON start_rs.stop_id = start_stop.stop_id
                    JOIN "stop" end_stop ON end_rs.stop_id = end_stop.stop_id
                    LEFT JOIN route_geometry rg ON t.route_geom_id = rg.route_geom_id
                    WHERE t.gtfs_trip_id = %s  -- Replace with your GTFS trip ID
                    AND start_rs.stop_sequence = %s  -- Start stop sequence
                    AND end_rs.stop_sequence = %s;   -- End stop sequence
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
            params = (trip_id, start_stop, end_stop)
            logger.info(
                f"Executing get_distance_between_two_stops_within_route with params: trip_id={trip_id}, start_stop={start_stop}, end_stop={end_stop}"
            )
            logger.debug(f"Full query params: {params}")
            cur.execute(stops_query, params)
            row = cur.fetchone()
            logger.info(f"Fecthed data: {row}")
            # row[0] will be the length in the SRID units (meters for 22992)
            if row and row[0] is not None:
                logger.info(f"Successfully calculated distance: {row[0]} meters")
                return float(row[0])
            else:
                logger.warning(
                    f"No distance found for gtfs_trip_id={trip_id}, start_stop_id={start_stop}, end_stop_id={end_stop}"
                )
                logger.warning(
                    "Possible reasons: trip not found, stops not in trip, or missing geometry data"
                )
                return None
        except Exception as e:
            logger.error(
                f"Database error in get_distance_between_two_stops_within_route: {e}"
            )
            logger.error(f"Query: {stops_query}")
            logger.error(
                f"Params: trip_id={trip_id}, start_stop={start_stop}, end_stop={end_stop}"
            )
            logger.error(f"Full params tuple: {params}")
            logger.error(f"Full exception details:", exc_info=True)
            # If the error appears to be a connection issue, try reconnecting once
            try:
                logger.info("Attempting to reconnect...")
                self.connect()
                cur = self._connection.cursor()
                params = (trip_id, start_stop, end_stop)
                cur.execute(stops_query, params)
                row = cur.fetchone()
                if row and row[0] is not None:
                    return float(row[0])
                return None
            except Exception as retry_e:
                logger.error(f"Retry failed: {retry_e}", exc_info=True)
                raise
        finally:
            if cur:
                try:
                    cur.close()
                except Exception:
                    pass

import os
import psycopg2
from psycopg2 import OperationalError
import logging

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import os
import psycopg2
from psycopg2 import OperationalError


class PostgresConnector:
    _instance = None
    _connection = None

    def __new__(cls):
        """
        Singleton pattern
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
        """Establishes or re-establishes the database connection with retry logic."""
        if self._connection is not None:
            # If already connected, close the old connection first (optional, but clean)
            try:
                self._connection.close()
                self._connection = None
            except OperationalError:
                pass  # Connection might already be closed/broken

        # Retry logic for container startup delays
        import time

        max_retries = 30
        retry_delay = 2  # seconds
        for attempt in range(1, max_retries + 1):
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
                return  # Success, exit retry loop
            except OperationalError as e:
                if attempt < max_retries:
                    print(
                        f"⏳ Connection attempt {attempt}/{max_retries} failed, retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    print(f"❌ All {max_retries} connection attempts failed: {e}")
                    self._connection = None
                    raise  # Re-raise after all retries exhausted

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
        self, trip_id, start_lat, start_lon, end_lat, end_lon
    ):
        """Return the length (meters) of the route segment between two coordinates along a route.

        This executes a PostGIS query that:
        - Projects the input coordinates into metric SRID 22992,
        - Locates the positions along the stored route geometry line,
        - Extracts the line substring between those two positions and returns its length.

        Parameters are passed safely via psycopg2 parameter substitution.
        """

        template_query = """
                            WITH trip_info AS (
                                SELECT 
                                    t.trip_id,
                                    t.route_geom_id,
                                    rg.geom_22992
                                FROM trip t
                                JOIN route_geometry rg ON t.route_geom_id = rg.route_geom_id
                                WHERE t.gtfs_trip_id = %s  -- GTFS trip ID
                            ),
                            point_positions AS (
                                SELECT 
                                    ST_LineLocatePoint(
                                        ti.geom_22992,
                                        ST_Transform(ST_SetSRID(ST_MakePoint(%s, %s), 4326), 22992) -- start lon, start lat
                                    ) AS start_fraction,
                                    ST_LineLocatePoint(
                                        ti.geom_22992,
                                        ST_Transform(ST_SetSRID(ST_MakePoint(%s, %s), 4326), 22992) -- end lon, end lat
                                    ) AS end_fraction
                                FROM trip_info ti
                            )
                            SELECT 
                                ST_Length(
                                    ST_LineSubstring(
                                        ti.geom_22992,
                                        LEAST(pp.start_fraction, pp.end_fraction),
                                        GREATEST(pp.start_fraction, pp.end_fraction)
                                    )
                                ) AS distance_meters
                            FROM trip_info ti
                            CROSS JOIN point_positions pp;
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
            params = (trip_id, start_lon, start_lat, end_lon, end_lat)
            cur.execute(template_query, params)
            row = cur.fetchone()
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
            try:
                logger.info("Attempting to reconnect...")
                self.connect()
                cur = self._connection.cursor()
                params = (trip_id, start_lon, start_lat, end_lon, end_lat)
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
                        WITH trip_info AS (
                            SELECT 
                                t.trip_id,
                                rg.geom_22992
                            FROM trip t
                            JOIN route_geometry rg ON t.route_geom_id = rg.route_geom_id
                            WHERE t.gtfs_trip_id = %s
                        ),
                        stop_positions AS (
                            SELECT 
                                ST_LineLocatePoint(ti.geom_22992, s.geom_22992) AS location_fraction
                            FROM route_stop rs
                            JOIN "stop" s ON rs.stop_id = s.stop_id
                            JOIN trip_info ti ON rs.trip_id = ti.trip_id
                            WHERE s.gtfs_stop_id IN (%s, %s)
                        )
                        SELECT 
                            ST_Length(
                                ST_LineSubstring(
                                    ti.geom_22992,
                                    (SELECT MIN(location_fraction) FROM stop_positions),
                                    (SELECT MAX(location_fraction) FROM stop_positions)
                                )
                            ) AS distance_meters
                        FROM trip_info ti;
                """

        start_stop = str(start_stop)
        end_stop = str(end_stop)

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

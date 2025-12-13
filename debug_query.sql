-- Debug queries to check your data
-- 1. Check if the trip exists
SELECT trip_id,
    gtfs_trip_id,
    route_id,
    route_geom_id
FROM trip
WHERE gtfs_trip_id = 'FUo5FExiKwUTpyTUJYA7R-07:00:00';
-- 2. Check if the stops exist
SELECT stop_id,
    gtfs_stop_id,
    name
FROM stop
WHERE stop_id IN (263, 45);
-- 3. Check if route_stops exist for this trip
SELECT rs.route_stop_id,
    rs.trip_id,
    rs.stop_id,
    rs.stop_sequence,
    s.gtfs_stop_id,
    s.name
FROM route_stop rs
    JOIN stop s ON s.stop_id = rs.stop_id
WHERE rs.trip_id = (
        SELECT trip_id
        FROM trip
        WHERE gtfs_trip_id = 'FUo5FExiKwUTpyTUJYA7R-07:00:00'
    )
ORDER BY rs.stop_sequence;
-- 4. Check if the trip has a route_geom_id
SELECT t.trip_id,
    t.gtfs_trip_id,
    t.route_geom_id,
    CASE
        WHEN rg.route_geom_id IS NOT NULL THEN 'geometry exists'
        ELSE 'NO GEOMETRY'
    END as geom_status
FROM trip t
    LEFT JOIN route_geometry rg ON rg.route_geom_id = t.route_geom_id
WHERE t.gtfs_trip_id = 'FUo5FExiKwUTpyTUJYA7R-07:00:00';
-- 5. Run the actual query with the parameters
SELECT ST_Length(
        ST_LineSubstring(
            rg.geom_22992,
            LEAST(
                ST_LineLocatePoint(rg.geom_22992, s1.geom_22992),
                ST_LineLocatePoint(rg.geom_22992, s2.geom_22992)
            ),
            GREATEST(
                ST_LineLocatePoint(rg.geom_22992, s1.geom_22992),
                ST_LineLocatePoint(rg.geom_22992, s2.geom_22992)
            )
        )
    ) AS distance_meters
FROM trip t
    JOIN route_geometry rg ON rg.route_geom_id = t.route_geom_id
    CROSS JOIN LATERAL (
        SELECT s.geom_22992
        FROM route_stop rs
            JOIN stop s ON s.stop_id = rs.stop_id
        WHERE rs.trip_id = t.trip_id
            AND rs.stop_id = 263
        LIMIT 1
    ) s1
    CROSS JOIN LATERAL (
        SELECT s.geom_22992
        FROM route_stop rs
            JOIN stop s ON s.stop_id = rs.stop_id
        WHERE rs.trip_id = t.trip_id
            AND rs.stop_id = 45
        LIMIT 1
    ) s2
WHERE t.gtfs_trip_id = 'FUo5FExiKwUTpyTUJYA7R-07:00:00';
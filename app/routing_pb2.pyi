from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class HealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("status", "message")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    status: str
    message: str
    def __init__(self, status: _Optional[str] = ..., message: _Optional[str] = ...) -> None: ...

class RouteRequest(_message.Message):
    __slots__ = ("start_lon", "start_lat", "end_lon", "end_lat", "max_transfers", "walking_cutoff")
    START_LON_FIELD_NUMBER: _ClassVar[int]
    START_LAT_FIELD_NUMBER: _ClassVar[int]
    END_LON_FIELD_NUMBER: _ClassVar[int]
    END_LAT_FIELD_NUMBER: _ClassVar[int]
    MAX_TRANSFERS_FIELD_NUMBER: _ClassVar[int]
    WALKING_CUTOFF_FIELD_NUMBER: _ClassVar[int]
    start_lon: float
    start_lat: float
    end_lon: float
    end_lat: float
    max_transfers: int
    walking_cutoff: float
    def __init__(self, start_lon: _Optional[float] = ..., start_lat: _Optional[float] = ..., end_lon: _Optional[float] = ..., end_lat: _Optional[float] = ..., max_transfers: _Optional[int] = ..., walking_cutoff: _Optional[float] = ...) -> None: ...

class StopInfo(_message.Message):
    __slots__ = ("stop_id", "name", "coord")
    STOP_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    COORD_FIELD_NUMBER: _ClassVar[int]
    stop_id: str
    name: str
    coord: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, stop_id: _Optional[str] = ..., name: _Optional[str] = ..., coord: _Optional[_Iterable[float]] = ...) -> None: ...

class WalkLeg(_message.Message):
    __slots__ = ("type", "distance_meters", "duration_minutes", "path")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    DISTANCE_METERS_FIELD_NUMBER: _ClassVar[int]
    DURATION_MINUTES_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    type: str
    distance_meters: int
    duration_minutes: int
    path: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, type: _Optional[str] = ..., distance_meters: _Optional[int] = ..., duration_minutes: _Optional[int] = ..., path: _Optional[_Iterable[str]] = ...) -> None: ...

class TripLeg(_message.Message):
    __slots__ = ("type", "trip_id", "mode", "route_name", "route_short_name", "headsign", "fare", "duration_minutes", "to", "path")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    TRIP_ID_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    ROUTE_NAME_FIELD_NUMBER: _ClassVar[int]
    ROUTE_SHORT_NAME_FIELD_NUMBER: _ClassVar[int]
    HEADSIGN_FIELD_NUMBER: _ClassVar[int]
    FARE_FIELD_NUMBER: _ClassVar[int]
    DURATION_MINUTES_FIELD_NUMBER: _ClassVar[int]
    FROM_FIELD_NUMBER: _ClassVar[int]
    TO_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    type: str
    trip_id: str
    mode: str
    route_name: str
    route_short_name: str
    headsign: str
    fare: float
    duration_minutes: int
    to: StopInfo
    path: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, type: _Optional[str] = ..., trip_id: _Optional[str] = ..., mode: _Optional[str] = ..., route_name: _Optional[str] = ..., route_short_name: _Optional[str] = ..., headsign: _Optional[str] = ..., fare: _Optional[float] = ..., duration_minutes: _Optional[int] = ..., to: _Optional[_Union[StopInfo, _Mapping]] = ..., path: _Optional[_Iterable[str]] = ..., **kwargs) -> None: ...

class TransferLeg(_message.Message):
    __slots__ = ("type", "from_trip_id", "to_trip_id", "from_trip_name", "to_trip_name", "walking_distance_meters", "duration_minutes", "path")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    FROM_TRIP_ID_FIELD_NUMBER: _ClassVar[int]
    TO_TRIP_ID_FIELD_NUMBER: _ClassVar[int]
    FROM_TRIP_NAME_FIELD_NUMBER: _ClassVar[int]
    TO_TRIP_NAME_FIELD_NUMBER: _ClassVar[int]
    WALKING_DISTANCE_METERS_FIELD_NUMBER: _ClassVar[int]
    DURATION_MINUTES_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    type: str
    from_trip_id: str
    to_trip_id: str
    from_trip_name: str
    to_trip_name: str
    walking_distance_meters: int
    duration_minutes: int
    path: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, type: _Optional[str] = ..., from_trip_id: _Optional[str] = ..., to_trip_id: _Optional[str] = ..., from_trip_name: _Optional[str] = ..., to_trip_name: _Optional[str] = ..., walking_distance_meters: _Optional[int] = ..., duration_minutes: _Optional[int] = ..., path: _Optional[_Iterable[str]] = ...) -> None: ...

class Leg(_message.Message):
    __slots__ = ("walk", "trip", "transfer")
    WALK_FIELD_NUMBER: _ClassVar[int]
    TRIP_FIELD_NUMBER: _ClassVar[int]
    TRANSFER_FIELD_NUMBER: _ClassVar[int]
    walk: WalkLeg
    trip: TripLeg
    transfer: TransferLeg
    def __init__(self, walk: _Optional[_Union[WalkLeg, _Mapping]] = ..., trip: _Optional[_Union[TripLeg, _Mapping]] = ..., transfer: _Optional[_Union[TransferLeg, _Mapping]] = ...) -> None: ...

class JourneySummary(_message.Message):
    __slots__ = ("total_time_minutes", "total_distance_meters", "walking_distance_meters", "transfers", "cost", "modes")
    TOTAL_TIME_MINUTES_FIELD_NUMBER: _ClassVar[int]
    TOTAL_DISTANCE_METERS_FIELD_NUMBER: _ClassVar[int]
    WALKING_DISTANCE_METERS_FIELD_NUMBER: _ClassVar[int]
    TRANSFERS_FIELD_NUMBER: _ClassVar[int]
    COST_FIELD_NUMBER: _ClassVar[int]
    MODES_FIELD_NUMBER: _ClassVar[int]
    total_time_minutes: int
    total_distance_meters: int
    walking_distance_meters: int
    transfers: int
    cost: float
    modes: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, total_time_minutes: _Optional[int] = ..., total_distance_meters: _Optional[int] = ..., walking_distance_meters: _Optional[int] = ..., transfers: _Optional[int] = ..., cost: _Optional[float] = ..., modes: _Optional[_Iterable[str]] = ...) -> None: ...

class Journey(_message.Message):
    __slots__ = ("id", "text_summary", "summary", "legs")
    ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_SUMMARY_FIELD_NUMBER: _ClassVar[int]
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    LEGS_FIELD_NUMBER: _ClassVar[int]
    id: str
    text_summary: str
    summary: JourneySummary
    legs: _containers.RepeatedCompositeFieldContainer[Leg]
    def __init__(self, id: _Optional[str] = ..., text_summary: _Optional[str] = ..., summary: _Optional[_Union[JourneySummary, _Mapping]] = ..., legs: _Optional[_Iterable[_Union[Leg, _Mapping]]] = ...) -> None: ...

class RouteResponse(_message.Message):
    __slots__ = ("num_journeys", "journeys", "start_trips_found", "end_trips_found")
    NUM_JOURNEYS_FIELD_NUMBER: _ClassVar[int]
    JOURNEYS_FIELD_NUMBER: _ClassVar[int]
    START_TRIPS_FOUND_FIELD_NUMBER: _ClassVar[int]
    END_TRIPS_FOUND_FIELD_NUMBER: _ClassVar[int]
    num_journeys: int
    journeys: _containers.RepeatedCompositeFieldContainer[Journey]
    start_trips_found: int
    end_trips_found: int
    def __init__(self, num_journeys: _Optional[int] = ..., journeys: _Optional[_Iterable[_Union[Journey, _Mapping]]] = ..., start_trips_found: _Optional[int] = ..., end_trips_found: _Optional[int] = ...) -> None: ...

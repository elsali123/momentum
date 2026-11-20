"""Source readers (container formats) and mappers (per-customer column semantics)."""

from robot_ingest.sources.base import Mapper
from robot_ingest.sources.droid import DroidMapper
from robot_ingest.sources.rh20t import Rh20tMapper

# configs/sources.toml `mapper = "..."` -> class
MAPPERS: dict[str, type[Mapper]] = {
    "rh20t": Rh20tMapper,
    "droid": DroidMapper,
}

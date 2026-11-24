"""High-rise office geometry + parametric IDF rendering (forked from demo3/run_analysis.py).

Building: rectangular tower, 36 m x 36 m plate, 3.9 m floor height, FLOOR_COUNT floors.
Zoning per floor: 4 perimeter zones (S/N/E/W, 5 m deep) + 1 core => 5 zones/floor.
Outlines carry T-junction vertices so interior partitions share identical edges and
are auto-detected by the shared-edge matcher (reused from demo3).
HVAC: ZoneHVAC:IdealLoadsAirSystem on every zone (pure loads, like 论文5).
Fenestration: ribbon window per perimeter exterior wall, sized so window/wall area = WWR.
Setpoints: occupied cooling/heating setpoints injected via template placeholders.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from building_project import BuildingProjectSpec


ROOT = Path(__file__).resolve().parent
TEMPLATE_PATH = ROOT / "office_template.idf.in"

# ---- Building geometry constants ----
BUILDING_X_M = 36.0          # east-west extent (x)
BUILDING_Y_M = 36.0          # north-south extent (y); south = y=0 face
FLOOR_HEIGHT_M = 3.9
FLOOR_COUNT = 15
PERIM_DEPTH_M = 5.0
TOTAL_HEIGHT_M = FLOOR_HEIGHT_M * FLOOR_COUNT
FLOOR_AREA_M2 = BUILDING_X_M * BUILDING_Y_M * FLOOR_COUNT

# inner core boundaries
_X0, _X1 = PERIM_DEPTH_M, BUILDING_X_M - PERIM_DEPTH_M   # 5, 31
_Y0, _Y1 = PERIM_DEPTH_M, BUILDING_Y_M - PERIM_DEPTH_M   # 5, 31

# primary exterior orientation that carries the window for each perimeter kind
PRIMARY_ORIENTATION = {
    "perim_s": "south",
    "perim_n": "north",
    "perim_e": "east",
    "perim_w": "west",
}

# weather lookup (prefer local 7/weather, fall back to demo3/weather)
WEATHER_DIRS = [ROOT / "weather", ROOT.parent / "demo3" / "weather"]

ENERGYPLUS_CANDIDATES = [
    r"D:\Program Files\EnergyPlusV25-2-0\energyplus.exe",
]


@dataclass(frozen=True)
class ZoneSpec:
    name: str
    label: str
    kind: str
    floor: int
    outline: tuple[tuple[float, float], ...]
    patches: tuple[tuple[tuple[float, float], ...], ...]
    z0: float
    z1: float
    # Which compass wall (if any) carries the ribbon window for this zone.
    # Set explicitly at zone-construction time (rectangle: from the static
    # PRIMARY_ORIENTATION table; l_shape: from the notch-corner-adjusted map)
    # so fenestration_block never has to guess it back out of the geometry.
    primary_orientation: str | None = None

    @property
    def area(self) -> float:
        return sum(_polygon_area(patch) for patch in self.patches)

    @property
    def volume(self) -> float:
        return self.area * (self.z1 - self.z0)

    @property
    def is_perimeter(self) -> bool:
        # Core zones carry no window orientation (primary_orientation=None for
        # both the rectangle "core" kind and the L-shape "wa_core"/"wb_core"
        # kinds); every perimeter kind has one. Keying off that instead of a
        # "perim_" substring keeps this correct for both zoning schemes.
        return self.primary_orientation is not None


@dataclass(frozen=True)
class GeometryConfig:
    plan_shape: str = "rectangle"          # "rectangle" | "l_shape"
    length_m: float = BUILDING_X_M
    width_m: float = BUILDING_Y_M
    ground_floor_height_m: float = FLOOR_HEIGHT_M
    typical_floor_height_m: float = FLOOR_HEIGHT_M
    floor_count: int = FLOOR_COUNT
    perimeter_depth_m: float = PERIM_DEPTH_M
    # l_shape only: a notch_length_m x notch_width_m rectangle is removed
    # from one corner (notch_corner) of the length_m x width_m bounding box.
    notch_corner: str = "ne"               # "ne" | "nw" | "se" | "sw"
    notch_length_m: float = 0.0
    notch_width_m: float = 0.0

    @property
    def total_height_m(self) -> float:
        return self.ground_floor_height_m + self.typical_floor_height_m * max(0, self.floor_count - 1)

    @property
    def notch_area_m2(self) -> float:
        if self.plan_shape != "l_shape":
            return 0.0
        return self.notch_length_m * self.notch_width_m

    @property
    def floor_plate_area_m2(self) -> float:
        return self.length_m * self.width_m - self.notch_area_m2

    @property
    def floor_area_m2(self) -> float:
        return self.floor_plate_area_m2 * self.floor_count


DEFAULT_GEOMETRY = GeometryConfig()


def geometry_from_project_spec(spec: BuildingProjectSpec) -> GeometryConfig:
    g = spec.geometry
    return GeometryConfig(
        plan_shape=str(g.plan_shape),
        length_m=float(g.length_m),
        width_m=float(g.width_m),
        ground_floor_height_m=float(g.ground_floor_height_m),
        typical_floor_height_m=float(g.typical_floor_height_m),
        floor_count=int(g.floor_count),
        perimeter_depth_m=float(g.perimeter_depth_m),
        notch_corner=str(g.notch_corner),
        notch_length_m=float(g.notch_length_m),
        notch_width_m=float(g.notch_width_m),
    )


def _floor_z_range(floor: int, config: GeometryConfig = DEFAULT_GEOMETRY) -> tuple[float, float]:
    """Z-range for a given floor, honoring the ground/typical floor-height split."""
    if floor <= 1:
        return 0.0, config.ground_floor_height_m
    z0 = config.ground_floor_height_m + (floor - 2) * config.typical_floor_height_m
    return z0, z0 + config.typical_floor_height_m


def fmt(value: float) -> str:
    return f"{value:.3f}"


def idf_object(object_type: str, fields: list[str]) -> str:
    if not fields:
        raise ValueError(f"{object_type} requires at least one field")
    rendered = [str(field) for field in fields]
    lines = [f"{object_type},"]
    for field in rendered[:-1]:
        lines.append(f"  {field},")
    lines.append(f"  {rendered[-1]};")
    return "\n".join(lines)


def parse_epw_header(epw_path: Path) -> dict[str, str]:
    first_line = epw_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) < 10 or parts[0].upper() != "LOCATION":
        raise RuntimeError(f"Unexpected EPW header in {epw_path}")
    return {
        "location_name": f"{parts[1]} {parts[2]} {parts[3]}",
        "latitude": parts[6],
        "longitude": parts[7],
        "timezone": parts[8],
        "elevation": parts[9],
    }


def choose_weather_file() -> tuple[Path, str]:
    for directory in WEATHER_DIRS:
        if directory.exists():
            epws = sorted(directory.glob("*.epw"))
            if epws:
                return epws[0], "local"
    raise FileNotFoundError("No EPW found in 7/weather or demo3/weather")


# ---------------------------------------------------------------------------
# Zoning: 4 perimeter + 1 core per floor, with conforming (shared) interior edges
# ---------------------------------------------------------------------------
def _floor_zone_outlines(config: GeometryConfig = DEFAULT_GEOMETRY) -> dict[str, tuple]:
    X, Y = config.length_m, config.width_m
    depth = config.perimeter_depth_m
    x0, x1, y0, y1 = depth, X - depth, depth, Y - depth
    if x1 <= x0 or y1 <= y0:
        raise ValueError("perimeter_depth_m is too large for the rectangular floor plate")
    outlines = {
        # South strip: full width, y in [0, y0]; top edge split at x0,x1
        "perim_s": ((0.0, 0.0), (X, 0.0), (X, y0), (x1, y0), (x0, y0), (0.0, y0)),
        # North strip: full width, y in [y1, Y]; bottom edge split at x0,x1
        "perim_n": ((0.0, y1), (x0, y1), (x1, y1), (X, y1), (X, Y), (0.0, Y)),
        # East strip: x in [x1, X], y in [y0, y1]
        "perim_e": ((x1, y0), (X, y0), (X, y1), (x1, y1)),
        # West strip: x in [0, x0], y in [y0, y1]
        "perim_w": ((0.0, y0), (x0, y0), (x0, y1), (0.0, y1)),
        # Core: x in [x0, x1], y in [y0, y1]
        "core": ((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
    }
    patches = {
        "perim_s": (((0.0, 0.0), (X, 0.0), (X, y0), (0.0, y0)),),
        "perim_n": (((0.0, y1), (X, y1), (X, Y), (0.0, Y)),),
        "perim_e": (((x1, y0), (X, y0), (X, y1), (x1, y1)),),
        "perim_w": (((0.0, y0), (x0, y0), (x0, y1), (0.0, y1)),),
        "core": (((x0, y0), (x1, y0), (x1, y1), (x0, y1)),),
    }
    return outlines, patches


ZONE_LABELS = {
    "perim_s": "南周边", "perim_n": "北周边", "perim_e": "东周边",
    "perim_w": "西周边", "core": "核心",
}
ZONE_ORDER = ["perim_s", "perim_n", "perim_e", "perim_w", "core"]


def build_zone_specs(config: GeometryConfig = DEFAULT_GEOMETRY) -> list[ZoneSpec]:
    if config.plan_shape == "l_shape":
        return _build_l_shape_zone_specs(config)
    return _build_rectangle_zone_specs(config)


def _build_rectangle_zone_specs(config: GeometryConfig = DEFAULT_GEOMETRY) -> list[ZoneSpec]:
    outlines, patches = _floor_zone_outlines(config)
    zones: list[ZoneSpec] = []
    for floor in range(1, config.floor_count + 1):
        z0, z1 = _floor_z_range(floor, config)
        for kind in ZONE_ORDER:
            zones.append(
                ZoneSpec(
                    name=f"F{floor}_{kind.upper()}",
                    label=f"{floor}层{ZONE_LABELS[kind]}",
                    kind=kind,
                    floor=floor,
                    outline=outlines[kind],
                    patches=patches[kind],
                    z0=z0,
                    z1=z1,
                    primary_orientation=PRIMARY_ORIENTATION.get(kind),
                )
            )
    return zones


# ---------------------------------------------------------------------------
# L-shape zoning: two adjacent, non-overlapping rectangular wings sharing one
# interior boundary edge, each independently zoned like the rectangle case.
#
#   wing A ("wa_*"): the full-length bottom wing, x in [0,X], y in [0,Y1]
#   wing B ("wb_*"): the narrower top wing,        x in [0,X1], y in [Y1,Y]
#
# where Y1 = width_m - notch_width_m and X1 = length_m - notch_length_m. This
# is the canonical layout for notch_corner="ne" (top-right corner removed);
# the other three corners are obtained by reflecting the canonical vertices
# (see _NOTCH_CORNER_REFLECTION) rather than re-deriving the geometry, so the
# same code path covers all four corners.
#
# Wing A's north edge gets a T-junction vertex at X1 splitting it into an
# interior segment (shared with wing B's south edge, auto-matched by the
# existing shared-edge detector in _wall_surface_records) and an exterior
# segment (the real notch-facing wall) -- the same T-junction trick already
# used by _floor_zone_outlines to stitch perimeter zones to the core.
# ---------------------------------------------------------------------------
_L_WING_KEYS = [
    "wa_perim_s", "wa_perim_n", "wa_perim_e", "wa_perim_w", "wa_core",
    "wb_perim_s", "wb_perim_n", "wb_perim_e", "wb_perim_w", "wb_core",
]

L_ZONE_LABELS = {
    "wa_perim_s": "主翼南周边", "wa_perim_n": "主翼缺角面周边", "wa_perim_e": "主翼东周边",
    "wa_perim_w": "主翼西周边", "wa_core": "主翼核心",
    "wb_perim_s": "副翼共享边周边", "wb_perim_n": "副翼北周边", "wb_perim_e": "副翼缺角面周边",
    "wb_perim_w": "副翼西周边", "wb_core": "副翼核心",
}

# Primary window-carrying orientation per canonical (pre-reflection) zone
# kind; None => core zone, no window. wb_perim_s is labeled "south" for
# bookkeeping but its main face is always the interior wing boundary, so in
# practice it never resolves to an exterior wall (see fenestration_block).
_L_PRIMARY_ORIENTATION = {
    "wa_perim_s": "south", "wa_perim_n": "north", "wa_perim_e": "east", "wa_perim_w": "west", "wa_core": None,
    "wb_perim_s": "south", "wb_perim_n": "north", "wb_perim_e": "east", "wb_perim_w": "west", "wb_core": None,
}

_ORIENTATION_OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east"}

# (reflect_x, reflect_y) needed to map the canonical "ne" layout onto each corner.
_NOTCH_CORNER_REFLECTION = {
    "ne": (False, False),
    "nw": (True, False),
    "se": (False, True),
    "sw": (True, True),
}


def _reflect_orientation(label: str | None, reflect_x: bool, reflect_y: bool) -> str | None:
    if label is None:
        return None
    if reflect_x and label in ("east", "west"):
        label = _ORIENTATION_OPPOSITE[label]
    if reflect_y and label in ("north", "south"):
        label = _ORIENTATION_OPPOSITE[label]
    return label


def _transform_points(points, *, reflect_x: bool, reflect_y: bool, x_size: float, y_size: float):
    """Mirror a polygon's vertices across the given axes and restore CCW winding.

    A single-axis reflection flips winding (CCW<->CW); reflecting both axes
    (equivalent to a 180 deg rotation) preserves it. Reversing the vertex
    order after an odd number of reflections keeps every zone's outline CCW,
    which _edge_orientation and GlobalGeometryRules both assume.
    """
    pts = list(points)
    if reflect_x:
        pts = [(x_size - x, y) for x, y in pts]
    if reflect_y:
        pts = [(x, y_size - y) for x, y in pts]
    if reflect_x != reflect_y:
        pts = list(reversed(pts))
    return tuple(pts)


def _l_shape_zone_outlines(config: GeometryConfig) -> tuple[dict[str, tuple], dict[str, tuple]]:
    X, Y, d = config.length_m, config.width_m, config.perimeter_depth_m
    nl, nw = config.notch_length_m, config.notch_width_m
    if config.notch_corner not in _NOTCH_CORNER_REFLECTION:
        raise ValueError(f"Unknown notch_corner: {config.notch_corner!r}")
    y1 = Y - nw   # wing A's outer (north) extent
    x1 = X - nl   # wing B's outer (east) extent
    xa0, xa1, ya0, ya1 = d, X - d, d, y1 - d
    xb0, xb1, yb0, yb1 = d, x1 - d, y1 + d, Y - d
    if xa1 <= xa0 or ya1 <= ya0:
        raise ValueError("perimeter_depth_m too large for wing A (main wing) of the L-shape plate")
    if xb1 <= xb0 or yb1 <= yb0:
        raise ValueError("perimeter_depth_m too large for wing B (side wing) of the L-shape plate")

    outlines = {
        "wa_perim_s": ((0.0, 0.0), (X, 0.0), (X, ya0), (xa1, ya0), (xa0, ya0), (0.0, ya0)),
        "wa_perim_n": ((0.0, ya1), (xa0, ya1), (xa1, ya1), (X, ya1), (X, y1), (x1, y1), (0.0, y1)),
        "wa_perim_e": ((xa1, ya0), (X, ya0), (X, ya1), (xa1, ya1)),
        "wa_perim_w": ((0.0, ya0), (xa0, ya0), (xa0, ya1), (0.0, ya1)),
        "wa_core": ((xa0, ya0), (xa1, ya0), (xa1, ya1), (xa0, ya1)),
        "wb_perim_s": ((0.0, y1), (x1, y1), (x1, yb0), (xb1, yb0), (xb0, yb0), (0.0, yb0)),
        "wb_perim_n": ((0.0, yb1), (xb0, yb1), (xb1, yb1), (x1, yb1), (x1, Y), (0.0, Y)),
        "wb_perim_e": ((xb1, yb0), (x1, yb0), (x1, yb1), (xb1, yb1)),
        "wb_perim_w": ((0.0, yb0), (xb0, yb0), (xb0, yb1), (0.0, yb1)),
        "wb_core": ((xb0, yb0), (xb1, yb0), (xb1, yb1), (xb0, yb1)),
    }
    patches = {
        "wa_perim_s": ((0.0, 0.0), (X, 0.0), (X, ya0), (0.0, ya0)),
        "wa_perim_n": ((0.0, ya1), (X, ya1), (X, y1), (0.0, y1)),
        "wa_perim_e": ((xa1, ya0), (X, ya0), (X, ya1), (xa1, ya1)),
        "wa_perim_w": ((0.0, ya0), (xa0, ya0), (xa0, ya1), (0.0, ya1)),
        "wa_core": ((xa0, ya0), (xa1, ya0), (xa1, ya1), (xa0, ya1)),
        "wb_perim_s": ((0.0, y1), (x1, y1), (x1, yb0), (0.0, yb0)),
        "wb_perim_n": ((0.0, yb1), (x1, yb1), (x1, Y), (0.0, Y)),
        "wb_perim_e": ((xb1, yb0), (x1, yb0), (x1, yb1), (xb1, yb1)),
        "wb_perim_w": ((0.0, yb0), (xb0, yb0), (xb0, yb1), (0.0, yb1)),
        "wb_core": ((xb0, yb0), (xb1, yb0), (xb1, yb1), (xb0, yb1)),
    }

    reflect_x, reflect_y = _NOTCH_CORNER_REFLECTION[config.notch_corner]
    if reflect_x or reflect_y:
        outlines = {
            k: _transform_points(v, reflect_x=reflect_x, reflect_y=reflect_y, x_size=X, y_size=Y)
            for k, v in outlines.items()
        }
        patches = {
            k: _transform_points(v, reflect_x=reflect_x, reflect_y=reflect_y, x_size=X, y_size=Y)
            for k, v in patches.items()
        }
    return outlines, patches


def _build_l_shape_zone_specs(config: GeometryConfig) -> list[ZoneSpec]:
    outlines, patches = _l_shape_zone_outlines(config)
    reflect_x, reflect_y = _NOTCH_CORNER_REFLECTION[config.notch_corner]
    zones: list[ZoneSpec] = []
    for floor in range(1, config.floor_count + 1):
        z0, z1 = _floor_z_range(floor, config)
        for kind in _L_WING_KEYS:
            orientation = _reflect_orientation(_L_PRIMARY_ORIENTATION[kind], reflect_x, reflect_y)
            zones.append(
                ZoneSpec(
                    name=f"F{floor}_{kind.upper()}",
                    label=f"{floor}层{L_ZONE_LABELS[kind]}",
                    kind=kind,
                    floor=floor,
                    outline=outlines[kind],
                    patches=(patches[kind],),
                    z0=z0,
                    z1=z1,
                    primary_orientation=orientation,
                )
            )
    return zones


def _polygon_area(points: tuple[tuple[float, float], ...]) -> float:
    total = 0.0
    for index, (x0, y0) in enumerate(points):
        x1, y1 = points[(index + 1) % len(points)]
        total += x0 * y1 - x1 * y0
    return abs(total) * 0.5


def _polygon_edges(points):
    return [(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]


def _edge_orientation(start, end) -> str:
    x0, y0 = start
    x1, y1 = end
    if abs(y0 - y1) < 1e-6:
        return "south" if x1 > x0 else "north"
    if abs(x0 - x1) < 1e-6:
        return "east" if y1 > y0 else "west"
    raise ValueError(f"Only axis-aligned edges supported: {start}->{end}")


def _normalized_edge(start, end):
    rounded = ((round(start[0], 6), round(start[1], 6)), (round(end[0], 6), round(end[1], 6)))
    return tuple(sorted(rounded))


def _wall_surface_records(zones: list[ZoneSpec]) -> dict[str, list[dict]]:
    records_by_zone = {zone.name: [] for zone in zones}
    counts_by_zone: dict[str, dict[str, int]] = {}
    edge_map: dict[tuple, list[dict]] = {}
    for zone in zones:
        counts_by_zone[zone.name] = {}
        for start, end in _polygon_edges(zone.outline):
            orientation = _edge_orientation(start, end)
            counts_by_zone[zone.name][orientation] = counts_by_zone[zone.name].get(orientation, 0) + 1
            ordinal = counts_by_zone[zone.name][orientation]
            record = {
                "surface_name": f"{zone.name}_{orientation.upper()}_{ordinal}",
                "orientation": orientation,
                "start": start,
                "end": end,
                "adjacent_surface_name": "",
            }
            records_by_zone[zone.name].append(record)
            edge_map.setdefault((zone.floor, _normalized_edge(start, end)), []).append(record)
    for shared in edge_map.values():
        if len(shared) > 2:
            raise RuntimeError(f"Unexpected multi-zone edge: {shared}")
        if len(shared) == 2:
            left, right = shared
            left["adjacent_surface_name"] = str(right["surface_name"])
            right["adjacent_surface_name"] = str(left["surface_name"])
    return records_by_zone


def _vertical_surface_vertices(orientation, start, end, z0, z1):
    x0, y0 = start
    x1, y1 = end
    if orientation == "south":
        xa, xb = sorted((x0, x1))
        return [(xa, y0, z1), (xa, y0, z0), (xb, y0, z0), (xb, y0, z1)]
    if orientation == "north":
        xa, xb = sorted((x0, x1))
        return [(xb, y0, z1), (xb, y0, z0), (xa, y0, z0), (xa, y0, z1)]
    if orientation == "east":
        ya, yb = sorted((y0, y1))
        return [(x0, ya, z1), (x0, ya, z0), (x0, yb, z0), (x0, yb, z1)]
    if orientation == "west":
        ya, yb = sorted((y0, y1))
        return [(x0, yb, z1), (x0, yb, z0), (x0, ya, z0), (x0, ya, z1)]
    raise ValueError(orientation)


def _horizontal_surface_vertices(patch, z, surface_kind):
    points = list(patch)
    if surface_kind == "floor":
        points = list(reversed(points))
    return [(x, y, z) for x, y in points]


def _stack_lookup(zones):
    return {(zone.floor, zone.kind): zone for zone in zones}


def _exterior_wall_by_orientation(zones, zone):
    result: dict[str, dict] = {}
    for record in _wall_surface_records(zones)[zone.name]:
        if record["adjacent_surface_name"]:
            continue
        result.setdefault(str(record["orientation"]), record)
    return result


def zone_block(zones):
    blocks = []
    for zone in zones:
        blocks.append(idf_object("Zone", [
            zone.name, "0.0", "0.0", "0.0", "0.0", "1", "1",
            fmt(zone.z1 - zone.z0), fmt(zone.volume),
        ]))
    return "\n\n".join(blocks)


def surface_block(zones):
    blocks = []
    wall_records = _wall_surface_records(zones)
    stacks = _stack_lookup(zones)
    for zone in zones:
        for record in wall_records[zone.name]:
            orientation = str(record["orientation"])
            if record["adjacent_surface_name"]:
                outside_condition, outside_object = "Surface", str(record["adjacent_surface_name"])
                sun, wind, construction, view = "NoSun", "NoWind", "Interior Partition", "0.0"
            else:
                outside_condition, outside_object = "Outdoors", ""
                sun, wind, construction, view = "SunExposed", "WindExposed", "Exterior Wall", "0.50"
            vertices = [f"{fmt(x)},{fmt(y)},{fmt(z)}" for x, y, z in _vertical_surface_vertices(
                orientation, tuple(record["start"]), tuple(record["end"]), zone.z0, zone.z1)]
            blocks.append(idf_object("BuildingSurface:Detailed", [
                str(record["surface_name"]), "Wall", construction, zone.name, "",
                outside_condition, outside_object, sun, wind, view, "4", *vertices,
            ]))

        for patch_index, patch in enumerate(zone.patches, start=1):
            floor_name = f"{zone.name}_FLOOR_{patch_index}"
            if zone.floor == 1:
                floor_condition, floor_object, floor_construction = "Ground", "", "Ground Floor"
            else:
                below = stacks[(zone.floor - 1, zone.kind)]
                floor_condition, floor_object, floor_construction = "Surface", f"{below.name}_CEILING_{patch_index}", "Interior Slab"
            floor_vertices = [f"{fmt(x)},{fmt(y)},{fmt(z)}" for x, y, z in _horizontal_surface_vertices(patch, zone.z0, "floor")]
            blocks.append(idf_object("BuildingSurface:Detailed", [
                floor_name, "Floor", floor_construction, zone.name, "",
                floor_condition, floor_object, "NoSun", "NoWind", "0.0",
                str(len(floor_vertices)), *floor_vertices,
            ]))

            ceiling_name = f"{zone.name}_CEILING_{patch_index}"
            if zone.floor == max(z.floor for z in zones):
                ceiling_condition, ceiling_object, ceiling_construction = "Outdoors", "", "Roof"
                ceiling_sun, ceiling_wind = "SunExposed", "WindExposed"
                ceiling_type = "Roof"
            else:
                above = stacks[(zone.floor + 1, zone.kind)]
                ceiling_condition, ceiling_object, ceiling_construction = "Surface", f"{above.name}_FLOOR_{patch_index}", "Interior Slab"
                ceiling_sun, ceiling_wind = "NoSun", "NoWind"
                ceiling_type = "Ceiling"
            ceiling_vertices = [f"{fmt(x)},{fmt(y)},{fmt(z)}" for x, y, z in _horizontal_surface_vertices(patch, zone.z1, "ceiling")]
            blocks.append(idf_object("BuildingSurface:Detailed", [
                ceiling_name, ceiling_type, ceiling_construction, zone.name, "",
                ceiling_condition, ceiling_object, ceiling_sun, ceiling_wind, "0.0",
                str(len(ceiling_vertices)), *ceiling_vertices,
            ]))
    return "\n\n".join(blocks)


def _wwr_window_vertices(record, zone, wwr, width_frac=0.90):
    """Centered ribbon window; window/wall area == wwr (width_frac of wall width,
    height = wwr*floor_height/width_frac, vertically centered)."""
    orientation = str(record["orientation"])
    start, end = tuple(record["start"]), tuple(record["end"])
    fh = zone.z1 - zone.z0
    height = wwr * fh / width_frac
    height = min(height, fh - 0.6)          # keep >=0.3 sill top and bottom
    sill = (fh - height) / 2.0
    z_low = zone.z0 + sill
    z_high = z_low + height
    if orientation in {"south", "north"}:
        x_low, x_high = min(start[0], end[0]), max(start[0], end[0])
        wall_w = x_high - x_low
        w = wall_w * width_frac
        x0 = x_low + (wall_w - w) / 2.0
        x1 = x0 + w
        y = start[1]
        if orientation == "south":
            return [(x0, y, z_high), (x0, y, z_low), (x1, y, z_low), (x1, y, z_high)]
        return [(x1, y, z_high), (x1, y, z_low), (x0, y, z_low), (x0, y, z_high)]
    y_low, y_high = min(start[1], end[1]), max(start[1], end[1])
    wall_w = y_high - y_low
    w = wall_w * width_frac
    y0 = y_low + (wall_w - w) / 2.0
    y1 = y0 + w
    x = start[0]
    if orientation == "east":
        return [(x, y0, z_high), (x, y0, z_low), (x, y1, z_low), (x, y1, z_high)]
    return [(x, y1, z_high), (x, y1, z_low), (x, y0, z_low), (x, y0, z_high)]


def fenestration_block(zones, window_wall_ratio: float):
    blocks = []
    if window_wall_ratio <= 0.0:
        return ""
    for zone in zones:
        orientation = zone.primary_orientation
        if orientation is None:
            continue
        exterior_walls = _exterior_wall_by_orientation(zones, zone)
        if orientation not in exterior_walls:
            # Legitimate for L-shape "shared wing boundary" zones (e.g.
            # wb_perim_s in the canonical NE layout): their nominal primary
            # face sits on the interior wing seam, not an exterior wall, so
            # they end up with no window -- not an error.
            continue
        record = exterior_walls[orientation]
        parent = str(record["surface_name"])
        vertices = [f"{fmt(x)},{fmt(y)},{fmt(z)}" for x, y, z in _wwr_window_vertices(record, zone, window_wall_ratio)]
        blocks.append(idf_object("FenestrationSurface:Detailed", [
            f"{parent}_WIN", "Window", "Exterior Window", parent, "",
            "autocalculate", "", "1", "4", *vertices,
        ]))
    return "\n\n".join(blocks)


def zone_list_block(zones):
    return idf_object("ZoneList", ["OFFICE_ZONES", *[zone.name for zone in zones]])


def hvac_block(zones):
    blocks = []
    for zone in zones:
        eq_list = f"{zone.name}_EQUIPMENT_LIST"
        supply = f"{zone.name}_SUPPLY_INLET"
        air_node = f"{zone.name}_AIR_NODE"
        ret = f"{zone.name}_RETURN_OUTLET"
        ideal = f"{zone.name}_IDEAL_LOADS"
        blocks.append(idf_object("ZoneHVAC:EquipmentConnections", [
            zone.name, eq_list, supply, "", air_node, ret]))
        blocks.append(idf_object("ZoneHVAC:EquipmentList", [
            eq_list, "SequentialLoad", "ZoneHVAC:IdealLoadsAirSystem", ideal, "1", "1"]))
        blocks.append(idf_object("ZoneHVAC:IdealLoadsAirSystem", [
            ideal, "", supply, "", "", "50", "13", "", "", "NoLimit", "", "",
            "NoLimit", "", "", "", "", "None", "", "None"]))
    return "\n\n".join(blocks)


def render_idf(template: str, weather_meta: dict, scenario: dict, zones: list[ZoneSpec]) -> str:
    wwr = float(scenario.get("window_wall_ratio", 0.40))
    replacements = {
        "__SCENARIO_NAME__": str(scenario.get("label", "office")),
        "__LOCATION_NAME__": weather_meta["location_name"],
        "__LATITUDE__": weather_meta["latitude"],
        "__LONGITUDE__": weather_meta["longitude"],
        "__TIMEZONE__": weather_meta["timezone"],
        "__ELEVATION__": weather_meta["elevation"],
        "__WALL_INSULATION_M__": f'{float(scenario["wall_insulation_m"]):.4f}',
        "__ROOF_INSULATION_M__": f'{float(scenario["roof_insulation_m"]):.4f}',
        "__WINDOW_U__": f'{float(scenario["window_u"]):.4f}',
        "__WINDOW_SHGC__": f'{float(scenario["window_shgc"]):.4f}',
        "__INFILTRATION_ACH__": f'{float(scenario["infiltration_ach"]):.4f}',
        "__HEATING_SETPOINT__": f'{float(scenario["heating_setpoint_c"]):.2f}',
        "__COOLING_SETPOINT__": f'{float(scenario["cooling_setpoint_c"]):.2f}',
        "__ZONE_BLOCK__": zone_block(zones),
        "__SURFACE_BLOCK__": surface_block(zones),
        "__FENESTRATION_BLOCK__": fenestration_block(zones, wwr),
        "__ZONE_LIST_BLOCK__": zone_list_block(zones),
        "__HVAC_BLOCK__": hvac_block(zones),
    }
    text = template
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text


def render_idf_for_spec(spec: BuildingProjectSpec, weather_meta: dict, label: str | None = None) -> str:
    """Render a rectangular project spec into an EnergyPlus IDF.

    This is the natural-language training workflow entrypoint. Existing scripts
    can keep using build_zone_specs() + render_idf() with default geometry.
    """
    config = geometry_from_project_spec(spec)
    zones = build_zone_specs(config)
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    return render_idf(template, weather_meta, spec.to_scenario(label=label), zones)


def geometry_summary(config: GeometryConfig = DEFAULT_GEOMETRY) -> dict[str, float | int]:
    zones_per_floor = len(_L_WING_KEYS) if config.plan_shape == "l_shape" else len(ZONE_ORDER)
    return {
        "plan_shape": config.plan_shape,
        "length_m": config.length_m,
        "width_m": config.width_m,
        "floor_count": config.floor_count,
        "ground_floor_height_m": config.ground_floor_height_m,
        "typical_floor_height_m": config.typical_floor_height_m,
        "perimeter_depth_m": config.perimeter_depth_m,
        "height_m": config.total_height_m,
        "floor_area_m2": config.floor_area_m2,
        "notch_corner": config.notch_corner if config.plan_shape == "l_shape" else None,
        "notch_length_m": config.notch_length_m if config.plan_shape == "l_shape" else None,
        "notch_width_m": config.notch_width_m if config.plan_shape == "l_shape" else None,
        "zone_count": config.floor_count * zones_per_floor,
    }


def find_energyplus() -> str:
    exe = shutil.which("energyplus")
    if exe:
        return exe
    for candidate in ENERGYPLUS_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("energyplus executable not found (PATH or known install dirs)")


def run_energyplus(epw: Path, idf_path: Path, output_dir: Path) -> None:
    exe = find_energyplus()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [exe, "-r", "-w", str(epw), "-d", str(output_dir), str(idf_path)]
    # cwd=output_dir so ReadVars writes its `readvars.audit` into this run's own dir;
    # without it, parallel workers share one cwd and collide on readvars.audit (fatal).
    result = subprocess.run(command, capture_output=True, text=True, cwd=str(output_dir))
    if result.returncode != 0:
        raise RuntimeError(f"EnergyPlus failed for {idf_path.name}\nSTDOUT:\n{result.stdout[-2000:]}\nSTDERR:\n{result.stderr[-2000:]}")

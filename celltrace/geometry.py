from dataclasses import dataclass
import math

import numpy as np
from scipy import ndimage

from .model import Issue, Placement, Plane, cell_id


def signed_area(points: np.ndarray) -> float:
    following = np.roll(points, -1, axis=0)
    return float(np.sum(points[:, 0] * following[:, 1] - points[:, 1] * following[:, 0]) / 2)


def cross_sign(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def validate_quad(corners: np.ndarray):
    corners = np.asarray(corners, dtype=np.float64)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise Issue("Four finite image-space corners are required.")
    next_corners = np.roll(corners, -1, axis=0)
    edges = next_corners - corners
    turns = cross_sign(edges, np.roll(edges, -1, axis=0))
    if np.min(np.linalg.norm(edges, axis=1)) < 16 or np.any(turns < 48):
        raise Issue("Choose clockwise corners around a convex panel, starting at top left.")
    if signed_area(corners) < 500:
        raise Issue("The selected panel is too small.")


def default_placement(width: int, height: int) -> Placement:
    return Placement(((0.0, 0.0), (float(width), 0.0), (float(width), float(height)), (0.0, float(height))))


def normalize_quad(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(points, axis=0)
    radius = np.mean(np.linalg.norm(points - center, axis=1))
    if radius < 1e-8:
        raise Issue("Panel coordinates are degenerate.")
    scale = math.sqrt(2) / radius
    transform = np.array([
        [scale, 0, -scale * center[0]],
        [0, scale, -scale * center[1]],
        [0, 0, 1],
    ])
    return (points - center) * scale, transform


def homography(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_normal, source_scale = normalize_quad(source)
    target_normal, target_scale = normalize_quad(target)
    rows = []
    for (x, y), (u, v) in zip(source_normal, target_normal):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    try:
        _, singular, right = np.linalg.svd(rows, full_matrices=True)
        if singular[-1] <= 1e-7:
            raise Issue("The four panel coordinates have no stable transform.")
        rectification = np.linalg.solve(
            target_scale, right[-1].reshape(3, 3) @ source_scale
        )
        denominator = rectification[2, 2]
        if not np.isfinite(rectification).all() or abs(denominator) < 1e-10:
            raise Issue("The selected panel has an unstable transform.")
        return rectification / denominator
    except np.linalg.LinAlgError:
        raise Issue("The selected panel could not be rectified.") from None


def project(transform: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = transform[2, 0] * x + transform[2, 1] * y + transform[2, 2]
    if np.any(abs(n) < 1e-8):
        raise Issue("The transform passes through infinity in the inspected plane.")
    u = (transform[0, 0] * x + transform[0, 1] * y + transform[0, 2]) / n
    v = (transform[1, 0] * x + transform[1, 1] * y + transform[1, 2]) / n
    return u, v


def basis_corners(plane: Plane) -> np.ndarray:
    return np.array(((0, 0), (plane.width, 0), (plane.width, plane.height), (0, plane.height)),
                    dtype=np.float64)


def transform_for(plane: Plane, placement: Placement) -> np.ndarray:
    corners = np.asarray(placement.corners)
    validate_quad(corners)
    order = {
        "identity": [0, 1, 2, 3],
        "rotate180": [2, 3, 0, 1],
        "mirror-x": [1, 0, 3, 2],
        "mirror-y": [3, 2, 1, 0],
    }[placement.orientation]
    return homography(basis_corners(plane), corners[order])


@dataclass(frozen=True)
class SamplingMap:
    matrix: np.ndarray
    x: np.ndarray
    y: np.ndarray
    in_bounds: np.ndarray

    @classmethod
    def for_plane(cls, plane: Plane, placement: Placement, source_width: int, source_height: int):
        matrix = transform_for(plane, placement)
        y, x = np.ogrid[0.5:plane.height:1.0, 0.5:plane.width:1.0]
        image_x, image_y = project(matrix, x, y)
        image_x = image_x.astype(np.float32) - 0.5
        image_y = image_y.astype(np.float32) - 0.5
        tolerance = 1e-4
        in_bounds = (
            (image_x >= -tolerance) & (image_x <= source_width - 1 + tolerance)
            & (image_y >= -tolerance) & (image_y <= source_height - 1 + tolerance)
        )
        np.clip(image_x, 0, source_width - 1, out=image_x)
        np.clip(image_y, 0, source_height - 1, out=image_y)
        return cls(matrix, image_x, image_y, in_bounds)

    def samples(self, values: np.ndarray) -> np.ndarray:
        return ndimage.map_coordinates(values, (self.y, self.x), order=1, mode="nearest", prefilter=False)

    def usable(self, mask: np.ndarray) -> np.ndarray:
        weights = self.samples(mask.astype(np.float32))
        return self.in_bounds & (weights > 0.9999)

    def present(self, mask: np.ndarray) -> np.ndarray:
        weights = self.samples(mask.astype(np.float32))
        return self.in_bounds & (weights > 0.0001)


@dataclass(frozen=True)
class Cell:
    row: int
    col: int
    bounds: tuple[int, int, int, int]
    interior: tuple[int, int, int, int]
    string: int

    @property
    def id(self) -> str:
        return cell_id(self.row, self.col)

    @property
    def view(self) -> tuple[slice, slice]:
        x0, y0, x1, y1 = self.interior
        return slice(y0, y1), slice(x0, x1)

    @property
    def pixels(self) -> int:
        x0, y0, x1, y1 = self.interior
        return (x1 - x0) * (y1 - y0)

    def covers(self, x: float, y: float) -> bool:
        left, upper, right, lower = self.bounds
        return left <= x < right and upper <= y < lower


def cells_for(plane: Plane) -> list[Cell]:
    cells = []
    x_edges = np.rint(np.linspace(0, plane.width, plane.cols + 1)).astype(int)
    y_edges = np.rint(np.linspace(0, plane.height, plane.rows + 1)).astype(int)
    half_rows = plane.rows // 2 if plane.string_axis == "half_columns" else 0
    for row, (y0, y1) in enumerate(zip(y_edges, y_edges[1:])):
        for col, (x0, x1) in enumerate(zip(x_edges, x_edges[1:])):
            ix = int((x1 - x0) * plane.inset)
            iy = int((y1 - y0) * plane.inset)
            if plane.string_axis == "half_columns":
                string = (col // plane.string_size) * 2 + min(row // half_rows, 1) + 1
            else:
                axis_index = col if plane.string_axis == "columns" else row
                string = axis_index // plane.string_size + 1
            cells.append(Cell(
                row, col,
                (int(x0), int(y0), int(x1), int(y1)),
                (int(x0) + ix, int(y0) + iy, int(x1) - ix, int(y1) - iy),
                string,
            ))
    return cells


def plane_polygon(points: tuple, plane: Plane) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) * np.array((plane.width, plane.height))


def validate_polygon(points: np.ndarray, min_area: float = 8):
    if len(points) < 3 or not np.isfinite(points).all():
        raise Issue("The polygon requires at least three finite points.")
    following = np.roll(points, -1, axis=0)
    if np.any(np.linalg.norm(points - following, axis=1) < 0.75):
        raise Issue("Polygon vertices must have distinct positions.")
    if abs(signed_area(points)) < min_area:
        raise Issue("The polygon is too small for measurement.")
    for index, (a, b) in enumerate(zip(points, following)):
        for other in range(index + 2, len(points)):
            if index == 0 and other == len(points) - 1:
                continue
            if segments_intersect(a, b, points[other], following[other]):
                raise Issue("The polygon must have a simple boundary.")


def segments_intersect(a, b, c, d) -> bool:
    def side(first, second, point):
        return float(cross_sign(second - first, point - first))

    epsilon = 1e-7
    side_ab = side(a, b, c), side(a, b, d)
    side_cd = side(c, d, a), side(c, d, b)
    if side_ab[0] * side_ab[1] < -epsilon and side_cd[0] * side_cd[1] < -epsilon:
        return True
    for first, second, point, cross in (
        (a, b, c, side_ab[0]), (a, b, d, side_ab[1]),
        (c, d, a, side_cd[0]), (c, d, b, side_cd[1]),
    ):
        if abs(cross) <= epsilon and np.all(point >= np.minimum(first, second) - epsilon) and np.all(
            point <= np.maximum(first, second) + epsilon
        ):
            return True
    return False


def polygon_mask(points: np.ndarray, width: int, height: int) -> np.ndarray:
    raster = np.zeros((height, width), dtype=bool)
    x0 = max(0, int(np.floor(points[:, 0].min())))
    y0 = max(0, int(np.floor(points[:, 1].min())))
    x1 = min(width, int(np.ceil(points[:, 0].max())))
    y1 = min(height, int(np.ceil(points[:, 1].max())))
    if x1 <= x0 or y1 <= y0:
        return raster
    y, x = np.ogrid[y0 + 0.5:y1:1.0, x0 + 0.5:x1:1.0]
    inside = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    for (ax, ay), (bx, by) in zip(points, np.roll(points, -1, axis=0)):
        if ay == by:
            continue
        crossings = ((ay > y) != (by > y)) & (x < ax + (y - ay) * (bx - ax) / (by - ay))
        inside ^= crossings
    raster[y0:y1, x0:x1] = inside
    return raster


def millimeter_scale(plane: Plane) -> tuple[float, float] | None:
    if plane.physical_width is None or plane.physical_height is None:
        return None
    return plane.physical_width / plane.width, plane.physical_height / plane.height

#!/usr/bin/env python3
"""Reproducible offline racing-line candidate. Never imports ROS or drives.

Regenerate the official reference from the map/track definition, seal cone
rows in memory, then minimise integrated squared curvature with smoothing.
This is NOT minimum lap time or a proof of controller feasibility.
"""

import argparse
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'scripts'))
import track_tool


@dataclass(frozen=True)
class Geometry:
    length_m: float = .58
    width_m: float = .31
    body_center_x_m: float = .1651
    margin_m: float = .15
    max_offset_m: float = .8
    sample_spacing_m: float = .01
    output_spacing_m: float = .2

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or (name != 'body_center_x_m' and value <= 0):
                raise ValueError(f'Invalid geometry: {name}')


class MapClearance:
    """Distance lower bound to occupied square cells and outer map boundary.

    Nearest occupied-cell-centre distance minus half the cell diagonal is
    conservative. Uses current simulator occupancy threshold, not stale README
    text saying all pixels below 128 are occupied.
    """

    def __init__(self, grid):
        self.grid = grid
        gray = np.asarray(grid.pixels).reshape(grid.height, grid.width)
        probability = gray/255 if grid.negate else (255-gray)/255
        self.occupied = probability > grid.occupied_thresh
        row, col = np.where(self.occupied)
        if not len(row):
            raise ValueError('Map has no occupied cells')
        self.tree = cKDTree(np.c_[grid.origin[0]+(col+.5)*grid.resolution,
                                 grid.origin[1]+(grid.height-row-.5)*grid.resolution])
        self.half_diagonal = grid.resolution / np.sqrt(2)
        self.lower = np.asarray(grid.origin[:2])
        self.upper = self.lower + np.array([grid.width, grid.height])*grid.resolution
        if abs(grid.origin[2]) > 1e-12:
            raise ValueError('Rotated map origins are not supported')

    def query(self, points):
        points = np.asarray(points)
        shape = points.shape[:-1]
        flat = points.reshape(-1, 2)
        d = self.tree.query(flat)[0] - self.half_diagonal
        edges = np.min(np.c_[flat-self.lower, self.upper-flat], axis=1)
        return np.minimum(d, edges).reshape(shape)


def path_spline(points):
    points = np.asarray(points)
    ds = np.linalg.norm(np.roll(points, -1, axis=0)-points, axis=1)
    if np.any(ds <= 1e-8):
        raise ValueError('Duplicate path points')
    s = np.r_[0., np.cumsum(ds)]
    return s, CubicSpline(s, np.vstack([points, points[0]]), bc_type='periodic')


def reference_from_map():
    tracks, name = track_tool._load(str(ROOT/'maps/tracks.yaml'), 'icra26')
    track = tracks[name]
    grid = track_tool.OccupancyMap(track_tool._resolve_map_yaml(track.map_path, track.map_image_ext))
    links = grid.link_cones(1.1)
    raw = track_tool.find_lap(grid, track, half_width=.35, link_cones=0)
    if raw is None:
        raise ValueError('No legal closed reference lap')
    ref = track_tool.smooth_closed(raw, 21)
    ref = track_tool.enforce_clearance(grid, ref, .35)
    ref = np.asarray(track_tool.resample_closed(ref, .2))
    # Match official CSV precision; no manually placed waypoints.
    return np.round(ref, 4), grid, links


def corridor_bounds(points, normals, clearance, geometry):
    """Connected normal corridors; stop at the first clearance obstruction."""
    lower, upper = [], []
    step = .01
    required = geometry.width_m/2 + geometry.margin_m
    if np.min(clearance.query(points)) < required:
        raise ValueError('Reference lacks the requested centre clearance')
    for point, normal in zip(points, normals):
        ends = []
        for sign in [-1, 1]:
            offsets = np.arange(0, geometry.max_offset_m + step/2, step)
            values = clearance.query(point + sign * offsets[:, None] * normal)
            invalid = np.flatnonzero(values < required + step)
            last = max(0, invalid[0]-1) if len(invalid) else len(offsets)-1
            ends.append(sign * offsets[last])
        lower.append(ends[0]); upper.append(ends[1])
    return np.asarray(lower), np.asarray(upper)


def optimise_offsets(points, normals, lower, upper, length_weight=.01, initial_offsets=None):
    """Periodic spline curvature energy, with analytic derivatives.

    Integrate kappa**2 over actual candidate arc length, not reference distance.
    Penalise curvature variation and lightly regularise offsets/length.
    Bounds enforce connected normal corridors; full-body validation follows.
    """
    if not np.isfinite(length_weight) or length_weight < 0:
        raise ValueError('Length weight must be finite and nonnegative')
    count = len(points)
    ds = np.linalg.norm(np.roll(points, -1, axis=0)-points, axis=1)
    s = np.r_[0., np.cumsum(ds)]
    identity = np.eye(count)
    basis = CubicSpline(s, np.vstack([identity, identity[0]]), bc_type='periodic')
    q = np.sort(np.r_[s[:-1], s[:-1]+ds/2])
    dq = (np.roll(q, -1)-q) % s[-1]
    weight = (dq+np.roll(dq, 1))/2
    matrices = []
    for order in [1, 2]:
        matrix = basis(q, order)
        matrix[abs(matrix) < 1e-12] = 0
        matrices.append(csr_matrix(matrix))
    first, second = matrices

    def objective(offsets):
        moved = points + normals*offsets[:, None]
        d1, d2 = first @ moved, second @ moved
        v = np.maximum(np.linalg.norm(d1, axis=1), 1e-8)
        cross = d1[:, 0]*d2[:, 1]-d1[:, 1]*d2[:, 0]
        kappa = cross/v**3
        dk = np.roll(kappa, -1)-kappa
        value = np.sum(weight*cross**2/v**5) + .04*np.sum(dk**2/dq)
        value += .01*np.sum(offsets**2*ds) + length_weight*np.sum(weight*v)
        cross_d1 = np.c_[d2[:, 1], -d2[:, 0]]
        cross_d2 = np.c_[-d1[:, 1], d1[:, 0]]
        grad1 = (2*weight*cross/v**5)[:, None]*cross_d1
        grad1 -= (5*weight*cross**2/v**7)[:, None]*d1
        grad2 = (2*weight*cross/v**5)[:, None]*cross_d2
        grad_k = .08*(np.roll(dk/dq, 1)-dk/dq)
        grad1 += grad_k[:, None]*(cross_d1/v[:, None]**3 - (3*cross/v**5)[:, None]*d1)
        grad2 += (grad_k/v**3)[:, None]*cross_d2
        grad1 += (length_weight*weight/v)[:, None]*d1
        gradient = np.sum(normals*(first.T @ grad1 + second.T @ grad2), axis=1)
        gradient += .02*offsets*ds
        return float(value), gradient

    initial = np.zeros(count) if initial_offsets is None else np.clip(initial_offsets, lower, upper)
    result = minimize(objective, initial, jac=True, method='L-BFGS-B',
                      bounds=list(zip(lower, upper)),
                      options={'maxiter': 20000, 'ftol': 1e-8, 'gtol': 1e-6, 'maxcor': 30})
    if not result.success:
        raise RuntimeError(f'Offset optimisation failed: {result.message}')
    return result.x, {'iterations': int(result.nit), 'objective_before': float(objective(np.zeros(count))[0]),
                      'objective_after': float(result.fun), 'solver_message': str(result.message)}


def body_margin(points, clearance, geometry):
    """Dense check of three covering discs around the offset rectangular body.

    Three discs cover three adjacent thirds of the rectangle. This checks all
    body points conservatively, not just its centre/corners. Subtract a sampling
    allowance from clearance; still a geometric check, not a dynamic rollout.
    """
    s, spline = path_spline(points)
    values = np.linspace(0, s[-1], int(np.ceil(s[-1]/geometry.sample_spacing_m)), endpoint=False)
    xy, derivative = spline(values), spline(values, 1)
    norm = np.linalg.norm(derivative, axis=1)
    if np.min(norm) < .1:
        raise ValueError('Degenerate path tangent')
    tangent = derivative / norm[:, None]
    offsets = geometry.body_center_x_m + np.array([-1, 0, 1])*geometry.length_m/3
    centres = xy[:, None, :] + tangent[:, None, :] * offsets[None, :, None]
    radius = np.hypot(geometry.width_m/2, geometry.length_m/6)
    distances = clearance.query(centres) - radius
    sampling_allowance = float(np.max(np.linalg.norm(np.roll(centres, -1, axis=0)-centres, axis=2))/2)
    return float(np.min(distances)-sampling_allowance)


def resample(points, spacing):
    s, spline = path_spline(points)
    dense = spline(np.linspace(0, s[-1], int(np.ceil(s[-1]/.005))+1))
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    q = np.linspace(0, arc[-1], int(np.ceil(arc[-1]/spacing)), endpoint=False)
    return np.c_[np.interp(q, arc, dense[:, 0]), np.interp(q, arc, dense[:, 1])]


def metrics(points, clearance, geometry):
    s, spline = path_spline(points)
    q = np.linspace(0, s[-1], int(np.ceil(s[-1]/.01)), endpoint=False)
    d1, d2 = spline(q, 1), spline(q, 2)
    speed = np.linalg.norm(d1, axis=1)
    curvature = (d1[:, 0]*d2[:, 1]-d1[:, 1]*d2[:, 0]) / speed**3
    length_step = speed*s[-1]/len(q)
    local_curvature = []
    for i in range(len(points)):
        ix = (i+np.arange(-3, 4)) % len(points)
        local_s = np.r_[0., np.cumsum(np.diff(s)[ix[:-1]])]
        local_s -= local_s[3]
        coefficients = np.polynomial.polynomial.polyfit(local_s, points[ix]-points[i], 3)
        a, b = coefficients[1], 2*coefficients[2]
        local_curvature.append((a[0]*b[1]-a[1]*b[0])/np.linalg.norm(a)**3)
    return {'point_count': len(points), 'polyline_length_m': float(s[-1]),
            'spline_length_m': float(np.sum(length_step)),
            'peak_abs_curvature_7point_1pm': float(np.max(abs(np.asarray(local_curvature)))),
            'peak_abs_curvature_1pm': float(np.max(abs(curvature))),
            'rms_curvature_1pm': float(np.sqrt(np.sum(curvature**2*length_step)/sum(length_step))),
            'integrated_curvature_squared_1pm': float(np.sum(curvature**2*length_step)),
            'min_body_clearance_lower_bound_m': body_margin(points, clearance, geometry)}


def check_topology(points, track):
    """Reject proper self-crossings and require one legal finish crossing."""
    dense = resample(points, .05)
    ends = np.roll(dense, -1, axis=0)
    def cross(a, b):
        return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]
    for i in range(len(dense)):
        j = np.arange(i+2, len(dense))
        j = j[j != (i-1) % len(dense)]
        a, b, c, d = dense[i], ends[i], dense[j], ends[j]
        ab, cd = b-a, d-c
        if np.any((cross(ab, c-a)*cross(ab, d-a) < -1e-12)
                  & (cross(cd, a-c)*cross(cd, b-c) < -1e-12)):
            raise ValueError('Self-intersecting candidate')
    a, b = np.asarray(track.finish_line)
    side = cross(b-a, dense-a)
    next_side = np.roll(side, -1)
    crossing = (side*next_side < 0)
    fractions = side[crossing]/(side[crossing]-next_side[crossing])
    locations = dense[crossing]+fractions[:, None]*(ends[crossing]-dense[crossing])
    position = (locations-a) @ (b-a) / np.sum((b-a)**2)
    valid = (position >= 0) & (position <= 1)
    direction = np.sign(next_side[crossing]-side[crossing])[valid]
    if len(direction) != 1 or direction[0] != track.crossing_direction:
        raise ValueError('Expected exactly one forward finish-line crossing')
    return {'proper_self_intersections': 0, 'forward_finish_crossings': 1,
            'topology_sample_spacing_m': .05}


def generate(geometry, length_weight=.01, coarse_stride=1):
    reference, grid, links = reference_from_map()
    clearance = MapClearance(grid)
    _, spline = path_spline(reference)
    ds = np.linalg.norm(np.roll(reference, -1, axis=0)-reference, axis=1)
    tangent = spline(np.r_[0., np.cumsum(ds[:-1])], 1)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    normals = np.c_[-tangent[:, 1], tangent[:, 0]]
    lower, upper = corridor_bounds(reference, normals, clearance, geometry)
    if coarse_stride < 1 or len(reference[::coarse_stride]) < 12:
        raise ValueError('Coarse stride must retain at least 12 reference points')
    initial = None
    coarse_solver = None
    if coarse_stride > 1:
        # Resolve long-range line placement before fine spline details. This is
        # an automatic map-derived seed, not a manually selected set of apexes.
        ix = np.arange(0, len(reference), coarse_stride)
        coarse, coarse_solver = optimise_offsets(reference[ix], normals[ix], lower[ix], upper[ix], length_weight)
        station = np.r_[0., np.cumsum(ds)]
        interpolation = CubicSpline(np.r_[station[ix], station[-1]], np.r_[coarse, coarse[0]], bc_type='periodic')
        initial = np.clip(interpolation(station[:-1]), lower, upper)
    offsets, solver = optimise_offsets(reference, normals, lower, upper, length_weight, initial)
    # A bounded centreline corridor alone does not guarantee that the car body
    # clears a corner. Back off the entire smooth solution towards the safe
    # reference until the final resampled output satisfies the footprint check.
    chosen = None
    for fraction in np.linspace(1, 0, 21):
        candidate = resample(reference + fraction*offsets[:, None]*normals, geometry.output_spacing_m)
        if body_margin(candidate, clearance, geometry) >= geometry.margin_m:
            chosen = candidate
            break
    if chosen is None or fraction == 0:
        raise RuntimeError('No improved candidate meeting the requested body margin')
    before, after = metrics(reference, clearance, geometry), metrics(chosen, clearance, geometry)
    if after['integrated_curvature_squared_1pm'] >= before['integrated_curvature_squared_1pm']:
        raise RuntimeError('Candidate does not improve measured curvature energy')
    tracks, name = track_tool._load(str(ROOT/'maps/tracks.yaml'), 'icra26')
    topology = check_topology(chosen, tracks[name])
    report = {'status': 'offline_geometric_candidate',
              'geometry': asdict(geometry), 'cone_links': links, 'solver': solver,
              'coarse_stride': coarse_stride, 'coarse_solver': coarse_solver,
              'objective_weights': {'curvature_energy': 1., 'curvature_variation': .04,
                                    'offset_regularisation': .01, 'length': length_weight},
              'retained_offset_fraction': float(fraction),
              'max_normal_offset_m': float(np.max(abs(fraction*offsets))),
              'reference': before, 'candidate': after, 'topology': topology,
              'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [ROOT/'maps/icra26.yaml', ROOT/'maps/icra26.pgm',
                                         ROOT/'maps/tracks.yaml', ROOT/'scripts/track_tool.py']},
              'limitations': ['Local numerical curvature optimisation with regularisation, not global optimum or minimum lap time.',
                              'Body follows candidate tangent; actual tracking/yaw errors not simulated.',
                              'Dense sampled footprint check is not a continuous-time collision proof.',
                              'No speed profile, actuator, tyre dynamics or closed-loop racing evaluation.',
                              'Curvature uses periodic spline derivatives; not the old 7-point smoothed estimator.']}
    return reference, chosen, grid, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New output directory; refuses overwrite')
    parser.add_argument('--margin', type=float, default=.15, help='Requested geometric body margin, metres')
    parser.add_argument('--length-weight', type=float, default=.01,
                        help='Length penalty relative to curvature energy; larger favours shorter routes, not faster laps')
    parser.add_argument('--max-offset', type=float, default=.8, help='Maximum normal search distance; obstacles still stop the search')
    parser.add_argument('--coarse-stride', type=int, default=1, help='Automatically seed broad line placement using every Nth reference point')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output directory already exists')
    geometry = Geometry(margin_m=args.margin, max_offset_m=args.max_offset)
    reference, candidate, grid, report = generate(geometry, args.length_weight, args.coarse_stride)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, points in [('reference_from_map.csv', reference), ('raceline_candidate.csv', candidate)]:
        np.savetxt(args.output/name, points, delimiter=',', fmt='%.8f', header='x_m,y_m')
    report['generator_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

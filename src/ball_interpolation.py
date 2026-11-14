"""
Ball trajectory interpolation.
Fills gaps in ball detection using linear interpolation + optional Kalman smoothing.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d


def interpolate_ball_trajectory(
    ball_positions: dict[int, tuple[float, float] | None],
    max_gap: int = 45,
    smooth_window: int = 5,
) -> dict[int, tuple[float, float] | None]:
    """
    Interpolate missing ball positions and smooth the trajectory.

    Args:
        ball_positions: dict mapping frame_number -> (x, y) or None if not detected.
        max_gap: Maximum number of consecutive missing frames to interpolate across.
        smooth_window: Window size for trajectory smoothing.

    Returns:
        dict mapping frame_number -> (x, y) with gaps filled where possible.
    """
    if not ball_positions:
        return ball_positions

    all_frames = sorted(ball_positions.keys())
    detected_frames = [f for f in all_frames if ball_positions[f] is not None]

    if len(detected_frames) < 2:
        return ball_positions

    # Extract detected positions
    det_x = np.array([ball_positions[f][0] for f in detected_frames])
    det_y = np.array([ball_positions[f][1] for f in detected_frames])
    det_t = np.array(detected_frames, dtype=np.float64)

    # Build interpolation functions
    interp_x = interp1d(det_t, det_x, kind='linear', fill_value='extrapolate')
    interp_y = interp1d(det_t, det_y, kind='linear', fill_value='extrapolate')

    result = dict(ball_positions)

    # Fill gaps
    for i in range(len(detected_frames) - 1):
        start_frame = detected_frames[i]
        end_frame = detected_frames[i + 1]
        gap_size = end_frame - start_frame - 1

        if 0 < gap_size <= max_gap:
            for f in range(start_frame + 1, end_frame):
                if result.get(f) is None:
                    result[f] = (float(interp_x(f)), float(interp_y(f)))

    # Apply smoothing to all filled positions
    if smooth_window > 1:
        filled_frames = sorted([f for f in result if result[f] is not None])
        if len(filled_frames) > smooth_window:
            xs = np.array([result[f][0] for f in filled_frames])
            ys = np.array([result[f][1] for f in filled_frames])
            xs_smooth = uniform_filter1d(xs, size=smooth_window)
            ys_smooth = uniform_filter1d(ys, size=smooth_window)
            for idx, f in enumerate(filled_frames):
                result[f] = (float(xs_smooth[idx]), float(ys_smooth[idx]))

    return result


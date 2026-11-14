from src.roster import apply_roster

FPS = 30.0


def player(tid, f, x, team=0, box=None):
    px = 100 + x * 10
    return {"tracker_id": tid, "team_id": team, "pitch_xy": [x, 20.0],
            "bbox": box or [px, 300, px + 20, 360]}


def build(tracks):
    """tracks: {tid: (first, last, fn(f)->player)}"""
    frames = {str(f): {"players": []} for f in range(0, 200)}
    for tid, (a, b, fn) in tracks.items():
        for f in range(a, b + 1):
            frames[str(f)]["players"].append(fn(tid, f))
    return frames


def ids(frames):
    return {p["tracker_id"] for v in frames.values() for p in v["players"]}


def full_team(n, extra):
    t = {i: (0, 199, lambda tid, f: player(tid, f, 10 + tid * 4)) for i in range(1, n + 1)}
    t.update(extra)
    return t


def test_under_cap_keeps_new_track():
    fr = build(full_team(2, {9: (50, 60, lambda t, f: player(t, f, 40))}))
    assert 9 in ids(apply_roster(fr, 3, FPS, 60, 40))


def test_full_team_short_midpitch_spawn_dropped():
    fr = build(full_team(3, {9: (50, 60, lambda t, f: player(t, f, 40))}))
    assert 9 not in ids(apply_roster(fr, 3, FPS, 60, 40))


def test_full_team_edge_spawn_kept_as_sub():
    fr = build(full_team(3, {9: (50, 60, lambda t, f: {**player(t, f, 1.0)})}))
    assert 9 in ids(apply_roster(fr, 3, FPS, 60, 40))


def test_duplicate_box_dropped_even_if_long():
    host = lambda tid, f: player(tid, f, 10 + tid * 4)
    dup = lambda tid, f: player(tid, f, 14.2, box=[242, 310, 258, 350])  # inside player 1's box
    fr = build(full_team(3, {9: (20, 150, dup)}))
    out = apply_roster(fr, 3, FPS, 60, 40)
    assert 9 not in ids(out) and {1, 2, 3} <= ids(out)


def test_reattach_to_lost_teammate():
    t = full_team(3, {})
    t[3] = (0, 100, lambda tid, f: player(tid, f, 22))
    t[9] = (110, 199, lambda tid, f: player(tid, f, 23))
    fr = build(t)
    out = apply_roster(fr, 3, FPS, 60, 40)
    assert ids(out) == {1, 2, 3}
    assert any(p["tracker_id"] == 3 for p in out["150"]["players"])


def test_other_team_not_counted():
    fr = build(full_team(3, {9: (50, 60, lambda t, f: player(t, f, 40, team=1))}))
    assert 9 in ids(apply_roster(fr, 3, FPS, 60, 40))

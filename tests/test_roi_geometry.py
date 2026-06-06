from vision.roi_geometry import (
    point_in_polygon,
    point_to_segment_distance,
    nearest_edge,
    average_quads,
    EDGE_LABELS,
)

SQUARE = [(0, 0), (100, 0), (100, 100), (0, 100)]  # TL,TR,BR,BL


def test_point_inside_square():
    assert point_in_polygon((50, 50), SQUARE) is True


def test_point_outside_square():
    assert point_in_polygon((150, 50), SQUARE) is False


def test_point_outside_above():
    assert point_in_polygon((50, -10), SQUARE) is False


def test_point_in_skewed_quad():
    quad = [(10, 0), (100, 20), (90, 100), (0, 80)]
    assert point_in_polygon((50, 50), quad) is True
    assert point_in_polygon((5, 5), quad) is False


def test_segment_distance_perpendicular():
    assert point_to_segment_distance((50, 50), (0, 0), (100, 0)) == 50.0


def test_segment_distance_beyond_endpoint():
    d = point_to_segment_distance((150, 0), (0, 0), (100, 0))
    assert d == 50.0


def test_nearest_edge_top():
    idx, dist = nearest_edge((50, 5), SQUARE)
    assert idx == 0
    assert dist == 5.0


def test_nearest_edge_left():
    idx, dist = nearest_edge((5, 50), SQUARE)
    assert idx == 3
    assert dist == 5.0


def test_average_quads_single():
    assert average_quads([SQUARE]) == [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]


def test_average_quads_mean():
    q1 = [(0, 0), (100, 0), (100, 100), (0, 100)]
    q2 = [(10, 10), (110, 10), (110, 110), (10, 110)]
    assert average_quads([q1, q2]) == [(5.0, 5.0), (105.0, 5.0), (105.0, 105.0), (5.0, 105.0)]


def test_average_quads_empty():
    assert average_quads([]) is None


def test_edge_labels():
    assert EDGE_LABELS == ("top", "right", "bottom", "left")

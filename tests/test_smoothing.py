from vision.smoothing import EMA


def test_first_update_returns_input():
    ema = EMA(alpha=0.4)
    assert ema.update(10.0, 20.0) == (10.0, 20.0)


def test_second_update_blends():
    ema = EMA(alpha=0.5)
    ema.update(0.0, 0.0)
    x, y = ema.update(10.0, 20.0)
    assert x == 5.0
    assert y == 10.0


def test_converges_toward_constant_input():
    ema = EMA(alpha=0.5)
    ema.update(0.0, 0.0)
    for _ in range(20):
        ema.update(100.0, 100.0)
    x, y = ema.value
    assert 99.0 < x <= 100.0
    assert 99.0 < y <= 100.0


def test_reset_clears_state():
    ema = EMA(alpha=0.4)
    ema.update(50.0, 60.0)
    ema.reset()
    assert ema.value is None
    assert ema.update(1.0, 2.0) == (1.0, 2.0)

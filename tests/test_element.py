from src.element import Element


def test_element_init():
    e = Element(0, 0, 1, 1)

    assert e.u_min == 0 and e.u_max == 1 and e.v_min == 0 and e.v_max == 1 and e.supported_b_splines == []


def test_element_containment():
    e = Element(0, 0, 1, 1)

    assert e.contains(0.5, 0.5)
    assert e.contains(0.5, 0)

    assert not e.contains(-1, 0.5)
    assert not e.contains(0, -1.0e-14)


def test_element_support():
    e = Element(0, 0, 1, 1)
    e.add_supported_b_spline(max)

    assert e.has_supported_b_spline(max)
    assert e.get_supported_b_spline(0) == max

    e.remove_supported_b_spline(max)
    assert len(e.supported_b_splines) == 0


def test_element_split():
    e1 = Element(0, 0, 1, 1)
    e2 = e1.split(0, 0.5)

    assert e1.u_min == 0 and e1.v_min == 0 and e1.u_max == 0.5 and e1.v_max == 1
    assert e2.u_min == 0.5 and e2.v_min == 0 and e2.u_max == 1 and e2.v_max == 1


def test_element_invalid():
    e1 = Element(0, 0, 1, 1)
    e2 = e1.split(0, 1)

    assert e2 is None
    assert e1.u_min == 0 and e1.u_max == 1 and e1.v_min == 0 and e1.v_max == 1

def test_element_midpoint():
    e1 = Element(0, 0, 1, 1)
    mp = e1.midpoint()

    assert mp == (0.5, 0.5)
from lr_spline import init_tensor_product_LR_spline
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
    mp = e1.midpoint

    assert mp == (0.5, 0.5)


def test_element_area():
    e1 = Element(0, 0, 1, 1)

    assert e1.area == 1


def test_element_intersects():
    e1 = Element(0, 0, 1, 1)
    e2 = Element(0.25, 0.25, 0.75, 0.75)
    e3 = Element(0.5, 0.5, 1.5, 1.5)
    e4 = Element(1, 0, 2, 1)

    assert e1.intersects(e2) and e2.intersects(e1)
    assert e1.intersects(e3) and e3.intersects(e1)
    assert not e1.intersects(e4)

    e5 = e1.split(axis=0, split_value=.5)
    assert not e1.intersects(e5)


def test_element_equality():
    e1 = Element(0, 0, 1, 1)
    e2 = Element(0, 0, 1, 1)
    e3 = Element(0, 0, 1, 1.1)

    assert e1 == e2
    assert e1 != e3


def test_element_overloaded():
    LR = init_tensor_product_LR_spline(1, 1, [0, 0, 1, 2, 2], [0, 0, 1, 2, 2])

    e = LR.M[2]

    assert not e.is_overloaded()

    e.add_supported_b_spline(LR.S[2])

    assert e.is_overloaded()

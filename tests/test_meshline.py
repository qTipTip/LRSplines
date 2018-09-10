from LRSplines.b_spline import BSpline
from LRSplines.element import Element
from LRSplines.meshline import Meshline


def test_meshline_init():
    start = 0
    stop = 1
    axis = 0
    constant_value = 0.5

    m = Meshline(start, stop, constant_value, axis)

    assert m.start == start
    assert m.stop == stop
    assert m.constant_value == constant_value
    assert m.axis == axis
    assert m.multiplicity == 1


def test_meshline_splits_element():
    start = 0
    stop = 1
    constant_value = 0.5

    m1 = Meshline(start, stop, constant_value, 0)
    m2 = Meshline(start, stop, constant_value, 1)
    e1 = Element(0, 0, 1, 1)
    e2 = Element(0, 0, 0.4, 1)
    e3 = Element(0, 0, 0.5, 1)

    assert m1.splits_element(e1)
    assert not m1.splits_element(e2)
    assert not m1.splits_element(e3)

    assert m2.splits_element(e1)
    assert m2.splits_element(e2)
    assert m2.splits_element(e3)


def test_meshline_splits_basis():
    start = 0
    stop = 1
    constant_value = 0.5
    m1 = Meshline(start, stop, constant_value, 0)
    m2 = Meshline(start, stop, constant_value, 1)

    b1 = BSpline(2, 2, [0, 1, 2, 3], [0, 1, 2, 3])
    b2 = BSpline(2, 2, [0, 1, 2, 3], [0, 0.25, 0.5, 0.75, 1])

    assert not m1.splits_basis(b1)
    assert m1.splits_basis(b2)

    assert not m2.splits_basis(b1)
    assert not m2.splits_basis(b2)


def test_meshline_number_of_knots():
    b1 = BSpline(1, 1, [0, 1, 2], [0, 1, 2])
    m1 = Meshline(0, 2, constant_value=1, axis=0)
    m2 = Meshline(0, 2, constant_value=0.5, axis=1)

    assert m1.number_of_knots_contained(b1) == 1
    assert m2.number_of_knots_contained(b1) == 0


def test_meshline_contains():
    m1 = Meshline(0, 2, constant_value=1, axis=0)
    m2 = Meshline(0.1, 1.9, constant_value=1, axis=0)
    m3 = Meshline(0.1, 1.9, constant_value=1.2, axis=0)
    m4 = Meshline(0.1, 1.9, constant_value=1, axis=1)

    assert m1.contains(m2)
    assert not m1.contains(m3)
    assert not m1.contains(m4)


def test_meshline_equals():
    m1 = Meshline(0.1, 1.9, constant_value=1, axis=0)
    m2 = Meshline(0.1, 1.9, constant_value=1.2, axis=0)
    m3 = Meshline(0.1, 1.9, constant_value=1, axis=1)

    assert m1 == m1
    assert m1 != m2
    assert m1 != m3


def test_meshline_overlaps():
    m1 = Meshline(0, 1, constant_value=0.5, axis=1)
    m2 = Meshline(0, 2, constant_value=0.5, axis=1)
    m3 = Meshline(0.5, 1.5, constant_value=0.5, axis=1)
    m4 = Meshline(0.5, 1.5, constant_value=0.5, axis=0)
    m5 = Meshline(0.5, 1.5, constant_value=0.4, axis=1)

    assert m1.overlaps(m2)
    assert m1.overlaps(m3)
    assert not m1.overlaps(m4)
    assert not m1.overlaps(m5)


def test_meshline_midpoint():
    m1 = Meshline(0, 2, constant_value=0.5, axis=1)
    m2 = Meshline(0, 2, constant_value=0.5, axis=0)

    assert m1.midpoint == (1, 0.5)
    assert m2.midpoint == (0.5, 1)


def test_minimal_support_split():
    B = BSpline(1, 1, [3, 3.5, 4], [2, 3, 4])
    m = Meshline(2, 4, 2.5, axis=1)

    assert m.splits_basis(B)


def test_meshline_overlaps_disjoint():
    m1 = Meshline(start=0, stop=2, constant_value=3, axis=0)
    m2 = Meshline(start=5, stop=6, constant_value=3, axis=0)

    assert not m1.overlaps(m2)

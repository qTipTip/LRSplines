from src.b_spline import BSpline
from src.element import Element
from src.meshline import Meshline


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

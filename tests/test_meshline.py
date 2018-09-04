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
    axis = 0
    constant_value = 0.5
    m = Meshline(start, stop, constant_value, axis)

    e1 = Element(0, 0, 1, 1)
    e2 = Element(0, 0, 0.4, 1)
    e3 = Element(0, 0, 0.5, 1)
    assert m.splits_element(e1)
    assert not m.splits_element(e2)
    assert not m.splits_element(e3)

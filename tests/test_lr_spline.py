from collections import Counter

import pytest

from LRSplines.b_spline import BSpline
from LRSplines.element import Element
from LRSplines.lr_spline import init_tensor_product_LR_spline, LRSpline
from LRSplines.meshline import Meshline


def test_lr_spline_minimal_span_line():
    d1, d2 = (2, 2)
    ku = [0, 1, 2, 3]
    kv = [0, 1, 2, 3]
    LR = init_tensor_product_LR_spline(d1, d2, ku, kv)

    element_to_refine = LR.M[1]
    minimal_span_meshline = LR.get_minimal_span_meshline(element_to_refine, axis=0)

    assert minimal_span_meshline.start == 0
    assert minimal_span_meshline.stop == 3
    assert minimal_span_meshline.constant_value == 0.5


def test_lr_spline_minimal_span_line_superficial():
    """
    Constructed example, won't occur in practice.
    """

    element = Element(0.5, 0, 1, 2)
    ku1 = [-1, 0, 1]
    kv1 = [0, 1, 2]
    ku2 = [0, 0.5, 1]
    kv2 = [0, 1, 3]

    b1 = BSpline(1, 1, ku1, kv1)
    b2 = BSpline(1, 1, ku2, kv2)

    element.add_supported_b_spline(b1)
    element.add_supported_b_spline(b2)

    small_span_meshline_vertical = LRSpline.get_minimal_span_meshline(element, axis=0)
    small_span_meshline_horizontal = LRSpline.get_minimal_span_meshline(element, axis=1)

    assert small_span_meshline_vertical.start == 0
    assert small_span_meshline_vertical.stop == 2
    assert small_span_meshline_vertical.constant_value == 0.75

    assert small_span_meshline_horizontal.start == 0
    assert small_span_meshline_horizontal.stop == 1
    assert small_span_meshline_horizontal.constant_value == 1


def test_lr_spline_insert_line_single():
    LR = init_tensor_product_LR_spline(1, 1, [0, 1, 2, 3], [0, 1, 2])
    M = Meshline(0, 2, constant_value=0.5, axis=1)  # horizontal split
    LR.insert_line(M)

    # expected functions
    b1 = BSpline(1, 1, [0, 1, 2], [0, 0.5, 1])
    b2 = BSpline(1, 1, [0, 1, 2], [0.5, 1, 2])
    b3 = BSpline(1, 1, [1, 2, 3], [0, 1, 2])

    assert b1 in LR.S
    assert b2 in LR.S
    assert b3 in LR.S
    assert len(LR.S) == 3

    expected_elements = [
        Element(0, 0, 1, 0.5),
        Element(1, 0, 2, 0.5),
        Element(0, 0.5, 1, 1),
        Element(1, 0.5, 2, 1),
        Element(2, 0, 3, 1),
        Element(0, 1, 1, 2),
        Element(1, 1, 2, 2),
        Element(2, 1, 3, 2)
    ]

    assert all([LR.contains_element(e) for e in expected_elements])
    assert len(LR.M) == 8


def test_lr_spline_insert_line_multiple():
    LR = init_tensor_product_LR_spline(1, 1, [0, 1, 2, 3], [0, 1, 2])
    M1 = Meshline(0, 2, constant_value=0.5, axis=1)  # horizontal split
    M2 = Meshline(0, 1, constant_value=1.5, axis=0)  # vertical split
    LR.insert_line(M1)
    LR.insert_line(M2)

    # expected functions
    b1 = BSpline(1, 1, [0, 1, 1.5], [0, 0.5, 1])
    b2 = BSpline(1, 1, [1, 1.5, 2], [0, 0.5, 1])
    b3 = BSpline(1, 1, [0, 1, 2], [0.5, 1, 2])
    b4 = BSpline(1, 1, [1, 2, 3], [0, 1, 2])

    assert b1 in LR.S
    assert b2 in LR.S
    assert b3 in LR.S
    assert b3 in LR.S
    assert len(LR.S) == 4

    expected_elements = [
        Element(0, 0, 1, 0.5),
        Element(1, 0, 1.5, 0.5),
        Element(1.5, 0, 2, 0.5),
        Element(0, 0.5, 1, 1),
        Element(1, 0.5, 1.5, 1),
        Element(1.5, 0.5, 2, 1),
        Element(2, 0, 3, 1),
        Element(0, 1, 1, 2),
        Element(1, 1, 2, 2),
        Element(2, 1, 3, 2)
    ]

    assert all([LR.contains_element(e) for e in expected_elements])
    assert len(LR.M) == 10


@pytest.mark.skip(reason='This is probably wrong - I havent checked the exact dimension')
def test_lr_spline_insert_multiple():
    LR = init_tensor_product_LR_spline(2, 2, [0, 0, 0, 1, 2, 4, 5, 6, 6, 6], [0, 0, 0, 1, 2, 4, 5, 6, 6, 6])
    m1 = Meshline(1, 5, constant_value=3, axis=0)
    m2 = Meshline(1, 5, constant_value=3, axis=1)

    assert len(LR.S) == 49
    LR.insert_line(m1)
    assert len(LR.S) == 52
    LR.insert_line(m2)
    assert len(LR.S) == 62

    # assorted expected functions
    b1 = BSpline(2, 2, [1, 2, 3, 4], [2, 3, 4, 5])
    b2 = BSpline(2, 2, [2, 3, 4, 5], [2, 3, 4, 5])
    b3 = BSpline(2, 2, [1, 2, 3, 4], [2, 3, 4, 5])
    b4 = BSpline(2, 2, [1, 2, 4, 5], [3, 4, 5, 6])

    assert b1 in LR.S
    assert b2 in LR.S
    assert b3 in LR.S
    assert b4 in LR.S


@pytest.mark.skip(reason='This test is platform dependent. Passes on Linux, fails on OSX')
def test_lr_spline_cleanup():
    LR = init_tensor_product_LR_spline(1, 1, [0, 1, 2], [0, 1, 2])
    M = Meshline(0, 2, constant_value=0.5, axis=0)
    LR.insert_line(M)

    elements = [
        Element(0, 0, 0.5, 1),
        Element(0, 1, 0.5, 2),
        Element(0.5, 0, 1, 1),
        Element(0.5, 1, 1, 2),
        Element(1, 0, 2, 1),
        Element(1, 1, 2, 2)
    ]

    b1 = LR.S[0]
    b2 = LR.S[1]

    assert all(e in b1.elements_of_support for e in elements[0:4])
    assert all(e in b2.elements_of_support for e in elements[2:])


def test_lr_spline_merge_meshlines_count():
    LR = init_tensor_product_LR_spline(1, 1, [0, 1, 2], [0, 1, 2])

    old_number_of_meshlines = len(LR.meshlines)
    m1 = Meshline(0, 2, constant_value=1, axis=0)
    LR.insert_line(meshline=m1)
    new_number_of_meshlines = len(LR.meshlines)

    assert old_number_of_meshlines == new_number_of_meshlines


def test_lr_spline_unique_basis_functions():
    LR = init_tensor_product_LR_spline(1, 1, [0, 0, 1, 1], [0, 1, 2])
    m = Meshline(0, 2, constant_value=0.5, axis=0)
    LR.insert_line(m, debug=False)
    m = Meshline(0, 1, constant_value=0.5, axis=1)
    LR.insert_line(m, debug=False)

    c = Counter(LR.S)
    assert all([count == 1 for count in c.values()])


def test_lr_spline_previously_split_functions():
    LR = init_tensor_product_LR_spline(2, 2, [0, 0, 0, 1, 2, 4, 5, 6, 6, 6], [0, 0, 0, 1, 2, 4, 5, 6, 6, 6])
    m = Meshline(1, 5, constant_value=3, axis=0)
    LR.insert_line(m)


def test_lr_spline_overloading_count_bilinear():
    LR = init_tensor_product_LR_spline(1, 1, [0, 0, 1, 2, 3, 4, 5, 6, 6], [0, 0, 1, 2, 3, 4, 5, 6, 6])

    m1 = Meshline(2, 4, constant_value=3.5, axis=0)
    m2 = Meshline(2, 4, constant_value=2.5, axis=0)
    m3 = Meshline(2, 4, constant_value=2.5, axis=1)
    m4 = Meshline(2, 4, constant_value=3.5, axis=1)

    # culprit = 3, 3.5, 4 x 2 3 4
    LR.insert_line(m1, debug=True)
    LR.insert_line(m2, debug=True)
    LR.insert_line(m3, debug=True)
    LR.insert_line(m4, debug=True)

    for e in LR.M:
        c = Counter(e.supported_b_splines)
        assert all([count == 1 for count in c.values()])

    for e in LR.M:
        assert len(e.supported_b_splines) in [4, 5]

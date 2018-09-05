from b_spline import BSpline
from element import Element
from lr_spline import init_tensor_product_LR_spline, LRSpline
from meshline import Meshline


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


def test_lr_spline_insert_line():
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


def test_lr_spline_insert_multiple():
    pass
    # LR = init_tensor_product_LR_spline([2, 2, [0, 0, 0, 1,2, 4, 5, 6, 6, 6], [0, 0, 0, 1, 2, 4, 5, 6, 6, 6]])
    # LR.insert_line()

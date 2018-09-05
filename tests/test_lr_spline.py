from lr_spline import init_tensor_product_LR_spline


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

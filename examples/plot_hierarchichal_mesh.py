from LRSplines import init_tensor_product_LR_spline, hierarchical_meshline_rectangle
from LRSplines.statistics import overloads_per_level

ku = [0, 0, 0, 1, 2, 3, 3, 3]
d = 2
lrs = init_tensor_product_LR_spline(d, d, ku, ku)

M = hierarchical_meshline_rectangle(1, 1, 2, 2, step=0.5)
for m in M:
    print(m)
    lrs.insert_line(m)
M = hierarchical_meshline_rectangle(1, 1, 2, 2, step=0.25)
for m in M:
    print(m)
    lrs.insert_line(m)

M = hierarchical_meshline_rectangle(1.5, 1, 2.5, 2, step=0.125) + hierarchical_meshline_rectangle(1.5, 1, 2.5, 1.5,
                                                                                                  step=0.0075)

for m in M:
    lrs.insert_line(m)

lrs.visualize_mesh(False, True)

stats = overloads_per_level(lrs)

print(stats)

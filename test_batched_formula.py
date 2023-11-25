import torch
from line_sphere_intersection_formula import line_sphere_intersection, b_line_sphere_intersection


u = torch.tensor([1, 0, 0])
o = torch.tensor([-2, 2, 0])
r = 2

expected_output = line_sphere_intersection(r, o, u)
print(f"{expected_output=}")

b_u = u.repeat(5, 1)
b_o = o.repeat(5, 1)

actual_output = b_line_sphere_intersection(r, b_o, b_u)
print(f"{actual_output.shape=}")
print(f"{actual_output=}")

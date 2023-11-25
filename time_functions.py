import timeit
import torch
from line_sphere_intersection import find_intersections, line_sphere_intersection


n = 100000
print("No solutions:")
r = 1
o = torch.tensor([-5.0, 10.0, 0.0])
u = torch.tensor([1.0, 0, 0])
print(f"{timeit.Timer(lambda: line_sphere_intersection(r, o, u)).timeit(n)=}") 
print(f"{timeit.Timer(lambda: find_intersections(r, o, u)).timeit(n)=}") 
print()
print("One solution:")
r = 1
o = torch.tensor([-5.0, 1.0, 0.0])
u = torch.tensor([1.0, 0, 0])
print(f"{timeit.Timer(lambda: line_sphere_intersection(r, o, u)).timeit(n)=}") 
print(f"{timeit.Timer(lambda: find_intersections(r, o, u)).timeit(n)=}") 
print()
print("Two solutions:")
r = 1
o = torch.tensor([-5.0, 0.0, 0.0])
u = torch.tensor([1.0, 0, 0])
print(f"{timeit.Timer(lambda: line_sphere_intersection(r, o, u)).timeit(n)=}") 
print(f"{timeit.Timer(lambda: find_intersections(r, o, u)).timeit(n)=}") 

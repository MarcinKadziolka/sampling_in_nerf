import torch
import math

def find_intersections(r, o, u):
    # finds intersections of a line x = o + du
    # with a sphere ||x||^2 = r^2 (centered at [0, 0, 0])
    # in a form of list of values of d

    u_norm = torch.linalg.vector_norm(u).item()
    u1 = torch.nn.functional.normalize(u, dim=0)   # u1 is an unit vector, same u as u

    delta = (torch.dot(u1, o)**2 - (torch.linalg.vector_norm(o)**2 - r**2)).item()

    if math.isclose(delta, 0, abs_tol=1e-5):    # delta ~ 0
        # one solution
        return [-torch.dot(u1, o).item() / u_norm]
    elif delta < 0:                             # delta < 0
        # no solutions
        return []
    else:                                       # delta > 0
        # two solutions
        return [(-torch.dot(u1, o).item() - math.sqrt(delta)) / u_norm,
                (-torch.dot(u1, o).item() + math.sqrt(delta)) / u_norm]
    

def line_sphere_intersection(r, o, u):
    a = torch.dot(u, u).item()
    b = 2*( torch.dot(u, o).item())
    c = torch.dot(o, o).item() - r**2
    delta = b**2 - 4 * a * c

    if math.isclose(delta, 0, abs_tol=1e-5):
        return [-b / (2 * a)]
    if delta < 0:
        return []
    else:  
        return [(-b - math.sqrt(delta)) / (2*a), (-b + math.sqrt(delta)) / (2*a)]



def b_line_sphere_intersection(r, o, u):
    # https://discuss.pytorch.org/t/dot-product-batch-wise/9746
    a = (u*u).sum(-1)
    b = 2 * (u*o).sum(-1)
    c = (o*o).sum(-1) - r**2
    delta = b**2 - 4 * a * c
   # TODO:
   # find indexes of negative delta
   # mask values when delta is negative

    points = (-b - torch.sqrt(delta)) / (2*a), (-b + torch.sqrt(delta)) / (2*a)

    return torch.stack(points, dim=-1)

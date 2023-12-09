import torch
import math

def line_sphere_intersection(r, o, u):
    """Calculate distances of intersections with a sphere
    Args:
        r: float. Radius of the sphere        
        o: array of shape [3]. Coordinates of the ray origin
        u: array of shape [3]. Vector of the ray direction
    Returns:
        Distances of intersections from the ray origin
        if no intersections:
            empty array.
        if one intersection:
            array of shape [1].
        if two intersections:
            array of shape [2]
    """
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

def batched_line_sphere_intersection(r, o, u):
    """Calculate batched distances of intersections with a sphere.
    Args:
        r: float. Radius of the sphere.        
        o: array of shape [n_rays, 3]. Coordinates of the rays origins.
        u: array of shape [n_rays, 3]. Vectors of the rays directions.
    Returns:
        array of shape [n_rays, 2]. Distances of intersections from the rays origins.
    """
    # https://discuss.pytorch.org/t/dot-product-batch-wise/9746
    a = (u*u).sum(-1)
    b = 2 * (u*o).sum(-1)
    c = (o*o).sum(-1) - r**2
    delta = b**2 - 4 * a * c
   # Maybe TODO:
   # find indexes of negative delta
   # mask values when delta is negative
    points = (-b - torch.sqrt(delta)) / (2*a), (-b + torch.sqrt(delta)) / (2*a)
    return torch.stack(points, dim=-1)

def intersection_coordinates(o, u, d):
    intersection_p1 = o + u[:, 0].view(-1, 1) * d
    intersection_p2 = o + u[:, 1].view(-1, 1) * d

    intersection_points = torch.stack((intersection_p1, intersection_p2), dim=1)
    return intersection_points

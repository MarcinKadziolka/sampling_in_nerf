import pytest
import torch
from line_sphere_intersection_formula import line_sphere_intersection, batched_line_sphere_intersection, intersection_coordinates 

class TestLineSphereIntersection:
    def test_vertical_line_through_center(self):
        res = line_sphere_intersection(1, torch.tensor([0.0, 0.0, 10.0]), torch.tensor([0.0, 0.0, -2.0]))
        assert len(res) == 2
        assert res[0] == pytest.approx(4.5, abs=1e-5)
        assert res[1] == pytest.approx(5.5, abs=1e-5)

    def test_line_far_from_sphere(self):
        res = line_sphere_intersection(1, torch.tensor([0.0, 0.0, 10.0]), torch.tensor([2.0, 0.0, 0.0]))
        assert len(res) == 0

    def test_line_tangent(self):
        res = line_sphere_intersection(1, torch.tensor([10.0, 0.0, 1.0]), torch.tensor([-3.0, 0.0, 0.0]))
        assert len(res) == 1
        assert res[0] == pytest.approx(10/3, abs=1e-5)

    # checked with http://www.ambrnet.com/TrigoCalc/Sphere/SpherLineIntersection_.htm
    def test1(self):
        res = line_sphere_intersection(53.42, torch.tensor([7.0, 4.0, 4.0]), torch.tensor([5.0, 5.0, 4.0]))
        assert len(res) == 2
        assert res[0] == pytest.approx(-7.645978719, abs=1e-5)
        assert res[1] == pytest.approx(5.494463568, abs=1e-5)

    def test2(self):
        res = line_sphere_intersection(0.8, torch.tensor([7.7, -0.5, 0.44]), torch.tensor([144, 4.11, 4.144]))
        assert len(res) == 2
        assert res[0] == pytest.approx(-0.0552663833, abs=1e-5)
        assert res[1] == pytest.approx(-0.0514803572, abs=1e-5)

    def test3(self):
        res = line_sphere_intersection(464, torch.tensor([7.0, 1.0, 0.0]), torch.tensor([-6.0, 2.0, 42.0]))
        assert len(res) == 2
        assert res[0] == pytest.approx(-10.90103428, abs=1e-5)
        assert res[1] == pytest.approx(10.945380178, abs=1e-5)

    def test4(self):
        res = line_sphere_intersection(60000, torch.tensor([9977.0, 1042.0, 455.0]), torch.tensor([-1.0, 1.0, 4.0]))
        assert len(res) == 2
        assert res[0] == pytest.approx(-13552.997869112, abs=2)
        assert res[1] == pytest.approx(14343.553424667, abs=2)

    def test5(self):
        res = line_sphere_intersection(148, torch.tensor([97.0, 104.0, -45.0]), torch.tensor([-5.0, 10.0, 8.0]))
        assert len(res) == 0

class TestBatchedLineSphereIntersection:
    def test_tangent_line_same_values_in_batch(self):
        b_o = torch.tensor([[-2, 2, 0],
                           [-2, 2, 0],
                           [-2, 2, 0]])
        b_u = torch.tensor([[1, 0, 0],
                            [1, 0, 0],
                            [1, 0, 0]])
        r = 2
        expected_output = torch.tensor([[2., 2.], [2., 2.], [2., 2.]])
        actual_output = batched_line_sphere_intersection(r, b_o, b_u)
        assert torch.equal(expected_output, actual_output)

    def test_different_values_in_batch(self):
        b_o = torch.tensor([[-2, 2, 0], [0, 0, 10], [0, -6, 0]])
        b_u = torch.tensor([[1, 0, 0], [0, 0, -2], [0, 2, 0]])
        r = 2
        expected_output = torch.tensor([[2., 2.], [4, 6], [2, 4]])
        actual_output = batched_line_sphere_intersection(r, b_o, b_u)
        assert torch.equal(expected_output, actual_output)

class TestIntersectionCoordinates:
    def test_single_axis_origin_direction(self):
        o = torch.tensor([[-8, 0, 0], [0, 12, 0], [0, 0, 6]])
        u = torch.tensor([[2, 0, 0], [0, -2, 0], [0, 0, -1]])
        r = 2
        d = torch.tensor([[3, 5], [5, 7], [4, 8]])
        expected_output = torch.tensor([
            [[-2, 0, 0], [2, 0, 0]], 
            [[0, 2, 0], [0, -2, 0]], 
            [[0, 0, 2], [0, 0, -2]]
            ])
        actual_output = intersection_coordinates(o, u, d) 
        assert torch.equal(expected_output, actual_output)

import unittest, torch
from line_sphere_intersection_formula import line_sphere_intersection, find_intersections

class TestFindIntersections(unittest.TestCase):
    
    def test_vertical_line_through_center(self):
        res = find_intersections(1, torch.tensor([0.0, 0.0, 10.0]), torch.tensor([0.0, 0.0, -2.0]))
        self.assertEqual(len(res), 2)
        self.assertAlmostEqual(res[0], 4.5, places=5)
        self.assertAlmostEqual(res[1], 5.5, places=5)
    
    def test_line_far_from_sphere(self):
        res = find_intersections(1, torch.tensor([0.0, 0.0, 10.0]), torch.tensor([2.0, 0.0, 0.0]))
        self.assertEqual(len(res), 0)

    def test_line_tangent(self):
        res = find_intersections(1, torch.tensor([10.0, 0.0, 1.0]), torch.tensor([-3.0, 0.0, 0.0]))
        self.assertEqual(len(res), 1)
        self.assertAlmostEqual(res[0], 10/3, places=5)

    # checked with http://www.ambrnet.com/TrigoCalc/Sphere/SpherLineIntersection_.htm
    def test1(self):
        res = find_intersections(53.42, torch.tensor([7.0, 4.0, 4.0]), torch.tensor([5.0, 5.0, 4.0]))
        self.assertEqual(len(res), 2)
        self.assertAlmostEqual(res[0], -7.645978719, places=5)
        self.assertAlmostEqual(res[1], 5.494463568, places=5)

    def test2(self):
        res = find_intersections(0.8, torch.tensor([7.7, -0.5, 0.44]), torch.tensor([144, 4.11, 4.144]))
        self.assertEqual(len(res), 2)
        self.assertAlmostEqual(res[0], -0.0552663833, places=5)
        self.assertAlmostEqual(res[1], -0.0514803572, places=5)

    def test3(self):
        res = find_intersections(464, torch.tensor([7.0, 1.0, 0.0]), torch.tensor([-6.0, 2.0, 42.0]))
        self.assertEqual(len(res), 2)
        self.assertAlmostEqual(res[0], -10.90103428, places=5)
        self.assertAlmostEqual(res[1], 10.945380178, places=5)

    def test4(self):
        res = find_intersections(60000, torch.tensor([9977.0, 1042.0, 455.0]), torch.tensor([-1.0, 1.0, 4.0]))
        self.assertEqual(len(res), 2)
        self.assertAlmostEqual(res[0], -13552.997869112, places=2)
        self.assertAlmostEqual(res[1], 14343.553424667, places=2)

    def test5(self):
        res = find_intersections(148, torch.tensor([97.0, 104.0, -45.0]), torch.tensor([-5.0, 10.0, 8.0]))
        self.assertEqual(len(res), 0)

class TestLineSphereIntersection(unittest.TestCase):
    def test_vertical_line_through_center(self):
        res = line_sphere_intersection(1, torch.tensor([0.0, 0.0, 10.0]), torch.tensor([0.0, 0.0, -2.0]))
        self.assertEqual(len(res), 2)
        self.assertAlmostEqual(res[0], 4.5, places=5)
        self.assertAlmostEqual(res[1], 5.5, places=5)
    
    def test_line_far_from_sphere(self):
        res = line_sphere_intersection(1, torch.tensor([0.0, 0.0, 10.0]), torch.tensor([2.0, 0.0, 0.0]))
        self.assertEqual(len(res), 0)

    def test_line_tangent(self):
        res = line_sphere_intersection(1, torch.tensor([10.0, 0.0, 1.0]), torch.tensor([-3.0, 0.0, 0.0]))
        self.assertEqual(len(res), 1)
        self.assertAlmostEqual(res[0], 10/3, places=5)

    # checked with http://www.ambrnet.com/TrigoCalc/Sphere/SpherLineIntersection_.htm
    def test1(self):
        res = line_sphere_intersection(53.42, torch.tensor([7.0, 4.0, 4.0]), torch.tensor([5.0, 5.0, 4.0]))
        self.assertEqual(len(res), 2)
        self.assertAlmostEqual(res[0], -7.645978719, places=5)
        self.assertAlmostEqual(res[1], 5.494463568, places=5)

    def test2(self):
        res = line_sphere_intersection(0.8, torch.tensor([7.7, -0.5, 0.44]), torch.tensor([144, 4.11, 4.144]))
        self.assertEqual(len(res), 2)
        self.assertAlmostEqual(res[0], -0.0552663833, places=5)
        self.assertAlmostEqual(res[1], -0.0514803572, places=5)

    def test3(self):
        res = line_sphere_intersection(464, torch.tensor([7.0, 1.0, 0.0]), torch.tensor([-6.0, 2.0, 42.0]))
        self.assertEqual(len(res), 2)
        self.assertAlmostEqual(res[0], -10.90103428, places=5)
        self.assertAlmostEqual(res[1], 10.945380178, places=5)

    def test4(self):
        res = line_sphere_intersection(60000, torch.tensor([9977.0, 1042.0, 455.0]), torch.tensor([-1.0, 1.0, 4.0]))
        self.assertEqual(len(res), 2)
        self.assertAlmostEqual(res[0], -13552.997869112, places=2)
        self.assertAlmostEqual(res[1], 14343.553424667, places=2)

    def test5(self):
        res = line_sphere_intersection(148, torch.tensor([97.0, 104.0, -45.0]), torch.tensor([-5.0, 10.0, 8.0]))
        self.assertEqual(len(res), 0)

if __name__ == "__main__":
    unittest.main()

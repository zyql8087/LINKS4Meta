import unittest
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.kinematics_extract import compute_angle, normalize_trajectory, extract_kinematics

class TestKinematics(unittest.TestCase):
    def test_compute_angle(self):
        # 90 degrees
        p1 = np.array([1, 0])
        p2 = np.array([0, 0])
        p3 = np.array([0, 1])
        angle = compute_angle(p1, p2, p3)
        self.assertAlmostEqual(angle, np.pi/2)
        
        # 180 degrees
        p1 = np.array([1, 0])
        p2 = np.array([0, 0])
        p3 = np.array([-1, 0])
        angle = compute_angle(p1, p2, p3)
        self.assertAlmostEqual(angle, np.pi, places=3)
        
        # 0 degrees (degenerate)
        p1 = np.array([1, 0])
        p2 = np.array([0, 0])
        p3 = np.array([1, 0])
        angle = compute_angle(p1, p2, p3)
        self.assertAlmostEqual(angle, 0.0, places=3)

    def test_normalize_trajectory(self):
        traj = np.array([[0, 0], [10, 20], [5, 10]])
        norm_traj = normalize_trajectory(traj)
        
        self.assertAlmostEqual(norm_traj.min(), 0.0)
        self.assertAlmostEqual(norm_traj.max(), 1.0)
        
        # Check specific points
        # x range [0, 10], y range [0, 20]
        # [0,0] -> [0,0]
        self.assertAlmostEqual(norm_traj[0, 0], 0.0) 
        self.assertAlmostEqual(norm_traj[0, 1], 0.0)
        # [10, 20] -> [1, 1]
        self.assertAlmostEqual(norm_traj[1, 0], 1.0)
        self.assertAlmostEqual(norm_traj[1, 1], 1.0)
        # [5, 10] -> [0.5, 0.5]
        self.assertAlmostEqual(norm_traj[2, 0], 0.5)
        self.assertAlmostEqual(norm_traj[2, 1], 0.5)

if __name__ == '__main__':
    unittest.main()

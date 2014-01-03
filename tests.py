#!/usr/bin/env python3
import unittest
from matrix import *

class TestMatrixOperations(unittest.TestCase):

    def setUp(self):
        self.v1 = Vector(1, 2, 3)
        self.v2 = Matrix([4, 5, 6])
        self.m1 = Matrix([1, 0, 0], [0, 1, 0], [0, 0, 1])
        self.m2 = Matrix([4, 1, -7, 2], [-1, 9, 6, 3])
        self.m3 = Matrix([8, -3, 1], [4, -6, 2], [7, 3, 5], [-2, -5, 1])

    def testDimensions(self):
        self.assertEqual(self.v1.height, 3)
        self.assertEqual(self.v1.width, 1)
        self.assertEqual(self.v2.height, 1)
        self.assertEqual(self.v2.width, 3)
        self.assertEqual(self.m1.height, 3)
        self.assertEqual(self.m1.width, 3)
        self.assertEqual(self.m2.height, 2)
        self.assertEqual(self.m2.width, 4)
        self.assertEqual(self.m3.height, 4)
        self.assertEqual(self.m3.width, 3)

    def testComparisons(self):
        self.assertTrue(self.v1 == Matrix([[1], [2], [3]]))
        self.assertFalse(self.v1 != Matrix([[1], [2], [3]]))
        self.assertFalse(self.m3 == self.m2)
        self.assertTrue(self.m1 != self.v2)
        self.assertFalse(self.m1 == 14)

    def testAddition(self):
        tm1 = self.v1 + self.v2.T
        self.assertEqual(tm1, Vector(5, 7, 9))
        tm2 = self.m2 + Matrix([1, 2, 3, 4], [-5, -6, -7, -8])
        self.assertEqual(tm2, Matrix([5, 3, -4, 6], [-6, 3, -1, -5]))
        with self.assertRaises(Exception):
            self.m2 + self.m3
        with self.assertRaises(Exception):
            self.v1 + self.v2

    def testSubtraction(self):
        tm1 = self.v1 - self.v2.T
        self.assertEqual(tm1, Vector(-3, -3, -3))
        tm2 = self.m3 - Matrix([1, 2, 3], [-4, 5, 6], [7, 8, -9], [10, -11, 12])
        self.assertEqual(tm2, Matrix([7, -5, -2], [8, -11, -4], [0, -5, 14], [-12, 6, -11]))
        with self.assertRaises(Exception):
            self.m2 - self.m3
        with self.assertRaises(Exception):
            self.v1 - self.v2

    def testWrongConstructions(self):
        self.assertRaises(Exception, Matrix, [[1, 2, 3], [1, 2, 3, 4]])
        self.assertRaises(Exception, Matrix, [[1, 2, 'a']])
        self.assertRaises(Exception, Matrix, [[1, 2, 3], [4, 5, [6, 7, 8]]])
        self.assertRaises(Exception, Matrix, [])

    def testMultiplication(self):
        self.assertEqual(self.v1 * 3, Vector(3, 6, 9))
        self.assertEqual(self.m2 * 0.5, Matrix([[2, 0.5, -3.5, 1], [-0.5, 4.5, 3, 1.5]]))
        self.assertEqual(2 * self.m1, Matrix([2, 0, 0], [0, 2, 0], [0, 0, 2]))
        self.assertEqual(self.m2 * self.m3, Matrix([-17, -49, -27], [64, -48, 50]))
        self.assertEqual(self.v2 * self.v1, 32)
        self.assertEqual(self.m1 * (self.v2 * self.v1), Matrix([32, 0, 0], [0, 32, 0], [0, 0, 32]))
        self.assertEqual(self.v2 * self.v1 * self.m1, Matrix([32, 0, 0], [0, 32, 0], [0, 0, 32]))
        with self.assertRaises(Exception):
            m3 * m2

    def testTranspose(self):
        self.assertEqual(self.m1, self.m1.T)
        self.assertEqual(self.m2.T.T, self.m2)
        self.assertEqual(self.m3.T, Matrix([8, 4, 7, -2], [-3, -6, 3, -5], [1, 2, 5, 1]))
        self.assertEqual(self.v2.T, Vector([4, 5, 6]))

    def testVector(self):
        self.assertTrue(self.v1.is_vector)
        self.assertTrue(self.v1.is_col_vector)
        self.assertFalse(self.v1.is_row_vector)
        self.assertTrue(self.v2.is_vector)
        self.assertFalse(self.v2.is_col_vector)
        self.assertTrue(self.v2.is_row_vector)
        self.assertFalse(self.m1.is_vector)
        self.assertFalse(self.m1.is_col_vector)
        self.assertFalse(self.m1.is_row_vector)

    def testNorm(self):
        self.assertAlmostEqual(self.m3.norm(type=1), 21)
        self.assertAlmostEqual(self.v1.norm(type=2), 3.741657387)
        self.assertAlmostEqual(self.v2.norm(type=2), 8.774964387)
        #self.assertAlmostEqual(self.m3.norm(type=2), 12.51910405)
        self.assertAlmostEqual(self.m3.norm(type='inf'), 15)
        self.assertAlmostEqual(self.m3.norm(type='fro'), 15.58845727)
        self.assertEqual(self.v1.norm(), self.v1.T.norm())
        self.assertEqual(self.v2.norm(), self.v2.T.norm())
        self.assertRaises(Exception, self.v1.norm, type='non-existant')

    def testGetRowCol(self):
        self.assertEqual(self.v1, self.v1.get_col(0))
        self.assertEqual(self.m3.get_col(2), Vector(1, 2, 5, 1))
        self.assertEqual(self.m2.get_row(0), Vector(4, 1, -7, 2).T)
        self.assertRaises(Exception, self.m1.get_row, -1)
        self.assertRaises(Exception, self.m1.get_row, 3)
        self.assertRaises(Exception, self.m1.get_col, -1)
        self.assertRaises(Exception, self.m1.get_col, 3)

    def testComplexNumbers(self):
        tm1 = Matrix([5+6j, 2+4j, -6-2j])
        tm2 = Vector(5-6j, 2, 7-2j)
        # TODO: how should the manhattan norm be computed with a row vector?
        # Matlab does it like the test below
        #self.assertAlmostEqual(tm1.norm(type=1), 18.60694095)
        # but I think this variant is the correct one:
        self.assertAlmostEqual(tm1.norm(type=1), 7.81024968)
        self.assertAlmostEqual(tm2.norm(type=1), 17.09035957)
        self.assertAlmostEqual(tm1.norm(type=2), 11)
        self.assertAlmostEqual(tm2.norm(type=2), 10.86278049)
        self.assertAlmostEqual(tm1.norm(type='fro'), 11)
        self.assertAlmostEqual(tm2.norm(type='fro'), 10.86278049)
        tm3 = tm2 * tm1
        self.assertEqual(tm3, Matrix([61, 34+8j, -42+26j], [10+12j, 4+8j, -12-4j], [47+32j, 22+24j, -46-2j]))
        self.assertAlmostEqual(tm3.norm(type=1), 133.47997526)
        #self.assertAlmostEqual(tm3.norm(type=2), 119.4905854)
        self.assertAlmostEqual(tm3.norm(type='inf'), 145.32485453)
        self.assertAlmostEqual(tm3.norm(type='fro'), 119.4905854)
        self.assertEqual(tm1 * tm2, 19+6j)

    def testReshape(self):
        tm1 = Matrix(8, -3, 1, 4, -6, 2, 7, 3, 5, -2, -5, 1)
        tm2 = tm1.reshape(4, 3)
        tm3 = tm1.reshape(height=4)
        tm4 = tm1.reshape(width=3)
        self.assertEqual(self.m3, tm2)
        self.assertEqual(tm2, tm3)
        self.assertEqual(tm3, tm4)
        self.assertEqual(tm1.width, 12)
        self.assertEqual(tm1.height, 1)
        self.assertEqual(tm2.width, 3)
        self.assertEqual(tm2.height, 4)
        self.assertEqual(tm3.width, 3)
        self.assertEqual(tm3.height, 4)
        self.assertEqual(tm4.width, 3)
        self.assertEqual(tm4.height, 4)
        tm5 = tm3.reshape(3, 4)
        self.assertEqual(tm5, Matrix([8, -3, 1, 4], [-6, 2, 7, 3], [5, -2, -5, 1]))
        self.assertEqual(tm5.width, 4)
        self.assertEqual(tm5.height, 3)
        self.assertEqual(self.v1.T, self.v1.reshape(height=1))
        self.assertRaises(Exception, tm1.reshape, height=5)
        self.assertRaises(Exception, tm1.reshape, width=0)
        self.assertRaises(Exception, tm1.reshape, 3, 3)

    def testDiag(self):
        # Get the diagonal of a Matrix
        self.assertEqual(self.m1.diag, Vector(1, 1, 1))
        self.assertEqual(self.m2.diag, Vector(4, 9))
        self.assertEqual(self.m3.diag, Vector(8, -6, 5))
        self.assertEqual(self.m3.diag, self.m3.T.diag)
        # Make a Vector to a diagonal Matrix
        self.assertEqual(diag(Vector(6)), Matrix(6))
        self.assertEqual(diag(self.v2.T), Matrix([4, 0, 0], [0, 5, 0], [0, 0, 6]))
        self.assertRaises(Exception, diag, self.v2)

    def testIdentity(self):
        self.assertEqual(identity(1), Vector(1))
        self.assertEqual(identity(2), Matrix([1, 0], [0, 1]))
        self.assertEqual(identity(3), Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        self.assertRaises(Exception, identity, 0)
        self.assertRaises(Exception, identity, -1)

    def testOnes(self):
        self.assertEqual(ones(1, 1), Matrix(1))
        self.assertEqual(ones(2, 2), Matrix([1, 1], [1, 1]))
        self.assertEqual(ones(5, 1), Vector(1, 1, 1, 1, 1))
        self.assertRaises(Exception, ones, 0, 2)
        self.assertRaises(Exception, ones, 4, -1)

    def testZeros(self):
        self.assertEqual(zeros(1, 1), Matrix(0))
        self.assertEqual(zeros(2, 2), Matrix([0, 0], [0, 0]))
        self.assertEqual(zeros(5, 1), Vector(0, 0, 0, 0, 0))
        self.assertRaises(Exception, zeros, 0, 2)
        self.assertRaises(Exception, zeros, 4, -1)

    def testCrossProduct(self):
        tv1 = Vector(4, 5, 6)
        self.assertEqual(self.v1 ^ tv1, Vector(-3, 6, -3))
        self.assertEqual(Vector(54, -234, 4+9j) ^ Vector(-6, 32, 8),
            Vector(-2000-288j, -456-54j, 324))
        with self.assertRaises(Exception):
            self.v1 ^ self.v2

    def testTrace(self):
        self.assertEqual(Matrix(5).trace, 5)
        self.assertEqual(Matrix([-2, 2, -4], [-1, 1, 3], [2, 0, -1]).trace, -2)
        with self.assertRaises(Exception):
            self.m3.trace

    def testInv(self):
        with self.assertRaises(Exception):
            Matrix((1, 2, 3), (5, 6, 7), (9, 10, 11)).inv
        self.assertEqual(identity(5).inv, identity(5))
        tm1 = Matrix([1, 3, 3], [1, 4, 3], [1, 3, 4])
        self.assertEqual(tm1.inv, Matrix([7, -3, -3], [-1, 1, 0], [-1, 0, 1]))
        tm2 = Matrix([1, 2, 3], [0, 1, 4], [5, 6, 0])
        self.assertEqual(tm2.inv, Matrix([-24, 18, 5], [20, -15, -4], [-5, 4, 1]))

if __name__ == '__main__':
    unittest.main()

pymatrix
========

pymatrix is a Python library that offers easily usable matrix arithmetics.
Just do an `import matrix` to use it.

__Note:__ Matrix instances are immutable and all operations will return a new instance

Usage
=====

Matrix Creation
---------------

To create a 2x3 matrix you can use one of the following (not the only options):

`matrix.Matrix([1, 2, 3], [4, 5, 6])`  
`matrix.Matrix([[1, 2, 3], [4, 5, 6]])`  
`matrix.Matrix(1, 2, 3, 4, 5, 6).reshape(2, 3)`  

The `Vector` function is a shorthand for creating Nx1 matrices:

`matrix.Vector(1, 2, 3)` is equal to `matrix.Matrix([1], [2], [3])`

Identities and matrices filled with ones or zeros can easily be created using:

`matrix.identity(size=3)`  
`matrix.ones(height=3, width=4)`  
`matrix.zeros(height=3, width=4)`  

Arithmetic Operations
---------------------

Matrix addition and multiplication work like expected:

`matrix1 * matrix2 - matrix3 + 2 * matrix4`

The cross-product between two vectors (3x1 matrices) can be computed as:

`vector1 ^ vector2`

Special functions
-----------------

The `T` property returns the transposed matrix:

`matrix1.T`

The `norm` function returns the computed norm:

`matrix1.norm()` and `matrix.norm(2)` return the Eucledian norm (only implemented for vectors)

`matrix1.norm(1)` returns the Manhattan norm

`matrix1.norm('inf')` returns the Uniform norm

`matrix1.norm('fro')` returns the Frobenius norm

The `det` property returns the determinant of the matrix:

`matrix1.det`

The `adj` property returns the adjugate matrix:

`matrix1.adj`

The `inv` property returns the inverse matrix:

`matrix1.inv`

The `trace` property returns the trace of a square matrix:

`matrix1.trace`

The `diag` property returns the diagonal of a matrix as a vector:

`matrix1.diag`

You can also stack matrices horizontally or vertically using the `stackh` and `stackv` functions:

`matrix.stackh(matrix1, matrix2, ...)`

`matrix.stackv(matrix1, matrix2, ...)`

The `cut` function cuts out a rectangular piece of a matrix:

`matrix1.cut(left=1, right=3, top=2, bottom=4)` (including left and top, excluding right and bottom)

The dimensions of a matrix can be changed by calling the `reshape` function:

`matrix1.reshape(height=3, width=2)`

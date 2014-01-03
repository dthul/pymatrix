import math

import sys
python_version = sys.version_info
major_version = python_version[0]
if major_version < 3:
    raise Exception('Python>=3 required')

class Matrix(object):
    def __init__(self, *values):
        assert len(values) > 0, 'The Matrix may not be empty'
        first_elem = values[0]
        if isnumeric(first_elem):
            self._fill([values])
            return
        assert isseries(first_elem), 'A Matrix needs to be created from numbers, tuples or lists.'
        assert len(first_elem) > 0, 'The Matrix needs to have at least one column and row'
        if isnumeric(first_elem[0]):
            self._fill(values)
        elif isseries(first_elem[0]):
            assert len(values) == 1, 'A Matrix can\'t have more than columns and rows'
            self._fill(first_elem)

    def _fill(self, values):
        first_elem = values[0]
        self._height = len(values)
        self._width = len(first_elem)
        for row in values:
            assert len(row) == self.width, 'All rows need to have the same length'
            for number in row:
                assert isnumeric(number), 'The Matrix may only contain numbers'
        self._values = tuple([tuple(row) for row in values])

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def values(self):
        return self._values

    def __repr__(self):
        s = 'Matrix('
        if len(self.values) > 1:
            s = s + '\n'
        for row in self.values:
            s = s + repr(row) + ',\n'
        return s[:-2] + ')'

    def __add__(self, other):
        assert isinstance(other, Matrix), 'Can only add Matrices'
        assert self.height == other.height and self.width == other.width, 'The Matrices to be added need to have the same size'
        return Matrix(tuple(map(lambda srow, orow: tuple(map(lambda x, y: x + y, srow, orow)), self.values, other.values)))

    def __sub__(self, other):
        assert isinstance(other, Matrix), 'Can only subtract Matrices'
        assert self.height == other.height and self.width == other.width, 'The Matrices to be subtracted need to have the same size'
        return Matrix(tuple(map(lambda srow, orow: tuple(map(lambda x, y: x - y, srow, orow)), self.values, other.values)))

    def __mul__(self, other):
        """Multiplication between a Matrix and a scalar or Matrix"""
        assert isinstance(other, Matrix) or isnumeric(other), 'Can only mutliply with a Matrix or a scalar'
        if isinstance(other, Matrix):
            return self._mulmm(other)
        elif isnumeric(other):
            return self._mulms(other)

    def __rmul__(self, other):
        """Multiplication that has a scalar as its first component"""
        return self.__mul__(other)

    def _mulmm(self, other):
        if other.is_scalar:
            return self._mulms(other.values[0][0])
        elif self.is_scalar:
            return other._mulms(self.values[0][0])
        assert self.width == other.height, 'The dimensions of the Matrices don\'t match'
        new_values = []
        for num_row in range(0, self.height):
            new_row = []
            for num_col in range(0, other.width):
                row = self.get_row(num_row, raw=True)
                col = other.get_col(num_col, raw=True)
                value = multvv(row, col)
                new_row.append(value)
            new_values.append(new_row)
        return Matrix(new_values)

    def _mulms(self, other):
        return Matrix(tuple(map(lambda row: tuple(map(lambda x: x * other, row)), self.values)))

    @property
    def T(self):
        return Matrix(tuple([self.get_col(i, raw=True) for i in range(0, self.width)]))

    def get_row(self, num_row, raw=False):
        assert num_row < self.height and num_row >= 0, 'Row doesn\'t exist'
        values = self.values[num_row]
        if raw:
            return values
        else:
            return Matrix(values)

    def get_col(self, num_col, raw=False):
        assert num_col < self.width and num_col >= 0, 'Column doesn\'t exist'
        values = tuple([self.values[i][num_col] for i in range(0, self.height)])
        if raw:
            return values
        else:
            return Vector(values)

    def get(self, num_row, num_col):
        return self._values[num_row][num_col]

    def __eq__(self, other):
        if isnumeric(other):
            if self.is_scalar:
                return self.values[0][0] == other
            else:
                return False
        assert isinstance(other, Matrix), 'Can only compare with matrices and scalars'
        return self.values == other.values

    def __ne__(self, other):
        if isnumeric(other):
            if self.is_scalar:
                return self.values[0][0] != other
            else:
                raise Exception('Can\'t compare matrix and scalar')
        assert isinstance(other, Matrix), 'Can only compare with matrices and scalars'
        return self.values != other.values

    def __hash__(self):
        return hash(self.values)

    def __complex__(self):
        assert self.is_scalar, 'Can only convert 1x1 matrices to a scalar'
        return complex(self.values[0][0])

    def __float__(self):
        assert self.is_scalar, 'Can only convert 1x1 matrices to a scalar'
        return float(self.values[0][0])

    def __xor__(self, other):
        """Computes the cross-product of two vectors"""
        assert isinstance(other, Matrix), 'Can only take the cross product between two vectors'
        assert self.height == 3 and self.width == 1 and other.height == 3 and other.width == 1,\
            'Can only take the cross-product of two 3x1 Matrices'
        u = self.get_col(0, raw=True)
        v = other.get_col(0, raw=True)
        return Vector(u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0])

    def __getitem__(self, index):
        return self._flat_values()[index]

    @property
    def is_row_vector(self):
        return self.height == 1

    @property
    def is_col_vector(self):
        return self.width == 1

    @property
    def is_vector(self):
        return self.is_row_vector or self.is_col_vector

    @property
    def is_scalar(self):
        return self.height == 1 and self.width == 1

    def norm(self, type=2):
        if type == 1:
            return self._norm1()
        elif type == 2:
            return self._norm2()
        elif type == 'inf':
            return self._norm_inf()
        elif type == 'fro':
            return self._norm_fro()
        else:
            raise Exception('Illegal norm type')

    def _norm1(self):
        """Manhattan Norm"""
        max = -1
        for j in range(0, self.width):
            value = sum(tuple(map(abs, self.get_col(j, raw=True))))
            if value > max:
                max = value
        return max

    def _norm2(self):
        """Eucledian Norm"""
        if not self.is_vector:
            # sqrt(dominant eigen value of A'A)
            raise NotImplementedError
        elif self.is_row_vector:
            # yes, the abs call is necessary to handle complex numbers
            return math.sqrt(sum(tuple(map(lambda x: abs(x**2), self.get_row(0, raw=True)))))
        elif self.is_col_vector:
            # yes, the abs call is necessary to handle complex numbers
            return math.sqrt(sum(tuple(map(lambda x: abs(x**2), self.get_col(0, raw=True)))))

    def _norm_inf(self):
        """Uniform Norm"""
        max = -1
        for i in range(0, self.height):
            value = sum(tuple(map(abs, self.get_row(i, raw=True))))
            if value > max:
                max = value
        return max

    def _norm_fro(self):
        """Frobenius Norm"""
        sum = 0
        for i in range(0, self.height):
            for j in range(0, self.width):
                value = self.values[i][j]
                # yes, the abs call is necessary to handle complex numbers
                sum += abs(value**2)
        return math.sqrt(sum)

    def reshape(self, height = -1, width = -1):
        assert height > 0 or width > 0, 'One dimension needs to be at least 1'
        assert is_whole(height) and is_whole(width), 'Can only use whole numbers'
        if height == -1 and width > 0:
            height = (self.width * self.height) / width
            assert is_whole(height), 'Can\'t reshape because new dimension doesn\'t fit'
            height = int(height)
        elif width == -1 and height > 0:
            width = (self.width * self.height) / height
            assert is_whole(width), 'Can\'t reshape because new dimension doesn\'t fit'
            width = int(width)
        elif width > 0 and height > 0:
            assert width * height == self.width * self.height, 'Can\'t reshape because new dimension doesn\'t fit'
        else:
            raise Exception('Illegal dimensions specified')
        values = self._flat_values()
        new_values = []
        for row in range(0, height):
            new_row = []
            for col in range(0, width):
                new_row.append(values[row * width + col])
            new_values.append(new_row)
        return Matrix(new_values)

    @property
    def diag(self):
        """Returns the diagonal of a Matrix as a Vector"""
        mindim = min(self.height, self.width)
        return Vector([self.values[i][i] for i in range(0, mindim)])

    @property
    def trace(self):
        assert self.is_square, 'Can only compute the trace of a square Matrix'
        sum = 0
        for i in range(0, self.height):
            sum += self.values[i][i]
        return sum

    def cut(self, left = 0, right = None, top = 0, bottom = None):
        """
        Cuts a rectangular piece out of this Matrix.
        From row top (inclusive) to row bottom (exclusive) and
        from column left (inclusive) to column right (exclusive).
        The resulting Matrix will have a height of bottom - top
        and a width of right - left
        """
        if right is None:
            right = self.width
        if bottom is None:
            bottom = self.height
        assert left >= 0 and left < self.width, 'left out of bounds'
        assert right > 0 and right <= self.width, 'right out of bounds'
        assert top >= 0 and top < self.height, 'top out of bounds'
        assert bottom > 0 and bottom <= self.height, 'bottom out of bounds'
        assert left < right, 'left must be smaller than right'
        assert top < bottom, 'top must be smaller than bottom'
        width = right - left
        height = bottom - top
        flat_values = self._flat_values()
        values = []
        for row in range(0, height):
            newrow = []
            for col in range(0, width):
                value = flat_values[self.width * top + left + self.width * row + col]
                newrow.append(value)
            values.append(newrow)
        return Matrix(values)

    def _A_ij(self, i, j):
        """Returns the Matrix with row i and column j removed"""
        assert i >= 0 and i < self.height, 'i out of bounds'
        assert j >= 0 and j < self.width, 'j out of bounds'
        if i == 0:
            m1 = self.cut(top=1)
        elif i == self.height - 1:
            m1 = self.cut(bottom=self.height - 1)
        else:
            tm1 = self.cut(bottom=i)
            tm2 = self.cut(top=i+1)
            m1 = stackv(tm1, tm2)
        if j == 0:
            m2 = m1.cut(left=1)
        elif j == m1.width - 1:
            m2 = m1.cut(right=m1.width - 1)
        else:
            tm1 = m1.cut(right=j)
            tm2 = m1.cut(left=j+1)
            m2 = stackh(tm1, tm2)
        return m2

    @property
    def det(self):
        """Computes the determinant of the Matrix"""
        assert self.is_square, 'Can only compute the determinant of a square Matrix'
        if self.height == 1:
            return self.values[0][0]
        i = 0 # can be chosen arbitrarily (smaller than self.height)
        sum = 0
        for j in range(0, self.width):
            if self.values[i][j] == 0:
                continue
            value = (-1)**(i+j) * self.values[i][j] * self._A_ij(i, j).det
            sum += value
        return sum

    @property
    def adj(self):
        """Computes the adjugate of the Matrix"""
        assert self.is_square, 'Can only compute the adjugate of a square Matrix'
        values = []
        for i in range(0, self.height):
            new_row = []
            for j in range(0, self.width):
                value = (-1)**(i+j) * self._A_ij(j, i).det
                new_row.append(value)
            values.append(new_row)
        return Matrix(values)

    @property
    def inv(self):
        assert self.is_square, 'Can only compute the inverse of a square Matrix'
        if self.height == 1:
            return Matrix(1 / self.values[0][0])
        d = self.det
        if abs(d) < 10**-4:
            raise Exception('Matrix is not invertible')
        return 1 / d * self.adj

    @property
    def is_square(self):
        return self.width == self.height

    def _flat_values(self):
        return [number for sublist in self.values for number in sublist]

def stackh(*matrices):
    matrices = _normalize_args(matrices)
    assert len(matrices) > 0, 'Can\'t stack zero matrices'
    for matrix in matrices:
        assert isinstance(matrix, Matrix), 'Can only stack matrices'
    height = matrices[0].height
    for matrix in matrices:
        assert matrix.height == height, 'Can\'t horizontally stack matrices with different heights'
    values = []
    for row in range(0, height):
        newrow = []
        for matrix in matrices:
            newrow += matrix.get_row(row, raw=True)
        values.append(newrow)
    return Matrix(values)

def stackv(*matrices):
    matrices = _normalize_args(matrices)
    assert len(matrices) > 0, 'Can\'t stack zero matrices'
    for matrix in matrices:
        assert isinstance(matrix, Matrix), 'Can only stack matrices'
    width = matrices[0].width
    for matrix in matrices:
        assert matrix.width == width, 'Can\'t vertically stack matrices with different widths'
    values = []
    for matrix in matrices:
        values += matrix.values
    return Matrix(values)

def _normalize_args(matrices):
    if len(matrices) > 0:
        first_elem = matrices[0]
        if isseries(first_elem):
            assert len(matrices) == 1, 'Couldn\'t normalize arguments'
            return first_elem
        return matrices
    return matrices

def is_whole(x):
    return x % 1 == 0

def Vector(*values):
    assert len(values) > 0, 'A Vector may not be empty'
    first_elem = values[0]
    if isnumeric(first_elem):
        numbers = values
    elif isseries(first_elem):
        assert len(values) == 1, 'A Vector may only contain one column'
        assert len(first_elem) > 0, 'A Vector may not be empty'
        numbers = first_elem
    for number in numbers:
        assert isnumeric(number), 'A Vector may only contain numbers'
    return Matrix(tuple([(x,) for x in numbers]))

def multvv(v1, v2):
    assert isseries(v1) and isseries(v2)
    assert len(v1) == len(v2)
    return sum(tuple(map(lambda x, y: x * y, v1, v2)))

def identity(size):
    m = Matrix([1 if i == j else 0 for i in range(0, size) for j in range(0, size)])
    return m.reshape(size)

def ones(height, width):
    m = Matrix(height * width * [1])
    return m.reshape(height, width)

def zeros(height, width):
    m = Matrix(height * width * [0])
    return m.reshape(height, width)

def diag(v):
    """Creates a diagonal Matrix from a Vector"""
    assert v.is_col_vector, 'Can only put column vector on the diagonal'
    m = Matrix([v.values[i][0] if i == j else 0 for i in range(0, v.height) for j in range(0, v.height)])
    return m.reshape(v.height)

def isnumeric(value):
    return isinstance(value, int) or isinstance(value, float) or isinstance(value, complex)

def isseries(value):
    return isinstance(value, list) or isinstance(value, tuple)

import math

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

    def __eq__(self, other):
        if isnumeric(other):
            if self.is_scalar:
                return self.values[0][0] == other
            else:
                raise Exception('Can\'t compare matrix and scalar')
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

def isnumeric(value):
    return isinstance(value, int) or isinstance(value, float) or isinstance(value, complex)

def isseries(value):
    return isinstance(value, list) or isinstance(value, tuple)

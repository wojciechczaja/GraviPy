"""
GraviPy.

Tensorial module implements a tensor components objects for
Coordinates, Metric, Christoffel, Ricci, Riemann etc.
"""

from collections import OrderedDict
from sympy import symbols, Symbol, Matrix, Function, sympify, variations, KroneckerDelta, Rational
from sympy import diag, sin, zeros
if 'reduce' not in vars():
    from functools import reduce
apply_tensor_symmetry = True


class GeneralTensor(object):
    """GeneralTensor.

    Represents tensor components objects in a particular Coordinate System.
    Tensor class should be extended rather than GeneralTensor class to create
    a new Tensor object.
    """

    GeneralTensorObjects = []

    def __init__(self, symbol, rank, coords, metric=None,
                 conn=None, *args, **kwargs):
        GeneralTensor.GeneralTensorObjects.append(self)
        self.is_tensor = True
        self.is_coordinate_tensor = False
        self.is_metric_tensor = False
        self.is_connection = False
        self.symbol = Symbol(symbol)
        self.rank = rank
        self.coords = coords
        self.metric = metric
        self.conn = conn
        self.dim = len(coords)
        self.components = {}
        self.partial_derivative_components = {}
        self.covariant_derivative_components = {}
        self.index_values = {-1: list(range(-self.dim, 0)),
                             0: list(range(-self.dim, 0)) +
                             list(range(1, self.dim + 1)),
                             1: list(range(1, self.dim + 1))}
        self._set_index_types(kwargs)
        if 'apply_tensor_symmetry' in kwargs:
            self.apply_tensor_symmetry = bool(kwargs['apply_tensor_symmetry'])
        else:
            self.apply_tensor_symmetry = apply_tensor_symmetry

    def __call__(self, *idxs):
        if self._proper_tensor_indexes(idxs) == 'component_mode':
            idxs = tuple(map(int, idxs))
            if idxs in self.components.keys():
                return self.components[idxs]
            else:
                if all([idxs[i] > 0 for i in range(self.rank)]):
                    return self._compute_covariant_component(idxs)
                else:
                    return self._compute_exttensor_component(idxs)
        elif self._proper_tensor_indexes(idxs) == 'matrix_mode':
            return self._matrix_form(idxs)
        else:
            raise GraviPyError('Unexpected tensor index error')

    def _matrix_form(self, idxs):
        allidxs = {i + 1: idxs[i] / abs(idxs[i])
                   for i in range(self.rank)
                   if isinstance(abs(idxs[i]), AllIdx)}
        if len(allidxs) % 2 != 0:
            allidxs.update({0: 0})
        allidxs = OrderedDict(sorted(allidxs.items(), key=lambda t: t[0]))
        paidxs = tuple((list(allidxs.keys())[i:i + 2]
                        for i in range(len(allidxs)) if i % 2 == 0))
        nidxs = list(idxs)
        # print(allidxs, paidxs, nidxs)

        def _rnidxs(idxs, pair, allidxs, k, l):
            for i in range(len(idxs)):
                if i + 1 == pair[0]:
                    idxs[i] = k * allidxs[pair[0]]
                if i + 1 == pair[1]:
                    idxs[i] = l * allidxs[pair[1]]
            return idxs

        if paidxs[0][0] == 0:
            M = Matrix(1, self.dim, lambda k, l:
                       self.__call__(*(_rnidxs(
                           nidxs, paidxs[0], allidxs, k + 1, l + 1))))
        else:
            M = Matrix(self.dim, self.dim, lambda k, l:
                       self.__call__(*(_rnidxs(
                           nidxs, paidxs[0], allidxs, k + 1, l + 1))))
        return M

    def _set_index_types(self, kwargs):
        if 'index_types' in kwargs:
            if isinstance(kwargs['index_types'], (list, tuple)) \
               and len(kwargs['index_types']) == self.rank \
               and all(kwargs['index_types'][i] in [-1, 0, 1]
                       for i in range(self.rank)):
                self.index_types = kwargs['index_types']
            else:
                raise GraviPyError('Incorrect index_types list')
        else:
            self.index_types = [0] * self.rank

    def _proper_tensor_indexes(self, *idxs):
        prank = len(idxs[0]) == self.rank
        if prank \
           and all([idxs[0][i] in self.index_values[self.index_types[i]]
                    for i in range(self.rank)]):
            return 'component_mode'
        elif prank \
                and all([idxs[0][i] in self.index_values[self.index_types[i]] or
                         (isinstance(abs(idxs[0][i]), AllIdx) and
                          idxs[0][i] / abs(idxs[0][i]) +
                          self.index_types[i] != 0)
                         for i in range(self.rank)]):
            return 'matrix_mode'
        else:
            raise GraviPyError('Tensor component ' + str(self.symbol) +
                               str(idxs[0]) + ' doesn\'t  exist')

    def _proper_derivative_indexes(self, *idxs):
        if all([idxs[0][i] in self.index_values[1]
                for i in range(len(idxs[0]))]):
            return True
        else:
            raise GraviPyError('Derivative component ' + str(idxs[0]) +
                               ' doesn\'t  exist')

    def _connection_required(self, metric):
        if metric.conn is None or not isinstance(metric.conn, Christoffel):
            raise GraviPyError('Christoffel object for metric ' +
                               str(metric.symbol) + ' is required')

    @staticmethod
    def get_nmatrixel(M, idxs):
        if not isinstance(idxs, (tuple)):
            raise GraviPyError('Incorrect "idxs" parameter')
        idxl = list(idxs)
        if len(idxl) == 1:
            return M[idxl[0]]
        elif len(idxl) == 2:
            return M[idxl[0], idxl[1]]
        else:
            if len(idxl) % 2 == 1:
                idx = idxl.pop(0)
                return Tensor.get_nmatrixel(M[idx], tuple(idxl))
            else:
                idx1 = idxl.pop(0)
                idx2 = idxl.pop(0)
                return Tensor.get_nmatrixel(M[idx1, idx2], tuple(idxl))

    def _compute_covariant_component(self, idxs):
        if len(idxs) == 0:
            component = Function(str(self.symbol))(*self.coords.c)
        elif len(idxs) == 1:
            component = Function(str(self.symbol) +
                                 '(' + str(idxs[0]) + ')')(*self.coords.c)
        else:
            component = Function(str(self.symbol) + str(idxs))(*self.coords.c)
        return component

    def _compute_exttensor_component(self, idxs):
        idxdict = dict(enumerate(idxs))
        idxargs = dict(enumerate(idxs))
        idxargs.update(dict({(i, 'c' + str(i)) for i in range(len(idxs))
                             if sympify(idxs[i]).is_negative}))
        rl = [k for k in idxdict if sympify(idxdict[k]).is_negative]
        ii = tuple([list(idxdict.values())[i] for i in rl])
        ij = tuple([list(idxargs.values())[i] for i in rl])
        tsum = 0
        for ij in list(variations(range(1, self.dim + 1), len(ij), True)):
            idxargs.update(zip(rl, ij))
            tmul = self(*idxargs.values())
            for i in range(len(ii)):
                tmul = tmul * self.metric(ii[i], -(ij[i]))
            tsum = tsum + tmul
        component = tsum.together()
        self.components.update({idxs: component})
        return component

    def partialD(self, *idxs):
        idxs = tuple(map(int, idxs))
        if len(idxs) <= self.rank:
            raise GraviPyError('Number of indexes must be greater' +
                               ' than tensor rank')
        tidxs = idxs[0:self.rank]
        didxs = idxs[self.rank:]
        self._proper_tensor_indexes(tidxs)
        self._proper_derivative_indexes(didxs)
        if idxs in self.partial_derivative_components.keys():
            return self.partial_derivative_components[idxs]
        else:
            component = self(*tidxs).diff(*map(self.coords,
                                               map(lambda x: -x, didxs)))
            self.partial_derivative_components.update({idxs: component})
            return component

    def covariantD(self, *idxs):
        self._connection_required(self.metric)
        idxs = tuple(map(int, idxs))
        if len(idxs) <= self.rank:
            raise GraviPyError('Number of indexes must be greater' +
                               ' than tensor rank')
        tidxs = idxs[0:self.rank]
        didxs = idxs[self.rank:]
        self._proper_tensor_indexes(tidxs)
        self._proper_derivative_indexes(didxs)
        if idxs in self.covariant_derivative_components.keys():
            return self.covariant_derivative_components[idxs]
        else:
            nidxs = list(idxs)
            cidx = nidxs.pop(-1)
            if len(didxs) == 1:
                component = self(*tidxs).diff(self.coords(-cidx))
                for i in range(len(tidxs)):
                    sgn = tidxs[i] / abs(tidxs[i])
                    ci = dict(enumerate(tidxs))
                    for k in range(1, self.dim + 1):
                        ci.update({i: sgn * k})
                        cil = tuple(ci.values())
                        if tidxs[i] > 0:
                            component = component - \
                                self.metric.conn(-k, tidxs[i], cidx) \
                                * self(*cil)
                        else:
                            component = component + \
                                self.metric.conn(tidxs[i], k, cidx) \
                                * self(*cil)
            else:
                component = self.covariantD(*nidxs).diff(self.coords(-cidx))
                for i in range(len(nidxs)):
                    sgn = nidxs[i] / abs(nidxs[i])
                    ci = dict(enumerate(nidxs))
                    for k in range(1, self.dim + 1):
                        ci.update({i: sgn * k})
                        cil = tuple(ci.values())
                        if nidxs[i] > 0:
                            component = component - \
                                self.metric.conn(-k, nidxs[i], cidx) * \
                                self.covariantD(*cil)
                        else:
                            component = component + \
                                self.metric.conn(nidxs[i], k, cidx) * \
                                self.covariantD(*cil)
            component = component.together()
            self.covariant_derivative_components.update({idxs: component})
            return component


class Coordinates(GeneralTensor):
    r"""
    Represents a class of Coordinate n-vectors.

    Parameters
    ==========

    symbol : python string - name of the Coordinate n-vector
    coords : list of SymPy Symbol objects - list of coordinates

    Examples
    ========

    Define a Coordinate 4-vector:

    >>> from gravipy.tensorial import *
    >>> t, r, theta, phi = symbols('t, r, \\theta, \phi')
    >>> chi = Coordinates('\chi', [t, r, theta, phi])
    >>> chi(-1)
    t
    >>> chi(-3)
    \theta

    Python list representation of the coordinate tensor chi

    >>> chi.c
    [t, r, \theta, \phi]

    and it's SymPy Matrix representation

    >>> chi(-All)
    Matrix([[t, r, \theta, \phi]])

    >>> chi.components
    {(-1,): t, (-2,): r, (-3,): \theta, (-4,): \phi}

    Covariant components are not defined until MetricTensor object is not
    created

    >>> chi(1) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    GraviPyError: "Tensor component \\chi(1,) doesn't  exist"

    """

    def __init__(self, symbol, coords):
        super(Coordinates, self).__init__(symbol, 1, coords,
                                          index_types=[-1])
        self.is_coordinate_tensor = True
        self.coords = self
        self.c = coords
        self.components = {(-i - 1,): coords[i] for i in range(self.dim)}

    def __len__(self):
        return self.dim


class MetricTensor(GeneralTensor):
    r"""
    Represents a class of Metric Tensors.

    Parameters
    ==========

    symbol : python string - name of the Coordinate n-vector
    coords : GraviPy Coordinates object
    metric : SymPy Matrix object - metric tensor components in ``coords`` sytem

    Examples
    ========

    Define the Schwarzshild MetricTensor:

    >>> from gravipy.tensorial import *
    >>> t, r, theta, phi = symbols('t, r, \\theta, \phi')
    >>> chi = Coordinates('\chi', [t, r, theta, phi])
    >>> M = Symbol('M')
    >>> Metric = diag(-(1 - 2 * M / r), 1 / (1 - 2 * M / r), r ** 2,
    ...                  r ** 2 * sin(theta) ** 2)
    >>> g = MetricTensor('g', chi, Metric)

    SymPy matrix representation of the metric tensor

    >>> Matrix(4, 4, lambda i, j: g(i + 1, j + 1))
    Matrix([
    [2*M/r - 1,              0,    0,                   0],
    [        0, 1/(-2*M/r + 1),    0,                   0],
    [        0,              0, r**2,                   0],
    [        0,              0,    0, r**2*sin(\theta)**2]])

    or for short

    >>> g(All, All)
    Matrix([
    [2*M/r - 1,              0,    0,                   0],
    [        0, 1/(-2*M/r + 1),    0,                   0],
    [        0,              0, r**2,                   0],
    [        0,              0,    0, r**2*sin(\theta)**2]])

    Contravariant and mixed tensor component

    >>> g(-All, -All)
    Matrix([
    [1/(2*M/r - 1),          0,       0,                       0],
    [            0, -2*M/r + 1,       0,                       0],
    [            0,          0, r**(-2),                       0],
    [            0,          0,       0, 1/(r**2*sin(\theta)**2)]])

    >>> g(All, -All)
    Matrix([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]])

    Contravariant and covariant components of Coordinate 4-vector

    >>> chi(-All)
    Matrix([[t, r, \theta, \phi]])

    >>> chi(All)
    Matrix([[t*(2*M/r - 1), r/(-2*M/r + 1), \theta*r**2, \phi*r**2*sin(\theta)**2]])

    """

    def __init__(self, symbol, coords, metric):
        super(MetricTensor, self).__init__(symbol, 2, coords)
        self.is_metric_tensor = True
        self.coords = coords
        self.metric = self
        self.components.update({(i, j): metric[i - 1, j - 1]
                                for i in self.index_values[1]
                                for j in self.index_values[1]})
        self.components.update({(i, j): (metric ** -1)[-i - 1, -j - 1]
                                for i in self.index_values[-1]
                                for j in self.index_values[-1]})
        self.components.update({(i, j): KroneckerDelta(abs(i), abs(j))
                                for i in self.index_values[0]
                                for j in self.index_values[0] if i * j < 0})
        coords.metric = self
        coords.index_types = [0]
        coords.components.update({(i,): sum(self.components[i, j] *
                                            self.coords(-j)
                                            for j in self.index_values[1])
                                  for i in self.index_values[1]})


class Tensor(GeneralTensor):
    r"""Tensor.

    Represents a class of GraviPy or User defined tensor components objects
    in a particular Coordinate System.

    Tensor class should be extended to create a new tensor components object.

    Parameters
    ==========

    symbol : python string - name of the Coordinate n-vector
    rank: integer - rank of a tensor
    metric : GraviPy MtricTensor object
    components: (optional) nested SymPy Matrix object - user defined
        tensor components
    components_type: (optional) python tuple - type of user defined tensor
        components : 1 for covariant -1 for contravariant indexes - for example
        (1, -1) for T_i^k mixed components of tensor T

    Examples
    ========

    Define a second rank tensor T in the Schwarzshild spacetime:

    >>> from gravipy.tensorial import *
    >>> t, r, theta, phi = symbols('t, r, \\theta, \phi')
    >>> chi = Coordinates('\chi', [t, r, theta, phi])
    >>> M = Symbol('M')
    >>> Metric = diag(-(1 - 2 * M / r), 1 / (1 - 2 * M / r), r ** 2,
    ...                  r ** 2 * sin(theta) ** 2)
    >>> g = MetricTensor('g', chi, Metric)
    >>> T = Tensor('T', 2, g)
    >>> T(3, 4)
    T(3, 4)(t, r, \theta, \phi)

    Contravariant and mixed components of T-tensor:

    >>> T(3, -4)
    T(3, 4)(t, r, \theta, \phi)/(r**2*sin(\theta)**2)
    >>> T(3, 5) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    GraviPyError: "Tensor component T(3, 5) doesn't  exist"

    Partial derivative of the tensor:

    >>> T.partialD(3, -4, 1)
    Derivative(T(3, 4)(t, r, \theta, \phi), t)/(r**2*sin(\theta)**2)
    >>> T.partialD(3, -4, 3)
    -2*T(3, 4)(t, r, \theta, \phi)*cos(\theta)/(r**2*sin(\theta)**3) + Derivative(T(3, 4)(t, r, \theta, \phi), \theta)/(r**2*sin(\theta)**2)

    Covariant derivative of the tensor:
    >>> Ga = Christoffel('Ga', g)
    >>> T.covariantD(1, 2, 3)
    (r*Derivative(T(1, 2)(t, r, \theta, \phi), \theta) - T(1, 3)(t, r, \theta, \phi))/r

    """

    TensorObjects = []

    def __init__(self, symbol, rank, metric, conn=None,
                 components=None, components_type=None, *args, **kwargs):
        if not isinstance(metric, MetricTensor):
            raise GraviPyError(str(metric) + ' is not a MetricTensor object')
        super(Tensor, self).__init__(symbol, rank, metric.coords, metric,
                                     metric.conn, *args, **kwargs)
        if components is not None:
            co = components
            if not isinstance(co, Matrix):
                raise GraviPyError('The "components" parameter must be ' +
                                   'the SymPy Matrix object')
            if components_type is None:
                ct = tuple(1 for i in range(self.rank))
            else:
                if isinstance(components_type, tuple)\
                   and len(components_type) == self.rank\
                   and all(abs(i) == 1 for i in components_type):
                    ct = components_type
                else:
                    raise GraviPyError('Incorrect "component_type" parameter')
            ctdict = dict(enumerate(ct))
            rl = [k for k in ctdict if ctdict[k] < 0]
            for id in list(variations(range(self.dim), self.rank, True)):
                idxs = tuple([i + 1 for i in id])
                if len(rl) == 0:
                    self.components.update({idxs:
                                            Tensor.get_nmatrixel(co, id)})
                else:
                    midxs = tuple([(id[i] + 1) * ct[i]
                                   for i in range(len(id))])
                    self.components.update({midxs:
                                            Tensor.get_nmatrixel(co, id)})
                    idxdict = dict(enumerate(idxs))
                    idxargs = dict(enumerate(idxs))
                    idxargs.update(dict({(i, 'c' + str(i))
                                         for i in range(len(idxs))
                                         if ct[i] < 0}))
                    ii = tuple([list(idxdict.values())[i] for i in rl])
                    ij = tuple([list(idxargs.values())[i] for i in rl])
                    tsum = 0
                    for ij in list(variations(range(1, self.dim + 1), len(ij),
                                              True)):
                        idxargs.update(zip(rl, ij))
                        tmul = Tensor.get_nmatrixel(
                            co, tuple([j - 1 for j in idxargs.values()]))
                        for i in range(len(ii)):
                            tmul = tmul * self.metric(ii[i], ij[i])
                        tsum = tsum + tmul
                    self.components.update({idxs: tsum})

        Tensor.TensorObjects.append(self)


class Christoffel(Tensor):
    r"""Christoffel.

    Represents a class of Christoffel symbols of the first and second kind.

    Parameters
    ==========

    symbol : python string - name of the Christoffel symbol
    metric : GraviPy MtricTensor object

    Examples
    ========

    Define a Christoffel symbols for the Schwarzschild metric:

    >>> from gravipy.tensorial import *
    >>> t, r, theta, phi = symbols('t, r, \\theta, \phi')
    >>> chi = Coordinates('\chi', [t, r, theta, phi])
    >>> M = Symbol('M')
    >>> Metric = diag(-(1 - 2 * M / r), 1 / (1 - 2 * M / r), r ** 2,
    ...                  r ** 2 * sin(theta) ** 2)
    >>> g = MetricTensor('g', chi, Metric)
    >>> Ga = Christoffel('Ga', g)
    >>> Ga(-1, 2, 1)
    -M/(r*(2*M - r))
    >>> Ga(2, All, All)
    Matrix([
    [M/r**2,               0,  0,                 0],
    [     0, -M/(2*M - r)**2,  0,                 0],
    [     0,               0, -r,                 0],
    [     0,               0,  0, -r*sin(\theta)**2]])
    >>> Ga(1, -1, 2) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    GraviPyError: "Tensor component Ga(1, -1, 2) doesn't  exist"

    """

    def __init__(self, symbol, metric, *args, **kwargs):
        super(Christoffel, self).__init__(
            symbol, 3, metric, index_types=(0, 1, 1), *args, **kwargs)
        self.is_connection = True
        self.conn = self
        self.metric.conn = self

    def _compute_covariant_component(self, idxs):
        component = Rational(1, 2) * (
            self.metric(idxs[0], idxs[1]).diff(self.coords(-idxs[2])) +
            self.metric(idxs[0], idxs[2]).diff(self.coords(-idxs[1])) -
            self.metric(idxs[1], idxs[2]).diff(self.coords(-idxs[0]))) \
            .together().simplify()
        self.components.update({idxs: component})
        if self.apply_tensor_symmetry:
            self.components.update({(idxs[0], idxs[2], idxs[1]): component})
        return component


class Ricci(Tensor):
    r"""Ricci.

    Represents a class of Ricci Tensors.

    Parameters
    ==========

    symbol : python string - name of the Ricci Tensor
    metric : GraviPy MtricTensor object

    Examples
    ========

    Define and calculate components of the Ricci Tensor for the Schwarzschild
    Metric:

    >>> from gravipy.tensorial import *
    >>> t, r, theta, phi = symbols('t, r, \\theta, \phi')
    >>> chi = Coordinates('\chi', [t, r, theta, phi])
    >>> M = Symbol('M')
    >>> Metric = diag(-(1 - 2 * M / r), 1 / (1 - 2 * M / r), r ** 2,
    ...                  r ** 2 * sin(theta) ** 2)
    >>> g = MetricTensor('g', chi, Metric)
    >>> Ga = Christoffel('Ga', g)
    >>> Ri = Ricci('Ri', g)
    >>> Ri(All, All)
    Matrix([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]])

    Ricci sacalar

    >>> Ri.scalar()
    0

    """

    def __init__(self, symbol, metric, *args, **kwargs):
        self._connection_required(metric)
        super(Ricci, self).__init__(symbol, 2, metric, metric.conn,
                                    *args, **kwargs)
        self._s = None

    def _compute_covariant_component(self, idxs):
        component = (sum(
            self.conn(-k, idxs[0], idxs[1]).diff(self.coords(-k))
            for k in self.index_values[1]) -
            sum(
                self.conn(-k, idxs[0], k).diff(self.coords(-idxs[1]))
                for k in self.index_values[1]) +
            sum(
                self.conn(-k, idxs[0], idxs[1]) * self.conn(-l, k, l)
                for k, l in list(variations(self.index_values[1], 2, True))) -
            sum(
                self.conn(-k, idxs[0], l) * self.conn(-l, idxs[1], k)
                for k, l in list(variations(self.index_values[1], 2, True)))
        ).together().simplify()
        self.components.update({idxs: component})
        if self.apply_tensor_symmetry:
            self.components.update({(idxs[1], idxs[0]): component})
        return component

    def scalar(self):
        if self._s is None:
            self._s = sum(
                self.metric(-k, -l) * self(k, l) for k, l in
                list(variations(self.index_values[1], 2, True))
            ).together().simplify()
            return self._s
        else:
            return self._s


class Riemann(Tensor):
    r"""Riemann.

    Represents a class of Riemann Tensors.

    Parameters
    ==========

    symbol : python string - name of the Riemann Tensor
    metric : GraviPy MtricTensor object

    Examples
    ========

    Define and calculate some components of the Riemann Curvature Tensor
    for the Schwarzschild Metric:

    >>> from gravipy.tensorial import *
    >>> t, r, theta, phi = symbols('t, r, \\theta, \phi')
    >>> chi = Coordinates('\chi', [t, r, theta, phi])
    >>> M = Symbol('M')
    >>> Metric = diag(-(1 - 2 * M / r), 1 / (1 - 2 * M / r), r ** 2,
    ...                  r ** 2 * sin(theta) ** 2)
    >>> g = MetricTensor('g', chi, Metric)
    >>> Rm = Riemann('Rm', g) # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    GraviPyError: 'Christoffel object for metric g is required'
    >>> Ga = Christoffel('Ga', g)
    >>> Rm = Riemann('Rm', g)
    >>> Rm(1, -3, -1, 3)
    M*(-2*M + r)/(r**3*(2*M - r))
    >>> Rm(1, All, 1, All)
    Matrix([
    [0,         0,                 0,                                0],
    [0, -2*M/r**3,                 0,                                0],
    [0,         0, M*(-2*M + r)/r**2,                                0],
    [0,         0,                 0, M*(-2*M + r)*sin(\theta)**2/r**2]])

    Several ways that one can contract the Riemann Tensor:
    - by manual creation of SymPy Matrix object
    >>> Matrix(4, 4, lambda i, j: sum(Rm(-k, i+1, k, j+1)
    ...                                  for k in range(1,5)).simplify())
    Matrix([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]])

    - using sum or reduce and GraviPy Matrix creation procedure
    >>> cRm = sum([Rm(-i, All, i, All) for i in range(1, 5)], zeros(4))
    >>> cRm = reduce(Matrix.add, [Rm(-i, All, i, All) for i in range(1, 5)])
    >>> cRm.simplify()
    >>> cRm
    Matrix([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]])

    - by creation of a new Tensor object
    >>> cRm = Tensor('cRm', 2, g)
    >>> def cov_component(idxs):
    ...        return sum(Rm(-i, idxs[0], i, idxs[1])
    ...                   for i in range(1, 5)).simplify()
    >>> cRm._compute_covariant_component = cov_component
    >>> cRm(All, All)
    Matrix([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]])

    """

    def __init__(self, symbol, metric, *args, **kwargs):
        self._connection_required(metric)
        super(Riemann, self).__init__(symbol, 4, metric, metric.conn,
                                      *args, **kwargs)

    def _compute_covariant_component(self, idxs):
        if (idxs[0] == idxs[1] or idxs[2] == idxs[3]) \
           and apply_tensor_symmetry:
            component = sympify(0)
        else:
            component = (
                self.conn(idxs[0], idxs[1], idxs[3])
                .diff(self.coords(-idxs[2])) -
                self.conn(idxs[0], idxs[1], idxs[2])
                .diff(self.coords(-idxs[3])) +
                sum(self.conn(-k, idxs[1], idxs[3]) *
                    self.conn(idxs[0], idxs[2], k)
                    for k in self.index_values[1]) -
                sum(self.conn(-k, idxs[1], idxs[2]) *
                    self.conn(idxs[0], idxs[3], k)
                    for k in self.index_values[1]) -
                sum(self.metric(idxs[0], k).diff(self.coords(-idxs[2])) *
                    self.conn(-k, idxs[1], idxs[3])
                    for k in self.index_values[1]) +
                sum(self.metric(idxs[0], k).diff(self.coords(-idxs[3])) *
                    self.conn(-k, idxs[1], idxs[2])
                    for k in self.index_values[1])
            ).together().simplify()
            self.components.update({idxs: component})
            if self.apply_tensor_symmetry:
                self.components.update({(idxs[1], idxs[0], idxs[2], idxs[3]): -component})
                self.components.update({(idxs[0], idxs[1], idxs[3], idxs[2]): -component})
                self.components.update({(idxs[1], idxs[0], idxs[3], idxs[2]): component})
                self.components.update({(idxs[2], idxs[3], idxs[0], idxs[1]): component})
                self.components.update({(idxs[3], idxs[2], idxs[0], idxs[1]): -component})
                self.components.update({(idxs[2], idxs[3], idxs[1], idxs[0]): -component})
                self.components.update({(idxs[3], idxs[2], idxs[1], idxs[0]): component})
        return component


class Einstein(Tensor):
    r"""Einstein.

    Represents a class of Einstein Tensors.

    Parameters
    ==========

    symbol : python string - name of the Einstein Tensor
    ricci : GraviPy RicciTensor object

    Examples
    ========

    Define and calculate Einstein Tensor for the Schwarzschild Metric:

    >>> from gravipy.tensorial import *
    >>> t, r, theta, phi = symbols('t, r, \\theta, \phi')
    >>> chi = Coordinates('\chi', [t, r, theta, phi])
    >>> M = Symbol('M')
    >>> Metric = diag(-(1 - 2 * M / r), 1 / (1 - 2 * M / r), r ** 2,
    ...                  r ** 2 * sin(theta) ** 2)
    >>> g = MetricTensor('g', chi, Metric)
    >>> Ga = Christoffel('Ga', g)
    >>> Ri = Ricci('Rm', g)
    >>> G = Einstein('G', Ri)
    >>> G(All, All)
    Matrix([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]])

    """

    def __init__(self, symbol, ricci, *args, **kwargs):
        if not isinstance(ricci, Ricci):
            raise GraviPyError(str(ricci.symbol) + ' is not a Ricci object')
        super(Einstein, self).__init__(
            symbol, 2, ricci.metric, ricci.metric.conn, *args, **kwargs)
        self.ricci = ricci

    def _compute_covariant_component(self, idxs):
        component = (self.ricci(idxs[0], idxs[1]) - Rational(1, 2) *
                     self.metric(idxs[0], idxs[1]) * self.ricci.scalar()
                     ).together().simplify()
        self.components.update({idxs: component})
        return component


class Geodesic(Tensor):
    r"""Geodesic.

    Represents a class of Geodesic vectors - absolute derivatives
    of a world line tangent vectors.

    Parameters
    ==========

    symbol : python string - name of the Einstein Tensor
    metric : GraviPy MetricTensor object
    ptr: SymPy Symbol object - world-line parameter

    Examples
    ========

    Define and calculate components of a Geodesic vector
    in the Schwarzschild spacetime:

    >>> from gravipy.tensorial import *
    >>> t, r, theta, phi = symbols('t, r, \\theta, \phi')
    >>> chi = Coordinates('\chi', [t, r, theta, phi])
    >>> M = Symbol('M')
    >>> Metric = diag(-(1 - 2 * M / r), 1 / (1 - 2 * M / r), r ** 2,
    ...                  r ** 2 * sin(theta) ** 2)
    >>> g = MetricTensor('g', chi, Metric)
    >>> Ga = Christoffel('Ga', g)
    >>> Ri = Ricci('Rm', g)
    >>> tau = Symbol('\\tau')
    >>> S = Geodesic('S', g, tau)
    >>> S(1)
    -2*M*Derivative(r(\tau), \tau)*Derivative(t(\tau), \tau)/r(\tau)**2 + (2*M/r(\tau) - 1)*Derivative(t(\tau), (\tau, 2))

    Note that creation of Geodesic object for the metric g switches GraviPy
    into mode with parametrized coordinates g.coords

    >>> Parametrization.info()
    [[\chi, \tau]]
    >>> chi(-All)
    Matrix([[t(\tau), r(\tau), \theta(\tau), \phi(\tau)]])

    You can deactivate this mode calling Parametrization.deactivate method

    >>> Parametrization.deactivate(chi)
    >>> Parametrization.info()
    No parametrization activated
    >>> chi(-All)
    Matrix([[t, r, \theta, \phi]])

    """

    def __init__(self, symbol, metric, ptr, *args, **kwargs):
        if not isinstance(ptr, Symbol):
            raise GraviPyError(str(ptr) + ' is not a Symbol object')
        if metric.coords.symbol not in list(Parametrization.ptr.keys()):
            Parametrization.activate(metric.coords, ptr)
        else:
            if Parametrization.ptr[metric.coords.symbol] != ptr:
                raise GraviPyError(
                    str(ptr) + ' is not valid parameter for the metric ' +
                    str(metric.symbol))
        self.ptr = ptr
        super(Geodesic, self).__init__(
            symbol, 1, metric, metric.conn, *args, **kwargs)

    def _compute_covariant_component(self, idxs):
        if self.coords.symbol not in Parametrization.ptr.keys():
            raise GraviPyError(
                'Parametrization mode is not active for the metric ' +
                str(self.metric.symbol))
        component = sum(
            (self.metric(idxs[0], k) * self.coords(-k).diff(self.ptr))
            .diff(self.ptr) for k in self.index_values[1]) - \
            Rational(1, 2) * sum(
                self.metric(k, l).diff(self.coords(-idxs[0])) *
                self.coords(-k).diff(self.ptr) *
                self.coords(-l).diff(self.ptr)
                for k, l in list(variations(self.index_values[1], 2, True)))
        self.components.update({idxs: component})
        return component


class Parametrization(object):
    r"""Parametrization.

    Class consists of methods for management of coordinates explicit
    dependence on an arbitrary parameter.

    """

    ptr = {}
    ptr_coords = {}
    noptr_coords = {}

    @staticmethod
    def activate(coords, ptr):
        if not isinstance(coords, Coordinates):
            raise GraviPyError(str(coords) + ' is not a Coordinates object')
        if not isinstance(ptr, Symbol):
            raise GraviPyError(str(ptr) + ' is not a Symbol object')
        if coords.symbol in list(Parametrization.ptr.keys()):
            Parametrization.deactivate(coords)
        Parametrization.ptr_coords.update(
            {coords.symbol: {coords(-i): Function(str(coords(-i)))(ptr)
                             for i in coords.index_values[1]}})
        Parametrization.noptr_coords.update(
            {coords.symbol: {Function(str(coords(-i)))(ptr): coords(-i)
                             for i in coords.index_values[1]}})
        Parametrization.ptr.update({coords.symbol: ptr})
        for tensor in GeneralTensor.GeneralTensorObjects:
            if tensor.coords == coords and not isinstance(tensor, Geodesic):
                tensor.components = Parametrization.apply_ptr_subs_to_dict(
                    coords, tensor.components)
                if isinstance(tensor, Coordinates):
                    tensor.c = Parametrization.apply_ptr_subs_to_list(coords,
                                                                      tensor.c)
                tensor.partial_derivative_components = \
                    Parametrization.apply_ptr_subs_to_dict(
                        coords, tensor.partial_derivative_components)
                tensor.covariant_derivative_components = \
                    Parametrization.apply_ptr_subs_to_dict(
                        coords, tensor.covariant_derivative_components)

    @staticmethod
    def deactivate(coords, ptr=None):
        if coords.symbol in list(Parametrization.ptr.keys()):
            for tensor in GeneralTensor.GeneralTensorObjects:
                if tensor.coords == coords \
                   and not isinstance(tensor, Geodesic):
                    tensor.components = \
                        Parametrization.apply_ptr_subs_to_dict(
                            coords, tensor.components, -1)
                    if isinstance(tensor, Coordinates):
                        tensor.c = \
                            Parametrization.apply_ptr_subs_to_list(
                                coords, tensor.c, -1)
                    tensor.partial_derivative_components = \
                        Parametrization.apply_ptr_subs_to_dict(
                            coords, tensor.partial_derivative_components, -1)
                    tensor.covariant_derivative_components = \
                        Parametrization.apply_ptr_subs_to_dict(
                            coords, tensor.covariant_derivative_components, -1)
        if coords.symbol in list(Parametrization.ptr_coords.keys()):
            Parametrization.ptr_coords.pop(coords.symbol)
        if coords.symbol in list(Parametrization.noptr_coords.keys()):
            Parametrization.noptr_coords.pop(coords.symbol)
        if coords.symbol in list(Parametrization.ptr.keys()):
            Parametrization.ptr.pop(coords.symbol)

    @staticmethod
    def coords_to_ptr_function(coords, expr, direction=1):
        ptr_subs_dict = {1: Parametrization.ptr_coords,
                         -1: Parametrization.noptr_coords}
        if direction not in list(ptr_subs_dict.keys()):
            raise GraviPyError('Direction parameter have to be of list ' +
                               list(ptr_subs_dict.keys()) + ' member')
        else:
            if hasattr(expr, 'subs'):
                return expr.subs(ptr_subs_dict[direction][coords.symbol]).doit()
            else:
                return expr

    @staticmethod
    def apply_ptr_subs_to_dict(coords, dict_expr, direction=1):
        return {key: Parametrization.coords_to_ptr_function(
            coords, dict_expr[key], direction)
            for key in list(dict_expr.keys())}

    @staticmethod
    def apply_ptr_subs_to_list(coords, list_expr, direction=1):
        return [Parametrization.coords_to_ptr_function(
            coords, expr, direction) for expr in list_expr]

    @staticmethod
    def info():
        if len(Parametrization.ptr):
            return [[key, Parametrization.ptr[key]]
                    for key in list(Parametrization.ptr.keys())]
        else:
            print('No parametrization activated')


class AllIdx(Symbol):
    pass


All = AllIdx('All', positive=True)


class GraviPyError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


if __name__ == "__main__":
    import doctest
    doctest.testmod()

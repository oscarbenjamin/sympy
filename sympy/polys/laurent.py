from __future__ import annotations

from functools import reduce
from operator import add, mul

from sympy.core.add import Add
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify, CantSympify
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.laurentpolynomialring import LaurentPolynomialRing
from sympy.polys.orderings import LexOrder
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.rings import PolyElement, PolyRing
from sympy.printing.defaults import DefaultPrinting

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sympy.core.expr import Expr
    from typing import Iterable

Monomial = tuple[int, ...]


def laurent_ring(symbols, domain, order='lex'):
    """Construct a Laurent polynomial ring.

    Examples
    ========

    >>> from sympy.polys.laurent import laurent_ring
    >>> from sympy import ZZ
    >>> R, x, y = laurent_ring('x,y', ZZ, order='lex')
    >>> R
    Laurent polynomial ring in x, y over ZZ with lex order
    >>> p = x**2 + 1/y
    >>> p
    x**2 + 1/y
    >>> p ** 2
    x**4 + 2*x**2/y + 1/y**2
    """
    ring = LaurentPolyRing(symbols, domain, order=order)
    return (ring,) + ring.gens


class LaurentPolyRing:
    """Ring of Laurent polynomials.

    Examples
    ========

    >>> from sympy.polys.laurent import LaurentPolyRing
    >>> from sympy import ZZ
    >>> from sympy.abc import x, y
    >>> R = LaurentPolyRing([x, y], ZZ, order='lex')
    >>> R
    Laurent polynomial ring in x, y over ZZ with lex order
    >>> p = R.from_expr(1/x + x*y)
    >>> x, y = R.gens
    >>> p = 1/x + x*y
    >>> p
    x*y + 1/x
    >>> p ** 2
    x**2*y**2 + 2*y + 1/x**2
    """
    def __init__(self, symbols, domain, order=LexOrder()):
        numer_ring = PolyRing(symbols, domain, order=order)

        self.numer_ring = numer_ring
        self.domain = numer_ring.domain
        self.symbols = numer_ring.symbols
        self.order = numer_ring.order

        self.zero = self.from_polyelement(numer_ring.zero)
        self.one = self.from_polyelement(numer_ring.one)
        self.gens = tuple(self.from_polyelement(g) for g in numer_ring.gens)

        # Make the symbols accessible as R.x, R.y etc.
        for symbol, generator in zip(self.symbols, self.gens):
            if isinstance(symbol, Symbol):
                name = symbol.name
                if not hasattr(self, name):
                    setattr(self, name, generator)

    def __eq__(self, other):
        if not isinstance(other, LaurentPolyRing):
            return NotImplemented
        return self.numer_ring == other.numer_ring

    def __hash__(self):
        return hash(self.numer_ring)

    def __repr__(self):
        syms = ', '.join(map(str, self.symbols))
        return f"Laurent polynomial ring in {syms} over {self.domain} with {self.order} order"

    def new(self, numer, denom):
        """Construct a new Laurent polynomial from a numerator and denominator."""
        return LaurentPolyElement(self, numer, denom)

    def ring_new(self, element):
        if isinstance(element, tuple) and len(element) == 2:
            numer, denom = element
            numer = self.numer_ring.ring_new(numer)
            denom = self.numer_ring.ring_new(denom)
            return self.new(numer, denom)
        else:
            return self.ground_new(element)

    def ground_new(self, coeff):
        """Construct a new Laurent polynomial from an element of the ground domain."""
        return self.from_polyelement(self.numer_ring.ground_new(coeff))

    def from_polyelement(self, numer: PolyElement):
        """Construct a new Laurent polynomial from a polynomial."""
        assert self.numer_ring == numer.ring
        return LaurentPolyElement(self, numer, self.numer_ring.one)

    def from_expr(self, expr):
        """Construct a new Laurent polynomial from a SymPy expression."""
        mapping = dict(list(zip(self.symbols, self.gens)))

        try:
            poly = self._rebuild_expr(expr, mapping)
        except CoercionFailed:
            raise ValueError("expected an expression convertible to a polynomial in %s, got %s" % (self, expr))
        else:
            return poly

    def _rebuild_expr(self, expr, mapping):
        domain = self.domain

        def _rebuild(expr):
            generator = mapping.get(expr)

            if generator is not None:
                return generator
            elif expr.is_Add:
                return reduce(add, list(map(_rebuild, expr.args)))
            elif expr.is_Mul:
                return reduce(mul, list(map(_rebuild, expr.args)))
            else:
                # XXX: Use as_base_exp() to handle Pow(x, n) and also exp(n)
                # XXX: E can be a generator e.g. sring([exp(2)]) -> ZZ[E]
                base, exp = expr.as_base_exp()
                if exp.is_Integer and exp != 1:
                    return _rebuild(base)**int(exp)
                else:
                    return self.ground_new(domain.convert(expr))

        return _rebuild(sympify(expr))

    def to_field(self):
        """Return a ``FracField`` with the same generators."""
        from sympy.polys.fields import FracField
        return FracField(self.symbols, self.domain, self.order)

    def to_domain(self):
        return LaurentPolynomialRing(self)


class LaurentPolyElement(DomainElement, DefaultPrinting, CantSympify):
    """A class for representing Laurent polynomials.

    Examples
    ========

    >>> from sympy.polys.laurent import laurent_ring
    >>> from sympy import QQ
    >>> R, x, y = laurent_ring('x,y', QQ)
    >>> p = 1/x + x*y
    >>> p
    x*y + 1/x
    >>> type(p)
    <class 'sympy.polys.laurent.LaurentPolyElement'>
    """
    def __init__(self,
        ring: LaurentPolynomialRing,
        numer: PolyElement,
        denom: PolyElement,
    ):
        self._check(ring, numer, denom)
        self.ring = ring
        self.numer = numer
        self.denom = denom

    def parent(self):
        return self.ring.to_domain()

    def __eq__(self, other):
        if not isinstance(other, LaurentPolyElement):
            return NotImplemented
        return self.ring == other.ring and self.numer == other.numer and self.denom == other.denom

    def __hash__(self):
        return hash((self.ring, self.numer, self.denom))

    @classmethod
    def _check(self, ring, numer, denom, check_cancelled=True):
        assert numer.ring == ring.numer_ring
        assert denom.ring == ring.numer_ring
        min_degrees = tuple(map(min, zip(*numer.itermonoms())))
        [(denom_monom, denom_coeff)] = denom.iterterms()
        for d, m in zip(denom_monom, min_degrees):
            assert d >= 0 and m >= 0, "Negative exponents in numer or denom"
            if check_cancelled:
                assert d == 0 or m == 0, "Uncancelled exponents between numer and denom"

    @classmethod
    def from_dict(cls, ring, numer, denom):
        numer, denom = cls._normalize_dict(ring, numer, denom)
        return cls(ring, numer, denom)

    @classmethod
    def from_poly_denom(cls, ring, numer, denom):
        numer, denom = cls._normalize_poly(ring, numer, denom)
        return cls(ring, numer, denom)

    @classmethod
    def _normalize_dict(cls, ring, numer, denom):
        """Normalize a dict/denom representation of a Laurent polynomial.

        The monomials in the dict representation can have negative exponents.
        """
        numer_ring = ring.numer_ring
        monomial_mul = numer_ring.monomial_mul
        domain = numer_ring.domain

        min_degrees = map(min, zip(*numer))
        denom_new = [min(d - m, 0) for d, m in zip(denom, min_degrees)]
        monom_diff = tuple(d - m for d, m in zip(denom, denom_new))
        numer_new = {monomial_mul(m, monom_diff): c for m, c in numer.items()}

        numer_new_poly = numer_ring.from_dict(numer_new)
        denom_new_poly = numer_ring.term_new((denom_new, domain.one))

        return numer_new_poly, denom_new_poly

    @classmethod
    def _normalize_poly(cls, ring, numer, denom):
        """Normalize a poly/denom representation of a Laurent polynomial.

        The monomials are required to have non-negative exponents.
        """
        cls._check(ring, numer, denom, check_cancelled=False)

        numer_ring = ring.numer_ring
        monomial_gcd = numer_ring.monomial_gcd
        monomial_ldiv = numer_ring.monomial_ldiv
        domain = numer_ring.domain

        [denom_monom] = denom.itermonoms()

        if denom_monom == numer_ring.zero_monom:
            return numer, denom

        monom_gcd = denom_monom

        for monom in numer.itermonoms():
            monom_gcd = monomial_gcd(monom_gcd, monom)
            if monom_gcd == numer_ring.zero_monom:
                return numer, denom

        denom_monom_new = monomial_ldiv(denom_monom, monom_gcd)
        numer_new = {monomial_ldiv(m, monom_gcd): c for m, c in numer.items()}

        denom_new = numer_ring.term_new(denom_monom_new, domain.one)
        numer_new = numer_ring.from_dict(numer_new)

        return numer_new, denom_new

    def _terms_num_den(self):
        denom = self.denom
        term_new = self.ring.numer_ring.term_new
        terms = [term_new(m, c) for m, c in self.numer.terms()]
        terms_num_den = []
        for term in terms:
            _, num, den = term._gcd_monom(denom)
            terms_num_den.append((num, den))
        return terms_num_den

    def as_expr(self, fraction=True):
        """Convert a Laurent polynomial to a SymPy expression."""
        if fraction:
            return self.as_expr_fraction()
        else:
            return self.as_expr_add()

    def as_expr_add(self):
        """Convert a Laurent polynomial to a SymPy expression."""
        terms = []
        for num, den in self._terms_num_den():
            if den == self.ring.numer_ring.one:
                terms.append(num.as_expr())
            else:
                terms.append(num.as_expr() / den.as_expr())
        return Add(*terms)

    def as_expr_fraction(self):
        """Convert a Laurent polynomial to a SymPy expression."""
        return self.numer.as_expr() / self.denom.as_expr()

    def __str__(self):
        terms_str = []
        for num, den in self._terms_num_den():
            if den == self.ring.numer_ring.one:
                terms_str.append(str(num))
            else:
                terms_str.append(f'{num}/{den}')

        if not terms_str:
            return '0'

        return ' + '.join(terms_str)

    @property
    def is_term(self):
        return self.numer.is_term

    @property
    def is_zero(self):
        return not self

    @property
    def is_one(self):
        return self.numer.is_one and self.denom.is_one

    @property
    def is_ground(self):
        return self.numer.is_ground and self.denom.is_ground

    @property
    def LC(self):
        return self.numer.LC

    def __bool__(self):
        return bool(self.numer)

    def __pos__(self):
        return self

    def __neg__(self):
        return self.from_poly_denom(self.ring, -self.numer, self.denom)

    def __add__(self, other):
        if not isinstance(other, LaurentPolyElement):
            try:
                other = self.ring.ground_new(other)
            except CoercionFailed:
                return NotImplemented
        elif self.ring != other.ring:
            return NotImplemented
        return self._add(other)

    def __radd__(self, other):
        try:
            other = self.ring.ground_new(other)
        except CoercionFailed:
            return NotImplemented
        return self._add(other)

    def _add(self, other):
        numer = self.numer * other.denom + self.denom * other.numer
        denom = self.denom * other.denom
        return self.from_poly_denom(self.ring, numer, denom)

    def __sub__(self, other):
        if not isinstance(other, LaurentPolyElement):
            try:
                other = self.ring.ground_new(other)
            except CoercionFailed:
                return NotImplemented
        elif self.ring != other.ring:
            return NotImplemented
        return self._sub(other)

    def __rsub__(self, other):
        try:
            other = self.ring.ground_new(other)
        except CoercionFailed:
            return NotImplemented
        return other._sub(self)

    def _sub(self, other):
        numer = self.numer * other.denom - self.denom * other.numer
        denom = self.denom * other.denom
        return self.from_poly_denom(self.ring, numer, denom)

    def __mul__(self, other):
        if isinstance(other, PolyElement) and other.ring == self.ring.numer_ring:
            other = self.ring.from_polyelement(other)

        if not isinstance(other, LaurentPolyElement):
            try:
                other = self.ring.ground_new(other)
            except CoercionFailed:
                return NotImplemented
        elif self.ring != other.ring:
            return NotImplemented

        return self._mul(other)

    def __rmul__(self, other):
        if isinstance(other, PolyElement) and other.ring == self.ring.numer_ring:
            other = self.ring.from_polyelement(other)
            return other * self

        try:
            other = self.ring.ground_new(other)
        except CoercionFailed:
            return NotImplemented
        return self._mul(other)

    def _mul(self, other):
        numer = self.numer * other.numer
        denom = self.denom * other.denom
        return self.from_poly_denom(self.ring, numer, denom)

    def __pow__(self, n: int):
        if not isinstance(n, int):
            return NotImplemented

        if n == 0:
            return self.ring.one

        numer, denom = self.numer, self.denom

        if n < 0:
            if not self:
                raise ZeroDivisionError("Inversion of zero")
            elif not numer.is_term:
                raise NotImplementedError("Inversion of non-term")
            numer, denom = denom, numer
            n = -n

        return self.from_poly_denom(self.ring, numer ** n, denom ** n)

    def __rtruediv__(self, other):
        try:
            other = self.ring.ground_new(other)
        except CoercionFailed:
            return NotImplemented
        return other._div_term(self)

    def __truediv__(self, other):
        if not isinstance(other, LaurentPolyElement):
            try:
                other = self.ring.ground_new(other)
            except CoercionFailed:
                return NotImplemented
        elif self.ring != other.ring:
            return NotImplemented
        return self._div_term(other)

    def _div_term(self, other):
        if other.is_zero:
            raise ZeroDivisionError("Division by zero")
        if not other.is_term:
            raise NotImplementedError("Division by non-term")
        numer = self.numer * other.denom
        denom = self.denom * other.numer
        return self.from_poly_denom(self.ring, numer, denom)

    def exquo(self, other):
        quo_numer = self.numer.exquo(other.numer)
        numer = quo_numer * other.denom
        denom = self.denom
        return self.from_poly_denom(self.ring, numer, denom)

    def gcd(self, other):
        gcd_numer = self.numer.gcd(other.numer)
        gcd_denom = self.denom.gcd(other.denom)
        return self.from_poly_denom(self.ring, gcd_numer, gcd_denom)

    def _factor_list(self):
        """Compute a factorization of ``self``."""
        # XXX: Should negative multiplicity should be used for the denominator
        # rather than positive powers of a reciprocal?
        coeff, numer_factors = self.numer.factor_list()
        _, denom_factors = self.denom.factor_list()

        one = self.ring.numer_ring.one
        factors = []
        for factor, exp in numer_factors:
            factor_laurent = self.from_poly_denom(self.ring, factor, one)
            factors.append((factor_laurent, exp))
        for factor, exp in denom_factors:
            factor_laurent = self.from_poly_denom(self.ring, one, factor)
            factors.append((factor_laurent, exp))

        return coeff, factors

from sympy.polys.domains.ring import Ring
from sympy.polys.domains.compositedomain import CompositeDomain


class LaurentPolynomialRing(Ring, CompositeDomain):
    """Domain for Laurent polynomials.

    Examples
    ========

    >>> from sympy import QQ
    >>> from sympy.abc import x, y
    >>> R = QQ.laurent_poly_ring([x, y])
    >>> R
    QQ[x,1/x,y,1/y]
    >>> p = R.from_sympy(x + 1/y)
    >>> p
    x + 1/y
    """

    is_LaurentPolynomialRing = is_Laurent = True

    has_assoc_Ring = True
    has_assoc_Field = True

    def __init__(self, domain_or_ring, symbols=None, order=None):
        from sympy.polys.laurent import LaurentPolyRing, LaurentPolyElement

        if isinstance(domain_or_ring, LaurentPolyRing) and symbols is None and order is None:
            ring = domain_or_ring
        else:
            ring = LaurentPolyRing(symbols, domain_or_ring, order)

        self.ring = ring
        self.dtype = LaurentPolyElement

        self.domain = ring.domain
        self.symbols = ring.symbols
        self.gens = ring.gens
        self.one = ring.one
        self.zero = ring.zero
        self.order = ring.order

        # Needed in some places:
        self.dom = self.domain

    def __str__(self):
        syms = [str(s) for s in self.symbols]
        gens = [f'{s},1/{s}' for s in syms]
        return f'{self.domain}[{",".join(gens)}]'

    def __eq__(self, other):
        if not isinstance(other, LaurentPolynomialRing):
            return NotImplemented
        return self.ring == other.ring

    def __hash__(self):
        return hash(self.ring)

    def new(self, element):
        return self.ring.ring_new(element)

    def get_ring(self):
        return self.ring.numer_ring.to_domain()

    def get_field(self):
        return self.ring.to_field().to_domain()

    def numer(self, a):
        """Returns numerator of ``a`` as an element of the numerator ring. """
        return a.numer

    def denom(self, a):
        """Returns denominator of ``a`` as an element of the numerator ring. """
        return a.denom

    def is_positive(self, a):
        """Returns True if `LC(a)` is positive. """
        return self.domain.is_positive(a.LC)

    def is_negative(self, a):
        """Returns True if `LC(a)` is negative. """
        return self.domain.is_negative(a.LC)

    def is_nonpositive(self, a):
        """Returns True if `LC(a)` is non-positive. """
        return self.domain.is_nonpositive(a.LC)

    def is_nonnegative(self, a):
        """Returns True if `LC(a)` is non-negative. """
        return self.domain.is_nonnegative(a.LC)

    def exquo(self, a, b):
        return a.exquo(b)

    def gcd(self, a, b):
        return a.gcd(b)

    def from_sympy(self, expr):
        return self.ring.from_expr(expr)

    def to_sympy(self, poly):
        return poly.as_expr()

    def from_ZZ(K1, a, K0):
        """Convert a an integer to `dtype`. """
        return K1(K1.domain.convert(a, K0))

    def from_RealField(K1, a, K0):
        """Convert a mpmath `mpf` object to `dtype`. """
        return K1(K1.domain.convert(a, K0))

    def from_PolynomialRing(K1, a, K0):
        """Convert a polynomial to `dtype`. """
        numer_ring = K1.ring.numer_ring
        if K0.ring == numer_ring:
            return K1.ring.from_polyelement(a)

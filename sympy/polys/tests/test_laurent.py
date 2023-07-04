from sympy.polys.laurent import laurent_ring, LaurentPolyRing, LaurentPolyElement
from sympy.polys.domains import ZZ, QQ
from sympy.polys.rings import ring
from sympy.polys.fields import FracField
from sympy.polys.orderings import LexOrder
from sympy.abc import x, y

from sympy.testing.pytest import raises


# test init
def test_LaurentRing_init():
    Rp, _, _ = ring("x y", ZZ)
    R1, xr, yr = laurent_ring("x y", ZZ)
    R2 = LaurentPolyRing([x, y], ZZ, "lex")
    R3, _, _ = laurent_ring([x, y], ZZ)
    lex = LexOrder()

    assert R1 == R2 == R3

    for R in [R1, R2, R3]:
        assert R.numer_ring == Rp
        assert R.domain == ZZ
        assert R.gens == (xr, yr)
        assert R.symbols == (x, y)
        assert R.order == lex
        assert R.zero == LaurentPolyElement(R, Rp.zero, Rp.one)
        assert R.one == LaurentPolyElement(R, Rp.one, Rp.one)
        assert R.x == xr
        assert R.y == yr


def test_LaurentRing_eq_hash():

    R1, _, _ = laurent_ring("x y", ZZ)
    R2, _, _ = laurent_ring("x y", ZZ)
    R3, _, _ = laurent_ring("y x", ZZ)
    R4, _, _ = laurent_ring("x y", QQ)
    R5, _, _ = laurent_ring("x y", ZZ, order='grevlex')
    Rs = [R1, R2, R3, R4, R5, ZZ, QQ, ZZ[x,y]]

    for i, Ri in enumerate(Rs):
        for j, Rj in enumerate(Rs):
            if i == j or {i, j} == {0, 1}:
                assert hash(Ri) == hash(Rj)
                assert (Ri == Rj) is True
                assert (Ri != Rj) is False
            else:
                assert (Ri == Rj) is False
                assert (Ri != Rj) is True

    assert (R1 == QQ) is False
    assert (R1 != QQ) is True


# test str/repr
def test_LaurentRing_str():
    R, x, y = laurent_ring("x y", ZZ)
    assert str(R) == "Laurent polynomial ring in x, y over ZZ with lex order"


# test new
def test_LaurentRing_new():
    Rp, _, _ = ring("x y", ZZ)
    Rl, _, _ = laurent_ring("x y", ZZ)

    assert Rl.ground_new(1) == Rl.one
    assert Rl.ground_new(0) == Rl.zero
    assert Rl.ground_new(ZZ(1)) == Rl.one
    assert Rl.ground_new(ZZ(0)) == Rl.zero

    assert Rl.ring_new(1) == Rl.one
    assert Rl.ring_new(0) == Rl.zero
    assert Rl.ring_new(Rp(1)) == Rl.one
    assert Rl.ring_new(Rp(0)) == Rl.zero

    numer = 2*Rp.x + Rp.y
    p = Rl.ring_new(numer)
    assert p.numer == numer
    assert p.denom == Rp.one

    numer = ZZ(2)
    denom = Rp.x
    p = Rl.ring_new((numer, denom))
    p2 = Rl.new(Rp(numer), Rp(denom))
    assert p.numer == numer
    assert p.denom == denom
    assert p == p2

    p = Rl.from_expr((2*x + y)/x)
    assert p.numer == 2*Rp.x + Rp.y
    assert p.denom == Rp.x
    assert p == (2*Rl.x + Rl.y)/Rl.x

    raises(ValueError, Rl.from_expr, x**y)


def test_ring_field():
    R, _, _ = laurent_ring("x y", ZZ)
    assert R.to_domain() == ZZ.laurent_poly_ring(x, y)
    assert R.to_field() == FracField([x, y], ZZ)


# test laurentpolyelement
def test_LaurentPolyElement_init():
    pass

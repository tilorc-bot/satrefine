"""Ground results proved zero (:mod:`satrefine.identities.core.ground`).

``refine`` returns ``0`` for a constant SymPy leaves unevaluated when exact
identities (denesting ``sqrt(a + b*sqrt(c))``, rationalizing ``1/(a +
b*sqrt(c))``) make it evaluate to 0, and leaves every other result as it was:
a constant near 0 but not 0, a constant outside the identities, anything with
a symbol.
"""
from __future__ import annotations

import pytest
from sympy import I, Q, Rational, S, cbrt, exp, expand, pi, sqrt, symbols

from satrefine import refine
from satrefine.identities.core import ground

x = symbols("x")

_tiny = Rational(1, 10**600)

ZERO = [
    sqrt(3 + 2*sqrt(2)) - 1 - sqrt(2),
    sqrt(7 - 4*sqrt(3)) - (2 - sqrt(3)),
    sqrt(5 + 2*sqrt(6)) - sqrt(2) - sqrt(3),
    sqrt(2 + sqrt(3)) - (sqrt(6) + sqrt(2))/2,       # the denested roots are not rational
    3*sqrt(11 - 6*sqrt(2)) - 9 + 3*sqrt(2),          # a coefficient on the root
    1/(sqrt(2) - 1) - sqrt(2) - 1,
    1/sqrt(3 + 2*sqrt(2)) - sqrt(2) + 1,             # denested, then rationalized
    I*(sqrt(3 + 2*sqrt(2)) - 1 - sqrt(2)),
]

NOT_ZERO = [   # near 0, or 0 outside the identities: left as they are
    sqrt(3 + 2*sqrt(2)) - 1 - sqrt(2) + Rational(1, 10**300),
    3 + _tiny*sqrt(2) - sqrt(expand((3 - _tiny*sqrt(2))**2)),   # 2*sqrt(2)/10**600: minimal_polynomial says 0
    (1 + sqrt(2))**60 - expand((1 + sqrt(2))**60 + (1 - sqrt(2))**60),   # about -1e-23
    sqrt(10001) - 100 - Rational(1, 200),
    exp(pi*sqrt(163)) - 640320**3 - 744,            # about -7.5e-13
    sqrt(3 + 2*sqrt(2)) - sqrt(2),                   # 1
    sqrt(3 - 2*sqrt(3)) - sqrt(3) + 1,               # 3 < 2*sqrt(3): the root of a negative number
    cbrt(sqrt(5) + 2) - cbrt(sqrt(5) - 2) - 1,       # 0, but cube roots are not denested
    sqrt(2)*sqrt(3 + 2*sqrt(2)) - sqrt(2) - 2,       # 0, but needs the product expanded
]


@pytest.mark.parametrize("e", ZERO, ids=str)
def test_zero(e):
    assert refine(e, True) == 0
    assert refine(e, Q.positive(x)) == 0


@pytest.mark.parametrize("e", NOT_ZERO, ids=str)
def test_not_zero_unchanged(e):
    assert ground.zero_or_self(e) is e
    assert refine(e, True) == e


def test_free_symbols_unchanged():
    e = sqrt(3 + 2*sqrt(2)) - 1 - sqrt(2)
    assert refine(x + e, True) == x + e
    assert refine(x*e, Q.positive(x)) == x*e
    assert refine(e**x, True) == e**x


def test_only_a_proof():
    """A result is not rewritten unless it is 0."""
    for e in (sqrt(3 + 2*sqrt(2)), 1/(sqrt(2) - 1), 1 + sqrt(7 - 4*sqrt(3))):
        assert ground.zero_or_self(e) is e
    assert refine(sqrt(3 + 2*sqrt(2)), True) == sqrt(3 + 2*sqrt(2))


def test_bounded():
    big = S.Zero
    for k in range(2, 2 + ground.MAX_NODES):
        big += sqrt(k)
    zero = sqrt(3 + 2*sqrt(2)) - 1 - sqrt(2)
    assert ground.zero_or_self(zero) == 0
    assert ground.zero_or_self(big + zero) == big + zero

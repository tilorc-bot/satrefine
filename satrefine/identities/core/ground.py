"""Ground results that are exactly zero.

SymPy leaves a symbol-free constant such as ``sqrt(3 + 2*sqrt(2)) - 1 - sqrt(2)``
unevaluated although it is 0, so ``refine`` returned it unchanged.  The driver
passes every top-level result through :func:`zero_or_self`, which rewrites a
copy of a ground result with two identities over the rationals and returns
``0`` when SymPy's constructors then evaluate the copy to ``0``; otherwise the
result is returned as it was (the rewritten copy is not a refinement, only a
proof).  For rationals ``a``, ``b``, ``c`` with ``c > 0`` and ``b != 0``:

* **denesting**: ``sqrt(a + b*sqrt(c)) = sqrt((a + s)/2) + sign(b)*sqrt((a - s)/2)``
  when ``a > 0`` and ``a**2 - b**2*c = s**2`` for a rational ``s >= 0``
  (square the right side: ``a + sign(b)*sqrt(b**2*c)``; both roots are of
  nonnegative numbers, ``s < a``, and the sum is nonnegative, so it is the
  principal root);
* **rationalizing**: ``1/(a + b*sqrt(c)) = (a - b*sqrt(c))/(a**2 - b**2*c)``
  when the denominator is nonzero.

Every condition is decided in exact rational arithmetic (``isqrt`` for the
square), so no numeric closeness is involved and a constant near 0 is never
called 0: the rewritten copy is equal to the result, and SymPy's evaluation of
a sum of like terms to ``0`` is exact.  Only sums, products and powers without
free symbols of at most :data:`MAX_NODES` nodes are tried.

Not used: SymPy's own tests, which are not proofs and are outside the engine
(issue: refine depends on SymPy as little as possible).  ``minimal_polynomial``
chooses among the factors of its resultants by comparing their values
numerically, and gives ``x`` for ``3 + sqrt(2)/10**600 - sqrt((3 -
sqrt(2)/10**600)**2)``, which is ``2*sqrt(2)/10**600``; ``equals(0)`` calls it
zero at ``10**-1000``; ``is_zero``, and ``ask(Q.zero(...))`` through it, rest on
``minimal_polynomial`` when ``evalf`` finds no significant digit.  satassume
reads ``sqrt(2)`` as an indeterminate with no relation ``t**2 = 2``, so it
cannot prove such a constant zero either.
"""
from __future__ import annotations

from itertools import islice
from math import isqrt
from typing import Any

from sympy import Rational, S, preorder_traversal
from sympy.core import Add, Mul, Pow

MAX_NODES = 60
"""Largest result (nodes of its tree) tried."""


def zero_or_self(e: Any) -> Any:
    """``0`` if the ground result ``e`` is proved zero by the identities above, else ``e``."""
    if not isinstance(e, (Add, Mul, Pow)) or e.free_symbols:
        return e
    if sum(1 for _ in islice(preorder_traversal(e), MAX_NODES + 1)) > MAX_NODES:
        return e
    return S.Zero if _rewritten(e) == 0 else e


def _rewritten(e: Any) -> Any:
    """``e`` with every node rewritten bottom-up by :func:`_denest` and :func:`_rationalized`
    (``1/sqrt(a + b*sqrt(c))`` by both)."""
    if not isinstance(e, (Add, Mul, Pow)):
        return e
    args = [_rewritten(a) for a in e.args]
    if args != list(e.args):
        e = e.func(*args)
    if isinstance(e, Pow) and e.exp in (S.Half, -S.Half, S.NegativeOne):
        parts = _surd(e.base)
        if parts is None:
            return e
        if e.exp == -1:
            return _rationalized(*parts) or e
        root = _denest(*parts)
        if root is None:
            return e
        return root if e.exp == S.Half else _rewritten(1/root)
    return e


def _surd(e: Any) -> tuple | None:
    """``(a, b, c)`` for ``e = a + b*sqrt(c)`` with rationals ``a``, ``b != 0`` and ``c > 0``, else ``None``."""
    if not isinstance(e, Add) or len(e.args) != 2:
        return None
    a, t = e.args
    if not a.is_Rational:
        a, t = t, a
    b, r = t.as_coeff_Mul()
    if not (a.is_Rational and b.is_Rational and isinstance(r, Pow) and r.exp is S.Half
            and r.base.is_Rational and r.base > 0):
        return None
    return a, b, r.base


def _denest(a: Rational, b: Rational, c: Rational) -> Any:
    """``sqrt(a + b*sqrt(c))`` as a sum of two square roots of rationals, or ``None``."""
    d = a**2 - b**2*c
    if not (a > 0 and d >= 0):
        return None
    p, q = isqrt(d.p), isqrt(d.q)
    if p*p != d.p or q*q != d.q:
        return None
    s = Rational(p, q)
    return Pow((a + s)/2, S.Half) + (1 if b > 0 else -1)*Pow((a - s)/2, S.Half)


def _rationalized(a: Rational, b: Rational, c: Rational) -> Any:
    """``1/(a + b*sqrt(c))`` with a rational denominator, or ``None``."""
    d = a**2 - b**2*c
    if d == 0:
        return None
    return (a - b*Pow(c, S.Half))/d

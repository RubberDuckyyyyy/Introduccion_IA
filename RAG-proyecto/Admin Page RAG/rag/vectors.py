"""Dot product, norm, and cosine similarity — every sum is visible."""

from __future__ import annotations

import math

Vector = list[float]


def dot(a: Vector | tuple[float, ...], b: Vector | tuple[float, ...]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: Vector | tuple[float, ...]) -> float:
    return math.sqrt(dot(a, a))


def cosine(a: Vector | tuple[float, ...], b: Vector | tuple[float, ...]) -> float:
    na, nb = norm(a), norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)

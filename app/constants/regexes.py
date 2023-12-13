from __future__ import annotations

import re

__all__ = ("OSU_VERSION", "USERNAME", "EMAIL", "BEST_OF")


OSU_VERSION = re.compile(
    r"^b(?P<date>\d{8})(?:\.(?P<revision>\d))?"
    r"(?P<stream>beta|cuttingedge|dev|tourney)?$",
)

USERNAME = re.compile(r"^[\w \[\]-]{2,12}$")
EMAIL = re.compile(r"^[^@\s]{1,200}@[^@\s\.]{1,30}(?:\.[^@\.\s]{2,24})+$")
UUID = re.compile(r"[a-f0-9]{8}-?[a-f0-9]{4}-?4[a-f0-9]{3}-?[89ab][a-f0-9]{3}-?[a-f0-9]{12}", re.I)

TOURNEY_MATCHNAME = re.compile(
    r"^(?P<name>[a-zA-Z0-9_ ]+): "
    r"\((?P<T1>[a-zA-Z0-9_ ]+)\)"
    r" vs\.? "
    r"\((?P<T2>[a-zA-Z0-9_ ]+)\)$",
    flags=re.IGNORECASE,
)

MAPPOOL_PICK = re.compile(r"^([a-zA-Z]+)([0-9]+)$")

BEST_OF = re.compile(r"^(?:bo)?(\d{1,2})$")

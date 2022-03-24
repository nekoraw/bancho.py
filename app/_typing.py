from __future__ import annotations

from ipaddress import IPv4Address
from ipaddress import IPv6Address
from typing import Union

IPAddress = Union[IPv4Address, IPv6Address]

BeatmapMD5 = str
BeatmapID = int

PlayerID = int
PlayerName = str
PlayerToken = str

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Union

from app._typing import BeatmapID
from app._typing import BeatmapMD5
from app._typing import PlayerID
from app._typing import PlayerName
from app._typing import PlayerToken

if TYPE_CHECKING:
    from app.objects.beatmap import Beatmap, BeatmapSet
    from app.objects.player import Player


bcrypt: dict[bytes, bytes] = {}
beatmap: dict[Union[BeatmapMD5, BeatmapID], Beatmap] = {}
beatmapset: dict[int, BeatmapSet] = {}
unsubmitted: set[str] = set()
needs_update: set[str] = set()
player: dict[Union[PlayerID, PlayerName, PlayerToken], Player] = {}

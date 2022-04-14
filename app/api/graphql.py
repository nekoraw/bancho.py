from __future__ import annotations

from datetime import datetime
from typing import Optional

import strawberry
from graphql import GraphQLError
from strawberry.types.info import Info
from strawberry.types.nodes import Selection

import app.state
import app.usecases
import app.utils
from app.constants.gamemodes import GameMode


def field_requested(selections: list[Selection], wanted_field: str) -> bool:
    for selection in selections:
        for field in selection.selections:
            if field.name == wanted_field:
                return True

    return False


@strawberry.type
class PlayerCounts:
    total: int
    online: int


@strawberry.type
class ClanType:
    id: str
    name: str
    tag: str

    # TODO: owner and stuff


@strawberry.type
class PlayerInfo:
    id: int
    name: str
    safe_name: str
    priv: int
    clan: Optional[ClanType]
    country: str
    silence_end: int


@strawberry.type
class GradeCounts:
    xh: int
    x: int
    sh: int
    s: int
    a: int


@strawberry.type
class Stats:
    tscore: int
    rscore: int
    pp: int
    plays: int
    playtime: int
    acc: float
    max_combo: int
    grades: GradeCounts

    rank: int
    country_rank: int


@strawberry.type
class PlayerStats:
    std: Stats
    taiko: Stats
    catch: Stats
    mania: Stats

    std_rx: Stats
    taiko_rx: Stats
    catch_rx: Stats

    std_ap: Stats


@strawberry.type
class BeatmapType:
    md5: str
    id: int
    set_id: int
    artist: int
    title: str
    version: str
    creator: str
    last_update: datetime
    total_length: int
    max_combo: int
    status: int
    plays: int
    passes: int
    mode: int
    bpm: float
    cs: float
    od: float
    ar: float
    hp: float
    diff: float


@strawberry.type
class PlayerStatus:
    online: bool
    last_seen: Optional[int]

    login_time: Optional[int]
    action: Optional[int]
    info_text: Optional[str]
    mode: Optional[int]
    mods: Optional[int]
    beatmap: Optional[BeatmapType]


@strawberry.type
class PlayerType:
    info: PlayerInfo
    stats: PlayerStats
    status: PlayerStatus


async def fetch_user_stats(user_id: int, user_country: str) -> PlayerStats:
    rows = await app.state.services.database.fetch_all(
        "SELECT mode, tscore, rscore, pp, plays, playtime, acc, max_combo, "
        "xh_count, x_count, sh_count, s_count, a_count FROM stats "
        "WHERE id = :userid",
        {"userid": user_id},
    )

    stats = {}
    for idx, mode_stats in enumerate([dict(row) for row in rows]):
        mode = GameMode(mode_stats.pop("mode"))

        rank = await app.state.services.redis.zrevrank(
            f"bancho:leaderboard:{idx}",
            str(user_id),
        )
        mode_stats["rank"] = rank + 1 if rank is not None else 0

        country_rank = await app.state.services.redis.zrevrank(
            f"bancho:leaderboard:{idx}:{user_country}",
            str(user_id),
        )
        mode_stats["country_rank"] = country_rank + 1 if country_rank is not None else 0

        grades = {}
        for grade in ("xh", "x", "sh", "s", "a"):
            grade_count = mode_stats.pop(f"{grade}_count")
            grades[grade] = grade_count

        mode_stats["grades"] = GradeCounts(**grades)

        stats[mode.graphql_str] = Stats(**mode_stats)

    return PlayerStats(**stats)


async def fetch_user_status(user_id: int, latest_activity: int) -> PlayerStatus:
    status = {}
    player = app.state.sessions.players.get(id=user_id)

    if player:
        bmap = None
        if player.status.map_md5:
            bmap = await app.usecases.beatmap.from_md5(player.status.map_md5)

        status |= {
            "online": True,
            "last_seen": latest_activity,
            "login_time": player.login_time,
            "action": int(player.status.action),
            "info_text": player.status.info_text,
            "mode": int(player.status.mode),
            "mods": int(player.status.mods),
            "beatmap": bmap.as_dict if bmap else None,
        }
    else:
        status |= {
            "online": False,
            "last_seen": latest_activity,
            "login_time": None,
            "action": None,
            "info_text": None,
            "mode": None,
            "mods": None,
            "beatmap": None,
        }

    return PlayerStatus(**status)


@strawberry.type
class Query:
    @strawberry.field
    async def player_counts(self, info: Info) -> PlayerCounts:
        selections = info.selected_fields

        total = 0
        if field_requested(selections, "total"):
            total = await app.state.services.database.fetch_val(
                "SELECT COUNT(*) FROM users",
                column=0,
            )

        return PlayerCounts(
            total=total,
            online=len(app.state.sessions.players.unrestricted) - 1,
        )

    @strawberry.field
    async def player(
        self,
        info: Info,
        id: Optional[int] = None,
        name: Optional[str] = None,
    ) -> Optional[PlayerType]:
        if not any((id, name)):
            raise GraphQLError("You must provide a username or user id!")

        response = {
            "info": None,
            "stats": None,
            "status": None,
        }

        if name:
            info = await app.state.services.database.fetch_one(
                "SELECT id, name, safe_name, latest_activity, "
                "priv, clan_id, country, silence_end "
                "FROM users WHERE safe_name = :safe_name",
                {"safe_name": app.utils.make_safe_name(name)},
            )
        else:
            info = await app.state.services.database.fetch_one(
                "SELECT id, name, safe_name, latest_activity, "
                "priv, clan_id, country, silence_end "
                "FROM users WHERE safe_name = :id",
                {"id": id},
            )

        if info is None:
            raise GraphQLError("Unknown user!")

        info = dict(info)
        user_id: int = info["id"]
        user_country: str = info["country"]
        latest_activity: int = info.pop("latest_activity")

        selections = info.selected_fields

        if field_requested(selections, "info"):
            clan_id = info.pop("clan_id")

            info["clan"] = app.state.sessions.clans.get(id=clan_id) if clan_id else None
            response["info"] = PlayerInfo(**info)

        if field_requested(selections, "stats"):
            response["stats"] = await fetch_user_stats(user_id, user_country)

        if field_requested(selections, "status"):
            response["status"] = await fetch_user_status(user_id, latest_activity)

        return PlayerType(**response)

from __future__ import annotations

from typing import Optional

import strawberry
from graphql import GraphQLError
from strawberry.types.info import Info
from strawberry.types.nodes import Selection

import app.state
import app.utils
from app.constants.gamemodes import GameMode


def field_requested(selections: list[Selection], wanted_field: str) -> bool:
    for (
        selection
    ) in (
        selections
    ):  # XX: this is probably bad, we could have multiple fields with the same name in different selections
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
class PlayerType:
    info: PlayerInfo
    stats: PlayerStats


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

        response = {}
        response["stats"] = None

        selections = info.selected_fields

        if name:
            info = await app.state.services.database.fetch_one(
                "SELECT id, name, safe_name, "
                "priv, clan_id, country, silence_end "
                "FROM users WHERE safe_name = :safe_name",
                {"safe_name": app.utils.make_safe_name(name)},
            )
        else:
            info = await app.state.services.database.fetch_one(
                "SELECT id, name, safe_name, "
                "priv, clan_id, country, silence_end "
                "FROM users WHERE safe_name = :id",
                {"id": id},
            )

        if info is None:
            raise GraphQLError("Unknown user!")

        user_id: int = info["id"]
        user_country: str = info["country"]

        if field_requested(selections, "info"):
            info = dict(info)
            clan_id = info.pop("clan_id")

            info["clan"] = app.state.sessions.clans.get(id=clan_id) if clan_id else None
            response["info"] = PlayerInfo(**info)

        if field_requested(selections, "stats"):
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
                mode_stats["country_rank"] = (
                    country_rank + 1 if country_rank is not None else 0
                )

                grades = {}
                for grade in ("xh", "x", "sh", "s", "a"):
                    grade_count = mode_stats.pop(f"{grade}_count")
                    grades[grade] = grade_count

                mode_stats["grades"] = GradeCounts(**grades)

                stats[mode.graphql_str] = Stats(**mode_stats)

            response["stats"] = PlayerStats(**stats)

        return PlayerType(**response)

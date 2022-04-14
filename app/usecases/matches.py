from __future__ import annotations

from collections import defaultdict
from typing import Sequence
from typing import TYPE_CHECKING
from typing import Union

import databases.core

from app.constants import regexes
from app.constants.mods import Mods
from app.logging import Ansi
from app.logging import log
from app.objects.beatmap import Beatmap
from app.objects.match import MapPool
from app.objects.match import Match
from app.objects.match import MatchTeams
from app.objects.match import MatchTeamTypes
from app.objects.match import MatchWinConditions
from app.objects.match import Slot

if TYPE_CHECKING:
    from app.objects.player import Player

from datetime import datetime, timedelta

import app.usecases
import asyncio


async def maps_from_sql(pool: MapPool, db_conn: databases.core.Connection) -> None:
    """Retrieve all maps from sql to populate `pool.maps`."""
    for row in await db_conn.fetch_all(
        "SELECT map_id, mods, slot FROM tourney_pool_maps WHERE pool_id = :pool_id",
        {"pool_id": pool.id},
    ):
        map_id = row["map_id"]
        bmap = await app.usecases.beatmap.from_bid(map_id)

        if not bmap:
            # map not found? remove it from the
            # pool and log this incident to console.
            # NOTE: it's intentional that this removes
            # it from not only this pool, but all pools.
            # TODO: perhaps discord webhook?
            log(f"Removing {map_id} from pool {pool.name} (not found).", Ansi.LRED)

            await db_conn.execute(
                "DELETE FROM tourney_pool_maps WHERE map_id = :map_id",
                {"map_id": map_id},
            )
            continue

        key: tuple[Mods, int] = (Mods(row["mods"]), row["slot"])
        pool.maps[key] = bmap


async def await_submissions(
    match: Match,
    was_playing: Sequence[Slot],
) -> tuple[dict[Union[MatchTeams, Player], int], Sequence[Player]]:
    """Await score submissions from all players in completed state."""
    scores: dict[Union[MatchTeams, Player], int] = defaultdict(int)
    didnt_submit: list[Player] = []
    time_waited = 0  # allow up to 10s (total, not per player)

    ffa = match.team_type in (MatchTeamTypes.head_to_head, MatchTeamTypes.tag_coop)

    if match.use_pp_scoring:
        win_cond = "pp"
    else:
        win_cond = ("score", "acc", "max_combo", "score")[match.win_condition]

    bmap = await app.usecases.beatmap.from_md5(match.map_md5)

    if not bmap:
        # map isn't submitted
        return {}, ()

    for s in was_playing:
        # continue trying to fetch each player's
        # scores until they've all been submitted.
        while True:
            rc_score = s.player.recent_score
            max_age = datetime.now() - timedelta(
                seconds=bmap.total_length + time_waited + 0.5,
            )

            if (
                rc_score
                and rc_score.bmap.md5 == match.map_md5
                and rc_score.server_time > max_age
            ):
                # score found, add to our scores dict if != 0.
                if score := getattr(rc_score, win_cond):
                    key = s.player if ffa else s.team
                    scores[key] += score

                break

            # wait 0.5s and try again
            await asyncio.sleep(0.5)
            time_waited += 0.5

            if time_waited > 10:
                # inform the match this user didn't
                # submit a score in time, and skip them.
                didnt_submit.append(s.player)
                break

    # all scores retrieved, update the match.
    return scores, didnt_submit


async def update_matchpoints(match: Match, was_playing: Sequence[Slot]) -> None:
    """\
    Determine the winner from `scores`, increment & inform players.

    This automatically works with the match settings (such as
    win condition, teams, & co-op) to determine the appropriate
    winner, and will use any team names included in the match name,
    along with the match name (fmt: OWC2020: (Team1) vs. (Team2)).

    For the examples, we'll use accuracy as a win condition.

    Teams, match title: `OWC2015: (United States) vs. (China)`.
        United States takes the point! (293.32% vs 292.12%)
        Total Score: United States | 7 - 2 | China
        United States takes the match, finishing OWC2015 with a score of 7 - 2!

    FFA, the top <=3 players will be listed for the total score.
        Justice takes the point! (94.32% [Match avg. 91.22%])
        Total Score: Justice - 3 | cmyui - 2 | FrostiDrinks - 2
        Justice takes the match, finishing with a score of 4 - 2!
    """

    scores, didnt_submit = await match.await_submissions(was_playing)

    for p in didnt_submit:
        match.chat.send_bot(f"{p} didn't submit a score (timeout: 10s).")

    if scores:
        ffa = match.team_type in (
            MatchTeamTypes.head_to_head,
            MatchTeamTypes.tag_coop,
        )

        # all scores are equal, it was a tie.
        if len(scores) != 1 and len(set(scores.values())) == 1:
            match.winners.append(None)
            match.chat.send_bot("The point has ended in a tie!")
            return None

        # Find the winner & increment their matchpoints.
        winner: Union[Player, MatchTeams] = max(scores, key=lambda k: scores[k])
        match.winners.append(winner)
        match.match_points[winner] += 1

        msg: list[str] = []

        def add_suffix(score: int | float) -> str | int | float:
            if match.use_pp_scoring:
                return f"{score:.2f}pp"
            elif match.win_condition == MatchWinConditions.accuracy:
                return f"{score:.2f}%"
            elif match.win_condition == MatchWinConditions.combo:
                return f"{score}x"
            else:
                return str(score)

        if ffa:
            msg.append(
                f"{winner.name} takes the point! ({add_suffix(scores[winner])} "
                f"[Match avg. {add_suffix(int(sum(scores.values()) / len(scores)))}])",
            )

            wmp = match.match_points[winner]

            # check if match point #1 has enough points to win.
            if match.winning_pts and wmp == match.winning_pts:
                # we have a champion, announce & reset our match.
                match.is_scrimming = False
                match.reset_scrim()
                match.bans.clear()

                m = f"{winner.name} takes the match! Congratulations!"
            else:
                # no winner, just announce the match points so far.
                # for ffa, we'll only announce the top <=3 players.
                m_points = sorted(match.match_points.items(), key=lambda x: x[1])
                m = f"Total Score: {' | '.join([f'{k.name} - {v}' for k, v in m_points])}"

            msg.append(m)
            del m

        else:  # teams
            if r_match := regexes.TOURNEY_MATCHNAME.match(match.name):
                match_name = r_match["name"]
                team_names = {
                    MatchTeams.blue: r_match["T1"],
                    MatchTeams.red: r_match["T2"],
                }
            else:
                match_name = match.name
                team_names = {MatchTeams.blue: "Blue", MatchTeams.red: "Red"}

            # teams are binary, so we have a loser.
            loser = MatchTeams({1: 2, 2: 1}[winner])

            # from match name if available, else blue/red.
            wname = team_names[winner]
            lname = team_names[loser]

            # scores from the recent play
            # (according to win condition)
            ws = add_suffix(scores[winner])
            ls = add_suffix(scores[loser])

            # total win/loss score in the match.
            wmp = match.match_points[winner]
            lmp = match.match_points[loser]

            # announce the score for the most recent play.
            msg.append(f"{wname} takes the point! ({ws} vs. {ls})")

            # check if the winner has enough match points to win the match.
            if match.winning_pts and wmp == match.winning_pts:
                # we have a champion, announce & reset our match.
                match.is_scrimming = False
                match.reset_scrim()

                msg.append(
                    f"{wname} takes the match, finishing {match_name} "
                    f"with a score of {wmp} - {lmp}! Congratulations!",
                )
            else:
                # no winner, just announce the match points so far.
                msg.append(f"Total Score: {wname} | {wmp} - {lmp} | {lname}")

        if didnt_submit:
            match.chat.send_bot(
                "If you'd like to perform a rematch, "
                "please use the `!mp rematch` command.",
            )

        for line in msg:
            match.chat.send_bot(line)

    else:
        match.chat.send_bot("Scores could not be calculated.")

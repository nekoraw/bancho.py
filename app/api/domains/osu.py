""" osu: handle connections from web, api, and beyond? """
from __future__ import annotations

import copy
import hashlib
import random
from enum import IntEnum
from enum import unique
from functools import cache
from pathlib import Path as SystemPath
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Literal
from typing import Mapping
from typing import Optional
from typing import TypeVar
from typing import Union
from urllib.parse import unquote
from urllib.parse import unquote_plus

import databases.core
from fastapi import status
from fastapi.datastructures import UploadFile
from fastapi.exceptions import HTTPException
from fastapi.param_functions import Depends
from fastapi.param_functions import File
from fastapi.param_functions import Form
from fastapi.param_functions import Header
from fastapi.param_functions import Path
from fastapi.param_functions import Query
from fastapi.requests import Request
from fastapi.responses import FileResponse
from fastapi.responses import ORJSONResponse
from fastapi.responses import RedirectResponse
from fastapi.responses import Response
from fastapi.routing import APIRouter

import app.packets
import app.settings
import app.state.cache
import app.state.services
import app.state.sessions
import app.utils
from app import repositories
from app import responses
from app import usecases
from app import validation
from app.constants.clientflags import LastFMFlags
from app.constants.gamemodes import GameMode
from app.constants.mods import Mods
from app.logging import Ansi
from app.logging import log
from app.logging import printc
from app.objects import models
from app.objects.beatmap import RankedStatus
from app.objects.player import Player
from app.objects.player import Privileges
from app.objects.score import Grade
from app.objects.score import Score
from app.objects.score import SubmissionStatus
from app.state.services import acquire_db_conn
from app.utils import escape_enum
from app.utils import pymysql_encode

AVATARS_PATH = SystemPath.cwd() / ".data/avatars"
BEATMAPS_PATH = SystemPath.cwd() / ".data/osu"
REPLAYS_PATH = SystemPath.cwd() / ".data/osr"
SCREENSHOTS_PATH = SystemPath.cwd() / ".data/ss"


router = APIRouter(
    tags=["osu! web API"],
    default_response_class=Response,
)


@cache
def authenticate_player_session(
    param_function: Callable[..., Any],
    username_alias: str = "u",
    pw_md5_alias: str = "p",
    err: Optional[Any] = None,
) -> Callable[[str, str], Awaitable[Player]]:
    async def wrapper(
        username: str = param_function(..., alias=username_alias),
        pw_md5: str = param_function(..., alias=pw_md5_alias),
    ) -> Player:
        player = await repositories.players.fetch(name=unquote(username))

        if player and usecases.players.validate_credentials(
            password=pw_md5.encode(),
            hashed_password=player.pw_bcrypt,  # type: ignore
        ):
            return player

        # player login incorrect
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=err,  # TODO: make sure this works
        )

    return wrapper


""" /web/ handlers """

# TODO
# POST /web/osu-session.php
# POST /web/osu-osz2-bmsubmit-post.php
# POST /web/osu-osz2-bmsubmit-upload.php
# GET /web/osu-osz2-bmsubmit-getid.php
# GET /web/osu-get-beatmap-topic.php


@router.post("/web/osu-error.php")
async def osuError(
    username: Optional[str] = Form(None, alias="u"),
    pw_md5: Optional[str] = Form(None, alias="h"),
    user_id: int = Form(..., alias="i", ge=3, le=2_147_483_647),
    osu_mode: str = Form(..., alias="osumode"),
    game_mode: str = Form(..., alias="gamemode"),
    game_time: int = Form(..., alias="gametime", ge=0),
    audio_time: int = Form(..., alias="audiotime"),
    culture: str = Form(...),
    map_id: int = Form(..., alias="beatmap_id", ge=0, le=2_147_483_647),
    map_md5: str = Form(..., alias="beatmap_checksum", min_length=32, max_length=32),
    exception: str = Form(...),
    feedback: str = Form(...),
    stacktrace: str = Form(...),
    soft: bool = Form(...),
    map_count: int = Form(..., alias="beatmap_count", ge=0),
    compatibility: bool = Form(...),
    ram: int = Form(...),
    osu_ver: str = Form(..., alias="version"),
    exe_hash: str = Form(..., alias="exehash"),
    config: str = Form(...),
    screenshot_file: Optional[UploadFile] = File(None, alias="ss"),
):
    """Handle an error submitted from the osu! client."""
    if not app.settings.DEBUG:
        # only handle osu-error in debug mode
        return

    if username and pw_md5:
        player = await repositories.players.fetch(name=unquote(username))

        if player is None or not usecases.players.validate_credentials(
            password=pw_md5.encode(),
            hashed_password=player.pw_bcrypt,  # type: ignore
        ):
            # player login incorrect
            await app.state.services.log_strange_occurrence("osu-error auth failed")
            player = None
    else:
        player = None

    err_desc = f"{feedback} ({exception})"
    log(f'{player or "Offline user"} sent osu-error: {err_desc}', Ansi.LCYAN)

    # NOTE: this stacktrace can be a LOT of data
    if app.settings.DEBUG and len(stacktrace) < 2000:
        printc(stacktrace[:-2], Ansi.LMAGENTA)

    # TODO: save error in db?


@router.post("/web/osu-screenshot.php")
async def osuScreenshot(
    player: Player = Depends(authenticate_player_session(Form, "u", "p")),
    endpoint_version: int = Form(..., alias="v"),
    screenshot_file: UploadFile = File(..., alias="ss"),  # TODO: why can't i use bytes?
):
    if endpoint_version != 1:
        await app.state.services.log_strange_occurrence(
            f"Incorrect endpoint version (/web/osu-screenshot.php v{endpoint_version})",
        )

    resp, data = await usecases.screenshots.create(screenshot_file)

    if not resp:
        return Response(
            content=data,
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    log(f"{player} uploaded {data}.")
    return data


@router.get("/web/osu-getfriends.php")
async def osuGetFriends(
    player: Player = Depends(authenticate_player_session(Query, "u", "h")),
):
    return "\n".join(map(str, player.friends)).encode()


@router.post("/web/osu-getbeatmapinfo.php")
async def get_beatmap_info(
    form_data: models.OsuBeatmapRequestForm,
    player: Player = Depends(authenticate_player_session(Query, "u", "h")),
    db_conn: databases.core.Connection = Depends(acquire_db_conn),
):
    return await usecases.beatmaps.get_beatmap_info(player, form_data)


def format_favourites(favourite_beatmap_set_ids: list[int]) -> bytes:
    return "\n".join([str(set_id) for set_id in favourite_beatmap_set_ids]).encode()


@router.get("/web/osu-getfavourites.php")
async def get_favourite_beatmap_sets(
    player: Player = Depends(authenticate_player_session(Query, "u", "h")),
):
    resp = await usecases.players.get_favourite_beatmap_sets(player)
    return format_favourites(resp)


@router.get("/web/osu-addfavourite.php")
async def add_favourite_beatmap(
    player: Player = Depends(authenticate_player_session(Query, "u", "h")),
    map_set_id: int = Query(..., alias="a"),
):
    return await usecases.players.add_favourite(player, map_set_id)


@router.get("/web/lastfm.php")
async def lastfm_handler(
    action: Literal["scrobble", "np"],
    beatmap_id_or_hidden_flag: str = Query(..., alias="b"),
    player: Player = Depends(authenticate_player_session(Query, "us", "ha")),
):
    if beatmap_id_or_hidden_flag[0] != "a":
        # not anticheat related, tell the
        # client not to send any more for now.
        return b"-3"

    flags = LastFMFlags(int(beatmap_id_or_hidden_flag[1:]))

    if flags & (LastFMFlags.HQ_ASSEMBLY | LastFMFlags.HQ_FILE):
        # Player is currently running hq!osu; could possibly
        # be a separate client, buuuut prooobably not lol.

        await usecases.players.restrict(
            player=player,
            admin=app.state.sessions.bot,
            reason=f"hq!osu running ({flags})",
        )

        if player.online:  # refresh their client state
            usecases.players.logout(player)

        return b"-3"

    if flags & LastFMFlags.REGISTRY_EDITS:
        # Player has registry edits left from
        # hq!osu's multiaccounting tool. This
        # does not necessarily mean they are
        # using it now, but they have in the past.

        if random.randrange(32) == 0:
            # Random chance (1/32) for a ban.
            await usecases.players.restrict(
                player=player,
                admin=app.state.sessions.bot,
                reason="hq!osu relife 1/32",
            )

            if player.online:  # refresh their client state
                usecases.players.logout(player)

            return b"-3"

        # TODO: make a tool to remove the flags & send this as a dm.
        #       also add to db so they never are restricted on first one.
        player.enqueue(
            app.packets.notification(
                "\n".join(
                    [
                        "Hey!",
                        "It appears you have hq!osu's multiaccounting tool (relife) enabled.",
                        "This tool leaves a change in your registry that the osu! client can detect.",
                        "Please re-install relife and disable the program to avoid any restrictions.",
                    ],
                ),
            ),
        )

        usecases.players.logout(player)

        return b"-3"

    """ These checks only worked for ~5 hours from release. rumoi's quick!
    if flags & (ClientFlags.libeay32Library | ClientFlags.aqnMenuSample):
        # AQN has been detected in the client, either
        # through the 'libeay32.dll' library being found
        # onboard, or from the menu sound being played in
        # the AQN menu while being in an inappropriate menu
        # for the context of the sound effect.
        pass
    """


@router.get("/web/osu-search.php")
async def beatmap_search(
    player: Player = Depends(authenticate_player_session(Query, "u", "h")),
    query: str = Query(..., alias="q"),
    mode: int = Query(..., alias="m", ge=-1, le=3),  # -1 for all
    ranked_status: int = Query(..., alias="r", ge=0, le=8),
    page_num: int = Query(..., alias="p"),
):
    return await usecases.direct.search(
        query,
        mode,
        ranked_status,
        page_num,
    )


@router.get("/web/osu-search-set.php")
async def get_beatmap_set_information(
    player: Player = Depends(authenticate_player_session(Query, "u", "h")),
    map_set_id: Optional[int] = Query(None, alias="s"),
    map_id: Optional[int] = Query(None, alias="b"),
):
    return await usecases.direct.search_set(
        map_id,
        map_set_id,
    )


T = TypeVar("T", bound=Union[int, float])


def chart_entry(name: str, before: Optional[T], after: T) -> str:
    return f"{name}Before:{before or ''}|{name}After:{after}"


@router.post("/web/osu-submit-modular-selector.php")
async def submit_score(
    request: Request,
    # TODO: should token be allowed
    # through but ac'd if not found?
    # TODO: validate token format
    # TODO: save token in the database
    token: str = Header(...),
    # TODO: do ft & st contain pauses?
    exited_out: bool = Form(..., alias="x"),
    fail_time: int = Form(..., alias="ft"),
    visual_settings_b64: bytes = Form(..., alias="fs"),
    updated_beatmap_hash: str = Form(..., alias="bmk"),
    storyboard_md5: Optional[str] = Form(None, alias="sbk"),
    iv_b64: bytes = Form(..., alias="iv"),
    unique_ids: str = Form(..., alias="c1"),  # TODO: more validaton
    score_time: int = Form(..., alias="st"),  # TODO: is this real name?
    pw_md5: str = Form(..., alias="pass"),
    osu_version: str = Form(..., alias="osuver"),  # TODO: regex
    client_hash_b64: bytes = Form(..., alias="s"),
    # TODO: do these need to be Optional?
    # TODO: validate this is actually what it is
    fl_cheat_screenshot: Optional[bytes] = File(None, alias="i"),
    db_conn: databases.core.Connection = Depends(acquire_db_conn),
):
    """Handle a score submission from an osu! client with an active session."""

    form_data = await request.form()

    # XXX:HACK the bancho protocol uses the "score" parameter name for
    # both the base64'ed score data, as well as the replay file in the multipart.
    # starlette/fastapi do not support this - this function provides a workaround
    score_parameters = usecases.scores.parse_form_data_score_params(form_data)
    if score_parameters is None:
        # failed to parse score data
        return

    # extract the score data and replay file from the score data
    score_data_b64, replay_file = score_parameters

    # decrypt the score data and client hash using the iv and osu version
    score_data, client_hash_decoded = usecases.scores.decrypt_score_aes_data(
        # to decrypt
        score_data_b64,
        client_hash_b64,
        # decryption keys
        iv_b64,
        osu_version,
    )

    # fetch the beatmap played in the score
    beatmap = await repositories.beatmaps.fetch_by_md5(score_data[0])

    if beatmap is None:
        # map does not exist, most likely unsubmitted.
        return b"error: beatmap"

    username = score_data[1].rstrip()  # rstrip 1 space if client has supporter

    player = await repositories.players.fetch(name=unquote(username))

    if player is None or usecases.players.validate_credentials(
        password=pw_md5.encode(),
        hashed_password=player.pw_bcrypt,  # type: ignore
    ):
        # player is not online, return nothing so that their
        # client will retry submission when they log in.
        return

    n300 = int(score_data[3])
    n100 = int(score_data[4])
    n50 = int(score_data[5])
    ngeki = int(score_data[6])
    nkatu = int(score_data[7])
    nmiss = int(score_data[8])
    vanilla_mode = int(score_data[15])

    passed = score_data[14] == "True"
    time_elapsed = score_time if passed else fail_time

    # all data read from submission.
    # now we can calculate things based on our data.
    accuracy = usecases.scores.calculate_accuracy(
        vanilla_mode,
        n300,
        n100,
        n50,
        ngeki,
        nkatu,
        nmiss,
    )

    # parse the score from the remaining data
    score = Score.from_submission(score_data, accuracy, time_elapsed)

    ## perform checksum validation
    try:
        usecases.scores.validate_score_submission_data(
            score,
            unique_ids,
            osu_version,
            client_hash_decoded,
            updated_beatmap_hash,
            storyboard_md5,
            player.client_details,  # type: ignore
        )
    except AssertionError:
        # NOTE: this is undergoing a temporary trial period,
        # after which, it will be enabled & perform restrictions.
        stacktrace = app.utils.get_appropriate_stacktrace()
        await app.state.services.log_strange_occurrence(stacktrace)

        # await usecases.players.restrict(
        #     player=player,
        #     admin=app.state.sessions.bot,
        #     reason="TODO",
        # )

        # if player.online: # refresh their client state
        #     usecases.players.logout(player)
        # return b"error: ban"
    except:
        raise

    if beatmap is not None:
        osu_file_path = BEATMAPS_PATH / f"{beatmap.id}.osu"
        if await usecases.beatmaps.ensure_local_osu_file(
            osu_file_path,
            beatmap.id,
            beatmap.md5,
        ):
            score.pp, score.sr = usecases.scores.calculate_performance(
                score,
                osu_file_path,
            )

            if score.passed:
                await usecases.scores.calculate_status(score, beatmap, player)

                if beatmap.status != RankedStatus.Pending:
                    score.rank = await usecases.scores.calculate_placement(
                        score,
                        beatmap,
                    )
            else:
                score.status = SubmissionStatus.FAILED
    else:
        score.pp = score.sr = 0.0
        if score.passed:
            score.status = SubmissionStatus.SUBMITTED
        else:
            score.status = SubmissionStatus.FAILED

    # we should update their activity no matter
    # what the result of the score submission is.
    usecases.players.update_latest_activity_soon(player)

    # attempt to update their stats if their
    # gm/gm-affecting-mods change at all.
    if score.mode != player.status.mode:
        player.status.mods = score.mods
        player.status.mode = score.mode

        if not player.restricted:
            app.state.sessions.players.enqueue(app.packets.user_stats(player))

    # Check for score duplicates
    if await db_conn.fetch_one(
        "SELECT 1 FROM scores WHERE online_checksum = :checksum",
        {"checksum": score.client_checksum},
    ):
        log(f"{player} submitted a duplicate score.", Ansi.LYELLOW)
        return b"error: no"

    if fl_cheat_screenshot:
        stacktrace = app.utils.get_appropriate_stacktrace()
        await app.state.services.log_strange_occurrence(stacktrace)

    if (  # check for pp caps on ranked & approved maps for appropriate players.
        beatmap.awards_ranked_pp
        and not (player.priv & Privileges.WHITELISTED or player.restricted)
    ):
        # Get the PP cap for the current context.
        """# TODO: find where to put autoban pp
        pp_cap = app.app.settings.AUTOBAN_PP[score.mode][score.mods & Mods.FLASHLIGHT != 0]

        if score.pp > pp_cap:
            await usecases.players.restrict(
                player=player,
                admin=app.state.sessions.bot,
                reason=f"[{score.mode!r} {score.mods!r}] autoban @ {score.pp:.2f}pp",
            )

            if player.online: # refresh their client state
                usecases.players.logout(player)
        """

    """ Score submission checks completed; submit the score. """

    if app.state.services.datadog is not None:
        app.state.services.datadog.increment("bancho.submitted_scores")

    if score.status == SubmissionStatus.BEST:
        if app.state.services.datadog is not None:
            app.state.services.datadog.increment("bancho.submitted_scores_best")

        if beatmap.has_leaderboard:
            if score.mode < GameMode.RELAX_OSU and beatmap.status == RankedStatus.Loved:
                # use score for vanilla loved only
                performance = f"{score.score:,} score"
            else:
                performance = f"{score.pp:,.2f}pp"

            player.enqueue(
                app.packets.notification(
                    f"You achieved #{score.rank}! ({performance})",
                ),
            )

            if score.rank == 1 and not player.restricted:
                # this is the new #1, post the play to #announce.
                announce_channel = await repositories.channels.fetch_by_name(
                    "#announce",
                )

                if announce_channel is not None:
                    # Announce the user's #1 score.
                    # TODO: truncate artist/title/version to fit on screen
                    ann = [
                        f"\x01ACTION achieved #1 on {beatmap.embed}",
                        f"with {score.acc:.2f}% for {performance}.",
                    ]

                    if score.mods:
                        ann.insert(1, f"+{score.mods!r}")

                    scoring_metric = (
                        "pp" if score.mode >= GameMode.RELAX_OSU else "score"
                    )

                    # If there was previously a score on the map, add old #1.
                    prev_n1 = await db_conn.fetch_one(
                        "SELECT u.id, name FROM users u "
                        "INNER JOIN scores s ON u.id = s.userid "
                        "WHERE s.map_md5 = :map_md5 AND s.mode = :mode "
                        "AND s.status = 2 AND u.priv & 1 "
                        f"ORDER BY s.{scoring_metric} DESC LIMIT 1",
                        {"map_md5": beatmap.md5, "mode": score.mode},
                    )

                    if prev_n1:
                        if player.id != prev_n1["id"]:
                            ann.append(
                                f"(Previous #1: [https://{app.settings.DOMAIN}/u/"
                                "{id} {name}])".format(**prev_n1),
                            )

                    usecases.channels.send_msg_to_clients(
                        announce_channel,
                        msg=" ".join(ann),
                        sender=player,
                        to_self=True,
                    )

        # this score is our best score.
        # update any preexisting personal best
        # records with SubmissionStatus.SUBMITTED.
        await db_conn.execute(
            "UPDATE scores SET status = 1 "
            "WHERE status = 2 AND map_md5 = :map_md5 "
            "AND userid = :user_id AND mode = :mode",
            {
                "map_md5": beatmap.md5,
                "user_id": player.id,
                "mode": score.mode,
            },
        )

    score.id = await db_conn.execute(
        "INSERT INTO scores "
        "VALUES (NULL, "
        ":map_md5, :score, :pp, :acc, "
        ":max_combo, :mods, :n300, :n100, "
        ":n50, :nmiss, :ngeki, :nkatu, "
        ":grade, :status, :mode, :play_time, "
        ":time_elapsed, :client_flags, :user_id, :perfect, "
        ":checksum)",
        {
            "map_md5": beatmap.md5,
            "score": score.score,
            "pp": score.pp,
            "acc": score.acc,
            "max_combo": score.max_combo,
            "mods": score.mods,
            "n300": score.n300,
            "n100": score.n100,
            "n50": score.n50,
            "nmiss": score.nmiss,
            "ngeki": score.ngeki,
            "nkatu": score.nkatu,
            "grade": score.grade.name,
            "status": score.status,
            "mode": score.mode,
            "play_time": score.server_time,
            "time_elapsed": score.time_elapsed,
            "client_flags": score.client_flags,
            "user_id": player.id,
            "perfect": score.perfect,
            "checksum": score.client_checksum,
        },
    )

    if score.passed:
        replay_data = await replay_file.read()

        # All submitted plays should have a replay.
        # If not, they may be using a score submitter.
        if len(replay_data) < 24 and not player.restricted:
            log(f"{player} submitted a score without a replay!", Ansi.LRED)
            await usecases.players.restrict(
                player=player,
                admin=app.state.sessions.bot,
                reason="submitted score with no replay",
            )

            if player.online:  # refresh their client state
                usecases.players.logout(player)
        else:
            # TODO: the replay is currently sent from the osu!
            # client compressed with LZMA; this compression can
            # be improved pretty decently by serializing it
            # manually, so we'll probably do that in the future.
            replay_file = REPLAYS_PATH / f"{score.id}.osr"
            replay_file.write_bytes(replay_data)

    """ Update the user's & beatmap's stats """

    # get the current stats, and take a
    # shallow copy for the response charts.
    stats = player.gm_stats
    prev_stats = copy.copy(stats)

    # stuff update for all submitted scores
    stats.playtime += score.time_elapsed // 1000
    stats.plays += 1
    stats.tscore += score.score
    stats.total_hits += score.n300 + score.n100 + score.n50

    if score.mode.as_vanilla in (1, 3):
        # taiko uses geki & katu for hitting big notes with 2 keys
        # mania uses geki & katu for rainbow 300 & 200
        stats.total_hits += score.ngeki + score.nkatu

    stats_query_l = [
        "UPDATE stats SET plays = :plays, playtime = :playtime, tscore = :tscore, "
        "total_hits = :total_hits",
    ]

    stats_query_args: dict[str, object] = {
        "plays": stats.plays,
        "playtime": stats.playtime,
        "tscore": stats.tscore,
        "total_hits": stats.total_hits,
    }

    if score.passed and beatmap.has_leaderboard:
        # player passed & map is ranked, approved, or loved.

        if score.max_combo > stats.max_combo:
            stats.max_combo = score.max_combo
            stats_query_l.append("max_combo = :max_combo")
            stats_query_args["max_combo"] = stats.max_combo

        if beatmap.awards_ranked_pp and score.status == SubmissionStatus.BEST:
            # map is ranked or approved, and it's our (new)
            # best score on the map. update the player's
            # ranked score, grades, pp, acc and global rank.

            additional_rscore = score.score
            if score.prev_best:
                # we previously had a score, so remove
                # it's score from our ranked score.
                additional_rscore -= score.prev_best.score

                if score.grade != score.prev_best.grade:
                    if score.grade >= Grade.A:
                        stats.grades[score.grade] += 1
                        grade_col = format(score.grade, "stats_column")
                        stats_query_l.append(f"{grade_col} = {grade_col} + 1")

                    if score.prev_best.grade >= Grade.A:
                        stats.grades[score.prev_best.grade] -= 1
                        grade_col = format(score.prev_best.grade, "stats_column")
                        stats_query_l.append(f"{grade_col} = {grade_col} - 1")
            else:
                # this is our first submitted score on the map
                if score.grade >= Grade.A:
                    stats.grades[score.grade] += 1
                    grade_col = format(score.grade, "stats_column")
                    stats_query_l.append(f"{grade_col} = {grade_col} + 1")

            stats.rscore += additional_rscore
            stats_query_l.append("rscore = :rscore")
            stats_query_args["rscore"] = stats.rscore

            # fetch scores sorted by pp for total acc/pp calc
            # NOTE: we select all plays (and not just top100)
            # because bonus pp counts the total amount of ranked
            # scores. i'm aware this scales horribly and it'll
            # likely be split into two queries in the future.
            best_scores = await db_conn.fetch_all(
                "SELECT s.pp, s.acc FROM scores s "
                "INNER JOIN maps m ON s.map_md5 = m.md5 "
                "WHERE s.userid = :user_id AND s.mode = :mode "
                "AND s.status = 2 AND m.status IN (2, 3) "  # ranked, approved
                "ORDER BY s.pp DESC",
                {"user_id": player.id, "mode": score.mode},
            )

            total_scores = len(best_scores)
            top_100_pp = best_scores[:100]

            # calculate new total weighted accuracy
            weighted_acc = sum(
                row["acc"] * 0.95**i for i, row in enumerate(top_100_pp)
            )
            bonus_acc = 100.0 / (20 * (1 - 0.95**total_scores))
            stats.acc = (weighted_acc * bonus_acc) / 100

            # add acc to query
            stats_query_l.append("acc = :acc")
            stats_query_args["acc"] = stats.acc

            # calculate new total weighted pp
            weighted_pp = sum(row["pp"] * 0.95**i for i, row in enumerate(top_100_pp))
            bonus_pp = 416.6667 * (1 - 0.95**total_scores)
            stats.pp = round(weighted_pp + bonus_pp)

            # add pp to query
            stats_query_l.append("pp = :pp")
            stats_query_args["pp"] = stats.pp

            # update global & country ranking
            stats.rank = await usecases.players.update_rank(player, score.mode)

    # create a single querystring from the list of updates
    stats_query = ", ".join(stats_query_l)

    stats_query += " WHERE id = :user_id AND mode = :mode"
    stats_query_args["user_id"] = player.id
    stats_query_args["mode"] = score.mode.value

    # send any stat changes to sql, and other players
    await db_conn.execute(stats_query, stats_query_args)

    if not player.restricted:
        # enqueue new stats info to all other users
        app.state.sessions.players.enqueue(app.packets.user_stats(player))

        # update beatmap with new stats
        beatmap.plays += 1
        if score.passed:
            beatmap.passes += 1

        await db_conn.execute(
            "UPDATE maps SET plays = :plays, passes = :passes WHERE md5 = :map_md5",
            {
                "plays": beatmap.plays,
                "passes": beatmap.passes,
                "map_md5": beatmap.md5,
            },
        )

    # update their recent score
    player.recent_scores[score.mode] = score.id

    """ score submission charts """

    if not score.passed or score.mode >= GameMode.RELAX_OSU:
        # charts & achievements won't be shown ingame.
        ret = b"error: no"

    else:
        # construct and send achievements & ranking charts to the client
        if beatmap.awards_ranked_pp and not player.restricted:
            achievements = []
            for achievement in app.state.sessions.achievements:
                if achievement in player.achievements:
                    # player already has this achievement.
                    continue

                if achievement.condition(score, score.mode.as_vanilla):
                    await usecases.players.unlock_achievement(player, achievement)
                    achievements.append(achievement)

            achievements_str = "/".join(map(repr, achievements))
        else:
            achievements_str = ""

        # create score submission charts for osu! client to display

        if score.prev_best:
            beatmap_ranking_chart_entries = (
                chart_entry("rank", score.prev_best.rank, score.rank),
                chart_entry("rankedScore", score.prev_best.score, score.score),
                chart_entry("totalScore", score.prev_best.score, score.score),
                chart_entry("maxCombo", score.prev_best.max_combo, score.max_combo),
                chart_entry(
                    "accuracy",
                    round(score.prev_best.acc, 2),
                    round(score.acc, 2),
                ),
                chart_entry("pp", score.prev_best.pp, score.pp),
            )
        else:
            # no previous best score
            beatmap_ranking_chart_entries = (
                chart_entry("rank", None, score.rank),
                chart_entry("rankedScore", None, score.score),
                chart_entry("totalScore", None, score.score),
                chart_entry("maxCombo", None, score.max_combo),
                chart_entry("accuracy", None, round(score.acc, 2)),
                chart_entry("pp", None, score.pp),
            )

        overall_ranking_chart_entries = (
            chart_entry("rank", prev_stats.rank, stats.rank),
            chart_entry("rankedScore", prev_stats.rscore, stats.rscore),
            chart_entry("totalScore", prev_stats.tscore, stats.tscore),
            chart_entry("maxCombo", prev_stats.max_combo, stats.max_combo),
            chart_entry("accuracy", round(prev_stats.acc, 2), round(stats.acc, 2)),
            chart_entry("pp", prev_stats.pp, stats.pp),
        )

        submission_charts = [
            # beatmap info chart
            f"beatmapId:{beatmap.id}",
            f"beatmapSetId:{beatmap.set_id}",
            f"beatmapPlaycount:{beatmap.plays}",
            f"beatmapPasscount:{beatmap.passes}",
            f"approvedDate:{beatmap.last_update}",
            "\n",
            # beatmap ranking chart
            "chartId:beatmap",
            f"chartUrl:https://osu.{app.settings.DOMAIN}/beatmapsets/{beatmap.set_id}",
            "chartName:Beatmap Ranking",
            *beatmap_ranking_chart_entries,
            f"onlineScoreId:{score.id}",
            "\n",
            # overall ranking chart
            "chartId:overall",
            f"chartUrl:https://{app.settings.DOMAIN}/u/{player.id}",
            "chartName:Overall Ranking",
            *overall_ranking_chart_entries,
            f"achievements-new:{achievements_str}",
        ]

        ret = "|".join(submission_charts).encode()

    log(
        f"[{score.mode!r}] {player} submitted a score! "
        f"({score.status!r}, {score.pp:,.2f}pp / {stats.pp:,}pp)",
        Ansi.LGREEN,
    )

    return ret


@router.get("/web/osu-getreplay.php")
async def get_score_replay(
    player: Player = Depends(authenticate_player_session(Query, "u", "h")),
    mode: int = Query(..., alias="m", ge=0, le=3),
    score_id: int = Query(..., alias="c", min=0, max=9_223_372_036_854_775_807),
):
    replay_file = await usecases.replays.fetch_file(score_id)
    if replay_file is None:
        return

    await usecases.scores.increment_replay_views(player.id, mode)
    return FileResponse(replay_file)


@router.get("/web/osu-rate.php")
async def post_beatmap_rating(
    player: Player = Depends(
        authenticate_player_session(Query, "u", "p", err=b"auth fail"),
    ),
    map_md5: str = Query(..., alias="c", min_length=32, max_length=32),
    rating: Optional[int] = Query(None, alias="v", ge=1, le=10),
    db_conn: databases.core.Connection = Depends(acquire_db_conn),
):
    if rating is None:
        # check if we have the map in our cache;
        # if not, the map probably doesn't exist.
        beatmap = repositories.beatmaps._fetch_by_key_cache(map_md5)
        if beatmap is None:
            return b"no exist"

        # only allow rating on maps with a leaderboard.
        if beatmap.status < RankedStatus.Ranked:
            return b"not ranked"

        # osu! client is checking whether we can rate the map or not.
        has_previous_rating = (
            await db_conn.fetch_one(
                "SELECT 1 FROM ratings WHERE map_md5 = :map_md5 AND userid = :user_id",
                {"map_md5": map_md5, "user_id": player.id},
            )
            is not None
        )

        # the client hasn't rated the map, so simply
        # tell them that they can submit a rating.
        if not has_previous_rating:
            return b"ok"
    else:
        # the client is submitting a rating for the map.
        await db_conn.execute(
            "INSERT INTO ratings VALUES (:user_id, :map_md5, :rating)",
            {"user_id": player.id, "map_md5": map_md5, "rating": int(rating)},
        )

    ratings = [
        row[0]
        for row in await db_conn.fetch_all(
            "SELECT rating FROM ratings WHERE map_md5 = :map_md5",
            {"map_md5": map_md5},
        )
    ]

    # send back the average rating
    avg = sum(ratings) / len(ratings)
    return f"alreadyvoted\n{avg}".encode()


@unique
@pymysql_encode(escape_enum)
class LeaderboardType(IntEnum):
    Local = 0
    Top = 1
    Mods = 2
    Friends = 3
    Country = 4


async def get_leaderboard_scores(
    leaderboard_type: Union[LeaderboardType, int],
    map_md5: str,
    mode: int,
    mods: Mods,
    player: Player,
    scoring_metric: Literal["pp", "score"],
) -> tuple[list[Mapping[str, Any]], Optional[Mapping[str, Any]]]:
    query = [
        f"SELECT s.id, s.{scoring_metric} AS _score, "
        "s.max_combo, s.n50, s.n100, s.n300, "
        "s.nmiss, s.nkatu, s.ngeki, s.perfect, s.mods, "
        "UNIX_TIMESTAMP(s.play_time) time, u.id userid, "
        "COALESCE(CONCAT('[', c.tag, '] ', u.name), u.name) AS name "
        "FROM scores s "
        "INNER JOIN users u ON u.id = s.userid "
        "LEFT JOIN clans c ON c.id = u.clan_id "
        "WHERE s.map_md5 = :map_md5 AND s.status = 2 "  # 2: =best score
        "AND (u.priv & 1 OR u.id = :user_id) AND mode = :mode",
    ]

    params = {"map_md5": map_md5, "user_id": player.id, "mode": mode}

    if leaderboard_type == LeaderboardType.Mods:
        query.append("AND s.mods = :mods")
        params["mods"] = mods
    elif leaderboard_type == LeaderboardType.Friends:
        query.append("AND s.userid IN :friends")
        params["friends"] = player.friends | {player.id}
    elif leaderboard_type == LeaderboardType.Country:
        query.append("AND u.country = :country")
        params["country"] = player.geoloc["country"]["acronym"]

    # TODO: customizability of the number of scores
    query.append("ORDER BY _score DESC LIMIT 50")

    async with app.state.services.database.connection() as db_conn:
        score_rows = await app.state.services.database.fetch_all(
            " ".join(query),
            params,
        )

        if score_rows:  # None or []
            # fetch player's personal best score
            personal_best_score_row = await db_conn.fetch_one(
                f"SELECT id, {scoring_metric} AS _score, "
                "max_combo, n50, n100, n300, "
                "nmiss, nkatu, ngeki, perfect, mods, "
                "UNIX_TIMESTAMP(play_time) time "
                "FROM scores "
                "WHERE map_md5 = :map_md5 AND mode = :mode "
                "AND userid = :user_id AND status = 2 "
                "ORDER BY _score DESC LIMIT 1",
                {"map_md5": map_md5, "mode": mode, "user_id": player.id},
            )

            if personal_best_score_row:
                # calculate the rank of the score.
                p_best_rank = 1 + await db_conn.fetch_val(
                    "SELECT COUNT(*) FROM scores s "
                    "INNER JOIN users u ON u.id = s.userid "
                    "WHERE s.map_md5 = :map_md5 AND s.mode = :mode "
                    "AND s.status = 2 AND u.priv & 1 "
                    f"AND s.{scoring_metric} > :score",
                    {
                        "map_md5": map_md5,
                        "mode": mode,
                        "score": personal_best_score_row["_score"],
                    },
                    column=0,  # COUNT(*)
                )

                # attach rank to personal best row
                personal_best_score_row = dict(personal_best_score_row)
                personal_best_score_row["rank"] = p_best_rank
            else:
                personal_best_score_row = None
        else:
            score_rows = []
            personal_best_score_row = None

    return score_rows, personal_best_score_row


SCORE_LISTING_FMTSTR = (
    "{id}|{name}|{score}|{max_combo}|"
    "{n50}|{n100}|{n300}|{nmiss}|{nkatu}|{ngeki}|"
    "{perfect}|{mods}|{userid}|{rank}|{time}|{has_replay}"
)


@router.get("/web/osu-osz2-getscores.php")
async def get_beatmap_leaderboard(
    player: Player = Depends(authenticate_player_session(Query, "us", "ha")),
    requesting_from_editor_song_select: bool = Query(..., alias="s"),
    leaderboard_version: int = Query(..., alias="vv"),
    leaderboard_type: int = Query(..., alias="v", ge=0, le=4),
    map_md5: str = Query(..., alias="c", min_length=32, max_length=32),
    map_filename: str = Query(..., alias="f"),  # TODO: regex?
    mode_arg: int = Query(..., alias="m", ge=0, le=3),
    map_set_id: int = Query(..., alias="i", ge=-1, le=2_147_483_647),
    mods_arg: int = Query(..., alias="mods", ge=0, le=2_147_483_647),
    map_package_hash: str = Query(..., alias="h"),  # TODO: further validation
    aqn_files_found: bool = Query(..., alias="a"),
    db_conn: databases.core.Connection = Depends(acquire_db_conn),
):
    if aqn_files_found:
        stacktrace = app.utils.get_appropriate_stacktrace()
        await app.state.services.log_strange_occurrence(stacktrace)

    # check if this md5 has already been  cached as
    # unsubmitted/needs update to reduce osu!api spam
    if map_md5 in app.state.cache.unsubmitted:
        return b"-1|false"
    if map_md5 in app.state.cache.needs_update:
        return b"1|false"

    if mods_arg & Mods.RELAX:
        if mode_arg == 3:  # rx!mania doesn't exist
            mods_arg &= ~Mods.RELAX
        else:
            mode_arg += 4
    elif mods_arg & Mods.AUTOPILOT:
        if mode_arg in (1, 2, 3):  # ap!catch, taiko and mania don't exist
            mods_arg &= ~Mods.AUTOPILOT
        else:
            mode_arg += 8

    mods = Mods(mods_arg)
    mode = GameMode(mode_arg)

    # attempt to update their stats if their
    # gm/gm-affecting-mods change at all.
    if mode != player.status.mode:
        player.status.mods = mods
        player.status.mode = mode

        if not player.restricted:
            app.state.sessions.players.enqueue(app.packets.user_stats(player))

    scoring_metric = "pp" if mode >= GameMode.RELAX_OSU else "score"

    if map_set_id > 0:
        # focus on long-term efficiency - cache the whole set
        await repositories.beatmap_sets.fetch_by_id(map_set_id)

    beatmap = await repositories.beatmaps.fetch_by_md5(map_md5)

    if beatmap is None:
        # map not found, figure out whether it needs an
        # update or isn't submitted using it's filename.

        map_filename = unquote_plus(map_filename)  # TODO: is unquote needed?
        map_exists = await usecases.beatmaps.filename_exists(map_filename)

        if map_exists:
            # map can be updated.
            app.state.cache.needs_update.add(map_md5)
            return b"1|false"
        else:
            # map is unsubmitted.
            # add this map to the unsubmitted cache, so
            # that we don't have to make this request again.
            app.state.cache.unsubmitted.add(map_md5)
            return b"-1|false"

    # we've found a beatmap for the request.

    if app.state.services.datadog is not None:
        app.state.services.datadog.increment("bancho.leaderboards_served")

    if beatmap.status < RankedStatus.Ranked:
        # only show leaderboards for ranked,
        # approved, qualified, or loved maps.
        return f"{int(beatmap.status)}|false".encode()

    # fetch scores & personal best
    # TODO: create a leaderboard cache
    if not requesting_from_editor_song_select:
        score_rows, personal_best_score_row = await get_leaderboard_scores(
            leaderboard_type,
            beatmap.md5,
            mode,
            mods,
            player,
            scoring_metric,
        )
    else:
        score_rows = []
        personal_best_score_row = None

    # fetch beatmap rating
    rating = await usecases.beatmaps.fetch_rating(beatmap)
    if rating is None:
        rating = 0.0

    ## construct response for osu! client

    response_lines: list[str] = [
        # NOTE: fa stands for featured artist (for the ones that may not know)
        # {ranked_status}|{serv_has_osz2}|{bid}|{bsid}|{len(scores)}|{fa_track_id}|{fa_license_text}
        f"{int(beatmap.status)}|false|{beatmap.id}|{beatmap.set_id}|{len(score_rows)}|0|",
        # {offset}\n{beatmap_name}\n{rating}
        # TODO: server side beatmap offsets
        f"0\n{beatmap.full_name}\n{rating}",
    ]

    if not score_rows:
        response_lines.extend(("", ""))  # no scores, no personal best
        return "\n".join(response_lines).encode()

    if personal_best_score_row is not None:
        response_lines.append(
            SCORE_LISTING_FMTSTR.format(
                **personal_best_score_row,
                name=player.name,
                userid=player.id,
                score=int(personal_best_score_row["_score"]),
                has_replay="1",
            ),
        )
    else:
        response_lines.append("")

    response_lines.extend(
        [
            SCORE_LISTING_FMTSTR.format(
                **s,
                score=int(s["_score"]),
                has_replay="1",
                rank=idx + 1,
            )
            for idx, s in enumerate(score_rows)
        ],
    )

    return "\n".join(response_lines).encode()


def format_comments(comments: list[Mapping[str, Any]]) -> bytes:
    ret: list[str] = []

    for cmt in comments:
        # TODO: maybe support player/creator colours?
        # pretty expensive for very low gain, but completion :D
        if cmt["priv"] & Privileges.NOMINATOR:
            fmt = "bat"
        elif cmt["priv"] & Privileges.DONATOR:
            fmt = "supporter"
        else:
            fmt = ""

        if cmt["colour"]:
            fmt += f'|{cmt["colour"]}'

        ret.append(
            "{time}\t{target_type}\t{fmt}\t{comment}".format(fmt=fmt, **cmt),
        )
    return "\n".join(ret).encode()


@router.post("/web/osu-comment.php")
async def beatmap_comments_handler(
    player: Player = Depends(authenticate_player_session(Form, "u", "p")),
    map_id: int = Form(..., alias="b"),
    map_set_id: int = Form(..., alias="s"),
    score_id: int = Form(..., alias="r", ge=0, le=9_223_372_036_854_775_807),
    mode_vn: int = Form(..., alias="m", ge=0, le=3),
    action: Literal["get", "post"] = Form(..., alias="a"),
    # only sent for post
    target: Optional[Literal["song", "map", "replay"]] = Form(None),
    colour: Optional[str] = Form(None, alias="f", min_length=6, max_length=6),
    start_time: Optional[int] = Form(None, alias="starttime"),
    comment: Optional[str] = Form(None, min_length=1, max_length=80),
):
    if action == "get":
        # client is requesting all comments
        comments = await usecases.comments.fetch_all(
            score_id,
            map_set_id,
            map_id,
        )

        resp = format_comments(comments)
    elif action == "post":
        # client is submitting a new comment

        # validate required parameters are present
        if target is None or comment is None or start_time is None:
            return None

        # get the corresponding id from the request
        if target == "song":
            target_id = map_set_id
        elif target == "map":
            target_id = map_id
        else:  # target == "replay"
            target_id = score_id

        if colour is not None and not player.priv & Privileges.DONATOR:
            # only supporters can use colours.
            # TODO: should we be restricting them?
            colour = None

        # insert into sql
        await usecases.comments.create(
            player,
            target,
            target_id,
            colour,
            comment,
            start_time,
        )

        resp = None  # empty resp is fine

    usecases.players.update_latest_activity_soon(player)
    return resp


@router.get("/web/osu-markasread.php")
async def mark_channel_as_read(
    player: Player = Depends(authenticate_player_session(Query, "u", "h")),
    channel: str = Query(..., min_length=0, max_length=32),
):
    if not (channel_name := unquote(channel)):  # TODO: unquote needed?
        return  # no channel specified

    target_player = await repositories.players.fetch(name=channel_name)

    if target_player is not None:
        await usecases.mail.mark_as_read(
            source_id=target_player.id,
            target_id=player.id,
        )


@router.get("/web/osu-getseasonal.php")
async def get_sesonal_backgrounds():
    """Handle a request from osu! to fetch seasonal background urls."""
    return ORJSONResponse(app.settings.SEASONAL_BGS._items)


@router.get("/web/bancho_connect.php")
async def bancho_connect_preflight(
    player: Player = Depends(authenticate_player_session(Query, "u", "h")),
    osu_ver: str = Query(..., alias="v"),
    active_endpoint: Optional[str] = Query(None, alias="fail"),
    net_framework_vers: Optional[str] = Query(None, alias="fx"),  # delimited by |
    client_hash: Optional[str] = Query(None, alias="ch"),
    retrying: Optional[bool] = Query(None, alias="retry"),  # '0' or '1'
):
    # TODO: support for client verification?

    return b""


@router.get("/web/check-updates.php")
async def check_updates(
    request: Request,
    action: Literal["check", "path", "error"],
    stream: Literal["cuttingedge", "stable40", "beta40", "stable"],
):
    return

    # NOTE: this code is unused now.
    # it was only used with server switchers,
    # which bancho.py has deprecated support for.
    return await usecases.client_versioning.check_updates(
        action,
        stream,
        request.query_params,
    )


""" Misc handlers """


if app.settings.REDIRECT_OSU_URLS:
    # NOTE: this will likely be removed with the addition of a frontend.
    async def osu_redirect(request: Request, _: int = Path(...)):
        return RedirectResponse(
            url=f"https://osu.ppy.sh{request['path']}",
            status_code=status.HTTP_301_MOVED_PERMANENTLY,
        )

    for pattern in (
        "/beatmapsets/{_}",
        "/beatmaps/{_}",
        "/community/forums/topics/{_}",
    ):
        router.get(pattern)(osu_redirect)


@router.get("/ss/{screenshot_id}.{extension}")
async def get_screenshot(
    screenshot_id: str = Path(..., regex=r"^[a-zA-Z0-9-_]{8}$"),
    extension: Literal["jpg", "jpeg", "png"] = Path(...),
):
    """Serve a screenshot from the server, by filename."""
    screenshot_file = usecases.screenshots.fetch_file(screenshot_id, extension)

    if screenshot_file is None:
        return ORJSONResponse(
            content={"status": "Screenshot not found."},
            status_code=status.HTTP_404_NOT_FOUND,
        )

    return FileResponse(
        screenshot_file,
        media_type=app.utils.get_media_type(extension),  # type: ignore
    )


@router.get("/d/{map_set_id}")
async def get_osz(
    map_set_id: str = Path(...),
):
    """Handle a map download request (osu.ppy.sh/d/*)."""
    no_video = map_set_id[-1] == "n"
    if no_video:
        map_set_id = map_set_id[:-1]

    download_url = usecases.direct.get_mapset_download_url(
        int(map_set_id),
        no_video,
    )

    return RedirectResponse(
        url=download_url,
        status_code=status.HTTP_301_MOVED_PERMANENTLY,
    )


@router.get("/web/maps/{map_filename}")
async def get_updated_beatmap(map_filename: str, host: str = Header(...)):
    """Send the latest .osu file the server has for a given map."""
    if host != "osu.ppy.sh":
        update_url = usecases.direct.get_mapset_update_url(map_filename)

        return RedirectResponse(
            url=update_url,
            status_code=status.HTTP_301_MOVED_PERMANENTLY,
        )

    return

    # NOTE: this code is unused now.
    # it was only used with server switchers,
    # which bancho.py has deprecated support for.

    # server switcher, use old method
    map_filename = unquote(map_filename)

    if not (
        res := await app.state.services.database.fetch_one(
            "SELECT id, md5 FROM maps WHERE filename = :filename",
            {"filename": map_filename},
        )
    ):
        return Response(status_code=status.HTTP_400_BAD_REQUEST)

    osu_file_path = BEATMAPS_PATH / f'{res["id"]}.osu'

    if (
        osu_file_path.exists()
        and res["md5"] == hashlib.md5(osu_file_path.read_bytes()).hexdigest()
    ):
        # up to date map found on disk.
        content = osu_file_path.read_bytes()
    else:
        # map not found, or out of date; get from osu!
        url = f"https://old.ppy.sh/osu/{res['id']}"

        async with app.state.services.http_client.get(url) as resp:
            if not resp or resp.status != 200:
                log(f"Could not find map {osu_file_path}!", Ansi.LRED)
                return (404, b"")  # couldn't find on osu!'s server

            content = await resp.read()

        # save it to disk for future
        osu_file_path.write_bytes(content)

    return content


@router.get("/p/doyoureallywanttoaskpeppy")
async def peppy_direct_message_handler():
    return (
        b"This user's ID is usually peppy's (when on bancho), "
        b"and is blocked from being messaged by the osu! client."
    )


""" ingame registration """


@router.post("/users")
async def register_account(
    request: Request,
    username: str = Form(..., alias="user[username]"),
    email: str = Form(..., alias="user[user_email]"),
    pw_plaintext: str = Form(..., alias="user[password]"),
    only_validating_params: int = Form(..., alias="check"),
    cloudflare_country: Optional[str] = Header(None, alias="CF-IPCountry"),
):
    errors = await validation.osu_registration(username, email, pw_plaintext)
    if errors:
        return responses.osu_registration_failure(errors)

    if not only_validating_params:  # == 0
        # fetch the country code from the request

        if cloudflare_country:
            # FASTPATH: use cloudflare country header
            country_acronym = cloudflare_country.lower()
        else:
            ip = app.state.services.ip_resolver.get_ip(request.headers)

            geoloc = await usecases.geolocation.lookup(ip)
            if geoloc is not None:
                country_acronym = geoloc["country"]["acronym"]
            else:
                country_acronym = "xx"

        player_id = await usecases.players.register(
            username,
            email,
            pw_plaintext,
            country_acronym,
        )

        if app.state.services.datadog is not None:
            app.state.services.datadog.increment("bancho.registrations")

        log(f"<{username} ({player_id})> has registered!", Ansi.LGREEN)

    return b"ok"  # success


@router.post("/difficulty-rating")
async def difficultyRatingHandler(request: Request):
    return RedirectResponse(
        url=f"https://osu.ppy.sh{request['path']}",
        status_code=status.HTTP_307_TEMPORARY_REDIRECT,
    )

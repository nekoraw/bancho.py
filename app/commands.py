from __future__ import annotations

import copy
import importlib.metadata
import os
import pprint
import random
import secrets
import signal
import struct
import time
import traceback
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from time import perf_counter_ns as clock_ns
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Mapping
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import TYPE_CHECKING
from typing import TypedDict
from typing import TypeVar
from typing import Union
from urllib.parse import urlparse

import psutil
import timeago
from pytimeparse.timeparse import timeparse

import app.logging
import app.pycord
import app.packets
import app.settings
import app.state
import app.usecases.performance
import app.utils
from app.constants import regexes
from app.constants.gamemodes import GameMode
from app.constants.gamemodes import GAMEMODE_REPR_LIST
from app.constants.mods import Mods
from app.constants.mods import SPEED_CHANGING_MODS
from app.constants.privileges import ClanPrivileges
from app.constants.privileges import Privileges
from app.objects.beatmap import Beatmap
from app.objects.beatmap import ensure_local_osu_file
from app.objects.beatmap import RankedStatus
from app.objects.clan import Clan
from app.objects.match import MapPool
from app.objects.match import Match
from app.objects.match import MatchTeams
from app.objects.match import MatchTeamTypes
from app.objects.match import MatchWinConditions
from app.objects.match import SlotStatus
from app.objects.player import Player
from app.objects.score import SubmissionStatus
from app.repositories import clans as clans_repo
from app.repositories import maps as maps_repo
from app.repositories import players as players_repo
from app.usecases.performance import ScoreParams
from app.utils import seconds_readable

if TYPE_CHECKING:
    from app.objects.channel import Channel


BEATMAPS_PATH = Path.cwd() / ".data/osu"


@dataclass
class Context:
    player: Player
    trigger: str
    args: Sequence[str]

    recipient: Union[Channel, Player]


Callback = Callable[[Context], Awaitable[Optional[str]]]


class Command(NamedTuple):
    triggers: list[str]
    callback: Callback
    priv: Privileges
    hidden: bool
    doc: Optional[str]


class CommandSet:
    def __init__(self, trigger: str, doc: str) -> None:
        self.trigger = trigger
        self.doc = doc

        self.commands: list[Command] = []

    def add(
        self,
        priv: Privileges,
        aliases: list[str] = [],
        hidden: bool = False,
    ) -> Callable[[Callback], Callback]:
        def wrapper(f: Callback) -> Callback:
            self.commands.append(
                Command(
                    # NOTE: this method assumes that functions without any
                    # triggers will be named like '{self.trigger}_{trigger}'.
                    triggers=(
                        [f.__name__.removeprefix(f"{self.trigger}_").strip()] + aliases
                    ),
                    callback=f,
                    priv=priv,
                    hidden=hidden,
                    doc=f.__doc__,
                ),
            )

            return f

        return wrapper


# TODO: refactor help commands into some base ver
#       since they're all the same anyway lol.

mp_commands = CommandSet("mp", "Comandos de Multiplayer.")
pool_commands = CommandSet("pool", "Comandos de Mappool.")
clan_commands = CommandSet("clan", "Comandos de Clã.")

regular_commands = []
command_sets = [
    mp_commands,
    pool_commands,
    clan_commands,
]


def command(
    priv: Privileges,
    aliases: list[str] = [],
    hidden: bool = False,
) -> Callable[[Callback], Callback]:
    def wrapper(f: Callback) -> Callback:
        regular_commands.append(
            Command(
                callback=f,
                priv=priv,
                hidden=hidden,
                triggers=[f.__name__.strip("_")] + aliases,
                doc=f.__doc__,
            ),
        )

        return f

    return wrapper


""" User commands
# The commands below are not considered dangerous,
# and are granted to any unbanned players.
"""


@command(Privileges.UNRESTRICTED, aliases=["", "h"], hidden=True)
async def _help(ctx: Context) -> Optional[str]:
    """Mostra todos os comandos documentados que um jogador consegue usar."""
    prefix = app.settings.COMMAND_PREFIX
    l = ["Comandos individuais", "-----------"]

    for cmd in regular_commands:
        if not cmd.doc or ctx.player.priv & cmd.priv != cmd.priv:
            # no doc, or insufficient permissions.
            continue

        l.append(f"{prefix}{cmd.triggers[0]}: {cmd.doc}")

    l.append("")  # newline
    l.extend(["Conjuntos de comandos", "-----------"])

    for cmd_set in command_sets:
        l.append(f"{prefix}{cmd_set.trigger}: {cmd_set.doc}")

    return "\n".join(l)


@command(Privileges.UNRESTRICTED)
async def roll(ctx: Context) -> Optional[str]:
    """Roda um dado de n lados onde n é o número que você vai dar. (padrão 100)"""
    if ctx.args and ctx.args[0].isdecimal():
        max_roll = min(int(ctx.args[0]), 0x7FFF)
    else:
        max_roll = 100

    if max_roll == 0:
        return "Rodar o quê?"

    points = random.randrange(0, max_roll)
    return f"{ctx.player.name} rolou {points} pontos!"


@command(Privileges.UNRESTRICTED, hidden=True)
async def block(ctx: Context) -> Optional[str]:
    """Bloqueia a comunicação de outro usuário com você."""
    target = await app.state.sessions.players.from_cache_or_sql(name=" ".join(ctx.args))

    if not target:
        return "Usuário não encontrado."

    if target is app.state.sessions.bot or target is ctx.player:
        return "Quê?"

    if target.id in ctx.player.blocks:
        return f"{target.name} já está bloqueado!"

    if target.id in ctx.player.friends:
        ctx.player.friends.remove(target.id)

    await ctx.player.add_block(target)
    return f"Adicionou {target.name} aos usuários bloqueados."


@command(Privileges.UNRESTRICTED, hidden=True)
async def unblock(ctx: Context) -> Optional[str]:
    """Desbloqueia a comunicação de outro usuário com você."""
    target = await app.state.sessions.players.from_cache_or_sql(name=" ".join(ctx.args))

    if not target:
        return "Usuário não encontrado."

    if target is app.state.sessions.bot or target is ctx.player:
        return "Quê?"

    if target.id not in ctx.player.blocks:
        return f"{target.name} não está bloqueado!"

    await ctx.player.remove_block(target)
    return f"Removeu {target.name} dos usuários bloqueados."


@command(Privileges.UNRESTRICTED)
async def reconnect(ctx: Context) -> Optional[str]:
    """Desconecta e reconecta um jogador (ou si) ao servidor."""
    if ctx.args:
        # !reconnect <player>
        if not ctx.player.priv & Privileges.ADMINISTRATOR:
            return None  # requires admin

        target = app.state.sessions.players.get(name=" ".join(ctx.args))
        if not target:
            return "Jogador não encontrado."
    else:
        # !reconnect
        target = ctx.player

    target.logout()

    return None


@command(Privileges.SUPPORTER)
async def changename(ctx: Context) -> Optional[str]:
    """Muda o seu nome de usuário."""
    name = " ".join(ctx.args).strip()

    if not regexes.USERNAME.match(name):
        return "Deve ter de 2-15 caracteres de tamanho."

    if "_" in name and " " in name:
        return 'Pode conter "_" e " ", mas não ambos.'

    if name in app.settings.DISALLOWED_NAMES:
        return "Nome não permitido; escolha outro."

    if await players_repo.fetch_one(name=name):
        return "Outro jogador já possui esse nome."

    # all checks passed, update their name
    await players_repo.update(ctx.player.id, name=name)

    ctx.player.enqueue(
        app.packets.notification(f"O seu nome de usuário mudou para {name}!"),
    )
    ctx.player.logout()

    return None


@command(Privileges.UNRESTRICTED, aliases=["bloodcat", "beatconnect", "chimu", "q"])
async def maplink(ctx: Context) -> Optional[str]:
    """Retorna um link de download para o mapa atual do usuário (depende da situação)."""
    bmap = None

    # priority: multiplayer -> spectator -> last np
    match = ctx.player.match
    spectating = ctx.player.spectating

    if match and match.map_id:
        bmap = await Beatmap.from_md5(match.map_md5)
    elif spectating and spectating.status.map_id:
        bmap = await Beatmap.from_md5(spectating.status.map_md5)
    elif ctx.player.last_np is not None and time.time() < ctx.player.last_np["timeout"]:
        bmap = ctx.player.last_np["bmap"]

    if bmap is None:
        return "Mapa não encontrado!"

    # gatari.pw & nerina.pw are pretty much the only
    # reliable mirrors I know of? perhaps beatconnect
    return f"[https://osu.gatari.pw/d/{bmap.set_id} {bmap.full_name}]"


async def can_generate_key(player: Player) -> bool:
    if player.n_keys == 1:
        return True
    
    user_keys = await app.state.services.database.fetch_one(
        f"SELECT creation_time FROM register_keys WHERE user_id_created = \"{player.id}\" ORDER BY creation_time DESC"
    )
    
    now = datetime.utcnow()
    if not user_keys:
        time_since_creation = now - datetime.fromtimestamp(player.creation_time)
        if time_since_creation >= timedelta(days=14):
            query = "UPDATE users SET n_available_keys = :n_available_keys WHERE id = :id"
            params = {
                "n_available_keys": 1,
                "id": player.id
            }
            await app.state.services.database.execute(query, params)
            return True
        else:
            return False
    else:
        latest_key = user_keys[0]
        time_since_last = now - datetime.fromtimestamp(latest_key)
        if time_since_last >= timedelta(days=14):
            query = "UPDATE users SET n_available_keys = :n_available_keys WHERE id = :id"
            params = {
                "n_available_keys": 1,
                "id": player.id
            }
            await app.state.services.database.execute(query, params)
            return True
        else:
            return False


@command(Privileges.UNRESTRICTED, aliases=["available_key"])
async def tenho_chave(ctx: Context) -> Optional[str]:
    """Verifica se o usuário possui uma chave de registro disponível."""
    player = ctx.player
    
    if player.n_keys == 1:
        return "Você possui uma chave disponível para resgate! Envie !gerar_chave para resgatá-la."
    
    user_keys = await app.state.services.database.fetch_one(
        f"SELECT creation_time FROM register_keys WHERE user_id_created = \"{player.id}\" ORDER BY creation_time DESC"
    )
    
    now = datetime.utcnow()
    
    if not user_keys:
        time_since_creation = now - datetime.fromtimestamp(player.creation_time)
        if time_since_creation >= timedelta(days=14):
            query = "UPDATE users SET n_available_keys = :n_available_keys WHERE id = :id"
            params = {
                "n_available_keys": 1,
                "id": player.id
            }
            await app.state.services.database.execute(query, params)
            return "Você possui uma chave disponível para resgate! Envie !gerar_chave para resgatá-la."
        else:
            remaining_time = timedelta(days=14) - time_since_creation
            timefmt = datetime.strftime(now + remaining_time, "%d/%m/%Y %H:%M")
            return f"Você ainda não consegue gerar uma chave. Você será capaz ás {timefmt} UTC-0."
    else:
        latest_key = user_keys[0]
        time_since_last = now - datetime.fromtimestamp(latest_key)
        if time_since_last >= timedelta(days=14):
            query = "UPDATE users SET n_available_keys = :n_available_keys WHERE id = :id"
            params = {
                "n_available_keys": 1,
                "id": player.id
            }
            await app.state.services.database.execute(query, params)
            return "Você possui uma chave disponível para resgate! Envie !gerar_chave para resgatá-la."
        else:
            remaining_time = timedelta(days=14) - time_since_last
            timefmt = datetime.strftime(now + remaining_time, "%d/%m/%Y %H:%M")
            return f"Você ainda não consegue gerar uma chave. Você será capaz ás {timefmt} UTC-0."
    

@command(Privileges.UNRESTRICTED, aliases=["generate_key"])
async def gerar_chave(ctx: Context) -> Optional[str]:
    """Gera uma chave para convidar um player terceiro ao game."""
    player = ctx.player
    # if player.n_keys < 1:
    #     if not (await can_generate_key(player)):
    #         return "Você não possui uma chave disponível para ser obtida. Envie !tenho_chave para verificar quando sua próxima chave estará disponível."
        
    new_key = str(uuid.uuid4())
    query = "INSERT INTO register_keys (reg_key, user_id_created, creation_time) VALUES (:reg_key, :user_id_created, UNIX_TIMESTAMP())"
    params = {
        "reg_key": new_key,
        "user_id_created": player.id
    }
    await app.state.services.database.execute(query, params)
    
    query = "UPDATE users SET n_available_keys = :n_available_keys WHERE id = :id"
    params = {
        "n_available_keys": 0,
        "id": player.id
    }
    await app.state.services.database.execute(query, params)
    
    return f"Sua chave é {new_key}. Guarde-a em um lugar seguro, pois essa é a única vez que vc pode ver ela."


@command(Privileges.UNRESTRICTED, aliases=["last", "r"])
async def recent(ctx: Context) -> Optional[str]:
    """Mostra informação de uma pontuação recente de um jogador."""
    if ctx.args:
        target = app.state.sessions.players.get(name=" ".join(ctx.args))
        if not target:
            return "Jogador não encontrado."
    else:
        target = ctx.player

    score = target.recent_score
    if not score:
        return "Sem pontuações encontradas :o (somente é salvo por sessão de jogo)."

    if score.bmap is None:
        return "We don't have a beatmap on file for your recent score."

    l = [f"[{score.mode!r}] {score.bmap.embed}", f"{score.acc:.2f}%"]

    if score.mods:
        l.insert(1, f"+{score.mods!r}")

    l = [" ".join(l)]

    if score.passed:
        rank = score.rank if score.status == SubmissionStatus.BEST else "NA"
        l.append(f"PASSOU {{{score.pp:.2f}pp #{rank}}}")
    else:
        # XXX: prior to v3.2.0, bancho.py didn't parse total_length from
        # the osu!api, and thus this can do some zerodivision moments.
        # this can probably be removed in the future, or better yet
        # replaced with a better system to fix the maps.
        if score.bmap.total_length != 0:
            completion = score.time_elapsed / (score.bmap.total_length * 1000)
            l.append(f"FALHOU {{{completion * 100:.2f}% completo}})")
        else:
            l.append("FALHOU")

    return " | ".join(l)


TOP_SCORE_FMTSTR = (
    "{idx}. ({pp:.2f}pp) [https://osu.{domain}/beatmapsets/{map_set_id}/{map_id} "
    "{artist} - {title} [{version}]]"
)


@command(Privileges.UNRESTRICTED, hidden=True)
async def top(ctx: Context) -> Optional[str]:
    """Mostra a informação do top 10 pontuações de um jogador."""
    # !top <mode> (player)
    args_len = len(ctx.args)
    if args_len not in (1, 2):
        return "Sintaxe inválida: !top <modo> (jogador)"

    if ctx.args[0] not in GAMEMODE_REPR_LIST:
        return f'Modos de jogo válidos: {", ".join(GAMEMODE_REPR_LIST)}.'

    if ctx.args[0] in (
        "rx!mania",
        "ap!taiko",
        "ap!catch",
        "ap!mania",
    ):
        return "Combinação impossível de modos de jogo."

    if args_len == 2:
        if not regexes.USERNAME.match(ctx.args[1]):
            return "Nome de usuário inválido."

        # specific player provided
        player = app.state.sessions.players.get(name=ctx.args[1])
        if not player:
            return "Jogador não encontrado."
    else:
        # no player provided, use self
        player = ctx.player

    # !top rx!std
    mode = GAMEMODE_REPR_LIST.index(ctx.args[0])

    scores = await app.state.services.database.fetch_all(
        "SELECT s.pp, b.artist, b.title, b.version, b.set_id map_set_id, b.id map_id "
        "FROM scores s "
        "LEFT JOIN maps b ON b.md5 = s.map_md5 "
        "WHERE s.userid = :user_id "
        "AND s.mode = :mode "
        "AND s.status = 2 "
        "AND b.status in (2, 3) "
        "ORDER BY s.pp DESC LIMIT 10",
        {"user_id": player.id, "mode": mode},
    )

    if not scores:
        return "Sem pontuações."

    return "\n".join(
        [f"Top 10 pontuações para {player.embed} ({ctx.args[0]})."]
        + [
            TOP_SCORE_FMTSTR.format(idx=idx + 1, domain=app.settings.DOMAIN, **s)
            for idx, s in enumerate(scores)
        ],
    )


# TODO: !compare (compare to previous !last/!top post's map)


class ParsingError(str):
    ...


def parse__with__command_args(
    mode: int,
    args: Sequence[str],
) -> Union[Mapping[str, Any], ParsingError]:
    """Parse arguments for the !with command."""

    # tried to balance complexity vs correctness for this function
    # TODO: it can surely be cleaned up further - need to rethink it?

    if not args or len(args) > 4:
        return ParsingError("Sintaxe inválida: !with <acc/nmiss/combo/mods ...>")

    # !with 95% 1m 429x hddt
    acc = mods = combo = nmiss = None

    # parse acc, misses, combo and mods from arguments.
    # tried to balance complexity vs correctness here
    for arg in (str.lower(arg) for arg in args):
        # mandatory suffix, combo & nmiss
        if combo is None and arg.endswith("x") and arg[:-1].isdecimal():
            combo = int(arg[:-1])
            # if combo > bmap.max_combo:
            #    return "Invalid combo."
        elif nmiss is None and arg.endswith("m") and arg[:-1].isdecimal():
            nmiss = int(arg[:-1])
            # TODO: store nobjects?
            # if nmiss > bmap.combo:
            #    return "Invalid misscount."
        else:
            # optional prefix/suffix, mods & accuracy
            arg_stripped = arg.removeprefix("+").removesuffix("%")
            if mods is None and arg_stripped.isalpha() and len(arg_stripped) % 2 == 0:
                mods = Mods.from_modstr(arg_stripped)
                mods = mods.filter_invalid_combos(mode)
            elif acc is None and arg_stripped.replace(".", "", 1).isdecimal():
                acc = float(arg_stripped)
                if not 0 <= acc <= 100:
                    return ParsingError("Precisão inválida.")
            else:
                return ParsingError(f"Argumento desconhecido: {arg}")

    return {
        "acc": acc,
        "mods": mods,
        "combo": combo,
        "nmiss": nmiss,
    }


@command(Privileges.UNRESTRICTED, aliases=["w"], hidden=True)
async def _with(ctx: Context) -> Optional[str]:
    """Especifica precisão arbitrária e combinação de mods usando o `/np`."""
    if ctx.recipient is not app.state.sessions.bot:
        return "Esse comando só pode ser usado no privado com o bot."

    if ctx.player.last_np is None or time.time() >= ctx.player.last_np["timeout"]:
        return "Por favor, dê /np em um mapa antes."

    bmap: Beatmap = ctx.player.last_np["bmap"]

    osu_file_path = BEATMAPS_PATH / f"{bmap.id}.osu"
    if not await ensure_local_osu_file(osu_file_path, bmap.id, bmap.md5):
        return "Arquivo de mapa não encontrado; esse erro foi reportado."

    mode_vn = ctx.player.last_np["mode_vn"]

    command_args = parse__with__command_args(mode_vn, ctx.args)
    if isinstance(command_args, ParsingError):
        return str(command_args)

    msg_fields = []

    score_args = ScoreParams(mode=mode_vn)

    mods = command_args["mods"]
    if mods is not None:
        score_args.mods = mods
        msg_fields.append(f"{mods!r}")

    nmiss = command_args["nmiss"]
    if nmiss:
        score_args.nmiss = nmiss
        msg_fields.append(f"{nmiss}m")

    combo = command_args["combo"]
    if combo is not None:
        score_args.combo = combo
        msg_fields.append(f"{combo}x")

    acc = command_args["acc"]
    if acc is not None:
        score_args.acc = acc
        msg_fields.append(f"{acc:.2f}%")

    result = app.usecases.performance.calculate_performances(
        osu_file_path=str(osu_file_path),
        scores=[score_args],  # calculate one score
    )

    return "{msg}: {pp:.2f}pp ({stars:.2f}*)".format(
        msg=" ".join(msg_fields),
        pp=result[0]["performance"]["pp"],
        stars=result[0]["difficulty"]["stars"],  # (first score result)
    )


@command(Privileges.UNRESTRICTED, aliases=["req"])
async def request(ctx: Context) -> Optional[str]:
    """Requisita um beatmap para nominação."""
    if ctx.args:
        return "Sintaxe inválida: !request"

    if ctx.player.last_np is None or time.time() >= ctx.player.last_np["timeout"]:
        return "Por favor, dê /np em um mapa antes."

    bmap = ctx.player.last_np["bmap"]

    if bmap.status != RankedStatus.Pending:
        return "Somente mapas pendentes podem ser requisitados para a mudança de status."

    await app.state.services.database.execute(
        "INSERT INTO map_requests "
        "(map_id, player_id, datetime, active) "
        "VALUES (:map_id, :user_id, NOW(), 1)",
        {"map_id": bmap.id, "user_id": ctx.player.id},
    )

    return "Requisição enviada."


@command(Privileges.UNRESTRICTED)
async def apikey(ctx: Context) -> Optional[str]:
    """Gera uma nova chave API e a designa para o jogador."""
    return "Comando desativado."
    if ctx.recipient is not app.state.sessions.bot:
        return f"Esse comando só pode ser usado no privado com o {app.state.sessions.bot.name}."

    # remove old token
    if ctx.player.api_key:
        app.state.sessions.api_keys.pop(ctx.player.api_key)

    # generate new token
    ctx.player.api_key = str(uuid.uuid4())

    await players_repo.update(ctx.player.id, api_key=ctx.player.api_key)
    app.state.sessions.api_keys[ctx.player.api_key] = ctx.player.id

    return f"Chave API gerada. Copie sua chave API (deste url)[http://{ctx.player.api_key}]."


""" Nominator commands
# The commands below allow users to
# manage  the server's state of beatmaps.
"""


@command(Privileges.NOMINATOR, aliases=["reqs"], hidden=True)
async def requests(ctx: Context) -> Optional[str]:
    """Verifica a fila de nominação de mapas."""
    if ctx.args:
        return "Sintaxe inválida: !requests"

    rows = await app.state.services.database.fetch_all(
        "SELECT map_id, player_id, datetime FROM map_requests WHERE active = 1",
    )

    if not rows:
        return "A fila está limpa! (0 requisições de mapa)"

    l = [f"Todas requisições: {len(rows)}"]

    for map_id, player_id, dt in rows:
        # find player & map for each row, and add to output.
        player = await app.state.sessions.players.from_cache_or_sql(id=player_id)
        if not player:
            l.append(f"Falha ao achar o jogador ({player_id})?")
            continue

        bmap = await Beatmap.from_bid(map_id)
        if not bmap:
            l.append(f"Falha ao achar o mapa requisitado ({map_id})?")
            continue

        l.append(f"[{player.embed} @ {dt:%d %b %I:%M%p}] {bmap.embed}.")

    return "\n".join(l)


_status_str_to_int_map = {"unrank": 0, "rank": 2, "love": 5}


def status_to_id(s: str) -> int:
    return _status_str_to_int_map[s]


@command(Privileges.NOMINATOR)
async def _map(ctx: Context) -> Optional[str]:
    """Muda o estado de ranque do mapa mais recente que foi mandado o /np."""
    if (
        len(ctx.args) != 2
        or ctx.args[0] not in ("rank", "unrank", "love")
        or ctx.args[1] not in ("set", "map")
    ):
        return "Sintaxe inválida: !map <rank/unrank/love> <map/set>"

    if ctx.player.last_np is None or time.time() >= ctx.player.last_np["timeout"]:
        return "Por favor, dê /np em um mapa antes."

    bmap = ctx.player.last_np["bmap"]
    new_status = RankedStatus(status_to_id(ctx.args[0]))

    if ctx.args[1] == "map":
        if bmap.status == new_status:
            return f"{bmap.embed} já está {new_status!s}!"
    else:  # ctx.args[1] == "set"
        if all(map.status == new_status for map in bmap.set.maps):
            return f"All maps from the set are already {new_status!s}!"

    # update sql & cache based on scope
    # XXX: not sure if getting md5s from sql
    # for updating cache would be faster?
    # surely this will not scale as well...

    async with app.state.services.database.connection() as db_conn:
        if ctx.args[1] == "set":
            # update whole set
            await db_conn.execute(
                "UPDATE maps SET status = :status, frozen = 1 WHERE set_id = :set_id",
                {"status": new_status, "set_id": bmap.set_id},
            )

            # select all map ids for clearing map requests.
            map_ids = [
                row["id"]
                for row in await maps_repo.fetch_many(
                    set_id=bmap.set_id,
                )
            ]

            if webhook_url := app.settings.DISCORD_BANCHO_UPDATES_WEBHOOK:
                await app.pycord.send_beatmapset_status_change(webhook_url, app.state.cache.beatmapset[bmap.set_id], new_status, player_info=ctx.player)


            for bmap in app.state.cache.beatmapset[bmap.set_id].maps:
                bmap.status = new_status

        else:
            # update only map
            await maps_repo.update(bmap.id, status=new_status, frozen=True)

            map_ids = [bmap.id]

            if bmap.md5 in app.state.cache.beatmap:
                if webhook_url := app.settings.DISCORD_BANCHO_UPDATES_WEBHOOK:
                    await app.pycord.send_beatmap_status_change(webhook_url, app.state.cache.beatmap[bmap.md5], new_status, player_info=ctx.player)

                app.state.cache.beatmap[bmap.md5].status = new_status

        # deactivate rank requests for all ids
        await db_conn.execute(
            "UPDATE map_requests SET active = 0 WHERE map_id IN :map_ids",
            {"map_ids": map_ids},
        )

    return f"{bmap.embed} atualizado para {new_status!s}."


""" Mod commands
# The commands below are somewhat dangerous,
# and are generally for managing players.
"""

ACTION_STRINGS = {
    "restrict": "Restringido por",
    "unrestrict": "Sem restrições por",
    "silence": "Silenciado por",
    "unsilence": "Não silenciado por",
    "note": "Nota adicionada:",
}


@command(Privileges.MODERATOR, hidden=True)
async def notes(ctx: Context) -> Optional[str]:
    """Recolhe os logs de um jogador pelo nome"""
    if len(ctx.args) != 2 or not ctx.args[1].isdecimal():
        return "Sintaxe inválida: !notes <nome> <dias_atrás>"

    target = await app.state.sessions.players.from_cache_or_sql(name=ctx.args[0])
    if not target:
        return f'"{ctx.args[0]}" não encontrado.'

    days = int(ctx.args[1])

    if days > 365:
        return "Por favor, (não) contate um desenvolvedor para conseguir essa informação."
    elif days <= 0:
        return "Sintaxe inválida: !notes <nome> <dias_atrás>"

    res = await app.state.services.database.fetch_all(
        "SELECT `action`, `msg`, `time`, `from` "
        "FROM `logs` WHERE `to` = :to "
        "AND UNIX_TIMESTAMP(`time`) >= UNIX_TIMESTAMP(NOW()) - :seconds "
        "ORDER BY `time` ASC",
        {"to": target.id, "seconds": days * 86400},
    )

    if not res:
        return f"Nada notável encontrado no {target} nos últimos {days} dias."

    notes = []
    for row in res:
        logger = await app.state.sessions.players.from_cache_or_sql(id=row["from"])
        if not logger:
            continue

        action_str = ACTION_STRINGS.get(row["action"], "Ação desconhecida:")
        time_str = row["time"]
        note = row["msg"]

        notes.append(f"[{time_str}] {action_str} {note} por {logger.name}")

    return "\n".join(notes)


@command(Privileges.MODERATOR, hidden=True)
async def addnote(ctx: Context) -> Optional[str]:
    """Adiciona uma anotação a um jogador especificado pelo nome."""
    if len(ctx.args) < 2:
        return "Sintaxe inválida: !addnote <nome> <anotação ...>"

    target = await app.state.sessions.players.from_cache_or_sql(name=ctx.args[0])
    if not target:
        return f'"{ctx.args[0]}" não encontrado.'

    await app.state.services.database.execute(
        "INSERT INTO logs "
        "(`from`, `to`, `action`, `msg`, `time`) "
        "VALUES (:from, :to, :action, :msg, NOW())",
        {
            "from": ctx.player.id,
            "to": target.id,
            "action": "note",
            "msg": " ".join(ctx.args[1:]),
        },
    )

    return f"Adicionou anotação a {target}."


# some shorthands that can be used as
# reasons in many moderative commands.
SHORTHAND_REASONS = {
    "aa": "tendo seu pedido aceitado",
    "cc": "usando um cliente modificado do osu!",
    "3p": "usando programas de terceiros",
    "rx": "usando programas de terceiros (relax)",
    "tw": "usando programas de terceiros (timewarp)",
    "au": "usando programas de terceiros (auto play)",
}


@command(Privileges.MODERATOR, hidden=True)
async def silence(ctx: Context) -> Optional[str]:
    """Silencia um jogador especificado com uma duração e motivo dado."""
    if len(ctx.args) < 3:
        return "Sintaxe inválida: !silence <nome> <duração> <motivo>"

    target = await app.state.sessions.players.from_cache_or_sql(name=ctx.args[0])
    if not target:
        return f'"{ctx.args[0]}" não encontrado.'

    if target.priv & Privileges.STAFF and not ctx.player.priv & Privileges.DEVELOPER:
        return "Apenas desenvolvedores podem gerir membros da staff."

    duration = timeparse(ctx.args[1])
    if not duration:
        return "Período de tempo inválido."

    reason = " ".join(ctx.args[2:])

    if reason in SHORTHAND_REASONS:
        reason = SHORTHAND_REASONS[reason]

    await target.silence(ctx.player, duration, reason)
    return f"{target} foi silenciado."


@command(Privileges.MODERATOR, hidden=True)
async def unsilence(ctx: Context) -> Optional[str]:
    """Retira a punição de silêncio de um jogador."""
    if len(ctx.args) < 2:
        return "Sintaxe inválida: !unsilence <nome> <reason>"

    target = await app.state.sessions.players.from_cache_or_sql(name=ctx.args[0])
    if not target:
        return f'"{ctx.args[0]}" não encontrado.'

    if not target.silenced:
        return f"{target} não está silenciado."

    if target.priv & Privileges.STAFF and not ctx.player.priv & Privileges.DEVELOPER:
        return "Apenas desenvolvedores podem gerir membros da staff."

    reason = " ".join(ctx.args[1:])

    await target.unsilence(ctx.player, reason)
    return f"{target} não está mais silenciado."


""" Admin commands
# The commands below are relatively dangerous,
# and are generally for managing players.
"""


@command(Privileges.ADMINISTRATOR, aliases=["u"], hidden=True)
async def user(ctx: Context) -> Optional[str]:
    """Retorna informação geral de um dado usuário."""
    if not ctx.args:
        # no username specified, use ctx.player
        player = ctx.player
    else:
        # username given, fetch the player
        player = await app.state.sessions.players.from_cache_or_sql(
            name=" ".join(ctx.args),
        )

        if not player:
            return "Jogador não encontrado."

    priv_list = [
        priv.name
        for priv in Privileges
        if player.priv & priv and bin(priv).count("1") == 1
    ][::-1]
    if player.last_np is not None and time.time() < player.last_np["timeout"]:
        last_np = player.last_np["bmap"].embed
    else:
        last_np = None

    osu_version = player.client_details.osu_version.date if player.online else "Unknown"
    donator_info = (
        f"Sim (ends {timeago.format(player.donor_end)})"
        if player.priv & Privileges.DONATOR != 0
        else "Não"
    )

    return "\n".join(
        (
            f'[{"Bot" if player.bot_client else "Jogador"}] {player.full_name} ({player.id})',
            f"Privilégios: {priv_list}",
            f"Doador: {donator_info}",
            f"Canais: {[c._name for c in player.channels]}",
            f"Logado em: {timeago.format(player.login_time)}",
            f"Ultima interação com o servidor: {timeago.format(player.last_recv_time)}",
            f"Versão do osu!: {osu_version} | Torneio: {player.tourney_client}",
            f"Silenciado: {player.silenced} | Espectando: {player.spectating}",
            f"Ultimo /np: {last_np}",
            f"Pontuação recente: {player.recent_score}",
            f"Partida: {player.match}",
            f"Espectadores: {player.spectators}",
        ),
    )


@command(Privileges.ADMINISTRATOR, hidden=True)
async def restrict(ctx: Context) -> Optional[str]:
    """Restringe a conta de um jogador especificado, e quem o convidou, com um motivo."""
    if len(ctx.args) < 2:
        return "Sintaxe inválida: !restrict <nome> <motivo>"

    # find any user matching (including offline).
    target = await app.state.sessions.players.from_cache_or_sql(name=ctx.args[0])
    if not target:
        return f'"{ctx.args[0]}" não encontrado.'

    if target.priv & Privileges.STAFF and not ctx.player.priv & Privileges.DEVELOPER:
        return "Apenas desenvolvedores podem gerir membros da staff."

    if target.restricted:
        return f"{target} já está restrito!"
    
    if target.id == 1:
        return f"Você não pode banir o (Mamiya Takuji)[https://files.catbox.moe/zjz9qo.png]."

    reason = " ".join(ctx.args[1:])

    if reason in SHORTHAND_REASONS:
        reason = SHORTHAND_REASONS[reason]

    await target.restrict(admin=ctx.player, reason=reason)

    # refresh their client state
    if target.online:
        target.logout()
        
    key_owner = await app.state.services.database.fetch_one(
        f"SELECT user_id_created FROM register_keys WHERE user_id_used = {target.id}"
    )
    
    u = await app.state.sessions.players.from_cache_or_sql(id=dict(key_owner).get("user_id_created"))
    if not u:
        return f"{target} foi restrito."        
    
    if u.restricted or u.id == 1:
        return f"{target} foi restrito."
    
    reason = f"Banido por convidar {target}, que foi banido por: {reason}."
    await u.restrict(admin=ctx.player, reason=reason)
    
    if u.online:
        u.logout()
        
    return f"{target} foi restrito, e {u} também foi por ter convidado."


@command(Privileges.ADMINISTRATOR, hidden=True)
async def unrestrict(ctx: Context) -> Optional[str]:
    """Retira a restrição da conta de um jogador especificado, com um motivo."""
    if len(ctx.args) < 2:
        return "Sintaxe inválida: !unrestrict <nome> <motivo>"

    # find any user matching (including offline).
    target = await app.state.sessions.players.from_cache_or_sql(name=ctx.args[0])
    if not target:
        return f'"{ctx.args[0]}" não encontrado.'

    if target.priv & Privileges.STAFF and not ctx.player.priv & Privileges.DEVELOPER:
        return "Apenas desenvolvedores podem gerir membros da staff."

    if not target.restricted:
        return f"{target} não está restrito!"

    reason = " ".join(ctx.args[1:])

    if reason in SHORTHAND_REASONS:
        reason = SHORTHAND_REASONS[reason]

    await target.unrestrict(ctx.player, reason)

    # refresh their client state
    if target.online:
        target.logout()

    return f"{target} não está mais restrito."


@command(Privileges.ADMINISTRATOR, hidden=True)
async def alert(ctx: Context) -> Optional[str]:
    """Envia uma notificação para todos os jogadores."""
    if len(ctx.args) < 1:
        return "Sintaxe inválida: !alert <mensagem>"

    notif_txt = " ".join(ctx.args)

    app.state.sessions.players.enqueue(app.packets.notification(notif_txt))
    return "Alerta enviado."


@command(Privileges.ADMINISTRATOR, aliases=["alertu"], hidden=True)
async def alertuser(ctx: Context) -> Optional[str]:
    """Envia uma notificação para um jogador especificado por nome."""
    if len(ctx.args) < 2:
        return "Sintaxe inválida: !alertu <name> <msg>"

    target = app.state.sessions.players.get(name=ctx.args[0])
    if not target:
        return "Não foi possível encontrar um usuário com este nome."

    notif_txt = " ".join(ctx.args[1:])

    target.enqueue(app.packets.notification(notif_txt))
    return "Alerta enviado."


# NOTE: this is pretty useless since it doesn't switch anything other
# than the c[e4].ppy.sh domains; it exists on bancho as a tournament
# server switch mechanism, perhaps we could leverage this in the future.
@command(Privileges.ADMINISTRATOR, hidden=True)
async def switchserv(ctx: Context) -> Optional[str]:
    """Muda os terminais internos do seu cliente para um endereço IP especificado."""
    if len(ctx.args) != 1:
        return "Sintaxe inválida: !switch <terminal>"

    new_bancho_ip = ctx.args[0]

    ctx.player.enqueue(app.packets.switch_tournament_server(new_bancho_ip))
    return "Tenha uma boa jornada.."


@command(Privileges.ADMINISTRATOR, aliases=["restart"])
async def shutdown(ctx: Context) -> Union[Optional[str], NoReturn]:
    """Desliga o servidor graciosamente."""
    if ctx.trigger == "reiniciar":
        _signal = signal.SIGUSR1
    else:
        _signal = signal.SIGTERM

    if ctx.args:  # shutdown after a delay
        delay = timeparse(ctx.args[0])
        if not delay:
            return "Período de tempo inválido."

        if delay < 15:
            return "Atraso minímo é 15 segundos."

        if len(ctx.args) > 1:
            # alert all online players of the reboot.
            alert_msg = (
                f"O servidor vai {ctx.trigger} em {ctx.args[0]}.\n\n"
                f'Razão: {" ".join(ctx.args[1:])}'
            )

            app.state.sessions.players.enqueue(app.packets.notification(alert_msg))

        app.state.loop.call_later(delay, os.kill, os.getpid(), _signal)
        return f"Foi enfileirado: {ctx.trigger}."
    else:  # shutdown immediately
        os.kill(os.getpid(), _signal)


""" Developer commands
# The commands below are either dangerous or
# simply not useful for any other roles.
"""


@command(Privileges.DEVELOPER)
async def stealth(ctx: Context) -> Optional[str]:
    """Alterna o modo furtivo do desenvolvedor, que o permite ficar invisível."""
    # NOTE: this command is a large work in progress and currently
    # half works; eventually it will be moved to the Admin level.
    ctx.player.stealth = not ctx.player.stealth

    return f'Modo furtivo {"ativado" if ctx.player.stealth else "desativado"}.'


@command(Privileges.DEVELOPER)
async def recalc(ctx: Context) -> Optional[str]:
    """Recalculate pp for a given map, or all maps."""
    return (
        "Please use tools/recalc.py instead.\n"
        "If you need any support, join our Discord @ https://discord.gg/nmu4hYWE4n."
    )


@command(Privileges.DEVELOPER, hidden=True)
async def debug(ctx: Context) -> Optional[str]:
    """Alterna a configuração de debug do console."""
    app.settings.DEBUG = not app.settings.DEBUG
    return f"{'Ativado' if app.settings.DEBUG else 'Desativado'}."


# NOTE: these commands will likely be removed
#       with the addition of a good frontend.
str_priv_dict = {
    "normal": Privileges.UNRESTRICTED,
    "verified": Privileges.VERIFIED,
    "whitelisted": Privileges.WHITELISTED,
    "supporter": Privileges.SUPPORTER,
    "premium": Privileges.PREMIUM,
    "alumni": Privileges.ALUMNI,
    "tournament": Privileges.TOURNEY_MANAGER,
    "nominator": Privileges.NOMINATOR,
    "mod": Privileges.MODERATOR,
    "admin": Privileges.ADMINISTRATOR,
    "developer": Privileges.DEVELOPER,
}


@command(Privileges.DEVELOPER, hidden=True)
async def addpriv(ctx: Context) -> Optional[str]:
    """Define privilégios para um jogador especificado (por nome)."""
    if len(ctx.args) < 2:
        return "Sintaxe inválida: !addpriv <nome> <cargo1 cargo2 cargo3 ...>"

    bits = Privileges(0)

    for m in [m.lower() for m in ctx.args[1:]]:
        if m not in str_priv_dict:
            return f"Não encontrado: {m}."

        bits |= str_priv_dict[m]

    target = await app.state.sessions.players.from_cache_or_sql(name=ctx.args[0])
    if not target:
        return "Não foi possível encontrar o usuário."

    if bits & Privileges.DONATOR != 0:
        return "Por favor use o comando !givedonator para atribuir os privilégios de doador aos jogadores."

    await target.add_privs(bits)
    return f"Os privilégios de {target} foram atualizados."


@command(Privileges.DEVELOPER, hidden=True)
async def rmpriv(ctx: Context) -> Optional[str]:
    """Remove privilégios para um jogador especificado (por nome)."""
    if len(ctx.args) < 2:
        return "Sintaxe inválida: !rmpriv <name> <cargo1 cargo2 cargo3 ...>"

    bits = Privileges(0)

    for m in [m.lower() for m in ctx.args[1:]]:
        if m not in str_priv_dict:
            return f"Não encontrado: {m}."

        bits |= str_priv_dict[m]

    target = await app.state.sessions.players.from_cache_or_sql(name=ctx.args[0])
    if not target:
        return "Não foi possível encontrar o usuário."

    await target.remove_privs(bits)

    if bits & Privileges.DONATOR != 0:
        target.donor_end = 0
        await app.state.services.database.execute(
            "UPDATE users SET donor_end = 0 WHERE id = :user_id",
            {"user_id": target.id},
        )

    return f"Os privilégios de {target} foram atualizados."


@command(Privileges.DEVELOPER, hidden=True)
async def givedonator(ctx: Context) -> Optional[str]:
    """Dá o cargo de doador para um jogador especificado (por nome) por uma quantidade específica de tempo, como '3h5m'."""
    if len(ctx.args) < 2:
        return "Sintaxe inválida: !givedonator <nome> <duração>"

    target = await app.state.sessions.players.from_cache_or_sql(name=ctx.args[0])
    if not target:
        return "Não foi possível encontrar o usuário."

    timespan = timeparse(ctx.args[1])
    if not timespan:
        return "Período de tempo inválido."

    if target.donor_end < time.time():
        timespan += int(time.time())
    else:
        timespan += target.donor_end

    target.donor_end = timespan
    await app.state.services.database.execute(
        "UPDATE users SET donor_end = :end WHERE id = :user_id",
        {"end": timespan, "user_id": target.id},
    )

    await target.add_privs(Privileges.SUPPORTER)

    return f"O {target} recebeu o cargo de doador que irá expirar em {ctx.args[1]}."


@command(Privileges.DEVELOPER)
async def wipemap(ctx: Context) -> Optional[str]:
    # (intentionally no docstring)
    if ctx.args:
        return "Sintaxe inválida: !wipemap"

    if ctx.player.last_np is None or time.time() >= ctx.player.last_np["timeout"]:
        return "Por favor, dê /np em um mapa antes."

    map_md5 = ctx.player.last_np["bmap"].md5

    # delete scores from all tables
    await app.state.services.database.execute(
        "DELETE FROM scores WHERE map_md5 = :map_md5",
        {"map_md5": map_md5},
    )

    return "Pontuações apagadas."


@command(Privileges.DEVELOPER, hidden=True)
async def menu(ctx: Context) -> Optional[str]:
    """Comando temporário para ilustrar a ideia de opção de menu."""
    ctx.player.send_current_menu()

    return None


@command(Privileges.DEVELOPER, aliases=["re"])
async def reload(ctx: Context) -> Optional[str]:
    """Recarrega um módulo do python."""
    if len(ctx.args) != 1:
        return "Sintaxe inválida: !reload <módulo>"

    parent, *children = ctx.args[0].split(".")

    try:
        mod = __import__(parent)
    except ModuleNotFoundError:
        return "Modulo não encontrado."

    try:
        for child in children:
            mod = getattr(mod, child)
    except AttributeError:
        return f"Falhou em {child}."  # type: ignore

    try:
        mod = importlib.reload(mod)
    except TypeError as exc:
        return f"{exc.args[0]}."

    return f"Recarregado {mod.__name__}"


@command(Privileges.UNRESTRICTED)
async def server(ctx: Context) -> Optional[str]:
    """Recuperar informações sobre a performance do servidor"""

    build_str = f"bancho.py v{app.settings.VERSION} ({app.settings.DOMAIN})"

    # get info about this process
    proc = psutil.Process(os.getpid())
    uptime = int(time.time() - proc.create_time())

    # get info about our cpu
    with open("/proc/cpuinfo") as f:
        header = "model name\t: "
        trailer = "\n"

        model_names = Counter(
            line[len(header) : -len(trailer)]
            for line in f.readlines()
            if line.startswith("model name")
        )

    # list of all cpus installed with thread count
    cpus_info = " | ".join(f"{v}x {k}" for k, v in model_names.most_common())

    # get system-wide ram usage
    sys_ram = psutil.virtual_memory()

    # output ram usage as `{bancho_used}MB / {sys_used}MB / {sys_total}MB`
    bancho_ram = proc.memory_info()[0]
    ram_values = (bancho_ram, sys_ram.used, sys_ram.total)
    ram_info = " / ".join([f"{v // 1024 ** 2}MB" for v in ram_values])

    # current state of settings
    mirror_search_url = urlparse(app.settings.MIRROR_SEARCH_ENDPOINT).netloc
    mirror_download_url = urlparse(app.settings.MIRROR_DOWNLOAD_ENDPOINT).netloc
    using_osuapi = app.settings.OSU_API_KEY != ""
    advanced_mode = app.settings.DEVELOPER_MODE
    auto_logging = app.settings.AUTOMATICALLY_REPORT_PROBLEMS

    # package versioning info
    # divide up pkg versions, 3 displayed per line, e.g.
    # aiohttp v3.6.3 | aiomysql v0.0.21 | bcrypt v3.2.0
    # cmyui v1.7.3 | datadog v0.40.1 | geoip2 v4.1.0
    # maniera v1.0.0 | mysql-connector-python v8.0.23 | orjson v3.5.1
    # psutil v5.8.0 | py3rijndael v0.3.3 | uvloop v0.15.2
    requirements = []

    for dist in importlib.metadata.distributions():
        requirements.append(f"{dist.name} v{dist.version}")  # type: ignore
    requirements.sort(key=lambda x: x.casefold())

    requirements_info = "\n".join(
        " | ".join(section)
        for section in (requirements[i : i + 3] for i in range(0, len(requirements), 3))
    )

    return "\n".join(
        (
            f"{build_str} | tempo de atividade: {seconds_readable(uptime)}",
            f"cpu(s): {cpus_info}",
            f"ram: {ram_info}",
            f"espelho de pesquisa: {mirror_search_url} | espelho de download: {mirror_download_url}",
            f"conexão com a osu!api: {using_osuapi}",
            f"modo avançado: {advanced_mode} | logging automático: {auto_logging}",
            "",
            "requisitos",
            requirements_info,
        ),
    )


if app.settings.DEVELOPER_MODE:
    """Comandos avançados (e potencialmente perigosos)."""

    # NOTE: some of these commands are potentially dangerous, and only
    # really intended for advanced users looking for access to lower level
    # utilities. Some may give direct access to utilties that could perform
    # harmful tasks to the underlying machine, so use at your own risk.

    from sys import modules as installed_mods

    __py_namespace: dict[str, Any] = globals() | {
        mod: __import__(mod)
        for mod in (
            "asyncio",
            "dis",
            "os",
            "sys",
            "struct",
            "discord",
            "datetime",
            "time",
            "inspect",
            "math",
            "importlib",
        )
        if mod in installed_mods
    }

    @command(Privileges.DEVELOPER)
    async def py(ctx: Context) -> Optional[str]:
        """Permite acesso assíncrono para o interpretador python."""
        # This can be very good for getting used to bancho.py's API; just look
        # around the codebase and find things to play with in your server.
        # Ex: !py return (await app.state.sessions.players.get(name='cmyui')).status.action
        if not ctx.args:
            return "owo"

        # turn our input args into a coroutine definition string.
        definition = "\n ".join(["async def __py(ctx):", " ".join(ctx.args)])

        try:  # def __py(ctx)
            exec(definition, __py_namespace)  # add to namespace
            ret = await __py_namespace["__py"](ctx)  # await it's return
        except Exception as exc:  # return exception in osu! chat
            ret = f"{exc.__class__}: {exc}"

        if "__py" in __py_namespace:
            del __py_namespace["__py"]

        # TODO: perhaps size checks?

        if not isinstance(ret, str):
            ret = pprint.pformat(ret, compact=True)

        return ret


""" Multiplayer commands
# The commands below for multiplayer match management.
# Most commands are open to player usage.
"""

R = TypeVar("R", bound=Optional[str])


def ensure_match(
    f: Callable[[Context, Match], Awaitable[Optional[R]]],
) -> Callable[[Context], Awaitable[Optional[R]]]:
    @wraps(f)
    async def wrapper(ctx: Context) -> Optional[R]:
        match = ctx.player.match

        # multi set is a bit of a special case,
        # as we do some additional checks.
        if match is None:
            # player not in a match
            return None

        if ctx.recipient is not match.chat:
            # message not in match channel
            return None

        if f is not mp_help and (
            ctx.player not in match.refs
            and not ctx.player.priv & Privileges.TOURNEY_MANAGER
        ):
            # doesn't have privs to use !mp commands (allow help).
            return None

        return await f(ctx, match)

    return wrapper


@mp_commands.add(Privileges.UNRESTRICTED, aliases=["h"])
@ensure_match
async def mp_help(ctx: Context, match: Match) -> Optional[str]:
    """Mostra todas as salas de multijogador visíeveis que um jogador pode acessar."""
    prefix = app.settings.COMMAND_PREFIX
    cmds = []

    for cmd in mp_commands.commands:
        if not cmd.doc or ctx.player.priv & cmd.priv != cmd.priv:
            # no doc, or insufficient permissions.
            continue

        cmds.append(f"{prefix}mp {cmd.triggers[0]}: {cmd.doc}")

    return "\n".join(cmds)


@mp_commands.add(Privileges.UNRESTRICTED, aliases=["st"])
@ensure_match
async def mp_start(ctx: Context, match: Match) -> Optional[str]:
    """Começa a partida multijogador, com quaisquer jogadores prontos."""
    if len(ctx.args) > 1:
        return "Sintaxe inválida: !mp start <force/segundos>"

    # this command can be used in a few different ways;
    # !mp start: start the match now (make sure all players are ready)
    # !mp start force: start the match now (don't check for ready)
    # !mp start N: start the match in N seconds (don't check for ready)
    # !mp start cancel: cancel the current match start timer

    if not ctx.args:
        # !mp start
        if match.starting is not None:
            time_remaining = int(match.starting["time"] - time.time())
            return f"Partida começando em {time_remaining} segundos."

        if any([s.status == SlotStatus.not_ready for s in match.slots]):
            return "Nem todos os jogadores estão prontos. (`!mp start force` para ignorar)."
    else:
        if ctx.args[0].isdecimal():
            # !mp start N
            if match.starting is not None:
                time_remaining = int(match.starting["time"] - time.time())
                return f"Partida começando em {time_remaining} segundos."

            # !mp start <seconds>
            duration = int(ctx.args[0])
            if not 0 < duration <= 300:
                return "Tempo selecionado deve estar entre 1-300 segundos."

            def _start() -> None:
                """Remove any pending timers & start the match."""
                # remove start & alert timers
                match.starting = None

                # make sure player didn't leave the
                # match since queueing this start lol...
                if ctx.player not in {slot.player for slot in match.slots}:
                    match.chat.send_bot("Jogador saiu da partida? (cancelado)")
                    return

                match.start()
                match.chat.send_bot("Iniciando partida.")

            def _alert_start(t: int) -> None:
                """Alert the match of the impending start."""
                match.chat.send_bot(f"Partida começando em {t} seconds.")

            # add timers to our match object,
            # so we can cancel them if needed.
            match.starting = {
                "start": app.state.loop.call_later(duration, _start),
                "alerts": [
                    app.state.loop.call_later(duration - t, lambda t=t: _alert_start(t))
                    for t in (60, 30, 10, 5, 4, 3, 2, 1)
                    if t < duration
                ],
                "time": time.time() + duration,
            }

            return f"Partida vai começar em {duration} segundos."
        elif ctx.args[0] in ("cancel", "c"):
            # !mp start cancel
            if match.starting is None:
                return "Temporizador da partida não está ativado!"

            match.starting["start"].cancel()
            for alert in match.starting["alerts"]:
                alert.cancel()

            match.starting = None

            return "Temporizador cancelado."
        elif ctx.args[0] not in ("force", "f"):
            return "Sintaxe inválida: !mp start <force/segundos>"
        # !mp start force simply passes through

    match.start()
    return "Boa sorte!"


@mp_commands.add(Privileges.UNRESTRICTED, aliases=["a"])
@ensure_match
async def mp_abort(ctx: Context, match: Match) -> Optional[str]:
    """Aborta a partida em progresso no modo multijogador."""
    if not match.in_progress:
        return "Abortar o quê?"

    match.unready_players(expected=SlotStatus.playing)

    match.in_progress = False
    match.enqueue(app.packets.match_abort())
    match.enqueue_state()
    return "Partida abortada."


@mp_commands.add(Privileges.UNRESTRICTED)
@ensure_match
async def mp_map(ctx: Context, match: Match) -> Optional[str]:
    """Define o mapa atual da partida por id."""
    if len(ctx.args) != 1 or not ctx.args[0].isdecimal():
        return "Sintaxe inválida: !mp map <beatmapid>"

    map_id = int(ctx.args[0])

    if map_id == match.map_id:
        return "O mapa já está selecionado."

    bmap = await Beatmap.from_bid(map_id)
    if not bmap:
        return "O mapa não foi encontrado."

    match.map_id = bmap.id
    match.map_md5 = bmap.md5
    match.map_name = bmap.full_name

    match.mode = bmap.mode

    match.enqueue_state()
    return f"Selecionado: {bmap.embed}."


@mp_commands.add(Privileges.UNRESTRICTED)
@ensure_match
async def mp_mods(ctx: Context, match: Match) -> Optional[str]:
    """Define os mods atuais da partida, na forma de string."""
    if len(ctx.args) != 1 or len(ctx.args[0]) % 2 != 0:
        return "Sintaxe inválida: !mp mods <mods>"

    mods = Mods.from_modstr(ctx.args[0])
    mods = mods.filter_invalid_combos(match.mode.as_vanilla)

    if match.freemods:
        if ctx.player is match.host:
            # allow host to set speed-changing mods.
            match.mods = mods & SPEED_CHANGING_MODS

        # set slot mods
        slot = match.get_slot(ctx.player)
        assert slot is not None

        slot.mods = mods & ~SPEED_CHANGING_MODS
    else:
        # not freemods, set match mods.
        match.mods = mods

    match.enqueue_state()
    return "Mods da partida atualizados."


@mp_commands.add(Privileges.UNRESTRICTED, aliases=["fm", "fmods"])
@ensure_match
async def mp_freemods(ctx: Context, match: Match) -> Optional[str]:
    """Alterna o estado dos freemods para a partida."""
    if len(ctx.args) != 1 or ctx.args[0] not in ("on", "off"):
        return "Sintaxe inválida: !mp freemods <on/off>"

    if ctx.args[0] == "on":
        # central mods -> all players mods.
        match.freemods = True

        for s in match.slots:
            if s.player is not None:
                # the slot takes any non-speed
                # changing mods from the match.
                s.mods = match.mods & ~SPEED_CHANGING_MODS

        match.mods &= SPEED_CHANGING_MODS
    else:
        # host mods -> central mods.
        match.freemods = False

        host_slot = match.get_host_slot()
        assert host_slot is not None

        # the match keeps any speed-changing mods,
        # and also takes any mods the host has enabled.
        match.mods &= SPEED_CHANGING_MODS
        match.mods |= host_slot.mods

        for s in match.slots:
            if s.player is not None:
                s.mods = Mods.NOMOD

    match.enqueue_state()
    return "Mods livres atualizado."


@mp_commands.add(Privileges.UNRESTRICTED)
@ensure_match
async def mp_host(ctx: Context, match: Match) -> Optional[str]:
    """Define o anfitrião da partida por nome."""
    if len(ctx.args) != 1:
        return "Sintaxe inválida: !mp host <nome>"

    target = app.state.sessions.players.get(name=ctx.args[0])
    if not target:
        return "Não foi possível encontrar um usuário por este nome."

    if target is match.host:
        return "Esta pessoa já é o anfitrião!"

    if target not in {slot.player for slot in match.slots}:
        return "Jogador não foi encontrado na partida."

    match.host_id = target.id

    match.host.enqueue(app.packets.match_transfer_host())
    match.enqueue_state(lobby=True)
    return "Anfitrião da partida foi atualizado."


@mp_commands.add(Privileges.UNRESTRICTED)
@ensure_match
async def mp_randpw(ctx: Context, match: Match) -> Optional[str]:
    """Randomiza a senha atual da partida."""
    match.passwd = secrets.token_hex(8)
    return "Senha da partida foi randomizada."


@mp_commands.add(Privileges.UNRESTRICTED, aliases=["inv"])
@ensure_match
async def mp_invite(ctx: Context, match: Match) -> Optional[str]:
    """Convida um jogador para a partida pelo nome"""
    if len(ctx.args) != 1:
        return "Sintaxe inválida: !mp invite <nome>"

    target = app.state.sessions.players.get(name=ctx.args[0])
    if not target:
        return "Não foi possível encontrar um usuário por este nome."

    if target is app.state.sessions.bot:
        return "Estou muito ocupado!"

    if target is ctx.player:
        return "Não tem como convidar a si mesmo!"

    target.enqueue(app.packets.match_invite(ctx.player, target.name))
    return f"Convidou {target} para a partida."


@mp_commands.add(Privileges.UNRESTRICTED)
@ensure_match
async def mp_addref(ctx: Context, match: Match) -> Optional[str]:
    """Adiciona um juíz para a partida pelo nome."""
    if len(ctx.args) != 1:
        return "Sintaxe inválida: !mp addref <nome>"

    target = app.state.sessions.players.get(name=ctx.args[0])
    if not target:
        return "Não foi possível encontrar um usuário por este nome."

    if target not in {slot.player for slot in match.slots}:
        return "Usuário deve estar na partida atual!"

    if target in match.refs:
        return f"{target} já é um juíz da partida!"

    match._refs.add(target)
    return f"{target.name} adicionado aos juízes da partida."


@mp_commands.add(Privileges.UNRESTRICTED)
@ensure_match
async def mp_rmref(ctx: Context, match: Match) -> Optional[str]:
    """Remove um juíz da partida pelo nome."""
    if len(ctx.args) != 1:
        return "Sintaxe inválida: !mp addref <nome>"

    target = app.state.sessions.players.get(name=ctx.args[0])
    if not target:
        return "Não foi possível encontrar um usuário por este nome."

    if target not in match.refs:
        return f"{target} não é um juíz da partida!"

    if target is match.host:
        return "O anfitrião sempre é o juíz!"

    match._refs.remove(target)
    return f"{target.name} não é mais um juíz da partida."


@mp_commands.add(Privileges.UNRESTRICTED)
@ensure_match
async def mp_listref(ctx: Context, match: Match) -> Optional[str]:
    """Lista todos os juízes da partida"""
    return ", ".join(map(str, match.refs)) + "."


@mp_commands.add(Privileges.UNRESTRICTED)
@ensure_match
async def mp_lock(ctx: Context, match: Match) -> Optional[str]:
    """Tranca todas as vagas vazias da partida."""
    for slot in match.slots:
        if slot.status == SlotStatus.open:
            slot.status = SlotStatus.locked

    match.enqueue_state()
    return "Todas as vagas vazias foram bloqueadas."


@mp_commands.add(Privileges.UNRESTRICTED)
@ensure_match
async def mp_unlock(ctx: Context, match: Match) -> Optional[str]:
    """Destranca todas as vagas trancadas da partida."""
    for slot in match.slots:
        if slot.status == SlotStatus.locked:
            slot.status = SlotStatus.open

    match.enqueue_state()
    return "Todas as vagas trancadas foram destrancadas."


@mp_commands.add(Privileges.UNRESTRICTED)
@ensure_match
async def mp_teams(ctx: Context, match: Match) -> Optional[str]:
    """Muda o tipo de time da partida."""
    if len(ctx.args) != 1:
        return "Sintaxe inválida: !mp teams <type>"

    team_type = ctx.args[0]

    if team_type in ("ffa", "freeforall", "head-to-head"):
        match.team_type = MatchTeamTypes.head_to_head
    elif team_type in ("tag", "coop", "co-op", "tag-coop"):
        match.team_type = MatchTeamTypes.tag_coop
    elif team_type in ("teams", "team-vs", "teams-vs"):
        match.team_type = MatchTeamTypes.team_vs
    elif team_type in ("tag-teams", "tag-team-vs", "tag-teams-vs"):
        match.team_type = MatchTeamTypes.tag_team_vs
    else:
        return "Tipo de time desconhecido. (ffa, tag, teams, tag-teams)"

    # find the new appropriate default team.
    # defaults are (ffa: neutral, teams: red).
    if match.team_type in (MatchTeamTypes.head_to_head, MatchTeamTypes.tag_coop):
        new_t = MatchTeams.neutral
    else:
        new_t = MatchTeams.red

    # change each active slots team to
    # fit the correspoding team type.
    for s in match.slots:
        if s.player is not None:
            s.team = new_t

    if match.is_scrimming:
        # reset score if scrimming.
        match.reset_scrim()

    match.enqueue_state()
    return "O tipo de time foi atualizado."


@mp_commands.add(Privileges.UNRESTRICTED, aliases=["cond"])
@ensure_match
async def mp_condition(ctx: Context, match: Match) -> Optional[str]:
    """Muda a condição de vitória da partida."""
    if len(ctx.args) != 1:
        return "Sintaxe inválida: !mp condition <tipo>"

    cond = ctx.args[0]

    if cond == "pp":
        # special case - pp can't actually be used as an ingame
        # win condition, but bancho.py allows it to be passed into
        # this command during a scrims to use pp as a win cond.
        if not match.is_scrimming:
            return "PP só pode ser usado como uma condição de vitória durante amistosos."
        if match.use_pp_scoring:
            return "Pontuação por PP já está ativada."

        match.use_pp_scoring = True
    else:
        if match.use_pp_scoring:
            match.use_pp_scoring = False

        if cond == "score":
            match.win_condition = MatchWinConditions.score
        elif cond in ("accuracy", "acc"):
            match.win_condition = MatchWinConditions.accuracy
        elif cond == "combo":
            match.win_condition = MatchWinConditions.combo
        elif cond in ("scorev2", "v2"):
            match.win_condition = MatchWinConditions.scorev2
        else:
            return "Condição de vitória inválida. (score, acc, combo, scorev2, *pp)"

    match.enqueue_state(lobby=False)
    return "Condição de vitória atualizada."


@mp_commands.add(Privileges.UNRESTRICTED, aliases=["autoref"])
@ensure_match
async def mp_scrim(ctx: Context, match: Match) -> Optional[str]:
    """Começa um amistoso na partida atual."""
    r_match = regexes.BEST_OF.fullmatch(ctx.args[0])
    if len(ctx.args) != 1 or not r_match:
        return "Sintaxe inválida: !mp scrim <md#>"

    best_of = int(r_match[1])
    if not 0 <= best_of < 16:
        return "Melhor deve estar entre 0-15."

    winning_pts = (best_of // 2) + 1

    if winning_pts != 0:
        # setting to real num
        if match.is_scrimming:
            return "Amistoso já está acontecendo!"

        if best_of % 2 == 0:
            return "MD# deve ser um número ímpar!"

        match.is_scrimming = True
        msg = (
            f"Um amistoso foi iniciado por {ctx.player.name}; "
            f"primeiro a conseguir {winning_pts} pontos, vence. Boa sorte!"
        )
    else:
        # setting to 0
        if not match.is_scrimming:
            return "Amistoso não está acontecendo!"

        match.is_scrimming = False
        match.reset_scrim()
        msg = "Amistoso cancelado."

    match.winning_pts = winning_pts
    return msg


@mp_commands.add(Privileges.UNRESTRICTED, aliases=["end"])
@ensure_match
async def mp_endscrim(ctx: Context, match: Match) -> Optional[str]:
    """Termina o amistoso atual da partida."""
    if not match.is_scrimming:
        return "Não está acontecendo o amistoso."

    match.is_scrimming = False
    match.reset_scrim()
    return "Amistoso terminou."  # TODO: final score (get_score method?)


@mp_commands.add(Privileges.UNRESTRICTED, aliases=["rm"])
@ensure_match
async def mp_rematch(ctx: Context, match: Match) -> Optional[str]:
    """Reinicia o amistoso, ou desfaz o último ponto da partida."""
    if ctx.args:
        return "Sintaxe inválida: !mp rematch"

    if ctx.player is not match.host:
        return "Apenas disponível para o anfitrião."

    if not match.is_scrimming:
        if match.winning_pts == 0:
            msg = "Não está em amistoso; para começar um, use !mp scrim."
        else:
            # re-start scrimming with old points
            match.is_scrimming = True
            msg = (
                f"Uma revanche foi iniciada por {ctx.player.name}; "
                f"o primeiro a fazer {match.winning_pts} pontos, vence. Boa sorte!"
            )
    else:
        # reset the last match point awarded
        if not match.winners:
            return "Nenhum ponto foi ganho ainda."

        recent_winner = match.winners[-1]
        if recent_winner is None:
            return "O último ponto foi um empate."

        match.match_points[recent_winner] -= 1  # TODO: team name
        match.winners.pop()

        msg = f"Um ponto foi deduzido de {recent_winner}."

    return msg


@mp_commands.add(Privileges.ADMINISTRATOR, aliases=["f"], hidden=True)
@ensure_match
async def mp_force(ctx: Context, match: Match) -> Optional[str]:
    """Força um jogador na partida pelo nome."""
    # NOTE: this overrides any limits such as silences or passwd.
    if len(ctx.args) != 1:
        return "Sintaxe inválida: !mp force <name>"

    target = app.state.sessions.players.get(name=ctx.args[0])
    if not target:
        return "Não foi possível encontrar um usuário por este nome."

    target.join_match(match, match.passwd)
    return "Bem-vindo."


# mappool-related mp commands


@mp_commands.add(Privileges.UNRESTRICTED, aliases=["lp"])
@ensure_match
async def mp_loadpool(ctx: Context, match: Match) -> Optional[str]:
    """Carrega uma mappool na partida atual."""
    if len(ctx.args) != 1:
        return "Sintaxe inválida: !mp loadpool <nome>"

    if ctx.player is not match.host:
        return "Apenas disponível para o anfitrião."

    name = ctx.args[0]

    pool = app.state.sessions.pools.get_by_name(name)
    if not pool:
        return "Não existe uma mappool com esse nome!"

    if match.pool is pool:
        return f"{pool!r} já está selecionada!"

    match.pool = pool
    return f"{pool!r} selecionada."


@mp_commands.add(Privileges.UNRESTRICTED, aliases=["ulp"])
@ensure_match
async def mp_unloadpool(ctx: Context, match: Match) -> Optional[str]:
    """Descarrega a mappool da partida atual."""
    if ctx.args:
        return "Sintaxe inválida: !mp unloadpool"

    if ctx.player is not match.host:
        return "Apenas disponível para o anfitrião."

    if not match.pool:
        return "Não há mappool selecionada na partida."

    match.pool = None
    return "Mappool descarregada."


@mp_commands.add(Privileges.UNRESTRICTED)
@ensure_match
async def mp_ban(ctx: Context, match: Match) -> Optional[str]:
    """Bane um mapa da mappool carregada na partida."""
    if len(ctx.args) != 1:
        return "Sintaxe inválida: !mp ban <ban>"

    if not match.pool:
        return "Não há uma mappool carregada na partida."

    mods_slot = ctx.args[0]

    # separate mods & slot
    r_match = regexes.MAPPOOL_PICK.fullmatch(mods_slot)
    if not r_match:
        return "Sintaxe inválida de escolha; exemplo correto: HD2"

    # not calling mods.filter_invalid_combos here intentionally.
    mods = Mods.from_modstr(r_match[1])
    slot = int(r_match[2])

    if (mods, slot) not in match.pool.maps:
        return f"Found no {mods_slot} pick in the pool."

    if (mods, slot) in match.bans:
        return "Esse mapa já foi banido!"

    match.bans.add((mods, slot))
    return f"{mods_slot} banido."


@mp_commands.add(Privileges.UNRESTRICTED)
@ensure_match
async def mp_unban(ctx: Context, match: Match) -> Optional[str]:
    """Desbane um mapa da mappool carregada na partida."""
    if len(ctx.args) != 1:
        return "Sintaxe inválida: !mp unban <ban>"

    if not match.pool:
        return "Não há uma mappool carregada na partida."

    mods_slot = ctx.args[0]

    # separate mods & slot
    r_match = regexes.MAPPOOL_PICK.fullmatch(mods_slot)
    if not r_match:
        return "Sintaxe inválida de escolha; exemplo correto: HD2"

    # not calling mods.filter_invalid_combos here intentionally.
    mods = Mods.from_modstr(r_match[1])
    slot = int(r_match[2])

    if (mods, slot) not in match.pool.maps:
        return f"Não existe {mods_slot} na mappool."

    if (mods, slot) not in match.bans:
        return "Essa escolha não está banida atualmente."

    match.bans.remove((mods, slot))
    return f"{mods_slot} desbanido."


@mp_commands.add(Privileges.UNRESTRICTED)
@ensure_match
async def mp_pick(ctx: Context, match: Match) -> Optional[str]:
    """Escolhe um mapa da mappool carregada na partida."""
    if len(ctx.args) != 1:
        return "Sintaxe inválida: !mp pick <pick>"

    if not match.pool:
        return "Não há uma mappool carregada na partida."

    mods_slot = ctx.args[0]

    # separate mods & slot
    r_match = regexes.MAPPOOL_PICK.fullmatch(mods_slot)
    if not r_match:
        return "Sintaxe inválida de escolha; exemplo correto: HD2"

    # not calling mods.filter_invalid_combos here intentionally.
    mods = Mods.from_modstr(r_match[1])
    slot = int(r_match[2])

    if (mods, slot) not in match.pool.maps:
        return f"Não existe {mods_slot} na mappool."

    if (mods, slot) in match.bans:
        return f"{mods_slot} foi banido, e você não pode escolher."

    # update match beatmap to the picked map.
    bmap = match.pool.maps[(mods, slot)]
    match.map_md5 = bmap.md5
    match.map_id = bmap.id
    match.map_name = bmap.full_name

    # TODO: some kind of abstraction allowing
    # for something like !mp pick fm.
    if match.freemods:
        # if freemods are enabled, disable them.
        match.freemods = False

        for s in match.slots:
            if s.player is not None:
                s.mods = Mods.NOMOD

    # update match mods to the picked map.
    match.mods = mods

    match.enqueue_state()

    return f"Picked {bmap.embed}. ({mods_slot})"


""" Mappool management commands
# The commands below are for event managers
# and tournament hosts/referees to help automate
# tedious processes of running tournaments.
"""


@pool_commands.add(Privileges.TOURNEY_MANAGER, aliases=["h"], hidden=True)
async def pool_help(ctx: Context) -> Optional[str]:
    """Mostra todos os comandos de mappool que um jogador pode usar."""
    prefix = app.settings.COMMAND_PREFIX
    cmds = []

    for cmd in pool_commands.commands:
        if not cmd.doc or ctx.player.priv & cmd.priv != cmd.priv:
            # no doc, or insufficient permissions.
            continue

        cmds.append(f"{prefix}pool {cmd.triggers[0]}: {cmd.doc}")

    return "\n".join(cmds)


@pool_commands.add(Privileges.TOURNEY_MANAGER, aliases=["c"], hidden=True)
async def pool_create(ctx: Context) -> Optional[str]:
    """Adiciona uma mappool ao banco de dados."""
    if len(ctx.args) != 1:
        return "Sintaxe inválida: !pool create <nome>"

    name = ctx.args[0]

    if app.state.sessions.pools.get_by_name(name):
        return "Uma pool com esse nome já existe!"

    # insert pool into db
    await app.state.services.database.execute(
        "INSERT INTO tourney_pools "
        "(name, created_at, created_by) "
        "VALUES (:name, NOW(), :user_id)",
        {"name": name, "user_id": ctx.player.id},
    )

    # add to cache (get from sql for id & time)
    row = await app.state.services.database.fetch_one(
        "SELECT * FROM tourney_pools WHERE name = :name",
        {"name": name},
    )
    assert row is not None

    row = dict(row)  # make mutable copy

    pool_creator = await app.state.sessions.players.from_cache_or_sql(
        id=row["created_by"],
    )
    assert pool_creator is not None

    app.state.sessions.pools.append(
        MapPool(
            id=row["id"],
            name=row["name"],
            created_at=row["created_at"],
            created_by=pool_creator,
        ),
    )

    return f"{name} criada."


@pool_commands.add(Privileges.TOURNEY_MANAGER, aliases=["del", "d"], hidden=True)
async def pool_delete(ctx: Context) -> Optional[str]:
    """Remove uma mappool do banco de dados."""
    if len(ctx.args) != 1:
        return "Sintaxe inválida: !pool delete <nome>"

    name = ctx.args[0]

    pool = app.state.sessions.pools.get_by_name(name)
    if not pool:
        return "Não foi possível achar uma mappool com esse nome."

    # delete from db
    await app.state.services.database.execute(
        "DELETE FROM tourney_pools WHERE id = :pool_id",
        {"pool_id": pool.id},
    )

    await app.state.services.database.execute(
        "DELETE FROM tourney_pool_maps WHERE pool_id = :pool_id",
        {"pool_id": pool.id},
    )

    # remove from cache
    app.state.sessions.pools.remove(pool)

    return f"{name} deletada."


@pool_commands.add(Privileges.TOURNEY_MANAGER, aliases=["a"], hidden=True)
async def pool_add(ctx: Context) -> Optional[str]:
    """Adiciona um novo mapa para uma mappool no banco de dados"""
    if len(ctx.args) != 2:
        return "Sintaxe inválida: !pool add <nome> <escolha>"

    if ctx.player.last_np is None or time.time() >= ctx.player.last_np["timeout"]:
        return "Por favor, dê /np em um mapa antes."

    name, mods_slot = ctx.args
    mods_slot = mods_slot.upper()  # ocd
    bmap = ctx.player.last_np["bmap"]

    # separate mods & slot
    r_match = regexes.MAPPOOL_PICK.fullmatch(mods_slot)
    if not r_match:
        return "Sintaxe inválida de escolha; exemplo correto: HD2"

    if len(r_match[1]) % 2 != 0:
        return "Mods inválidos."

    # not calling mods.filter_invalid_combos here intentionally.
    mods = Mods.from_modstr(r_match[1])
    slot = int(r_match[2])

    pool = app.state.sessions.pools.get_by_name(name)
    if not pool:
        return "Não foi possível achar uma mappool com esse nome."

    if (mods, slot) in pool.maps:
        return f"{mods_slot} is already {pool.maps[(mods, slot)].embed}!"

    if bmap in pool.maps.values():
        return "O mapa já está na mappool."

    # insert into db
    await app.state.services.database.execute(
        "INSERT INTO tourney_pool_maps "
        "(map_id, pool_id, mods, slot) "
        "VALUES (:map_id, :pool_id, :mods, :slot)",
        {"map_id": bmap.id, "pool_id": pool.id, "mods": mods, "slot": slot},
    )

    # add to cache
    pool.maps[(mods, slot)] = bmap

    return f"{bmap.embed} adicionado a {name} como {mods_slot}."


@pool_commands.add(Privileges.TOURNEY_MANAGER, aliases=["rm", "r"], hidden=True)
async def pool_remove(ctx: Context) -> Optional[str]:
    """Remove um mapa da mappool no banco de dados."""
    if len(ctx.args) != 2:
        return "Sintaxe inválida: !pool remove <nome> <escolha>"

    name, mods_slot = ctx.args
    mods_slot = mods_slot.upper()  # ocd

    # separate mods & slot
    r_match = regexes.MAPPOOL_PICK.fullmatch(mods_slot)
    if not r_match:
        return "Sintaxe inválida de escolha; exemplo correto: HD2"

    # not calling mods.filter_invalid_combos here intentionally.
    mods = Mods.from_modstr(r_match[1])
    slot = int(r_match[2])

    pool = app.state.sessions.pools.get_by_name(name)
    if not pool:
        return "Não foi possível achar uma mappool com esse nome."

    if (mods, slot) not in pool.maps:
        return f"Não existe {mods_slot} na mappool."

    # delete from db
    await app.state.services.database.execute(
        "DELETE FROM tourney_pool_maps WHERE mods = :mods AND slot = :slot",
        {"mods": mods, "slot": slot},
    )

    # remove from cache
    del pool.maps[(mods, slot)]

    return f"{mods_slot} removido de {name}."


@pool_commands.add(Privileges.TOURNEY_MANAGER, aliases=["l"], hidden=True)
async def pool_list(ctx: Context) -> Optional[str]:
    """Lista a informação de todas as mappools que existem."""
    pools = app.state.sessions.pools
    if not pools:
        return "Não existe nenhuma mappool."

    l = [f"Mappools ({len(pools)})"]

    for pool in pools:
        l.append(
            f"[{pool.created_at:%Y-%m-%d}] {pool.id}. "
            f"{pool.name}, por {pool.created_by}.",
        )

    return "\n".join(l)


@pool_commands.add(Privileges.TOURNEY_MANAGER, aliases=["i"], hidden=True)
async def pool_info(ctx: Context) -> Optional[str]:
    """Obtêm toda a informação de uma mappool específica."""
    if len(ctx.args) != 1:
        return "Sintaxe inválida: !pool info <nome>"

    name = ctx.args[0]

    pool = app.state.sessions.pools.get_by_name(name)
    if not pool:
        return "Não foi possível achar uma mappool com esse nome."

    _time = pool.created_at.strftime("%H:%M:%S%p")
    _date = pool.created_at.strftime("%Y-%m-%d")
    datetime_fmt = f"Criado em {_time} no dia {_date}"
    l = [f"{pool.id}. {pool.name}, por {pool.created_by} | {datetime_fmt}."]

    for (mods, slot), bmap in sorted(
        pool.maps.items(),
        key=lambda x: (Mods.to_string(x[0][0]), x[0][1]),
    ):
        l.append(f"{mods!r}{slot}: {bmap.embed}")

    return "\n".join(l)


""" Clan managment commands
# The commands below are for managing bancho.py
# clans, for users, clan staff, and server staff.
"""


@clan_commands.add(Privileges.UNRESTRICTED, aliases=["h"])
async def clan_help(ctx: Context) -> Optional[str]:
    """Mostra todos os comandos documentados de clã que um jogador pode usar."""
    prefix = app.settings.COMMAND_PREFIX
    cmds = []

    for cmd in clan_commands.commands:
        if not cmd.doc or ctx.player.priv & cmd.priv != cmd.priv:
            # no doc, or insufficient permissions.
            continue

        cmds.append(f"{prefix}clan {cmd.triggers[0]}: {cmd.doc}")

    return "\n".join(cmds)


@clan_commands.add(Privileges.UNRESTRICTED, aliases=["c"])
async def clan_create(ctx: Context) -> Optional[str]:
    """Criar um clã com uma tag e nome dados."""
    if len(ctx.args) < 2:
        return "Sintaxe inválida: !clan create <tag> <nome>"

    tag = ctx.args[0].upper()
    if not 1 <= len(tag) <= 6:
        return "Tag do clã deve ter de 1 a 6 caracteres."

    name = " ".join(ctx.args[1:])
    if not 2 <= len(name) <= 16:
        return "Nome do clã deve ter de 2 a 16 caracteres."

    if ctx.player.clan:
        return f"Você já é um membro de {ctx.player.clan}!"

    if app.state.sessions.clans.get(name=name):
        return "Esse nome já é utilizado por outro clã."

    if app.state.sessions.clans.get(tag=tag):
        return "Essa tag já é utilizada por outro clã."

    created_at = datetime.now()

    # add clan to sql
    clan = await clans_repo.create(
        name=name,
        tag=tag,
        owner=ctx.player.id,
    )

    # add clan to cache
    clan = Clan(
        id=clan["id"],
        name=name,
        tag=tag,
        created_at=created_at,
        owner_id=ctx.player.id,
    )
    app.state.sessions.clans.append(clan)

    # set owner's clan & clan priv (cache & sql)
    ctx.player.clan = clan
    ctx.player.clan_priv = ClanPrivileges.Owner

    clan.owner_id = ctx.player.id
    clan.member_ids.add(ctx.player.id)

    await players_repo.update(
        ctx.player.id,
        clan_id=clan.id,
        clan_priv=ClanPrivileges.Owner,
    )

    # announce clan creation
    announce_chan = app.state.sessions.channels["#announce"]
    if announce_chan:
        msg = f"\x01ACTION fundou {clan!r}."
        announce_chan.send(msg, sender=ctx.player, to_self=True)

    return f"{clan!r} criado."


@clan_commands.add(Privileges.UNRESTRICTED, aliases=["delete", "d"])
async def clan_disband(ctx: Context) -> Optional[str]:
    """Desfazer o clã (administradores podem desfazer outros clãs)."""
    if ctx.args:
        # disband a specified clan by tag
        if ctx.player not in app.state.sessions.players.staff:
            return "Somente administradores podem desfazer o clã de outras pessoas."

        clan = app.state.sessions.clans.get(tag=" ".join(ctx.args).upper())
        if not clan:
            return "Não foi possível achar um clã por esse nome."
    else:
        # disband the player's clan
        clan = ctx.player.clan
        if not clan:
            return "Você não é membro de nenhum clã!"

    await clans_repo.delete(clan.id)
    app.state.sessions.clans.remove(clan)

    # remove all members from the clan,
    # reset their clan privs (cache & sql).
    # NOTE: only online players need be to be uncached.
    for member_id in clan.member_ids:
        await players_repo.update(member_id, clan_id=0, clan_priv=0)

        member = app.state.sessions.players.get(id=member_id)
        if member:
            member.clan = None
            member.clan_priv = None

    # announce clan disbanding
    announce_chan = app.state.sessions.channels["#announce"]
    if announce_chan:
        msg = f"\x01ACTION desfez {clan!r}."
        announce_chan.send(msg, sender=ctx.player, to_self=True)

    return f"{clan!r} desfeito."


@clan_commands.add(Privileges.UNRESTRICTED, aliases=["i"])
async def clan_info(ctx: Context) -> Optional[str]:
    """Obtêm a informação de um clã pela sua tag."""
    if not ctx.args:
        return "Sintaxe inválida: !clan info <tag>"

    clan = app.state.sessions.clans.get(tag=" ".join(ctx.args).upper())
    if not clan:
        return "Não foi possível achar um clã por essa tag."

    msg = [f"{clan!r} | Fundado em {clan.created_at:%d %b, %Y}."]

    # get members privs from sql
    clan_members = await players_repo.fetch_many(clan_id=clan.id)
    for member in sorted(clan_members, key=lambda m: m["clan_priv"], reverse=True):
        priv_str = ("Member", "Officer", "Owner")[member["clan_priv"] - 1]
        msg.append(f"[{priv_str}] {member['name']}")

    return "\n".join(msg)


@clan_commands.add(Privileges.UNRESTRICTED)
async def clan_leave(ctx: Context):
    """Sai do clã do qual você faz parte."""
    if not ctx.player.clan:
        return "Você não está em um clã."
    elif ctx.player.clan_priv == ClanPrivileges.Owner:
        return "Você deve tranferir a liderança do seu clã antes de sair, ou você pode desfazer o clã usando !clan disband."

    await ctx.player.clan.remove_member(ctx.player)
    return f"Você saiu do clã {ctx.player.clan!r} com sucesso."


# TODO: !clan inv, !clan join, !clan leave


@clan_commands.add(Privileges.UNRESTRICTED, aliases=["l"])
async def clan_list(ctx: Context) -> Optional[str]:
    """Lista a informação de todos os clãs existentes."""
    if ctx.args:
        if len(ctx.args) != 1 or not ctx.args[0].isdecimal():
            return "Sintaxe inválida: !clan list (página)"
        else:
            offset = 25 * int(ctx.args[0])
    else:
        offset = 0

    total_clans = len(app.state.sessions.clans)
    if offset >= total_clans:
        return "Não existe nenhum clã."

    msg = [f"bancho.py listou ({total_clans} clãs no total)."]

    for idx, clan in enumerate(app.state.sessions.clans, offset):
        msg.append(f"{idx + 1}. {clan!r}")

    return "\n".join(msg)


class CommandResponse(TypedDict):
    resp: Optional[str]
    hidden: bool


async def process_commands(
    player: Player,
    target: Union["Channel", Player],
    msg: str,
) -> Optional[CommandResponse]:
    # response is either a CommandResponse if we hit a command,
    # or simply False if we don't have any command hits.
    start_time = clock_ns()

    prefix_len = len(app.settings.COMMAND_PREFIX)
    trigger, *args = msg[prefix_len:].strip().split(" ")

    # case-insensitive triggers
    trigger = trigger.lower()

    # check if any command sets match.
    for cmd_set in command_sets:
        if trigger == cmd_set.trigger:
            if not args:
                args = ["help"]

            trigger, *args = args  # get subcommand

            # case-insensitive triggers
            trigger = trigger.lower()

            commands = cmd_set.commands
            break
    else:
        # no set commands matched, check normal commands.
        commands = regular_commands

    for cmd in commands:
        if trigger in cmd.triggers and player.priv & cmd.priv == cmd.priv:
            # found matching trigger with sufficient privs
            try:
                res = await cmd.callback(
                    Context(
                        player=player,
                        trigger=trigger,
                        args=args,
                        recipient=target,
                    ),
                )
            except Exception:
                # print exception info to the console,
                # but do not break the player's session.
                traceback.print_exc()

                res = "Um erro inesperado aconteceu durante a execução desse comando."

            if res is not None:
                # we have a message to return, include elapsed time
                elapsed = app.logging.magnitude_fmt_time(clock_ns() - start_time)
                return {"resp": f"{res} | Tempo de execução: {elapsed}", "hidden": cmd.hidden}
            else:
                # no message to return
                return {"resp": None, "hidden": False}

    return None

"""Discord utilities that use pycord."""

from typing import Union

from discord.colour import Colour
from app.objects.beatmap import Beatmap, BeatmapSet, RankedStatus
from discord import Webhook, Embed
import aiohttp


def create_beatmap_changes_embed(beatmap:Beatmap, new_status:RankedStatus) -> Embed:

    # old_status_string = f"{beatmap.status!s}"
    new_status_string = f"{new_status!s}"

    title = f"Novo mapa adicionado a seção {new_status_string}:"
    
    embed = Embed(title=title, color=Colour(0).from_rgb(0, 0, 255))

    embed.set_thumbnail(url=f"https://b.ppy.sh/thumb/{beatmap.set_id}l.jpg")

    star_rating = f"{int(beatmap.diff*100)/100}🌟"

    embed.add_field(
        name=f"{beatmap.artist} - {beatmap.title} [{beatmap.version}]",
        value=(f"**{star_rating}** - **CS** {beatmap.cs} - **AR** {beatmap.ar} - **BPM** {beatmap.bpm}\
        \n **LINK:** https://osu.ppy.sh/beatmapsets/{beatmap.set_id}"),
        inline=False
    )

    return embed

def create_beatmapset_changes_embed(beatmapset:BeatmapSet, new_status:RankedStatus) -> Embed:

    # old_status_string = f"{beatmap.status!s}"
    new_status_string = f"{new_status!s}"

    title = f"Novo set de mapas adicionado a seção {new_status_string}:"
    
    embed = Embed(title=title, color=Colour(0).from_rgb(0, 0, 255))

    embed.set_thumbnail(url=f"https://b.ppy.sh/thumb/{beatmapset.id}l.jpg")

    embed.add_field(
        name=f"{beatmapset.maps[0].artist} - {beatmapset.maps[0].title}",
        value=(f"**DIFICULDADES NO SET:** {len(beatmapset.maps)}\
        \n **LINK:** https://osu.ppy.sh/beatmapsets/{beatmapset.id}"),
        inline=False
    )

    return embed



async def send_beatmap_status_change(webhook_url:str, beatmap:Beatmap, new_status:RankedStatus) -> None:
    """Send new ranked status from the beatmap to discord."""
    embed = create_beatmap_changes_embed(beatmap, new_status)

    async with aiohttp.ClientSession() as session:
        webhook = Webhook.from_url(webhook_url, session=session)
        await webhook.send(embed=embed)
    

async def send_beatmapset_status_change(webhook_url:str, beatmapset:BeatmapSet, new_status:RankedStatus) -> None:
    """Send new ranked status from the beatmapset to discord."""
    embed = create_beatmapset_changes_embed(beatmapset, new_status)

    async with aiohttp.ClientSession() as session:
        webhook = Webhook.from_url(webhook_url, session=session)
        await webhook.send(embed=embed)
    
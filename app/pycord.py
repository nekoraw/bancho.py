"""Discord utilities that use pycord."""

from discord.colour import Colour
from app.objects.beatmap import Beatmap, RankedStatus
from discord import Webhook, Embed
import aiohttp


def create_beatmap_changes_embed(beatmap:Beatmap, new_status:RankedStatus) -> Embed:
    embed = Embed(title=f"Status de rank do mapa modificado:", color=Colour(0).from_rgb(0, 0, 255))

    embed.set_thumbnail(url=f"https://b.ppy.sh/thumb/{beatmap.set_id}l.jpg")

    star_rating = f"{int(beatmap.diff*100)/100}🌟"

    old_status_string = f"{beatmap.status!s}"
    new_status_string = f"{new_status!s}"

    embed.add_field(
        name=f"{beatmap.artist} - {beatmap.title} [{beatmap.version}] ({star_rating})",
        value=f"Status: {old_status_string} -> {new_status_string}",
        inline=False
    )

    return embed

async def send_beatmap_status_change(webhook_url:str, beatmap:Beatmap, new_status:RankedStatus):
    """Send new ranked status from the beatmap to discord."""
    async with aiohttp.ClientSession() as session:
        webhook = Webhook.from_url(webhook_url, session=session)
        await webhook.send(embed=create_beatmap_changes_embed(beatmap, new_status), username="Beatmap Updates")
    



    
    
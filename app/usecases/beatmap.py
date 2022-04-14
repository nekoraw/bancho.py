from __future__ import annotations

from typing import Optional

import app.state
from app.objects.beatmap import Beatmap
from app.objects.beatmap import BeatmapSet
from app.objects.beatmap import osuapiv1_getbeatmaps


async def from_md5(md5: str, set_id: int = -1) -> Optional[Beatmap]:
    """Fetch a map from the cache, database, or osuapi by md5."""
    bmap = await Beatmap._from_md5_cache(md5)

    if not bmap:
        # map not found in cache

        # to be efficient, we want to cache the whole set
        # at once rather than caching the individual map

        if set_id <= 0:
            # set id not provided - fetch it from the map md5
            res = await app.state.services.database.fetch_one(
                "SELECT set_id FROM maps WHERE md5 = :map_md5",
                {"map_md5": md5},
            )

            if res is not None:
                # set found in db
                set_id = res["set_id"]
            else:
                # set not found in db, try osu!api
                api_data = await osuapiv1_getbeatmaps(h=md5)

                if not api_data:
                    return None

                set_id = int(api_data[0]["beatmapset_id"])

        # fetch (and cache) beatmap set
        beatmap_set = await BeatmapSet.from_bsid(set_id)

        if beatmap_set is not None:
            # the beatmap set has been cached - fetch beatmap from cache
            bmap = await Beatmap._from_md5_cache(md5, check_updates=False)

    return bmap


async def from_bid(bid: int) -> Optional[Beatmap]:
    """Fetch a map from the cache, database, or osuapi by id."""
    bmap = await Beatmap._from_bid_cache(bid)

    if not bmap:
        # map not found in cache

        # to be efficient, we want to cache the whole set
        # at once rather than caching the individual map

        res = await app.state.services.database.fetch_one(
            "SELECT set_id FROM maps WHERE id = :map_id",
            {"map_id": bid},
        )

        if res is not None:
            # set found in db
            set_id = res["set_id"]
        else:
            # set not found in db, try osu!api
            api_data = await osuapiv1_getbeatmaps(b=bid)

            if not api_data:
                return None

            set_id = int(api_data[0]["beatmapset_id"])

        # fetch (and cache) beatmap set
        beatmap_set = await BeatmapSet.from_bsid(set_id)

        if beatmap_set is not None:
            # the beatmap set has been cached - fetch beatmap from cache
            bmap = await Beatmap._from_bid_cache(bid, check_updates=False)

    return bmap

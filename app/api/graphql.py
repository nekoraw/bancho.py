import strawberry
import app.state

from strawberry.types.info import Info
from strawberry.types.nodes import Selection

def field_requested(selection: Selection, wanted_field: str) -> bool:
    for field in selection.selections:
        if field.name == wanted_field:
            return True
        
    return False

@strawberry.type
class PlayerCounts:
    total: int
    online: int

@strawberry.type
class Query:
    @strawberry.field
    async def player_counts(self, info: Info) -> PlayerCounts:
        selections = info.selected_fields
        
        total = 0
        if field_requested(selections[0], "total"):
            total = await app.state.services.database.fetch_val(
                "SELECT COUNT(*) FROM users",
                column=0,
            )
        
        return PlayerCounts(
            total=total,
            online=len(app.state.sessions.players.unrestricted) - 1,
        )
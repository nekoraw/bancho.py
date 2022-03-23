""" api: bancho.py's developer api for interacting with server state """
from __future__ import annotations

from strawberry.fastapi import GraphQLRouter
import strawberry

@strawberry.type
class Query:
    @strawberry.field
    def test(self) -> str:
        return "hi"

schema = strawberry.Schema(Query)
router = GraphQLRouter(schema, path="/")
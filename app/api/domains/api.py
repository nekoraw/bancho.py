""" api: bancho.py's developer api for interacting with server state """
from __future__ import annotations

import strawberry
from strawberry.fastapi import GraphQLRouter

from app.api.graphql import Query

schema = strawberry.Schema(Query)
router = GraphQLRouter(schema, path="/")

""" api: bancho.py's developer api for interacting with server state """
from __future__ import annotations

from app.api.graphql import Query
from strawberry.fastapi import GraphQLRouter
import strawberry

schema = strawberry.Schema(Query)
router = GraphQLRouter(schema, path="/")
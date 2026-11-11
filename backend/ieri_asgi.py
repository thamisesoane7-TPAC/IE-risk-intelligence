"""Preserve the existing FastAPI object, lifespan, DB integration and all old routes."""
import importlib
import os

from fastapi import FastAPI

from ieri_risk.api import router

module_name, object_name = os.environ.get("IERI_BASE_APP", "app.main:app").split(":", 1)
app = getattr(importlib.import_module(module_name), object_name)
if not isinstance(app, FastAPI):
    raise TypeError("IERI_BASE_APP must identify the existing FastAPI application")
if not getattr(app.state, "ieri_risk_installed", False):
    existing = set(app.openapi().get("paths", {}))
    if any(route.path in existing for route in router.routes):
        raise RuntimeError("An existing route conflicts with the /risk namespace")
    app.include_router(router)
    app.openapi_schema = None
    app.state.ieri_risk_installed = True

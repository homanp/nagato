from fastapi import APIRouter

from lib.api import ingest, invoke, webhook

router = APIRouter()
api_prefix = "/api/v1"

router.include_router(ingest.router, tags=["Ingest"], prefix=api_prefix)
router.include_router(invoke.router, tags=["Invoke"], prefix=api_prefix)
router.include_router(webhook.router, tags=["Webhook"], prefix=api_prefix)

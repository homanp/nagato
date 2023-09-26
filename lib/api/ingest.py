from fastapi import APIRouter

router = APIRouter()


@router.post(
    "/ingest",
    name="ingest",
    description="Ingest data",
)
async def ingest():
    """Endpoint for ingesting data"""
    return {"success": True, "data": None}

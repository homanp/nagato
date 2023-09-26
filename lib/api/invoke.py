from fastapi import APIRouter

router = APIRouter()


@router.post(
    "/invoke",
    name="invoke",
    description="Invoke model",
)
async def invoke():
    """Endpoint for invoking model"""
    return {"success": True, "data": None}

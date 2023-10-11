from fastapi import APIRouter

from lib.utils.prisma import prisma
from prisma import Json

router = APIRouter()


@router.post(
    "/webhook/finetune",
    name="finetune_webhook",
    description="Webhook events for fine-tuning",
)
async def finetune_webhook(body: dict):
    """Endpoint for invoking model"""
    finetune_id = body.get("id")
    status = body.get("status")
    datasources = await prisma.datasource.find_many()
    for ds in datasources:
        if ds.finetune and ds.finetune.get("id") == finetune_id:
            datasource_ = await prisma.datasource.update(
                where={"id": ds.id},
                data={
                    "finetune": Json({**body, **ds.finetune}),
                    "status": "DONE" if status == "succeeded" else "FAILED",
                },
            )
    return {"success": True, "data": datasource_}

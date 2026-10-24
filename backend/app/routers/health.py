from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request) -> dict:
    service = request.app.state.model_service
    models = {
        name: {"available": name in service.models, "model_name": service.models[name].model_name if name in service.models else None}
        for name in ("ehr", "pima")
    }
    ready = service.loaded and all(info["available"] for info in models.values())
    return {"status": "ok" if ready else "unavailable", "models": models}

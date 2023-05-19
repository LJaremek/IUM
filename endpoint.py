from fastapi import FastAPI, HTTPException
from random import randint

"""
    exec: uvicorn endpoint:app --reload
"""

app = FastAPI()


class ModelA:
    def __call__(self, x: str) -> int:
        return -1


class ModelB:
    def __call__(self, x: str) -> int:
        return -1


@app.get("/model/")
def execute_empty_model() -> str:
    return "you need to specify model variant"


@app.get("/model/{variant}")
async def execute_model(variant: str, data: list[int]) -> int:
    if variant == 'a':
        return ModelA()(data)
    elif variant == 'b':
        return ModelB()(data)
    else:
        raise HTTPException(status_code=404, detail="unknown model")


@app.get("/ab_experiment")
def ab_experiment(data: list[int]) -> int:
    return ModelA()(data) if randint(0, 1) == 0 else ModelB()(data)

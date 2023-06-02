from fastapi import FastAPI, HTTPException
import numpy as np
import torch

from models import get_models

"""
    exec: uvicorn endpoint:app --reload
"""


model_a_dict, model_b_dict = get_models()

model_a = model_a_dict["model"].eval()
model_b = model_b_dict["model"].eval()

app = FastAPI()


@app.get("/model/")
def execute_empty_model() -> str:
    return "you need to specify model variant"


@app.get("/model/{variant}")
async def execute_model(variant: str, data: list) -> tuple[int, float, float]:
    """
    Input:
     * variant: str - model variant
     * data: list - model params (series data and parameters data)

    Output:
     * results: tuple[int, float, float] - model result, model std, model mean
    """
    X_s, X_p = data
    X_s = torch.from_numpy(np.array(X_s).astype('float32'))
    X_p = torch.from_numpy(np.array(X_p).astype('float32'))

    if variant == "a":
        preds = model_a(X_s, X_p)
        result = int(
            float(preds[0][-1]*model_a_dict["std"]+model_a_dict["mean"])
            )
        return result, model_a_dict["std"], model_a_dict["mean"]
    elif variant == "b":
        preds = model_b(X_s, X_p)
        result = int(
            float(preds[0][-1]*model_b_dict["std"]+model_b_dict["mean"])
            )
        return result, model_a_dict["std"], model_a_dict["mean"]
    else:
        raise HTTPException(status_code=404, detail="unknown model")


@app.get("/ab_experiment")
def ab_experiment(data: list[int]) -> int:
    return 8

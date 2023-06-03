from fastapi import FastAPI, HTTPException
import numpy as np
import torch

from models import get_models
import experiment_ab

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
    X_s = torch.from_numpy(np.array(X_s).astype("float32"))
    X_p = torch.from_numpy(np.array(X_p).astype("float32"))

    if variant == "a":
        preds = model_a(X_s, X_p)
        result = int(
            float(preds[0][-1][0]*model_a_dict["std"]+model_a_dict["mean"])
            )
        return result, model_a_dict["std"], model_a_dict["mean"]

    elif variant == "b":
        preds = model_b(X_s, X_p)
        result = int(
            float(preds[0][-1][0]*model_b_dict["std"]+model_b_dict["mean"])
            )
        return result, model_b_dict["std"], model_b_dict["mean"]

    else:
        raise HTTPException(status_code=404, detail="unknown model")


@app.get("/ab_experiment")
def ab_experiment(
        time_series_path: str = "./data/X_s_test.csv",
        parameters_path: str = "./data/X_p_test.csv",
        expected_results_path: str = "./data/y_test.csv",
        parameters_size: tuple[int, int] = (7, 24)
        ) -> tuple[float, float]:
    """
    Input:
     * time_series_path: str - path to time series data
     * parameters_path: str - path to parameters data
     * expected_results_path: str - expected y data
     * parameters_size: tuple[int, int] - shape of parameters (-, p1,  p2)

    Output:
     * results: tuple[float, float] - model a avg loss, model b avg loss
    """
    return experiment_ab.run_ab(
        time_series_path,
        parameters_path,
        expected_results_path,
        parameters_size
        )

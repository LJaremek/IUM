# import requests
import logging

from fastapi import FastAPI, HTTPException
# import torch.utils.data as data
# import pandas as pd
import numpy as np
import torch

from logging_config import configure_logging
from models import get_models
import experiment_ab

"""
exec: uvicorn endpoint:app --reload
curl -X GET "127.0.0.1:8000/ab_experiment?time_series_path=./data/X_s_test.csv"
"""

configure_logging()

logger = logging.getLogger(__name__)

model_a_dict, model_b_dict = get_models()

model_a = model_a_dict["model"].eval()
model_b = model_b_dict["model"].eval()

app = FastAPI()


@app.get("/model/")
def execute_empty_model() -> str:
    logger.warning("Call api /model/ without model name")
    return "you need to specify model variant"


@app.get("/model/{variant}")
async def execute_model(variant: str, data_: list) -> tuple[int, float, float]:
    """
    Input:
     * variant: str - model variant
     * data: list - model params (series data and parameters data)

    Output:
     * results: tuple[int, float, float] - model result, model std, model mean
    """
    X_s, X_p = data_
    X_s = torch.from_numpy(np.array(X_s).astype("float32"))
    X_p = torch.from_numpy(np.array(X_p).astype("float32"))

    if variant == "a":
        preds = model_a(X_s, X_p)
        result = int(
            float(preds[0][-1][0]*model_a_dict["std"]+model_a_dict["mean"])
            )
        logger.info("Call api /model/ for model A")
        return result, model_a_dict["std"], model_a_dict["mean"]

    elif variant == "b":
        preds = model_b(X_s, X_p)
        result = int(
            float(preds[0][-1][0]*model_b_dict["std"]+model_b_dict["mean"])
            )
        logger.info("Call api /model/ for model B")
        return result, model_b_dict["std"], model_b_dict["mean"]

    else:
        logger.error(f"Call api /model/ for not exists model {variant}")
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
    logger.info("Call api for experiment A/B")
    return experiment_ab.run_ab(
        time_series_path,
        parameters_path,
        expected_results_path,
        parameters_size
        )


# @app.get("/success_criterion")
# def success_criterion(
#         model: str,
#         time_series_path: str = "./data/X_s_test.csv",
#         parameters_path: str = "./data/X_p_test.csv",
#         expected_results_path: str = "./data/y_test.csv",
#         timestamps_path: str = "./data/timestamp.csv",
#         parameters_size: tuple[int, int] = (7, 24)
#         ) -> tuple[float, float]:
#     """
#     Input:
#      * model: str - model name. Available: "a", "b"
#      * time_series_path: str - path to time series data
#      * parameters_path: str - path to parameters data
#      * expected_results_path: str - expected y data
#      * timestamps_path: str - path to timestamps of data
#      * parameters_size: tuple[int, int] - shape of parameters (-, p1,  p2)

#     Output:
#      * costs: tuple[float, float] - real costs, model costs
#     """
#     logger.info("Call api for success criterion")
#     X_s, X_p, y = experiment_ab.unpack_data(
#         time_series_path, parameters_path, expected_results_path
#         )
#     URL = "http://localhost:8000/model/"
#     dataset = data.TensorDataset(X_s, X_p, y)
#     loader = data.DataLoader(dataset, shuffle=False)
#     times = pd.read_csv(timestamps_path)

#     times_index = -1
#     last_time = times.iloc[0][0]
#     popular_today = {}
#     with torch.no_grad():
#         for X_s_, X_p_, y_ in loader:

#             X_s_ = X_s_.unsqueeze(2).tolist()
#             X_p_ = X_p_.tolist()
#             y_ = y_.unsqueeze(2).tolist()

#             times_index += 1

#             if times.iloc[times_index][0] != last_time:
#                 break

#             response = requests.get(URL+model, json=(X_s_, X_p_))

#             popular_today[tuple(X_p_[0][0])] = response.json()[0]

#         sorted_genres = sorted(
#                 popular_today, key=lambda k: popular_today[k]
#                 )

#         print(last_time, len(sorted_genres))
#     return (1.0, 1.0)

from random import randint
import requests

import torch.utils.data as data
import pandas as pd
import torch

URL = "http://localhost:8000/model/"
BATCH_SIZE = 8


def run_ab(
        time_series_path: str = "./data/X_s_test.csv",
        parameters_path: str = "./data/X_p_test.csv",
        expected_results_path: str = "./data/y_test.csv",
        parameters_size: tuple[int, int] = (7, 24)
        ) -> tuple[float, float]:

    X_s = pd.read_csv(time_series_path, header=None, index_col=False)
    X_p = pd.read_csv(parameters_path, header=None, index_col=False)
    y = pd.read_csv(expected_results_path, header=None, index_col=False)

    p1, p2 = parameters_size

    size = X_p.shape[0]//p1
    X_p = X_p.to_numpy().reshape((size, p1, p2))

    y = torch.tensor(y.values.astype("float32"))
    X_s = torch.tensor(X_s.values.astype("float32"))
    X_p = torch.tensor(X_p.astype("float32"))

    dataset = data.TensorDataset(X_s, X_p, y)
    loader = data.DataLoader(dataset, shuffle=False)

    results: dict[str, list[int]] = {"a": [], "b": []}
    with torch.no_grad():
        for X_s_, X_p_, y_ in loader:

            X_s_ = X_s_.unsqueeze(2).tolist()
            X_p_ = X_p_.tolist()
            y_ = y_.unsqueeze(2).tolist()

            if randint(0, 1):
                model = "a"
            else:
                model = "b"

            response = requests.get(URL+model, json=(X_s_, X_p_))
            model_preds, std, mean = response.json()

            y_expected = y_[0][-1][0]*std+mean

            results[model].append(model_preds-y_expected)

    print(sum(results["a"])/len(results["a"]))
    print(sum(results["b"])/len(results["b"]))


run_ab()

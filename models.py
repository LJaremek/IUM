import logging
import json

import torch.nn as nn
import torch

from logging_config import configure_logging

configure_logging()

logger = logging.getLogger(__name__)


class SuperMusicModel(nn.Module):
    def __init__(self, num_params) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=50, num_layers=1, batch_first=True
            )

        self.linear = nn.Linear(50 + num_params, 1)  # .cuda()

    def forward(self, x, params):
        x, (h_n, _) = self.lstm(x)
        combined = torch.cat((x, params), dim=2)
        output = self.linear(combined)
        return output


class WeakMusicModel(nn.Module):
    def __init__(self, num_params) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=10, num_layers=1, batch_first=True
            )
        self.linear = nn.Linear(10 + num_params, 1).cuda()

    def forward(self, x, params):
        x, (h_n, _) = self.lstm(x)
        return x


def get_models(conf_path: str = "./models.conf") -> tuple[dict, dict]:
    with open(conf_path, "r", -1, "utf-8") as file:
        models_data = json.load(file)

    model_a = {}
    model_a["path"] = models_data["a"]["path"]
    model_a["params"] = models_data["a"]["params"]
    model_a["std"] = models_data["a"]["std"]
    model_a["mean"] = models_data["a"]["mean"]

    try:
        model_a["model"] = torch.load(models_data["a"]["path"])
    except AttributeError:
        model_a["model"] = SuperMusicModel(model_a["params"])
        logger.error("Error during load model A")

    model_b = {}
    model_b["path"] = models_data["b"]["path"]
    model_b["params"] = models_data["b"]["params"]
    model_b["std"] = models_data["b"]["std"]
    model_b["mean"] = models_data["b"]["mean"]

    try:
        model_b["model"] = torch.load(models_data["b"]["path"])
    except AttributeError:
        model_b["model"] = WeakMusicModel(model_b["params"])
        logger.error("Error during load model B")

    return model_a, model_b


if __name__ == "__main__":
    model_a = torch.load("./lstm_gatunki.pt")

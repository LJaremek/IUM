import json

import torch.nn as nn
import torch


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


def get_models(conf_path: str = "./models.conf") -> tuple[dict, dict]:
    with open(conf_path, "r", -1, "utf-8") as file:
        models_data = json.load(file)

    model_a = {}
    model_a["path"] = models_data["a"]["path"]
    model_a["params"] = models_data["a"]["params"]
    model_a["std"] = models_data["a"]["std"]
    model_a["mean"] = models_data["a"]["mean"]
    model_a["model"] = SuperMusicModel(model_a["params"])

    model_b = {}
    model_b["path"] = models_data["b"]["path"]
    model_b["params"] = models_data["b"]["params"]
    model_b["std"] = models_data["b"]["std"]
    model_b["mean"] = models_data["b"]["mean"]
    model_b["model"] = SuperMusicModel(model_b["params"])

    return model_a, model_b


if __name__ == "__main__":
    model_a = torch.load("./lstm_gatunki.pt")

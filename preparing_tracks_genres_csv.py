import pandas as pd


class Genres:
    def __init__(self) -> None:
        self.genres: list[str] = []

    def get(self, x) -> None:
        x = ",".join(x)
        self.genres += x.split(",")
        return x


def build_track_genres(file_path: str = "./data/tracks_genres.csv") -> None:
    """
    Generating csv file which connect tracks with author genres
    """
    df_artists = pd.read_json("data/artists.jsonl", lines=True)
    df_tracks = pd.read_json("data/tracks.jsonl", lines=True)

    df_tracks["id_track"] = df_tracks["id"]
    df_tracks = df_tracks.drop(["id"], axis=1)

    df = pd.merge(
        df_tracks,
        df_artists.drop(["name"], axis=1),
        left_on="id_artist",
        right_on="id"
    ).drop(["id"], axis=1)

    df = df[["id_track", "genres"]]

    g = Genres()

    df["genres"].apply(lambda x: g.get(x))

    popular = {
        "hip hop": [],
        "pop": [],
        "rap": [],
        "rock": [],
        "disco": [],
        "metal": [],
        "punk": [],
        "jazz": [],
        "indie": [],
        "folk": [],
        "r&b": [],
        "soul": [],
        "drill": [],
        "electro": [],
        "house": [],
        "reggae": [],
        "funk": [],
        "dance": [],
        "latin": [],
        "romantic": [],
        "country": [],
        "classical": [],
        "edm": [],
        "_other_": []
        }

    for _, e in enumerate(set(g.genres)):
        done = False
        for key in popular.keys():
            if key in e:
                popular[key].append(e)
                done = True
        if not done:
            popular["_other_"].append(e)

    for key, values in popular.items():
        df[key] = df["genres"].apply(
            lambda x: 1 if set(values).intersection(x) else 0
            )

    df = df.drop(["genres"], axis=1)
    df.to_csv(file_path, index=False)


def get_test_data_with_authors() -> None:
    df_sessions = pd.read_json("data/sessions.jsonl", lines=True)
    df_artists = pd.read_json("data/artists.jsonl", lines=True)
    df_tracks = pd.read_json("data/tracks.jsonl", lines=True)

    df = pd.merge(
        df_sessions,
        df_tracks,
        left_on="track_id",
        right_on="id"
    ).drop(["id"], axis=1)

    df = pd.merge(
        df,
        df_artists,
        left_on="id_artist",
        right_on="id"
    ).drop(["id"], axis=1)

    df_with_tracks = df[["track_id", "timestamp", "genres"]]
    df_without_tracks = df[["timestamp", "genres"]]

    df_with_tracks['timestamp'] = pd.to_datetime(
        df_with_tracks['timestamp']
        ).dt.date
    df_without_tracks['timestamp'] = pd.to_datetime(
        df_without_tracks['timestamp']
        ).dt.date


def get_track_genres(
        track_id: str,
        data_path: str = "./data/tracks_genres.csv"
        ) -> list[int]:
    df = pd.read_csv(data_path)
    return df.loc[df["id_track"] == track_id].drop(["id_track"], axis=1)


if __name__ == "__main__":
    # build_track_genres()

    # df_track_storage = pd.read_json("data/track_storage.jsonl", lines=True)
    # tmp_storages = df_track_storage.groupby("storage_class").\
    #   count()["track_id"]
    # storage_size = {
    #     "fast": tmp_storages["fast"],
    #     "medium": tmp_storages["medium"],
    #     "slow": tmp_storages["slow"]
    # }

    print(get_track_genres("0LcNMuOiULmxJK3bdHTfDF"))

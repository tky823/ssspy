# It is expected to run from root ssspy directory
from tests.dummy.utils.dataset import download_sample_speech_data


def download_all() -> None:
    conditions = [
        {"n_sources": 2, "sisec2010_tag": "dev1_female3"},
        {"n_sources": 3, "sisec2010_tag": "dev1_female3"},
        {"n_sources": 4, "sisec2010_tag": "dev1_female4"},
    ]
    max_durations = [0.1, 0.5]

    for kwargs in conditions:
        for max_duration in max_durations:
            download_sample_speech_data(max_duration=max_duration, **kwargs)


if __name__ == "__main__":
    download_all()

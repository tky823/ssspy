# It is expected to run from root ssspy directory
from tests.dummy.utils.dataset import download_sample_speech_data


def download_all() -> None:
    conditions = [
        {"n_sources": 2, "sisec2010_tag": "dev1_female3"},
        {"n_sources": 3, "sisec2010_tag": "dev1_female3"},
        {"n_sources": 4, "sisec2010_tag": "dev1_female4"},
    ]
    for kwargs in conditions:
        download_sample_speech_data(**kwargs)


if __name__ == "__main__":
    download_all()

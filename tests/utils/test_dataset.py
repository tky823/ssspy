import pytest

from ssspy.utils.dataset import download_dummy_data

parameters_dataset = [
    (2, "dev1_female3", 20000),
    (3, "dev1_female3", 20000),
    (4, "dev1_female4", 20000),
]


@pytest.mark.parametrize("n_sources, sisec2010_tag, max_samples", parameters_dataset)
def test_dataset(n_sources: int, sisec2010_tag: str, max_samples: int):
    waveform_src_img = download_dummy_data(
        sisec2010_root="./tests/.data/SiSEC2010",
        mird_root="./tests/.data/MIRD",
        n_sources=n_sources,
        sisec2010_tag=sisec2010_tag,
        max_samples=max_samples,
    )

    n_channels = n_sources

    assert waveform_src_img.shape == (n_channels, n_sources, max_samples)

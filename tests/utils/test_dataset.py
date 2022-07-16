import pytest

from ssspy.utils.dataset import download_sample_speech_data

parameters_dataset = [
    (2, "dev1_female3"),
    (3, "dev1_female3"),
    (4, "dev1_female4"),
]
parameters_max_samples = [20000]
parameters_conv = [True, False]


@pytest.mark.parametrize("n_sources, sisec2010_tag", parameters_dataset)
@pytest.mark.parametrize("max_samples", parameters_max_samples)
@pytest.mark.parametrize("conv", parameters_conv)
def test_conv_dataset(n_sources: int, sisec2010_tag: str, max_samples: int, conv: bool):
    waveform_src_img = download_sample_speech_data(
        sisec2010_root="./tests/.data/SiSEC2010",
        mird_root="./tests/.data/MIRD",
        n_sources=n_sources,
        sisec2010_tag=sisec2010_tag,
        max_samples=max_samples,
        conv=conv,
    )

    n_channels = n_sources

    assert waveform_src_img.shape == (n_channels, n_sources, max_samples)

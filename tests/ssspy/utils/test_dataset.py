import pytest

from ssspy.utils.dataset import download_sample_speech_data

parameters_dataset = [
    (2, "dev1_female3"),
    (3, "dev1_female3"),
    (4, "dev1_female4"),
]
parameters_max_duration = [1.2]
parameters_conv = [True, False]


@pytest.mark.parametrize("n_sources, sisec2010_tag", parameters_dataset)
@pytest.mark.parametrize("max_duration", parameters_max_duration)
@pytest.mark.parametrize("conv", parameters_conv)
def test_conv_dataset(n_sources: int, sisec2010_tag: str, max_duration: int, conv: bool):
    waveform_src_img, sample_rate = download_sample_speech_data(
        sisec2010_root="./tests/.data/SiSEC2010",
        mird_root="./tests/.data/MIRD",
        n_sources=n_sources,
        sisec2010_tag=sisec2010_tag,
        max_duration=max_duration,
        conv=conv,
    )

    n_channels = n_sources

    assert waveform_src_img.shape == (n_channels, n_sources, int(sample_rate * max_duration))

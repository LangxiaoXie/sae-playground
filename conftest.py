import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs GPT-2 loaded (~10s)")


@pytest.fixture(scope="session")
def gpt2():
    """Real GPT-2, loaded once for the whole session."""
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained("gpt2", device="cpu")

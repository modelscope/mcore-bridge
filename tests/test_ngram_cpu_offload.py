# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest

from mcore_bridge.utils import use_ngram_embedding_cpu_offload


@pytest.mark.parametrize('shared,legacy,expected', [
    (None, None, False),
    ('1', None, True),
    ('0', None, False),
    (None, '1', True),
    (None, '0', False),
    ('0', '1', False),
    ('1', '0', True),
    ('true', 'false', True),
    ('false', 'true', False),
])
def test_ngram_cpu_offload_precedence(monkeypatch, shared, legacy, expected):
    for name, value in [('NGRAM_EMBEDDING_CPU_OFFLOAD', shared), ('PLE_CPU_OFFLOAD', legacy)]:
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert use_ngram_embedding_cpu_offload() is expected

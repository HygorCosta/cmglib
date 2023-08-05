import pytest
import numpy as np
from pathlib import Path
from cmglib.builder import Builder


@pytest.fixture(name='modelo')
def builder_model():
    nx, ny, nz = 10, 10, 10
    return Builder(nx, ny, nz)

def test_ler_coord(modelo):
    coord = modelo.ler_coord('notebook\Arquivos\Fluxo_COORD.inc')
    assert 1

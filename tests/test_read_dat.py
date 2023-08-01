import pytest
from cmglib.read_dat import DatReader

@pytest.fixture(name='modelo')
def dat_modelo():
    dat_file = r"C:\Users\DVOS\OneDrive - PETROBRAS\Documentos\Pessoais\Doutorado\Softwares\ReservoirsDATA\OLYMPUS\OLYMPUS\OLYMPUS_BHP_noWC_1_AMALGAMATE.dat"
    return DatReader(dat_file)


def test_remove_comment_lines(modelo):
    with open(modelo.dat, 'r', encoding='UTF-8') as file:
        file = modelo.remove_comment_lines(file)
    assert 1

def test_section_reservoir_grid_dimension(modelo):
    with open(modelo.dat, 'r', encoding='UTF-8') as file:
        text = file.read()
        modelo.reservoir_description(text)


def test_section_reservoir_grid_include(modelo):
    with open(modelo.dat, 'r', encoding='UTF-8') as file:
        text = file.read()
        modelo.reservoir_description(text)

def test_section_reservoir_perm(modelo):
    with open(modelo.dat, 'r', encoding='UTF-8') as file:
        text = file.read()
        modelo.reservoir_description(text)

def test_read(modelo):
    modelo.read()

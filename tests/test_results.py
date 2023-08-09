from cmglib.results import Results
import pytest

@pytest.fixture(name='model')
def model_instance():
    basename = 'notebook/Arquivos/OLYMPUS.dat'
    return Results(basename)

def test_parse_include_files(model):
    obtido = model._parse_include_files()
    assert isinstance(obtido, list)

def test_parse_general_info(model):
    model.parse_general_info()
    assert model.general is not None

def test_get_dimensionality(model):
    model.parse_general_info()
    num, den = model._get_dimensionality('WAQCUMSEC')
    assert num == 11
    assert den is None

def test_get_output_unit(model):
    obtido = model._get_output_unit('ALGO')
    assert isinstance(obtido, str)

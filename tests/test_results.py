from cmglib.results import Results
import pytest
import xarray as xr

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
    model.parse_general_info()
    gain, offset = model._get_output_unit('WAQCUMSEC')
    assert isinstance(gain, float)
    assert isinstance(offset, float)

def test_conversion_params(model):
    model.parse_general_info()
    num, den = model._get_dimensionality('WAQCUMSEC')
    gain, offset = model._conversion_params(num, 'm3', 'bbl')
    assert isinstance(gain, float)
    assert isinstance(offset, float)

def test_conversion_gain_offset(model):
    model.parse_general_info()
    gain, offset = model._conversion_gain_offset('WAQCUMSEC')
    assert isinstance(gain, float)
    assert isinstance(offset, float)

def test_read_timeseries_GROUP(model):
    obtido = model.read_timeseries('GROUPS')
    return isinstance(obtido, xr.Dataset)

def test_read_timeseries_WELLS(model):
    obtido = model.read_timeseries('WELLS')
    return isinstance(obtido, xr.Dataset)

import pytest
import numpy as np
from pathlib import Path
from cmglib.report import Report

@pytest.fixture(name='report_obj')
def inst_report():
    tpl = 'cmglib\TemplateReport.tpl'
    return Report(tpl)


def test_report_info(report_obj):
    assert report_obj._report_exe.is_file()


def test_basename_sr3(report_obj):
    sr3_file = Path("cmglib\OLYMPUS_BHP_noWC_1.sr3")
    names = report_obj._cmg_files(sr3_file)
    assert names.sr3 == sr3_file


def test_template(report_obj):
    sr3_file = Path("cmglib\OLYMPUS_BHP_noWC_1.sr3")
    report_obj.basename = report_obj._cmg_files(sr3_file)
    report_obj._tpl_to_rwd()
    assert report_obj.basename.rwd.is_file()


def test_run_report(report_obj):
    sr3_file = Path("cmglib\OLYMPUS_BHP_noWC_1.sr3")
    assert report_obj.run_report(sr3_file, rwo_exist_ok=True)


def test_read_rwo(report_obj):
    rwo_file = Path("cmglib\OLYMPUS_BHP_noWC_1.rwo")
    df_rwo = report_obj.read_rwo(rwo_file)
    assert 1


def test_read_cash_flow(report_obj):
    sr3_file = Path("cmglib\OLYMPUS_BHP_noWC_1.sr3")
    return_code = report_obj.run_report(sr3_file)
    prices = np.array([70, 0, -10, -5])
    df_rwo = report_obj.add_cash_flow(prices)
    assert 1


def test_read_npv(report_obj):
    sr3_file = Path("cmglib\OLYMPUS_BHP_noWC_1.sr3")
    return_code = report_obj.run_report(sr3_file)
    prices = np.array([70, 0, -10, -5])
    proj_npv = report_obj.npv(tma=0.1, prices=prices)
    assert proj_npv > 0

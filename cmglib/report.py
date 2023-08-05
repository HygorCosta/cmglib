"""Class to run REPORT in pos-processor."""
import logging
import os
import re
import subprocess
import pandas as pd
import xarray as xr
import numpy as np
from collections import namedtuple
from pathlib import Path
from string import Template

class Report:

    _basename = namedtuple('basename', 'sr3 rwd rwo')

    def __init__(self, template:str=None, key_word:str='SR3FILE') -> None:
        self.template = template
        self.key_word = key_word
        self._report_exe = self.report_info()

    @staticmethod
    def report_info():
        cmg_home = Path(os.environ['CMG_HOME'])
        report = list(cmg_home.rglob('report*.exe'))
        report_exe = sorted(report)[-1]
        return report_exe

    @classmethod
    def _cmg_files(cls, sr3_file):
        sr3_file = Path(sr3_file)
        root = sr3_file.parent
        basename = root / sr3_file.stem
        basename = cls._basename(
            sr3=basename.with_suffix(".sr3"),
            rwd=basename.with_suffix(".rwd"),
            rwo=basename.with_suffix(".rwo"),
        )
        return basename

    def _tpl_to_rwd(self):
        with open(self.template, 'r', encoding='UTF-8') as tpl:
            content = Template(tpl.read())
        with open(self.basename.rwd, 'w', encoding='UTF-8') as file:
            content = content.substitute(SR3FILE=self.basename.sr3.name)
            file.write(content)

    def __call_report_exe(self):
        try:
            logging.info(f'Running {self.basename.sr3.stem} in Results Report...')
            report_command = [self._report_exe,
                            '-f',
                            self.basename.rwd.name,
                            '-o',
                            self.basename.rwo.name]
            self.procedure = subprocess.run(
                report_command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                cwd=self.basename.sr3.parent.resolve(),
                shell=True,
                check=True
            )
            logging.info('Complete!')
        except subprocess.CalledProcessError as error:
            print(f"Report não pode ser executador, verificar: {error}")

    def run_report(self, sr3:str, rwo_exist_ok=False):
        self.basename = self._cmg_files(sr3)
        if not rwo_exist_ok and self.basename.rwo.is_file():
            logging.warning(f"O arquivo {self.basename.rwo.stem} já existe e não será sobrescrito.")
            return None
        if self.template:
            self._tpl_to_rwd()
        self.__call_report_exe()
        return 1

    @staticmethod
    def _atoi(text):
        return int(text) if text.isdigit() else text

    def _natural_keys(self, text):
        return [self._atoi(c) for c in re.split(r'(\d+)', text)]

    def _parse_rwo(self, rwo_file):
        attrs = dict()
        variables = dict()
        with open(rwo_file, 'r+', encoding='UTF-8') as file:
            line = file.readline()
            dados = []
            variables = []
            parameters = []
            coords = dict()
            while line != '':
                if line.startswith('TABLE'):
                    table_number = line.split()[-1]
                    file_name = file.readline().split()[-1]
                    attrs['File'] = file_name
                    line = file.readline()
                    parameter = '_'.join(line.split()[1:-2]).lower()
                    attrs[f'Table_{table_number}'] = parameter
                    coords_aux = [file.readline().split()[0]]
                    coords_aux = file.readline().split()
                    line = file.readline().split()
                    unit_time, parameter_unit = line[0], line[1]
                    attrs[f'Time_unit'] = unit_time
                    attrs[f'{parameter}'] = parameter_unit
                    line = file.readline()
                    if parameter.startswith('water'):
                        well_type = coords_aux[0].split('-')[0].lower()
                        parameter += f'_{well_type}'
                if line[0].isdigit():
                    values = [float(i) for i in line.split()]
                    dados.append(values)
                    line = file.readline()
                if line.startswith('TABLE') or not line:
                    parameters.append(parameter)
                    dados = np.array(dados)
                    variables.append(dados[:, 1:])
                    time = dados[:, 0]
                    dados = []
            coords['time'] = time
            prod = xr.Dataset(
                data_vars=variables,
                coords=coords,
                attrs=attrs
            )

    def read_rwo(self, rwo_file:str=None):
        if not rwo_file:
            rwo_file = self.basename.rwo
        with open(rwo_file, 'r+', encoding='UTF-8') as rwo:
            self.rwo = pd.read_csv(rwo, sep="\t", index_col=False,
                                    usecols=np.r_[:5], skiprows=np.r_[0, 1, 3:6])

    def add_cash_flow(self, prices):
        self.rwo = self.rwo.rename(
                columns={
                    'TIME': 'time',
                    'Period Oil Production - Monthly SC': "oil_prod",
                    'Period Gas Production - Monthly SC': "gas_prod",
                    'Period Water Production - Monthly SC': "water_prod",
                    'Period Water Production - Monthly SC.1': "water_inj"
                }
            )
        production = self.rwo.loc[:, ['oil_prod',
                                       'gas_prod',
                                       'water_prod',
                                       'water_inj']]
        self.rwo['cash_flow'] = production.mul(prices).sum(axis=1)
        # Ajuste dos casos que rodaram com 21 anos
        #### APAGAR DEPOIS
        self.rwo = self.rwo[self.rwo['time'] <= 7305]

    def npv(self, tma, prices):
        self.add_cash_flow(prices)
        periodic_rate = ((1 + tma) ** (1 / 365.25)) - 1
        time = self.rwo["time"].to_numpy()
        time = self.rwo["time"].to_numpy()
        tax = 1 / np.power((1 + periodic_rate), time)
        return np.sum(self.rwo.cash_flow.to_numpy() * tax)

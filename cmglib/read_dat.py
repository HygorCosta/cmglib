import os
import re
from pathlib import Path
from collections import namedtuple


class DatReader:

    def __init__(self, dat_file) -> None:
        self.dat_file = Path(dat_file)
        self.res_desc = None
        self.pvt = None
        self.rtype = None
        self.well_control = None
        self.well_location = None

    def read(self):
        with open(self.dat_file, 'r', encoding='UTF-8') as file:
            text = file.read()
            self.res_desc = self.reservoir_description(text)
            self.pvt = self.fluid_model(text)
            self.rtype = self.rock_type(text)
            self.well_control = self.search_well_control(text)
            self.well_location = self.search_well_location(text)

    @staticmethod
    def is_comment_or_blank(line):
            line = line.rstrip()
            if line:
                if not re.match(r'\s*\*\*', line):
                    return False
            return True

    def reservoir_description(self, text):
        match = re.search(r'\n\*?GRID.*\n\*?MODEL', text, re.S)
        if match:
            reservoir = match[0]
            grid_dim = self._search_grid(reservoir)
            grid_inc =  self._search_grid_include(reservoir)
            perm_inc =  self._search_parameters(text, 'PERMI')
            poro_inc =  self._search_parameters(text, 'POR')
            ntg_inc = self._search_parameters(text, 'NETGROSS')
            pinch_inc = self._search_parameters(text, 'PINCHOUTARRAY')
            transf_inc = self._search_transf(text)
            falhas_inc = self._search_falhas(text)
            res_dat = namedtuple('res_dat', ['grid_dim', 'grid',\
                                                'perm', 'por',\
                                                    'ntg', 'transf',\
                                                    'falhas', 'pinch'])
            return res_dat(grid_dim, grid_inc, perm_inc, poro_inc,
                           ntg_inc, transf_inc, falhas_inc, pinch_inc)
    @staticmethod
    def _search_grid(text):
        grid_pat = r'\n\*?GRID\s+\*?\w+\s+\d+\s*\d+\s*\d+'
        grid_line = re.search(grid_pat, text, re.S)[0]
        grid_line = grid_line.split()[-3:]
        return tuple([eval(i) for i in grid_line])

    @staticmethod
    def _search_grid_include(text):
        grid_inc_pat = r'\n\*?GRID.*?INCLUDE\s+\'.*?\''
        grid_inc = re.search(grid_inc_pat, text, re.S)[0]
        return grid_inc.split()[-1][1:-1]

    @staticmethod
    def _search_parameters(text, parameter):
        #TODO: ampliar para caso generico
            perm_pat = f'\n\s*\*?{parameter}\s*\*?ALL.*?\n\s*\*?INCLUDE\s+\'.*?\''
            perm_inc = re.search(perm_pat, text, re.S)[0]
            return perm_inc.split()[-1][1:-1]

    @staticmethod
    def _search_transf(text):
         transf_pat = r'INCLUDES/REALIZATIONS/OLYMPUS_\d/TRANSF_\d_\d\.inc'
         return re.findall(transf_pat, text)

    @staticmethod
    def _search_falhas(text):
         transf_pat = r'INCLUDES/FALHAS/FALHA_\d\.inc'
         return re.findall(transf_pat, text)

    def fluid_model(self, text):
        fluid_pat = r'\n\*?MODEL.*?\.inc\''
        fluid_inc = re.search(fluid_pat, text, re.S)[0]
        return fluid_inc.split()[-1][1:-1]

    def rock_type(self, text):
        rock_pat = r'\n\*?RTYPE.*?\.inc\''
        rock_inc = re.search(rock_pat, text, re.S)[0]
        return rock_inc.split()[-1][1:-1]

    def search_well_control(self, text):
        w_pat = r'\*?INCLUDE\s\'INCLUDES/WELL_CONTROL/.*?\.inc'
        w_inc = re.search(w_pat, text, re.S)[0]
        return w_inc.split()[-1][1:-1]

    def search_well_location(self, text):
        w_pat = r'\*?INCLUDE\s\'INCLUDES/REALIZATIONS/OLYMPUS_\d/WELL_OLYMPUS_\d.*?\.inc'
        w_inc = re.search(w_pat, text, re.S)[0]
        return w_inc.split()[-1][1:-1]

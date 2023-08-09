import h5py
import numpy as np
import os
import pandas as pd
import re
import xarray as xr
from collections import namedtuple
from functools import lru_cache



class Results:

    __General = namedtuple("General", "sim_info timetable units_table units_conversion_table name_record_table")
    __Extension = namedtuple("Extension", "basename dat out sr3 rwd rwo log")

    def __init__(self, basename: str) -> None:
        basename = os.path.splitext(basename)[0]
        self.cmgfile = self.__Extension(
            basename,
            basename + ".dat",
            basename + ".out",
            basename + ".sr3",
            basename + ".rwd",
            basename + ".rwo",
            basename + ".log",
        )

    def _parse_include_files(self, dat_file=None):
        """Parse simulation file for *INCLUDE files and return a list

        Args:
            dat_file (str, optional): files path. Defaults to None.

        Returns:
            list[str]: include files path
        """
        if dat_file is None:
            dat_file = self.cmgfile.dat

        with open(dat_file, 'r') as dat:
            lines = dat.read()

        pattern = r'\n\s*[*]?\s*include\s+[\'|"]([\.:\w/-]+)[\'|"]'
        return re.findall(pattern, lines, flags=re.IGNORECASE)

    def parse_general_info(self):
        """Parse information from SR3 General Subtree
        """
        if os.path.isfile(self.cmgfile.sr3):
            with h5py.File(self.cmgfile.sr3, 'r') as f:
                sim_info = {
                    i: v.decode('utf-8', errors='ignore') for i, v in f.attrs.items()
                }
                timetable = pd.DataFrame(f['General/MasterTimeTable'][()]).set_index('Index')
                units_table = pd.DataFrame(f['General/UnitsTable'][()])
                units_conversion_table = pd.DataFrame(
                    f['General/UnitConversionTable'][()]
                )
                param = ['Dimensionality', 'Output Unit', 'Internal Unit']
                units_table.loc[:, param] = units_table.loc[
                    :, param
                ].applymap(lambda x: x.decode('utf-8', errors='ignore'))

                units_conversion_table.loc[:, 'Unit Name'] = units_conversion_table.loc[
                    :, 'Unit Name'
                ].apply(lambda s: s.decode('utf-8', errors='ignore'))

                name_record_table = pd.DataFrame(
                    f['General/NameRecordTable'][()]
                ).applymap(
                    lambda x: x.decode(errors='ignore') if type(x) == bytes else x
                )

                self.general = self.__General(
                    sim_info,
                    timetable,
                    units_table,
                    units_conversion_table,
                    name_record_table
                )

    @lru_cache(maxsize=512)
    def _get_dimensionality(self, prop):
        """Get dimensionality of property from General/NameRecordTable
        on loaded SR3 file.

        Args:
            prop (str): property

        Returns:
            [(int, int)]: dimensionalities of Numerator and Denominator
            of property unit
        """
        if prop not in self.general.name_record_table.Keyword.values:
            return None, None

        dims = self.general.name_record_table[
            self.general.name_record_table.Keyword == prop
        ].Dimensionality.values[0]
        dims = [int(i) for i in dims.replace('-', '').split('|') if i.isdigit()]

        if len(dims) == 2:
            num_dim, den_dim = dims
        elif len(dims) == 1:
            num_dim = dims[0]
            den_dim = None
        else:
            num_dim = None
            den_dim = None

        return num_dim, den_dim

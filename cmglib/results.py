import h5py
import numpy as np
import os
import pandas as pd
import re
import xarray as xr
from collections import namedtuple, defaultdict
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

    @lru_cache(maxsize=512)
    def _get_output_unit(self, prop):
        num_dim, den_dim = self._get_dimensionality(prop)
        if num_dim is None:
            return ""

        num_inunit, num_outunit = self.general.units_table.loc[
            self.general.units_table.Index == num_dim,
            ["Internal Unit", "Output Unit"]
        ].values[0]

        if den_dim is not None:
            den_inunit, den_outunit = self.general.units_table.loc[
                self.general.units_table.Index == den_dim,
                ["Internal Unit", "Output Unit"]
            ].values[0]
        else:
            den_inunit, den_outunit = None, None

        if den_outunit:
            return num_outunit + "/" + den_outunit
        else:
            return num_outunit

    @lru_cache(maxsize=512)
    def _conversion_params(self, dimensionality, inunit, outunit):
        """Get unit conversion gain and offset parameters given a dimensionallity, input and output units

        Arguments:
            dimensionality {int} -- Property Dimensionality
            inunit {str} -- input unit with which data is stored for the desired property
            outunit {str} -- output unit for the desired property selected on the basefile

        Returns:
            (gain, offset) {float, float} -- gain and offset unit conversion parameters
        """

        unit_filter_in = (
            self.general.units_conversion_table["Unit Name"] == inunit
        ) & (self.general.units_conversion_table.Dimensionality == dimensionality)
        gain_in, offset_in = self.general.units_conversion_table.loc[
            unit_filter_in, ["Gain", "Offset"]
        ].values[0]

        unit_filter_out = (
            self.general.units_conversion_table["Unit Name"] == outunit
        ) & (self.general.units_conversion_table.Dimensionality == dimensionality)
        gain_out, offset_out = self.general.units_conversion_table.loc[
            unit_filter_out, ["Gain", "Offset"]
        ].values[0]

        return gain_out / gain_in, offset_out / gain_out - offset_in / gain_in

    @lru_cache(maxsize=512)
    def _conversion_gain_offset(self, prop):
        """Get unit conversion gain and offset parameters given a desired property

        Arguments:
            prop {str} -- property

        Returns:
            (gain, offset) {float, float} -- gain and offset unit conversion parameters
        """

        # Numerator and Denominator Dimensionalities
        num_dim, den_dim = self._get_dimensionality(prop)

        if num_dim is None:
            return 1.0, 0.0

        num_inunit, num_outunit = self.general.units_table.loc[
            self.general.units_table.Index == num_dim,
            ['Internal Unit', 'Output Unit']
        ].values[0]

        if den_dim is not None:
            den_inunit, den_outunit = self.general.units_table.loc[
                self.general.units_table.Index == den_dim,
                ['Internal Unit', 'Output Unit']
            ].values[0]
        else:
            den_inunit, den_outunit = None, None

        num_gain, num_offset = self._conversion_params(num_dim, num_inunit, num_outunit)

        if den_dim is not None:
            den_gain, den_offset = self._conversion_params(
                den_dim, den_inunit, den_outunit
            )
        else:
            den_gain, den_offset = 1.0, 0.0

        return den_gain / num_gain, den_offset / den_gain - num_offset / num_gain

    def read_timeseries(self, timeseries_group, timeseries_variables=None,
                        sr3_file=None):
        """Read Timeseries from SR3 file

        Arguments:
            timeseries_group {str} -- [WELLS, GROUPS, SECTORS, SPECIAL HISTORY, LEASES, WELL-TEST]

        Keyword Arguments:
            timeseries_variables {list} -- List of timeseries variables to be read (default: {None})
            sr3_file {str} -- SR3 file (default: {basename.sr3})

        Returns:
            {xarray.DataSet} -- DataSet containing group timeseries
        """
        if sr3_file is None:
            sr3_file = self.cmgfile.sr3
            if os.path.isfile(sr3_file):
                self.parse_general_info()
            else:
                print("No available sr3 file")

        timeseries_group = timeseries_group.upper()

        with h5py.File(sr3_file, 'r') as file:
            if (
                (file[f'/TimeSeries/{timeseries_group}'].get('Variables') is None)
                or (file[f'/TimeSeries/{timeseries_group}'].get('Origins') is None)
                or (file[f'/TimeSeries/{timeseries_group}'].get('Timesteps') is None)
            ):
                print(f'No variables detected for Group Type {timeseries_group}')
                return None

            aliases = {
                "OILRATSC": "Qo",
                "OILRATRC": "Qo_RC",
                "OILVOLSC": "Np",
                "WATRATSC": "Qw",
                "WATRATRC": "Qw_RC",
                "GASVOLSC": "Gp",
                "GASRATSC": "Qg",
                "GASRATRC": "Qg_RC",
                "WATVOLSC": "Wp",
                "BHP": "BHP",
                "WTTD": "tD",
                "WTPD": "pD",
                "WTAGART": "tAgarwal",
                "WTAGARP": "pAgarwal",
                "WTMDH": "tMDH",
                "WTHORNER": "tHorner",
            }

            if timeseries_variables is None:
                variables = file[f'/TimeSeries/{timeseries_group}/Variables'][()].astype(str)
                variables_and_indexes = list(enumerate(variables))
            elif isinstance(timeseries_variables, list):
                reverse_aliases = self.general.name_record_table[['Keyword', 'Name']]
                reverse_aliases = dict(
                    zip(reverse_aliases.Name, reverse_aliases.Keyword)
                )
                reverse_aliases.update({v: k for k, v in aliases.items()})

                timeseries_variables = [
                    reverse_aliases.get(v, v) for v in timeseries_variables
                ]
                variables = [
                    v for v in file[f'/TimeSeries/{timeseries_group}/Variables'][()].astype(str)
                    if v in timeseries_variables
                ]
                variables_and_indexes = [
                    (iv, v) for (iv, v) in enumerate(
                        file[f'/TimeSeries/{timeseries_group}/Variables'][()].astype(str)
                    )
                    if v in timeseries_variables
                ]
            else:
                raise(
                    TypeError('timeseries_variables muste be a list of variable names or None,\
                              to read all variables avaliable.')
                )

            origins = file[f'/TimeSeries/{timeseries_group}/Origins'][()].astype(str)
            timesteps = file[f'/TimeSeries/{timeseries_group}/Timesteps'][()]

            times = {
                col: self.general.timetable.loc[timesteps, col].values
                for col in self.general.timetable.columns
            }

            ds = xr.Dataset(coords={'origin': origins, **times})

            if 'Date' in times.keys():
                date_var = 'Date'
            else:
                date_var = times.keys()[-1]

            data_raw = file[f'/TimeSeries/{timeseries_group}/Data'][()]

            for iv, var in variables_and_indexes:
                if var in self.general.name_record_table.Keyword.values:
                    var_name = self.general.name_record_table[
                        self.general.name_record_table.Keyword == var
                    ]['Name'].values[0]
                    gain, offset = self._conversion_gain_offset(var.split(" ")[0])
                else:
                    var_name = var
                    gain, offset = self._conversion_gain_offset(var.split(" ")[0])

                ds[var_name] = (
                    (date_var, 'origin'),
                    data_raw[:, iv, :] * gain + offset,
                )
                unit = self._get_output_unit(var)
                ds[var_name].attrs['units'] = unit if unit is not None else ""

            for var, alias in aliases.items():
                if var in variables:
                    var_name = self.general.name_record_table[
                        self.general.name_record_table.Keyword == var
                    ]['Name'].values[0]
                    ds[alias] = ((date_var, 'origin'), ds[var_name].values)
                    ds[alias].attrs['units'] = self._get_output_unit(var)

            if ('OILRATSC' in variables) and ('WATRATSC' in variables):
                with np.errstate(invalid='ignore'):
                    ds['BSW'] = (
                        (date_var, 'origin'),
                        ds['Qw'].values / (ds['Qw'].values + ds['Qo'].values),
                    )
                    ds['BSW'].attrs['units'] = ''

            if ("OILRATSC" in variables) & ("GASRATSC" in variables):
                with np.errstate(invalid="ignore"):
                    ds["RGO"] = (
                        (date_var, "origin"),
                        ds["Qg"].values / ds["Qo"].values,
                    )
                    ds["RGO"].attrs["units"] = ""

            if 'Date' in times.keys():
                date = pd.to_datetime(
                    ds.Date.values, format="%Y%m%d"
                ) + pd.to_timedelta(ds.Date.values - np.floor(ds.Date.values), unit='D')
                ds.coords['Date'] = date

            return ds

    def list_properties(self, timesteps=False, descriptions=False, sr3_file=None):
        """List grid properties from SR3 file

        Keyword Arguments:
            timesteps {bool} -- Return recorded timesteps for each property? (default: False)
            descriptions {bool} -- Return keyword descriptions? (default: False})
            sr3_file {str} -- SR3 filename (default: {basename.sr3})

        Returns:
            {list | pd.DataFrame} -- List of keywords available
        """
        if sr3_file is None:
            sr3_file = self.cmgfile.sr3
        if os.path.isfile(sr3_file):
            self.parse_general_info()
        else:
            raise ValueError("No available sr3 file")

        prop_dict = defaultdict(list)
        with h5py.File(sr3_file, 'r') as f:
            for k in f['SpatialProperties/000000/GRID/'].keys():
                prop_dict[k].append(0)
            timestep_props = [(t, t_groups) for t, t_groups in f["SpatialProperties"].items() if t.isdigit()]
            for t, t_groups in timestep_props:
                for k in t_groups.keys():
                    if k != 'GRID':
                        prop_dict[k].append(int(t))

        name_record = self.general.name_record_table.set_index('Keyword').to_dict()

        if descriptions:
            return pd.DataFrame(sorted([(var, name_record['Name'][var], name_record['Long Name'][var]) for var in (prop_dict.keys()) if (var in name_record['Packing'].keys() and name_record['Packing'][var] == 'p')]),
                            columns=['keyword', 'name', 'long_name'])
        if timesteps:
            return {k: pd.DataFrame(dict(**{'idx': range(len(v)), 'idx_timetable': v}, **{
                col: self.general.timetable.loc[v, col].values
                for col in self.general.timetable.columns
            })) for (k, v) in  prop_dict.items()}
        else:
            return sorted([var for var in (prop_dict.keys()) if (var in name_record['Packing'].keys() and name_record['Packing'][var] == 'p')])

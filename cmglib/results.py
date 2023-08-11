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

    def read_properties(self, grid_properties=None, timestep_idx=None, sr3_file=None, twophi2k=False):
            """Read grid properties from SR3 file

            Keyword Arguments:
                grid_properties {list} -- Property or list of properties to be read (default: {None})
                timestep_idx {list} -- List of timestep indices to be read (default: {None})
                sr3_file {str} -- SR3 filename (default: {basename.sr3})
                twophi2k {bool} -- Load matrix-fracture (2phi2k) properties (default: False)

            Returns:
                {xarray.DataSet} -- DataSet with desired grid properties
            """

            if sr3_file is None:
                sr3_file = self.cmgfile.sr3
                if os.path.isfile(sr3_file):
                    self.parse_general_info()
                else:
                    print("No available sr3 file")

            if isinstance(grid_properties, str):
                grid_properties = [grid_properties]

            # Read Properties
            with h5py.File(sr3_file, "r") as f:
                dimI = int(f["SpatialProperties/000000/GRID/IGNTID"][0])
                dimJ = int(f["SpatialProperties/000000/GRID/IGNTJD"][0])
                dimK = int(f["SpatialProperties/000000/GRID/IGNTKD"][0])
                ind_valid_cells = f["SpatialProperties/000000/GRID/IPSTCS"][()] - 1
                ind_null_cells = np.setdiff1d(
                    np.arange(dimI * dimJ * dimK), ind_valid_cells
                )

                # Check if model is 2phi2k - Matrix-Fracture
                if ("TRMMF" in list(f["SpatialProperties/000000"])) or twophi2k:
                    twophi2k = True
                    dims = (dimI, dimJ, dimK, 2)
                    dims_size = dimI * dimJ * dimK * 2
                    coords = {
                        "I": np.arange(dimI),
                        "J": np.arange(dimJ),
                        "K": np.arange(dimK),
                        "model": ["matrix", "fracture"],
                    }
                else:
                    twophi2k = False
                    dims = (dimI, dimJ, dimK)
                    dims_size = dimI * dimJ * dimK
                    coords = {
                        "I": np.arange(dimI),
                        "J": np.arange(dimJ),
                        "K": np.arange(dimK),
                    }

                # List and filter timesteps
                timestep_props = [(t, t_groups) for t, t_groups in f["SpatialProperties"].items() if t.isdigit()]
                if timestep_idx is not None:
                    for i in timestep_idx:
                        if abs(i) >= len(timestep_props):
                            raise ValueError(f'Index {i} out of range')
                    timestep_idx = [i % len(timestep_props) for i in timestep_idx]
                    timestep_props = [(t, t_groups) for i, (t, t_groups) in enumerate(timestep_props) if i in timestep_idx]

                # read properties to list of props and timesteps
                timesteps = []
                props = {}
                prop_times = {}
                sectors = {}
                units = {}
                for t, t_groups in timestep_props:
                    timesteps.append(int(t))
                    for v, v_items in t_groups.items():
                        if (v != "GRID") and ((grid_properties is None) or (v in grid_properties)):
                            try:
                                gain, offset = self._conversion_gain_offset(v)
                                # print(f'v: {v} gain: {gain} offset: {offset}')
                                prop = np.zeros(dims_size)

                                if len(v_items[()]) == len(ind_valid_cells):
                                    prop[ind_valid_cells] = v_items[()] * gain + offset
                                    prop[ind_null_cells] = np.nan
                                elif len(v_items[()]) == len(prop):
                                    prop = v_items[()] * gain + offset
                                else:
                                    raise ValueError

                                prop = prop.reshape(dims, order="F")

                                if v not in props.keys():
                                    props[v] = []
                                    prop_times[v] = []
                                props[v].append(np.nan_to_num(prop))
                                prop_times[v].append(int(t))
                                units[v] = self._get_output_unit(v)
                            except:
                                pass
                                # print('Could not read property {}'.format(v))
                        elif v == "GRID":
                            for vg, vg_items in v_items.items():
                                if (vg != "ISECTGEOM") and ((grid_properties is None) or(vg in grid_properties)):
                                    try:
                                        gain, offset = self._conversion_gain_offset(vg)
                                        # print(f'vg: {vg} gain: {gain} offset: {offset}')
                                        prop = np.zeros(dims_size)

                                        if len(vg_items[()]) == len(ind_valid_cells):
                                            prop[ind_valid_cells] = (
                                                vg_items[()] * gain + offset
                                            )
                                            prop[ind_null_cells] = np.nan
                                        elif len(vg_items[()]) == len(prop):
                                            prop = vg_items[()] * gain + offset
                                        else:
                                            raise ValueError

                                        prop = prop.reshape(dims, order="F")

                                        if vg not in props.keys():
                                            props[vg] = []
                                            prop_times[vg] = []
                                        props[vg].append(np.nan_to_num(prop))
                                        prop_times[vg].append(int(t))
                                        units[vg] = self._get_output_unit(vg)
                                    except:
                                        pass
                                        # print('Could not read property {}'.format(vg))
                                elif (vg == "ISECTGEOM") and ((grid_properties is None) or (("ISECTGEOM" in grid_properties) or ("SECTORARRAY" in grid_properties))):
                                    try:
                                        sect_array = f["SpatialProperties/000000/GRID"][
                                            "ISECTGEOM"
                                        ][()]
                                        sect_index = np.where(sect_array < 0)[0]
                                        dimSect = len(sect_index)
                                        sect_array_bool = (
                                            np.zeros(
                                                (dimSect, dims_size), dtype=np.bool
                                            )
                                            * np.nan
                                        )

                                        for i in range(dimSect):
                                            upper_bound = (
                                                sect_index[i + 1]
                                                if i + 1 < dimSect
                                                else None
                                            )
                                            sect_array_bool[
                                                i,
                                                sect_array[
                                                    (sect_index[i] + 1) : upper_bound
                                                ],
                                            ] = True

                                        sectors["coords"] = {
                                            "sector": range(dimSect),
                                            "I": np.arange(dimI),
                                            "J": np.arange(dimJ),
                                            "K": np.arange(dimK),
                                        }
                                        sectors["values"] = sect_array_bool.reshape(
                                            (dimSect,) + dims, order="F"
                                        )
                                    except:
                                        pass

                times = {
                    col: self.general.timetable.loc[timesteps, col].values
                    for col in self.general.timetable.columns
                }
                ds = xr.Dataset(coords=dict(coords, **times))

                for v, value_list in props.items():
                    t = prop_times[v]
                    if t == [0]:
                        ds[v] = (tuple(coords.keys()), props[v][0])
                        ds[v].attrs["units"] = units[v] if units[v] is not None else ""
                    elif t == timesteps:
                        ds[v] = (("Date",) + tuple(coords.keys()), np.stack(props[v]))
                        ds[v].attrs["units"] = units[v] if units[v] is not None else ""

                if (grid_properties is None) or ("NULL" in grid_properties):
                    null = np.zeros(dims_size)
                    null[ind_null_cells] = 1
                    null = null.reshape(dims, order="F")
                    ds["NULL"] = (tuple(coords.keys()), null)
                    ds["NULL"].attrs["units"] = ""

                if sectors:
                    ds = ds.assign_coords(coords=sectors["coords"])
                    ds = ds.assign(
                        SECTORARRAY=(("sector",) + tuple(coords.keys()), sectors["values"])
                    )

                if "Date" in times.keys():
                    date = pd.to_datetime(
                        ds.Date.values, format="%Y%m%d"
                    ) + pd.to_timedelta(ds.Date.values - np.floor(ds.Date.values), unit="D")
                    ds.coords["Date"] = date

            return ds

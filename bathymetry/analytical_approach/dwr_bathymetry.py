# -*- coding: utf-8 -*-

# %%% Import -------------------------------------------------------------------

from abc import ABC, abstractmethod
import arcpy
import os
import shutil
import sys

sys.path.append(r'common')

import file_management as fm
import numpy as np
import yaml
import copy
import glob
import pathlib
import arcinfo
import textwrap
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import subprocess
import re
import scipy.stats as stats
import itertools


# NOTE: All variable names and functionality are subject to change. This is a pre-release
# version of this module and is currently under development. THIS MODULE AND ASSOCIATED
# SCRIPT FILES ARE PROVIDED ON AN "AS-IS" BASIS ONLY.

# NOTE: All variable names and functionality are subject to change.

# TODO: Make unit tests or control sites with some project locations.

# TODO: Currently have partial PEP 8 formatting with some long lines
#  extended. Full formatting to be done once more sites have been tested.

# TODO: Add docstrings to functions once more sites have been tested.

# %%% Main Body ----------------------------------------------------------------

def plot_linear_fit(data, case, **kws):
    df = data.copy()
    df = df.dropna()
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['POINT_Z'], df[case])
    ax = plt.gca()
    r_squared = r_value ** 2
    ax.plot(df['POINT_Z'], (slope * df['POINT_Z'] + intercept), '-', color='#dc143c')
    if intercept >= 0:
        ax.annotate(f'y = {slope:0.2f}x + {intercept:0.1f}\nR² = {r_squared:0.2f}', (0, 1), (10, -10),
                    xycoords='axes fraction', textcoords='offset points', ha='left', va='top')
    elif intercept < 0:
        ax.annotate(f'y = {slope:0.2f}x - {(intercept * -1):0.1f}\nR² = {r_squared:0.2f}', (0, 1), (10, -10),
                    xycoords='axes fraction', textcoords='offset points', ha='left', va='top')
    return slope, intercept, r_squared


def tabulate_qaqc_stats(data, case, point_type=None):
    df = data.copy()
    df = df.dropna()
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['POINT_Z'], df[case])
    r_squared = r_value ** 2
    diff = df[case] - df['POINT_Z']
    mean_error = diff.mean()
    abs_diff = diff.abs()
    mean_abs_error = abs_diff.mean()
    pct_error = diff.sum() / df['POINT_Z'].abs().sum() * 100
    pct_abs_error = abs_diff.sum() / df['POINT_Z'].abs().sum() * 100
    rmse = ((diff ** 2).sum() / df['POINT_Z'].shape[0]) ** 0.5
    result_dict = {}
    result_dict['Slope'] = slope
    result_dict['Intercept'] = intercept
    result_dict['R-Squared'] = r_squared
    result_dict['Mean Error'] = mean_error
    result_dict['Mean Absolute Error'] = mean_abs_error
    result_dict['Percent Error'] = pct_error
    result_dict['Percent Absolute Error'] = pct_abs_error
    result_dict['Root Mean Square Error'] = rmse
    name = f'{case}_{point_type}' if point_type is not None else f'{case}'
    result_df = pd.DataFrame(result_dict, index=[name]).T
    return result_df, diff


# Used in analysis_step decorators to nullify functions that have already been run.
def already_run(analysis, name, addl=''):
    print(rf'{analysis.reach.name}\{analysis.name}{addl}: {name}() has already been run.')


# Used in analysis_step decorators to point back to the original analysis object and append the record.
# identifier must result in a unique text file name so the script does not mistake actions
# from different objects as being the same action.
def inner_record(obj, analysis, func, identifier, command_str, addl, *args, **kwargs):
    id_file = rf'{analysis.working_directory.full_path}\{identifier}.txt'
    id_file_exists = os.path.exists(id_file)

    # Read the last line of the file. If the command text matches, do not write it again!
    # https://stackoverflow.com/questions/46258499/how-to-read-the-last-line-of-a-file-in-python
    if not os.path.exists(analysis.history_path):
        with open(analysis.history_path, 'w') as f:
            f.write(rf'Command history for {analysis.name}' + '\n' + '-' * 72)

    with open(analysis.history_path, 'r') as f:
        for line in f:
            pass
        last_line = line

    if command_str != last_line:
        with open(analysis.history_path, 'a') as f:
            f.write('\n' + rf'{command_str}')

    if not id_file_exists:
        # If this is a new entry, write the class name, method name,
        # and parameters to the registry.
        argstring = ', '.join([str(arg) for arg in args if arg is not None])
        kwstring = ', '.join([f'{kw}={val}' for kw, val in kwargs.items()])
        print(rf'Running {command_str}')
        analysis._registry.append(command_str)
        res = func(obj, *args, **kwargs)
        with open(id_file, 'w') as f:
            f.write('1')
    else:
        # If this is an existing analysis or function was already run,
        # substitute the method
        # for a function that prints a message that it was already
        # done and do not perform any actions.
        res = already_run(analysis, func.__name__, addl)
    return res


class ProjectionError(BaseException):
    pass


class GISChecker:
    def __init__(self, print_wkt=True):
        self.register = []
        self.last_wkt = None
        self.print_wkt = print_wkt

    def check_in(self, gis_file_path):
        print(gis_file_path)
        self.register.append(gis_file_path)
        wkt = arcpy.Describe(gis_file_path).spatialReference.exportToString()
        if self.print_wkt:
            print('\t' + wkt)
        return gis_file_path

    def export_file_paths(self, txt_file_name):
        with open(rf'{txt_file_name}.txt', 'w') as f:
            f.write('\n'.join([str(i) for i in self.register]))


class Project:
    def __init__(self, folder_path, base_dem_path_10, interpolation_script_directory, path_to_rscript, path_to_run_r):
        self._project_folder_path = folder_path
        self.base_dem_path_10 = base_dem_path_10
        self.interpolation_script_directory = interpolation_script_directory
        self.path_to_rscript = path_to_rscript
        self.path_to_run_r = path_to_run_r
        # self.breaklines_fc_paths = breaklines_fc_paths
        # self.levees = levee_path_list
        self._reaches = {}

    def __getitem__(self, reach_name):
        if type(reach_name) == str:
            return self._reaches[reach_name]
        elif type(reach_name) in (int, slice):
            return tuple(self._reaches.values())[reach_name]

    @property
    def reaches(self):
        return self._reaches

    @property
    def project_folder_path(self):
        return self._project_folder_path

    def add_reach(self, stream_abbrev, start_rkm=None, end_rkm=None):
        if (start_rkm is not None) | (end_rkm is not None):
            reach = Reach(stream_abbrev, start_rkm, end_rkm)
        else:
            reach = NodeReach(stream_abbrev)
        reach.project = self
        self._reaches[reach.name] = reach


class Reach:
    def __init__(self, stream_abbrev, start_rkm, end_rkm, version=None):
        self._stream_abbrev = stream_abbrev
        self._start_rkm = start_rkm
        self._end_rkm = end_rkm
        self._analyses = {}
        self.project = None
        self.version = version

    @property
    def name(self):
        name = f'{self._stream_abbrev}_{self._start_rkm:0>4.1f}_{self._end_rkm:0>4.1f}'.replace('.', '_')
        if self.version is not None:
            name += '_v' + str(self.version)
        return name

    @property
    def analyses(self):
        return tuple(self._analyses)

    @property
    def working_dir(self):
        path = rf'{self.project.project_folder_path}\{self.name}'
        return path

    def set_up_folder(self):
        if os.path.exists(self.working_dir):
            pass
        else:
            os.mkdir(self.working_dir)
            os.mkdir(rf'{self.working_dir}\_inputs')

    def add_analysis(self, analysis):
        analysis.reach = self
        analysis.number = len(self._analyses) + 1
        self._analyses[analysis.name] = analysis
        analysis.set_up_folder()

    def __getitem__(self, name):
        if type(name) == str:
            return self._analyses[name]
        elif type(name) in (int, slice):
            return tuple(self._analyses.values())[name]


class NodeReach(Reach):
    def __init__(self, stream_abbrev):
        super().__init__(stream_abbrev, None, None)

    @property
    def name(self):
        return self._stream_abbrev


# TODO: Class explosion present due to needs to capture many different flavors of ObservationPoints in a short period
#  of time. Refactor to more concisely describe relationship between bathymetry (singlebeam/multibeam and file sources),
#  LiDAR, automatically generated shoreline points, and augmented points. An updated UML diagram will be helpful.

# TODO: Change from_multibeam to now say from_coverage. This is a more accurate description of what is going on.
class ObservationPoints:
    def __init__(self, data_short_name, data_long_name, precision=None, fmt='.', ms=20, zorder=None, fillstyle='full',
                 alpha=1, color=None, point_erase_poly=None):
        self.mesh = None
        self.precision = precision
        self.data_short_name = data_short_name
        self.data_long_name = data_long_name
        self.fmt = fmt
        self.ms = ms
        self.zorder = zorder
        self.fillstyle = fillstyle
        self.alpha = alpha
        self.color = color
        self.point_erase_poly = point_erase_poly

    def analysis_step(func):
        def inner(self, *args, **kwargs):
            identifier = rf'_{self.mesh.analysis.__class__.__name__}_{self.mesh.name}_{self.name}_{func.__name__}'
            argstring = ', '.join([arg for arg in args if arg is not None])
            kwstring = ', '.join([f'{kw}={val}' for kw, val in kwargs.items()])
            command_str = rf'{self.mesh.analysis.__class__.__name__}[{self.mesh.name}, obs_pt={self.name}].{func.__name__}({argstring}, {kwstring})'.replace(
                ', )', ')')
            # Perform function substitution here. This function is written so analysis_step can be defined in
            # attached objects and add records to the analysis history.
            res = inner_record(self, self.mesh.analysis, func, identifier, command_str,
                               rf'\{self.mesh.name}, obs_pt={self.name}', *args, **kwargs)
            return res

        return inner

    @property
    def _working_dir(self):
        path = rf'{self.mesh.analysis.working_directory.full_path}\{self.mesh.name}'
        return path

    @property
    def fc_path(self):
        pass

    @property
    def csv_path(self):
        pass

    @property
    def name(self):
        pass


class BathyPoints(ObservationPoints):
    def __init__(self, data_short_name, data_long_name, precision=1, fmt='kx', ms=4, zorder=2.5, fillstyle='full',
                 alpha=1, color=None, point_erase_poly=None):
        super().__init__(data_short_name, data_long_name, precision, fmt, ms, zorder, fillstyle, alpha, color,
                         point_erase_poly=point_erase_poly)


class SingleBeamBathyPoints(BathyPoints):
    def __init__(self, bathy_pts_fc_path, data_short_name, data_long_name, precision=1, fmt='kx', ms=4, zorder=2.5,
                 fillstyle='full', alpha=1, z_ft_name=None, z_m_name=None, color=None, point_erase_poly=None):
        """
        An object representing singlebeam bathymetry data.
        :param bathy_pts_fc_path: Path to feature class containing bathymetry points.
        :param data_short_name: (str) A shorthand name for the bathymetry survey to be used in summary statistics.
        :param data_long_name: (str) A longhand name for the bathymetry survey to be used in the main control script.
                               May be used in the future for creating legends in plots if desired.
        :param precision: (float) A value to be used to represent the precision of the survey for plotting and statistics
                          purposes. Default is 1.
        :param fmt: (str) A matplotlib marker type to be used in cross-section plots. Default is 'kx'.
        :param ms: (str) A matplotlib marker size for cross-section plotting. Default is 4.
        :param zorder: (float) A zorder for matplotlib cross-section plotting. Default is 2.5.
        :param fillstyle: (str) A matplotlib marker fill style. Default is 'full'
        :param alpha: (float) A matplotlib float value representing transparency, with 1 being completely opaque and
                      0 being completely transparent. Default is 1.
        :param z_ft_name: (str) A field containing elevation values in NAVD88 ft. Should be None if z_m_name is used.
        :param z_m_name: (str) A field containing elevation values in NAVD88 m. Should be None if z_ft_name is used.
        :param color: (str) A matplotlib color for cross-section plotting. Not currently implemented. Default is None
                            (plot using matplotlib standards).
        :param point_erase_poly: (str) A path to a feature class with polygon representing area where points should
                                 be excluded from interpolation.

        Last revised: 07/05/2023
        """
        super().__init__(data_short_name, data_long_name, precision, fmt, ms, zorder, fillstyle, alpha, color,
                         point_erase_poly=point_erase_poly)
        self.z_ft_name = z_ft_name
        self.z_m_name = z_m_name
        self.bathy_pts_fc_path = bathy_pts_fc_path

    @ObservationPoints.analysis_step
    def clip_export_points(self):
        if not os.path.exists(self._working_dir):
            os.mkdir(self._working_dir)
        temp_gdb = fm.PathedGeoDataBase(rf'{self._working_dir}\clip_{self.name}_tmp.gdb')
        temp_gdb.set_env()
        bathy_clip = rf'{self._working_dir}\{self.mesh.name}_{self.name}_clip.shp'
        if self.point_erase_poly is not None:
            arcpy.CopyFeatures_management(self.point_erase_poly, 'bathy_point_erase_poly')
            arcpy.Clip_analysis(self.bathy_pts_fc_path, self.mesh.boundary, 'temp_clip')
            arcpy.Erase_analysis('temp_clip', self.point_erase_poly, bathy_clip)
        else:
            arcpy.Clip_analysis(self.bathy_pts_fc_path, self.mesh.boundary, bathy_clip)
        if int(arcpy.GetCount_management(bathy_clip)[0]) == 0:
            return False
        for field in ['POINT_X', 'POINT_Y', 'POINT_Z']:
            if field not in [fc_field.name for fc_field in arcpy.Describe(bathy_clip).fields]:
                arcpy.AddField_management(bathy_clip, field, 'DOUBLE')
        if (self.z_ft_name is not None) & (
                'POINT_Z' not in [field.name for field in arcpy.Describe(self.bathy_pts_fc_path).fields]):
            arcpy.CalculateField_management(bathy_clip, 'POINT_Z', f'!{self.z_ft_name}!*0.3048', 'PYTHON')
        if (self.z_m_name is not None) & (
                'POINT_Z' not in [field.name for field in arcpy.Describe(self.bathy_pts_fc_path).fields]):
            arcpy.CalculateField_management(bathy_clip, 'POINT_Z', f'!{self.z_m_name}!', 'PYTHON')
        if 'POINT_X' not in [field.name for field in arcpy.Describe(self.bathy_pts_fc_path).fields]:
            arcpy.CalculateField_management(bathy_clip, 'POINT_X', '!Shape.Centroid.X!', 'PYTHON')
        if 'POINT_Y' not in [field.name for field in arcpy.Describe(self.bathy_pts_fc_path).fields]:
            arcpy.CalculateField_management(bathy_clip, 'POINT_Y', '!Shape.Centroid.Y!', 'PYTHON')
        df_in = pd.DataFrame(arcpy.da.TableToNumPyArray(bathy_clip,
                                                        [field.name for field in arcpy.Describe(bathy_clip).fields
                                                         if (field.name in ['POINT_X', 'POINT_Y', 'POINT_Z']) | (
                                                                 field.type in ['OID'])]))
        df_in.to_csv(
            rf'{self._working_dir}\{self.mesh.name}_{self.name}_clip.csv',
            index=False)
        temp_gdb.reset_env()
        return True

    @property
    def name(self):
        path = pathlib.PureWindowsPath(self.bathy_pts_fc_path).stem
        return path

    @property
    def fc_path(self):
        bathy_clip = rf'{self._working_dir}\{self.mesh.name}_{self.name}_clip.shp'
        if os.path.exists(bathy_clip):
            return bathy_clip

    @property
    def csv_path(self):
        clip_csv = rf'{self._working_dir}\{self.mesh.name}_{self.name}_clip.csv'
        if os.path.exists(clip_csv):
            return clip_csv


class SHPMultiBeamBathyPoints(BathyPoints):
    def __init__(self, bathy_pts_fc_path, data_short_name, data_long_name, precision=1, fmt='kx', ms=4, zorder=2.5,
                 fillstyle='full', alpha=1, z_ft_name=None, z_m_name=None, color=None, buffer_out='5 Meters',
                 buffer_in='-10 Meters',
                 point_erase_poly=None):
        super().__init__(data_short_name, data_long_name, precision, fmt, ms, zorder, fillstyle, alpha, color,
                         point_erase_poly=point_erase_poly)
        self.z_ft_name = z_ft_name
        self.z_m_name = z_m_name
        self.bathy_pts_fc_path = bathy_pts_fc_path
        self.buffer_out = buffer_out
        self.buffer_in = buffer_in

    @ObservationPoints.analysis_step
    def clip_export_points(self):
        if not os.path.exists(self._working_dir):
            os.mkdir(self._working_dir)
        temp_gdb = fm.PathedGeoDataBase(rf'{self._working_dir}\clip_{self.name}_tmp.gdb')
        temp_gdb.set_env()
        bathy_clip = rf'{self._working_dir}\{self.mesh.name}_{self.name}_clip.shp'
        if self.point_erase_poly is not None:
            arcpy.CopyFeatures_management(self.point_erase_poly, 'bathy_point_erase_poly')
            arcpy.Clip_analysis(self.bathy_pts_fc_path, self.mesh.boundary, 'temp_clip')
            arcpy.Erase_analysis('temp_clip', self.point_erase_poly, bathy_clip)
        else:
            arcpy.Clip_analysis(self.bathy_pts_fc_path, self.mesh.boundary, bathy_clip)
        if int(arcpy.GetCount_management(bathy_clip)[0]) == 0:
            return False
        for field in ['POINT_X', 'POINT_Y', 'POINT_Z']:
            if field not in [fc_field.name for fc_field in arcpy.Describe(bathy_clip).fields]:
                arcpy.AddField_management(bathy_clip, field, 'DOUBLE')
        if (self.z_ft_name is not None) & (
                'POINT_Z' not in [field.name for field in arcpy.Describe(self.bathy_pts_fc_path).fields]):
            arcpy.CalculateField_management(bathy_clip, 'POINT_Z', f'!{self.z_ft_name}!*0.3048', 'PYTHON')
        if (self.z_m_name is not None) & (
                'POINT_Z' not in [field.name for field in arcpy.Describe(self.bathy_pts_fc_path).fields]):
            arcpy.CalculateField_management(bathy_clip, 'POINT_Z', f'!{self.z_m_name}!', 'PYTHON')
        if 'POINT_X' not in [field.name for field in arcpy.Describe(self.bathy_pts_fc_path).fields]:
            arcpy.CalculateField_management(bathy_clip, 'POINT_X', '!Shape.Centroid.X!', 'PYTHON')
        if 'POINT_Y' not in [field.name for field in arcpy.Describe(self.bathy_pts_fc_path).fields]:
            arcpy.CalculateField_management(bathy_clip, 'POINT_Y', '!Shape.Centroid.Y!', 'PYTHON')
        df_in = pd.DataFrame(arcpy.da.TableToNumPyArray(bathy_clip,
                                                        [field.name for field in arcpy.Describe(bathy_clip).fields
                                                         if (field.name in ['POINT_X', 'POINT_Y', 'POINT_Z']) | (
                                                                 field.type in ['OID'])]))
        df_in.to_csv(rf'{self._working_dir}\{self.mesh.name}_{self.name}_clip.csv', index=False)
        arcpy.Buffer_analysis(self.bathy_pts_fc_path, 'point_coverage_tmp', self.buffer_out, dissolve_option='ALL')
        coverage_polygon = rf'{self._working_dir}\{self.mesh.name}_{self.name}_a1.shp'
        arcpy.Buffer_analysis('point_coverage_tmp', coverage_polygon, self.buffer_in, 'FULL')
        temp_gdb.reset_env()
        return True

    @property
    def coverage_polygon_fc_path(self):
        coverage_polygon = rf'{self._working_dir}\{self.mesh.name}_{self.name}_a1.shp'
        if os.path.exists(coverage_polygon):
            return coverage_polygon

    @property
    def name(self):
        path = pathlib.PureWindowsPath(self.bathy_pts_fc_path).stem
        return path

    @property
    def fc_path(self):
        bathy_clip = rf'{self._working_dir}\{self.mesh.name}_{self.name}_clip.shp'
        if os.path.exists(bathy_clip):
            return bathy_clip

    @property
    def csv_path(self):
        clip_csv = rf'{self._working_dir}\{self.mesh.name}_{self.name}_clip.csv'
        if os.path.exists(clip_csv):
            return clip_csv


class MultiBeamBathyPoints(BathyPoints):
    def __init__(self, bathy_raster_path, data_short_name, data_long_name, precision=1, fmt='kx', ms=4, zorder=2.5,
                 fillstyle='full', alpha=1, color=None, buffer_in='-5 Meters', point_erase_poly=None):
        """
        An object representing multibeam (raster) bathymetry data.

        :param bathy_raster_path: (str) Path to raster containing multibeam data.
        :param data_short_name: (str) A shorthand name for the bathymetry survey to be used in summary statistics.
        :param data_long_name: (str) A longhand name for the bathymetry survey to be used in the main control script.
                               May be used in the future for creating legends in plots if desired.
        :param precision: (float) A value to be used to represent the precision of the survey for plotting and statistics
                          purposes. Default is 1.
        :param fmt: (str) A matplotlib marker type to be used in cross-section plots. Default is 'kx'.
        :param ms: (str) A matplotlib marker size for cross-section plotting. Default is 4.
        :param zorder: (float) A zorder for matplotlib cross-section plotting. Default is 2.5.
        :param fillstyle: (str) A matplotlib marker fill style. Default is 'full'
        :param alpha: (float) A matplotlib float value representing transparency, with 1 being completely opaque and
                      0 being completely transparent. Default is 1.
        :param color: (str) A matplotlib color for cross-section plotting. Not currently implemented. Default is None
                            (plot using matplotlib standards).
        :param buffer_in: (str, ArcGIS linear units) A distance to buffer the dataset coverage polygon inwards to
                          account for decreased accuracy at the edge of the data and define the region where the script
                          should fit the values closely. The default is '-5 Meters'.
        :param point_erase_poly: (str) Path to feature class containing polygons where observation points should
                                 be removed from final clipped points used for interpolation.

        Last revised: 11/09/2023
        """
        super().__init__(data_short_name, data_long_name, precision, fmt, ms, zorder, fillstyle, alpha, color,
                         point_erase_poly=point_erase_poly)
        self.bathy_raster_path = bathy_raster_path
        self.buffer_in = buffer_in

    @ObservationPoints.analysis_step
    def clip_export_points(self):
        if not os.path.exists(self._working_dir):
            os.mkdir(self._working_dir)
        temp_gdb = fm.PathedGeoDataBase(rf'{self._working_dir}\clip_{self.name}_tmp.gdb')
        temp_gdb.set_env()
        bathy_ras = rf'{self.name}_ras'
        bathy_resample = rf'{self.name}_resample'
        bathy_clip = rf'{self._working_dir}\{self.mesh.name}_{self.name}_clip.shp'
        arcpy.Clip_management(self.bathy_raster_path, '#', bathy_ras, self.mesh.boundary,
                              clipping_geometry='ClippingGeometry')
        arcpy.Resample_management(bathy_ras, bathy_resample, '2 2', 'BILINEAR')
        if self.point_erase_poly is not None:
            arcpy.CopyFeatures_management(self.point_erase_poly, 'point_erase_poly')
            arcpy.RasterToPoint_conversion(bathy_resample, 'temp_clip')
            arcpy.Erase_analysis('temp_clip', self.point_erase_poly, bathy_clip)
        else:
            arcpy.RasterToPoint_conversion(bathy_resample, bathy_clip)
        if int(arcpy.GetCount_management(bathy_clip)[0]) == 0:
            return False
        for field in ['POINT_X', 'POINT_Y', 'POINT_Z']:
            arcpy.AddField_management(bathy_clip, field, 'DOUBLE')
        arcpy.CalculateField_management(bathy_clip, 'POINT_X', '!Shape.Centroid.X!', 'PYTHON')
        arcpy.CalculateField_management(bathy_clip, 'POINT_Y', '!Shape.Centroid.Y!', 'PYTHON')
        arcpy.CalculateField_management(bathy_clip, 'POINT_Z', '!grid_code!', 'PYTHON')
        df_in = pd.DataFrame(arcpy.da.TableToNumPyArray(bathy_clip,
                                                        [field.name for field in arcpy.Describe(bathy_clip).fields
                                                         if (field.name in ['POINT_X', 'POINT_Y', 'POINT_Z']) | (
                                                                 field.type in ['OID'])]))
        df_in.to_csv(
            rf'{self._working_dir}\{self.mesh.name}_{self.name}_clip.csv',
            index=False)
        raster = arcpy.sa.Raster(self.bathy_raster_path)
        coverage_raster = arcpy.sa.Con(arcpy.sa.IsNull(raster), raster, 1)
        coverage_raster = arcpy.sa.Int(coverage_raster)
        coverage_raster.save(rf'{temp_gdb.full_path}\coverage_raster')

        # TODO: Should the buffered portion be moved to the shoreline aniso/diff creation argument? Nice to have
        # unbuffered polygon.
        arcpy.RasterToPolygon_conversion(rf'{temp_gdb.full_path}\coverage_raster', 'coverage_polygon_raw')
        arcpy.Dissolve_management('coverage_polygon_raw', 'coverage_polygon_dissolve')
        arcpy.EliminatePolygonPart_management('coverage_polygon_dissolve', 'coverage_polygon_elim', "AREA",
                                              "1000 SquareMeters", 0, "CONTAINED_ONLY")
        coverage_polygon = rf'{self._working_dir}\{self.mesh.name}_{self.name}_a1.shp'
        arcpy.Buffer_analysis('coverage_polygon_elim', coverage_polygon, self.buffer_in, 'FULL')
        temp_gdb.reset_env()
        return True

    @property
    def name(self):
        path = pathlib.PureWindowsPath(self.bathy_raster_path).stem
        return path

    @property
    def fc_path(self):
        bathy_clip = rf'{self._working_dir}\{self.mesh.name}_{self.name}_clip.shp'
        if os.path.exists(bathy_clip):
            return bathy_clip

    @property
    def csv_path(self):
        clip_csv = rf'{self._working_dir}\{self.mesh.name}_{self.name}_clip.csv'
        if os.path.exists(clip_csv):
            return clip_csv

    @property
    def coverage_polygon_fc_path(self):
        coverage_polygon = rf'{self._working_dir}\{self.mesh.name}_{self.name}_a1.shp'
        if os.path.exists(coverage_polygon):
            return coverage_polygon


class ManualBathyPoints(BathyPoints):
    def __init__(self, name, fc_clip_input_path, csv_path, data_short_name, data_long_name, precision=1, fmt='kx', ms=4,
                 zorder=2.5, fillstyle='full', alpha=1, coverage_polygon_fc_path=None, color=None,
                 buffer_in='-5 Meters',
                 point_erase_poly=None):
        super().__init__(data_short_name, data_long_name, precision, fmt, ms, zorder, fillstyle, alpha, color,
                         point_erase_poly=point_erase_poly)
        self._name = name
        self._fc_clip_input_path = fc_clip_input_path
        self._csv_path = csv_path
        self._coverage_polygon_fc_path = coverage_polygon_fc_path
        self.buffer_in = buffer_in

    @ObservationPoints.analysis_step
    def clip_export_points(self):
        if not os.path.exists(self._working_dir):
            os.mkdir(self._working_dir)
        if not os.path.exists(rf'{self._working_dir}\{self.mesh.name}_{self.name}_clip.shp'):
            arcpy.Copy_management(self._fc_clip_input_path,
                                  rf'{self._working_dir}\{self.mesh.name}_{self.name}_clip.shp')
        if not os.path.exists(rf'{self._working_dir}\{self.mesh.name}_{self.name}_clip.csv'):
            shutil.copy(self._csv_path, rf'{self._working_dir}\{self.mesh.name}_{self.name}_clip.csv')
        if self._coverage_polygon_fc_path is not None:
            coverage_polygon = rf'{self._working_dir}\{self.mesh.name}_{self.name}_a1.shp'
            if not os.path.exists(coverage_polygon):
                arcpy.Buffer_analysis(self._coverage_polygon_fc_path, coverage_polygon, self.buffer_in, 'FULL')
        return True

    @property
    def name(self):
        return self._name

    @property
    def fc_path(self):
        path = rf'{self._working_dir}\{self.mesh.name}_{self.name}_clip.shp'
        if os.path.exists(path):
            return path

    @property
    def csv_path(self):
        path = rf'{self._working_dir}\{self.mesh.name}_{self.name}_clip.csv'
        if os.path.exists(path):
            return path

    @property
    def coverage_polygon_fc_path(self):
        path = rf'{self._working_dir}\{self.mesh.name}_{self.name}_a1.shp'
        if os.path.exists(path):
            return path


class ShorelinePoints(ObservationPoints):
    def __init__(self, data_short_name, data_long_name, precision=0.283, fmt='k.', ms=20, zorder=2.6, fillstyle='full',
                 alpha=1, color=None, point_erase_poly=None):
        """
        An object representing the shoreline points generated between the target polygon and the mesh boundary.

        :param data_short_name: (str) A shorthand name to be used in summary statistics.
        :param data_long_name: (str) A longhand name to be used in the main control script.
                               May be used in the future for creating legends in plots if desired.
        :param precision: (float) A value to be used to represent the precision of the survey for plotting and statistics
                          purposes. Default is 0.283.
        :param fmt: (str) A matplotlib marker type to be used in cross-section plots. Default is 'k.'.
        :param ms: (str) A matplotlib marker size for cross-section plotting. Default is 20.
        :param zorder: (float) A zorder for matplotlib cross-section plotting. Default is 2.6.
        :param fillstyle: (str) A matplotlib marker fill style. Default is 'full'
        :param alpha: (float) A matplotlib float value representing transparency, with 1 being completely opaque and
                      0 being completely transparent. Default is 1.
        :param color: (str) A matplotlib color for cross-section plotting. Not currently implemented.
        :param point_erase_poly: (str) Path to feature class containing polygons where observation points should
                                 be removed from final clipped points used for interpolation.

        Last revised: 11/09/2023
        """
        super().__init__(data_short_name, data_long_name, precision, fmt, ms, zorder, fillstyle, alpha, color,
                         point_erase_poly=point_erase_poly)
        self.mesh = None
        self.precision = precision

    @property
    def _working_dir(self):
        path = rf'{self.mesh.analysis.working_directory.full_path}\{self.mesh.name}'
        return path

    @ObservationPoints.analysis_step
    def clip_export_points(self):
        temp_gdb = fm.PathedGeoDataBase(
            rf'{self._working_dir}\clip_{self.name}_tmp.gdb')
        temp_gdb.set_env()
        arcpy.Clip_management(self.mesh.dem_clip_1m, '#', 'shoreline',
                              in_template_dataset=self.mesh.boundary,
                              clipping_geometry='ClippingGeometry')
        shoreline_clip = rf'{self._working_dir}\{self.mesh.name}_{self.name}.shp'
        arcpy.RasterToPoint_conversion('shoreline', shoreline_clip)
        if int(arcpy.GetCount_management(shoreline_clip)[0]) == 0:
            return False
        for field in ['POINT_X', 'POINT_Y', 'POINT_Z']:
            arcpy.AddField_management(shoreline_clip, field, 'DOUBLE')
        arcpy.CalculateField_management(shoreline_clip, 'POINT_X', '!Shape.Centroid.X!', 'PYTHON')
        arcpy.CalculateField_management(shoreline_clip, 'POINT_Y', '!Shape.Centroid.Y!', 'PYTHON')
        arcpy.CalculateField_management(shoreline_clip, 'POINT_Z', '!grid_code!', 'PYTHON')
        df_in = pd.DataFrame(arcpy.da.TableToNumPyArray(shoreline_clip,
                                                        [field.name for field in arcpy.Describe(shoreline_clip).fields
                                                         if (field.name in ['POINT_X', 'POINT_Y', 'POINT_Z']) | (
                                                                 field.type in ['OID'])]))
        df_in.to_csv(
            rf'{self._working_dir}\{self.mesh.name}_{self.name}.csv',
            index=False)
        temp_gdb.reset_env()
        return True

    @property
    def csv_path(self):
        shoreline_csv = rf'{self._working_dir}\{self.mesh.name}_{self.name}.csv '
        if os.path.exists(shoreline_csv):
            return shoreline_csv

    @property
    def fc_path(self):
        shoreline_clip = rf'{self._working_dir}\{self.mesh.name}_{self.name}.shp'
        if os.path.exists(shoreline_clip):
            return shoreline_clip

    @property
    def name(self):
        return self.data_short_name.replace(' ', '_')


class LiDARPoints(ShorelinePoints):
    def __init__(self, lidar_pts_fc_path, data_short_name, data_long_name, precision=0.283, fmt='kx', ms=4, zorder=2.5,
                 fillstyle='full', alpha=1, z_ft_name=None, z_m_name=None, coverage_fc=None, color=None,
                 lidar_point_erase_poly=None, buffer_in='-5 Meters', point_erase_poly=None):
        """
        An object representing terrestrial LiDAR points.

        :param lidar_pts_fc_path: (str) Path to feature class containing LiDAR points.
        :param data_short_name: (str) A shorthand name for the bathymetry survey to be used in summary statistics.
        :param data_long_name: (str) A longhand name for the bathymetry survey to be used in the main control script.
                               May be used in the future for creating legends in plots if desired.
        :param precision: (float) A value to be used to represent the precision of the survey for plotting and statistics
                          purposes. Default is 0.283.
        :param fmt: (str) A matplotlib marker type to be used in cross-section plots. Default is 'kx'.
        :param ms: (str) A matplotlib marker size for cross-section plotting. Default is 4.
        :param zorder: (float) A zorder for matplotlib cross-section plotting. Default is 2.5.
        :param fillstyle: (str) A matplotlib marker fill style. Default is 'full'
        :param alpha: (float) A matplotlib float value representing transparency, with 1 being completely opaque and
                      0 being completely transparent. Default is 1.
        :param fmt: (str) A matplotlib marker type to be used in cross-section plots. Default is 'kx'.
        :param ms: (str) A matplotlib marker size for cross-section plotting. Default is 4.
        :param zorder: (float) A zorder for matplotlib cross-section plotting. Default is 2.5.
        :param fillstyle: (str) A matplotlib marker fill style. Default is 'full'
        :param alpha: (float) A matplotlib float value representing transparency, with 1 being completely opaque and
                      0 being completely transparent
        :param z_ft_name: (str) A field containing elevation values in NAVD88 ft. Should be None if z_m_name is used.
        :param z_m_name: (str) A field containing elevation values in NAVD88 m. Should be None if z_ft_name is used.
        :param coverage_fc: (str) A path to a feature class coverage polygon representing the
                            unbuffered extents of the data. Can be set to None if setting coefficients to 1 is undesirable.
        :param color: (str) A matplotlib color for cross-section plotting. Not currently implemented. Default is None
                            (plot using matplotlib standards).
        :param lidar_point_erase_poly: (str) A path to a feature class polygon representing areas where LiDAR points
                                       should not be included in the interpolation. Often used to remove points
                                       that overlap areas where shoreline points are already generated. Default is None.
        :param buffer_in: (str, ArcGIS linear unit) A buffer distance to be used for adjusting the coverage polygon
                                                    extents. Default is '-5 Meters'.
        :param point_erase_poly: (str) Path to feature class containing polygons where observation points should
                                 be removed from final clipped points used for interpolation.

        Last revised: 11/09/2023
        """
        super().__init__(data_short_name, data_long_name, precision, fmt, ms, zorder, fillstyle,
                         alpha, point_erase_poly=point_erase_poly)
        self.z_ft_name = z_ft_name
        self.z_m_name = z_m_name
        self.lidar_pts_fc_path = lidar_pts_fc_path
        self.coverage_fc = coverage_fc
        self.point_erase_poly = lidar_point_erase_poly
        self.buffer_in = buffer_in

    @ObservationPoints.analysis_step
    def clip_export_points(self):
        if not os.path.exists(self._working_dir):
            os.mkdir(self._working_dir)
        temp_gdb = fm.PathedGeoDataBase(rf'{self._working_dir}\clip_{self.name}_tmp.gdb')
        temp_gdb.set_env()
        lidar_clip = rf'{self._working_dir}\{self.mesh.name}_{self.name}_clip.shp'
        if self.point_erase_poly is not None:
            arcpy.CopyFeatures_management(self.point_erase_poly, 'lidar_point_erase_poly')
            arcpy.Clip_analysis(self.lidar_pts_fc_path, self.mesh.boundary, 'temp_clip')
            arcpy.Erase_analysis('temp_clip', self.point_erase_poly, lidar_clip)
        else:
            arcpy.Clip_analysis(self.lidar_pts_fc_path, self.mesh.boundary, lidar_clip)
        if int(arcpy.GetCount_management(lidar_clip)[0]) == 0:
            return False
        for field in ['POINT_X', 'POINT_Y', 'POINT_Z']:
            if field not in [fc_field.name for fc_field in arcpy.Describe(lidar_clip).fields]:
                arcpy.AddField_management(lidar_clip, field, 'DOUBLE')
        if (self.z_ft_name is not None) & (
                'POINT_Z' not in [field.name for field in arcpy.Describe(self.lidar_pts_fc_path).fields]):
            arcpy.CalculateField_management(lidar_clip, 'POINT_Z', f'!{self.z_ft_name}!*0.3048', 'PYTHON')
        if (self.z_m_name is not None) & (
                'POINT_Z' not in [field.name for field in arcpy.Describe(self.lidar_pts_fc_path).fields]):
            arcpy.CalculateField_management(lidar_clip, 'POINT_Z', f'!{self.z_m_name}!', 'PYTHON')
        if 'POINT_X' not in [field.name for field in arcpy.Describe(self.lidar_pts_fc_path).fields]:
            arcpy.CalculateField_management(lidar_clip, 'POINT_X', '!Shape.Centroid.X!', 'PYTHON')
        if 'POINT_Y' not in [field.name for field in arcpy.Describe(self.lidar_pts_fc_path).fields]:
            arcpy.CalculateField_management(lidar_clip, 'POINT_Y', '!Shape.Centroid.Y!', 'PYTHON')
        df_in = pd.DataFrame(arcpy.da.TableToNumPyArray(lidar_clip,
                                                        [field.name for field in arcpy.Describe(lidar_clip).fields
                                                         if (field.name in ['POINT_X', 'POINT_Y', 'POINT_Z']) | (
                                                                 field.type in ['OID'])]))
        df_in.to_csv(rf'{self._working_dir}\{self.mesh.name}_{self.name}_clip.csv', index=False)
        if self.coverage_fc is not None:
            coverage_polygon = rf'{self._working_dir}\{self.mesh.name}_{self.name}_a1.shp'
            if (self.buffer_in is None) or (self.buffer_in == '0 Meters'):
                arcpy.CopyFeatures_management(self.coverage_fc, coverage_polygon)
            else:
                arcpy.Buffer_analysis(self.coverage_fc, coverage_polygon, self.buffer_in, 'FULL')
        temp_gdb.reset_env()
        return True

    @property
    def name(self):
        path = pathlib.PureWindowsPath(self.lidar_pts_fc_path).stem
        return path

    @property
    def fc_path(self):
        lidar_clip = rf'{self._working_dir}\{self.mesh.name}_{self.name}_clip.shp'
        if os.path.exists(lidar_clip):
            return lidar_clip

    @property
    def csv_path(self):
        clip_csv = rf'{self._working_dir}\{self.mesh.name}_{self.name}_clip.csv'
        if os.path.exists(clip_csv):
            return clip_csv

    @property
    def coverage_polygon_fc_path(self):
        if self.coverage_fc is not None:
            coverage_polygon = rf'{self._working_dir}\{self.mesh.name}_{self.name}_a1.shp'
            if os.path.exists(coverage_polygon):
                return coverage_polygon
            else:
                return None
        else:
            return None


# TODO Remove precision keyword and update references.
# Same as LiDARPoints, but recognized by script not to be included in statistics.
class AugmentPoints(LiDARPoints):
    def __init__(self, lidar_pts_fc_path, data_short_name, data_long_name, precision=None, fmt='kx', ms=4, zorder=2.5,
                 fillstyle='full', alpha=1, z_ft_name=None, z_m_name=None, coverage_fc=None, color=None,
                 buffer_in='-5 Meters', point_erase_poly=None):
        """
        An object representing estimated LiDAR or terrestrial points. These points are excluded from summary statistics.

        :param lidar_pts_fc_path: (str) Path to feature class containing augmented points.
        :param data_short_name: (str) A shorthand name for the points to be used in internal data storage for plotting.
        :param data_long_name: (str) A longhand name for the points to be used in the main control script.
                               May be used in the future for creating legends in plots if desired.
        :param precision: (float) A value to be used to represent the precision of the survey for plotting and statistics
                          purposes. Default is None.
        :param fmt: (str) A matplotlib marker type to be used in cross-section plots. Default is 'kx'.
        :param ms: (str) A matplotlib marker size for cross-section plotting. Default is 4.
        :param zorder: (float) A zorder for matplotlib cross-section plotting. Default is 2.5.
        :param fillstyle: (str) A matplotlib marker fill style. Default is 'full'
        :param alpha: (float) A matplotlib float value representing transparency, with 1 being completely opaque and
                      0 being completely transparent. Default is 1.
        :param fmt: (str) A matplotlib marker type to be used in cross-section plots. Default is 'kx'.
        :param ms: (str) A matplotlib marker size for cross-section plotting. Default is 4.
        :param zorder: (float) A zorder for matplotlib cross-section plotting. Default is 2.5.
        :param fillstyle: (str) A matplotlib marker fill style. Default is 'full'
        :param alpha: (float) A matplotlib float value representing transparency, with 1 being completely opaque and
                      0 being completely transparent
        :param z_ft_name: (str) A field containing elevation values in NAVD88 ft. Should be None if z_m_name is used.
        :param z_m_name: (str) A field containing elevation values in NAVD88 m. Should be None if z_ft_name is used.
        :param coverage_fc: (str) A path to a feature class coverage polygon representing the
                            unbuffered extents of the data. Can be set to None if setting coefficients to 1 is undesirable.
        :param color: (str) A matplotlib color for cross-section plotting. Not currently implemented. Default is None
                            (plot using matplotlib standards).
        :param buffer_in: (str, ArcGIS linear unit) A buffer distance to be used for adjusting the coverage polygon
                                                    extents. Default is '-5 Meters'.
        :param point_erase_poly: (str) Path to feature class containing polygons where observation points should
                                 be removed from final clipped points used for interpolation.

        Last revised: 11/09/2023
        """
        super().__init__(lidar_pts_fc_path, data_short_name, data_long_name, precision, fmt=fmt, ms=ms, zorder=zorder,
                         fillstyle=fillstyle, alpha=alpha, z_ft_name=z_ft_name, z_m_name=z_m_name,
                         coverage_fc=coverage_fc, color=color, buffer_in=buffer_in, point_erase_poly=point_erase_poly)


class TargetPoints:
    def __init__(self):
        self.mesh = None

    @property
    def _working_dir(self):
        path = rf'{self.mesh.analysis.working_directory.full_path}\{self.mesh.name}'
        return path

    def create_target_points(self):
        temp_gdb = fm.PathedGeoDataBase(
            rf'{self._working_dir}\clip_target_tmp.gdb')
        temp_gdb.set_env()
        old_snap = arcpy.env.snapRaster
        arcpy.env.snapRaster = self.mesh.analysis.dem_2m_path
        arcpy.Clip_management(self.mesh.analysis.source_dem_path, '#', 'target_area',
                              in_template_dataset=self.mesh.boundary, clipping_geometry='ClippingGeometry')
        arcpy.Resample_management('target_area', 'target_area_2', '2 2', 'BILINEAR')
        target_clip = rf'{self._working_dir}\{self.mesh.name}_target_pts.shp'
        arcpy.RasterToPoint_conversion('target_area_2', target_clip)
        for field in ['POINT_X', 'POINT_Y']:
            arcpy.AddField_management(target_clip, field, 'DOUBLE')
        arcpy.CalculateField_management(target_clip, 'POINT_X', '!Shape.Centroid.X!', 'PYTHON')
        arcpy.CalculateField_management(target_clip, 'POINT_Y', '!Shape.Centroid.Y!', 'PYTHON')
        df_in = pd.DataFrame(arcpy.da.TableToNumPyArray(target_clip,
                                                        [field.name for field in arcpy.Describe(target_clip).fields
                                                         if (field.name in ['POINT_X', 'POINT_Y']) | (
                                                                 field.type in ['OID'])]))
        df_in.to_csv(
            rf'{self._working_dir}\{self.mesh.name}_target_pts.csv',
            index=False)
        arcpy.env.snapRaster = old_snap
        temp_gdb.reset_env()

    @property
    def target_pts_csv_path(self):
        target_csv = rf'{self._working_dir}\{self.mesh.name}_target_pts.csv'
        if os.path.exists(target_csv):
            return target_csv

    @property
    def target_pts_fc_path(self):
        target_clip = rf'{self._working_dir}\{self.mesh.name}_target_pts.shp'
        if os.path.exists(target_clip):
            return target_clip


class AbstractAnalysis(ABC):
    def __init__(self, dem_2m_path, source_dem_path=None, breaklines_fc_path=None, levee_path=None):
        self.reach = None
        self._number = None
        self._new = True
        self._registry = []
        self.source_dem_path = dem_2m_path if source_dem_path is None else source_dem_path
        self.working_directory = None
        self.gpr = GeoProcessingRecorder(self)
        self.breaklines_fc_path = breaklines_fc_path
        self.levee_path = levee_path
        self.dem_2m_path = dem_2m_path

    # Special decorator to remember what analysis steps were run previously
    # and skip analysis steps if a result is already present.
    def analysis_step(func):
        def inner(self, *args, **kwargs):
            identifier = rf'_{self.__class__.__name__}_{func.__name__}'
            argstring = ', '.join([str(arg) for arg in args if arg is not None])
            kwstring = ', '.join([f'{kw}={val}' for kw, val in kwargs.items()])
            command_str = rf'{self.__class__.__name__}.{func.__name__}({argstring}, {kwstring})'.replace(', )', ')')
            # Perform function substitution here. This function is written so analysis_step can be defined in
            # attached objects and add records to the analysis history.
            res = inner_record(self, self, func, identifier, command_str, '', *args, **kwargs)
            return res

        return inner

    @property
    def history_path(self):
        return rf'{self.working_directory.full_path}\history.txt'

    @property
    def number(self):
        return self._number

    @number.setter
    def number(self, number):
        if self._number is None:
            self._number = number

    @property
    def name(self):
        classname = self.__class__.__name__.replace('Analysis', '')
        return f'{self._number:02d}_{classname}'

    @property
    def starting_bathy_dem_path(self):
        return rf'{self.reach.name}\{self.name}_start.tif'

    @property
    def ending_bathy_dem_path(self):
        return rf'{self.reach.name}\{self.name}_end.tif'

    def set_up_folder(self):
        check = glob.glob(rf'{self.reach.project.project_folder_path}\{self.reach.name}\{self._number:1d}_*')
        if os.path.exists(rf'{self.reach.project.project_folder_path}\{self.reach.name}\{self.name}'):
            self.working_directory = fm.ExistingWorkingDirectory(
                rf'{self.reach.project.project_folder_path}\{self.reach.name}\{self.name}')
            return
        if len(check) > 0:
            if os.path.exists(check[0]):
                raise ValueError("""Another analysis of a different name exists.""")
        os.mkdir(rf'{self.reach.project.project_folder_path}\{self.reach.name}\{self.name}')
        self.working_directory = fm.ExistingWorkingDirectory(
            rf'{self.reach.project.project_folder_path}\{self.reach.name}\{self.name}')

    @analysis_step
    def update_bathy_dem(self):
        pass

    @analysis_step
    def finalize_dem(self):
        pass

    def save_history(self):
        with open(rf'{self.working_directory.full_path}\history.txt', 'w') as f:
            f.write('\n'.join(self._registry))


class FDAPDEAnalysis(AbstractAnalysis):
    def __init__(self, dem_2m_path, source_dem_path=None, breaklines_fc_path=None, levee_path=None):
        super().__init__(dem_2m_path, source_dem_path, breaklines_fc_path, levee_path)
        self.block_polygon_fc_path = None
        self.mesh = None

    def _set_up_input(self, fc_path):
        name = pathlib.PureWindowsPath(fc_path).stem
        new_path = rf'{self.working_directory.full_path}\{name}.shp'
        if not os.path.exists(new_path):
            arcpy.CopyFeatures_management(fc_path, new_path)
        return new_path

    def __getitem__(self, name):
        return self.mesh

    def add_mesh(self, mesh_name, domain_polygon_fc_path, block_polygon_fc_path, open_boundary_fc_path, res=3.0,
                 res_hint_poly_fc_path=None, h0=2.8, seed=1000, res_sdist_list=None, in_angle_raster=None):
        self.mesh = Mesh(mesh_name, self, domain_polygon_fc_path, block_polygon_fc_path, open_boundary_fc_path, res,
                         res_hint_poly_fc_path, h0, seed, res_sdist_list, in_angle_raster)


class Case:
    def __init__(self, name, aniso_constant, diff_constant, long_name=None, xs_plot=True, lw=1, zorder=None,
                 final=False):
        self.aniso_constant = aniso_constant
        self.diff_constant = diff_constant
        self.mesh = None
        self.name = name
        self.long_name = long_name
        self.xs_plot = xs_plot
        self.result = None
        self.lw = lw
        self.zorder = zorder
        self.final = final

    @property
    def _working_dir(self):
        path = rf'{self.mesh.analysis.working_directory.full_path}\{self.mesh.name}'
        return path

    def analysis_step(func):
        def inner(self, *args, **kwargs):
            identifier = rf'_{self.mesh.analysis.__class__.__name__}_{self.mesh.name}_{self.name}_{func.__name__}'
            argstring = ', '.join([str(arg) for arg in args if arg is not None])
            kwstring = ', '.join([f'{kw}={val}' for kw, val in kwargs.items()])
            command_str = rf'{self.mesh.analysis.__class__.__name__}[{self.mesh.name}][{self.name}].{func.__name__}({argstring}, {kwstring})'.replace(
                ', )', ')')
            # Perform function substitution here. This function is written so analysis_step can be defined in
            # attached objects and add records to the analysis history.
            res = inner_record(self, self.mesh.analysis, func, identifier, command_str,
                               rf'\{self.mesh.name}\{self.name}', *args, **kwargs)
            return res

        return inner

    def _create_constant_raster(self, var_name):
        value = getattr(self, f'{var_name}_constant')
        out_raster_path = rf'{self._working_dir}\{self.name}_{var_name}_{value:0.0f}.tif'
        # This if statement bypasses raster if it was previously generated AND avoids generating a duplicate raster
        # if another Case uses the same coefficient.
        if not os.path.exists(out_raster_path):
            angle = arcpy.sa.Raster(self.mesh.angle_raster)
            out_raster = value + 0 * angle
            out_raster.save(out_raster_path)
        return out_raster_path

    # custom_yaml_template is for specialized cases to include certain parameters by default. This script may be
    # modified in the future to accommodate multiple observation points.
    @analysis_step
    def set_up_yaml_file(self, custom_yaml_template=r"resources\r_input_template.yaml"):
        yaml_path = rf'{self.mesh.analysis.working_directory.full_path}\{self.mesh.name}_{self.name}.yaml'
        if not os.path.exists(yaml_path):
            shutil.copy(custom_yaml_template, yaml_path)
            int_script_directory = self.mesh.analysis.reach.project.interpolation_script_directory.replace('\\', '\\\\')
            mesh_parent_directory = rf'{self.mesh.analysis.working_directory.full_path}\{self.mesh.name}'.replace('\\',
                                                                                                                  '\\\\')
            mesh = f'{self.mesh.name}_out'
            target_pts = pathlib.PureWindowsPath(self.mesh.target_points.target_pts_csv_path).stem
            angle_raster = pathlib.PureWindowsPath(self.mesh.angle_raster).stem
            aniso_raster = pathlib.PureWindowsPath(self.aniso_raster).stem
            diff_raster = pathlib.PureWindowsPath(self.diff_raster).stem
            substitute_obs_pt_str = '\n'.join([f'- file_name: {pathlib.PureWindowsPath(obs_pt.csv_path).stem}.csv'
                                               for obs_pt in self.mesh.observation_points])
            substitute_text_tags(yaml_path,
                                 {'<INTERPOLATION_SCRIPT_DIRECTORY>': int_script_directory,
                                  '<MESH_PARENT_DIRECTORY>': mesh_parent_directory,
                                  '<MESH>': mesh,
                                  '<MESH_NAME>': self.mesh.name,
                                  '<TARGET_PTS>': target_pts,
                                  '<OBS_PTS>': substitute_obs_pt_str,
                                  '<CASE_NAME>': self.name,
                                  '<ANGLE_RASTER>': angle_raster,
                                  '<ANISO_RASTER>': aniso_raster,
                                  '<DIFF_COEFF_RASTER>': diff_raster})

    @analysis_step
    def run_fda_pde(self):
        project = self.mesh.analysis.reach.project
        command = rf'{project.path_to_rscript} {project.path_to_run_r} --d ' \
                  rf'{project.interpolation_script_directory} -f {self.mesh.name}_{self.name}.yaml'
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                cwd=self.mesh.analysis.working_directory.full_path)
        output, messages = proc.communicate()
        with open(rf'{self.mesh.analysis.working_directory.full_path}\{self.mesh.name}_{self.name}_fdaPDE_output.txt',
                  'w') as f:
            outstring = '=' * 32 + ' OUTPUT ' + '=' * 32 + '\r\n' + f'{output.decode()}' + '\r\n' + '=' * 31 \
                        + ' MESSAGES ' + '=' * 31 + '\r\n' + f'{messages.decode()}'
            f.write(outstring)
        print(output.decode())

    @property
    def result_raster(self):
        out_path = rf'{self.mesh.analysis.working_directory.full_path}\{self.mesh.name}\{self.mesh.name}_sp_var_{self.name}.tif'
        if os.path.exists(out_path):
            return out_path

    @property
    def aniso_raster(self):
        path = rf'{self._working_dir}\{self.name}_aniso_{self.aniso_constant:0.0f}.tif'
        if os.path.exists(path):
            return path

    @property
    def diff_raster(self):
        path = rf'{self._working_dir}\{self.name}_diff_{self.diff_constant:0.0f}.tif'
        if os.path.exists(path):
            return path

    def read_check_result(self, rsme_max=0.5):
        self.result = FDAPDEResult(self)
        print(self.result.rsme)
        if self.result.rsme >= rsme_max:
            print(f'{self.name} rsme exceeds rsme max of {rsme_max}. Removing from plotting')
            self.xs_plot = False


class ConstantCase(Case):
    @Case.analysis_step
    def create_constant_aniso_ratio(self):
        self._create_constant_raster('aniso')

    @Case.analysis_step
    def create_constant_diff_coeff(self):
        self._create_constant_raster('diff')


class VariableCase(Case):
    def __init__(self, name, aniso_constant, diff_constant, long_name=None, xs_plot=True, lw=1, zorder=None,
                 final=False):
        super().__init__(name, aniso_constant, diff_constant, long_name, xs_plot, lw, zorder, final)

    @Case.analysis_step
    def create_variable_aniso_ratio(self, aniso_polygons, priority_field=None, from_multibeam=False, multibeam_value=1,
                                    from_shoreline=False, shoreline_value=1, sdist_list=None):
        self._create_variable_raster('aniso', aniso_polygons, priority_field, from_multibeam,
                                     multibeam_value, from_shoreline, shoreline_value, sdist_list)

    @Case.analysis_step
    def create_variable_diff_coeff(self, diff_polygons, priority_field=None, from_multibeam=False, multibeam_value=1,
                                   from_shoreline=False, shoreline_value=1, sdist_list=None):
        self._create_variable_raster('diff', diff_polygons, priority_field, from_multibeam, multibeam_value,
                                     from_shoreline, shoreline_value, sdist_list)

    def _create_variable_raster(self, var_name, polygons, priority_field=None, from_multibeam=False, multibeam_value=1,
                                from_shoreline=False, shoreline_value=1, sdist_list=None, polygon_order='before'):
        # If a constant raster already exists for some reason, this step will be short-circuited.
        # sdist_list is a list of lists/tuples each containing a distance value and a variable value.
        constant_raster = self._create_constant_raster(var_name)
        out_raster_path = rf'{self._working_dir}\{self.mesh.name}_{var_name}_{self.name}.tif'

        if not os.path.exists(out_raster_path):
            temp_gdb = fm.PathedGeoDataBase(
                rf'{self._working_dir}\{self.name}_{var_name}.gdb')
            temp_gdb.set_env()
            in_polygons = polygons
            tmp_raster = None
            add_polygons = []
            if from_multibeam:
                # Get a list of coverages that contain the attribute coverage_polygon_fc_path and is not empty.
                obs_coverages_1 = [obs_pt for obs_pt in self.mesh.observation_points
                                   if (hasattr(obs_pt, 'coverage_polygon_fc_path'))]
                obs_coverages = [obs_pt.coverage_polygon_fc_path for obs_pt in obs_coverages_1
                                 if obs_pt.coverage_polygon_fc_path is not None]
                if len(obs_coverages) == 1:
                    arcpy.CopyFeatures_management(obs_coverages[0], 'bathy_polygon_tmp')
                elif len(obs_coverages) > 1:
                    arcpy.Merge_management(obs_coverages, 'coverages_merged')
                    arcpy.Dissolve_management('coverages_merged', 'bathy_polygon_tmp')
                else:
                    raise RuntimeError('from_multibeam specified but no coverage polygon exists.')

                # Need to clip, multibeam coverage may extend past domain.
                arcpy.RasterDomain_3d(constant_raster, 'constant_raster_domain', 'POLYGON')
                arcpy.Clip_analysis('bathy_polygon_tmp', 'constant_raster_domain', 'bathy_polygon')
                if int(arcpy.GetCount_management('bathy_polygon')[0]) > 0:
                    arcpy.AddField_management('bathy_polygon', var_name, 'DOUBLE')
                    arcpy.CalculateField_management('bathy_polygon', var_name, f'{multibeam_value}', 'PYTHON')
                    add_polygons += [rf'{temp_gdb.full_path}\bathy_polygon']
            if sdist_list is not None:
                for i, sdist_pair in enumerate(sdist_list):
                    sdist, value = sdist_pair
                    sdist *= -1
                    sdist_raster = arcpy.sa.Raster(rf'{self._working_dir}\sdist_{self.mesh.name}.tif')
                    if i == 0:
                        tmp_raster = arcpy.sa.SetNull((sdist_raster < sdist) | (sdist_raster >= 0), value)
                    else:
                        tmp_raster = arcpy.sa.Con((sdist_raster >= sdist) & (sdist_raster <= 0), value, tmp_raster)
                sdist_var_raster = arcpy.sa.Int(tmp_raster)
                sdist_var_raster.save(rf'{temp_gdb.full_path}\sdist_var_raster')
                arcpy.RasterToPolygon_conversion(rf'{temp_gdb.full_path}\sdist_var_raster', 'sdist_polygon',
                                                 raster_field='Value')
                arcpy.AddField_management('sdist_polygon', var_name, 'DOUBLE')
                arcpy.CalculateField_management('sdist_polygon', var_name, "!gridcode!", 'PYTHON')
                add_polygons += [rf'{temp_gdb.full_path}\sdist_polygon']
            if from_shoreline:
                dem = self.mesh.dem_clip_1m
                arcpy.Clip_management(dem, '#', f'{self.name}_clip',
                                      in_template_dataset=self.mesh.domain_erase_polygon_fc_path,
                                      nodata_value=-3.402823e38, clipping_geometry='ClippingGeometry')

                # raster = arcpy.sa.Raster(f'{self.name}_clip')
                raster = arcpy.sa.Raster(dem)
                coverage_raster = arcpy.sa.Con(arcpy.sa.IsNull(raster), raster, 1)
                coverage_raster = arcpy.sa.Int(coverage_raster)
                coverage_raster.save(rf'{temp_gdb.full_path}\coverage_raster')
                arcpy.RasterToPolygon_conversion(rf'{temp_gdb.full_path}\coverage_raster', 'coverage_polygon_raw')
                arcpy.Dissolve_management('coverage_polygon_raw', 'shoreline_polygon')
                arcpy.AddField_management('shoreline_polygon', var_name, 'DOUBLE')
                arcpy.CalculateField_management('shoreline_polygon', var_name, f'{shoreline_value}', 'PYTHON')
                add_polygons += [rf'{temp_gdb.full_path}\shoreline_polygon']
            if polygon_order == 'after':
                polygons = add_polygons + in_polygons
            elif polygon_order == 'before':
                polygons = in_polygons + add_polygons
            else:
                raise ValueError("polygon_order must be 'before' or 'after'.")
            mosaic_list = [constant_raster]
            old_snap = arcpy.env.snapRaster
            arcpy.env.snapRaster = self.mesh.angle_raster
            cellsize = arcpy.Describe(self.mesh.angle_raster).meanCellHeight
            for polygon in polygons:
                polygon_name = arcpy.Describe(polygon).baseName
                if priority_field is None:
                    arcpy.PolygonToRaster_conversion(polygon, var_name,
                                                     rf'{temp_gdb.full_path}\{polygon_name}_{var_name}_raster',
                                                     'CELL_CENTER', cellsize=cellsize)
                else:
                    arcpy.PolygonToRaster_conversion(polygon, var_name,
                                                     rf'{temp_gdb.full_path}\{polygon_name}_{var_name}_raster',
                                                     'CELL_CENTER', priority_field=priority_field, cellsize=cellsize)
                mosaic_list.append(rf'{temp_gdb.full_path}\{polygon_name}_{var_name}_raster')
            arcpy.MosaicToNewRaster_management(';'.join(mosaic_list), self._working_dir,
                                               f'{self.mesh.name}_{var_name}_{self.name}.tif',
                                               pixel_type='32_BIT_FLOAT',
                                               cellsize=cellsize, number_of_bands=1, mosaic_method='LAST')
            arcpy.env.snapRaster = old_snap
            temp_gdb.reset_env()
        return out_raster_path

    @property
    def aniso_raster(self):
        path = rf'{self._working_dir}\{self.mesh.name}_aniso_{self.name}.tif'
        if os.path.exists(path):
            return path

    @property
    def diff_raster(self):
        path = rf'{self._working_dir}\{self.mesh.name}_diff_{self.name}.tif'
        if os.path.exists(path):
            return path

    # For the time being, pass aniso_polygons and diff_polygons to the function since they are not part of the
    # case object.
    def export(self, export_folder_name, export_type='all', aniso_polygons=None, diff_polygons=None,
               custom_yaml_template_file=None, xs_fc=None, xs_qaqc_folder_name=None, fit_plot_qaqc_folder_name=None,
               island_removed_target_polygon=None):
        """Export case and all important inputs (except 2m, 10m, and source DEM's)."""
        mesh = self.mesh
        analysis = mesh.analysis
        reach = analysis.reach
        project = reach.project
        # Set up new folder. If it already exists, raise an error to prevent co-mingling of files.
        new_folder = os.path.join(project.project_folder_path, export_folder_name)
        if os.path.exists(new_folder):
            raise RuntimeError(f'{new_folder} already exists.')
        else:
            os.makedirs(new_folder)

        if export_type in ['all', 'inputs', 'inputs_results']:
            self._export_inputs(export_folder_name, aniso_polygons, diff_polygons, custom_yaml_template_file,
                                island_removed_target_polygon)

        if export_type in ['all', 'results', 'inputs_results', 'results_qaqc']:
            self._export_results(export_folder_name)

        if export_type in ['all', 'qaqc', 'results_qaqc']:
            self._export_qaqc(export_folder_name, xs_fc, xs_qaqc_folder_name, fit_plot_qaqc_folder_name)

    def _export_inputs(self, export_folder_name, aniso_polygons=None, diff_polygons=None, custom_yaml_template_file=None,
                       island_removed_target_polygon=None):
        mesh = self.mesh
        analysis = mesh.analysis
        reach = analysis.reach
        project = reach.project
        gis_file_list = []

        # Export analysis level files.
        gis_file_list += [analysis.breaklines_fc_path, mesh.block_polygon_input_fc_path]

        # Export mesh input files.
        gis_file_list += [mesh.domain_polygon_input_fc_path, mesh.open_boundary_input_fc_path]

        # Export custom angle raster, if present
        if mesh.in_angle_raster is not None:
            if island_removed_target_polygon is None:
                raise ValueError(f'Please provide island removed target polygon path for {mesh.in_angle_raster}. Use the island_removed_target_polygon keyword to specify this.')
            gis_file_list += [mesh.in_angle_raster, island_removed_target_polygon]

        # Export observation points.
        # TODO: In the future, read from the fdaPDE yaml file to ensure changes in observation point variables
        #  do not result in the wrong points being exported. This part is excluded for now to save on budget.
        for obs_pt in mesh.observation_points:
            if not (type(obs_pt) == ShorelinePoints) and not (type(obs_pt) == LiDARPoints):
                if hasattr(obs_pt, 'bathy_pts_fc_path') and (obs_pt.bathy_pts_fc_path is not None):
                    gis_file_list += [obs_pt.bathy_pts_fc_path] # Bathymetry points
                if hasattr(obs_pt, 'bathy_raster_path') and (obs_pt.bathy_raster_path is not None):
                    gis_file_list += [obs_pt.bathy_raster_path] # Bathymetry raster
                if hasattr(obs_pt, 'point_erase_poly') and (obs_pt.point_erase_poly is not None):
                    gis_file_list += [obs_pt.point_erase_poly] # Point removal polygon
                if hasattr(obs_pt, 'coverage_fc') and (obs_pt.coverage_fc is not None):
                    gis_file_list += [obs_pt.coverage_fc] # User-defined coverage polygon
                if type(obs_pt) == ManualBathyPoints:
                    gis_file_list += [obs_pt.csv_path]  # User-defined raw csv file

        # Export case level files.
        gis_file_list += [aniso_polygons, diff_polygons] # Anisotropy revision polygons (and diffusion if included)

        # Export custom yaml file, if present
        if custom_yaml_template_file is not None:
            self._copy_file(rf'{project.project_folder_path}\resources\{custom_yaml_template_file}', export_folder_name)

        # Use a set to remove duplicate file paths (often used in multipurpose files such as coverages that can also be used
        # to remove points, or aniso/diff files).
        for file in set(gis_file_list):
            self._copy_gis(file, export_folder_name)

    def _export_results(self, export_folder_name):
        mesh = self.mesh
        analysis = mesh.analysis
        reach = analysis.reach
        project = reach.project
        gis_file_list = []
        file_list = []

        # Export check files that are relevant to current case and mesh.
        txt_glob = glob.glob(rf'{analysis.working_directory.full_path}\*.txt', recursive=False)
        check_files = [txt for txt in txt_glob if f'_FDAPDEAnalysis_{mesh.name}' in txt]
        for check_file in check_files:
            fns = ['create_variable_aniso_ratio', 'create_variable_diff_coeff', 'set_up_yaml_file', 'run_fda_pde']
            fn_check = [fn in check_file for fn in fns]
            if any(fn_check) and (self.name in check_file):
                file_list += [check_file]
            elif not any(fn_check):
                file_list += [check_file]

        # Export masked raster.
        file_list += [rf'{analysis.working_directory.full_path}\rastermasked.tif']

        # Export fdaPDE yaml file.
        file_list += [rf'{analysis.working_directory.full_path}\{mesh.name}_{self.name}.yaml']

        # Export fdaPDE result file.
        file_list += [rf'{analysis.working_directory.full_path}\{mesh.name}_{self.name}_fdaPDE_output.txt']

        # Export all mesh preparation steps, case preparation steps, and fdaPDE results. Due to the number of files,
        # this is best done by including what is relevant and then subtracting out everything else.

        # Start with only the non-geodatabase files within the mesh folder.
        mesh_folder_files = [file for file in glob.glob(rf'{mesh.working_dir}\*', recursive=False) if os.path.isfile(file)]
        include_set = set()

        # If the case name matches, it belongs in the list regardless. This matches all fdaPDE outputs and coefficient
        # generation files that are not .gdb.
        include_set = include_set.union({file for file in mesh_folder_files if self.name in file})

        # If an observation point name is present, it belongs in the list.
        # TODO: In the future, read from the fdaPDE yaml file to ensure changes in observation point variables
        #  do not result in the wrong points being exported. This part is excluded for now to save on budget.
        for obs_pt in mesh.observation_points:
            include_set = include_set.union({file for file in mesh_folder_files if obs_pt.name in file})

        # Include target points
        include_set = include_set.union({file for file in mesh_folder_files if f'{mesh.name}_target_pts' in file})

        # Include signed distance
        include_set = include_set.union({file for file in mesh_folder_files if f'sdist_{mesh.name}' in file})

        # Include mesh points and polygon
        include_set = include_set.union({file for file in mesh_folder_files if 'out_point' in file})
        include_set = include_set.union({file for file in mesh_folder_files if 'out_polygon' in file})

        # Include final mesh
        include_set = include_set.union({file for file in mesh_folder_files if f'{mesh.name}_out.gr3' in file})

        # Include files from mesh generation steps
        for mesh_file in ['cache_labeled_masked.npy', 'cache_labeled2.npy', 'dem_list.yaml', 'dem_misses.txt',
                          'hgrid.gr3', 'hgrid.png', 'hgrid_mod.gr3', 'labeled_components.png', 'labeled_components.tif',
                          'labeled_masked.png', 'main.yaml', 'main_echo.yaml', 'prepare_schism.log']:
            include_set = include_set.union({file for file in mesh_folder_files if mesh_file in file})

        # Include angle raster files. If a custom angle file is specified, this may have any number of names.
        include_set = include_set.union({file for file in mesh_folder_files if f'angle_{mesh.name}' in file})

        # Up to this point, include_set should include all files that are not extraneous cases or observation points.
        # Copy these files over. Since this will include all GIS supporting files, treat them all as individual files.
        for file in include_set.union(set(file_list)):
            self._copy_file(file, export_folder_name)

        # Now focus on the individual file folders and geodatabases that are not part of the QAQC steps.
        mesh_folder_folders = [file for file in glob.glob(rf'{mesh.working_dir}\*', recursive=False) if
                             not os.path.isfile(file)]
        include_non_qaqc_folder_set = set()

        # Include all observation point generation geodatabases.
        for obs_pt in mesh.observation_points:
            include_non_qaqc_folder_set = include_non_qaqc_folder_set.union({folder for folder in mesh_folder_folders if obs_pt.name in folder})

        # Include all coefficient generation geodatabases.
        include_non_qaqc_folder_set = include_non_qaqc_folder_set.union({folder for folder in mesh_folder_folders if f'{self.name}_aniso.gdb' in folder})
        include_non_qaqc_folder_set = include_non_qaqc_folder_set.union(
            {folder for folder in mesh_folder_folders if f'{self.name}_diff.gdb' in folder})

        # Include intermediate process folders and target point geodatabase.
        for name in ['_inputs', 'boundary', 'clip_target_tmp.gdb']:
            include_non_qaqc_folder_set = include_non_qaqc_folder_set.union(
                {folder for folder in mesh_folder_folders if name in folder})

        # Now copy these folders over.
        for folder in include_non_qaqc_folder_set:
            self._copy_folder(folder, export_folder_name)

    def _export_qaqc(self, export_folder_name, xs_fc=None, xs_qaqc_folder=None, fit_plot_qaqc_folder=None):
        mesh = self.mesh
        analysis = mesh.analysis
        reach = analysis.reach
        project = reach.project
        gis_file_list = []
        file_list = []
        folder_list = []

        if xs_fc is not None:
            gis_file_list += [xs_fc]

        if xs_qaqc_folder is not None:
            folder_list += [rf'{mesh.working_dir}\{xs_qaqc_folder}']

        if fit_plot_qaqc_folder is not None:
            folder_list += [rf'{mesh.working_dir}\{fit_plot_qaqc_folder}']

        if gis_file_list:
            for fc in gis_file_list:
                self._copy_gis(fc, export_folder_name)

        if folder_list:
            for folder in folder_list:
                self._copy_folder(folder, export_folder_name)

    def _copy_main(self, file, export_folder_name, copy_function):
        rel_path = self._get_relative_path(file)
        mesh = self.mesh
        analysis = mesh.analysis
        reach = analysis.reach
        project = reach.project
        copy_folder = os.path.join(project.project_folder_path, export_folder_name)
        new_file = os.path.join(copy_folder, rel_path)
        new_file_folder = str(pathlib.PureWindowsPath(new_file).parent)
        if not os.path.exists(new_file_folder):
            os.makedirs(new_file_folder)
        print(f'Copying {file} to {new_file}...')
        copy_function(file, new_file)
        return new_file

    def _copy_folder(self, folder, export_folder_name):
        return self._copy_main(folder, export_folder_name, shutil.copytree)

    def __copy_geodatabase(self, gdb_data, export_folder_name):
        rel_path = self._get_relative_path(gdb_data)
        mesh = self.mesh
        analysis = mesh.analysis
        reach = analysis.reach
        project = reach.project
        copy_folder = os.path.join(project.project_folder_path, export_folder_name)
        new_gdb_data = os.path.join(copy_folder, rel_path)
        gdb_path = re.findall(r'(.*?\.gdb)*', new_gdb_data)[0]
        gdb_parent_path = str(pathlib.PureWindowsPath(gdb_path).parent)
        gdb_name = pathlib.PureWindowsPath(gdb_path).name
        if not os.path.exists(gdb_parent_path):
            os.makedirs(gdb_parent_path)
        if not os.path.exists(gdb_path):
            arcpy.CreateFileGDB_management(gdb_parent_path, gdb_name)
        print(f'Copying {gdb_data} to {new_gdb_data}...')
        arcpy.Copy_management(gdb_data, new_gdb_data)
        return new_gdb_data

    def _copy_gis(self, file, export_folder_name):
        if not '.gdb' in file:
            return self._copy_main(file, export_folder_name, arcpy.Copy_management)
        else:
            return self.__copy_geodatabase(file, export_folder_name)

    def _copy_file(self, file, export_folder_name):
        return self._copy_main(file, export_folder_name, shutil.copy2)

    def _get_relative_path(self, file):
        mesh = self.mesh
        analysis = mesh.analysis
        reach = analysis.reach
        project = reach.project
        if project.project_folder_path not in file:
            raise ValueError(f'{file} is not within project directory.')
        else:
            result = os.path.relpath(file, project.project_folder_path)
        return result


class FDAPDEResult:
    def __init__(self, case):
        self.case = case
        self.rsme = None
        self._read_result()

    def _read_result(self):
        patt = r'"summary of rmse for reference points"\n(?:.*?\n)+?\[1,\]\s*(.*?)\n'
        with open(self.result_file, 'r') as f:
            data = f.read()
            result = re.findall(patt, data)
            if len(result) == 0:
                self.rsme = np.nan
            else:
                self.rsme = float(re.findall(patt, data)[0])

    @property
    def result_file(self):
        path = rf'{self.case.mesh.analysis.working_directory.full_path}\{self.case.mesh.name}_{self.case.name}_fdaPDE_output.txt'
        if os.path.exists(path):
            return path


class Mesh:
    def __init__(self, name, analysis, domain_polygon_fc_path, block_polygon_fc_path, open_boundary_fc_path, res=3.0,
                 res_hint_poly_fc_path=None, h0=2.8, seed=1000, res_sdist_list=None, in_angle_raster=None,
                 bathymetry_raster_targets=None):
        self.analysis = analysis
        self.case = {}
        self.name = name
        self._res = res
        self.h0 = h0
        self._set_up_folder()
        self._domain_polygon_input_fc_path = domain_polygon_fc_path
        self._block_polygon_input_fc_path = block_polygon_fc_path
        self._open_boundary_input_fc_path = open_boundary_fc_path
        self._res_hint_poly_input_fc_path = res_hint_poly_fc_path
        self.observation_points = []
        self.target_points = None
        self.seed = seed
        self._initialize()
        self.res_sdist_list = res_sdist_list
        self._in_angle_raster = in_angle_raster
        self._bathymetry_raster_targets = bathymetry_raster_targets

    def __getitem__(self, name):
        if type(name) == str:
            return self.case[name]
        elif type(name) in (int, slice):
            return tuple(self.case.values())[name]

    @property
    def domain_polygon_input_fc_path(self):
        return self._domain_polygon_input_fc_path

    @property
    def block_polygon_input_fc_path(self):
        return self._block_polygon_input_fc_path

    @property
    def open_boundary_input_fc_path(self):
        return self._open_boundary_input_fc_path

    @property
    def bathymetry_raster_targets(self):
        return self._bathymetry_raster_targets

    @property
    def working_dir(self):
        path = rf'{self.analysis.working_directory.full_path}\{self.name}'
        return path

    @property
    def res_name(self):
        name = pathlib.PureWindowsPath(self._res_hint_poly_input_fc_path).stem
        return name

    @property
    def res_hint_fc_path(self):
        path = rf'{self.working_dir}\res_hint\{self.res_name}.shp'
        if os.path.exists(path):
            return path

    @property
    def in_angle_raster(self):
        return self._in_angle_raster

    @property
    def res(self):
        if os.path.isdir(str(self._res)):
            path = rf'{self.working_dir}\res_hint\{self.res_name}_constant.tif'
            if os.path.exists(path):
                return path
        else:
            return self._res

    # This should work with prep_mesh_sdist regardless of whether folder exists.
    def _set_up_folder(self):
        if not os.path.exists(rf'{self.working_dir}'):
            os.mkdir(rf'{self.working_dir}')
            os.mkdir(rf'{self.input_path}')

    def _set_up_input(self, folder_name, fc_path):
        name = pathlib.PureWindowsPath(fc_path).stem
        ext = pathlib.PureWindowsPath(fc_path).suffix
        new_path = rf'{self.working_dir}\{folder_name}\{name}{ext}'
        if not os.path.exists(new_path):
            self.analysis.gpr.record(arcpy.CopyFeatures_management, fc_path, new_path)
        return name, new_path

    def analysis_step(func):
        def inner(self, *args, **kwargs):
            identifier = rf'_{self.analysis.__class__.__name__}_{self.name}_{func.__name__}'
            argstring = ', '.join([arg for arg in args if arg is not None])
            kwstring = ', '.join([f'{kw}={val}' for kw, val in kwargs.items()])
            command_str = rf'{self.analysis.__class__.__name__}[{self.name}].{func.__name__}({argstring}, {kwstring})'.replace(
                ', )', ')')
            # Perform function substitution here. This function is written so analysis_step can be defined in
            # attached objects and add records to the analysis history.
            res = inner_record(self, self.analysis, func, identifier, command_str, rf'\{self.name}', *args, **kwargs)
            return res

        return inner

    def add_case(self, case):
        self.case[case.name] = case
        case.mesh = self

    def remove_cases(self, case_name_list):
        self.case = {case.name: case for case in self.case.values() if case.name not in case_name_list}

    @property
    def input_path(self):
        input_path = rf'{self.working_dir}\_inputs'
        return input_path

    def _initialize(self):
        if not os.path.exists(self.input_path):
            os.mkdir(self.input_path)
        self._set_up_input('_inputs', self._domain_polygon_input_fc_path)
        self._set_up_input('_inputs', self._block_polygon_input_fc_path)
        self._set_up_input('_inputs', self._open_boundary_input_fc_path)

    @analysis_step
    def clip_dem(self):
        bathymetry_rasters = []
        dem_mask = rf'{self.input_path}\dem_mask.tif'
        self.analysis.gpr.record(arcpy.AddField_management, rf'{self.domain_polygon_fc_path}', 'Elev',
                                 'DOUBLE')
        self.analysis.gpr.record(arcpy.CalculateField_management, rf'{self.domain_polygon_fc_path}', 'Elev',
                                 '1000', 'PYTHON')

        cellsize = arcpy.Describe(self.analysis.source_dem_path).meanCellHeight
        old_snap = arcpy.env.snapRaster
        arcpy.env.snapRaster = self.analysis.source_dem_path
        self.analysis.gpr.record(arcpy.PolygonToRaster_conversion, self.domain_polygon_fc_path, 'Elev', dem_mask,
                                 cellsize=cellsize)
        dem_clip = rf'{self.input_path}\dem_domain_clip.tif'

        self.analysis.gpr.record(arcpy.Clip_management, self.analysis.source_dem_path, '#',
                                 dem_clip, in_template_dataset=self.domain_polygon_fc_path,
                                 clipping_geometry='ClippingGeometry')

        rasters = [dem_clip, dem_mask]
        self.analysis.gpr.record(arcpy.MosaicToNewRaster_management, ';'.join(rasters), rf'{self.input_path}',
                                 f'dem_preprocessed.tif',
                                 pixel_type='32_BIT_FLOAT', cellsize=cellsize, number_of_bands=1, mosaic_method='FIRST')
        dem_preprocess = rf'{self.input_path}\dem_preprocessed.tif'
        arcpy.env.snapRaster = old_snap

        domain_erase = rf'{self.input_path}\{self.domain_name}_erase.shp'
        self.analysis.gpr.record(arcpy.Erase_analysis, self.domain_polygon_fc_path, self.analysis.breaklines_fc_path,
                                 domain_erase)

        dem_clip_2 = rf'{self.input_path}\dem_domain_clip_1m.tif'
        self.analysis.gpr.record(arcpy.Resample_management, dem_preprocess, dem_clip_2, '1 1', 'BILINEAR')

        if type(self._bathymetry_raster_targets) == list:
            old_snap = arcpy.env.snapRaster
            arcpy.env.snapRaster = dem_clip_2
            dem_preclip = rf'{self.input_path}\dem_domain_clip_1m_multibeam.tif'
            for bathy_raster in self._bathymetry_raster_targets:
                bathy_raster_name = pathlib.PureWindowsPath(bathy_raster).stem
                bathy_resample = rf'{self.input_path}\{bathy_raster_name}_resample.tif'
                self.analysis.gpr.record(arcpy.Resample_management, bathy_raster, bathy_resample, '1 1',
                                         'BILINEAR')
                bathymetry_rasters.append(bathy_resample)
            self.analysis.gpr.record(arcpy.MosaicToNewRaster_management, ';'.join(bathymetry_rasters + [dem_clip_2]),
                                     rf'{self.input_path}', 'dem_domain_clip_1m_bathy_raster.tif',
                                     pixel_type='32_BIT_FLOAT', cellsize=1, number_of_bands=1, mosaic_method='FIRST')
            # dem_preclip = resampled multibeams merged with dem_clip_2
            arcpy.env.snapRaster = old_snap
        else:
            dem_preclip = dem_clip_2

        dem_clip_1m = rf'{self.input_path}\dem_domain_clip_no_water_1m.tif'
        self.analysis.gpr.record(arcpy.Clip_management, dem_preclip, '#',
                                 dem_clip_1m,
                                 in_template_dataset=domain_erase, nodata_value=-3.402823e38,
                                 clipping_geometry='ClippingGeometry')

    @property
    def domain_name(self):
        name = pathlib.PureWindowsPath(self._domain_polygon_input_fc_path).stem
        return name

    @property
    def domain_polygon_fc_path(self):
        path = rf'{self.input_path}\{self.domain_name}.shp'
        if os.path.exists(path):
            return path

    @property
    def domain_erase_polygon_fc_path(self):
        if os.path.exists(rf'{self.input_path}\{self.domain_name}_erase.shp'):
            return rf'{self.input_path}\{self.domain_name}_erase.shp'

    @property
    def block_polygon_name(self):
        name = pathlib.PureWindowsPath(self._block_polygon_input_fc_path).stem
        return name

    @property
    def block_polygon_fc_path(self):
        path = rf'{self.input_path}\{self.block_polygon_name}.shp'
        if os.path.exists(path):
            return path

    @property
    def open_boundary_name(self):
        name = pathlib.PureWindowsPath(self._open_boundary_input_fc_path).stem
        return name

    @property
    def open_boundary_fc_path(self):
        path = rf'{self.input_path}\{self.open_boundary_name}.shp'
        if os.path.exists(path):
            return path

    @property
    def dem_clip_1m(self):
        path = rf'{self.input_path}\dem_domain_clip_no_water_1m.tif'
        if os.path.exists(path):
            return path

    @property
    def boundary(self):
        bdy = rf'{self.working_dir}\boundary\{self.name}_out_polygon_boundary.shp'
        if os.path.exists(bdy):
            return bdy

    @property
    def polygons(self):
        polygons = rf'{self.working_dir}\{self.name}_out_polygon.shp'
        if os.path.exists(polygons):
            return polygons

    @property
    def nodes(self):
        point = rf'{self.working_dir}\{self.name}_out_point.shp'
        if os.path.exists(point):
            return point

    @analysis_step
    def create_res_hint(self, base_value, field_name='res_hint'):
        # Field name assumed to exist when this function is run.
        create_res_hint_path = rf'{self.working_dir}\res_hint'
        if not os.path.exists(create_res_hint_path):
            os.mkdir(create_res_hint_path)
        name, path = self._set_up_input('res_hint', self._res_hint_poly_input_fc_path)
        res_hint_constant = rf'{self.working_dir}\res_hint\{self.res_name}_constant.tif'
        raster = arcpy.sa.CreateConstantRaster(base_value, 'Float', 1, self.dem_clip_1m)
        raster.save(res_hint_constant)
        old_snap = arcpy.env.snapRaster
        arcpy.env.snapRaster = res_hint_constant
        res_hint_poly_raster_path = rf'{self.working_dir}\res_hint\{self.res_name}_poly.tif'
        arcpy.PolygonToRaster_conversion(self.res_hint_fc_path, field_name, res_hint_poly_raster_path, cellsize=1)
        rasters = ';'.join([res_hint_poly_raster_path, res_hint_constant])
        arcpy.MosaicToNewRaster_management(rasters, rf'{self.working_dir}\res_hint',
                                           rf'{self.res_name}.tif',
                                           pixel_type='32_BIT_FLOAT', cellsize=1, number_of_bands=1,
                                           mosaic_method='FIRST')
        arcpy.env.snapRaster = old_snap

    @analysis_step
    def prep_mesh_sdist(self, from_sdist_list=None):
        self._prep_mesh_sdist_inner()
        tmp_raster = None
        if from_sdist_list is not None:
            old_extent = arcpy.env.extent
            arcpy.env.extent = self.dem_clip_1m
            create_res_hint_path = rf'{self.working_dir}\res_hint'
            if not os.path.exists(create_res_hint_path):
                os.mkdir(create_res_hint_path)
            for i, sdist_pair in enumerate(from_sdist_list):
                sdist, value = sdist_pair
                sdist *= -1
                arcpy.CopyRaster_management(rf'{self.working_dir}\sdist_{self.name}.tif',
                                            rf'{self.working_dir}\res_hint\prev_sdist.tif')
                sdist_raster = arcpy.sa.Raster(rf'{self.working_dir}\res_hint\prev_sdist.tif')
                if i == 0:
                    tmp_raster = arcpy.sa.SetNull((sdist_raster < sdist) | (sdist_raster >= 0), value)
                else:
                    tmp_raster = arcpy.sa.Con((sdist_raster >= sdist) & (sdist_raster <= 0), value, tmp_raster)
            tmp_raster = arcpy.sa.Con(arcpy.sa.IsNull(tmp_raster), self._res, tmp_raster)
            # if self.res_hint_fc_path is not None:
            #     orig_res_hint = arcpy.sa.Raster(rf'{create_res_hint_path}\{self.res_name}.tif')
            #     tmp_raster = arcpy.sa.Con(arcpy.sa.IsNull(orig_res_hint), tmp_raster, orig_res_hint)
            tmp_raster.save(rf'{self.working_dir}\res_hint\res_from_sdist_tmp.tif')
            arcpy.env.extent = old_extent
            self._prep_mesh_sdist_inner()

    def _prep_mesh_sdist_inner(self):
        infile = rf'{self.name}\_inputs\dem_domain_clip_no_water_1m.tif'
        from_sdist = rf'{self.working_dir}\res_hint\res_from_sdist_tmp.tif'
        if os.path.isdir(str(self.res)) & (not (os.path.exists(from_sdist))):
            res = rf'{self.name}\res_hint\{self.res_name}.tif'
        elif os.path.exists(from_sdist):
            res = rf'{self.working_dir}\res_hint\res_from_sdist_tmp.tif'
        else:
            res = self.res
        print('In the distmesh conda environment, copy and run the following commands')
        print(rf'cd {self.analysis.working_directory.full_path}')
        print(
            rf'prep_mesh_sdist --infile {infile} --bnd_shapefile {self.name}\_inputs\{self.block_polygon_name}.shp --res {res} --adapt_res --h0={self.h0} --outdir={self.name} --seed={self.seed}')
        input('Once this process is done, press ENTER to continue.')
        if not (os.path.exists(rf'{self.analysis.working_directory.full_path}\{self.name}\hgrid.gr3')):
            raise RuntimeError('hgrid.gr3 not found')

    @analysis_step
    def sdist_to_direction(self):
        if self._in_angle_raster is None:
            print('In the distmesh conda environment, copy and run the following commands')
            print(rf'cd {self.working_dir}')
            print(rf'sdist_to_direction --input sdist_{self.name}.tif --output angle_{self.name}.tif')
            input('Once this process is done, press ENTER to continue.')
            if not (os.path.exists(rf'{self.working_dir}\angle_{self.name}.tif')):
                raise RuntimeError('hgrid_mod.gr3 not found')
        elif self._in_angle_raster is not None:
            print('Angle raster overriden. Using user-provided angle raster.')
            name = pathlib.PureWindowsPath(self._in_angle_raster).stem
            arcpy.Copy_management(self._in_angle_raster, rf'{self.working_dir}\{name}.tif')

    @property
    def angle_raster(self):
        if self._in_angle_raster is None:
            angle = rf'{self.working_dir}\angle_{self.name}.tif'
            if os.path.exists(angle):
                return angle
        elif self._in_angle_raster is not None:
            name = pathlib.PureWindowsPath(self._in_angle_raster).stem
            angle_raster = rf'{self.working_dir}\{name}.tif'
            return angle_raster

    @analysis_step
    def remove_skewed_cells(self):
        print('In the distmesh conda environment, copy and run the following commands')
        print(rf'cd {self.working_dir}')
        print(rf'remove_skewed_cells --input hgrid.gr3 --output hgrid_mod.gr3')
        input('Once this process is done, press ENTER to continue.')
        if not (os.path.exists(rf'{self.working_dir}\hgrid_mod.gr3')):
            raise RuntimeError('hgrid_mod.gr3 not found')

    @analysis_step
    def convert_linestrings(self):
        print('In the distmesh conda environment, copy and run the following commands')
        print(rf'cd {self.working_dir}')
        print(rf'convert_linestrings --input _inputs\{self.open_boundary_name}.shp --output '
              rf'{self.open_boundary_name}.yaml')
        input('Once this process is done, press ENTER to continue.')
        if not (os.path.exists(rf'{self.working_dir}\{self.open_boundary_name}.yaml')):
            raise RuntimeError(rf'{self.open_boundary_name}.yaml not found')

    @analysis_step
    def copy_schism_yaml(self):
        shutil.copy(r"resources\main.yaml", rf'{self.working_dir}\main.yaml')
        bdy_name = pathlib.PureWindowsPath(self.open_boundary_fc_path).stem
        substitute_text_tags(rf'{self.working_dir}\main.yaml',
                             {'<BDY_NAME>': bdy_name,
                              '<NAME>': self.name})
        shutil.copy(r"resources\dem_list.yaml", rf'{self.working_dir}\dem_list.yaml')
        dem = self.analysis.source_dem_path
        dem_name = 'dem'
        dem_clip_1m = rf'{dem_name}_domain_clip_no_water_1m'
        substitute_text_tags(rf'{self.working_dir}\dem_list.yaml', {'<DEM_NAME>': dem_clip_1m})

    @analysis_step
    def prepare_schism(self):
        print('In the distmesh conda environment, copy and run the following commands')
        print(rf'cd {self.working_dir}')
        print(rf'prepare_schism main.yaml')
        input('Once this process is done, press ENTER to continue.')
        if not (os.path.exists(rf'{self.working_dir}\{self.name}_out.gr3')):
            raise RuntimeError(f'{self.name}_out.gr3 not found')

    @analysis_step
    def convert_mesh(self):
        print('In the distmesh conda environment, copy and run the following commands')
        print(rf'cd {self.working_dir}')
        print(
            rf'convert_mesh --input {self.name}_out.gr3 --output {self.name}_out.shp --crs "+proj=utm +zone=10 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"')
        input('Once this process is done, press ENTER to continue.')
        if not (
                os.path.exists(rf'{self.working_dir}\{self.name}_out_polygon.shp')):
            raise RuntimeError(f'{self.name}_out_polygon.shp not found')

    @analysis_step
    def create_check_boundaries(self):
        create_check_boundary_path = rf'{self.working_dir}\boundary'
        if not os.path.exists(create_check_boundary_path):
            os.mkdir(create_check_boundary_path)
        self.analysis.gpr.record(arcpy.Dissolve_management,
                                 rf'{self.working_dir}\{self.name}_out_polygon.shp',
                                 rf'{self.working_dir}\boundary\{self.name}_out_polygon_boundary.shp')
        mesh_boundary = f'lyr_fdaPDE_{self.analysis.reach.name}_{self.analysis.name}_{self.name}_boundary'
        arcpy.MakeFeatureLayer_management(rf'{self.working_dir}\boundary\{self.name}_out_polygon_boundary.shp',
                                          mesh_boundary)
        mesh_nodes = f'lyr_fdaPDE_{self.analysis.reach.name}_{self.analysis.name}_{self.name}_nodes'
        arcpy.MakeFeatureLayer_management(rf'{self.working_dir}\{self.name}_out_point.shp', mesh_nodes)
        arcpy.SelectLayerByLocation_management(mesh_nodes, 'BOUNDARY_TOUCHES', mesh_boundary)
        n_touching = arcpy.GetCount_management(mesh_nodes)[0]
        if int(n_touching) > 0:
            data_field = [field.name for field in arcpy.ListFields(mesh_nodes) if field.type == 'Double'][0]
            arcpy.SelectLayerByAttribute_management(mesh_nodes, 'SUBSET_SELECTION', f'"{data_field}" >= -2')
            block_polygon_lyr = f'lyr_fdaPDE_{self.name}_block_polygon'
            arcpy.MakeFeatureLayer_management(self.block_polygon_fc_path, block_polygon_lyr)
            arcpy.SelectLayerByLocation_management(mesh_nodes, 'WITHIN_A_DISTANCE', block_polygon_lyr,
                                                   search_distance=f'{self._res / 2} Meters',
                                                   selection_type='REMOVE_FROM_SELECTION')
            n_qaqc = arcpy.GetCount_management(mesh_nodes)[0]
            if int(n_qaqc) > 0:
                arcpy.CopyFeatures_management(mesh_nodes, rf'{self.working_dir}\boundary\{self.name}_qaqc.shp')
                print(
                    f'{n_qaqc} points touch the mesh boundary with values >= -2. Review and decide whether a different'
                    f' resolution mesh is needed.')
            else:
                print('No points touching the mesh boundary with values >= -2 were detected.')
        else:
            print('No points touching the mesh boundary with values >= -2 were detected.')

    def add_observation_points(self, obs_points_list):
        # This must be copied - otherwise analysis will get pointed to something different when this object is used
        # for a different analysis.
        if type(obs_points_list) != list:
            obs_points_list = [obs_points_list]
        for obs_point in obs_points_list:
            obs_point_copy = copy.deepcopy(obs_point)
            self.observation_points.append(obs_point_copy)
            obs_point_copy.mesh = self

    def clip_export_points(self):
        remove_points = []
        for obs_point in self.observation_points:
            result = obs_point.clip_export_points()
            if result == False:
                print(f'{obs_point.name} not within target polygon. Removing from observation points.')
                remove_points.append(obs_point)
        self.observation_points = [obs_point for obs_point in self.observation_points if obs_point not in remove_points]

    def add_target(self, target_obj=None):
        self.target_points = TargetPoints() if target_obj is None else target_obj
        self.target_points.mesh = self

    @analysis_step
    def create_target_points(self):
        self.target_points.create_target_points()

    def prep_plot_fit(self, plot=True, figsize=(4.5, 4.5), folder_name='fit_qaqc', precision=False, linear_fit=True,
                      hexbin=False, plot_pct=False):
        qaqc_dir = rf'{self.working_dir}\{folder_name}'
        tmp_dir = fm.PathedWorkingDirectory(qaqc_dir)
        tmp_gdb = fm.PathedGeoDataBase(rf'{qaqc_dir}\fit_plot.gdb')
        tmp_gdb.set_env()
        result_rasters = [arcpy.sa.Raster(case.result_raster) for case in self.case.values()
                          if (case.result_raster is not None) & (case.xs_plot is True)]
        result_cases = [case.name for case in self.case.values()
                        if (case.result_raster is not None) & (case.xs_plot is True)]
        results = [[raster, name] for raster, name in zip(result_rasters, result_cases)]
        fields = ['POINT_Z'] + result_cases
        for obs_pt in self.observation_points:
            if not isinstance(obs_pt, AugmentPoints):
                arcpy.CopyFeatures_management(obs_pt.fc_path, rf'{obs_pt.name}')
                method = 'BILINEAR'
                # method = 'BILINEAR' if not isinstance(obs_pt, ShorelinePoints) else 'NONE'
                print(obs_pt.name)
                arcpy.sa.ExtractMultiValuesToPoints(rf'{obs_pt.name}', results, method)
                df = pd.DataFrame(arcpy.da.TableToNumPyArray(rf'{obs_pt.name}',
                                                             [field.name for field in
                                                              arcpy.Describe(rf'{obs_pt.name}').fields
                                                              if field.name in fields]))
                df.to_csv(rf'{tmp_dir.full_path}\{obs_pt.name}.csv', index=False)

        if plot:
            self.plot_fit_by_case(figsize, folder_name, precision, linear_fit, hexbin, plot_pct)

    def plot_fit_by_case(self, figsize=(5.37, 4.5), folder_name='fit_qaqc', precision=True, linear_fit=True,
                         hexbin=False,
                         plot_pct=False):
        qaqc_dir = rf'{self.working_dir}\{folder_name}'
        df_bathy_list = []
        df_shoreline_list = []
        for obs_pt in self.observation_points:
            if not isinstance(obs_pt, AugmentPoints):
                df_i = pd.read_csv(rf'{qaqc_dir}\{obs_pt.name}.csv')
                if isinstance(obs_pt, BathyPoints):
                    df_bathy_list.append(df_i)
                elif isinstance(obs_pt, ShorelinePoints):
                    df_shoreline_list.append(df_i)
        df_bathy = pd.concat(df_bathy_list).reset_index()
        df_shoreline = pd.concat(df_shoreline_list).reset_index()
        df_all = pd.concat([df_bathy, df_shoreline]).reset_index()
        result_cases = [case for case in self.case.values()
                        if (case.result_raster is not None) & (case.xs_plot is True)]
        result_df_list = []
        result_all_df_list = []
        for case in result_cases:
            for obs_pt in self.observation_points:
                if not isinstance(obs_pt, AugmentPoints):
                    if isinstance(obs_pt, BathyPoints):
                        pt_type = 'Bathymetric points'
                    elif isinstance(obs_pt, ShorelinePoints):
                        pt_type = 'Shoreline points'
                    self._create_fit_plots([obs_pt],
                                           case, figsize, pt_type, folder_name, linear_fit, obs_pt.precision, hexbin,
                                           plot_pct)
            result_df_list.append(
                self._create_fit_plots(
                    [obs_pt for obs_pt in self.observation_points
                     if (isinstance(obs_pt, BathyPoints)) and not (isinstance(obs_pt, AugmentPoints))],
                    case, figsize, 'Bathymetric points', folder_name, linear_fit, 1.0, hexbin, plot_pct))
            result_df_list.append(
                self._create_fit_plots(
                    [obs_pt for obs_pt in self.observation_points if isinstance(obs_pt, ShorelinePoints)],
                    case, figsize, 'Shoreline points', folder_name, linear_fit, 0.283, hexbin, plot_pct))

        for case in result_cases:
            name = case.name
            result_all_df, all_diff = tabulate_qaqc_stats(df_all, name)
            result_all_df_list.append(result_all_df)
        pd.concat(result_all_df_list, axis=1).to_csv(rf'{qaqc_dir}\{self.name}_all_points_stats.csv')
        pd.concat(result_df_list, axis=1).to_csv(rf'{qaqc_dir}\{self.name}_points_stats_by_location.csv')

    def _create_fit_plots(self, obs_pt_list, case, figsize, point_type, folder_name, linear_fit, precision, hexbin,
                          plot_pct):
        colors = cm.YlOrRd(np.linspace(1 / 9, 8 / 9, 9))
        cmap = mcolors.ListedColormap(colors)
        qaqc_dir = rf'{self.working_dir}\{folder_name}'
        fig, ax = plt.subplots(figsize=figsize)
        df_list = []
        result_name_title = obs_pt_list[0].data_short_name if len(
            obs_pt_list) == 1 else point_type
        # If data short name is shoreline_points, this will coincide with the name used for labeling all shoreline
        # points including from LiDAR. Give it a different name to avoid this conflict.
        if (obs_pt_list[0].data_short_name == 'shoreline_points') & (len(obs_pt_list) == 1):
            result_name_title = 'Shoreline points (boundary)'
        result_name = result_name_title.lower().replace('(', '').replace(')', '').replace(' ', '_')
        for obs_pt in obs_pt_list:
            if not isinstance(obs_pt, AugmentPoints):
                df_i = pd.read_csv(rf'{qaqc_dir}\{obs_pt.name}.csv')
                df_list.append(df_i)
        df = pd.concat(df_list).reset_index().drop(columns=['index'])
        df_min = df.min().min()
        df_max = df.max().max()
        name = case.name
        for obs_pt, df_i in zip(obs_pt_list, df_list):
            if not isinstance(obs_pt, AugmentPoints):
                if len(df_list) != 1:
                    if hexbin:
                        ax.hexbin(df['POINT_Z'], df[name], gridsize=100, cmap=cmap, mincnt=1)
                    else:
                        ax.plot(df['POINT_Z'], df[name], '.', alpha=0.15, label=obs_pt.data_long_name)
                else:
                    if hexbin:
                        ax.hexbin(df['POINT_Z'], df[name], gridsize=100, cmap=cmap, mincnt=1)
                    else:
                        ax.plot(df['POINT_Z'], df[name], '.', alpha=0.15)
        ax.set_ylabel(f'Predicted (NAVD88 m)')
        ax.set_xlabel(f'Observed (NAVD88 m)')
        ax.plot([df_min, df_max], [df_min, df_max], 'k-', lw=1)
        ax.plot([df_min, df_max], [df_min + precision, df_max + precision], 'k--', lw=1)
        ax.plot([df_min, df_max], [df_min - precision, df_max - precision], 'k--', lw=1)
        if linear_fit:
            slope, intercept, r_squared = plot_linear_fit(df, name)
        result_df, diff = tabulate_qaqc_stats(df, name, point_type.lower())
        ax.set_xlim(df_min, df_max)
        ax.set_ylim(df_min, df_max)
        ax.set_title(result_name_title)
        plt.grid(which='minor', color='#dddddd', lw=0.8)
        # ax.yaxis.set_minor_locator(tkr.AutoMinorLocator())
        # ax.xaxis.set_minor_locator(tkr.AutoMinorLocator())
        fig.tight_layout()
        fig.savefig(rf'{qaqc_dir}\{self.name}_{name}_{result_name}_fit.png', dpi=300)
        fig2, ax2 = plt.subplots(figsize=(figsize[0], figsize[1]))
        # abs_diff = diff.abs()
        # abs_diff_max = np.ceil(abs_diff.max() * 10) / 10
        abs_diff_max = 3
        bins = np.arange(-abs_diff_max, abs_diff_max, 0.1)
        _, _, patches = ax2.hist(diff, bins=bins, histtype='bar', edgecolor='#000000', facecolor='#dddddd')
        total = diff.shape[0]
        n_less = diff[diff < -precision].shape[0]
        n_center = diff[(diff >= -precision) & (diff <= precision)].shape[0]
        n_greater = diff[diff > precision].shape[0]
        percent_less = n_less / total * 100
        percent_center = n_center / total * 100
        percent_greater = n_greater / total * 100
        ymax = ax2.get_ylim()[1]
        xmin = ax2.get_xlim()[0]
        xmax = ax2.get_xlim()[1]
        offset = -10
        if plot_pct:
            ax2.annotate(f'{percent_less:0.1f}%', ((-precision - xmin) / 2 + xmin, ymax),
                         (0, offset), textcoords='offset points', ha='center', va='top')
            ax2.annotate(f'{percent_center:0.1f}%', (0, ymax), (0, offset), textcoords='offset points', ha='center',
                         va='top')
            ax2.annotate(f'{percent_greater:0.1f}%', (xmax - (xmax - precision) / 2, ymax), (0, offset),
                         textcoords='offset points',
                         ha='center', va='top')
        ax2.axvline(-precision, ls='--', color='black', lw=1)
        ax2.axvline(precision, ls='--', color='black', lw=1)
        ax2.set_title(result_name_title)
        plt.grid(which='minor', color='#dddddd', lw=0.8)
        ax2.yaxis.set_major_locator(tkr.MultipleLocator(20000))
        # ax2.yaxis.set_minor_locator(tkr.AutoMinorLocator())
        # ax2.xaxis.set_minor_locator(tkr.AutoMinorLocator())
        ax2.yaxis.set_major_formatter(tkr.StrMethodFormatter('{x:0,.0f}'))
        ax2.set_xlabel(f'Predicted - Observed (m)')
        ax2.set_ylabel(f'Frequency')
        plt.yticks(rotation=90, ha='right')
        fig2.tight_layout()
        fig2.savefig(rf'{qaqc_dir}\{self.name}_{name}_{result_name}_hist_by_location.png',
                     dpi=300)
        fig.clf()
        fig2.clf()
        plt.close(fig)
        plt.close(fig2)
        del fig
        del fig2
        return result_df

    def plot_xs_lines(self, xs_line_fc, radius='2 Meter', plot=True, figsize=(12, 6),
                      folder_name='xs_qaqc', error_bars=False, final=False):
        radius_value = re.findall(r'(.*?)\s', radius)[0]
        radius_unit = re.findall(r'\s(.*)', radius)[0]
        print(f'{radius_value} {radius_unit}')
        qaqc_dir = rf'{self.working_dir}\{folder_name}'
        tmp_dir = fm.PathedWorkingDirectory(qaqc_dir)
        tmp_gdb = fm.PathedGeoDataBase(rf'{qaqc_dir}\xs_plot.gdb')
        tmp_gdb.set_env()
        arcpy.env.outputMFlag = 'Enabled'
        arcpy.CopyFeatures_management(xs_line_fc, rf'{self.name}_xs_line')
        arcpy.AddField_management(rf'{self.name}_xs_line', 'Start', 'DOUBLE')
        arcpy.AddField_management(rf'{self.name}_xs_line', 'End', 'DOUBLE')
        arcpy.CalculateField_management(rf'{self.name}_xs_line', 'Start', '0', 'Python 3')
        arcpy.CalculateField_management(rf'{self.name}_xs_line', 'End', '!Shape.Length!', 'Python 3')
        arcpy.env.outputMFlag = 'SameAsInput'
        old_snap = arcpy.env.snapRaster
        arcpy.env.snapRaster = self[0].result_raster
        arcpy.Resample_management(self.dem_clip_1m, 'dem_clip', '2 2', 'BILINEAR')
        arcpy.env.snapRaster = old_snap
        arcpy.GeneratePointsAlongLines_management(rf'{self.name}_xs_line', rf'{self.name}_xs_points', 'DISTANCE',
                                                  '1 Meter',
                                                  Include_End_Points='END_POINTS')
        result_rasters = [arcpy.sa.Raster(case.result_raster) for case in self.case.values()
                          if (case.result_raster is not None) & (case.xs_plot is True)]
        result_cases = [case.name for case in self.case.values()
                        if (case.result_raster is not None) & (case.xs_plot is True)]

        results = [[raster, name] for raster, name in zip(result_rasters, result_cases)]
        arcpy.sa.ExtractMultiValuesToPoints(rf'{self.name}_xs_points', results, 'BILINEAR')
        arcpy.CreateRoutes_lr(rf'{self.name}_xs_line', 'Name', rf'{self.name}_xs_line_route', 'TWO_FIELDS',
                              'Start', 'End')
        arcpy.Buffer_analysis(xs_line_fc, 'xs_buffer', f'{float(radius_value) * 2} {radius_unit}',
                              dissolve_option='ALL')
        fc_list = []
        for obs_pt in self.observation_points:
            fc = obs_pt.data_short_name.lower().replace(" ", "_")
            arcpy.Clip_analysis(obs_pt.fc_path, 'xs_buffer', fc)
            arcpy.AddField_management(fc, 'DataName', 'TEXT')
            drop_fields = [field.name for field in arcpy.Describe(fc).fields
                           if (field.name not in ['OID', 'POINT_X', 'POINT_Y',
                                                  'POINT_Z', 'DataName'])
                           & (field.type not in ['OID', 'Geometry'])]
            if len(drop_fields) > 0:
                arcpy.DeleteField_management(fc, drop_fields)
            arcpy.CalculateField_management(fc, 'DataName', f"'{fc}'", 'PYTHON')
            fc_list.append(fc)
        arcpy.Merge_management(fc_list, 'observation_points')

        arcpy.LocateFeaturesAlongRoutes_lr(rf'{self.name}_xs_points', rf'{self.name}_xs_line_route', 'Name',
                                           radius, rf'{qaqc_dir}\{self.name}_xs_points.csv', 'Name POINT Station')
        arcpy.LocateFeaturesAlongRoutes_lr('observation_points',
                                           rf'{self.name}_xs_line_route', 'Name',
                                           radius, rf'{qaqc_dir}\{self.name}_observation_points.csv',
                                           'Name POINT Station')

        if plot:
            arcpy.EnableAttachments_management(rf'{self.name}_xs_line')
            self.plot_results(figsize, folder_name, error_bars, first=True, final=final)

    def plot_results(self, figsize=(12, 6),
                     folder_name='xs_qaqc', error_bars=False, first=False, final=False):
        qaqc_dir = rf'{self.working_dir}\{folder_name}'
        result_cases = [case for case in self.case.values()
                        if (case.result_raster is not None) & (case.xs_plot is True)]
        result_long_names = [case.long_name for case in self.case.values()
                             if (case.result_raster is not None) & (case.xs_plot is True)]

        obs = pd.read_csv(rf'{qaqc_dir}\{self.name}_observation_points.csv',
                          usecols=['Name', 'Station', 'Distance', 'POINT_X', 'POINT_Y', 'POINT_Z', 'DataName'])
        obs_dict = {obs_pt.data_short_name.lower().replace(" ", "_"): obs_pt for obs_pt in self.observation_points}
        results = pd.read_csv(rf'{qaqc_dir}\{self.name}_xs_points.csv')

        xs_names = results['Name'].unique().tolist()

        df_dict_attach = {'Name': [],
                          'Path': []}
        for xs_name in xs_names:
            print(xs_name)
            df_dict_attach['Name'].append(xs_name)
            df_dict_attach['Path'].append(f'{xs_name}.png')
            fig, ax = plt.subplots(figsize=figsize)
            obs_query = obs.query('Name == @xs_name')
            results_query = results.query('Name == @xs_name')
            for i, (case, long_name) in enumerate(zip(result_cases, result_long_names)):
                name = case.name
                # result = results_query[['Station', name]].dropna(subset=[name]).copy()
                result = results_query[['Station', name]].copy()
                print(case.name)
                print(result['Station'].min())
                if (i <= 9) & (not final):
                    lt = 'solid'
                elif (i >= 10) & (i <= 19) & (not final):
                    lt = 'dashed'
                elif (i >= 20) & (i <= 29) & (not final):
                    lt = 'dashdot'
                elif (i >= 30) & (i <= 39) & (not final):
                    lt = 'dotted'
                elif (i >= 40) & (i <= 49) & (not final):
                    lt = (0, (5, 1))
                else:
                    lt = '-'
                if long_name is None:
                    if final:
                        if case.final:
                            ax.plot(result['Station'], result[name], linestyle=lt, lw=case.lw,
                                    zorder=case.zorder, color='#DC143C')
                        else:
                            ax.plot(result['Station'], result[name], linestyle=lt, lw=case.lw,
                                    zorder=case.zorder, color='#91BAE2')
                    else:
                        ax.plot(result['Station'], result[name], linestyle=lt, lw=case.lw, label=name,
                                zorder=case.zorder)
                else:
                    if final:
                        if case.final:
                            ax.plot(result['Station'], result[name], linestyle=lt, lw=case.lw,
                                    zorder=case.zorder, color='#DC143C')
                        else:
                            ax.plot(result['Station'], result[name], linestyle=lt, lw=case.lw,
                                    zorder=case.zorder, color='#91BAE2')
                    else:
                        ax.plot(result['Station'], result[name], linestyle=lt, lw=case.lw, label=long_name,
                                zorder=case.zorder)
            for data_name, obs_pt in obs_dict.items():
                b_data_name = obs_query['DataName'] == data_name
                if error_bars:
                    ax.errorbar(obs_query.loc[b_data_name, 'Station'],
                                obs_query.loc[b_data_name, 'POINT_Z'], obs_pt.precision, fmt=' ',
                                color='#888888',
                                capsize=0, capthick=0.5, lw=0.8, zorder=2.7)
                # TODO: Fix point zorder. Hard code fix present so final plot points come out in front of other cases
                # tried.
                if obs_pt.color is not None:
                    ax.plot(obs_query.loc[b_data_name, 'Station'], obs_query.loc[b_data_name, 'POINT_Z'],
                            obs_pt.fmt, obs_pt.ms, obs_pt.zorder, fillstyle=obs_pt.fillstyle, alpha=obs_pt.alpha,
                            color=obs_pt.color)
                else:
                    if final:
                        ax.plot(obs_query.loc[b_data_name, 'Station'], obs_query.loc[b_data_name, 'POINT_Z'],
                                obs_pt.fmt, obs_pt.ms, obs_pt.zorder, fillstyle=obs_pt.fillstyle, alpha=obs_pt.alpha,
                                zorder=3.05)
                    else:
                        ax.plot(obs_query.loc[b_data_name, 'Station'], obs_query.loc[b_data_name, 'POINT_Z'],
                                obs_pt.fmt, obs_pt.ms, obs_pt.zorder, fillstyle=obs_pt.fillstyle, alpha=obs_pt.alpha)
            ax.set_xlabel('Station (m)')
            ax.set_ylabel('Elevation (m NAVD88)')
            if not final:
                ax.legend(loc='lower right')
            ax.yaxis.set_minor_locator(tkr.AutoMinorLocator())
            ax.xaxis.set_minor_locator(tkr.AutoMinorLocator())
            plt.grid(which='minor', color='#dddddd', lw=0.8)
            plt.grid(which='major', color='#cccccc', lw=1)
            fig.tight_layout()
            fig.savefig(rf'{qaqc_dir}\{xs_name}.png', dpi=300)
            plt.clf()
            plt.close(fig)
        df_attach = pd.DataFrame(df_dict_attach)
        df_attach.to_csv(rf'{qaqc_dir}\attach.csv', index=False)
        if not first:
            arcpy.RemoveAttachments_management(rf'{qaqc_dir}\xs_plot.gdb\{self.name}_xs_line',
                                               'Name', rf'{qaqc_dir}\attach.csv', 'Name', 'Path')
        arcpy.AddAttachments_management(rf'{qaqc_dir}\xs_plot.gdb\{self.name}_xs_line',
                                        'Name', rf'{qaqc_dir}\attach.csv', 'Name',
                                        'Path', qaqc_dir)


class GeoProcessingRecorder:
    def __init__(self, analysis):
        self._analysis = analysis
        self._count = 1

    def record(self, arcpy_fn, *args, **kwargs):
        argstring = ', '.join([arg for arg in args if arg is not None])
        kwstring = ', '.join([f'{kwargs}={val}' for kw, val in kwargs.items()])
        command_str = f'{arcpy_fn.__name__}({argstring}, {kwstring})'.replace(', )', ')')
        print(f'\tRunning {command_str}')
        result = arcpy_fn(*args, **kwargs)
        with open(rf'{self._analysis.working_directory.full_path}\history.txt', 'a') as f:
            f.write('\n\t' + command_str)
        return result


def substitute_text_tags(file_path, replace_dict):
    with open(file_path, 'r') as f:
        data = f.read()

    for key, value in replace_dict.items():
        data = data.replace(key, value)

    with open(file_path, 'w') as f:
        f.write(data)


def add_point_fields(fc, point_z=True):
    pz = ['POINT_Z'] if point_z else []
    for field in ['POINT_X', 'POINT_Y'] + pz:
        arcpy.AddField_management(fc, field, 'DOUBLE')
    arcpy.CalculateField_management(fc, 'POINT_X', '!Shape.Centroid.X!', 'PYTHON')
    arcpy.CalculateField_managem
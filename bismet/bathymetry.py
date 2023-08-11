# -*- coding: utf-8 -*-

# %%% Import -------------------------------------------------------------------

from abc import ABC, abstractmethod
import arcpy
import os
import shutil
import sys

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
import subprocess
import re
import scipy.stats as stats

# NOTE: All variable names and functionality are subject to change.

# TODO: Make a separate python file for fdaPDEAnalysis - this file is getting lengthy.

# TODO: Make unit tests or control sites with some project locations.

# TODO: Currently have partial PEP 8 formatting with some long lines
#  extended. Full formatting to be done once more sites have been tested.

# TODO: Add docstrings to functions once more sites have been tested.

# TODO: Add example usage with a main script file to illustrate how this script.

# %%% Main Body ----------------------------------------------------------------

def plot_linear_fit(data, case, **kws):
    df = data.copy()
    df = df.dropna()
    slope, intercept, r_value, p_value, std_err = stats.linregress(df[case], df['POINT_Z'])
    ax = plt.gca()
    r_squared = r_value ** 2
    ax.plot(df[case], (slope*df[case]+intercept), '-', color='#dc143c')
    if intercept >= 0:
        ax.annotate(f'y = {slope:0.2f}x + {intercept:0.1f}\nR² = {r_squared:0.2f}', (0, 1), (10, -10),
                    xycoords='axes fraction', textcoords='offset points', ha='left', va='top')
    elif intercept < 0:
        ax.annotate(f'y = {slope:0.2f}x - {(intercept*-1):0.1f}\nR² = {r_squared:0.2f}', (0, 1), (10, -10),
                    xycoords='axes fraction', textcoords='offset points', ha='left', va='top')
    return slope, intercept, r_squared

def tabulate_qaqc_stats(data, case, point_type=None):
    df = data.copy()
    df = df.dropna()
    slope, intercept, r_value, p_value, std_err = stats.linregress(df[case], df['POINT_Z'])
    r_squared = r_value ** 2
    diff = df[case] - df['POINT_Z']
    mean_error = diff.mean()
    abs_diff = diff.abs()
    mean_abs_error = abs_diff.mean()
    pct_error = diff.sum() / df['POINT_Z'].sum() * 100
    pct_abs_error = abs_diff.sum() / df['POINT_Z'].sum() * 100
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
            f.write(rf'Command history for {analysis.name}' + '\n' + '-'*72)

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
        self.register.append(gis_file_path)
        wkt = arcpy.Describe(gis_file_path).spatialReference.exportToString()
        if self.print_wkt:
            print(gis_file_path)
            print('\t' + wkt)
        return gis_file_path


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
    def __init__(self, stream_abbrev, start_rkm, end_rkm):
        self._stream_abbrev = stream_abbrev
        self._start_rkm = start_rkm
        self._end_rkm = end_rkm
        self._analyses = {}
        self.project = None

    @property
    def name(self):
        name = f'{self._stream_abbrev}_{self._start_rkm:0>4.1f}_{self._end_rkm:0>4.1f}'.replace('.', '_')
        return name

    @property
    def analyses(self):
        return tuple(self._analyses)

    def set_up_folder(self):
        if os.path.exists(rf'{self.project.project_folder_path}\{self.name}'):
            pass
        else:
            os.mkdir(rf'{self.project.project_folder_path}\{self.name}')
            os.mkdir(rf'{self.project.project_folder_path}\{self.name}\_inputs')

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


class BathyPoints(ABC):
    def __init__(self, precision=1):
        self.mesh = None
        self.precision = precision

    @property
    def _working_dir(self):
        path = rf'{self.mesh.analysis.working_directory.full_path}\{self.mesh.name}'
        return path

    @property
    def bathy_pts_fc_clip_path(self):
        pass

    @property
    def bathy_pts_csv_path(self):
        pass

    @property
    def bathy_name(self):
        pass


class SingleBeamBathyPoints(BathyPoints):
    def __init__(self, bathy_pts_fc_path, z_ft_name=None, z_m_name=None):
        super().__init__()
        self.z_ft_name = z_ft_name
        self.z_m_name = z_m_name
        self.bathy_pts_fc_path = bathy_pts_fc_path

    def clip_export_bathy_points(self):
        if not os.path.exists(self._working_dir):
            os.mkdir(self._working_dir)
        temp_gdb = fm.PathedGeoDataBase(rf'{self._working_dir}\clip_{self.bathy_name}_tmp.gdb')
        temp_gdb.set_env()
        bathy_clip = rf'{self._working_dir}\{self.mesh.name}_{self.bathy_name}_clip.shp'
        arcpy.Clip_analysis(self.bathy_pts_fc_path, self.mesh.boundary, bathy_clip)
        for field in ['POINT_X', 'POINT_Y', 'POINT_Z']:
            arcpy.AddField_management(bathy_clip, field, 'DOUBLE')
        if self.z_ft_name is not None:
            arcpy.CalculateField_management(bathy_clip, 'POINT_Z', f'!{self.z_ft_name}!*0.3048', 'PYTHON')
        if self.z_m_name is not None:
            arcpy.CalculateField_management(bathy_clip, 'POINT_Z', f'!{self.z_m_name}!', 'PYTHON')
        arcpy.CalculateField_management(bathy_clip, 'POINT_X', '!Shape.Centroid.X!', 'PYTHON')
        arcpy.CalculateField_management(bathy_clip, 'POINT_Y', '!Shape.Centroid.Y!', 'PYTHON')
        df_in = pd.DataFrame(arcpy.da.TableToNumPyArray(bathy_clip,
                                                        [field.name for field in arcpy.Describe(bathy_clip).fields
                                                         if (field.name in ['POINT_X', 'POINT_Y', 'POINT_Z']) | (
                                                                 field.type in ['OID'])]))
        df_in.to_csv(
            rf'{self._working_dir}\{self.mesh.name}_{self.bathy_name}_clip.csv',
            index=False)
        temp_gdb.reset_env()

    @property
    def bathy_name(self):
        path = pathlib.PureWindowsPath(self.bathy_pts_fc_path).stem
        return path

    @property
    def bathy_pts_fc_clip_path(self):
        bathy_name = pathlib.PureWindowsPath(self.bathy_pts_fc_path).stem
        bathy_clip = rf'{self._working_dir}\{self.mesh.name}_{bathy_name}_clip.shp'
        if os.path.exists(bathy_clip):
            return bathy_clip

    @property
    def bathy_pts_csv_path(self):
        bathy_name = pathlib.PureWindowsPath(self.bathy_pts_fc_path).stem
        clip_csv = rf'{self._working_dir}\{self.mesh.name}_{bathy_name}_clip.csv'
        if os.path.exists(clip_csv):
            return clip_csv


# TODO: Should script automatically project to the correct coordinate system?
class MultiBeamBathyPoints(BathyPoints):
    def __init__(self, bathy_raster_path):
        super().__init__()
        self.bathy_raster_path = bathy_raster_path

    def clip_export_bathy_points(self):
        if not os.path.exists(self._working_dir):
            os.mkdir(self._working_dir)
        temp_gdb = fm.PathedGeoDataBase(rf'{self._working_dir}\clip_{self.bathy_name}_tmp.gdb')
        temp_gdb.set_env()
        bathy_ras = rf'{self.bathy_name}_ras'
        bathy_resample = rf'{self.bathy_name}_resample'
        bathy_clip = rf'{self._working_dir}\{self.mesh.name}_{self.bathy_name}_clip.shp'
        arcpy.Clip_management(self.bathy_raster_path, '#', bathy_ras, self.mesh_boundary, clipping_geometry='ClippingGeometry')
        arcpy.Resample_management(bathy_ras, bathy_resample, '2 2', 'BILINEAR')
        arcpy.RasterToPoint_conversion(bathy_resample, bathy_clip)
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
            rf'{self._working_dir}\{self.mesh.name}_{self.bathy_name}_clip.csv',
            index=False)
        raster = arcpy.sa.Raster(self.bathy_raster_path)
        coverage_raster = arcpy.sa.Con(arcpy.sa.IsNull(raster), raster, 1)
        coverage_raster = arcpy.sa.Int(coverage_raster)
        coverage_raster.save(rf'{temp_gdb.full_path}\coverage_raster')

        #TODO: Should the buffered portion be moved to the shoreline aniso/diff creation argument? Nice to have
        # unbuffered polygon.
        arcpy.RasterToPolygon_conversion(rf'{temp_gdb.full_path}\coverage_raster', 'coverage_polygon_raw')
        arcpy.Dissolve_management('coverage_polygon_raw', 'coverage_polygon_dissolve')
        arcpy.EliminatePolygonPart_management('coverage_polygon_dissolve', 'coverage_polygon_elim')
        coverage_polygon = rf'{self._working_dir}\{self.mesh.name}_{self.bathy_name}_a1.shp'
        arcpy.Buffer_analysis('coverage_polygon_elim', coverage_polygon, '-5 Meters', 'FULL')
        temp_gdb.reset_env()

    @property
    def bathy_name(self):
        path = pathlib.PureWindowsPath(self.bathy_raster_path).stem
        return path

    @property
    def bathy_pts_fc_clip_path(self):
        bathy_name = pathlib.PureWindowsPath(self.bathy_raster_path).stem
        bathy_clip = rf'{self._working_dir}\{self.mesh.name}_{bathy_name}_clip.shp'
        if os.path.exists(bathy_clip):
            return bathy_clip

    @property
    def bathy_pts_csv_path(self):
        bathy_name = pathlib.PureWindowsPath(self.bathy_raster_path).stem
        clip_csv = rf'{self._working_dir}\{self.mesh.name}_{bathy_name}_clip.csv'
        if os.path.exists(clip_csv):
            return clip_csv

    @property
    def bathy_coverage_polygon_fc_path(self):
        bathy_name = pathlib.PureWindowsPath(self.bathy_raster_path).stem
        coverage_polygon = rf'{self._working_dir}\{self.mesh.name}_{bathy_name}_a1.shp'
        if os.path.exists(coverage_polygon):
            return coverage_polygon


class ManualBathyPoints(BathyPoints):
    def __init__(self, bathy_name, bathy_pts_fc_clip_input_path, bathy_pts_csv_path, bathy_coverage_polygon_fc_path=None):
        super().__init__()
        self._bathy_name = bathy_name
        self._bathy_pts_fc_clip_input_path = bathy_pts_fc_clip_input_path
        self._bathy_pts_csv_path = bathy_pts_csv_path
        self._bathy_coverage_polygon_fc_path = bathy_coverage_polygon_fc_path

    def clip_export_bathy_points(self):
        if not os.path.exists(self._working_dir):
            os.mkdir(self._working_dir)
        if not os.path.exists(rf'{self._working_dir}\{self.mesh.name}_{self.bathy_name}_clip.shp'):
            arcpy.Copy_management(self._bathy_pts_fc_clip_input_path,
                                          rf'{self._working_dir}\{self.mesh.name}_{self.bathy_name}_clip.shp')
        if not os.path.exists(rf'{self._working_dir}\{self.mesh.name}_{self.bathy_name}_clip.csv'):
            shutil.copy(self._bathy_pts_csv_path, rf'{self._working_dir}\{self.mesh.name}_{self.bathy_name}_clip.csv')
        if self._bathy_coverage_polygon_fc_path is not None:
            coverage_polygon = rf'{self._working_dir}\{self.mesh.name}_{self.bathy_name}_a1.shp'
            if not os.path.exists(coverage_polygon):
                arcpy.Buffer_analysis(self._bathy_coverage_polygon_fc_path, coverage_polygon, '-5 Meters', 'FULL')

    @property
    def bathy_name(self):
        return self._bathy_name

    @property
    def bathy_pts_fc_clip_path(self):
        path = rf'{self._working_dir}\{self.mesh.name}_{self.bathy_name}_clip.shp'
        if os.path.exists(path):
            return path

    @property
    def bathy_pts_csv_path(self):
        path = rf'{self._working_dir}\{self.mesh.name}_{self.bathy_name}_clip.csv'
        if os.path.exists(path):
            return path

    @property
    def bathy_coverage_polygon_fc_path(self):
        path = rf'{self._working_dir}\{self.mesh.name}_{self.bathy_name}_a1.shp'
        if os.path.exists(path):
            return path


class ShorelinePoints:
    def __init__(self, precision=0.3):
        self.mesh = None
        self.precision = precision

    @property
    def _working_dir(self):
        path = rf'{self.mesh.analysis.working_directory.full_path}\{self.mesh.name}'
        return path

    def clip_export_shoreline_points(self):
        temp_gdb = fm.PathedGeoDataBase(
            rf'{self._working_dir}\clip_shoreline_tmp.gdb')
        temp_gdb.set_env()
        arcpy.Clip_management(self.mesh.dem_clip_1m, '#', 'shoreline',
                              in_template_dataset=self.mesh.boundary,
                              clipping_geometry='ClippingGeometry')
        shoreline_clip = rf'{self._working_dir}\{self.mesh.name}_shoreline_pts.shp'
        arcpy.RasterToPoint_conversion('shoreline', shoreline_clip)
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
            rf'{self._working_dir}\{self.mesh.name}_shoreline_pts.csv',
            index=False)
        temp_gdb.reset_env()

    @property
    def shoreline_pts_csv_path(self):
        shoreline_csv = rf'{self._working_dir}\{self.mesh.name}_shoreline_pts.csv '
        if os.path.exists(shoreline_csv):
            return shoreline_csv

    @property
    def shoreline_pts_fc_path(self):
        shoreline_clip = rf'{self._working_dir}\{self.mesh.name}_shoreline_pts.shp'
        if os.path.exists(shoreline_clip):
            return shoreline_clip


class ManualShorelinePoints(ShorelinePoints):
    def __init__(self, shoreline_pts_csv_input_path, shoreline_pts_fc_input_path):
        super().__init__()
        self._shoreline_pts_csv_input_path = shoreline_pts_csv_input_path
        self._shoreline_pts_fc_input_path = shoreline_pts_fc_input_path

    def clip_export_shoreline_points(self):
        arcpy.CopyFeatures_management(self._shoreline_pts_fc_input_path, rf'{self._working_dir}\{self.mesh.name}_shoreline_pts.shp')
        shutil.copy(self._shoreline_pts_csv_input_path, rf'{self._working_dir}\{self.mesh.name}_shoreline_pts.csv ')


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
        arcpy.Clip_management(self.mesh.analysis.dem_2m_path, '#', 'target_area',
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


class ManualTargetPoints(TargetPoints):
    def __init__(self, target_pts_csv_input_path, target_pts_fc_input_path):
        super().__init__()
        self._target_pts_csv_input_path = target_pts_csv_input_path
        self._target_pts_fc_input_path = target_pts_fc_input_path

    def create_target_points(self):
        arcpy.CopyFeatures_management(self._target_pts_fc_input_path, rf'{self._working_dir}\{self.mesh.name}_target_pts.shp')
        shutil.copy(self._target_pts_csv_input_path, rf'{self._working_dir}\{self.mesh.name}_target_pts.csv')


class AbstractAnalysis(ABC):
    def __init__(self, dem_2m_path, breaklines_fc_path=None, levee_path=None):
        self.reach = None
        self._number = None
        self._new = True
        self._registry = []
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
    def __init__(self, dem_2m_path, breaklines_fc_path=None, levee_path=None):
        super().__init__(dem_2m_path, breaklines_fc_path, levee_path)
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

    def add_mesh(self, mesh_name, domain_polygon_fc_path, block_polygon_fc_path, open_boundary_fc_path, res=3.0, res_hint_poly_fc_path=None, h0=2.8, seed=1000):
        self.mesh = Mesh(mesh_name, self, domain_polygon_fc_path, block_polygon_fc_path, open_boundary_fc_path, res,
                         res_hint_poly_fc_path, h0, seed)

class Case:
    def __init__(self, name, aniso_constant, diff_constant, long_name=None, xs_plot=True, lw=1, zorder=None):
        self.aniso_constant = aniso_constant
        self.diff_constant = diff_constant
        self.mesh = None
        self.name = name
        self.long_name = long_name
        self.xs_plot = xs_plot
        self.result = None
        self.lw = lw
        self.zorder = zorder

    @property
    def _working_dir(self):
        path = rf'{self.mesh.analysis.working_directory.full_path}\{self.mesh.name}'
        return path

    def analysis_step(func):
        def inner(self, *args, **kwargs):
            identifier = rf'_{self.mesh.analysis.__class__.__name__}_{self.mesh.name}_{self.name}_{func.__name__}'
            argstring = ', '.join([str(arg) for arg in args if arg is not None])
            kwstring = ', '.join([f'{kw}={val}' for kw, val in kwargs.items()])
            command_str = rf'{self.mesh.analysis.__class__.__name__}[{self.mesh.name}][{self.name}].{func.__name__}({argstring}, {kwstring})'.replace(', )', ')')
            # Perform function substitution here. This function is written so analysis_step can be defined in
            # attached objects and add records to the analysis history.
            res = inner_record(self, self.mesh.analysis, func, identifier, command_str, rf'\{self.mesh.name}\{self.name}', *args, **kwargs)
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
            int_script_directory = self.mesh.analysis.reach.project.interpolation_script_directory.replace('\\','\\\\')
            mesh_parent_directory = rf'{self.mesh.analysis.working_directory.full_path}\{self.mesh.name}'.replace('\\','\\\\')
            mesh = f'{self.mesh.name}_out'
            bathy_pts = pathlib.PureWindowsPath(self.mesh.bathymetry_points.bathy_pts_csv_path).stem
            shoreline_pts = pathlib.PureWindowsPath(self.mesh.shoreline_points.shoreline_pts_csv_path).stem
            target_pts = pathlib.PureWindowsPath(self.mesh.target_points.target_pts_csv_path).stem
            angle_raster = pathlib.PureWindowsPath(self.mesh.angle_raster).stem
            aniso_raster = pathlib.PureWindowsPath(self.aniso_raster).stem
            diff_raster = pathlib.PureWindowsPath(self.diff_raster).stem
            substitute_text_tags(yaml_path,
                                 {'<INTERPOLATION_SCRIPT_DIRECTORY>': int_script_directory,
                                  '<MESH_PARENT_DIRECTORY>': mesh_parent_directory,
                                  '<MESH>': mesh,
                                  '<MESH_NAME>': self.mesh.name,
                                  '<TARGET_PTS>': target_pts,
                                  '<BATHY_PTS>': bathy_pts,
                                  '<SHORELINE_PTS>': shoreline_pts,
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
        with open(rf'{self.mesh.analysis.working_directory.full_path}\{self.mesh.name}_{self.name}_fdaPDE_output.txt', 'w') as f:
            outstring = '='*32 + ' OUTPUT ' + '=' * 32 + '\r\n' + f'{output.decode()}' + '\r\n' + '='*31 \
                        + ' MESSAGES ' + '='*31 + '\r\n' + f'{messages.decode()}'
            f.write(outstring)
        print(output.decode())

    @property
    def result_raster(self):
        out_path = rf'{self.mesh.analysis.working_directory.full_path}\{self.mesh.name}\{self.mesh.name}_sp_var_{self.name}.tif'
        if os.path.exists(out_path):
            return out_path

    def check_shoreline_fit(self):
        qaqc_dir = rf'{self._working_dir}\{self.name}_shoreline_qaqc'
        tmp_dir = fm.PathedWorkingDirectory(qaqc_dir)
        tmp_gdb = fm.PathedGeoDataBase(rf'{qaqc_dir}\shoreline_check.gdb')
        tmp_gdb.set_env()

        arcpy.Clip_management(self.mesh.analysis.dem_2m_path, '#', f'{self.name}_lidar_clip', in_template_dataset=self.mesh.domain_erase_polygon_fc_path,
                              nodata_value=-3.402823e38, clipping_geometry='ClippingGeometry')
        arcpy.Clip_management(self.result_raster, '#', f'{self.name}_result_clip', in_template_dataset=self.mesh.domain_erase_polygon_fc_path,
                              nodata_value=-3.402823e38, clipping_geometry='ClippingGeometry')
        check_raster = arcpy.sa.Minus(arcpy.sa.Raster(self.result_raster), arcpy.sa.Raster(f'{self.name}_lidar_clip'))
        check_raster.save(rf'{tmp_gdb.full_path}\{self.name}_check')

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
                                from_shoreline=False, shoreline_value=1, sdist_list=None):
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
                arcpy.CopyFeatures_management(self.mesh.bathymetry_points.bathy_coverage_polygon_fc_path,
                                              'bathy_polygon')
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
                arcpy.RasterToPolygon_conversion(rf'{temp_gdb.full_path}\sdist_var_raster', 'sdist_polygon', raster_field='Value')
                arcpy.AddField_management('sdist_polygon', var_name, 'DOUBLE')
                arcpy.CalculateField_management('sdist_polygon', var_name, "!gridcode!", 'PYTHON')
                add_polygons += [rf'{temp_gdb.full_path}\sdist_polygon']
            if from_shoreline:
                #dem = self.mesh.analysis.dem_2m_path
                dem = self.mesh.dem_clip_1m
                arcpy.Clip_management(dem, '#', f'{self.name}_clip',
                                      in_template_dataset=self.mesh.domain_erase_polygon_fc_path,
                                      nodata_value=-3.402823e38, clipping_geometry='ClippingGeometry')

                #raster = arcpy.sa.Raster(f'{self.name}_clip')
                raster = arcpy.sa.Raster(dem)
                coverage_raster = arcpy.sa.Con(arcpy.sa.IsNull(raster), raster, 1)
                coverage_raster = arcpy.sa.Int(coverage_raster)
                coverage_raster.save(rf'{temp_gdb.full_path}\coverage_raster')
                arcpy.RasterToPolygon_conversion(rf'{temp_gdb.full_path}\coverage_raster', 'coverage_polygon_raw')
                arcpy.Dissolve_management('coverage_polygon_raw', 'shoreline_polygon')
                arcpy.AddField_management('shoreline_polygon', var_name, 'DOUBLE')
                arcpy.CalculateField_management('shoreline_polygon', var_name, f'{shoreline_value}', 'PYTHON')
                add_polygons += [rf'{temp_gdb.full_path}\shoreline_polygon']
            polygons = add_polygons + in_polygons
            mosaic_list = [constant_raster]
            old_snap = arcpy.env.snapRaster
            arcpy.env.snapRaster = self.mesh.angle_raster
            cellsize = arcpy.Describe(self.mesh.angle_raster).meanCellHeight
            for polygon in polygons:
                polygon_name = arcpy.Describe(polygon).baseName
                if priority_field is None:
                    arcpy.PolygonToRaster_conversion(polygon, var_name, rf'{temp_gdb.full_path}\{polygon_name}_{var_name}_raster',
                                                     'CELL_CENTER', cellsize=cellsize)
                else:
                    arcpy.PolygonToRaster_conversion(polygon, var_name, rf'{temp_gdb.full_path}\{polygon_name}_{var_name}_raster',
                                                     'CELL_CENTER', priority_field=priority_field, cellsize=cellsize)
                mosaic_list.append(rf'{temp_gdb.full_path}\{polygon_name}_{var_name}_raster')
            arcpy.MosaicToNewRaster_management(';'.join(mosaic_list), self._working_dir,
                                               f'{self.mesh.name}_{var_name}_{self.name}.tif', pixel_type='32_BIT_FLOAT',
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
                 res_hint_poly_fc_path=None, h0=2.8, seed=1000, res_sdist_list=None):
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
        self.shoreline_points = None
        self.target_points = None
        self.bathymetry_points = None
        self.seed = seed
        self._initialize()
        self.res_sdist_list = res_sdist_list

    def __getitem__(self, name):
        if type(name) == str:
            return self.case[name]
        elif type(name) in (int, slice):
            return tuple(self.case.values())[name]

    @property
    def _working_dir(self):
        path = rf'{self.analysis.working_directory.full_path}\{self.name}'
        return path

    @property
    def res_name(self):
        name = pathlib.PureWindowsPath(self._res_hint_poly_input_fc_path).stem
        return name

    @property
    def res_hint_fc_path(self):
        path = rf'{self._working_dir}\res_hint\{self.res_name}.shp'
        if os.path.exists(path):
            return path

    @property
    def res(self):
        if os.path.isdir(str(self._res)):
            path = rf'{self._working_dir}\res_hint\{self.res_name}_constant.tif'
            if os.path.exists(path):
                return path
        else:
            return self._res

    # This should work with prep_mesh_sdist regardless of whether folder exists.
    def _set_up_folder(self):
        if not os.path.exists(rf'{self._working_dir}'):
            os.mkdir(rf'{self._working_dir}')
            os.mkdir(rf'{self.input_path}')

    def _set_up_input(self, folder_name, fc_path):
        name = pathlib.PureWindowsPath(fc_path).stem
        ext = pathlib.PureWindowsPath(fc_path).suffix
        new_path = rf'{self._working_dir}\{folder_name}\{name}{ext}'
        if not os.path.exists(new_path):
            self.analysis.gpr.record(arcpy.CopyFeatures_management, fc_path, new_path)
        return name, new_path

    def analysis_step(func):
        def inner(self, *args, **kwargs):
            identifier = rf'_{self.analysis.__class__.__name__}_{self.name}_{func.__name__}'
            argstring = ', '.join([arg for arg in args if arg is not None])
            kwstring = ', '.join([f'{kw}={val}' for kw, val in kwargs.items()])
            command_str = rf'{self.analysis.__class__.__name__}[{self.name}].{func.__name__}({argstring}, {kwstring})'.replace(', )', ')')
            # Perform function substitution here. This function is written so analysis_step can be defined in
            # attached objects and add records to the analysis history.
            res = inner_record(self, self.analysis, func, identifier, command_str, rf'\{self.name}', *args, **kwargs)
            return res
        return inner

    def add_case(self, case):
        self.case[case.name] = case
        case.mesh = self

    @property
    def input_path(self):
        input_path = rf'{self._working_dir}\_inputs'
        return input_path

    def _initialize(self):
        if not os.path.exists(self.input_path):
            os.mkdir(self.input_path)
        self._set_up_input('_inputs', self._domain_polygon_input_fc_path)
        self._set_up_input('_inputs', self._block_polygon_input_fc_path)
        self._set_up_input('_inputs', self._open_boundary_input_fc_path)

    @analysis_step
    def clip_dem(self):
        dem_mask = rf'{self.input_path}\dem_mask.tif'
        self.analysis.gpr.record(arcpy.AddField_management, rf'{self.domain_polygon_fc_path}', 'Elev',
                        'DOUBLE')
        self.analysis.gpr.record(arcpy.CalculateField_management, rf'{self.domain_polygon_fc_path}', 'Elev',
                        '1000', 'PYTHON')

        cellsize = arcpy.Describe(self.analysis.dem_2m_path).meanCellHeight
        old_snap = arcpy.env.snapRaster
        arcpy.env.snapRaster = self.analysis.dem_2m_path
        self.analysis.gpr.record(arcpy.PolygonToRaster_conversion, self.domain_polygon_fc_path, 'Elev', dem_mask,
                        cellsize=cellsize)
        dem_clip = rf'{self.input_path}\dem_domain_clip.tif'

        self.analysis.gpr.record(arcpy.Clip_management, self.analysis.dem_2m_path, '#',
                        dem_clip, in_template_dataset=self.domain_polygon_fc_path, clipping_geometry='ClippingGeometry')

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

        dem_clip_1m = rf'{self.input_path}\dem_domain_clip_no_water_1m.tif'
        self.analysis.gpr.record(arcpy.Clip_management, dem_clip_2, '#',
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
        bdy = rf'{self._working_dir}\boundary\{self.name}_out_polygon_boundary.shp'
        if os.path.exists(bdy):
            return bdy

    @property
    def polygons(self):
        polygons = rf'{self._working_dir}\{self.name}_out_polygon.shp'
        if os.path.exists(polygons):
            return polygons

    @property
    def nodes(self):
        point = rf'{self._working_dir}\{self.name}_out_point.shp'
        if os.path.exists(point):
            return point

    @analysis_step
    def create_res_hint(self, base_value, field_name='res_hint'):
        # Field name assumed to exist when this function is run.
        create_res_hint_path = rf'{self._working_dir}\res_hint'
        if not os.path.exists(create_res_hint_path):
            os.mkdir(create_res_hint_path)
        name, path = self._set_up_input('res_hint', self._res_hint_poly_input_fc_path)
        res_hint_constant = rf'{self._working_dir}\res_hint\{self.res_name}_constant.tif'
        raster = arcpy.sa.CreateConstantRaster(base_value, 'Float', 1, self.dem_clip_1m)
        raster.save(res_hint_constant)
        old_snap = arcpy.env.snapRaster
        arcpy.env.snapRaster = res_hint_constant
        res_hint_poly_raster_path = rf'{self._working_dir}\res_hint\{self.res_name}_poly.tif'
        arcpy.PolygonToRaster_conversion(self.res_hint_fc_path, field_name, res_hint_poly_raster_path, cellsize=1)
        rasters = ';'.join([res_hint_poly_raster_path, res_hint_constant])
        arcpy.MosaicToNewRaster_management(rasters, rf'{self._working_dir}\res_hint',
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
            create_res_hint_path = rf'{self._working_dir}\res_hint'
            if not os.path.exists(create_res_hint_path):
                os.mkdir(create_res_hint_path)
            for i, sdist_pair in enumerate(from_sdist_list):
                sdist, value = sdist_pair
                sdist *= -1
                arcpy.CopyRaster_management(rf'{self._working_dir}\sdist_{self.name}.tif', rf'{self._working_dir}\res_hint\prev_sdist.tif')
                sdist_raster = arcpy.sa.Raster(rf'{self._working_dir}\res_hint\prev_sdist.tif')
                if i == 0:
                    tmp_raster = arcpy.sa.SetNull((sdist_raster < sdist) | (sdist_raster >= 0), value)
                else:
                    tmp_raster = arcpy.sa.Con((sdist_raster >= sdist) & (sdist_raster <= 0), value, tmp_raster)
            tmp_raster = arcpy.sa.Con(arcpy.sa.IsNull(tmp_raster), self._res, tmp_raster)
            # if self.res_hint_fc_path is not None:
            #     orig_res_hint = arcpy.sa.Raster(rf'{create_res_hint_path}\{self.res_name}.tif')
            #     tmp_raster = arcpy.sa.Con(arcpy.sa.IsNull(orig_res_hint), tmp_raster, orig_res_hint)
            tmp_raster.save(rf'{self._working_dir}\res_hint\res_from_sdist_tmp.tif')
            arcpy.env.extent = old_extent
            self._prep_mesh_sdist_inner()

    def _prep_mesh_sdist_inner(self):
        infile = rf'{self.name}\_inputs\dem_domain_clip_no_water_1m.tif'
        from_sdist = rf'{self._working_dir}\res_hint\res_from_sdist_tmp.tif'
        if os.path.isdir(str(self.res)) & (not(os.path.exists(from_sdist))):
            res = rf'{self.name}\res_hint\{self.res_name}.tif'
        elif os.path.exists(from_sdist):
            res = rf'{self._working_dir}\res_hint\res_from_sdist_tmp.tif'
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
        print('In the distmesh conda environment, copy and run the following commands')
        print(rf'cd {self._working_dir}')
        print(rf'sdist_to_direction --input sdist_{self.name}.tif --output angle_{self.name}.tif')
        input('Once this process is done, press ENTER to continue.')
        if not (os.path.exists(rf'{self._working_dir}\angle_{self.name}.tif')):
            raise RuntimeError('hgrid_mod.gr3 not found')

    @property
    def angle_raster(self):
        angle = rf'{self._working_dir}\angle_{self.name}.tif'
        if os.path.exists(angle):
            return angle

    @analysis_step
    def remove_skewed_cells(self):
        print('In the distmesh conda environment, copy and run the following commands')
        print(rf'cd {self._working_dir}')
        print(rf'remove_skewed_cells --input hgrid.gr3 --output hgrid_mod.gr3')
        input('Once this process is done, press ENTER to continue.')
        if not (os.path.exists(rf'{self._working_dir}\hgrid_mod.gr3')):
            raise RuntimeError('hgrid_mod.gr3 not found')

    @analysis_step
    def convert_linestrings(self):
        print('In the distmesh conda environment, copy and run the following commands')
        print(rf'cd {self._working_dir}')
        print(rf'convert_linestrings --input _inputs\{self.open_boundary_name}.shp --output '
              rf'{self.open_boundary_name}.yaml')
        input('Once this process is done, press ENTER to continue.')
        if not (os.path.exists(rf'{self._working_dir}\{self.open_boundary_name}.yaml')):
            raise RuntimeError(rf'{self.open_boundary_name}.yaml not found')

    @analysis_step
    def copy_schism_yaml(self):
        shutil.copy(r"resources\main.yaml", rf'{self._working_dir}\main.yaml')
        bdy_name = pathlib.PureWindowsPath(self.open_boundary_fc_path).stem
        substitute_text_tags(rf'{self._working_dir}\main.yaml',
                             {'<BDY_NAME>': bdy_name,
                              '<NAME>': self.name})
        shutil.copy(r"resources\dem_list.yaml", rf'{self._working_dir}\dem_list.yaml')
        dem = self.analysis.dem_2m_path
        dem_name = 'dem'
        dem_clip_1m = rf'{dem_name}_domain_clip_no_water_1m'
        substitute_text_tags(rf'{self._working_dir}\dem_list.yaml', {'<DEM_NAME>': dem_clip_1m})

    @analysis_step
    def prepare_schism(self):
        print('In the distmesh conda environment, copy and run the following commands')
        print(rf'cd {self._working_dir}')
        print(rf'prepare_schism main.yaml')
        input('Once this process is done, press ENTER to continue.')
        if not (os.path.exists(rf'{self._working_dir}\{self.name}_out.gr3')):
            raise RuntimeError(f'{self.name}_out.gr3 not found')

    @analysis_step
    def convert_mesh(self):
        print('In the distmesh conda environment, copy and run the following commands')
        print(rf'cd {self._working_dir}')
        print(
            rf'convert_mesh --input {self.name}_out.gr3 --output {self.name}_out.shp --proj4 "+proj=utm +zone=10 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"')
        input('Once this process is done, press ENTER to continue.')
        if not (
        os.path.exists(rf'{self._working_dir}\{self.name}_out_polygon.shp')):
            raise RuntimeError(f'{self.name}_out_polygon.shp not found')

    @analysis_step
    def create_check_boundaries(self):
        create_check_boundary_path = rf'{self._working_dir}\boundary'
        if not os.path.exists(create_check_boundary_path):
            os.mkdir(create_check_boundary_path)
        self.analysis.gpr.record(arcpy.Dissolve_management,
                        rf'{self._working_dir}\{self.name}_out_polygon.shp',
                        rf'{self._working_dir}\boundary\{self.name}_out_polygon_boundary.shp')
        mesh_boundary = f'lyr_fdaPDE_{self.analysis.reach.name}_{self.analysis.name}_{self.name}_boundary'
        arcpy.MakeFeatureLayer_management(rf'{self._working_dir}\boundary\{self.name}_out_polygon_boundary.shp',
                                          mesh_boundary)
        mesh_nodes = f'lyr_fdaPDE_{self.analysis.reach.name}_{self.analysis.name}_{self.name}_nodes'
        arcpy.MakeFeatureLayer_management(rf'{self._working_dir}\{self.name}_out_point.shp', mesh_nodes)
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
                arcpy.CopyFeatures_management(mesh_nodes, rf'{self._working_dir}\boundary\{self.name}_qaqc.shp')
                print(f'{n_qaqc} points touch the mesh boundary with values >= -2. Review and decide whether a different'
                      f' resolution mesh is needed.')
            else:
                print('No points touching the mesh boundary with values >= -2 were detected.')
        else:
            print('No points touching the mesh boundary with values >= -2 were detected.')

    def add_bathy(self, bathymetry_points):
        # This must be copied - otherwise analysis will get pointed to something different when this object is used
        # for a different analysis.
        self.bathymetry_points = copy.deepcopy(bathymetry_points)
        self.bathymetry_points.mesh = self

    @analysis_step
    def clip_export_bathymetry(self):
        self.bathymetry_points.clip_export_bathy_points()

    def add_shoreline(self, shoreline_obj=None):
        self.shoreline_points = ShorelinePoints() if shoreline_obj is None else shoreline_obj
        self.shoreline_points.mesh = self

    @analysis_step
    def clip_export_shoreline(self):
        self.shoreline_points.clip_export_shoreline_points()

    def add_target(self, target_obj=None):
        self.target_points = TargetPoints() if target_obj is None else target_obj
        self.target_points.mesh = self

    @analysis_step
    def create_target_points(self):
        self.target_points.create_target_points()

    def prep_plot_fit(self, plot=True, figsize=(7.5, 7.5), folder_name='fit_qaqc', precision=False, by_case=False):
        qaqc_dir = rf'{self._working_dir}\{folder_name}'
        tmp_dir = fm.PathedWorkingDirectory(qaqc_dir)
        tmp_gdb = fm.PathedGeoDataBase(rf'{qaqc_dir}\fit_plot.gdb')
        tmp_gdb.set_env()
        result_rasters = [arcpy.sa.Raster(case.result_raster) for case in self.case.values()
                          if (case.result_raster is not None) & (case.xs_plot is True)]
        result_cases = [case.name for case in self.case.values()
                        if (case.result_raster is not None) & (case.xs_plot is True)]
        results = [[raster, name] for raster, name in zip(result_rasters, result_cases)]
        arcpy.CopyFeatures_management(self.bathymetry_points.bathy_pts_fc_clip_path, rf'{self.name}_bathy_pts')
        arcpy.CopyFeatures_management(self.shoreline_points.shoreline_pts_fc_path, rf'{self.name}_shoreline_pts')

        arcpy.sa.ExtractMultiValuesToPoints(rf'{self.name}_bathy_pts', results, 'BILINEAR')
        arcpy.sa.ExtractMultiValuesToPoints(rf'{self.name}_shoreline_pts', results, 'BILINEAR')
        fields = ['POINT_Z'] + result_cases
        df_bathy = pd.DataFrame(arcpy.da.TableToNumPyArray(rf'{self.name}_bathy_pts',
                                                        [field.name for field in arcpy.Describe(rf'{self.name}_bathy_pts').fields
                                                         if field.name in fields]))
        df_shoreline = pd.DataFrame(arcpy.da.TableToNumPyArray(rf'{self.name}_shoreline_pts',
                                                        [field.name for field in arcpy.Describe(rf'{self.name}_shoreline_pts').fields
                                                         if field.name in fields]))
        df_bathy.to_csv(rf'{tmp_dir.full_path}\{self.name}_bathy_pts.csv', index=False)
        df_shoreline.to_csv(rf'{tmp_dir.full_path}\{self.name}_shoreline_pts.csv', index=False)


        if plot:
            if by_case==False:
                self.plot_fit(figsize, folder_name, precision)
            else:
                self.plot_fit_by_case(figsize, folder_name, precision)

    def plot_fit_by_case(self, figsize=(7.5, 7.5), folder_name='fit_qaqc', precision=True, linear_fit=True):
        qaqc_dir = rf'{self._working_dir}\{folder_name}'
        df_bathy = pd.read_csv(rf'{qaqc_dir}\{self.name}_bathy_pts.csv')
        df_shoreline = pd.read_csv(rf'{qaqc_dir}\{self.name}_shoreline_pts.csv')
        df_all = pd.concat([df_bathy, df_shoreline]).reset_index()
        result_cases = [case for case in self.case.values()
                        if (case.result_raster is not None) & (case.xs_plot is True)]
        result_long_names = [case.long_name for case in self.case.values()
                             if (case.result_raster is not None) & (case.xs_plot is True)]
        result_df_list = []
        result_all_df_list = []
        for df, point_type in zip([df_bathy, df_shoreline], ['Bathymetry', 'Shoreline']):
            for case, long_name in zip(result_cases, result_long_names):
                fig, ax = plt.subplots(figsize=figsize)
                df_min = df.min().min()
                df_max = df.max().max()
                name = case.name
                ax.plot(df[name], df['POINT_Z'], '.', alpha=0.15)
                ax.set_xlabel(f'Interpolated {point_type} Elevation (m NAVD88)')
                ax.set_ylabel(f'Observed {point_type} Elevation (m NAVD88)')
                ax.plot([df_min, df_max], [df_min, df_max], 'k-', lw=1)
                if precision:
                    if point_type=='Shoreline':
                        delta = self.shoreline_points.precision
                    elif point_type=='Bathymetry':
                        delta = self.bathymetry_points.precision
                    ax.plot([df_min, df_max], [df_min + delta, df_max + delta], 'k--', lw=0.5)
                    ax.plot([df_min, df_max], [df_min - delta, df_max - delta], 'k--', lw=0.5)
                if linear_fit:
                    slope, intercept, r_squared = plot_linear_fit(df, name)
                result_df, diff = tabulate_qaqc_stats(df, name, point_type.lower())
                result_df_list.append(result_df)
                ax.set_xlim(df_min, df_max)
                ax.set_ylim(df_min, df_max)
                plt.grid(which='minor', color='#dddddd', lw=0.8)
                ax.yaxis.set_minor_locator(tkr.AutoMinorLocator())
                ax.xaxis.set_minor_locator(tkr.AutoMinorLocator())
                fig.tight_layout()
                fig.savefig(rf'{qaqc_dir}\{self.name}_{name}_{point_type.replace(" ","_").lower()}_fit.png',
                            dpi=300)
                fig2, ax2 = plt.subplots(figsize=(figsize[0], figsize[1]))
                abs_diff = diff.abs()
                abs_diff_max = np.ceil(abs_diff.max()*10)/10
                bins = np.arange(-abs_diff_max, abs_diff_max, 0.1)
                _, _, patches = ax2.hist(diff, bins=bins, histtype='step', alpha=0.6, edgecolor='C0')
                _, _, patches = ax2.hist(diff, bins=bins, alpha=0.4, facecolor='C0')
                total = diff.shape[0]
                n_less = diff[diff < -delta].shape[0]
                n_center = diff[(diff >= -delta) & (diff <= delta)].shape[0]
                n_greater = diff[diff > delta].shape[0]
                percent_less = n_less / total * 100
                percent_center = n_center / total * 100
                percent_greater = n_greater / total * 100
                ymax = ax2.get_ylim()[1]
                xmin = ax2.get_xlim()[0]
                xmax = ax2.get_xlim()[1]
                offset = -10
                ax2.annotate(f'{percent_less:0.1f}%', ((-delta - xmin)/2 + xmin, ymax),
                             (0, offset), textcoords='offset points', ha='center', va='top')
                ax2.annotate(f'{percent_center:0.1f}%', (0, ymax), (0, offset), textcoords='offset points', ha='center', va='top')
                ax2.annotate(f'{percent_greater:0.1f}%', (xmax - (xmax - delta)/2, ymax), (0, offset), textcoords='offset points',
                         ha='center', va='top')
                if precision:
                    ax2.axvline(-delta, ls='--', color='black', lw=0.5)
                    ax2.axvline(delta, ls='--', color='black', lw=0.5)
                plt.grid(which='minor', color='#dddddd', lw=0.8)
                ax2.yaxis.set_minor_locator(tkr.AutoMinorLocator())
                ax2.xaxis.set_minor_locator(tkr.AutoMinorLocator())
                ax2.yaxis.set_major_formatter(tkr.StrMethodFormatter('{x:0,.0f}'))
                ax2.set_xlabel(f'Elevation Diff (m NAVD88)')
                ax2.set_ylabel(f'Count')
                fig2.tight_layout()
                fig2.savefig(rf'{qaqc_dir}\{self.name}_{name}_{point_type.replace(" ","_").lower()}_hist_by_location.png',
                             dpi=300)

        for case in result_cases:
            name = case.name
            result_all_df, all_diff = tabulate_qaqc_stats(df_all, name)
            result_all_df_list.append(result_all_df)
        pd.concat(result_all_df_list, axis=1).to_csv(rf'{qaqc_dir}\{self.name}_all_points_stats.csv')
        pd.concat(result_df_list, axis=1).to_csv(rf'{qaqc_dir}\{self.name}_points_stats_by_location.csv')

    def plot_fit(self, figsize=(7.5, 7.5), folder_name='fit_qaqc', precision=False):
        qaqc_dir = rf'{self._working_dir}\{folder_name}'
        df_bathy = pd.read_csv(rf'{qaqc_dir}\{self.name}_bathy_pts.csv')
        df_shoreline = pd.read_csv(rf'{qaqc_dir}\{self.name}_shoreline_pts.csv')
        result_cases = [case for case in self.case.values()
                        if (case.result_raster is not None) & (case.xs_plot is True)]
        result_long_names = [case.long_name for case in self.case.values()
                             if (case.result_raster is not None) & (case.xs_plot is True)]
        for df, point_type in zip([df_bathy, df_shoreline], ['Bathymetry', 'Shoreline']):
            fig, ax = plt.subplots(figsize=figsize)
            df_min = df.min().min()
            df_max = df.max().max()
            for i, (case, long_name) in enumerate(zip(result_cases, result_long_names)):
                name = case.name
                if i <= 9:
                    mt = '.'
                elif i >= 10 & i <= 19:
                    mt = 'x'
                elif i >= 20:
                    mt = 't'
                else:
                    mt = '.'
                ax.plot(df[name], df['POINT_Z'], mt, alpha=0.15, label=long_name)
            ax.set_xlabel(f'Interpolated {point_type} Elevation (m NAVD88)')
            ax.set_ylabel(f'Observed {point_type} Elevation (m NAVD88)')
            ax.plot([df_min, df_max], [df_min, df_max], 'k-', lw=1)
            if precision:
                if point_type=='Shoreline':
                    delta = self.shoreline_points.precision
                elif point_type=='Bathymetry':
                    delta = self.bathymetry_points.precision
                ax.plot([df_min, df_max], [df_min + delta, df_max + delta], 'k--', lw=0.5)
                ax.plot([df_min, df_max], [df_min - delta, df_max - delta], 'k--', lw=0.5)
            ax.set_xlim(df_min, df_max)
            ax.set_ylim(df_min, df_max)
            ax.legend()
            plt.grid(which='minor', color='#dddddd', lw=0.8)
            ax.yaxis.set_minor_locator(tkr.AutoMinorLocator())
            ax.xaxis.set_minor_locator(tkr.AutoMinorLocator())
            fig.tight_layout()
            fig.savefig(rf'{qaqc_dir}\{self.name}_{point_type.replace(" ","_").lower()}_fit.png', dpi=300)

    def plot_xs_lines(self, xs_line_fc, radius='2 Meter', plot=True, figsize=(7.5, 7.5),
                      folder_name='xs_qaqc', error_bars=None):
        radius_value = re.findall(r'(.*?)\s', radius)[0]
        radius_unit = re.findall(r'\s(.*)', radius)[0]
        print(f'{radius_value} {radius_unit}')
        qaqc_dir = rf'{self._working_dir}\{folder_name}'
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
        arcpy.GeneratePointsAlongLines_management(rf'{self.name}_xs_line', rf'{self.name}_xs_points', 'DISTANCE', '1 Meter',
                                                  Include_End_Points='END_POINTS')
        result_rasters = [arcpy.sa.Raster(case.result_raster) for case in self.case.values()
                          if (case.result_raster is not None) & (case.xs_plot is True)]
        result_cases = [case.name for case in self.case.values()
                        if (case.result_raster is not None) & (case.xs_plot is True)]

        results = [[raster, name] for raster, name in zip(result_rasters, result_cases)]
        arcpy.sa.ExtractMultiValuesToPoints(rf'{self.name}_xs_points', results, 'BILINEAR')
        arcpy.CreateRoutes_lr(rf'{self.name}_xs_line', 'Name', rf'{self.name}_xs_line_route', 'TWO_FIELDS',
                              'Start', 'End')
        arcpy.Buffer_analysis(xs_line_fc, 'xs_buffer', f'{float(radius_value)*2} {radius_unit}', dissolve_option='ALL')
        arcpy.Clip_analysis(self.shoreline_points.shoreline_pts_fc_path, 'xs_buffer', 'shoreline_clip')
        arcpy.Clip_analysis(self.bathymetry_points.bathy_pts_fc_clip_path, 'xs_buffer', 'bathy_clip')
        arcpy.LocateFeaturesAlongRoutes_lr(rf'{self.name}_xs_points', rf'{self.name}_xs_line_route', 'Name',
                                           radius, rf'{qaqc_dir}\{self.name}_xs_points.csv', 'Name POINT Station')
        arcpy.LocateFeaturesAlongRoutes_lr('shoreline_clip',
                                           rf'{self.name}_xs_line_route', 'Name',
                                           radius, rf'{qaqc_dir}\{self.name}_shoreline_points.csv',
                                           'Name POINT Station')
        arcpy.LocateFeaturesAlongRoutes_lr('bathy_clip',
                                           rf'{self.name}_xs_line_route', 'Name',
                                           radius, rf'{qaqc_dir}\{self.name}_bathy_points.csv', 'Name POINT Station')

        if plot:
            self.plot_results(figsize, folder_name, error_bars)

    def plot_results(self, figsize=(13.333*0.6, 7.5*1/2),
                      folder_name='xs_qaqc', error_bars=None):
        qaqc_dir = rf'{self._working_dir}\{folder_name}'
        result_cases = [case for case in self.case.values()
                        if (case.result_raster is not None) & (case.xs_plot is True)]
        result_long_names = [case.long_name for case in self.case.values()
                             if (case.result_raster is not None) & (case.xs_plot is True)]

        bathy = pd.read_csv(rf'{qaqc_dir}\{self.name}_bathy_points.csv')
        shoreline = pd.read_csv(rf'{qaqc_dir}\{self.name}_shoreline_points.csv')
        results = pd.read_csv(rf'{qaqc_dir}\{self.name}_xs_points.csv')

        xs_names = results['Name'].unique().tolist()

        for xs_name in xs_names:
            fig, ax = plt.subplots(figsize=figsize)
            bathy_query = bathy.query('Name == @xs_name')
            shoreline_query = shoreline.query('Name == @xs_name')
            results_query = results.query('Name == @xs_name')
            for i, (case, long_name) in enumerate(zip(result_cases, result_long_names)):
                name = case.name
                if i <= 9:
                    lt = '-'
                elif i >= 10 & i <= 19:
                    lt = '--'
                elif i >= 20:
                    lt = '-.'
                else:
                    lt = '-'
                if long_name is None:
                    ax.plot(results_query['Station'], results_query[name], lt, lw=case.lw, label=name, zorder=case.zorder)
                else:
                    ax.plot(results_query['Station'], results_query[name], lt, lw=case.lw, label=long_name, zorder=case.zorder)
            if error_bars is not None:
                ax.errorbar(bathy_query['Station'], bathy_query['POINT_Z'], error_bars, fmt=' ', color='#888888',
                            capsize=0, capthick=0.5, lw=0.8, zorder=2.7)
                ax.errorbar(shoreline_query['Station'], shoreline_query['POINT_Z'], error_bars, fmt=' ',
                            color='#888888',
                            capsize=0, capthick=0.5, lw=0.8, zorder=2.8)
            ax.plot(bathy_query['Station'], bathy_query['POINT_Z'], 'kx', ms=4, zorder=2.5)
            ax.plot(shoreline_query['Station'], shoreline_query['POINT_Z'], 'k.', 2.5, zorder=2.6)
            ax.set_xlabel('Station (m)')
            ax.set_ylabel('Elevation (m NAVD88)')
            ax.legend()
            ax.yaxis.set_minor_locator(tkr.AutoMinorLocator())
            ax.xaxis.set_minor_locator(tkr.AutoMinorLocator())
            plt.grid(which='minor', color='#dddddd', lw=0.8)
            fig.tight_layout()
            fig.savefig(rf'{qaqc_dir}\{xs_name}.png', dpi=300)
            del fig


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
    arcpy.CalculateField_management(fc, 'POINT_Y', '!Shape.Centroid.Y!', 'PYTHON')
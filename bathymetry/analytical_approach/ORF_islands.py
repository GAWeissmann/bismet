# -*- coding: utf-8 -*-

# %%% Import -------------------------------------------------------------------

import sys
import os

sys.path.append(r'..\common')
import dwr_bathymetry as bathy
import time
import arcpy
import matplotlib.pyplot as plt
import itertools
import multiprocessing as mp

# %%% Main Body ----------------------------------------------------------------

# Function for supporting parallel processing with fdaPDE.
def parallel_run_fda_pde(case):
    case.run_fda_pde()
    return

def main():
    reach_name = 'ORF'

    # Object for checking projection of GIS files.
    pc = bathy.GISChecker()
    
    # Alias for pc.check_in. Wrapping this around a feature class or .tif file will print out
    # the projection, then store the string path to the variable.
    gis = pc.check_in

    # Get current working directory.
    cwd = os.getcwd()

    # Path to fdaPDE run script directory.
    interpolation_script_directory = r'C:\Users\GWeissmann\Git\fdaPDE_bathy\code'

    # Path to R interpreter for running fdaPDE.
    path_to_rscript = r'C:\Users\GWeissmann\AppData\Local\Programs\R\R-4.2.2\bin\Rscript.exe'

    # Path to fdaPDE_run_cmd.R.
    path_to_run_r = r'C:\Users\GWeissmann\Git\fdaPDE_bathy\code\fdaPDE_run_cmd.R'

    # 10m dem. 10m covers the entire project.
    dem_10m = gis(rf'{cwd}\project_inputs\bay_delta_dem_v4.2.gdb\dem_delta_10m_20201207')

    # Create a Project object for storing project settings and reach information.
    proj = bathy.Project(cwd, dem_10m, interpolation_script_directory, path_to_rscript, path_to_run_r)

    # 2m dem. 2m dem is defined at the analysis level.
    dem_2m = gis(rf'{cwd}\project_inputs\bay_delta_dem_v4.2.gdb\dem_ccfb_south_delta_san_joaquin_rvr_2m_20200625')

    # DEM to use for shoreline points and image clipping.
    source_dem = gis(rf'{cwd}\project_inputs\SacDeltaLidar_2017_clip_m.tif')

    # Once the folder is set up, add the reach-scale inputs in the newly created folder under the _inputs folder.

    # %%% OLD RIVER FIVE POINTS --------------------------------------------------------------------

    # Add reach. Arguments are reach name, then optionally start and end river kilometers to represent linear reaches.
    # Leave start_rkm and end_rkm out to represent confluences.
    proj.add_reach(reach_name)

    # Reaches can be accessed through indices by name or by slice. Entering -1 will access the last added reach.
    # This line sets up a folder for the reach. If this line is run again, the script will keep the folder in the same
    # place. Deleting the folder requires manually deleting the folder. This is done by design to prevent accidental
    # data loss.
    proj[-1].set_up_folder()

    # Target polygon.
    target_poly_pc_or = gis(rf"{cwd}\{reach_name}\_inputs\ORF_Target_Polygon_islands.shp")

    # Block polygon.
    block_poly_pc_or = gis(rf"{cwd}\{reach_name}\_inputs\ORF_block_polygon.shp")

    # Domain polygon.
    domain_poly_pc_or = gis(rf"{cwd}\{reach_name}\_inputs\ORF_domain_polygon.shp")

    # Open boundary lines.
    open_boundary_pc_or = gis(rf"{cwd}\{reach_name}\_inputs\ORF_open_boundaries.shp")

    # Add an analysis to the reach. Currently multiple analysis are supported per reach but only one analysis
    # and analysis type (bathy.FDAPDEAnalysis) is supported.
    # Currently supported arguments for bathy.FDAPDEAnalysis are shown below.
    # When this line is run, the script will add a numbered folder. If this line is run after the folder has been
    # created, it will not create a new folder.
    proj[-1].add_analysis(bathy.FDAPDEAnalysis(dem_2m, source_dem_path=source_dem,  breaklines_fc_path=target_poly_pc_or))

    # Reach objects also support indexing to access Analysis objects the same way as Reaches. The line below
    # accesses the most recent Analysis in the most recent Reach.
    analysis = proj[-1][-1]

    # Add a mesh to the FDAPDEAnalysis. Give it a descriptive name. Current convention is r[res]h_[h0].
    # A seed value of 1000 is currently the standard for consistent mesh generation. The value used is arbitrary.
    # The FDAPDEAnalysis object will then create a folder for the mesh.
    # a parameter for res_hint_poly_fc_path exists but this functionality and mesh.create_res_hint() has not been tested.
    analysis.add_mesh('r3h2_8_islands', domain_poly_pc_or, block_poly_pc_or, open_boundary_pc_or,
                      res=3.0, h0=2.8, seed=2000)

    # Access the newly created mesh.
    mesh = analysis['r3h2_8_islands']

    # (*) This line runs the process of clipping the DEM from the domain polygon and blanking out the water area from
    # the target polygon. When this process successfully finishes running, the script generates a text file named
    # "_FDAPDEAnalysis_r3h2_8_clip_dem.txt". This file prevents the line below from being re-run again and the script
    # will skip it and move to the next line. Similar files are generated for other commands below. These commands
    # are marked with "(*)" on the comment line.
    # File outputs and copies of inputs are are stored in the *mesh* folder under _inputs.
    mesh.clip_dem()

    # (*) This line prints the command lines needed to run prep_mesh_sdist. Open up an Anaconda prompt, switch to the
    # distmesh environment, and copy and paste the commands. When it is finished running, click in the PyCharm
    # console and press Enter.
    mesh.prep_mesh_sdist()

    # (*) This line prints the command lines needed to run sdist_to_direction. Open up an Anaconda prompt, switch to the
    # distmesh environment, and copy and paste the commands. When it is finished running, click in the PyCharm
    # console and press Enter.
    mesh.sdist_to_direction()

# Class for handling GIS licensing errors.
class LicenseError(BaseException):
    pass


if __name__ == '__main__':
    try:
        if (arcpy.CheckExtension('Spatial') == 'Available') & (arcpy.CheckExtension('3D') == 'Available'):
            arcpy.CheckOutExtension('Spatial')
            arcpy.CheckOutExtension('3D')
        else:
            raise LicenseError
            # Time start of script
        start = time.time()

        # Use seaborn whitegrid style for plots.
        plt.style.use('seaborn-whitegrid')

        # Force plots to start and end on major ticks.
        # plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
        # plt.rcParams['axes.xmargin'] = 0
        # plt.rcParams['axes.ymargin'] = 0

        # Execute main code.
        main()

        # Report total script run time.
        end = time.time()
        total_time = end - start
        print(f'Total run time: {total_time:0.1f} seconds.')

    except LicenseError:
        print('Spatial Analyst and/or 3D Analyst license is unavailable')
    except arcpy.ExecuteError:
        print(arcpy.GetMessages(2))
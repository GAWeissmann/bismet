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
import matplotlib

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
    cwd = r'U:\GIS\GIS\Projects\2020xxx\D202000589_03_DBDEM_01\04_Documents\03_Scripts\Python\PyCharm-ArcPy-Pro\analytical_approach'

    # Path to fdaPDE run script directory.
    interpolation_script_directory = r'C:\Users\GWeissmann\Git\fdaPDE_bathy\code'

    # Path to R interpreter for running fdaPDE.
    path_to_rscript = r'C:\Users\GWeissmann\AppData\Local\Programs\R\R-4.2.2\bin\Rscript.exe'

    # Path to fdaPDE_run_cmd.R.
    path_to_run_r = r'C:\Users\GWeissmann\Git\fdaPDE_bathy\code\fdaPDE_run_cmd.R'

    # 10m dem. 10m covers the entire project.
    dem_10m = gis(rf'{cwd}\project_inputs\bay_delta_dem_v4.2.gdb\dem_delta_10m_20201207')

    # Create a Project object for storing project settings and reach information.
    proj = bathy.Project(
        cwd,
        dem_10m,
        interpolation_script_directory, path_to_rscript, path_to_run_r)

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
    target_poly_pc_or = gis(rf"{cwd}\{reach_name}\_inputs\ORF_Target_Polygon.shp")

    # Block polygon.
    block_poly_pc_or = gis(rf"{cwd}\{reach_name}\_inputs\ORF_block_polygon.shp")

    # Domain polygon.
    domain_poly_pc_or = gis(rf"{cwd}\{reach_name}\_inputs\ORF_domain_polygon.shp")

    # Open boundary lines.
    open_boundary_pc_or = gis(rf"{cwd}\{reach_name}\_inputs\ORF_open_boundaries.shp")

    aniso_adjust = r"U:\GIS\GIS\Projects\2020xxx\D202000589_03_DBDEM_01\04_Documents\03_Scripts\Python\PyCharm-ArcPy-Pro\analytical_approach\ORF\_inputs\aniso_revision.shp"
    diff_adjust = r"U:\GIS\GIS\Projects\2020xxx\D202000589_03_DBDEM_01\04_Documents\03_Scripts\Python\PyCharm-ArcPy-Pro\analytical_approach\ORF\_inputs\diff_revision.shp"

    orf_mb = bathy.MultiBeamBathyPoints(gis(rf"{cwd}\{reach_name}\_inputs\OR_East_MB_UTM.gdb\OR_East_MB_UTM_m"),
                                        'cinquini_mb_east', 'Cinquini (2022) MultiBeam (East)')
    glc_mb = bathy.MultiBeamBathyPoints(gis(rf"{cwd}\project_inputs\GLC_MB_UTM.gdb\GLC_MB_UTM_m"),
                                        'cinquini_mb_glc', 'Cinquini (2022) MultiBeam (Grant Line Canal)')
    orf_sb = bathy.SingleBeamBathyPoints(gis(rf"{cwd}\{reach_name}\_inputs\OR_East_SB_UTM_ft.shp"),
                                         'cinquini_sb_east', 'Cinquini (2022) SingleBeam (East)', z_ft_name='Field3')
    pc_sb = bathy.SingleBeamBathyPoints(gis(rf"{cwd}\{reach_name}\_inputs\Paradise_Cut_SB_UTM_ft.shp"),
                                         'cinquini_sb_pc', 'Cinquini (2022) SingleBeam (Paradise Cut)', z_ft_name='Field3')
    ncro_sb = bathy.SingleBeamBathyPoints(gis(rf"{cwd}\project_inputs\OldRiver_gaps_clip.shp"),
                                         'ncro_sb_or', 'NCRO (2018) Old River SingleBeam', z_m_name='POINT_Z')
    ncro_orf_sb = bathy.SingleBeamBathyPoints(gis(rf"{cwd}\{reach_name}\_inputs\FivePoints_gaps_clip.shp"),
                                         'ncro_sb_fp', 'NCRO (2013) Five Points SingleBeam', z_m_name='POINT_Z',
                                         point_erase_poly=gis(rf"{cwd}\{reach_name}\_inputs\five_points_gaps_clip_erase.shp"))
    ss_mb = bathy.SHPMultiBeamBathyPoints(gis(rf"{cwd}\project_inputs\OldRvr_at_SalmonSluMB_clip_utm.shp"),
                                         'dbc_ss_mb', 'DBC (2013) Salmon Slough MultiBeam', z_m_name='POINT_Z')

    lidar_point_erase_poly = gis(r'U:\GIS\GIS\Projects\2020xxx\D202000589_03_DBDEM_01\04_Documents\03_Scripts\Python\PyCharm-ArcPy-Pro\analytical_approach\ORF\_inputs\lidar_point_erase_poly.shp')
    # In theory, there should be a coverage for each set of island points since they occupy a 2-dimensional space.
    # However, since we have a combined coverage for all island points and they are all being included in the script,
    # it is just being included in the 2017 dataset. If each dataset were given their own coverage,
    # the result would ultimately be the same or similar enough since the coverages would be merged when the
    # coefficients are generated.
    lidar_2017 = bathy.LiDARPoints(rf'{cwd}\project_inputs\SD_Island_Points_2017_1m_1.gdb\T2017_LiDAR_Resampled_to_1m',
                                   'lidar_2017', '2017 Lidar', fmt='k^', precision=0.283, alpha=0.3, z_m_name='POINT_Z',
                                   coverage_fc=gis(rf'{cwd}\project_inputs\SD_Island_Coverage_all.shp'),
                                   lidar_point_erase_poly=lidar_point_erase_poly, buffer_in='0 Meters')
    lidar_2007 = bathy.LiDARPoints(rf'{cwd}\project_inputs\SD_Island_Points_2017_1m_1.gdb\SacDelta_Lidar_2007_Islands_m_pts',
                                   'lidar_2007', '2007 Lidar', fmt='k^', precision=0.283, alpha=0.3, z_m_name='POINT_Z',
                                   buffer_in='0 Meters')
    aug_1 = bathy.AugmentPoints(rf'{cwd}\project_inputs\SD_Island_Points_2017_1m_1.gdb\Islands_Augmented_pts1',
                                   'augment_1', 'Augmented Points 1', fmt='ko', alpha=0.3, z_m_name='POINT_Z',
                                buffer_in='0 Meters')
    aug_2 = bathy.AugmentPoints(rf'{cwd}\project_inputs\SD_Island_Points_2017_1m_1.gdb\Islands_Augmented_pts2',
                                   'augment_2', 'Augmented Points 2', fmt='ko', alpha=0.3, z_m_name='POINT_Z',
                                buffer_in='0 Meters')
    shoreline_points = bathy.ShorelinePoints('shoreline_points', 'shoreline', fmt='k.', alpha=1)

    obs_pts = [orf_mb, orf_sb, pc_sb, ncro_sb, ncro_orf_sb, ss_mb, lidar_2017, shoreline_points]

    # Feature class containing cross-sections for drawing cross-section plots. Must contain a field named "Name" with
    # a unique cross-section name.
    pc_or_xs = gis(rf"{cwd}\{reach_name}\_inputs\ORF_xs_qaqc_2.shp")
    xs_plot = gis(rf"{cwd}\{reach_name}\_inputs\ORF_xs_plot.shp")

    # Add an analysis to the reach. Currently multiple analysis are supported per reach but only one analysis
    # and analysis type (bathy.FDAPDEAnalysis) is supported.
    # Currently supported arguments for bathy.FDAPDEAnalysis are shown below.
    # When this line is run, the script will add a numbered folder. If this line is run after the folder has been
    # created, it will not create a new folder.
    proj[-1].add_analysis(bathy.FDAPDEAnalysis(dem_2m, source_dem_path=source_dem,  breaklines_fc_path=target_poly_pc_or))

    # Reach objects also support indexing to access Analysis objects the same way as Reaches. The line below
    # accesses the most recent Analysis in the most recent Reach.
    analysis = proj[-1][-1]

    # User-defined angle raster
    angle = r"U:\GIS\GIS\Projects\2020xxx\D202000589_03_DBDEM_01\04_Documents\03_Scripts\Python\PyCharm-ArcPy-Pro\analytical_approach\ORF\01_FDAPDE\r3h2_8_islands\angle_r3h2_8_islands.tif"

    # Add a mesh to the FDAPDEAnalysis. Give it a descriptive name. Current convention is r[res]h_[h0].
    # A seed value of 1000 is currently the standard for consistent mesh generation. The value used is arbitrary.
    # The FDAPDEAnalysis object will then create a folder for the mesh.
    # a parameter for res_hint_poly_fc_path exists but this functionality and mesh.create_res_hint() has not been tested.
    mesh_name = 'r3h2_8_r5'
    # mesh_name = 'test'
    analysis.add_mesh(mesh_name, domain_poly_pc_or, block_poly_pc_or, open_boundary_pc_or,
                      res=3.0, h0=2.8, seed=2000, in_angle_raster=angle)

    # Access the newly created mesh.
    mesh = analysis[mesh_name]

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

    # (*) This line prints the command lines needed to run remove_skewed_cells. Open up an Anaconda prompt, switch to the
    # distmesh environment, and copy and paste the commands. When it is finished running, click in the PyCharm
    # console and press Enter.
    mesh.remove_skewed_cells()

    # (*) This line prints the command lines needed to run convert_linestrings. Open up an Anaconda prompt, switch to the
    # distmesh environment, and copy and paste the commands. When it is finished running, click in the PyCharm
    # console and press Enter.
    mesh.convert_linestrings()

    # (*) This line copies a template yaml file and dem_list yaml file and then substitutes the correct inputs for
    # prepare schism.
    mesh.copy_schism_yaml()

    # (*) This line prints the command lines needed to run prepare_schism. Open up an Anaconda prompt, switch to the
    # distmesh environment, and copy and paste the commands. When it is finished running, click in the PyCharm
    # console and press Enter.
    mesh.prepare_schism()

    # (*) This line prints the command lines needed to run convert_mesh. Open up an Anaconda prompt, switch to the
    # distmesh environment, and copy and paste the commands. When it is finished running, click in the PyCharm
    # console and press Enter.
    mesh.convert_mesh()

    # (*) This generates a mesh boundary and checks boundary node depths. The mesh boundary and QAQC points are
    # created in the "boundary" folder. QAQC points are created when boundary node depths are > -2m.
    mesh.create_check_boundaries()

    mesh.add_observation_points(obs_pts)

    # (*) This exports all observation points to the correct format.
    mesh.clip_export_points()

    # This adds target points to the mesh. Currently there are no parameters for TargetPoint objects.
    mesh.add_target()

    # (*) This generates the target points within the target polygon/breaklines, adds the POINT_X and POINT_Y values,
    # and exports the points to .csv.
    mesh.create_target_points()

    aniso_list = [40, 80, 120]
    diff_list = [40, 80, 120, 240, 360]
    for a, d in itertools.product(aniso_list, diff_list):
        mesh.add_case(bathy.VariableCase(f'a{a}d{d}s1m1', aniso_constant=a, diff_constant=d, long_name=f'Aniso={a}, Diff={d}, '
                                                                                                    'Slopes=1, MB=1',
                                         xs_plot=True, zorder=3, lw=1.5))
        case = mesh[f'a{a}d{d}s1m1']
        case.create_variable_aniso_ratio([], from_shoreline=True, from_multibeam=True)
        case.create_variable_diff_coeff([], from_shoreline=True, from_multibeam=True)
        case.set_up_yaml_file()
    mesh['a120d120s1m1'].zorder = 3.1
    mesh['a120d120s1m1'].lw = 2.5

    aniso_list_2 = [100, 120]
    diff_list_2 = [100, 120]
    for a, d in itertools.product(aniso_list_2, diff_list_2):
        mesh.add_case(bathy.VariableCase(f'a{a}d{d}s1m1_r1', aniso_constant=a, diff_constant=d, long_name=f'Aniso={a}, Diff={d}, '
                                                                                                    'Slopes=1, MB=1',
                                         xs_plot=True, zorder=3, lw=1.5))
        case = mesh[f'a{a}d{d}s1m1_r1']
        case.create_variable_aniso_ratio([aniso_adjust], from_shoreline=True, from_multibeam=True)
        case.create_variable_diff_coeff([diff_adjust], from_shoreline=True, from_multibeam=True)
        case.set_up_yaml_file()
    mesh['a120d120s1m1_r1'].zorder = 3.1
    mesh['a120d120s1m1_r1'].final = True
    mesh['a120d120s1m1_r1'].lw = 2.5

    cases_1 = [mesh[f'a{a}d{d}s1m1'] for a, d in itertools.product(aniso_list, diff_list)]
    cases_2 = [mesh[f'a{a}d{d}s1m1_r1'] for a, d in itertools.product(aniso_list_2, diff_list_2)]
    cases = cases_1 + cases_2

    # # # MULTIPROCESSING BLOCK
    with mp.Pool(8) as p:
        p.map(parallel_run_fda_pde, cases)

    # # Run read_check_result after multiprocessing is done.
    for a, d in itertools.product(aniso_list, diff_list):
        case = mesh[f'a{a}d{d}s1m1']
        # case.read_check_result()

    for a, d in itertools.product(aniso_list_2, diff_list_2):
        case = mesh[f'a{a}d{d}s1m1_r1']
        # case.read_check_result()

    # QAQC cross-section plots of the mesh results. This uses an if statement since check files won't recognize when
    # more cases are added. Plotting is accomplished by creating points along xs lines at 1m spacing, then
    # using ArcGIS routes to locate those points. radius is a linear unit provided to LocateFeaturesAlongRoutes.
    # error_bars adds tolerance bars representing shoreline point and bathymetry point precision values.
    figsize_review = (12, 6)
    figsize_final = (5.37, 3.1)
    figsize = figsize_final
    if not os.path.exists(rf'{mesh.working_dir}\xs_plot'):
        mesh.plot_xs_lines(xs_plot, folder_name='xs_plot', radius='2 Meter', plot=True, figsize=figsize, error_bars=False,
                           final=True)
    else:
        mesh.plot_results(folder_name='xs_plot', figsize=figsize, error_bars=False, final=True)

    # Scatterplots, histograms, and summary tables. By default, plots with precision lines.
    if not os.path.exists(rf'{mesh.working_dir}\fit_qaqc'):
        mesh.prep_plot_fit(folder_name='fit_qaqc', precision=True, linear_fit=False, hexbin=True)
    else:
        mesh.plot_fit_by_case(folder_name='fit_qaqc', precision=True, linear_fit=False, hexbin=True)

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

        # Try using Agg backend to avoid 'Fail to allocate bitmap' with fit plots.
        # https://stackoverflow.com/questions/15455029/python-matplotlib-agg-vs-interactive-plotting-and-tight-layout
        matplotlib.use('Agg')

        # Use seaborn whitegrid style for plots.
        plt.style.use('default')

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
import arcpy
import os
import shutil
import pathlib

# WARNING, DO NOT MODIFY THE CODE BETWEEN THIS LINE AND THE COMMENTED LINE MARKED "END WARNING". THIS IS DEFENSIVE
# CODE TO PREVENT ACCIDENTAL DATA LOSS.
def check_path(path):
    if 'python' not in path.lower():
        raise RuntimeError('Path must contain "Python" in its name')

cwd = os.getcwd()
check_path(cwd)
# END WARNING

def main():
    gdb = GeoDataBase('test')
    gdb.set_env()
    arcpy.CreateFeatureclass_management(os.getcwd(), 'test.shp', 'Polygon')
    arcpy.CopyFeatures_management(rf'{os.getcwd()}\test.shp', 'test2')

class GeoDataBase:
    def __init__(self, name):
        self.path = os.getcwd()
        self.name = name
        self.full_path = rf'{self.path}\{self.name}.gdb'
        self.recreate()
        self._old_env = None

    def recreate(self):
        if os.path.exists(self.full_path):
            try:
                check_path(self.full_path)
                shutil.rmtree(self.full_path)
            except:
                raise OSError("File access denied.")
        arcpy.CreateFileGDB_management(self.path, f'{self.name}.gdb')

    def set_env(self):
        self._old_env = arcpy.env.workspace
        arcpy.env.workspace = self.full_path

    def reset_env(self):
        arcpy.env.workspace = self._old_env

    def create_dataset(self, name, sp_ref):
        arcpy.CreateFeatureDataset_management(self.full_path, name, sp_ref)

class PathedGeoDataBase(GeoDataBase):
    def __init__(self, path):
        self.path = str(pathlib.PureWindowsPath(path).parent)
        self.name = pathlib.PureWindowsPath(path).stem
        self.full_path = rf'{self.path}\{self.name}.gdb'
        self.recreate()
        self._old_env = None


class ExistingGeoDataBase(GeoDataBase):
    def __init__(self, path):
        self.path = str(pathlib.PureWindowsPath(path).parent)
        self.name = pathlib.PureWindowsPath(path).stem
        self.full_path = rf'{self.path}\{self.name}.gdb'
        self._old_env = None


class WorkingDirectory:
    def __init__(self, name):
        self.path = os.getcwd()
        self.name = name
        self.full_path = rf'{self.path}\{self.name}'
        self.recreate()
        self._old_env = None

    def recreate(self):
        if os.path.exists(self.full_path):
            try:
                check_path(self.full_path)
                shutil.rmtree(self.full_path)
            except:
                raise OSError("File access denied.")
        os.mkdir(self.full_path)

    def set_env(self):
        self._old_env = arcpy.env.workspace
        arcpy.env.workspace = self.full_path

    def reset_env(self):
        arcpy.env.workspace = self._old_env


class ExistingWorkingDirectory(WorkingDirectory):
    def __init__(self, path):
        self.path = str(pathlib.PureWindowsPath(path).parent)
        self.name = pathlib.PureWindowsPath(path).stem
        self.full_path = path
        self._old_env = None


class PathedWorkingDirectory(WorkingDirectory):
    def __init__(self, path):
        self.path = str(pathlib.PureWindowsPath(path).parent)
        self.name = pathlib.PureWindowsPath(path).stem
        self.full_path = path
        self.recreate()
        self._old_env = None


if __name__ == '__main__':
    main()
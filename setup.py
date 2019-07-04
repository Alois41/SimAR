# import sys
# from cx_Freeze import setup, Executable
# import os.path
# sys.path.append(os.path.abspath("./source"))


# base = None
# if sys.platform == "win32":
#     base = "Win32GUI"  # pour application graphique sous Windows
#
# build_exe_options = {"includes": ["OpenGL"]}
# additional_mods = ['numpy.core._methods', 'numpy.lib.format']
# excludes = ['matplotlib', 'tornado', 'PIL', 'jupyter_client', 'jupyter_core']
# include_files = ['config.yml']
#
# PYTHON_INSTALL_DIR = os.path.dirname(os.path.dirname(os.__file__))
# os.environ['TCL_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tcl8.6')
# os.environ['TK_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tk8.6')
#
# setup(
#     name="SteelBox",
#     version="1.0",
#     description="",
#     options={"build_exe": {"packages": ["OpenGL", "numpy"], 'includes': additional_mods, 'excludes': excludes,
#                            'include_files': include_files}},
#     executables=[Executable("run.py", base=base)]
# )

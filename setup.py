import sys
import os
from cx_Freeze import setup, Executable

build_exe_options = {"includes": ["OpenGL"]}
additional_mods = ['numpy.core._methods', 'numpy.lib.format']

setup(
    name="SteelBox",
    version="1.0",
    description="",
    options={"build_exe": {"packages": ["OpenGL"], 'includes': additional_mods}},
    executables=[Executable("run.py", base="Console")]
)

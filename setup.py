"""Build the small native library into platform wheels."""

import os
import shutil
import subprocess
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel


class BuildNative(build_py):
    def run(self):
        super().run()
        if os.environ.get("OMNI_SKIP_NATIVE") == "1":
            return
        root = Path(__file__).resolve().parent
        build = Path(self.build_lib).resolve().parent / "native"
        cuda = os.environ.get("OMNI_BUILD_CUDA", "OFF")
        subprocess.check_call(
            [
                "cmake",
                "-S",
                str(root),
                "-B",
                str(build),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DOMNI_TESTS=OFF",
                "-DOMNI_BLAS=OFF",
                "-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON",
                "-DOMNI_CUDA=" + cuda,
            ]
        )
        subprocess.check_call(
            ["cmake", "--build", str(build), "--config", "Release", "--parallel", "4"]
        )
        destination = Path(self.build_lib) / "omni_tokenizer" / "_native"
        destination.mkdir(parents=True, exist_ok=True)
        libraries = [
            p
            for p in build.rglob("*")
            if p.name in {"omni.dll", "libomni.so", "libomni.dylib"}
        ]
        if len(libraries) != 1:
            raise RuntimeError("Could not locate built native library")
        shutil.copy2(libraries[0], destination / libraries[0].name)


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class CtypesWheel(bdist_wheel):
    def get_tag(self):
        _, _, platform = super().get_tag()
        return "py3", "none", platform


setup(
    cmdclass={"build_py": BuildNative, "bdist_wheel": CtypesWheel},
    distclass=BinaryDistribution,
)

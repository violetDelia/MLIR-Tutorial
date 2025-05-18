import os
import sys

import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
import lit.util

def find_runtime(name):
    path = ""
    for prefix in ["", "lib"]:
        path = os.path.join(
            config.llvm_lib_dir, f"{prefix}{name}{config.llvm_shlib_ext}"
        )
        if os.path.isfile(path):
            break
    return path

def add_runtime(name):
    return ToolSubst(f"%{name}", find_runtime(name))

# name: The name of this test suite.
config.name = 'DEMO_TEST'

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir',".py"]

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

config.substitutions.append(('%PATH%', config.environment['PATH']))
#config.substitutions.append(('%shlibext', config.llvm_shlib_ext))

llvm_config.with_system_environment(['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])
llvm_config.use_default_substitutions()
# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

config.excludes = [
    "CMakeLists.txt",
    "lit.cfg.py",
    "lit.site.cfg.py",
]

tool_dirs = [
    config.mlir_binary_dir,
    config.mlir_tutorial_tool_dir
]
tools = [
    ToolSubst("ns-opt", config.mlir_tutorial_ns_opt, unresolved="ignore"),
    add_runtime("mlir_runner_utils"),
    add_runtime("mlir_c_runner_utils"),
]
llvm_config.add_tool_substitutions(tools, tool_dirs)

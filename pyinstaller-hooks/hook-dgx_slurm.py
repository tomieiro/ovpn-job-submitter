"""Bundles the job templates into the frozen executable.

``--collect-data`` skips ``.py`` files, which silently left
``templates/execute_notebook.py`` out of the binary.
"""

from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files(
    "dgx_slurm",
    include_py_files=True,
    includes=["templates/*"],
)

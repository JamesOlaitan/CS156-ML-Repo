"""Execute the project notebook with the isolated cs156 kernel; preserve outputs.

Run from any directory with this project's .venv/bin/python. Re-execution uses
the frozen procedure, not a fresh search for a favorable split.
"""
from pathlib import Path
import os

ROOT=Path(__file__).resolve().parents[1]
os.environ['JUPYTER_PATH']=str(ROOT/'.venv/share/jupyter')
os.environ['JUPYTER_RUNTIME_DIR']=str(ROOT/'.cache/jupyter')
os.environ['IPYTHONDIR']=str(ROOT/'.cache/ipython')
os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
for folder in ['jupyter','ipython','matplotlib']:(ROOT/'.cache'/folder).mkdir(parents=True,exist_ok=True)

import nbformat
from nbclient import NotebookClient

path=ROOT/'notebooks/first_pipeline.ipynb'
notebook=nbformat.read(path,as_version=4)
client=NotebookClient(notebook,timeout=1800,kernel_name='cs156',resources={'metadata':{'path':str(ROOT)}})
try:
    client.execute()
finally:
    # Keep diagnostic output even if a real execution error occurs.
    nbformat.write(notebook,path)
errors=[output for cell in notebook.cells for output in cell.get('outputs',[]) if output.get('output_type')=='error']
if errors:raise RuntimeError(f'Notebook contains {len(errors)} error outputs.')
print(f'Executed {sum(cell.cell_type=="code" for cell in notebook.cells)} code cells successfully: {path}')

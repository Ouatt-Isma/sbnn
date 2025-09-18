import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from tqdm import tqdm

# Load the notebook
notebook_filename = "snn.ipynb"
with open(notebook_filename) as f:
    nb = nbformat.read(f, as_version=4)

# Count code cells
code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
total = len(code_cells)

# Wrap the preprocessor with tqdm
class TqdmExecutePreprocessor(ExecutePreprocessor):
    def preprocess_cell(self, cell, resources, cell_index):
        if cell.cell_type == "code":
            self.pbar.update(1)
        return super().preprocess_cell(cell, resources, cell_index)

ep = TqdmExecutePreprocessor(timeout=-1, kernel_name="python3")
ep.pbar = tqdm(total=total, desc="Executing notebook", unit="cell")

# Execute
ep.preprocess(nb, {"metadata": {"path": "."}})
ep.pbar.close()

# Save executed notebook
with open("executed_notebook.ipynb", "w") as f:
    nbformat.write(nb, f)

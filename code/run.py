import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from tqdm import tqdm
import subprocess
import sys
import datetime
from pathlib import Path

def run_notebook(notebook_path: str):
    # Paths
    nb_path = Path(notebook_path)
    if not nb_path.exists():
        print(f"❌ Notebook {notebook_path} not found.")
        sys.exit(1)

    output_path = nb_path.with_name(nb_path.stem + "_executed.ipynb")

    # Load notebook
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Count code cells
    code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
    total = len(code_cells)

    # Progress bar wrapper
    class TqdmExecutePreprocessor(ExecutePreprocessor):
        def preprocess_cell(self, cell, resources, cell_index):
            if cell.cell_type == "code":
                self.pbar.update(1)
            return super().preprocess_cell(cell, resources, cell_index)

    ep = TqdmExecutePreprocessor(timeout=-1, kernel_name="python3")
    ep.pbar = tqdm(total=total, desc=f"Executing {nb_path.name}", unit="cell")

    # Execute
    ep.preprocess(nb, {"metadata": {"path": "."}})
    ep.pbar.close()

    # Save executed notebook
    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"✅ Executed notebook saved to: {output_path}")

    return output_path

def git_commit_and_push(file_path: Path):
    commit_message = f"Auto-update {file_path.name} at {datetime.datetime.now().isoformat(timespec='seconds')}"

    try:
        # Check if there are changes
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        if not status.stdout.strip():
            print("ℹ️ No changes to commit.")
            return

        subprocess.run(["git", "add", str(file_path)], check=True)
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        subprocess.run(["git", "push"], check=True)
        print("🚀 Changes pushed to GitHub.")
    except subprocess.CalledProcessError:
        print("⚠️ Git push failed. Make sure your repo is set up correctly.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_nb_with_progress.py <notebook.ipynb>")
        sys.exit(1)

    notebook_file = sys.argv[1]
    executed_file = run_notebook(notebook_file)
    git_commit_and_push(executed_file)

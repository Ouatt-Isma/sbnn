import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from tqdm import tqdm
import subprocess
import sys
import datetime
from pathlib import Path

def git_commit_and_push(file_path: Path, cell_index: int, total: int):
    commit_message = f"Auto-update {file_path.name} after cell {cell_index+1}/{total} at {datetime.datetime.now().isoformat(timespec='seconds')}"

    try:
        # Check if there are changes
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        if not status.stdout.strip():
            return

        subprocess.run(["git", "add", str(file_path)], check=True)
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        subprocess.run(["git", "push"], check=True)
        print(f"📤 Pushed notebook update after cell {cell_index+1}/{total}")
    except subprocess.CalledProcessError:
        print("⚠️ Git push failed (check repo setup/authentication).")

def run_notebook_live(notebook_path: str):
    nb_path = Path(notebook_path)
    if not nb_path.exists():
        print(f"❌ Notebook {notebook_path} not found.")
        sys.exit(1)

    # Load notebook
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Count code cells
    code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
    total = len(code_cells)

    # Progress bar wrapper
    class LiveCommitExecutePreprocessor(ExecutePreprocessor):
        def preprocess_cell(self, cell, resources, cell_index):
            if cell.cell_type == "code":
                # Execute cell
                result = super().preprocess_cell(cell, resources, cell_index)

                # Save notebook after each cell
                with open(nb_path, "w", encoding="utf-8") as f:
                    nbformat.write(nb, f)

                # Commit + push
                git_commit_and_push(nb_path, cell_index, total)

                # Update progress bar
                self.pbar.update(1)
                return result
            return super().preprocess_cell(cell, resources, cell_index)

    ep = LiveCommitExecutePreprocessor(timeout=-1, kernel_name="python3")
    ep.pbar = tqdm(total=total, desc=f"Executing {nb_path.name}", unit="cell")

    # Execute
    ep.preprocess(nb, {"metadata": {"path": "."}})
    ep.pbar.close()

    print(f"✅ Finished execution of {nb_path.name}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_nb_with_progress_live.py <notebook.ipynb>")
        sys.exit(1)

    run_notebook_live(sys.argv[1])

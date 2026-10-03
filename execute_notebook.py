"""Execute every notebook cell in a real kernel; never overwrite a prior output."""
import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", action="store_true",
                        help="Also run a fresh full seed-0 lifted experiment")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output notebook; existing notebooks are protected")
    root = Path(__file__).resolve().parent
    notebook = nbformat.read(root / "PINN_2D_Biology_improved.ipynb", as_version=4)
    nbformat.validate(notebook)
    client = NotebookClient(notebook, timeout=900, kernel_name="python3",
                            allow_errors=False, record_timing=False,
                            resources={"metadata": {"path": str(root)}})
    # Only this child kernel sees the flag. No user/global environment is changed.
    import os
    kernel_env = dict(os.environ, PINN_RUN_NEW="1" if args.train else "0")
    client.execute(env=kernel_env)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if not all(cell.execution_count is not None for cell in code_cells):
        raise RuntimeError("Not all code cells executed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        nbformat.write(notebook, stream)
    print(f"Executed {len(code_cells)} code cells successfully; new training: {args.train}")
    print(f"Saved {args.output.name}")


if __name__ == "__main__":
    main()

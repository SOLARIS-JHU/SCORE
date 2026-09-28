#!/usr/bin/env python3
"""Validate and execute source notebooks without modifying their outputs."""
import argparse
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout', type=int, default=300, help='Seconds per cell')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = root / 'tutorials/outputs/executed'
    destination.mkdir(parents=True, exist_ok=True)
    # Avoid installing or changing a user's kernelspec. Always use this Python.
    with tempfile.TemporaryDirectory(prefix='lyapunov-kernel-') as directory:
        import json
        spec = Path(directory) / 'lyapunov-tutorial'
        spec.mkdir()
        (spec / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Lyapunov tutorials', 'language': 'python',
        }))
        manager = KernelSpecManager(kernel_dirs=[directory])
        for path in sorted((root / 'tutorials').glob('*.ipynb')):
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            client = NotebookClient(notebook, timeout=args.timeout,
                                    kernel_name='lyapunov-tutorial',
                                    resources={'metadata': {'path': str(path.parent)}})
            client.km = client.create_kernel_manager()
            client.km.kernel_spec_manager = manager
            print(f'Executing {path.name}', flush=True)
            client.execute(env={**os.environ, 'MPLBACKEND': 'module://matplotlib_inline.backend_inline'})
            nbformat.write(notebook, destination / path.name)
            print(f'PASS {path.name}', flush=True)
    print(f'Executed notebooks: {destination}')


if __name__ == '__main__':
    main()

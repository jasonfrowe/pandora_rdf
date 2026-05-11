# Pandora RDF workbook

## Installation

1. Create a virtual environment.
2. Activate it.
3. Install project dependencies, including `ipykernel` for notebook use.

```bash
python -m venv .pandora_rdf
source .pandora_rdf/bin/activate
python -m pip install --upgrade pip
python -m pip install numba matplotlib astropy ipykernel
```

If you want this environment to appear as a notebook kernel:

```bash
python -m ipykernel install --user --name pandora_rdf --display-name "Python (pandora_rdf)"
```
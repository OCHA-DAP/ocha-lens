# Installation

## From PyPI

Install ocha-lens using pip:

```bash
pip install ocha-lens
```

## Development Installation

For development, install from source:

1. Clone the repository:
```bash
git clone https://github.com/OCHA-DAP/ocha-lens.git
cd ocha-lens
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

## Environment Configuration

## Dependencies

ocha-lens requires Python 3.10 or later and depends on:

- pandas
- xarray
- netcdf4

These will be installed automatically when you install ocha-lens.

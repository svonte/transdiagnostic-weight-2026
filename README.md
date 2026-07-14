# Transdiagnostic Weight 2026

This repository is a minimal public code archive accompanying the paper on
transdiagnostic weight trajectories and clinically significant weight gain
across psychiatric treatment studies.

## Included scripts

1. `transdiagnostic_weight_01_harmonize.py` extracts and harmonizes
   longitudinal weight measurements and baseline covariates.
2. `transdiagnostic_weight_02_models.py` fits the primary longitudinal and
   clinically significant weight-gain models.
3. `transdiagnostic_weight_02b_metabolic.py` analyses the secondary
   study-specific metabolic outcomes.

The intended analysis order is
`transdiagnostic_weight_01_harmonize.py`,
`transdiagnostic_weight_02_models.py`, and then
`transdiagnostic_weight_02b_metabolic.py`.

## Data and outputs

Participant-level data, derived datasets, model outputs, figures, and paper
files are not included. The data are restricted to the approved research
environment and cannot be redistributed in this public repository.

The scripts therefore require access to the original project data layout and
appropriate permissions before they can be executed. They write derived data
and analysis outputs to the project results directory.

## Software

The scripts were developed for Python 3 with the following main packages:

- NumPy
- pandas
- PyArrow
- SciPy
- statsmodels
- openpyxl

Exact package versions should be taken from the original analysis environment.

## Scope of this archive

This is a publication code archive containing the principal analysis scripts.
It is not a standalone reproduction package, because the restricted data and
some project-level helper files are intentionally omitted.

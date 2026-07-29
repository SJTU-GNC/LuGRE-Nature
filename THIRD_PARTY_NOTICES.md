# Third-party notices and exclusions

## Python dependencies

The public-code staging repository does not vendor the portable Python runtime
or the previous local copies of `openpyxl` and `pypdf`. Dependencies are
declared in `environment/requirements.txt` and retain their own upstream
licences.

## Scientific data and images

Data, generated figures, reference figures, and non-code image assets are
distributed through the separate Zenodo dataset record, not this GitHub code
repository. Their public release remains subject to the rights audit supplied
with the Zenodo staging package.

## Excluded source archives

Commercial, licensed, unavailable, and very large provider archives are not
included. The scientific effect of each exclusion is documented in
`manifest/external_large_sources.csv` and
`docs/REPRODUCIBILITY_MATRIX.md`.

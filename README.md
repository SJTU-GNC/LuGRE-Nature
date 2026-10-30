# LuGRE R3 Reproduction Code

Python and PowerShell code supporting the manuscript *Observing Earth's
ionospheric irregularities from the moon*.

This repository contains the public code release for the frozen R3 selected
182-station analysis dated 30 October 2026. Numerical inputs, source data,
official figure references, and verified outputs are versioned separately in
the associated Zenodo data record.

## R3 scope

- Five main figures and eight Extended Data figures.
- A selected Ground population of 182 stations, 29,803,114 native 1-Hz
  receiver-satellite epochs, and 498,430 accepted minute windows.
- The accepted LuGRE h50 and LEO populations.
- Portable relative-path entry points for Windows and Python 3.12.
- Environment, provenance, selection, and validation metadata needed to audit
  the analysis code.

## Associated manuscript

**Title:** *Observing Earth's ionospheric irregularities from the moon*

**Authors:** Rong Yang, Zhihong Li, Andrea Nardin, Alex Minetto, Fabio Dovis,
Claudio Cesaroni, Luca Spogli, Lucilla Alfonsi, Xirui Miao, Yuquan Ma, Wei Gao,
Li-Ta Hsu, and Xingqun Zhan.

**Corresponding author:** Rong Yang (rongyang@sjtu.edu.cn)

Full creator and affiliation metadata are recorded in `CITATION.cff` and
`.zenodo.json`.

## Data download

Download the latest R3 data version from the Zenodo dataset record associated
with DOI `10.5281/zenodo.21672261`, then extract its contents into this
repository root. The data package restores the numerical-input, reference, and
result paths expected by the code.

The public data package deliberately excludes the YunYao profiles under
`components/fig5joint/data/restricted_yunyao/` because provider permission for
public redistribution has not been documented. The Fig.5 workflow therefore
requires those separately authorised inputs for complete execution. No
restricted YunYao file is present in this GitHub repository.

## Environment and execution

Create a Python 3.12 environment and install the pinned requirements:

```text
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

After merging the Zenodo data package into the repository root:

```text
.venv\Scripts\python run_all.py --list
.venv\Scripts\python run_all.py
.venv\Scripts\python run_all.py --tasks fig3 fig4
```

The bundled Windows setup script in the complete Zenodo reproduction package
can install from its verified offline wheels. Those wheels are not duplicated
in this code repository.

## Reproduction boundaries

The code regenerates scientific plots from the versioned numerical inputs.
Some approved manuscript figures contain authored artwork or manual composite
layout elements; their task metadata states where pixel identity is not
claimed. Large upstream mission, RINEX, radar, and radio-occultation archives
remain with their original providers and are documented through provenance and
access inventories rather than redistributed here.

## Citation and responsible reuse

Please cite the archived R3 software release and the associated Zenodo data
version. Cite the manuscript when a public citation becomes available. Reuse
must preserve applicable copyright and licence notices, give appropriate
attribution, identify modifications, and must not imply endorsement or
misrepresent the original data, methods, results, figures, or limitations.

See `REUSE_AND_CITATION.md` for the complete statement.

## Licences

Copyright (c) 2026 Rong Yang, Zhihong Li, Andrea Nardin, Alex Minetto, Fabio
Dovis, Claudio Cesaroni, Luca Spogli, Lucilla Alfonsi, Xirui Miao, Yuquan Ma,
Wei Gao, Li-Ta Hsu, and Xingqun Zhan.

Original project software is released under the MIT License; see `LICENSE`.
Author-generated data are licensed separately under CC BY 4.0 in the Zenodo
data record. Third-party materials retain their original terms and are not
relicensed by this repository.

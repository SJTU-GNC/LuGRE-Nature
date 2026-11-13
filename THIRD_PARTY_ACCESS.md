# Third-party sources and access

## YunYao profiles excluded from the public archives

The prior local review bundle explicitly states that provider approval is required before public redistribution of the two YunYao observations **and their extracted numerical profiles**. Such approval has not been established for this packaging task. The public release therefore excludes all files below `components/fig5joint/data/restricted_yunyao/`, including the original NetCDFs, extracted electron-density profiles, match table and matched-location IRI curves. Their expected layout and identifiers are documented in the component exclusion manifest. Local originals and older releases are preserved.

Request authorized data from Tianjin Yunyao Aerospace Technology Co., Ltd. through https://open.tjyyspace.com/#/dataDownload. Default public Fig.5 reproduction does not read the restricted directory and explicitly marks the missing panels/overlays. To reproduce them, obtain permission and supply the documented inputs with `run_R4.py --with-restricted --restricted-dir ...`. The exact approved complete Fig.5 is retained as a paper-result reference, not used as a numerical background or a substitute for those data. ED9 and all other public tasks remain runnable.

## Other sources

`provenance/DATA_ACCESS_INVENTORY.csv` lists the original mission, GNSS, radio-occultation, radar, model, space-weather, antenna and cartographic sources. Original data-network acknowledgements and conditions remain applicable. Raw mission/RINEX/multi-network acquisition archives are not all duplicated here; the compact scientific inputs and provenance identify the computational boundary.

Natural Earth resources and their source information are included for offline maps. NAVCEN/JRC antenna resources and mission illustrations retain their source notices. No author licence automatically applies to third-party data/artwork or Springer Nature manuscript templates. Dependency wheels retain their embedded distribution licences. No proprietary Windows fonts or Microsoft Office runtime are included. Figure1's optional editable-artwork export requires the user's own PowerPoint installation; the Python numerical tasks do not.

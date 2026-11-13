# Figure 5 and Extended Data 9: R4 public reproduction

The approved Fig5_R4.png and joint-observable figure are byte-identical to the frozen R3 targets. These figures retain their separate case-study and high-rate populations, not the Ground182 Fig3/4 cohort.

Run `python verify_inputs_R4.py`, then `python run_R4.py --figure all --output-dir results_R4`.

The default public entry point does not read YunYao files even if a local private copy is present. It writes Fig5_R4_public_partial.png, with missing-data labels in panels g/i. The two YunYao electron-density profiles, their matched-location IRI curves and YunYao track overlays cannot be reproduced from the public subset. PFISR and all other included numerical panels still redraw. The approved complete publication PNG is a separate reference, never used as a runtime data background. ED9 redraw and its true-P95 checks are unaffected.

For a complete numerical Fig.5 redraw, obtain provider permission and the two profiles through https://open.tjyyspace.com/#/dataDownload, then run `python run_R4.py --figure Fig5 --with-restricted --restricted-dir /path/to/authorized/layout --output-dir results_authorized`. The expected relative file paths are recorded in provenance/YunYao_public_exclusion_manifest_R4.json. Raw NetCDF files are documented source provenance; the plotting entry point reads the extracted profiles and match table, not NetCDF. No raw-to-extracted conversion is claimed here.

Public archive preparation must exclude the entire data/restricted_yunyao directory, including extracted profiles and match-location IRI tables, unless permission is explicitly established. Local originals and the older review bundle remain preserved.

The scientific redrawing target is the included numerical population. The historical PowerPoint montage and fonts are not claimed pixel-exact. Current approved PNGs remain reference targets. All third-party data/artwork retain their source terms; authors' code licensing does not relicense them.

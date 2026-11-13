# R4 upload notes

Two complementary ZIP files are prepared for the existing records:

- `LuGRE_R4_Data_20261113_References_checked.zip`: current figures, full-quality manuscript snapshots, scientific inputs, source tables, data dictionaries and provenance. Update the data record https://doi.org/10.5281/zenodo.21672260.
- `LuGRE_R4_Code_20261113_References_checked.zip`: supported Python workflows, analysis/plotting code, environment lock, Windows/Python 3.12 offline dependency wheels, validators and documentation. Update the code record https://doi.org/10.5281/zenodo.21675357.

Both ZIPs use the same top-level folder `LuGRE_R4_release/`. Extract both into the same parent directory to obtain the complete runnable release. Shared documentation is byte-identical. Neither ZIP alone supplies the whole computational environment and input set.

Create new versions of the existing records rather than editing an old scientific version in place. The repository assigns any new version-specific DOI; this package does not invent one. Cross-link the data and software records and cite the final assigned versions. The prior 21672261/21675358 version identifiers appear only where needed for provenance/licence history.

The archive includes no email drafts, coauthor review files, passwords, acquisition caches or unrelated candidate plots. The root and component access notices state the boundary between packaged processed inputs and upstream raw archives. Complete raw ground/mission archives are not duplicated. Review THIRD_PARTY_ACCESS.md for the YunYao exception before publication.

The user will upload these files. No upload, repository publication or online synchronization was performed in preparing this release.

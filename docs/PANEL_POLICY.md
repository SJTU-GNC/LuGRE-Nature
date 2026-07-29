# Panel Reproduction Policy

Panels are split only when the accepted full-figure chain is incomplete.

Each panel task:

1. has a root-level entry point;
2. reads numerical inputs or scientific source assets, not an accepted PNG;
3. clears only its own output directory before running;
4. leaves exactly one PNG in that directory;
5. runs in a fresh Python process with the pinned environment;
6. declares whether it is pixel-exact or a scientific-only redraw;
7. retains the accepted or best archived panel image only under `reference/`
   for visual QA.

Panel outputs are not automatically assembled into a substitute accepted
figure. A newly written composer could be useful for exploration, but it would
not restore the missing accepted layout provenance and therefore would not be
reported as an exact manuscript reproduction.

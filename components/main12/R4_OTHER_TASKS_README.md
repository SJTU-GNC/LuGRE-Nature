# R4 Fig1, space-weather and antenna tasks

- Fig1: unchanged authored schematic; compute_fig1_fresnel_R4.py independently verifies the six PPTX finite-source Fresnel series. The whole schematic is a reference export, not a data-to-pixels reconstruction.
- Space weather: render_space_weather_R4.py redraws the current approved R4 six-row layout from the frozen independent context cohort: 2,244 windows, 693 polar. These are historical OP-context summaries, not the 2,196 post-QC Fig2 event windows. Keep the populations distinct.
- Antenna: render_antenna_R4.py redraws the approved A-over-B / C-at-right layout from frozen off-boresight density, range and source-cut summaries. C shades actual full min/max across raw SVN-by-azimuth cuts and keeps the archived equal-SVN reference median. Galileo is the GRAP population-model azimuth set, not individual spacecraft. Optional prepare_full_pattern_range_R4.py recomputes and checks these reference curves from the included official source pattern files.

Scripts use MAIN12_OUTPUT as the output parent and never write to source/manuscript folders. PNG-only 600dpi, exact approved layout preserved. Arial is not redistributed; a locally installed font or Matplotlib fallback may change rasterized pixels. Original artwork and official third-party antenna source files retain their own rights; no blanket CC-BY/MIT claim is made for them.


## Exact portable commands and outputs

Run from this component with the packaged Python environment:

```
python run_other_r4.py --task fig1 --output outputs_R4
python run_other_r4.py --task space_weather --output outputs_R4
python run_other_r4.py --task antenna --output outputs_R4
```

Outputs: `fig1/Fig1_R4.png` (explicit authored reference copy), `fig1/Fig1_fresnel_radius_recomputed.csv` and `Fig1_fresnel_verification.json`; `space_weather/Extended_Data_Space_weather_index.png` plus style QA; `antenna/Extended_Data_GNSS_antenna_geometry.png` plus `antenna_QA_R4.json`. All folders above are below the requested output directory. Optional raw-pattern verification: `python code/prepare_full_pattern_range_R4.py` (or set MAIN12_OUTPUT to select its output parent).

Fig1 theoretical inputs: transmitter-screen distance26,000km, carriers1575.42/1176.45MHz; first-Fresnel radius uses sqrt(lambda*Dtx*Drx/(Dtx+Drx)). No synthetic measurements. Antenna angle evidence retains137,686 original samples at50–1000km: GPS13.9–16.3deg and Galileo12.5–14.4deg. Reference first valleys are27/33/24/28deg for GPSL1/L5/GalileoE1/E5a. Full min/max range and the original median have distinct definitions and are not interchanged.

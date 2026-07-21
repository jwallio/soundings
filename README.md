# CONUS Upper-Air Data Watch

Public, data-availability dashboard for CONUS upper-air observations and NCO operational-message status.

Live site: https://soundings.wall.cloud

The scheduled GitHub Actions workflow checks NCO status four times daily (01:45Z, 03:45Z, 13:45Z, and 15:45Z) and publishes the static dashboard to GitHub Pages. The 15:45Z run and manual dispatch perform the full IGRA historical refresh and social-graphics build; the other scheduled runs retain the current IGRA/social assets and update NCO status/site content.

## Local build

```powershell
python -m pip install -r requirements-dashboard.txt
python scripts/run_upper_air_monitor.py --refresh --years 6
python scripts/build_upper_air_social_graphics.py
python scripts/build_upper_air_public_site.py
```

The generated public bundle is `upper-air-site/dist/`.

The dashboard compares archive records with a same-date baseline and shows NCO operational-message/product-record availability as a separate signal. GFS, NAM, and NCEP rows are operational-message products, not unique station counts; missing source rows remain missing and are not converted to zero. NCO status is not a confirmed balloon-launch count, and neither layer is a model-skill measure.

The generated page exposes the latest source-record date, last successful NCO refresh, build time, and a retained-data warning when an upstream NCO request or parser fails. A failed NCO refresh does not replace valid historical CSV data with an empty file.

Known limitation: the current station master supplies one active expected CONUS inventory (69 stations), so that denominator is applied retrospectively. Historical station activation/deactivation metadata is required before changing that methodology; it is intentionally not inferred here.


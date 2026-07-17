# CONUS Upper-Air Data Watch

Public, data-availability dashboard for CONUS upper-air observations and NCO operational-message status.

Live site: https://soundings.wall.cloud

The scheduled GitHub Actions workflow refreshes the IGRA archive comparison, NCO status products, station map, and static dashboard, then publishes the result to GitHub Pages. The workflow runs daily after the afternoon source refresh and can also be started manually.

## Local build

```powershell
python -m pip install -r requirements-dashboard.txt
python scripts/run_upper_air_monitor.py --refresh --years 6
python scripts/build_upper_air_social_graphics.py
python scripts/build_upper_air_public_site.py
```

The generated public bundle is `upper-air-site/dist/`.

The dashboard compares archive records with a same-date baseline and shows NCO operational-message reporting as a separate signal. It does not treat either layer as a confirmed count of launches or a model-skill measure.
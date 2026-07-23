# CONUS Upper-Air Data Watch

Public, data-availability dashboard for CONUS upper-air observations and NCO operational-message status.

Live site: https://soundings.wall.cloud

The scheduled GitHub Actions workflow checks NCO status four times daily (01:45Z, 03:45Z, 13:45Z, and 15:45Z) and publishes the static dashboard to GitHub Pages. The 15:45Z run and manual dispatch perform the full IGRA historical refresh and social-graphics build; the other scheduled runs restore retained IGRA/station data, skip the expensive archive refresh, and update NCO status/site content. If retained archive data is unavailable, an NCO-only run is promoted to a full refresh.

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

## Reading a recent recovery

The July 21, 2026 snapshot showed a sharp short-term recovery, but not a sustained 30-day reversal:

- The latest 7 days were **891 observed / 890.9 expected (100.0%)**, up from **93.7%** in the preceding 7 days.
- The latest 30 days were still **96.3% of baseline**, down 1.5 percentage points from the prior 30 days.
- Among the stations visible in the ranking panels, **LZK (Little Rock)** was the clearest recovery signal, followed by **BMX (Shelby County)**, **CHS (Charleston)**, and **OUN (Norman)**. LZK improved sharply but remained below its 30-day baseline.
- **TWC (Tucson)** remained at zero reported archive records in the latest 7-day window. **DRT (Del Rio)**, **DTX (Detroit/Pontiac)**, **OAK (Oakland)**, and **VEF (Las Vegas)** remained important shortfall contributors.

These station findings are archive-record comparisons against the 2021–2024 same-date baseline, not confirmed launch-cause findings. The public charts display the top eight shortfalls and surpluses for each selected window rather than a complete 69-station decomposition. The recent-contributor comparison derives the preceding period by subtracting nested station windows (for example, 30-day totals from 90-day totals), so it is a directional diagnostic rather than a separate daily station product. Current NCO issue status is a separate operational-message signal and should not be treated as an explanation for archive counts.

For reproducible current values, use the downloadable [archive availability CSV](https://soundings.wall.cloud/archive-availability.csv) and [latest station status CSV](https://soundings.wall.cloud/latest-station-status.csv).

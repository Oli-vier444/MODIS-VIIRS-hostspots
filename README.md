# MODIS-VIIRS-hostspots
Visual Fire

Visual Fire is a web tool for exploring satellite fire detections from 2000 to today and for judging whether a given month or fire season is unusual. It runs on a harmonized MODIS + VIIRS dataset (one common schema for both sensors) served by a FastAPI + DuckDB backend, with a single-page front end.

Built for the NASA Space Apps Challenge.

What it does

Map and table. Detections are drawn on an interactive Leaflet map (OpenStreetMap, CARTO and Esri satellite base layers), colored by sensor (orange = MODIS, blue = VIIRS), and linked to a paginated table: click a point to highlight its row, or a row to find it on the map. The full result of any query can be downloaded as CSV.

Calendar and charts. A monthly calendar colored by detections, high-confidence detections, total FRP or pixel area. Up to five more years can be added to compare the same month side by side on a shared color scale. Charts cover daily and monthly series, confidence, day/night, satellites, fire source type and top countries. Every chart can be saved as PNG or CSV.

Historical series and anomalies. The whole 2000-to-today series as a years × months (or years × weeks) matrix, colored by value, by anomaly relative to the typical month, or by a traffic-light rating. For any month it computes the historical median and typical range (P25–P75) of a zone, reports how many times the median the month is, and gives a green / yellow / orange / red status. It also plots the cumulative season against the historical band. A zone is a country, or simply the area currently visible on the map. The current (incomplete) month is compared with the same days in earlier years.

Shared filters. Minimum confidence (any / nominal or higher / high only) and day or night apply to every view.

How to run

You need Python 3.10+ and an internet connection (Leaflet, Chart.js and the map tiles are loaded from CDNs).

1. Install the dependencies (once)

powershell
cd backend
python -m venv venv
venv\Scripts\activate
pip install fastapi "uvicorn[standard]" duckdb pandas

2.Download Data

The fire data in this repository is incomplete due to github's restrictions, download it from https://drive.google.com/drive/folders/11ekA7Tj2lJCB6KWbsgz-R0VySxQEfyRU?usp=sharing and place it under the "data" folder.

3. Start the API (terminal 1)

powershell
cd backend
venv\Scripts\activate
uvicorn app.main:app --reload

The API runs at http://127.0.0.1:8000 and its interactive documentation at http://127.0.0.1:8000/docs.

4. Serve the front end (terminal 2). Go to the folder that contains index.html (visual_fire in our setup):

powershell
cd visual_fire
python -m http.server 5500

5. Open the app in a web browser at http://localhost:5500

The API only accepts requests from http://localhost:5500 and http://127.0.0.1:5500, so the front end must be served on port 5500.

Data

The API reads Parquet files from the project root:

data/
├── harmonized/
│   ├── year=2000/   *.parquet
│   ├── year=2001/   *.parquet
│   └── ...
├── processed/       (optional; only used by the original per-sensor endpoints)
└── countries.csv    (list of countries for the search suggestions)

Each harmonized Parquet file has these columns: latitude, longitude, acq_date, acq_time, instrument (MODIS or VIIRS), satellite, confidence (h, n, l), daynight, country, frp, type, version and pixel_area. The notebooks data_ingestion.ipynb and armonization.ipynb prepare these files. The data files themselves are not included in the repository because of their size.

API
Endpoint	Purpose
GET /harmonized/years	Years available on disk
GET /harmonized/fires	Paginated detections
GET /harmonized/fires.csv	All detections of a query, streamed as CSV
GET /harmonized/calendar	Daily aggregates for one month
GET /harmonized/analysis	Aggregates for the charts
GET /harmonized/series	Full daily series (all years), cached in memory per set of filters
GET /countries	Country list
GET /fires, /calendar, /analysis	Original per-sensor endpoints (data/processed)

The harmonized endpoints accept country, instrument, min_confidence, fire_type and daynight. /harmonized/series also accepts a rectangular zone (south, north, west, east).

Notes and limitations
MODIS starts in November 2000 and VIIRS in 2012. VIIRS has a finer resolution and detects more fires, so raw counts jump when it joins the series. For comparisons across eras, FRP or pixel area are safer metrics than counts.
The first time /harmonized/series is requested it reads every year and can take a while; afterwards the result is kept in memory until the server restarts.
The traffic-light thresholds (1.25×, 1.5× and 2× the historical median) are constants at the top of the historical-series script in index.html and are easy to change.
Year validation in the original endpoints is currently limited to 2000–2025.
Built with
Backend: Python, FastAPI, Uvicorn, DuckDB, pandas, Parquet
Frontend: HTML, CSS and plain JavaScript, Leaflet 1.9.4, Chart.js 4.4.1
Map tiles: OpenStreetMap, CARTO, Esri World Imagery

Acknowledgments

We acknowledge the use of data and/or imagery from NASA's Fire Information for Resource Management System (FIRMS) (https://earthdata.nasa.gov/firms), part of NASA's Land, Atmosphere Near real-time Capability for Earth observations (LANCE) (https://earthdata.nasa.gov/lance) and NASA's Earth Science Data and Information System (ESDIS).

Parts of this project were developed with the assistance of Claude, an AI assistant made by Anthropic. Claude helped with converting the original Folium map to Leaflet and integrating it with the table, calendar and charts; the harmonized-data endpoints and views; the multi-year calendar comparison; the CSV and PNG exports; the historical series, anomaly and traffic-light view; and debugging and code review. The project idea, the data ingestion and harmonization, and the base API are the team's own work, and the team reviewed and tested the results.

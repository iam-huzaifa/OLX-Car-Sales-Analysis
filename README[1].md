# OLX Used Cars Dashboard (Streamlit)

This repository contains a Streamlit dashboard template to explore OLX / used cars data.
It is designed to be GitHub-deployable and easy to customize.

## How to use

1. Place your dataset as `olx_cars.csv` in the repository root **OR** use the file uploader in the sidebar to upload a CSV.
2. Run locally:
   ```
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. Deploy on Streamlit Cloud:
   - Push this repository to GitHub.
   - Connect the repo on Streamlit Cloud and deploy `app.py`.

## UI choices made
- Wide layout, sidebar for filters and file upload.
- Top KPIs line with `st.metric`.
- Responsive Plotly charts for interactivity.
- Data table with download button.

## Notes
- The app attempts to auto-detect common column names (price, brand, year, mileage, city).
- If auto-detect fails, use the "Column mapping" selectors in the sidebar to map your dataset.
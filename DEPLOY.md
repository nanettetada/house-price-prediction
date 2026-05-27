# Deploy this dashboard

## Streamlit Community Cloud (free, ~3 min)

1. Go to [streamlit.io/cloud](https://streamlit.io/cloud) → **Sign in with GitHub** (use `nanettetada`).
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `nanettetada/harare-property-prices`
   - **Branch:** `main`
   - **Main file path:** `dashboard.py`
4. **Deploy**. Build takes 2–3 minutes.
5. Paste the resulting URL into the README's live-demo badge.

## Notes

- Auto-rebuilds on every push to `main`.
- Free tier sleeps after ~7 days idle. First load wakes it.
- Alternative: Hugging Face Spaces (SDK = Streamlit, same repo).

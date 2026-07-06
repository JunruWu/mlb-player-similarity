# MLB Player Similarity Finder

A Streamlit app for finding MLB hitter player-year samples with similar batting profiles.

## Setup

Place your CSV file here:

```text
data/stats.csv
```

The app expects the CSV to include `last_name, first_name`, `player_id`, `year`, `xwoba`, and the batting feature columns listed in `app.py`.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Similarity Model

Each row is treated as one player-year sample. The app renames `last_name, first_name` to `player_name`, standardizes every model feature with `StandardScaler`, and calculates weighted z-score distance:

```text
z_diff_i = target_z_i - candidate_z_i
weighted_z_distance = sqrt(sum(weight_i * z_diff_i^2) / sum(weight_i))
avg_abs_z_diff = sum(weight_i * abs(z_diff_i)) / sum(weight_i)
similarity_score = 100 * exp(-0.5 * weighted_z_distance^2)
```

By default, `xwoba` is not included in the similarity model because it is more of an outcome metric than a style metric. You can enable it with the sidebar checkbox.

Feature weights are stored in the `FEATURE_WEIGHTS` dictionary in `app.py`, so you can adjust the model without changing the rest of the code.

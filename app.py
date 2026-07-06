from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler


APP_TITLE = "MLB Player Similarity Finder"
DATA_PATH = Path(__file__).parent / "data" / "stats.csv"

CONTACT_QUALITY_FEATURES = [
    "avg_swing_speed",
    "squared_up_contact",
    "sweet_spot_percent",
    "barrel_batted_rate",
    "solidcontact_percent",
    "flareburner_percent",
    "poorlyunder_percent",
    "poorlytopped_percent",
    "hard_hit_percent",
    "avg_best_speed",
]

PLATE_DISCIPLINE_FEATURES = [
    "k_percent",
    "bb_percent",
    "z_swing_percent",
    "oz_swing_percent",
    "oz_contact_percent",
    "iz_contact_percent",
    "edge_percent",
    "whiff_percent",
    "swing_percent",
]

BATTED_BALL_FEATURES = [
    "pull_percent",
    "opposite_percent",
    "groundballs_percent",
    "flyballs_percent",
    "linedrives_percent",
]

BASE_FEATURES = (
    CONTACT_QUALITY_FEATURES + PLATE_DISCIPLINE_FEATURES + BATTED_BALL_FEATURES
)

FEATURE_WEIGHTS = {
    # Contact Quality
    "avg_swing_speed": 0.8,
    "squared_up_contact": 1.2,
    "sweet_spot_percent": 1.0,
    "barrel_batted_rate": 1.4,
    "solidcontact_percent": 0.9,
    "flareburner_percent": 0.6,
    "poorlyunder_percent": 0.7,
    "poorlytopped_percent": 0.3,
    "hard_hit_percent": 0.5,
    "avg_best_speed": 1.0,
    # Plate Discipline
    "k_percent": 0.8,
    "bb_percent": 1.0,
    "z_swing_percent": 0.7,
    "oz_swing_percent": 1.3,
    "oz_contact_percent": 0.6,
    "iz_contact_percent": 0.4,
    "edge_percent": 0.1,
    "whiff_percent": 1.2,
    "swing_percent": 0.3,
    # Batted Ball Profile
    "pull_percent": 1.1,
    "opposite_percent": 0.9,
    "groundballs_percent": 1.0,
    "flyballs_percent": 1.0,
    "linedrives_percent": 1.2,
}

OPTIONAL_OUTCOME_WEIGHTS = {
    "xwoba": 0.5,
}

REQUIRED_COLUMNS = ["last_name, first_name", "player_id", "year", "xwoba"] + BASE_FEATURES

RESULT_COLUMNS = [
    "rank",
    "player_name",
    "year",
    "player_id",
    "similarity_score",
    "xwoba",
    "k_percent",
    "bb_percent",
    "z_swing_percent",
    "oz_swing_percent",
    "oz_contact_percent",
    "iz_contact_percent",
    "edge_percent",
    "whiff_percent",
    "swing_percent",
    "barrel_batted_rate",
    "hard_hit_percent",
    "avg_swing_speed",
    "squared_up_contact",
    "pull_percent",
    "opposite_percent",
    "groundballs_percent",
    "flyballs_percent",
    "linedrives_percent",
]

COMPARISON_FEATURES = [
    "xwoba",
    "k_percent",
    "bb_percent",
    "barrel_batted_rate",
    "hard_hit_percent",
    "avg_swing_speed",
    "squared_up_contact",
    "pull_percent",
    "opposite_percent",
    "groundballs_percent",
    "flyballs_percent",
    "linedrives_percent",
]


def get_similarity_features(include_xwoba: bool) -> list[str]:
    """Return model features with xwoba included only when requested."""
    if include_xwoba:
        return BASE_FEATURES + ["xwoba"]
    return BASE_FEATURES.copy()


@st.cache_data
def load_data(csv_path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Read the CSV and normalize the player name/feature columns."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    if "last_name, first_name" in df.columns:
        df = df.rename(columns={"last_name, first_name": "player_name"})

    numeric_columns = ["player_id", "year", "xwoba"] + BASE_FEATURES
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "player_name" in df.columns:
        df["player_name"] = df["player_name"].astype(str).str.strip()

    return df


def get_missing_columns(columns: pd.Index) -> list[str]:
    """Return required source columns that are absent after player-name cleanup."""
    normalized = set(columns)
    if "player_name" in normalized:
        normalized.add("last_name, first_name")
    return [column for column in REQUIRED_COLUMNS if column not in normalized]


def build_player_options(df: pd.DataFrame) -> pd.DataFrame:
    """Build a stable display label for each player-year sample."""
    options = df[["player_name", "player_id", "year"]].copy()
    options = options.dropna(subset=["player_name", "player_id", "year"])
    options["year"] = options["year"].astype(int)
    options["player_id"] = options["player_id"].astype(int)
    options["label"] = (
        options["player_name"] + " - " + options["year"].astype(str)
    )
    return options.sort_values(["player_name", "year"], ascending=[True, False])


def filter_player_options(options: pd.DataFrame, search_term: str) -> pd.DataFrame:
    """Filter player-year options by player name, year, or player id."""
    tokens = [token for token in search_term.strip().lower().split() if token]
    if not tokens:
        return options

    searchable = (
        options["label"].str.lower()
        + " "
        + options["player_name"].str.lower()
        + " "
        + options["year"].astype(str)
        + " "
        + options["player_id"].astype(str)
    )
    mask = pd.Series(True, index=options.index)
    for token in tokens:
        mask &= searchable.str.contains(token, regex=False)
    return options.loc[mask]


def prepare_model_matrix(
    df: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Drop unusable rows and return standardized feature values."""
    model_df = df.dropna(subset=features + ["player_name", "player_id", "year"]).copy()
    if model_df.empty:
        return model_df, np.empty((0, len(features)))

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(model_df[features])
    return model_df.reset_index(drop=True), scaled_values


def calculate_similarity(
    model_df: pd.DataFrame,
    scaled_values: np.ndarray,
    target_index: int,
    features: list[str],
    top_n: int,
    allow_same_player: bool,
) -> pd.DataFrame:
    """Calculate Top N most similar player-year samples."""
    feature_weights = FEATURE_WEIGHTS | OPTIONAL_OUTCOME_WEIGHTS
    weights = np.array([feature_weights[feature] for feature in features], dtype=float)
    target_z_values = scaled_values[target_index]
    z_diffs = target_z_values - scaled_values
    weighted_squared_z_diffs = weights * np.square(z_diffs)
    weighted_z_distances = np.sqrt(weighted_squared_z_diffs.sum(axis=1) / weights.sum())
    avg_abs_z_diffs = (weights * np.abs(z_diffs)).sum(axis=1) / weights.sum()

    results = model_df.copy()
    results["weighted_z_distance"] = weighted_z_distances
    results["avg_abs_z_diff"] = avg_abs_z_diffs
    results["similarity_score"] = 100 * np.exp(-0.5 * np.square(weighted_z_distances))

    target_row = model_df.iloc[target_index]
    mask = results.index != target_index
    if not allow_same_player:
        mask &= results["player_id"] != target_row["player_id"]

    results = results.loc[mask].sort_values(
        ["similarity_score", "player_name", "year"], ascending=[False, True, False]
    )
    results = results.head(top_n).copy()
    results.insert(0, "rank", range(1, len(results) + 1))
    results["similarity_score"] = results["similarity_score"].round(1)
    results["weighted_z_distance"] = results["weighted_z_distance"].round(3)
    results["avg_abs_z_diff"] = results["avg_abs_z_diff"].round(3)
    return results


def find_target_index(
    model_df: pd.DataFrame, player_id: int, year: int
) -> int | None:
    """Return model row index for the selected player-year sample."""
    matches = model_df.index[
        (model_df["player_id"].astype(int) == int(player_id))
        & (model_df["year"].astype(int) == int(year))
    ]
    if len(matches) == 0:
        return None
    return int(matches[0])


def build_feature_difference(
    target_row: pd.Series, comparison_row: pd.Series, features: list[str]
) -> pd.DataFrame:
    """Create a target-vs-comparison feature difference table."""
    rows = []
    for feature in features:
        target_value = target_row[feature]
        comparison_value = comparison_row[feature]
        rows.append(
            {
                "feature": feature,
                "target": target_value,
                "similar_player": comparison_value,
                "difference": comparison_value - target_value,
            }
        )
    diff_df = pd.DataFrame(rows)
    numeric_cols = ["target", "similar_player", "difference"]
    diff_df[numeric_cols] = diff_df[numeric_cols].round(3)
    return diff_df


def build_result_table_with_target(
    target_row: pd.Series, results: pd.DataFrame
) -> pd.DataFrame:
    """Add the selected player as the first visible comparison row."""
    target_display = target_row.to_frame().T.copy()
    target_display.insert(0, "rank", "")
    target_display["similarity_score"] = 100.0
    target_display["weighted_z_distance"] = 0.0
    target_display["avg_abs_z_diff"] = 0.0

    visible_results = results.copy()
    visible_results["rank"] = visible_results["rank"].astype(str)
    return pd.concat([target_display, visible_results], ignore_index=True)


def display_bar_chart(target_row: pd.Series, comparison_row: pd.Series) -> None:
    """Show a compact normalized bar chart for selected headline traits."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.info("Install plotly to enable the comparison chart.")
        return

    chart_features = [
        "barrel_batted_rate",
        "hard_hit_percent",
        "avg_swing_speed",
        "squared_up_contact",
        "k_percent",
        "bb_percent",
        "pull_percent",
        "groundballs_percent",
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name=f"{target_row['player_name']} {int(target_row['year'])}",
            x=chart_features,
            y=[target_row[feature] for feature in chart_features],
        )
    )
    fig.add_trace(
        go.Bar(
            name=f"{comparison_row['player_name']} {int(comparison_row['year'])}",
            x=chart_features,
            y=[comparison_row[feature] for feature in chart_features],
        )
    )
    fig.update_layout(
        barmode="group",
        height=430,
        margin=dict(l=20, r=20, t=35, b=90),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis_title="Raw value",
    )
    st.plotly_chart(fig, use_container_width=True)


def format_result_table(results: pd.DataFrame) -> pd.DataFrame:
    """Return only user-facing columns with clean numeric formatting."""
    visible = results[RESULT_COLUMNS].copy()
    visible["player_id"] = pd.to_numeric(visible["player_id"], errors="coerce").astype(
        int
    )
    visible["year"] = pd.to_numeric(visible["year"], errors="coerce").astype(int)
    for column in visible.columns:
        if column in ["rank", "player_name", "player_id", "year"]:
            continue
        visible[column] = pd.to_numeric(visible[column], errors="coerce")
        if column == "similarity_score":
            visible[column] = visible[column].round(1)
        else:
            visible[column] = visible[column].round(3)
    visible["player_id"] = visible["player_id"].astype(int)
    visible["year"] = visible["year"].astype(int)
    return visible


def render_missing_file_message() -> None:
    st.warning(
        "Missing CSV file. Please place your MLB hitter data at `data/stats.csv` "
        "and rerun the app."
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    if not DATA_PATH.exists():
        render_missing_file_message()
        st.stop()

    try:
        df = load_data(DATA_PATH)
    except Exception as exc:
        st.error(f"Could not read `data/stats.csv`: {exc}")
        st.stop()

    missing_columns = get_missing_columns(df.columns)
    if missing_columns:
        st.error("The CSV is missing required fields:")
        st.write(missing_columns)
        st.stop()

    player_options = build_player_options(df)
    if player_options.empty:
        st.warning("No player-year rows are available after loading the CSV.")
        st.stop()

    st.sidebar.header("Controls")
    search_term = st.sidebar.text_input(
        "Search player, year, or player ID",
        placeholder="Aaron Judge, 2025, 592450",
    )
    filtered_options = filter_player_options(player_options, search_term)
    if filtered_options.empty:
        st.warning("No player-year options match your search.")
        st.stop()

    selected_label = st.sidebar.selectbox(
        "Player-season",
        options=filtered_options["label"].tolist(),
        index=0,
    )
    include_xwoba = st.sidebar.checkbox(
        "Include xwoba in similarity model", value=False
    )
    allow_same_player = st.sidebar.checkbox(
        "Allow same player from other seasons", value=False
    )
    top_n = st.sidebar.slider("Top N", min_value=5, max_value=20, value=10)

    selected_option = filtered_options.loc[
        filtered_options["label"] == selected_label
    ].iloc[0]
    features = get_similarity_features(include_xwoba)

    model_df, scaled_values = prepare_model_matrix(df, features)
    if model_df.empty:
        st.warning("No rows have complete data for the selected model features.")
        st.stop()

    target_index = find_target_index(
        model_df,
        int(selected_option["player_id"]),
        int(selected_option["year"]),
    )

    if target_index is None:
        st.warning(
            "The selected player-year has missing model features, so it cannot be "
            "used for similarity search."
        )
        st.stop()

    target_row = model_df.iloc[target_index]
    results = calculate_similarity(
        model_df=model_df,
        scaled_values=scaled_values,
        target_index=target_index,
        features=features,
        top_n=top_n,
        allow_same_player=allow_same_player,
    )

    if results.empty:
        st.warning("No similar players found with the current filters.")
        st.stop()

    st.subheader("Selected Player")
    col1, col2, col3 = st.columns(3)
    col1.metric("Player", str(target_row["player_name"]))
    col2.metric("Player ID", int(target_row["player_id"]))
    col3.metric("Year", int(target_row["year"]))

    st.subheader(f"Selected Player + Top {len(results)} Similar Players")
    table_with_target = build_result_table_with_target(target_row, results)
    st.dataframe(
        format_result_table(table_with_target),
        use_container_width=True,
        hide_index=True,
    )

    comparison_labels = (
        results["rank"].astype(str)
        + ". "
        + results["player_name"]
        + " - "
        + results["year"].astype(int).astype(str)
    )
    selected_comparison_label = st.selectbox(
        "Compare feature differences against",
        options=comparison_labels.tolist(),
    )
    selected_rank = int(selected_comparison_label.split(".", 1)[0])
    comparison_row = results.loc[results["rank"] == selected_rank].iloc[0]

    st.subheader("Feature Differences")
    diff_df = build_feature_difference(target_row, comparison_row, COMPARISON_FEATURES)
    st.dataframe(diff_df, use_container_width=True, hide_index=True)

    st.subheader("Key Feature Comparison")
    display_bar_chart(target_row, comparison_row)


if __name__ == "__main__":
    main()

    

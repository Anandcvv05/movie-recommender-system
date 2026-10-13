
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #10121c 0%, #171426 100%);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    .hero {
        padding: 30px;
        border-radius: 22px;
        background: linear-gradient(115deg, #32204f, #192b49);
        border: 1px solid #51416e;
        margin-bottom: 25px;
    }

    .hero h1 {
        font-size: 42px;
        margin-bottom: 8px;
        color: #ffffff;
    }

    .hero p {
        font-size: 16px;
        color: #ded8f5;
    }

    .movie-card {
        background: #201d30;
        border: 1px solid #3b3452;
        border-radius: 15px;
        padding: 18px;
        margin-bottom: 12px;
        min-height: 150px;
    }

    .movie-title {
        color: #ffffff;
        font-size: 17px;
        font-weight: 700;
    }

    .movie-genre {
        color: #c3b8e5;
        font-size: 13px;
        margin-top: 8px;
    }

    .movie-score {
        color: #c4b5fd;
        font-weight: 700;
        margin-top: 12px;
    }

    div[data-testid="stMetric"] {
        background: #201d30;
        border: 1px solid #39344d;
        padding: 16px;
        border-radius: 14px;
    }

    .section-caption {
        color: #bdb6d5;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data(show_spinner=False)
def load_data():
    ratings = pd.read_csv(BASE_DIR / "ratings.csv")
    movies = pd.read_csv(BASE_DIR / "movies.csv")

    required_ratings = {"userId", "movieId", "rating"}
    required_movies = {"movieId", "title", "genres"}

    if not required_ratings.issubset(ratings.columns):
        raise ValueError(
            "ratings.csv must contain userId, movieId and rating."
        )

    if not required_movies.issubset(movies.columns):
        raise ValueError(
            "movies.csv must contain movieId, title and genres."
        )

    ratings = ratings.dropna(
        subset=["userId", "movieId", "rating"]
    ).copy()

    movies = movies.dropna(
        subset=["movieId", "title"]
    ).copy()

    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    movies["movieId"] = movies["movieId"].astype(int)

    movies["genres"] = movies["genres"].fillna(
        "(no genres listed)"
    )

    return ratings, movies


try:
    ratings, movies = load_data()

    if ratings.empty or movies.empty:
        st.error("The dataset is empty. Check your CSV files.")
        st.stop()

except Exception as exc:
    st.error(f"Unable to load the dataset: {exc}")
    st.info(
        "Keep ratings.csv and movies.csv in the same folder as app.py."
    )
    st.stop()


# =====================================================
# BUILD USER SIMILARITY MODEL
# =====================================================
@st.cache_data(show_spinner="Preparing recommendation model...")
def build_model(ratings):
    rating_matrix = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    )

    user_means = rating_matrix.mean(axis=1)

    centered_matrix = rating_matrix.sub(
        user_means, axis=0
    ).fillna(0)

    similarity_values = cosine_similarity(centered_matrix)

    similarity_df = pd.DataFrame(
        similarity_values,
        index=rating_matrix.index,
        columns=rating_matrix.index
    )

    return rating_matrix, user_means, similarity_df


matrix, user_means, similarity_df = build_model(ratings)

user_ids = sorted(matrix.index.tolist())

genre_options = sorted({
    genre
    for value in movies["genres"]
    for genre in str(value).split("|")
    if genre and genre != "(no genres listed)"
})


# =====================================================
# RECOMMENDATION ENGINE
# =====================================================
def recommend_movies(
    user_id,
    top_n=10,
    neighbor_count=20,
    selected_genres=None,
    min_rating=3.0
):
    if user_id not in matrix.index:
        return pd.DataFrame()

    known_ratings = matrix.loc[user_id].dropna()

    if known_ratings.empty:
        return pd.DataFrame()

    similarities = similarity_df.loc[user_id].drop(
        index=user_id, errors="ignore"
    )

    neighbors = similarities.sort_values(
        ascending=False
    ).head(neighbor_count)

    # Use only positively similar users
    neighbors = neighbors[neighbors > 0]

    if neighbors.empty:
        return pd.DataFrame()

    neighbor_matrix = matrix.loc[neighbors.index]
    neighbor_means = user_means.loc[neighbors.index]

    # Center each neighbor's ratings around their personal average
    deviations = neighbor_matrix.sub(
        neighbor_means, axis=0
    ).fillna(0)

    observed = neighbor_matrix.notna().astype(float)

    numerator = deviations.T.dot(neighbors)

    denominator = observed.T.dot(neighbors.abs())

    # Predict using the target user's average plus weighted deviations
    predicted = (
        user_means.loc[user_id]
        + numerator / denominator.replace(0, np.nan)
    )

    # Exclude movies already rated by this user
    predicted = predicted.drop(
        index=known_ratings.index,
        errors="ignore"
    )

    predicted = predicted.dropna().clip(0.5, 5.0)

    result = predicted.rename(
        "Predicted Rating"
    ).reset_index()

    result.columns = ["movieId", "Predicted Rating"]

    result = result.merge(
        movies[["movieId", "title", "genres"]],
        on="movieId",
        how="inner"
    )

    if selected_genres:
        result = result[
            result["genres"].apply(
                lambda value: any(
                    genre in str(value).split("|")
                    for genre in selected_genres
                )
            )
        ]

    result = result[
        result["Predicted Rating"] >= min_rating
    ]

    return (
        result.sort_values(
            "Predicted Rating", ascending=False
        )
        .head(top_n)
        .reset_index(drop=True)
    )


# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("## 🎬 CineMatch")
    st.caption("Your personal movie discovery space")
    st.divider()

    selected_user = st.selectbox(
        "👤 Select user profile",
        user_ids,
        format_func=lambda uid: f"User {uid}"
    )

    # Number input with increment/decrement arrows
    top_n = int(st.number_input(
        "Number of recommendations",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
        help="Choose how many movies to recommend."
    ))

    # Number input with increment/decrement arrows
    neighbor_count = int(st.number_input(
        "Similar users to consider",
        min_value=5,
        max_value=50,
        value=20,
        step=1,
        help="Choose how many similar users influence predictions."
    ))

    st.divider()
    st.markdown("### 🎭 Genre preferences")

    selected_genres = st.multiselect(
        "Filter recommendations",
        options=genre_options,
        placeholder="All genres"
    )

    min_rating = st.select_slider(
        "Minimum predicted rating",
        options=[0.5, 1.0, 1.5, 2.0, 2.5,
                 3.0, 3.5, 4.0, 4.5, 5.0],
        value=3.0
    )

    st.caption(
        "Filters affect displayed recommendations, not model training."
    )


# =====================================================
# HERO HEADER
# =====================================================
st.markdown("""
<div class="hero">
    <h1>🎬 CineMatch</h1>
    <p>Discover your next favourite movie through collaborative filtering.</p>
    <p>Explore recommendations based on users with similar rating patterns.</p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# DATASET METRICS
# =====================================================
c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "🎞️ Movies",
    f"{movies['movieId'].nunique():,}"
)

c2.metric(
    "⭐ Ratings",
    f"{len(ratings):,}"
)

c3.metric(
    "👥 Users",
    f"{ratings['userId'].nunique():,}"
)

c4.metric(
    "📊 Average rating",
    f"{ratings['rating'].mean():.2f}/5"
)

st.write("")


# =====================================================
# NAVIGATION TABS
# =====================================================
tab_recommend, tab_discover, tab_history, tab_insights = st.tabs([
    "✨ For You",
    "🔎 Discover Movies",
    "📚 My Rating History",
    "📈 Insights"
])


# =====================================================
# TAB 1: PERSONALIZED RECOMMENDATIONS
# =====================================================
with tab_recommend:
    st.subheader(f"Recommended for User {selected_user}")

    st.markdown(
        '<p class="section-caption">'
        'Recommendations are estimates based on similar users, '
        'not guaranteed preferences.'
        '</p>',
        unsafe_allow_html=True
    )

    if st.button(
        "✨ Generate Recommendations",
        type="primary",
        key="generate_recommendations"
    ):
        results = recommend_movies(
            user_id=selected_user,
            top_n=top_n,
            neighbor_count=neighbor_count,
            selected_genres=selected_genres,
            min_rating=min_rating
        )

        st.session_state["recommendation_results"] = results
        st.session_state["recommendation_user"] = selected_user

    if (
        "recommendation_results" in st.session_state
        and st.session_state.get("recommendation_user") == selected_user
    ):
        results = st.session_state["recommendation_results"]

        if results.empty:
            st.warning(
                "No recommendations match these settings. "
                "Try removing genre filters or lowering the minimum rating."
            )
        else:
            st.success(
                f"Found {len(results)} movie recommendations."
            )

            for start in range(0, len(results), 2):
                cols = st.columns(2)

                for col, (_, movie) in zip(
                    cols,
                    results.iloc[start:start + 2].iterrows()
                ):
                    with col:
                        st.markdown(
                            f"""
                            <div class="movie-card">
                                <div class="movie-title">
                                    🎥 {movie['title']}
                                </div>
                                <div class="movie-genre">
                                    {movie['genres']}
                                </div>
                                <div class="movie-score">
                                    ⭐ Predicted rating:
                                    {movie['Predicted Rating']:.2f}/5
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            st.download_button(
                "⬇️ Download recommendations as CSV",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name=f"recommendations_user_{selected_user}.csv",
                mime="text/csv"
            )

            with st.expander("View recommendations as a table"):
                st.dataframe(
                    results,
                    hide_index=True,
                    use_container_width=True
                )
    else:
        st.info(
            "Choose your preferences in the sidebar and click "
            "**Generate Recommendations** to get started."
        )


# =====================================================
# TAB 2: DISCOVER MOVIES
# =====================================================
with tab_discover:
    st.subheader("Explore the MovieLens library")

    search_text = st.text_input(
        "🔎 Search movie titles",
        placeholder="Example: Toy Story, Matrix, Titanic..."
    )

    discover_genres = st.multiselect(
        "Filter by genre",
        options=genre_options,
        key="discover_genres"
    )

    catalog = movies.copy()

    if search_text.strip():
        catalog = catalog[
            catalog["title"].str.contains(
                search_text.strip(),
                case=False,
                na=False,
                regex=False
            )
        ]

    if discover_genres:
        catalog = catalog[
            catalog["genres"].apply(
                lambda value: any(
                    genre in str(value).split("|")
                    for genre in discover_genres
                )
            )
        ]

    st.caption(f"{len(catalog):,} matching movies")

    if catalog.empty:
        st.warning(
            "No movies match your search. Try another title or genre."
        )
    else:
        stats = ratings.groupby("movieId").agg(
            Average_Rating=("rating", "mean"),
            Number_of_Ratings=("rating", "count")
        ).reset_index()

        catalog = catalog.merge(
            stats,
            on="movieId",
            how="left"
        )

        catalog["Average_Rating"] = catalog[
            "Average_Rating"
        ].round(2)

        catalog["Number_of_Ratings"] = catalog[
            "Number_of_Ratings"
        ].fillna(0).astype(int)

        st.dataframe(
            catalog[[
                "title",
                "genres",
                "Average_Rating",
                "Number_of_Ratings"
            ]].rename(columns={
                "title": "Movie",
                "genres": "Genres",
                "Average_Rating": "Average rating",
                "Number_of_Ratings": "Rating count"
            }),
            use_container_width=True,
            hide_index=True
        )


# =====================================================
# TAB 3: USER RATING HISTORY
# =====================================================
with tab_history:
    st.subheader(f"Rating history for User {selected_user}")

    history = ratings[
        ratings["userId"] == selected_user
    ].merge(
        movies[["movieId", "title", "genres"]],
        on="movieId",
        how="left"
    )

    if history.empty:
        st.info("No rating history is available.")
    else:
        hc1, hc2, hc3 = st.columns(3)

        hc1.metric("Movies rated", len(history))

        hc2.metric(
            "Average rating given",
            f"{history['rating'].mean():.2f}"
        )

        hc3.metric(
            "Highly rated movies",
            int((history["rating"] >= 4).sum())
        )

        history = history.sort_values(
            "rating", ascending=False
        )

        st.dataframe(
            history[["title", "genres", "rating"]].rename(
                columns={
                    "title": "Movie",
                    "genres": "Genres",
                    "rating": "Your rating"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download rating history",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name=f"rating_history_user_{selected_user}.csv",
            mime="text/csv"
        )


# =====================================================
# TAB 4: DATASET INSIGHTS
# =====================================================
with tab_insights:
    st.subheader("Dataset insights")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Rating distribution")

        fig, ax = plt.subplots(figsize=(7, 4))

        ratings["rating"].value_counts().sort_index().plot(
            kind="bar",
            ax=ax
        )

        ax.set_xlabel("Rating")
        ax.set_ylabel("Number of ratings")
        ax.set_title("Distribution of MovieLens Ratings")
        ax.grid(axis="y", alpha=0.2)

        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown("#### Most-rated movies")

        popular = (
            ratings.groupby("movieId")
            .agg(
                rating_count=("rating", "count"),
                average_rating=("rating", "mean")
            )
            .reset_index()
            .merge(
                movies[["movieId", "title"]],
                on="movieId",
                how="left"
            )
            .sort_values("rating_count", ascending=False)
            .head(10)
        )

        popular["average_rating"] = popular[
            "average_rating"
        ].round(2)

        st.dataframe(
            popular[[
                "title", "rating_count", "average_rating"
            ]].rename(columns={
                "title": "Movie",
                "rating_count": "Number of ratings",
                "average_rating": "Average rating"
            }),
            hide_index=True,
            use_container_width=True
        )

    st.markdown("#### Recommendation methodology")

    st.write(
        "The system centers user ratings around personal averages, "
        "computes cosine similarity between users, selects positively "
        "similar neighbors, and predicts unseen ratings using "
        "similarity-weighted rating deviations."
    )

    st.caption(
        "Dataset: MovieLens Small. Statistics are calculated from the "
        "local CSV files. Movie posters and external metadata are not fetched."
    )


# =====================================================
# FOOTER
# =====================================================
st.divider()

st.caption(
    "CineMatch • Movie Recommender System • "
    "Python, Pandas, Scikit-learn and Streamlit"
)

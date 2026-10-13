# 🎬 CineMatch — Movie Recommender System

### Discover movies you'll love with Collaborative Filtering

CineMatch is an interactive movie recommendation web application built using **Python, Streamlit, Pandas, NumPy, and Scikit-learn**. It uses **User-Based Collaborative Filtering** to recommend movies by identifying users with similar rating patterns and estimating which unseen movies a user may enjoy.

The application provides personalized recommendations, movie search, genre filtering, rating history, and dataset analytics through a modern, user-friendly dashboard.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Objectives](#-objectives)
- [Features](#-features)
- [Dataset](#-dataset)
- [How It Works](#-how-it-works)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation and Setup](#-installation-and-setup)
- [Run the Application](#-run-the-application)
- [How to Use](#-how-to-use)
- [Recommendation Algorithm](#-recommendation-algorithm)
- [Evaluation](#-evaluation)
- [Limitations](#-limitations)
- [Future Enhancements](#-future-enhancements)
- [Dataset Citation](#-dataset-citation)
- [License and Dataset Usage](#-license-and-dataset-usage)
- [Author](#-author)

---

## 📖 Overview

Movie recommendation systems help users discover movies that match their interests and preferences.

CineMatch uses historical user ratings from the MovieLens dataset to identify similarities between users. It then predicts ratings for movies that a selected user has not rated and recommends the highest-scoring candidates.

Unlike a system that recommends movies only because they belong to a popular genre, CineMatch uses collaborative filtering to learn from the preferences expressed through rating patterns.

## 🎯 Objectives

- Implement a movie recommendation system using collaborative filtering.
- Analyze user–movie rating data.
- Calculate similarities between users using cosine similarity.
- Generate personalized Top-N movie recommendations.
- Explore movie genres, rating histories, and dataset statistics.
- Develop an interactive web interface using Streamlit.
- Demonstrate the application of machine learning techniques to a real-world recommendation problem.

## ✨ Features

### 1. Personalized Movie Recommendations
- Select a MovieLens user profile.
- Generate recommendations based on similar users.
- Configure the number of recommendations.
- Adjust the number of similar users considered.
- Filter results by genre and minimum predicted rating.

### 2. Movie Discovery
- Search movies by title.
- Filter movies by genre.
- View movie genres, average ratings, and rating counts.
- Explore the MovieLens movie catalog.

### 3. User Rating History
- View movies previously rated by a selected user.
- Inspect individual ratings and genres.
- View the user's average rating and number of highly rated movies.
- Download rating history as a CSV file.

### 4. Dataset Analytics
- View the number of users, movies, and ratings.
- Explore the rating distribution.
- Identify the most-rated movies.
- Examine basic statistics about the dataset.

### 5. Data Export
- Download generated recommendations as CSV.
- Export a selected user's rating history.

### 6. Interactive Web Interface
- Modern dark-themed dashboard.
- Sidebar controls and organized navigation tabs.
- Recommendation cards and data tables.
- Interactive charts and expandable information sections.

---

## 📊 Dataset

CineMatch uses the **MovieLens Latest Small Dataset** published by GroupLens Research at the University of Minnesota.

**Official dataset page:**  
https://grouplens.org/datasets/movielens/latest/

The Small Dataset contains approximately:

| Attribute | Description |
|---|---|
| Users | 610 |
| Movies | 9,742 |
| Ratings | 100,836 |
| Rating scale | 0.5 to 5.0 |
| Movie metadata | Titles and genres |

The dataset statistics above describe the MovieLens Small Dataset. The application displays statistics calculated from the CSV files actually loaded.

### Files Used

**`ratings.csv`**

Contains user rating records.

| Column | Description |
|---|---|
| `userId` | Anonymized user identifier |
| `movieId` | Movie identifier |
| `rating` | Rating given by the user |
| `timestamp` | Time when the rating was recorded |

**`movies.csv`**

Contains movie metadata.

| Column | Description |
|---|---|
| `movieId` | Movie identifier |
| `title` | Movie title and release year, when available |
| `genres` | Pipe-separated movie genres |

The `tags.csv` and `links.csv` files are not required by the current implementation.

---

## ⚙️ How It Works

CineMatch follows these steps:

1. **Data loading:** Read movie information and user ratings from local CSV files.
2. **Rating matrix construction:** Build a user–movie matrix in which rows represent users, columns represent movies, and cells contain ratings.
3. **Rating centering:** Subtract each user's average rating from their observed ratings to account for differences in individual rating habits.
4. **Similarity calculation:** Apply cosine similarity to the centered user-rating vectors.
5. **Neighbor selection:** Identify the most similar users to the selected user.
6. **Rating prediction:** Estimate ratings for unseen movies using similarity-weighted rating deviations.
7. **Recommendation generation:** Exclude movies already rated by the selected user and rank the remaining candidates.
8. **Filtering and display:** Apply the selected genre and predicted-rating filters, then display the recommendations.

### System Architecture

```text
           MovieLens Dataset
                   |
          +--------+--------+
          |                 |
      ratings.csv        movies.csv
          |                 |
          v                 |
    User-Movie Matrix        |
          |                 |
          v                 |
    Rating Centering         |
          |                 |
          v                 |
    Cosine Similarity        |
          |                 |
          v                 |
    Similar User Selection   |
          |                 |
          v                 |
    Predicted Ratings <------+
          |
          v
    Top-N Recommendations
          |
          v
    Genre and Rating Filters
          |
          v
    Streamlit Dashboard
```

---

## 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Dataset loading, transformation, and analysis |
| NumPy | Numerical computations |
| Scikit-learn | Cosine similarity calculation |
| Matplotlib | Data visualization |
| Git and GitHub | Version control and source-code hosting |
| Streamlit Community Cloud | Optional application deployment |

---

## 📁 Project Structure

```text
movie_recommender_project/
│
├── app.py
│   └── Streamlit application and recommendation logic
│
├── movies.csv
│   └── Movie titles and genres
│
├── ratings.csv
│   └── User movie ratings
│
├── requirements.txt
│   └── Required Python packages
│
├── README.md
│   └── Project documentation
│
└── test_app.py
    └── Optional temporary Streamlit test application
```

The `test_app.py` file is only needed for testing Streamlit independently. It is not part of the core recommendation algorithm.

---

## 💻 Installation and Setup

### Prerequisites

- Python 3.10 or a compatible Python version.
- pip package installer.
- Git, if you want to clone the repository.
- The MovieLens `ratings.csv` and `movies.csv` files.

### Step 1: Clone the Repository

Replace the placeholder URL with your actual GitHub repository URL.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd movie_recommender_project
```

Alternatively, download the repository as a ZIP file and extract it.

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```cmd
venv\Scripts\activate.bat
```

### Step 3: Install Dependencies

Create or update `requirements.txt` with:

```text
streamlit
pandas
numpy
scikit-learn
matplotlib
```

Install the packages:

```bash
python -m pip install -r requirements.txt
```

### Step 4: Add the Dataset

Place `ratings.csv` and `movies.csv` in the same directory as `app.py`.

The application reads these files locally and does not need to download them at startup.

---

## ▶️ Run the Application

From the project directory, execute:

```bash
python -m streamlit run app.py
```

Streamlit will display a local URL, usually:

http://localhost:8501

Open that URL in your browser to use CineMatch.

To stop the server, press `Ctrl + C` in the terminal.

---

## 🖥️ How to Use

1. Launch CineMatch in your browser.
2. Select a user profile from the sidebar.
3. Choose the desired number of recommendations.
4. Set the number of similar users to consider.
5. Optionally select preferred genres and a minimum predicted rating.
6. Open the **For You** tab and click **Generate Recommendations**.
7. Review the recommended movie titles, genres, and predicted ratings.
8. Use **Discover Movies** to search the catalog.
9. Use **My Rating History** to inspect the selected user's previous ratings.
10. Open **Insights** to explore rating distributions and popular movies.
11. Download recommendations or rating history as CSV files when needed.

---

## 🧠 Recommendation Algorithm

### User-Based Collaborative Filtering

User-Based Collaborative Filtering assumes that users who have expressed similar preferences in the past may also enjoy similar movies in the future.

### Cosine Similarity

Cosine similarity measures the similarity between two user-rating vectors.

\[
\operatorname{sim}(u,v)=
\frac{\mathbf{r}_u\cdot\mathbf{r}_v}
{\|\mathbf{r}_u\|\|\mathbf{r}_v\|}
\]

Here:

- \(u\) and \(v\) represent two users.
- \(\mathbf{r}_u\) and \(\mathbf{r}_v\) represent their rating vectors after centering around their average ratings.
- A higher positive similarity indicates more similar rating patterns.

### Rating Prediction

The implementation estimates a target user's rating for an unseen movie using the target user's average rating and the weighted rating deviations of similar users.

\[
\hat r_{u,i}=
\bar r_u+
\frac{\sum_{v\in N(u)}s(u,v)(r_{v,i}-\bar r_v)}
{\sum_{v\in N(u):\,i\text{ rated}}|s(u,v)|}
\]

Where:

- \(\hat r_{u,i}\) is the predicted rating for user \(u\) and movie \(i\).
- \(\bar r_u\) is the target user's average rating.
- \(\bar r_v\) is a neighboring user's average rating.
- \(s(u,v)\) is the cosine similarity between the users.
- \(N(u)\) is the selected neighborhood of similar users.

The implementation considers positively similar neighbors and uses observed ratings when calculating each movie's prediction.

Predicted ratings are clipped to the MovieLens rating range of 0.5 to 5.0.

---

## 📈 Evaluation

The current application provides interactive recommendations and descriptive dataset analytics.

**Important:** RMSE, MAE, Precision@K, and Recall@K have not been implemented as part of the current application interface. They should not be reported as measured results until the model has been evaluated.

For a rigorous evaluation, the next steps are:

- Split ratings into training and testing sets.
- Build the similarity matrix using training data only.
- Predict held-out ratings.
- Calculate Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).
- Evaluate recommendation rankings using Precision@K and Recall@K.
- Compare collaborative filtering against a popularity-based baseline.

A proper evaluation must avoid using test ratings when building the recommendation model.

---

## ⚠️ Limitations

- **Cold-start users:** New users without historical ratings cannot receive fully personalized recommendations.
- **Cold-start movies:** Movies with little or no rating information may not be recommended effectively.
- **Sparse ratings:** Most users rate only a small portion of the available movies.
- **Dataset age:** The MovieLens Small Dataset describes historical ratings rather than current viewing trends.
- **Similarity assumptions:** Similarity in past ratings does not guarantee similar future preferences.
- **Memory usage:** A full user-to-user similarity matrix requires memory proportional to the square of the number of users.
- **No live movie metadata:** The application does not currently fetch movie posters or current information from external movie services.

---

## 🚀 Future Enhancements

Potential improvements include:

- Item-Based Collaborative Filtering.
- Matrix factorization using Singular Value Decomposition (SVD).
- Hybrid recommendations combining collaborative filtering and movie genres.
- RMSE, MAE, Precision@K, and Recall@K evaluation.
- Movie posters and richer metadata integration.
- User accounts and saved preferences.
- Persistent storage for ratings and recommendations.
- Automated model training and performance monitoring.
- Deployment with a shareable public URL.

---

## 📚 Dataset Citation

If you use the MovieLens dataset in your academic work, cite:

Harper, F. Maxwell, and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems, 5(4), Article 19.

https://doi.org/10.1145/2827872

Dataset source: https://grouplens.org/datasets/movielens/latest/

## 📄 License and Dataset Usage

The project source code may be distributed under a license you choose for your own work. The MovieLens dataset has separate usage conditions specified by GroupLens.

Consult the dataset's accompanying README and terms before redistributing the data or using it commercially. A public GitHub repository containing the dataset should respect those conditions.

## 👨‍💻 Author

**CVV ANAND**

B.Tech — Artificial Intelligence and Machine Learning

JAIN (Deemed-to-be University), Bengaluru

*Project: Movie Recommender System using Collaborative Filtering*

---

## ⭐ Acknowledgements

- GroupLens Research, University of Minnesota, for providing the MovieLens dataset.
- The Python open-source ecosystem, including Pandas, NumPy, Scikit-learn, Matplotlib, and Streamlit.

If you find this project useful, consider starring the repository on GitHub.
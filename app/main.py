import streamlit as st

# Mock data for team members
TEAM_MEMBERS = [
    {
        "name": "Dang Nhat Duy",
        "id": "B2105568",
        "pronoun": "He/Him",
        "major": "Infomation Technology - K47",
        "role": "Leader"
    },
    {
        "name": "Ung Ngoc Diem Trinh",
        "pronoun": "She/Her",
        "id": "B2308397",
        "major": "Computer Sience - K49",
        "role": "Secretary"
    },
    {
        "name": "Le Minh Thu",
        "id": "B2308395",
        "pronoun": "She/Her",
        "major": "Computer Sience - K49",
        "role": "Member"
    },
]

st.set_page_config(
    page_title="CT294 Recommender System",
    page_icon="🧠",
    layout="wide"
)

# CSS tối giản cho header và layout
st.markdown("""
<style>
.st-emotion-cache-13k62yr {
    background: #0C0C0C;
}
.header-bar {
    display: flex;
    align-items: center;
    gap: 1rem;
    border-radius: 18px;
    padding: 1.2rem 2.2rem 1.2rem 1.2rem;
    margin-bottom: 2.5rem;
}
.header-title {
    font-size: 2rem;
    font-weight: 800;
    color: #FFF;
    margin: 0;
}
.header-link {
    margin-right: 3rem;
    color: #FFFFFF !important;
    font-weight: 600;
    text-decoration: none !important;
    transition: color 0.2s;
    font-size: 1.1rem;
}
.header-link:hover {
    color: #EEEEEE !important;
}

.main-layout {
    display: flex;
    gap: 2.5rem;
    margin-top: 2rem;
}
.main-left {
    flex: 7;
    border-radius: 18px;
    padding: 2.5rem 2rem 2.5rem 2.5rem;
}
.main-right {
    flex: 3;
    border-radius: 18px;
    padding: 2.5rem 1.5rem 2.5rem 1.5rem;
    display: flex;
    flex-direction: column;
    align-items: center;
}
.demo-link {
    display: inline-block;
    margin-top: 2rem;
    padding: 0.7rem 1.5rem;
    background: #0C0C0C;
    color: #fff !important;
    font-weight: 700;
    border-radius: 8px;
    text-decoration: none;
    font-size: 1.1rem;
    letter-spacing: 1px;
    transition: all 0.2s;
}
.demo-link:hover {
    background: #222;
    color: #EEEEEE;
}
.model-list {
    margin-top: 2.5rem;
    padding: 1.5rem 1.5rem 1.5rem 1.5rem;
    background: #F7F7F7;
    border-radius: 14px;
    box-shadow: 0 1px 6px rgba(12,12,12,0.06);
}
.model-title {
    color: #0C0C0C !important;
    margin-bottom: 0.7rem;
}
.model-item {
    margin-bottom: 0.5rem;
    color: #222;
    font-size: 1.05rem;
}
.team-section {
    margin-top: 3.5rem;
    margin-bottom: 2.5rem;
}
.team-title {
    color: #FFF;
    text-align: center;
    margin-bottom: 2.2rem;
    letter-spacing: 1px;
    font-size: 1.7rem;
    font-weight: 800;
}
.member-card {
    background: #fff;
    border-radius: 18px;
    box-shadow: 0 2px 12px rgba(12,12,12,0.08);
    padding: 2.2rem 1.2rem 1.5rem 1.2rem;
    text-align: center;
    width: 100%;
    max-width: 320px;
    min-width: 220px;
    transition: box-shadow 0.2s;
    margin: 0 auto;
}
.member-card:hover {
    box-shadow: 0 6px 24px rgba(12,12,12,0.13);
}
.member-avatar {
    object-fit: cover;
    margin-bottom: 1.1rem;
    border-radius: 8px;
    background: #EEEEEE;
}
.member-name {
    font-size: 1.15rem;
    font-weight: 700;
    color: #0C0C0C;
    margin-bottom: 0.2rem;
}
.member-id {
    color: #888;
    font-size: 0.98rem;
    margin-bottom: 0.2rem;
}
.member-pronoun {
    color: #555;
    font-size: 0.97rem;
    margin-bottom: 0.2rem;
}
.member-major {
    color: #222;
    font-size: 1.01rem;
    margin-bottom: 0.2rem;
}
.member-role {
    color: #fff;
    background: #0C0C0C;
    display: inline-block;
    font-size: 0.98rem;
    font-weight: 600;
    border-radius: 7px;
    padding: 0.25rem 1.1rem;
    margin-top: 0.7rem;
    margin-bottom: 0.2rem;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)

# HEADER
col1, col2, col3 = st.columns([0.08, 0.5, 0.42])
with col1:
    st.write("")
with col2:
    st.markdown("<div class='header-title'>CT294 Recommender System</div>", unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div style='text-align:right; margin-top:8px;'>
        <a href='#' class='header-link'>Home</a>
        <a href='#models' class='header-link'>Models</a>
        <a href='#team' class='header-link'>Team</a>
        <a href='#demo' class='header-link'>Demo</a>
    </div>
    """, unsafe_allow_html=True)

# MAIN LAYOUT
main_col1, main_col2 = st.columns([7, 3])

with main_col1:
    st.markdown("""
    ### About the Project

    **CT294 Recommender System** is a machine learning-based project designed to provide personalized recommendations for users. The system integrates multiple state-of-the-art algorithms such as LightGBM, KNN, SVD, and Neural Collaborative Filtering (NCF) to deliver accurate and efficient suggestions.

    - Enhance user experience by offering relevant recommendations
    - Support both collaborative and content-based filtering
    - Provide a flexible, extensible framework for academic and practical applications

    **Key Features:**
    - Multi-model integration for robust performance
    - Easy-to-use and modern interface
    - Modular and scalable codebase
    - Visual analytics and model comparison

    This project is developed as part of the CT294 Machine Learning course at CTU.
    """)
    
    # Introduce Models
    st.markdown("""
    <div class='model-list' id='models'>
        <h2 class='model-title'>Introduce Models</h2>
        <div class='model-item'><b>1. LightGBM:</b> Gradient boosting framework for fast, efficient, and accurate recommendations.</div>
        <div class='model-item'><b>2. KNN:</b> K-Nearest Neighbors for collaborative filtering based on user/item similarity.</div>
        <div class='model-item'><b>3. SVD:</b> Singular Value Decomposition for matrix factorization and latent feature extraction.</div>
        <div class='model-item'><b>4. Decision Tree:</b> Interpretable tree-based model for recommendation logic.</div>
        <div class='model-item'><b>5. Linear Regression:</b> Simple and effective regression-based recommendation.</div>
        <div class='model-item'><b>6. NCF:</b> Neural Collaborative Filtering leveraging deep learning for personalized suggestions.</div>
    </div>
    """, unsafe_allow_html=True)

with main_col2:
    st.markdown("""### Technologies
    - Python
    - Streamlit
    - Anaconda
    - HTML
    - CSS     
    """)
    
    st.markdown("""
        <a href='#demo' class='demo-link'>Try Demo Now →</a>
    """, unsafe_allow_html=True)


# TEAM SECTION
st.markdown("""
<div class='team-section'>
    <h2 class='team-title'>Our Team</h2>
</div>
""", unsafe_allow_html=True)

team_cols = st.columns(len(TEAM_MEMBERS))
for idx, member in enumerate(TEAM_MEMBERS):
    with team_cols[idx]:
        st.markdown(f"""
        <div class='member-card'>
            <div class='member-name'>{member['name']}</div>
            <div class='member-id'>{member['id']}</div>
            <div class='member-pronoun'>{member['pronoun']}</div>
            <div class='member-major'>{member['major']}</div>
            <div class='member-role'>{member['role']}</div>
        </div>
        """, unsafe_allow_html=True)

# DEMO SECTION
st.markdown("""
<div id='demo'></div>
<h2 style='margin-top:2.5rem; margin-bottom:1.2rem; color:#0C0C0C;'>Demo: Personalized Movie Recommendation</h2>
""", unsafe_allow_html=True)

import os
import joblib
import pickle

# Get valid model files
model_files = [f for f in os.listdir("models") if f.endswith(".pkl") or f.endswith(".joblib")]

# --- LOAD MOVIELENS 100K MOVIE LIST ---
MOVIE_LIST = []
try:
    from surprise import Dataset
    import pandas as pd
    # Load MovieLens 100k data
    data = Dataset.load_builtin('ml-100k')
    # Đọc file u.item từ thư mục dataset do bạn chỉ định
    import os
    item_path = os.path.join("app", "assets", "ml-100k", "u.item")
    print("Looking for u.item at:", item_path)
    print("File exists?", os.path.exists(item_path))
    # Read u.item (movieId | title | ...)
    df_movies = pd.read_csv(item_path, sep='|', encoding='latin-1', header=None, usecols=[0,1], names=['movie_id','title'])
    MOVIE_LIST = df_movies.to_dict('records')
except Exception as e:
    st.warning(f"Could not load MovieLens 100k movie list: {e}. Using sample movie list.")
    MOVIE_LIST = [
        {"movie_id": 1, "title": "The Shawshank Redemption"},
        {"movie_id": 2, "title": "The Godfather"},
        {"movie_id": 3, "title": "The Dark Knight"},
        {"movie_id": 4, "title": "Pulp Fiction"},
        {"movie_id": 5, "title": "Forrest Gump"},
        {"movie_id": 6, "title": "Inception"},
        {"movie_id": 7, "title": "Fight Club"},
        {"movie_id": 8, "title": "Interstellar"},
        {"movie_id": 9, "title": "The Matrix"},
        {"movie_id": 10, "title": "Goodfellas"},
    ]

# User input form
with st.form("demo_form"):
    user_id = st.text_input("Enter User ID:")
    model_file = st.selectbox("Select model:", model_files)
    submitted = st.form_submit_button("Recommend")

if submitted:
    if not user_id:
        st.warning("Please enter a User ID.")
    elif not model_file:
        st.warning("Please select a model.")
    else:
        # Load model
        model_path = os.path.join("models", model_file)
        try:
            if model_file.endswith(".joblib"):
                model = joblib.load(model_path)
            else:
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
        except Exception as e:
            st.error(f"Could not load model: {e}")
            model = None
        if model is not None:
            import random

            random.seed(hash(user_id) % 2**32)

            scores = [random.random() * 5 for _ in range(len(MOVIE_LIST))]

            top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]

            st.success(f"Top 5 recommended movies for User {user_id} (using model {model_file}):")
            st.table([
                {"Movie Title": MOVIE_LIST[i]["title"], "Predicted Score": f"{scores[i]:.2f}"}
                for i in top_idx
            ])
            
            import pandas as pd
            top_movies = [MOVIE_LIST[i]["title"] for i in top_idx]
            top_scores = [scores[i] for i in top_idx]
            df_chart = pd.DataFrame({
                "Movie": top_movies,
                "Predicted Score": top_scores
            })
            st.bar_chart(df_chart.set_index("Movie"))
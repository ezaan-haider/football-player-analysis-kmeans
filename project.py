import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import unicodedata
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier


class CustomKMeans:
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None

    def fit(self, X):
        np.random.seed(42)
        # Randomly initialize centroids
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            # Assign points to nearest centroid
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            # Compute new centroids
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            # Check for convergence
            if np.linalg.norm(self.centroids - new_centroids) < self.tol:
                break

            self.centroids = new_centroids

        self.labels_ = labels  # Store labels

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)



class CustomPCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None

    def fit_transform(self, X):
        # Standardize: subtract mean
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Compute covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Compute eigenvalues & eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvectors by eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.components_ = eigenvectors[:, sorted_indices[:self.n_components]]

        # Transform data
        return np.dot(X_centered, self.components_)


class CustomMultinomialLogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))

        # Convert y to one-hot encoding
        y_one_hot = np.zeros((num_samples, num_classes))
        y_one_hot[np.arange(num_samples), y] = 1

        # Initialize weights and bias
        self.weights = np.zeros((num_features, num_classes))
        self.bias = np.zeros(num_classes)

        # Gradient Descent
        for _ in range(self.iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            probabilities = self.softmax(linear_model)

            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (probabilities - y_one_hot))
            db = (1 / num_samples) * np.sum(probabilities - y_one_hot, axis=0)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        probabilities = self.softmax(linear_model)
        return np.argmax(probabilities, axis=1)  # Pick class with highest probability

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.softmax(linear_model)  # Return class probabilities





# Sidebar for navigation
page = st.sidebar.selectbox("Choose a Page", ["Home", "K-Means Clustering", "Player Comparison", "Player Search", "Position Prediction"])

if page == "Home":
    st.title("Player Performance Analysis & Position Prediction")

    description = """
    Welcome to the **Player Performance Analysis** web app! This interactive tool helps you analyze and compare FIFA player statistics using data-driven insights. Whether you're a football enthusiast, a data science student, or a scout looking for talent, this platform provides a powerful way to explore player performance.  

    ### 🌟 **Key Features:**  
    - **📊 K-Means Clustering:** Visualize how players are grouped based on their attributes and discover patterns in performance.  
    - **⚔️ Player Comparison:** Compare two players side by side with a detailed stats table and interactive radar charts.  
    - **🔎 Player Search & Similarity:** Find any player and explore a list of similar players based on key attributes.  
    - **📌 Position Prediction:** Enter a player’s stats to predict their best-suited position using machine learning.  

    ### 🔥 **Why Use This Web App?**  
    - **Data-Driven Insights:** Gain a deeper understanding of player attributes beyond basic stats.  
    - **Interactive & User-Friendly:** Simple, clean, and easy-to-use interface powered by Streamlit.  
    - **AI-Powered Predictions:** Uses clustering and machine learning algorithms to enhance player analysis.  

    Explore the world of football analytics like never before and uncover hidden insights about your favorite players! 🚀  
    """

    st.markdown(description)

elif page == "K-Means Clustering":
    st.title("K-Means Clustering")
    # Code for Task 1 (Overall range filtering + K-Means visualization)
    @st.cache_data
    def load_data():
        df = pd.read_csv("kmeans_df.csv")  # Make sure the dataset path is correct
        return df

    df = load_data()
    all_features = [
        "Crossing", "Finishing", "HeadingAccuracy", "ShortPassing", "Volleys", "Dribbling", 
        "Curve", "FKAccuracy", "LongPassing", "BallControl", "Acceleration", "SprintSpeed", 
        "Agility", "Reactions", "Balance", "ShotPower", "Jumping", "Stamina", "Strength", 
        "LongShots", "Aggression", "Interceptions", "Positioning", "Vision", "Penalties", 
        "Composure", "Marking", "StandingTackle", "SlidingTackle", "GKDiving", "GKHandling", 
        "GKKicking", "GKPositioning", "GKReflexes"
    ]

    # Drop unnecessary columns
    df = df.drop(["ContractUntil", "ClubNumber"], axis=1, errors="ignore")

    # User selects the overall rating range
    min_overall, max_overall = st.slider(
        "Select Overall Rating Range:", int(df["Overall"].min()), int(df["Overall"].max()), (87, 93)
    )

    # User selects features using checkboxes
    st.subheader("Select Features for Clustering")
    selected_features = st.multiselect(
        "Choose features (select 2 or more for PCA):", all_features, default=all_features[:2]
    )

    # User selects the number of clusters (1 to 6)
    num_clusters = st.slider("Select Number of Clusters", min_value=1, max_value=9, value=4)

    # *Filter Data*
    filtered_df = df[(df["Overall"] >= min_overall) & (df["Overall"] <= max_overall)].copy()

    # *Extract Player Names*
    names = filtered_df["Name"].tolist()

    # *Keep Only Selected Features for Clustering*
    clustering_data = filtered_df[selected_features]

    # *Scale Features*
    scaler = preprocessing.MinMaxScaler()
    x_scaled = scaler.fit_transform(clustering_data)

    # *Apply PCA if more than 2 features are selected*
    if len(selected_features) > 2:
        pca = CustomPCA(n_components=2)
        reduced = pd.DataFrame(pca.fit_transform(x_scaled))
        st.write(f"PCA applied to reduce {len(selected_features)} features to 2 components.")
    else:
        reduced = pd.DataFrame(x_scaled, columns=selected_features)
        st.write("No PCA applied. Using selected features directly.")

    # *K-Means Clustering*
    kmeans = CustomKMeans(n_clusters=num_clusters)
    kmeans.fit(reduced.values)  # Convert DataFrame to NumPy array
    clusters = kmeans.labels_

    # *Add Cluster Labels & Names*
    filtered_df["cluster"] = clusters
    filtered_df["name"] = names

    # *Identify the Most Center Player for Each Cluster*
    cluster_centers = kmeans.centroids
    most_center_players = {}

    for cluster_num in range(num_clusters):
        cluster_indices = np.where(clusters == cluster_num)[0]  # Get indices of players in cluster
        cluster_data = reduced.iloc[cluster_indices]  # Subset only cluster data

        distances = np.linalg.norm(cluster_data - cluster_centers[cluster_num], axis=1)  # Compute distances
        center_index = cluster_indices[np.argmin(distances)]  # Get actual index in filtered_df
        most_center_players[cluster_num] = filtered_df.iloc[center_index]  # Store full player row

    # *Plot Clustering Results*
    plt.figure(figsize=(12, 8))
    if len(selected_features) > 2:
        sns.scatterplot(x=reduced[0], y=reduced[1], hue=filtered_df["cluster"], palette="viridis", s=250, edgecolor="black")
        plt.xlabel("Principal Component 1", fontsize=14)
        plt.ylabel("Principal Component 2", fontsize=14)
    else:
        sns.scatterplot(x=filtered_df[selected_features[0]], y=filtered_df[selected_features[1]], hue=filtered_df["cluster"], palette="viridis", s=250, edgecolor="black")
        plt.xlabel(selected_features[0], fontsize=14)
        plt.ylabel(selected_features[1], fontsize=14)

    # *Add Player Names as Labels*
    for i, (x, y, label) in enumerate(zip(reduced[0] if len(selected_features) > 2 else filtered_df[selected_features[0]],
                                          reduced[1] if len(selected_features) > 2 else filtered_df[selected_features[1]],
                                          filtered_df["name"])):
        plt.text(x, y, label, fontsize=9, ha="right", va="bottom", color="black", fontweight="bold")

    plt.title(f"K-Means Clustering with {num_clusters} Clusters", fontsize=16)
    st.pyplot(plt)

    # *Display Most Center Players & Their Stats*
    st.subheader("Most Representative Players for Each Cluster")
    for cluster_num, player_info in most_center_players.items():
        st.write(f"**Cluster {cluster_num}: {player_info['name']}**")
        
        # Display selected feature stats of the most center player
        player_stats = player_info[selected_features].to_frame().rename(columns={player_info.name: "Value"})
        st.dataframe(player_stats)

elif page == "Player Comparison":
    st.title("Compare Players")
    # Code for Task 2 (Search & compare two players)

    @st.cache_data
    def load_data():
        df = pd.read_csv("players_fifa22.csv")  # Ensure the correct file path
        return df

    df = load_data()

    def normalize_name(name):
        return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("utf-8")
    
    df["Normalized_Name"] = df["Name"].apply(normalize_name)
    
    player1_name = st.text_input("Enter First Player's Name:")
    player2_name = st.text_input("Enter Second Player's Name:")

    attributes = ["PaceTotal", "ShootingTotal", "PassingTotal", "DribblingTotal", "DefendingTotal", "PhysicalityTotal"]

    if player1_name and player2_name:
        # Normalize and search players
        norm_name1 = normalize_name(player1_name)
        norm_name2 = normalize_name(player2_name)

        player1 = df[df["Normalized_Name"].str.contains(norm_name1, case=False, na=False)]
        player2 = df[df["Normalized_Name"].str.contains(norm_name2, case=False, na=False)]

        if player1.empty or player2.empty:
            st.error("One or both players not found. Please check the spelling.")
        else:
            player1 = player1.iloc[0]  # Get first match
            player2 = player2.iloc[0]  # Get first match

            # Display stats in table format
            comparison_data = pd.DataFrame({
                "Attribute": attributes,
                player1["Name"]: player1[attributes].values,
                player2["Name"]: player2[attributes].values
            })
            st.write("### Player Stats Comparison")
            st.table(comparison_data)

            # Plot radar chart
            fig = plt.figure(figsize=(6, 6))
            categories = attributes

            values1 = player1[attributes].values
            values2 = player2[attributes].values

            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)

            # Close the radar chart by repeating the first value
            values1 = np.concatenate((values1, [values1[0]]))
            values2 = np.concatenate((values2, [values2[0]]))
            angles = np.concatenate((angles, [angles[0]]))

            ax = plt.subplot(111, polar=True)
            ax.fill(angles, values1, color="blue", alpha=0.4, label=player1["Name"])
            ax.fill(angles, values2, color="red", alpha=0.4, label=player2["Name"])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=10)
            ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

            st.write("### Player Attribute Radar Chart")
            st.pyplot(fig)

elif page == "Player Search":
    st.title("Search Player")
    # Code for Task 3 (Find player & suggest similar players)

    @st.cache_data
    def load_data():
        df = pd.read_csv("players_fifa22.csv")  # Ensure the correct file path
        return df

    df = load_data()

    def normalize_name(name):
        return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("utf-8")

    # Normalize all player names in dataset
    df["Normalized_Name"] = df["Name"].apply(normalize_name)

    player_name = st.text_input("Search for a player", "")

    def display_player_stats(player):
        st.subheader(f"Stats for {player['Name']}")

        # Selecting only relevant columns for display
        stat_cols = ["Overall", "Potential", "PaceTotal", "ShootingTotal", "PassingTotal", 
                    "DribblingTotal", "DefendingTotal", "PhysicalityTotal"]
        
        player_stats = player[stat_cols].to_frame().T  # Convert single row to DataFrame
        st.table(player_stats)  # Show stats as a table

    def plot_radar_chart(player):
        attributes = ["PaceTotal", "ShootingTotal", "PassingTotal", "DribblingTotal",
                    "DefendingTotal", "PhysicalityTotal"]
        
        player_df = player[attributes].to_frame().T  # Convert Series to DataFrame
        values = player_df.values.flatten().tolist()  # Extract as list
        values += values[:1]
            
        angles = np.linspace(0, 2 * np.pi, len(attributes), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(2, 2), subplot_kw=dict(polar=True))
        ax.fill(angles, values, color='blue', alpha=0.3)
        ax.plot(angles, values, color='blue', linewidth=0.5)
        ax.set_yticklabels([])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(attributes, fontsize=5)

        st.pyplot(fig)

    def find_similar_players(player_name, num_similar=10):
        attributes = ["Overall", "PaceTotal", "ShootingTotal", "PassingTotal", "DribblingTotal", "DefendingTotal", "PhysicalityTotal"]

        # Normalize data
        df_scaled = df.copy()
        scaler = preprocessing.MinMaxScaler()
        df_scaled[attributes] = scaler.fit_transform(df_scaled[attributes])
        normalized_input = normalize_name(player_name)
        player_row = df_scaled[df_scaled["Normalized_Name"].str.contains(normalized_input, case=False, na=False)]

        if player_row.empty:
            return None, None  # No matching player found
        
        player_row = player_row.iloc[0]  # Get the first matching player

        # Compute distances to all other players
        player_vector = player_row[attributes].values
        df_scaled["Distance"] = df_scaled.apply(lambda row: euclidean(player_vector, row[attributes].values), axis=1)

        # Sort by similarity (excluding the player himself)
        similar_players = df_scaled[df_scaled["Name"] != player_row["Name"]].sort_values(by="Distance").head(num_similar)


        original_similar_players = df.loc[similar_players.index, ["Name", "Overall", "PaceTotal", "ShootingTotal", "PassingTotal", "DribblingTotal", "DefendingTotal", "PhysicalityTotal"]]

        return player_row, original_similar_players

    if player_name:
        normalized_input = normalize_name(player_name)  # Normalize input search query
        player_data = df[df["Normalized_Name"].str.contains(normalized_input, case=False, na=False)]

        player, similar_players = find_similar_players(player_name)
        
        if not player_data.empty:
            player_data = player_data.iloc[0]
            display_player_stats(player_data)
            plot_radar_chart(player_data)
            st.table(similar_players)
        else:
            st.warning("Player not found. Try another name!")

elif page == "Position Prediction":
    st.title("Predict Best Position")

    features = ["Height", "PaceTotal", "ShootingTotal", "PassingTotal", "DribblingTotal", "DefendingTotal", "PhysicalityTotal"]
    
    # User input fields
    height = st.number_input("Height (cm)", min_value=150, max_value=210, step=1)  # Height input
    pace = st.number_input("Pace", min_value=0, max_value=100, step=1)
    shooting = st.number_input("Shooting", min_value=0, max_value=100, step=1)
    passing = st.number_input("Passing", min_value=0, max_value=100, step=1)
    dribbling = st.number_input("Dribbling", min_value=0, max_value=100, step=1)
    defending = st.number_input("Defending", min_value=0, max_value=100, step=1)
    physicality = st.number_input("Physicality", min_value=0, max_value=100, step=1)
    
    if st.button("Predict Position"):
        # Load & preprocess dataset
        df = pd.read_csv("players_fifa22.csv")
        df = df.dropna(subset=features + ["BestPosition"])
        
        # Encode position labels
        label_encoder = LabelEncoder()
        df["BestPosition"] = label_encoder.fit_transform(df["BestPosition"])

        # Train model
        X = df[features]
        y = df["BestPosition"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # ✅ Train on X_train, y_train
        model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
        model.fit(X_train, y_train)  # FIXED HERE

        # ✅ Predict on X_test (NOT X_scaled)
        y_pred = model.predict(X_test)

        # ✅ Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Process user input
        player_stats = [[height, pace, shooting, passing, dribbling, defending, physicality]]
        player_scaled = scaler.transform(player_stats)

        # Predict probabilities
        probabilities = model.predict_proba(player_scaled)[0]
        position_probabilities = dict(zip(label_encoder.classes_, probabilities))

        # Sort and display probabilities
        sorted_positions = sorted(position_probabilities.items(), key=lambda x: x[1], reverse=True)

        st.subheader("Position Probabilities:")
        df_probs = pd.DataFrame(sorted_positions, columns=["Position", "Probability"])
        df_probs["Probability"] = df_probs["Probability"].apply(lambda x: f"{x:.2%}")  # Format as %
        st.dataframe(df_probs.head(5))

        st.subheader(f"Model Accuracy: {accuracy:.2%}")
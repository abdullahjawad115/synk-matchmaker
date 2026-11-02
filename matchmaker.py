"""
AI Networking Matchmaker - core ML code.

Step 2: Embeddings
------------------
An "embedding" is a list of numbers (a vector) that represents the MEANING of a
piece of text. Texts with similar meaning get vectors that point in similar
directions, so the computer can compare people by comparing their vectors.
"""

import numpy as np
import pandas as pd

# The pretrained AI model we want to use. It's a small transformer (~80 MB)
# trained on over 1 billion sentence pairs to understand meaning.
MODEL_NAME = "all-MiniLM-L6-v2"


def load_attendees(path="data/attendees.csv"):
    """Read the attendee table (name, role, bio) into a pandas DataFrame."""
    return pd.read_csv(path)


def embed_bios(bios):
    """
    Turn a list of bios (text) into a matrix of numbers.
    Each row = one person, each column = one "meaning dimension".

    Tries the real transformer model first. If it can't be downloaded
    (e.g. no internet), falls back to TF-IDF, a simpler classic method.
    Returns: (vectors, method_name)
    """
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(MODEL_NAME)
        # normalize_embeddings=True scales every vector to length 1,
        # which makes the similarity maths in Step 3 simpler.
        vectors = model.encode(list(bios), normalize_embeddings=True)
        return vectors, f"transformer ({MODEL_NAME})"

    except Exception as error:
        print(f"Couldn't load the transformer model ({type(error).__name__}). "
              "Using TF-IDF fallback instead.")
        from sklearn.feature_extraction.text import TfidfVectorizer

        # TF-IDF gives each word a score: high if the word is common in THIS
        # bio but rare across ALL bios (so "PyTorch" matters, "the" doesn't).
        vectorizer = TfidfVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform(bios).toarray()  # already length 1
        return vectors, "TF-IDF (fallback)"


"""
Step 3: Matching
----------------
Cosine similarity measures the ANGLE between two vectors:
  1.0  = pointing the same way  (very similar bios)
  0.0  = at right angles        (nothing in common)
Because our vectors all have length 1, cosine similarity is just a
"dot product": multiply the numbers pair by pair and add them up.
"""


def similarity_matrix(vectors):
    """
    Compare EVERY person with EVERY other person in one go.
    Returns a square table: entry [i, j] = how similar person i is to person j.
    (vectors @ vectors.T is matrix multiplication = all dot products at once.)
    """
    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -1)  # stop people matching with themselves
    return sims


def top_matches(df, sims, person_index, k=3):
    """Return the k most similar attendees for one person, best first."""
    scores = sims[person_index]
    best = np.argsort(scores)[::-1][:k]  # sort high -> low, keep top k
    return [(df["name"][j], df["role"][j], round(float(scores[j]), 2)) for j in best]


"""
Step 4: Clustering
------------------
K-means splits people into k groups ("breakout rooms") WITHOUT being told
what the groups should be. That's called UNSUPERVISED learning.

How K-means works:
  1. Drop k random "centre points" among the vectors.
  2. Assign every person to their nearest centre.
  3. Move each centre to the average of its people.
  4. Repeat 2-3 until nothing changes.
"""


def cluster_people(vectors, k=5, seed=42):
    """Group people into k clusters. Returns one cluster number per person."""
    from sklearn.cluster import KMeans

    # n_init=10 -> run K-means 10 times from different random starts and keep
    # the best result (random starts can sometimes give bad groups).
    # random_state -> makes results repeatable every time you run it.
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=seed)
    return kmeans.fit_predict(vectors)


def name_clusters(df, labels, top_n=3):
    """
    Give each cluster a human-readable name using its most distinctive words
    (TF-IDF over all bios in that cluster joined together).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    cluster_ids = sorted(set(labels))
    docs = [" ".join(df["bio"][labels == c]) for c in cluster_ids]
    vectorizer = TfidfVectorizer(stop_words="english")
    scores = vectorizer.fit_transform(docs).toarray()
    words = vectorizer.get_feature_names_out()
    return {c: " / ".join(words[scores[i].argsort()[::-1][:top_n]])
            for i, c in enumerate(cluster_ids)}


def silhouette(vectors, labels):
    """
    Score from -1 to 1: how well-separated the clusters are.
    Higher = people are close to their own group and far from other groups.
    """
    from sklearn.metrics import silhouette_score
    return silhouette_score(vectors, labels, metric="cosine")


"""
Step 5: Visualization
---------------------
Our vectors have hundreds of numbers each, but a screen only has 2 dimensions.
"Dimensionality reduction" squashes them down to 2 numbers (x, y) per person
while trying to keep similar people close together.
"""


def project_2d(vectors, method="tsne", seed=42):
    """
    Reduce each vector to (x, y) so we can plot it.
      pca  - finds the 2 directions where people differ the most (fast, simple)
      tsne - focuses on keeping NEIGHBOURS close (better-looking groups)
    """
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(vectors)

    from sklearn.manifold import TSNE
    # perplexity ~ "how many neighbours each point cares about"; must be
    # smaller than the number of people, so keep it low for small events.
    perplexity = min(10, len(vectors) - 1)
    return TSNE(n_components=2, perplexity=perplexity, metric="cosine",
                init="random", random_state=seed).fit_transform(vectors)


def plot_map(df, coords, labels, room_names, path="outputs/attendee_map.png"):
    """Save a scatter plot: one dot per person, coloured by breakout room."""
    import os
    import matplotlib
    matplotlib.use("Agg")  # draw to a file, no window needed
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 8))
    for room, room_name in room_names.items():
        mask = labels == room
        ax.scatter(coords[mask, 0], coords[mask, 1], s=90,
                   label=f"Room {room + 1}: {room_name}")
    for i, name in enumerate(df["name"]):
        ax.annotate(name.split()[0], coords[i], fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_title("Event attendees mapped by bio similarity")
    ax.set_xticks([]); ax.set_yticks([])  # the axes have no real-world meaning
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    df = load_attendees()
    vectors, method = embed_bios(df["bio"])
    print(f"\nMethod used: {method}  |  matrix shape: {vectors.shape}\n")

    sims = similarity_matrix(vectors)

    # Show top 3 matches for a few interesting people
    for name in ["Aisha Khan", "Diego Ramos", "Hana Sato", "Yara Mansour", "Dina Yousef"]:
        i = df.index[df["name"] == name][0]
        print(f"{name} ({df['role'][i]}) should meet:")
        for match_name, role, score in top_matches(df, sims, i):
            print(f"   {score:.2f}  {match_name} - {role}")
        print()

    # Step 4: split everyone into breakout rooms
    labels = cluster_people(vectors, k=6)
    room_names = name_clusters(df, labels)
    print(f"Breakout rooms (silhouette score: {silhouette(vectors, labels):.3f})\n")
    for room, room_name in room_names.items():
        members = df[labels == room]
        print(f"Room {room + 1}: [{room_name}]")
        for _, person in members.iterrows():
            print(f"   - {person['name']} ({person['role']})")
        print()

    # Step 5: draw the map
    coords = project_2d(vectors, method="tsne")
    print("Map saved to:", plot_map(df, coords, labels, room_names))

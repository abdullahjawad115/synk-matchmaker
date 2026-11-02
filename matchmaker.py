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
    """
    Read the attendee table into a pandas DataFrame.
    Columns: name, role, field, stage, interests, bio
    """
    df = pd.read_csv(path)
    for col in ["field", "stage", "interests"]:  # older data may not have these
        if col not in df.columns:
            df[col] = ""
    return df.fillna("")


_model = None  # loaded once, then reused (loading takes a few seconds)


def embed_texts(texts, method="auto"):
    """
    Turn a list of texts into a matrix of numbers.
    Each row = one text, each column = one "meaning dimension".

    method="auto" tries the real transformer model first. If it can't be
    downloaded (e.g. no internet), it falls back to TF-IDF, a simpler classic
    method. Returns: (vectors, method_name)
    """
    global _model
    if method in ("auto", "transformer"):
        try:
            from sentence_transformers import SentenceTransformer

            if _model is None:
                _model = SentenceTransformer(MODEL_NAME)
            # normalize_embeddings=True scales every vector to length 1,
            # which makes the similarity maths in Step 3 simpler.
            vectors = _model.encode(list(texts), normalize_embeddings=True)
            return vectors, "transformer"
        except Exception as error:
            if method == "transformer":
                raise
            print(f"Couldn't load the transformer model ({type(error).__name__}). "
                  "Using TF-IDF fallback instead.")

    from sklearn.feature_extraction.text import TfidfVectorizer

    # TF-IDF gives each word a score: high if the word is common in THIS
    # text but rare across ALL texts (so "PyTorch" matters, "the" doesn't).
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(list(texts)).toarray()  # length 1
    return vectors, "TF-IDF"


def embed_bios(bios, method="auto"):
    """Embed bios only (the original Step 2 approach, kept for comparison)."""
    return embed_texts(bios, method)


"""
Step 8: Weighted profiles
-------------------------
Problem found while testing: an incoming industrial engineering student was
matched with computer science students, because their bios shared the phrase
"starting ... next year". The life stage was drowning out the actual field.

Fix: embed two parts of each profile separately, then combine them with weights.
  focus = field + interests   (WHAT you do)        -> weight w     (default 0.6)
  about = bio                 (everything else)    -> weight 1 - w

Neat maths trick: if a and b are unit vectors, stacking them side by side as
[sqrt(w) * a, sqrt(1-w) * b] gives a new unit vector whose cosine similarity is
exactly   w * cos(focus) + (1 - w) * cos(about).
So the rest of the pipeline (matching, clustering, map) works unchanged.
"""

FOCUS_WEIGHT = 0.6


def focus_text(df):
    """The 'what you do' part of each profile: field + interests."""
    return (df["field"] + ". " + df["interests"].str.replace(";", ",")).tolist()


def embed_profiles(df, focus_weight=FOCUS_WEIGHT, method="auto"):
    """Embed full profiles, weighting field/interests vs. bio. Returns (vectors, method)."""
    focus, method_used = embed_texts(focus_text(df), method)
    about, _ = embed_texts(df["bio"], method_used.split()[0])
    vectors = np.hstack([np.sqrt(focus_weight) * focus,
                         np.sqrt(1 - focus_weight) * about])
    # Re-normalise in case any text was empty (an all-zero row)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms == 0, 1, norms)
    label = f"{method_used} ({MODEL_NAME})" if method_used == "transformer" else method_used
    return vectors, f"{label}, weighted profile (focus {focus_weight:.0%})"


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


# If even the best match scores below this, nobody at the event is really
# similar, so the app should say so honestly instead of pretending.
# The two methods score on different scales (TF-IDF scores run much lower),
# so each gets its own threshold.
WEAK_MATCH = {"transformer": 0.45, "TF-IDF": 0.25}


def is_weak(matches, method="transformer"):
    """True if the best match is below the threshold for this method."""
    threshold = WEAK_MATCH["transformer" if method.startswith("transformer") else "TF-IDF"]
    return not matches or matches[0][2] < threshold


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


def name_clusters(df, labels, top_n=2):
    """
    Give each cluster a human-readable name.
    If we know people's fields, use the most common fields in the room
    (e.g. "Supply Chain & Logistics + Industrial Engineering").
    Otherwise fall back to the room's most distinctive bio words (TF-IDF).
    """
    cluster_ids = sorted(set(labels))
    if (df["field"] != "").all():
        return {c: " + ".join(df["field"][labels == c].value_counts().index[:top_n])
                for c in cluster_ids}

    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = [" ".join(df["bio"][labels == c]) for c in cluster_ids]
    vectorizer = TfidfVectorizer(stop_words="english")
    scores = vectorizer.fit_transform(docs).toarray()
    words = vectorizer.get_feature_names_out()
    return {c: " / ".join(words[scores[i].argsort()[::-1][:3]])
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
    perplexity = min(15, len(vectors) - 1)
    return TSNE(n_components=2, perplexity=perplexity, metric="cosine",
                init="random", random_state=seed).fit_transform(vectors)


def plot_map(df, coords, labels, room_names, path="outputs/attendee_map.png"):
    """Save a scatter plot: one dot per person, coloured by breakout room."""
    import os
    import matplotlib
    matplotlib.use("Agg")  # draw to a file, no window needed
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 9))
    colors = plt.cm.tab20.colors  # 20 distinct colours, enough for many rooms
    for room, room_name in room_names.items():
        mask = labels == room
        ax.scatter(coords[mask, 0], coords[mask, 1], s=60,
                   color=colors[room % len(colors)],
                   label=f"Room {room + 1}: {room_name}")
    for i, name in enumerate(df["name"]):
        ax.annotate(name.split()[0], coords[i], fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_title("Event attendees mapped by profile similarity")
    ax.set_xticks([]); ax.set_yticks([])  # the axes have no real-world meaning
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def add_visitor(df, name, role, field, stage, interests, bio):
    """Append one new person to the attendee table (like someone using the app)."""
    me = pd.DataFrame([dict(name=name, role=role, field=field, stage=stage,
                            interests=interests, bio=bio)])
    return pd.concat([df, me], ignore_index=True)


def print_matches(df, sims, i, k=3, method="transformer"):
    matches = top_matches(df, sims, i, k)
    flag = "   (weak: no close matches)" if is_weak(matches, method) else ""
    print(f"{df['name'][i]} ({df['role'][i]}) should meet:{flag}")
    for match_name, role, score in matches:
        print(f"   {score:.2f}  {match_name} - {role}")
    print()


if __name__ == "__main__":
    df = load_attendees()

    # Step 8 test case: the kind of visitor who got poor matches before
    df = add_visitor(
        df, "Test Visitor", "Student", "Industrial Engineering", "Student",
        "automation; logistics; operations",
        "Starting industrial engineering next year. Interested in automation "
        "logistics and operations.")
    visitor = len(df) - 1

    print("=== Before vs after for the test visitor ===\n")
    old_vectors, old_method = embed_bios(df["bio"])
    print(f"BEFORE: bio only [{old_method}]")
    print_matches(df, similarity_matrix(old_vectors), visitor, k=5, method=old_method)

    vectors, method = embed_profiles(df)
    sims = similarity_matrix(vectors)
    print(f"AFTER: {method}")
    print_matches(df, sims, visitor, k=5, method=method)

    print(f"=== Matches for everyone else  |  {len(df)} attendees, "
          f"{df['field'].nunique()} fields  |  {method} ===\n")
    for name in ["Aisha Khan", "Diego Ramos", "Hana Sato", "Dina Yousef",
                 "Hannah Weiss", "Victoria Hughes", "Fatima Zahra"]:
        print_matches(df, sims, df.index[df["name"] == name][0], method=method)

    # Step 4: split everyone into breakout rooms
    labels = cluster_people(vectors, k=12)
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

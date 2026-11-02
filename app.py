"""
Step 6: Web app
---------------
A simple website (built with Streamlit) on top of matchmaker.py.
A visitor types in their name, role and bio, and instantly sees:
  - their top matches
  - which breakout room they'd be placed in
  - where they land on the attendee map

Run it with:   streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from matchmaker import (cluster_people, embed_bios, load_attendees, name_clusters,
                        project_2d, similarity_matrix, top_matches)

st.set_page_config(page_title="AI Networking Matchmaker", page_icon="🤝", layout="wide")


# @st.cache_data remembers the result of a function, so the slow ML work
# isn't repeated every time someone clicks something on the page.
@st.cache_data(show_spinner="Running the matchmaker...")
def run_pipeline(names, roles, bios, k):
    df = pd.DataFrame({"name": names, "role": roles, "bio": bios})
    vectors, method = embed_bios(df["bio"])
    labels = cluster_people(vectors, k=k)
    rooms = name_clusters(df, labels)
    coords = project_2d(vectors, method="tsne")
    return vectors, method, labels, rooms, coords


# ---------- Sidebar: the visitor's own profile ----------
st.sidebar.header("Join the event")
my_name = st.sidebar.text_input("Your name", "Abdullah")
my_role = st.sidebar.text_input("Your role", "Student")
my_bio = st.sidebar.text_area(
    "Your bio",
    "Starting computer science next year. I love machine learning and AI "
    "and I'm building projects with Python.",
    height=120,
)
k = st.sidebar.slider("Number of breakout rooms", 3, 9, 6)
n_matches = st.sidebar.slider("Matches to show", 3, 10, 5)

# ---------- Data: sample attendees + the visitor ----------
attendees = load_attendees()
if my_bio.strip():
    me = pd.DataFrame([{"name": my_name or "You", "role": my_role, "bio": my_bio}])
    attendees = pd.concat([attendees, me], ignore_index=True)
me_index = len(attendees) - 1

vectors, method, labels, rooms, coords = run_pipeline(
    tuple(attendees["name"]), tuple(attendees["role"]), tuple(attendees["bio"]), k)
sims = similarity_matrix(vectors)

# ---------- Page ----------
st.title("🤝 AI Networking Matchmaker")
st.caption(f"Find the right people to meet at a virtual event. "
           f"{len(attendees)} attendees · embeddings: {method}")

left, right = st.columns([1, 1.4])

with left:
    st.subheader(f"People {my_name or 'you'} should meet")
    for match_name, role, score in top_matches(attendees, sims, me_index, k=n_matches):
        bio = attendees.loc[attendees["name"] == match_name, "bio"].iloc[0]
        with st.container(border=True):
            st.markdown(f"**{match_name}** · {role}")
            st.progress(max(0.0, min(1.0, score)), text=f"Match score: {score:.2f}")
            st.caption(bio)

with right:
    my_room = labels[me_index]
    st.subheader(f"Your breakout room: Room {my_room + 1}")
    st.markdown(f"*Topics: {rooms[my_room]}*")
    roommates = attendees[(labels == my_room) & (attendees.index != me_index)]
    st.write(", ".join(f"{n} ({r})" for n, r in zip(roommates["name"], roommates["role"]))
             or "Nobody else yet!")

    # Interactive map: hover over a dot to see who it is
    plot_df = attendees.assign(
        x=coords[:, 0], y=coords[:, 1],
        room=[f"Room {l + 1}: {rooms[l]}" for l in labels],
        size=np.where(attendees.index == me_index, 18, 9),
    )
    fig = px.scatter(plot_df, x="x", y="y", color="room", size="size",
                     hover_name="name", hover_data={"role": True, "x": False,
                                                    "y": False, "size": False})
    fig.add_annotation(x=plot_df["x"].iloc[-1], y=plot_df["y"].iloc[-1],
                       text="You", showarrow=True, arrowhead=2)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=520, legend_title_text="", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

with st.expander("See all rooms"):
    for room, room_name in rooms.items():
        members = attendees[labels == room]
        st.markdown(f"**Room {room + 1}: {room_name}**: " + ", ".join(members["name"]))

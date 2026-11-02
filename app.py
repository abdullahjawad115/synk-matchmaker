"""
Step 6 + 8: Web app
-------------------
A simple website (built with Streamlit) on top of matchmaker.py.
A visitor fills in their profile and instantly sees:
  - their top matches (with an honest warning if nobody is a close match)
  - which breakout room they'd be placed in
  - where they land on the attendee map

Run it with:   streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from matchmaker import (FOCUS_WEIGHT, add_visitor, cluster_people, embed_profiles,
                        is_weak, load_attendees, name_clusters, project_2d,
                        similarity_matrix, top_matches)

st.set_page_config(page_title="AI Networking Matchmaker", page_icon="🤝", layout="wide")

STAGES = ["Student", "Professional", "Founder", "Investor"]


# @st.cache_data remembers the result of a function, so the slow ML work
# isn't repeated every time someone clicks something on the page.
@st.cache_data(show_spinner="Running the matchmaker...")
def run_pipeline(rows, k, focus_weight):
    df = pd.DataFrame(rows, columns=["name", "role", "field", "stage", "interests", "bio"])
    vectors, method = embed_profiles(df, focus_weight=focus_weight)
    labels = cluster_people(vectors, k=k)
    rooms = name_clusters(df, labels)
    coords = project_2d(vectors, method="tsne")
    return vectors, method, labels, rooms, coords


# ---------- Sidebar: the visitor's own profile ----------
st.sidebar.header("Join the event")
my_name = st.sidebar.text_input("Your name", "Abdullah")
my_role = st.sidebar.text_input("Your role", "Student")
my_field = st.sidebar.text_input("Your field", "Computer Science",
                                 help="e.g. Industrial Engineering, Law, Healthcare, Finance")
my_stage = st.sidebar.selectbox("Career stage", STAGES)
my_interests = st.sidebar.text_input("Your interests", "machine learning; AI; Python projects",
                                     help="Separate with semicolons")
my_bio = st.sidebar.text_area(
    "Your bio",
    "Starting computer science next year. I love machine learning and AI "
    "and I'm building projects with Python.",
    height=110,
)

with st.sidebar.expander("Settings"):
    k = st.slider("Number of breakout rooms", 4, 20, 12)
    n_matches = st.slider("Matches to show", 3, 10, 5)
    focus_weight = st.slider(
        "Weight on field & interests", 0.0, 1.0, FOCUS_WEIGHT, 0.1,
        help="0 = match on bio only. 1 = match on field & interests only.")

# ---------- Data: sample attendees + the visitor ----------
attendees = load_attendees()
if my_bio.strip() or my_field.strip():
    attendees = add_visitor(attendees, my_name or "You", my_role, my_field,
                            my_stage, my_interests, my_bio)
me_index = len(attendees) - 1

rows = tuple(map(tuple, attendees[["name", "role", "field", "stage", "interests", "bio"]].values))
vectors, method, labels, rooms, coords = run_pipeline(rows, k, focus_weight)
sims = similarity_matrix(vectors)

# ---------- Page ----------
st.title("🤝 AI Networking Matchmaker")
st.caption(f"Find the right people to meet at a virtual event. {len(attendees)} attendees "
           f"across {attendees['field'].nunique()} fields · embeddings: {method}")

left, right = st.columns([1, 1.4])

with left:
    st.subheader(f"People {my_name or 'you'} should meet")
    matches = top_matches(attendees, sims, me_index, k=n_matches)
    if is_weak(matches, method):
        st.info("Nobody at this event is a close match for your field yet, "
                "so these are the nearest people we found.")
    for match_name, role, score in matches:
        person = attendees.loc[attendees["name"] == match_name].iloc[0]
        with st.container(border=True):
            st.markdown(f"**{match_name}** · {role}  \n:gray[{person['field']} · {person['stage']}]")
            st.progress(max(0.0, min(1.0, score)), text=f"Match score: {score:.2f}")
            st.caption(person["bio"])

with right:
    my_room = labels[me_index]
    st.subheader(f"Your breakout room: Room {my_room + 1}")
    st.markdown(f"*{rooms[my_room]}*")
    roommates = attendees[(labels == my_room) & (attendees.index != me_index)]
    st.write(", ".join(f"{n} ({r})" for n, r in zip(roommates["name"], roommates["role"]))
             or "Nobody else yet!")

    # Interactive map: hover over a dot to see who it is
    plot_df = attendees.assign(
        x=coords[:, 0], y=coords[:, 1],
        room=[f"Room {l + 1}: {rooms[l]}" for l in labels],
        size=np.where(attendees.index == me_index, 18, 8),
    )
    fig = px.scatter(plot_df, x="x", y="y", color="room", size="size",
                     color_discrete_sequence=px.colors.qualitative.Dark24,
                     hover_name="name",
                     hover_data={"role": True, "field": True, "x": False,
                                 "y": False, "size": False, "room": False})
    fig.add_annotation(x=plot_df["x"].iloc[-1], y=plot_df["y"].iloc[-1],
                       text="You", showarrow=True, arrowhead=2)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=700, legend_title_text="", margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", yanchor="top", y=-0.02, font_size=10))
    st.plotly_chart(fig, use_container_width=True)

with st.expander("See all rooms"):
    for room, room_name in rooms.items():
        members = attendees[labels == room]
        st.markdown(f"**Room {room + 1}: {room_name}**: " + ", ".join(members["name"]))

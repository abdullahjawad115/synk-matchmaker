# 🤝 AI Networking Matchmaker

**Find the right people to meet at a virtual event, automatically.**

Attendees write a short bio. The matchmaker uses NLP embeddings to understand what each person does, then:

- **Recommends who to meet**: each person's top matches, ranked by similarity
- **Creates breakout rooms**: groups attendees by topic with K-means clustering
- **Maps the event**: plots every attendee in 2D so you can see the communities

Built as a prototype feature for AI-powered virtual event platforms (conferences, trade shows and networking events), where helping people find each other is the hardest part.

---

## How it works

```
bios ──► embeddings ──► cosine similarity ──► top matches
                   ├──► K-means          ──► breakout rooms
                   └──► t-SNE            ──► 2D attendee map
```

| Step | What happens | Technique |
|---|---|---|
| 1. Embed | Each bio becomes a vector of numbers that captures its meaning | Sentence transformer [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (384 dimensions), with TF-IDF as an offline fallback |
| 2. Match | Every attendee is compared with every other attendee in one matrix multiplication | Cosine similarity |
| 3. Group | Attendees are split into topic-based breakout rooms, named after each room's most distinctive words | K-means clustering (unsupervised learning), silhouette score for evaluation |
| 4. Visualize | High-dimensional vectors are squashed to 2D while keeping neighbours close | t-SNE (PCA also supported) |
| 5. Serve | Visitors enter their own bio and instantly see matches, their room and their spot on the map | Streamlit + Plotly |

### Example matches

| Attendee | Top suggestion | Why it makes sense |
|---|---|---|
| Fintech startup founder | Fintech founder, then angel investor | Same space, plus someone who funds it |
| 3D artist | Game developer | Both build 3D worlds |
| Incoming CS student into AI | Two other AI-focused students, then an ML engineer | Peers first, then a mentor |

---

## What I learned

- **Keyword matching vs. meaning.** TF-IDF only matches exact words, so a "computer vision / object detection" engineer didn't match an "image recognition / CNN" engineer, even though they do the same job. Sentence transformers fix this because they encode meaning, not vocabulary.
- **Choosing k is a judgment call.** The silhouette score kept rising as rooms got smaller, so I picked the number of rooms that made sense for a real event, not just the highest score.
- **Always look at results.** The clustering and the 2D map sometimes disagreed (for example, an event organizer was clustered with founders but sat next to the 3D builders on the map). Inspecting both caught mistakes a single metric would hide.

---

## Run it yourself

```bash
git clone https://github.com/abdullahjawad115/synk-matchmaker.git
cd synk-matchmaker
pip install -r requirements.txt

python matchmaker.py      # prints matches and rooms, saves outputs/attendee_map.png
streamlit run app.py      # opens the web app in your browser
```

The first run downloads the embedding model (~80 MB). Without internet, it falls back to TF-IDF automatically.

## Project structure

```
├── app.py               # Streamlit web app
├── matchmaker.py        # Core ML: embeddings, matching, clustering, visualization
├── data/attendees.csv   # 34 sample attendees (name, role, bio)
└── requirements.txt
```

## Ideas for next steps

- [ ] LLM-generated icebreakers for each match ("You both work on computer vision. Ask about their drone project.")
- [ ] Learn from feedback: re-rank matches using 👍/👎 from attendees
- [ ] Match on goals as well as bios (e.g. *founder looking for investors* ↔ *investor looking for startups*)
- [ ] Scale to thousands of attendees with a vector database

---

Built by **Abdullah Jawad**. [GitHub](https://github.com/abdullahjawad115)

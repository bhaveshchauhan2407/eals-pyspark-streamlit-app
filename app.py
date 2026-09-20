import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import FastALSConfig
from src.data_loader import (
    dataframe_to_events,
    dataframe_to_interactions,
    load_events,
    load_interactions,
)
from src.evaluate import hit_rate_at_k, ndcg_at_k, online_protocol_metrics
from src.model import FastALSModel
from src.online_split import chronological_90_10_split
from src.recommender import recommend_top_k
from src.split import leave_one_out_split
from src.train import run_one_iteration

# ---------------------------------------------------------------------------
# Project settings: edit these if a name or link changes
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "yelp.rating"
RESULTS_DIR = ROOT / "results"

THIS_REPO_URL = "https://github.com/bhaveshchauhan2407/eals-pyspark-streamlit-app"
M1_REPO_URL = "https://github.com/bhaveshchauhan2407/als-pyspark"
PAPER_CODE_URL = "https://github.com/hexiangnan/sigir16-eals"

TOP_K = 10
ONLINE_CHUNK = 50  # events processed between two updates of the online chart
SCALABILITY_UPDATES = 100  # online updates timed per dataset size

st.set_page_config(page_title="eALS recommender", page_icon="🍽️")


# ---------------------------------------------------------------------------
# Cached data loading (reloading only happens when the row limit changes)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading the Yelp data...")
def get_interactions(limit_rows: int) -> pd.DataFrame:
    return load_interactions(str(DATA_PATH), limit_rows=limit_rows)


@st.cache_data(show_spinner="Loading the Yelp events...")
def get_events(limit_rows: int) -> pd.DataFrame:
    return load_events(str(DATA_PATH), limit_rows=limit_rows)


def build_config(settings: dict) -> FastALSConfig:
    return FastALSConfig(
        factors=settings["factors"],
        max_iter=settings["max_iter"],
        reg=settings["reg"],
        w0=settings["w0"],
        alpha=settings["alpha"],
        top_k=TOP_K,
        show_progress=False,
        show_loss=False,
    )


def train_with_live_loss(model: FastALSModel, n_iter: int) -> list:
    """Train with the M1 training step, drawing the loss after each iteration."""
    progress = st.progress(0.0, text="Starting training...")
    chart_slot = st.empty()
    losses = []
    for iteration in range(n_iter):
        run_one_iteration(model)
        losses.append(model.loss())
        progress.progress(
            (iteration + 1) / n_iter,
            text=f"Training: iteration {iteration + 1} of {n_iter}",
        )
        chart_slot.line_chart(loss_frame(losses), x="Iteration", y="Training loss")
    progress.empty()
    chart_slot.empty()
    return losses


def loss_frame(losses: list) -> pd.DataFrame:
    return pd.DataFrame(
        {"Iteration": range(1, len(losses) + 1), "Training loss": losses}
    )


def settings_caption(settings: dict) -> str:
    return (
        f"{settings['rows']:,} rows, K = {settings['factors']}, "
        f"{settings['max_iter']} iterations, w0 = {settings['w0']}, "
        f"α = {settings['alpha']}, λ = {settings['reg']}"
    )


# ---------------------------------------------------------------------------
# Introduction
# ---------------------------------------------------------------------------

st.title("Recommending Yelp Businesses With Fast Matrix Factorization")
st.caption(
    "Bhavesh Chauhan. Final project for Tooling for Data Scientists, "
    "MScT Data Science and AI for Business (HEC Paris and École Polytechnique)."
)

st.subheader("What this project is")
st.markdown(
    f"""
In my first year (M1), my project partner and I reimplemented the **eALS** algorithm
from He et al., *Fast Matrix Factorization for Online Recommendation with
Implicit Feedback* (SIGIR 2016). The authors published their code in Java.
Our task in the Database Management course was to rebuild it in Python with
**PySpark**, using Spark DataFrames and RDDs for the data pipeline and NumPy for
training, and to reproduce the paper's experiments on the Yelp dataset.

This app packages that work so anyone can explore the data, train the model
and look at its recommendations, with nothing to install except Docker.
"""
)

st.subheader("Where recommenders like this are used")
st.markdown(
    """
Every time Netflix suggests a show, Spotify builds a playlist or Amazon shows
"customers also bought", a recommender system is ranking items for you. These
systems mostly learn from **implicit feedback**: what you clicked, played or
bought, rather than star ratings, because people rarely rate things. That is
the setting eALS is designed for. Here, a user "interacting" with a Yelp
business simply means they reviewed it.
"""
)

st.subheader("ALS in plain words")
st.markdown(
    """
Picture a huge table with one row per user and one column per business, where
a tick means the user reviewed that business. Almost every cell is empty.
**Matrix factorization** describes each user and each business with a short
list of K hidden numbers. E.g., it could be tastes such as "likes cheap eats"
or "into nightlife", but it is the model discovers them on its own. A user's
predicted interest in a business is how well their two lists match.

**Alternating least squares (ALS)** learns these numbers by taking turns: it
holds the businesses fixed and finds the best numbers for each user, then holds
the users fixed and does the same for the businesses, and repeats. Each turn
has an exact answer, so there is no learning rate to tune.

The paper's element-wise version, **eALS**, adds three ideas:

- it updates one number at a time, which avoids costly matrix inversions and
  makes it K times faster than standard ALS;
- it treats an empty cell for a popular business as stronger evidence of
  disinterest than an empty cell for an obscure one (controlled by α);
- it absorbs a new interaction instantly by updating only that user and that
  business, with no full retraining. This is the "online" part.
"""
)

st.subheader("What changed from the M1 version")
st.markdown(
    """
- **Spark was replaced by pandas for data loading.** The dataset fits in
  memory and training already ran locally in NumPy, so Spark added a Java
  dependency and a much heavier container without making anything faster.
  The Spark version remains available in the M1 repository.
- **Loading is now deterministic.** The same settings always select the same
  rows, so every result can be reproduced.
- **New in this version:** unit tests for the data pipeline, continuous
  integration on every push, this Streamlit app and a Docker image.
- **The eALS algorithm itself is unchanged.**
"""
)

st.subheader("Credits and links")
st.markdown(
    f"""
Unlike this project, the original M1 project was co-authored with another student from our program as my project partner and under the
supervision of Prof. Dario Colazzo at Polytechnique. The algorithm and the Yelp data file come from
the paper's authors.
"""
)
link_cols = st.columns(3)
link_cols[0].link_button("This project's repository", THIS_REPO_URL)
link_cols[1].link_button("Original M1 project (PySpark)", M1_REPO_URL)
link_cols[2].link_button("Paper authors' code and data", PAPER_CODE_URL)

st.divider()

if not DATA_PATH.exists():
    st.error(
        f"The data file was not found at `{DATA_PATH}`. "
        "Place `yelp.rating` in the `data/` folder and reload the page."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar: settings shared by the training and online tabs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    st.caption(
        "Used by every tab that trains a model. The defaults are the M1 "
        "baseline, chosen by grid search. See the Terminology tab for what "
        "each setting means."
    )
    settings = {
        "rows": st.select_slider(
            "Rows of data", options=[5000, 10000, 20000, 30000, 50000], value=10000
        ),
        "factors": st.select_slider(
            "Latent factors (K)", options=[4, 8, 12, 16, 24, 32], value=12
        ),
        "max_iter": st.slider("Training iterations", 1, 20, 5),
        "w0": st.select_slider(
            "Missing-data weight (w0)", options=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0], value=1.0
        ),
        "alpha": st.select_slider(
            "Popularity exponent (α)",
            options=[0.0, 0.1, 0.25, 0.4, 0.5, 0.75, 1.0],
            value=0.5,
        ),
        "reg": st.number_input(
            "Regularization (λ)", min_value=0.001, max_value=1.0, value=0.1, step=0.01
        ),
    }
    if settings["rows"] > 20000:
        st.warning(
            "Large datasets can take several minutes, because the training "
            "code is written in plain Python."
        )

(
    tab_terms,
    tab_data,
    tab_train,
    tab_recs,
    tab_online,
    tab_scale,
    tab_m1,
) = st.tabs(
    [
        "Terminology",
        "Data",
        "Train and evaluate",
        "Recommendations",
        "Online updates",
        "Scalability",
        "M1 results",
    ]
)


# ---------------------------------------------------------------------------
# Tab 0: terminology
# ---------------------------------------------------------------------------

with tab_terms:

    st.subheader("Basic ideas")
    st.markdown(
        """
**Interaction.** One user reviewing one business. The star rating is ignored:
only the fact that the review happened is used. This is called *implicit
feedback*, and it is how most real recommenders learn, from clicks, plays or
purchases rather than ratings.

**Hidden interactions (test set).** To judge a recommender fairly, some
interactions are hidden during training, and afterwards we check whether the
model can predict them. The model never sees them while learning.

**Offline protocol.** Used in the *Train and evaluate* and *Recommendations*
tabs. For every user with at least two interactions, one interaction (their
last one in the data file) is hidden. This is called *leave-one-out*. The
model is trained once on everything else, then evaluated.

**Online protocol.** Used in the *Online updates* tab. Events are sorted by
time. The model trains on the oldest 90%, then the newest 10% arrive one by
one, as they would in a live system. For each event, the model first makes its
recommendations, is checked against what the user actually did, and only then
learns from the event.

**Cold start.** When a user or business appears for the first time during the
online stream, the model has never seen them before. It creates a fresh set of
numbers for them on the fly.
"""
    )

    st.subheader("Settings in the sidebar")
    st.markdown(
        """
**Rows of data.** How much of the Yelp file to use. The offline tabs take the
first N unique user–business pairs in the file, and the online tab takes the N
oldest events. More rows give the model more to learn from, but everything
runs slower. The full file has about 731,000 interactions. The app stops at
50,000 because the training code is written in plain Python.

**Latent factors (K).** The length of the list of hidden "taste" numbers
learned for each user and each business. A small K can only capture broad
patterns, such as popular versus niche. A larger K can capture finer
distinctions, but it needs more data, can start memorising noise, and is
slower: running time grows roughly with K².

**Training iterations.** One iteration is one full pass in which the model
updates the numbers of every user, then of every business. Each pass improves
the fit. Most of the improvement happens in the first few passes, after which
the loss flattens out.

**Missing-data weight (w0).** How strongly the model treats an empty cell, a
business the user has *not* reviewed, as a sign that the user is not
interested. With w0 close to 0, empty cells are almost ignored and the model
may end up predicting that everyone likes everything. With a large w0, the
model pushes all its predictions for unreviewed businesses towards zero. The
paper calls this c₀.

**Popularity exponent (α).** How the missing-data weight is shared between
businesses. With α = 0, the empty cells of every business count equally. As α
grows, the empty cells of popular businesses count more. The reasoning: if a
user has never reviewed a very popular restaurant, they have probably heard of
it and chosen not to go, whereas not reviewing an obscure one tells us little.
The paper found that α around 0.4 to 0.5 works best.

**Regularization (λ).** A penalty on large values in the taste lists. It stops
the model from fitting the training data too closely (*overfitting*), which
would make it memorise past reviews instead of learning patterns that carry
over to new ones. A larger λ gives a simpler, more cautious model.

**Weight of new events (w_new).** Online tab only. During training, every
interaction has weight 1. When a new event arrives during the stream, it gets
weight w_new instead, so the model reacts more strongly to fresh behaviour,
which is usually a better signal of a user's current taste. The paper found
that w_new = 4 works best on Yelp.
"""
    )

    st.subheader("Numbers the app reports")
    st.markdown(
        f"""
**Top-{TOP_K} and "@{TOP_K}".** For each user, the model gives a score to every
business they have not reviewed yet, sorts them, and recommends the {TOP_K}
highest. "@{TOP_K}" means a metric only looks at these top {TOP_K}
recommendations: a hidden business ranked 11th counts as a miss. This cutoff
has nothing to do with K, the number of latent factors. The paper used @100;
the M1 project and this app use the stricter @{TOP_K}.

**Score.** The model's predicted interest of a user in a business: how well
their two taste lists match. Scores are not probabilities, and only their
order matters, since that order decides the ranking.

**Hit rate (HR@{TOP_K}).** The share of hidden interactions for which the
hidden business appears anywhere in the user's top {TOP_K}. E.g.,
HR@{TOP_K} = 0.05 means the hidden business was recommended in 5% of cases.
That can look low, but with around 5,000 businesses to choose from, picking
{TOP_K} at random would succeed only about 0.2% of the time
({TOP_K} ÷ 5,000).

**NDCG@{TOP_K} (normalized discounted cumulative gain).** Like the hit rate,
but it also rewards *where* the hit appears in the list. A hit at rank r
scores 1 / log₂(r + 1): 1.0 at rank 1, 0.63 at rank 2, 0.5 at rank 3, down to
0.29 at rank 10, and 0 for a miss. NDCG@{TOP_K} is the average of these
scores, so it is never higher than HR@{TOP_K}. The closer the two are, the
nearer the top of the list the hits are.

**Training loss.** The quantity the algorithm tries to make as small as
possible while learning. It adds up three parts: the error on reviewed pairs
(where the model should predict close to 1), a penalty for predicting high
scores on empty cells (set by w0 and α), and the regularization penalty (set by
λ). A lower loss means the model fits its training data better. Two cautions:
losses are only comparable between runs with the same w0, α and λ, because
those settings change what is being measured; and a lower loss does not
guarantee better recommendations. The M1 experiments found several cases where
the loss kept falling while HR and NDCG did not improve.

**New users and businesses seen.** Online tab only. How many cold-start users
and businesses appeared for the first time during the stream.

**Training time and running time.** Seconds measured on the computer running
this app. They depend on the machine, so compare times within one session
rather than across computers.
"""
    )


# ---------------------------------------------------------------------------
# Tab 1: data explorer
# ---------------------------------------------------------------------------

with tab_data:
    df = get_interactions(settings["rows"])
    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()
    sparsity = 1 - len(df) / (n_users * n_items)

    st.markdown(
        "Each row is one user who reviewed one business. Ratings are ignored "
        "and repeated reviews of the same business count once."
    )
    cols = st.columns(4)
    cols[0].metric("Interactions", f"{len(df):,}")
    cols[1].metric("Users", f"{n_users:,}")
    cols[2].metric("Businesses", f"{n_items:,}")
    cols[3].metric("Empty cells", f"{sparsity:.2%}")

    st.subheader("A few businesses get most of the reviews")
    popularity = df["item_id"].value_counts().reset_index(drop=True)
    top_share = popularity.head(max(1, len(popularity) // 10)).sum() / len(df)
    st.line_chart(
        pd.DataFrame(
            {
                "Business rank by popularity": range(1, len(popularity) + 1),
                "Number of reviews": popularity.values,
            }
        ),
        x="Business rank by popularity",
        y="Number of reviews",
    )
    st.caption(
        f"The most popular 10% of businesses receive {top_share:.0%} of all "
        "interactions. This long tail is why eALS weights missing data by "
        "popularity (α)."
    )

    st.subheader("First rows")
    st.dataframe(df.head(20), hide_index=True)


# ---------------------------------------------------------------------------
# Tab 2: offline training and evaluation
# ---------------------------------------------------------------------------

with tab_train:
    st.markdown(
        "For each user, one business (their last one in the data file) is "
        "hidden from training. After "
        "training, the model recommends 10 businesses per user, and we check "
        "whether the hidden one is among them."
    )
    st.markdown(
        f"- **HR@{TOP_K}** (hit rate): share of users whose hidden business "
        "appears in their top 10.\n"
        f"- **NDCG@{TOP_K}**: like the hit rate, but a hit at rank 1 counts "
        "more than a hit at rank 10."
    )

    if st.button("Train model", type="primary", key="train_button"):
        interactions = dataframe_to_interactions(get_interactions(settings["rows"]))
        train, test = leave_one_out_split(interactions)
        model = FastALSModel(interactions=train, config=build_config(settings))

        start = time.time()
        losses = train_with_live_loss(model, settings["max_iter"])
        train_seconds = time.time() - start

        with st.spinner("Scoring the test users..."):
            hr = hit_rate_at_k(model, test, k=TOP_K)
            ndcg = ndcg_at_k(model, test, k=TOP_K)

        st.session_state["offline"] = {
            "model": model,
            "train": train,
            "test": test,
            "losses": losses,
            "hr": hr,
            "ndcg": ndcg,
            "seconds": train_seconds,
            "settings": dict(settings),
        }

    result = st.session_state.get("offline")
    if result is None:
        st.info("Choose settings in the sidebar, then select **Train model**.")
    else:
        st.caption("Trained with " + settings_caption(result["settings"]))
        cols = st.columns(4)
        cols[0].metric(f"HR@{TOP_K}", f"{result['hr']:.4f}")
        cols[1].metric(f"NDCG@{TOP_K}", f"{result['ndcg']:.4f}")
        cols[2].metric("Final loss", f"{result['losses'][-1]:,.1f}")
        cols[3].metric("Training time", f"{result['seconds']:.1f} s")
        st.line_chart(loss_frame(result["losses"]), x="Iteration", y="Training loss")
        st.caption(
            f"Trained on {len(result['train']):,} interactions, evaluated on "
            f"{len(result['test']):,} test users. The M1 report used HR@10 and "
            "NDCG@10 too, so these numbers are directly comparable to it."
        )


# ---------------------------------------------------------------------------
# Tab 3: recommendations for one user
# ---------------------------------------------------------------------------

with tab_recs:
    result = st.session_state.get("offline")
    if result is None:
        st.info("Train a model in the **Train and evaluate** tab first.")
    else:
        model = result["model"]
        held_out = dict(result["test"])  # user -> hidden business

        st.markdown(
            "Pick a user to see the businesses they reviewed, the one hidden "
            "from training, and what the model recommends. The Yelp file uses "
            "anonymised numbers instead of names."
        )
        user = st.selectbox("User", options=sorted(held_out), key="rec_user")

        u = model.user_to_index[user]
        history = [model.index_to_item[i] for i in model.user_items[u]]
        hidden = held_out[user]
        recs = recommend_top_k(model, user, k=TOP_K)

        cols = st.columns(2)
        cols[0].metric("Businesses reviewed in training", len(history))
        cols[1].metric("Hidden business", hidden)
        st.caption("Reviewed in training: " + ", ".join(str(i) for i in history))

        if hidden in recs:
            st.success(
                f"Hit: the hidden business is recommended at rank "
                f"{recs.index(hidden) + 1}."
            )
        else:
            st.warning(
                f"Miss: the hidden business is not in this user's top {TOP_K}."
            )

        st.dataframe(
            pd.DataFrame(
                {
                    "Rank": range(1, len(recs) + 1),
                    "Business": recs,
                    "Score": [
                        round(model.predict(u, model.item_to_index[i]), 4) for i in recs
                    ],
                    "Hidden business": ["Yes" if i == hidden else "" for i in recs],
                }
            ),
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# Tab 4: online protocol
# ---------------------------------------------------------------------------

with tab_online:
    st.markdown(
        "This reproduces the paper's online protocol. The model is trained on "
        "the oldest 90% of events. Then the newest 10% arrive one at a time: "
        "for each event, the model first recommends 10 businesses to that user, "
        "we check whether the business they actually reviewed is among them, "
        "and only then does the model learn from the event with a quick local "
        "update."
    )
    w_new = st.select_slider(
        "Weight of new events (w_new)",
        options=[1.0, 2.0, 4.0, 8.0],
        value=4.0,
        help="How much more a new event counts than a training interaction. "
        "The paper found 4 works best on Yelp.",
    )

    if st.button("Run online simulation", type="primary", key="online_button"):
        events = dataframe_to_events(get_events(settings["rows"]))
        train, stream = chronological_90_10_split(events)
        model = FastALSModel(interactions=train, config=build_config(settings))
        users_before, items_before = model.user_count, model.item_count

        losses = train_with_live_loss(model, settings["max_iter"])

        progress = st.progress(0.0, text="Streaming new events...")
        chart_slot = st.empty()
        hits, ndcg_sum, done, curve = 0.0, 0.0, 0, []
        for start in range(0, len(stream), ONLINE_CHUNK):
            chunk = stream[start : start + ONLINE_CHUNK]
            metrics = online_protocol_metrics(
                model, chunk, k=TOP_K, w_new=w_new, online_iter=1
            )
            hits += metrics["hr_at_k"] * len(chunk)
            ndcg_sum += metrics["ndcg_at_k"] * len(chunk)
            done += len(chunk)
            curve.append(
                {
                    "Events processed": done,
                    f"HR@{TOP_K} so far": hits / done,
                    f"NDCG@{TOP_K} so far": ndcg_sum / done,
                }
            )
            progress.progress(
                done / len(stream), text=f"Streaming: {done:,} of {len(stream):,} events"
            )
            chart_slot.line_chart(pd.DataFrame(curve), x="Events processed")
        progress.empty()
        chart_slot.empty()

        st.session_state["online"] = {
            "curve": pd.DataFrame(curve),
            "hr": hits / done if done else 0.0,
            "ndcg": ndcg_sum / done if done else 0.0,
            "events": done,
            "new_users": model.user_count - users_before,
            "new_items": model.item_count - items_before,
            "settings": {**settings, "w_new": w_new},
        }

    result = st.session_state.get("online")
    if result is None:
        st.info("Choose settings in the sidebar, then select **Run online simulation**.")
    else:
        st.caption(
            "Run with " + settings_caption(result["settings"])
            + f", w_new = {result['settings']['w_new']}"
        )
        cols = st.columns(4)
        cols[0].metric(f"HR@{TOP_K}", f"{result['hr']:.4f}")
        cols[1].metric(f"NDCG@{TOP_K}", f"{result['ndcg']:.4f}")
        cols[2].metric("New users seen", f"{result['new_users']:,}")
        cols[3].metric("New businesses seen", f"{result['new_items']:,}")
        if not result["curve"].empty:
            st.line_chart(result["curve"], x="Events processed")
        st.caption(
            f"{result['events']:,} events streamed. New users and businesses "
            "appear for the first time during the stream: the model adds them "
            "on the fly (the cold-start case)."
        )


# ---------------------------------------------------------------------------
# Tab: scalability (running time versus data size)
# ---------------------------------------------------------------------------

with tab_scale:
    st.markdown(
        "This mirrors the "
        "scalability section of the paper and of the M1 report. For each "
        "dataset size, the app trains a fresh model with the sidebar settings "
        "(the sidebar's row setting is ignored here) and measures two things:"
    )
    st.markdown(
        "- **Training time:** for all iterations, and per iteration.\n"
        f"- **Online update time:** the average time to absorb one new "
        f"interaction, measured on {SCALABILITY_UPDATES} hidden interactions."
    )
    st.markdown(
        "In theory, the time per iteration grows roughly in proportion to the "
        "number of users, businesses and interactions, while an online update "
        "depends only on the history of the one user and business involved. "
        "In practice, this implementation recomputes a summary of all "
        "businesses whenever a brand-new business appears (a limitation noted "
        "in M1), so online updates can also slow down on larger data."
    )

    sizes = st.multiselect(
        "Dataset sizes (rows)",
        options=[5000, 10000, 20000, 30000, 50000],
        default=[5000, 10000, 20000],
        key="scal_sizes",
    )
    if max(sizes, default=0) > 20000:
        st.warning("Sizes above 20,000 rows can take several minutes each.")

    if st.button(
        "Measure running time", type="primary", key="scal_button", disabled=not sizes
    ):
        measurements = []
        progress = st.progress(0.0, text="Starting...")
        for step, rows in enumerate(sorted(sizes)):
            progress.progress(
                step / len(sizes), text=f"Training on {rows:,} rows..."
            )
            interactions = dataframe_to_interactions(get_interactions(rows))
            train, test = leave_one_out_split(interactions)
            model = FastALSModel(interactions=train, config=build_config(settings))

            start = time.perf_counter()
            for _ in range(settings["max_iter"]):
                run_one_iteration(model)
            train_seconds = time.perf_counter() - start

            sample = test[:SCALABILITY_UPDATES]
            start = time.perf_counter()
            for user, item in sample:
                model.update_model(user, item, w_new=4.0, online_iter=1)
            update_ms = (time.perf_counter() - start) / max(1, len(sample)) * 1000

            measurements.append(
                {
                    "Rows": rows,
                    "Interactions": len(interactions),
                    "Users": len({u for u, _ in interactions}),
                    "Businesses": len({i for _, i in interactions}),
                    "Training time (s)": round(train_seconds, 2),
                    "Time per iteration (s)": round(
                        train_seconds / settings["max_iter"], 3
                    ),
                    "Online update (ms)": round(update_ms, 2),
                }
            )
        progress.empty()
        st.session_state["scalability"] = {
            "table": pd.DataFrame(measurements),
            "settings": dict(settings),
        }

    result = st.session_state.get("scalability")
    if result is None:
        st.info("Choose dataset sizes, then select **Measure running time**.")
    else:
        table = result["table"]
        st.caption(
            f"K = {result['settings']['factors']}, "
            f"{result['settings']['max_iter']} iterations, "
            f"w0 = {result['settings']['w0']}, α = {result['settings']['alpha']}"
        )
        st.subheader("Training time")
        st.line_chart(table, x="Rows", y="Training time (s)")
        st.subheader("Average time of one online update")
        st.line_chart(table, x="Rows", y="Online update (ms)")
        st.dataframe(table, hide_index=True)
        st.caption(
            "Compare with the Dataset size experiment in the M1 results tab. "
            "Absolute times depend on the computer, but the shape of the "
            "curves should be similar."
        )


# ---------------------------------------------------------------------------
# Tab 5: results from the M1 report
# ---------------------------------------------------------------------------

SWEEPS = {
    "Latent factors (K)": ("k_sweep", "factors"),
    "Popularity exponent (α)": ("alpha_sweep", "alpha"),
    "Missing-data weight (w0)": ("w0_sweep", "w0"),
    "Training iterations": ("iteration_sweep", "max_iter"),
    "Dataset size": ("scalability_sweep", "row_limit"),
}

with tab_m1:
    st.markdown(
        "These are the saved results of the M1 experiments, where one setting "
        "was changed at a time while the others stayed at the baseline. They "
        "are loaded from the `results/` folder, so nothing is recomputed."
    )
    choice = st.selectbox("Experiment", options=list(SWEEPS), key="m1_sweep")
    prefix, column = SWEEPS[choice]

    for protocol in ["offline", "online"]:
        path = RESULTS_DIR / f"{prefix}_{protocol}.csv"
        st.subheader(f"{protocol.capitalize()} protocol")
        if not path.exists():
            st.info(f"`{path.name}` was not found in the `results/` folder.")
            continue
        sweep = pd.read_csv(path)
        metric_cols = [c for c in ["hr_at_k", "ndcg_at_k"] if c in sweep.columns]
        if column in sweep.columns and metric_cols:
            chart = sweep[[column] + metric_cols].rename(
                columns={"hr_at_k": f"HR@{TOP_K}", "ndcg_at_k": f"NDCG@{TOP_K}"}
            )
            st.line_chart(chart, x=column)
        shown = [
            c
            for c in [column, "hr_at_k", "ndcg_at_k", "final_loss", "runtime_seconds"]
            if c in sweep.columns
        ]
        st.dataframe(sweep[shown] if shown else sweep, hide_index=True)

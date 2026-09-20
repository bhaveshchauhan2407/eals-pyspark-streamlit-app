# eALS recommender: fast matrix factorization on Yelp data

This is an interactive Streamlit app for **eALS**, the recommendation algorithm from
He et al., *Fast Matrix Factorization for Online Recommendation with Implicit
Feedback* (SIGIR 2016), trained and evaluated on Yelp data. The app is tested,
checked automatically on every push, and packaged as a Docker image.

Final project for **Tooling for Data Scientists**, MScT Data Science and AI for
Business X-HEC, by Bhavesh Chauhan.

**Links**

- Docker image: https://hub.docker.com/r/bhaveshchauhan24/eals-pyspark-streamlit-app
- Original M1 project (PySpark): https://github.com/bhaveshchauhan2407/als-pyspark
- Paper authors' Java code and data: https://github.com/hexiangnan/sigir16-eals

---

The only requirement is Docker.

```bash
docker run -p 8501:8501 bhaveshchauhan24/eals-pyspark-streamlit-app:1.0.0
```

Then open http://localhost:8501 in your browser.

## Run it locally without Docker

Requires Python 3.13.7.

```bash
git clone https://github.com/bhaveshchauhan2407/eals-pyspark-streamlit-app.git
cd eals-pyspark-streamlit-app

python -m venv .venv
source .venv/bin/activate  

pip install -r requirements.txt
streamlit run app.py
```

## Run the tests

```bash
pytest
```

This runs all unit tests and prints a coverage report (settings are in
`pytest.ini`).

---

## About the project

### What this project is

In my first year (M1), my project partner and I reimplemented the
eALS algorithm for the Database Management course. The paper's authors
published their code in Java. Our task was to rebuild it in Python with
**PySpark**, using Spark DataFrames and RDDs for the data pipeline and NumPy for
training, and to reproduce the paper's experiments on the Yelp dataset.

This project packages that work as an app that anyone can run.

### Where recommenders like this are used

Every time Netflix suggests a show, Spotify builds a playlist or Amazon shows
"customers also bought", a recommender system is ranking items. These
systems mostly learn from **implicit feedback**: what you clicked, played or
bought, rather than star ratings, because people rarely rate things. That is
the setting eALS is designed for. Here, a user "interacting" with a Yelp
business simply means they reviewed it.

### ALS in plain words

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

### What changed from the M1 version

- **Spark was replaced by pandas for data loading.** The dataset fits in
  memory and training already ran locally in NumPy, so Spark added a Java
  dependency and a much heavier container without making anything faster.
  The Spark version remains available in the
  [M1 repository](https://github.com/bhaveshchauhan2407/als-pyspark).
- **Loading is now deterministic.** The same settings always select the same
  rows, so every result can be reproduced.
- **New in this version:** unit tests for the data pipeline, continuous
  integration on every push, the Streamlit app and a Docker image.
- **The eALS algorithm itself is unchanged.**

---

## Using the app

The top of the page introduces the project. Model settings are in the
sidebar, and the rest of the app is organised in tabs:

| Tab | What it does |
|---|---|
| Terminology | Explains every setting and every number the app reports |
| Data | Counts of interactions, users and businesses, and the popularity long tail |
| Train and evaluate | Trains the model with a live loss chart, then reports HR@10 and NDCG@10 |
| Recommendations | Shows one user's history, their hidden business and their top 10 |
| Online updates | Simulates new events arriving one by one, with the model updating after each |
| Scalability | Measures how training and online update times grow with data size |
| M1 results | Interactive charts of the experiments from the M1 report |

---

## Terminology

### Basic ideas

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

### Settings

| Setting | Meaning |
|---|---|
| **Rows of data** | How much of the Yelp file to use: the first N unique user–business pairs (offline) or the N oldest events (online). More rows give the model more to learn from but make everything slower. |
| **Latent factors (K)** | The length of the list of hidden "taste" numbers per user and per business. A small K only captures broad patterns; a larger K captures finer ones but needs more data, can memorise noise, and is slower (time grows roughly with K²). |
| **Training iterations** | One iteration is a full pass updating every user, then every business. Most of the improvement happens in the first few passes. |
| **Missing-data weight (w0)** | How strongly an empty cell (a business the user has *not* reviewed) counts as a sign of disinterest. Near 0, empty cells are almost ignored; a large w0 pushes all predictions for unreviewed businesses towards zero. Called c₀ in the paper. |
| **Popularity exponent (α)** | How that weight is shared between businesses. At α = 0 all businesses count equally; as α grows, empty cells of popular businesses count more, since a user who never reviewed a very popular place probably chose not to go. The paper found α ≈ 0.4–0.5 works best. |
| **Regularization (λ)** | A penalty on large values in the taste lists, which stops the model from memorising the training data (*overfitting*). A larger λ gives a simpler, more cautious model. |
| **Weight of new events (w_new)** | Online only. Training interactions have weight 1; a new event gets weight w_new, so the model reacts more strongly to fresh behaviour. The paper found w_new = 4 works best on Yelp. |

### Metrics

| Metric | Meaning |
|---|---|
| **Top-10 and "@10"** | The model scores every business a user has not reviewed and recommends the 10 highest. "@10" means a metric only looks at these 10: a hidden business ranked 11th is a miss. This cutoff is unrelated to K, the number of latent factors. The paper used @100; this project uses the stricter @10. |
| **Score** | The model's predicted interest of a user in a business. Not a probability: only the order matters. |
| **HR@10 (hit rate)** | The share of hidden interactions whose business appears anywhere in the user's top 10. With around 5,000 businesses, picking 10 at random would succeed only about 0.2% of the time. |
| **NDCG@10** | Like the hit rate, but it rewards hits near the top. A hit at rank r scores 1 / log₂(r + 1): 1.0 at rank 1, 0.63 at rank 2, down to 0.29 at rank 10, and 0 for a miss. Always at most HR@10. |
| **Training loss** | The quantity the algorithm minimises: errors on reviewed pairs, plus a penalty for high scores on empty cells (set by w0 and α), plus the regularization penalty. Only comparable between runs with the same w0, α and λ, and a lower loss does not guarantee better recommendations. |
| **Training time** | Seconds measured on the machine running the app, so only comparable within one session. |

---

## Project structure

```
.
├── .github/workflows/tests.yml   # CI: runs the tests on every push and pull request
├── app.py                        # Streamlit app
├── data/yelp.rating              # Yelp interactions (user, business, rating, timestamp)
├── results/                      # Saved experiment results from the M1 project
├── src/
│   ├── data_loader.py            # Reading, cleaning and filtering the data (pandas)
│   ├── split.py                  # Offline leave-one-out split
│   ├── online_split.py           # Online chronological 90/10 split
│   ├── model.py                  # eALS model
│   ├── train.py                  # Training loop
│   ├── evaluate.py               # HR@10, NDCG@10 and the online protocol
│   ├── recommender.py            # Top-10 recommendations
│   └── config.py, predict.py, utils.py
├── tests/                        # Unit tests (pytest)
├── Dockerfile, .dockerignore     # Container build
├── pytest.ini                    # Test and coverage settings
└── requirements.txt              # Pinned dependencies
```

## Testing scope

Following the assignment brief, unit tests cover the data pipeline: file
reading, cleaning and filtering (`data_loader.py`), the offline and online
train/test splits (`split.py`, `online_split.py`), and the filtering of
already-seen items in recommendations (`recommender.py`). All of these are at
100% coverage. The tests use small hand-written input files, so the expected
result of every test is known exactly and the suite runs in under a second.

The eALS training algorithm itself (`model.py`, `train.py`) is outside the
scope of this assignment and is not unit-tested, so total coverage is about
46%. CI fails if coverage drops below 45%, which guards against tests being
removed or new code being added without tests.

## Reproducibility

- **Pinned dependencies** in `requirements.txt`, so every install gets the same
  library versions.
- **The same Python version** locally, in CI and in the Docker image.
- **A pinned CI runner** (`ubuntu-24.04`) rather than a moving `ubuntu-latest`.
- **Deterministic data loading** and fixed random seeds, so the same settings
  give the same results.
- **Multi-platform image** Container is emulated so that it runs on both Intel/AMD and Apple Silicon.

## Credits

The original M1 project was co-authored with my project partner, another student in our program,
under the supervision of Prof. Dario Colazzo from École Polytechnique. The
eALS algorithm and the preprocessed Yelp data file are from He, Zhang, Kan and
Chua (2016), available in the
[authors' repository](https://github.com/hexiangnan/sigir16-eals).

# %% [markdown]
#
# # Classification with imbalanced datasets
#
# # TODO: describe what we call imbalanced datasets with classification settings

# %%
import pandas as pd
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1_000_000,
    n_classes=2,
    weights=[0.99, 0.01],
    random_state=0,
)
X, y = pd.DataFrame(X), pd.Series(y)

# %%
y.value_counts(normalize=True) * 100

# %% [markdown]
#
# Looking at the true target distribution, we therefore observe that the probability
# for a sample to be the positive class with label 1 is rare (~1%).

# %%
from sklearn.linear_model import LogisticRegression

model = LogisticRegression().fit(X, y)
y_proba = model.predict_proba(X)
y_proba = pd.DataFrame(y_proba, columns=["p(y=0)", "p(y=1)"])

# %%
y_proba.head()

# %%
y_proba.plot.hist(bins=100, figsize=(10, 5), subplots=True, layout=(1, 2), sharey=True)

# %%
from sklearn.calibration import CalibrationDisplay

CalibrationDisplay.from_estimator(model, X, y, n_bins=20, strategy="quantile")

# %%
from sklearn.metrics import classification_report

print(classification_report(y, model.predict(X)))

# %%
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(model, X, y)

# %% [markdown]
#
# # What people do and you should not do (naively)

# %%
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline

model = make_pipeline(
    RandomUnderSampler(sampling_strategy=0.1, random_state=0), LogisticRegression()
)
model.fit(X, y)

# %%
print(classification_report(y, model.predict(X)))

# %%
ConfusionMatrixDisplay.from_estimator(model, X, y)

# %%
CalibrationDisplay.from_estimator(model, X, y, n_bins=20, strategy="quantile")

# %%
from sklearn.calibration import CalibratedClassifierCV

model = CalibratedClassifierCV(
    make_pipeline(
        RandomUnderSampler(sampling_strategy=0.1, random_state=0), LogisticRegression()
    ),
    method="isotonic",
)
model.fit(X, y)

# %%
CalibrationDisplay.from_estimator(model, X, y, n_bins=20, strategy="quantile")

# %%
print(classification_report(y, model.predict(X)))

# %%
ConfusionMatrixDisplay.from_estimator(model, X, y)

# %% [markdown]
#
# # What you should look at

# %%
from sklearn.model_selection import FixedThresholdClassifier

model = FixedThresholdClassifier(
    LogisticRegression(), threshold=y.value_counts(normalize=True)[1]
)
model.fit(X, y)

# %%
CalibrationDisplay.from_estimator(model, X, y, n_bins=20, strategy="quantile")

# %%
print(classification_report(y, model.predict(X)))

# %%
ConfusionMatrixDisplay.from_estimator(model, X, y)

# %%
import numpy as np
from sklearn.metrics import get_scorer
from sklearn.metrics._scorer import _CurveScorer

thresholds = np.linspace(0, 1, 15)
precision_curve_scorer = _CurveScorer.from_scorer(
    get_scorer("precision"), response_method="predict_proba", thresholds=thresholds
)
recall_curve_scorer = _CurveScorer.from_scorer(
    get_scorer("recall"), response_method="predict_proba", thresholds=thresholds
)

precision_scores, precision_thresholds = precision_curve_scorer(model, X, y)
recall_scores, recall_thresholds = recall_curve_scorer(model, X, y)

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.plot(precision_thresholds, precision_scores, marker="+", label="Precision")
ax.plot(recall_thresholds, recall_scores, marker="+", label="Recall")

# Annotate threshold values on markers
for i, (threshold, score) in enumerate(zip(precision_thresholds, precision_scores)):
    ax.annotate(
        f"{threshold:.2f}",
        (threshold, score),
        textcoords="offset points",
        xytext=(5, 0),
        ha="left",
        fontsize=8,
    )

for i, (threshold, score) in enumerate(zip(recall_thresholds, recall_scores)):
    ax.annotate(
        f"{threshold:.2f}",
        (threshold, score),
        textcoords="offset points",
        xytext=(5, 0),
        ha="left",
        fontsize=8,
    )

ax.set(xlabel="Threshold", ylabel="Score")
ax.legend()

# %%
# Interactive version with plotly
import plotly.graph_objects as go

fig_plotly = go.Figure()
fig_plotly.add_trace(
    go.Scatter(
        x=precision_thresholds,
        y=precision_scores,
        mode="lines+markers",
        name="Precision",
        marker=dict(symbol="cross"),
        hovertemplate="Threshold: %{x:.2f}<br>Precision: %{y:.3f}",
    )
)
fig_plotly.add_trace(
    go.Scatter(
        x=recall_thresholds,
        y=recall_scores,
        mode="lines+markers",
        name="Recall",
        marker=dict(symbol="cross"),
        hovertemplate="Threshold: %{x:.2f}<br>Recall: %{y:.3f}",
    )
)
fig_plotly.update_layout(
    xaxis_title="Threshold",
    yaxis_title="Score",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="closest",
    width=600,
    height=500,
)
fig_plotly.show()

# %%
from sklearn.metrics import PrecisionRecallDisplay

PrecisionRecallDisplay.from_estimator(model, X, y)

# %%
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import TunedThresholdClassifierCV


def maximize_precision_under_constrained_recall(y_true, y_pred, recall_level):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    if recall < recall_level:
        return -np.inf
    return precision


model = TunedThresholdClassifierCV(
    estimator=LogisticRegression(),
    scoring=make_scorer(maximize_precision_under_constrained_recall, recall_level=0.3),
    n_jobs=-1,
).fit(X, y)

# %%
print(classification_report(y, model.predict(X)))

# %%
ConfusionMatrixDisplay.from_estimator(model, X, y)

# %%
model.best_threshold_

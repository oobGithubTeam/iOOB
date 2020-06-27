import numpy as np
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from strlearn.streams import StreamGenerator
from strlearn.evaluators import TestThenTrain
from strlearn.ensembles import SEA, OnlineBagging ,OOB, UOB
from strlearn.metrics import recall, specificity, balanced_accuracy_score

stream = StreamGenerator(n_classes=2,
                         n_chunks=200,
                         chunk_size=400,
                         n_features=10,
                         concept_sigmoid_spacing=5,
                         incremental=True,
                         n_drifts=1,
                         weights=[0.8, 0.2],
                         random_state=95)

clfs = [
    SEA(GaussianNB(), n_estimators=5),
    OnlineBagging(GaussianNB(), n_estimators=5),
    OOB(GaussianNB(), n_estimators=5),
    UOB(GaussianNB(), n_estimators=5),
]

clf_names = [
    "SEA",
    "OB",
    "OOB",
    "UOB",
]

metrics = [
    # recall,
    specificity,
    balanced_accuracy_score,
]

metrics_names = [
    # "recall",
    "specificity",
    "BAC",
]

evaluator = TestThenTrain(metrics)

evaluator.process(stream, clfs)

fig, ax = plt.subplots(1, len(metrics), figsize=(24, 8))
for m, metric in enumerate(metrics):
    ax[m].set_title(metrics_names[m])
    ax[m].set_ylim(0, 1)
    for i, clf in enumerate(clfs):
        ax[m].plot(evaluator.scores[i, :, m], label=clf_names[i])
    plt.ylabel("Metric")
    plt.xlabel("Chunk")
    ax[m].legend()
plt.savefig('example2.png')
plt.show()
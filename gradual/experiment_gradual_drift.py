from strlearn.streams import StreamGenerator
from strlearn.ensembles import SEA, OnlineBagging ,OOB, UOB
from sklearn.naive_bayes import GaussianNB
from strlearn.metrics import recall, specificity, balanced_accuracy_score
import numpy as np
from tqdm import tqdm
from strlearn.evaluators import TestThenTrain


mcargs = {
    "n_classes": 2,
    "n_chunks": 250,
    "chunk_size": 400,
    "n_features": 10,
}

streams = {
    "gradual1": StreamGenerator(**mcargs, n_drifts=1, concept_sigmoid_spacing=5, weights=[0.8, 0.2], random_state=14),
    "gradual2": StreamGenerator(**mcargs, n_drifts=1, concept_sigmoid_spacing=5, weights=[0.8, 0.2], random_state=67),
    "gradual3": StreamGenerator(**mcargs, n_drifts=1, concept_sigmoid_spacing=5, weights=[0.8, 0.2], random_state=95),
    "gradual4": StreamGenerator(**mcargs, n_drifts=1, concept_sigmoid_spacing=5, weights=[0.8, 0.2], random_state=234),
    "gradual5": StreamGenerator(**mcargs, n_drifts=1, concept_sigmoid_spacing=5, weights=[0.8, 0.2], random_state=876),
    "gradual6": StreamGenerator(**mcargs, n_drifts=1, concept_sigmoid_spacing=5, weights=[0.8, 0.2], random_state=1410),
    "gradual7": StreamGenerator(**mcargs, n_drifts=1, concept_sigmoid_spacing=5, weights=[0.8, 0.2], random_state=2137),
    "gradual8": StreamGenerator(**mcargs, n_drifts=1, concept_sigmoid_spacing=5, weights=[0.8, 0.2], random_state=3222),
    "gradual9": StreamGenerator(**mcargs, n_drifts=1, concept_sigmoid_spacing=5, weights=[0.8, 0.2], random_state=4367),
    "gradual10": StreamGenerator(**mcargs, n_drifts=1, concept_sigmoid_spacing=5, weights=[0.8, 0.2], random_state=5000),
}

clfs = [
    SEA(GaussianNB(), n_estimators=5),
    OnlineBagging(GaussianNB(), n_estimators=5),
    OOB(GaussianNB(), n_estimators=5),
    UOB(GaussianNB(), n_estimators=5),
]

metrics = [
    recall,
    specificity,
    balanced_accuracy_score,
]

recall_arr = np.array([[],[],[],[]])
specificity_arr = np.array([[],[],[],[]])
bac_arr = np.array([[],[],[],[]])

for stream_name in tqdm(streams):

    stream = streams[stream_name]
    evaluator = TestThenTrain(metrics)
    evaluator.process(stream, clfs)
    scores = np.mean(evaluator.scores, axis=1)
    npscores = np.array(scores)
    div = np.array_split(npscores, len(metrics), axis=1)

    recall_arr = np.concatenate((recall_arr, div[0]), axis=1)
    specificity_arr = np.concatenate((specificity_arr, div[1]), axis=1)
    bac_arr = np.concatenate((bac_arr, div[2]), axis=1)

    # print("recall: ", recall_arr, "\n")
    # print("specificity: ", specificity_arr, "\n")
    # print("bac_arr: ", bac_arr, "\n")

np.savetxt("gradual_recall.csv", recall_arr, delimiter=",")
np.savetxt("gradual_specificity.csv", specificity_arr, delimiter=",")
np.savetxt("gradual_bac.csv", bac_arr, delimiter=",")
# iOOB
Improved OOB algorithm in Python

This repository contains research on static data streams for metrics e.g recall, specificity and balanced accuracy. We compare here 4 algorithms which work with 
classifier ensemble like SEA (Streaming Ensemble Algorithm), OB (Online Bagging), OOB (Oversampling-based Online Bagging), UOB (Undersampling-based Online Bagging).

The purpose of this research is to compare algorithms: SEA, OB, OOB, UOB in the processing of imbalanced static data streams for metrics which give important information in case of imbalanced streams. Additionally, we want to check if concept drift affects on statistical results for given classification methods. To execute the experiment synthetic data generated using the "StreamGenerator()" function from the "stream-learn" library for the Python programming language has been used.

The tests were placed in three directories:
- sudden
- gradual
- incremental
 
Directories listed above contains these files:
- experiment_<drift_name>_drift.py - in this file streams for 4 algorithms and 3 metrics are created and processed. The results are finally saved to files named <drift_name> _ <metrics_name>.csv. These files store the results of the evaluation process for the metrics used in the experiment.
   
- statistical_tests_<drift_name>.py - this file contains statistical tests for the results received from the first part of the experiment located in the file "experiment_ <drift_name> _drift.py".
The results of the statistical tests are in the files: "stat_better_table_ <metrics_name>.npy". Unfortunately, the GitHub platform does not support reading files in this format - to read them, run the file in the PyCharm Community Edition. Read in rows, to see which method is statistically better. The code placed in the file also creates a table with the results placed in the "results.csv" file, and the content of this file is in the article.

- example.py - this file contains an example of experimental evaluation for a selected stream used in the experiment, but this time as a quality graph for a given metric from data chunks. Graphics received from this file are in "example1.png" and "exapmle2.png".

Our implementation of the OOB algorithm is in the "iOOB.py" file.

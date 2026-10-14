# Heart Disease Prediction with k-Nearest Neighbors

## Part I

### Introduction

The goal of this part was to predict whether a patient has heart disease using the Cleveland Heart Disease dataset from the UCI Machine Learning Repository. The dataset has 303 patients and 13 attributes, such as age, chest pain type, resting blood pressure, cholesterol, maximum heart rate and the results of a thallium stress test. The `num` column records the diagnosis on a scale from 0 (no disease) to 4. Since we only care whether disease is present or not, we collapsed values 1–4 into a single "disease" value, which turns this into a yes/no classification problem. That leaves 164 healthy patients and 139 with disease, so the two classes are fairly balanced.

We used k-nearest neighbors (kNN): to classify a patient, we find the k most similar patients we already know about and go with the majority. Because sklearn's `KNeighborsClassifier` was not allowed, we built our own version on top of sklearn's `NearestNeighbors`. The two questions we had to answer were which value of k to use and which attributes to include, with F1 score as our measure of success.

### Methods

**Cleaning the data.** Six patients had a missing value (`?`) in either `ca` or `thal`, so we dropped them, leaving 297 patients.

**Splitting the data.** We used `sklearn.model_selection.train_test_split` to randomly split the data into a training set of 237 patients (80%) and a test set of 60 patients (20%). The test set was put aside and not used at all while choosing k and the attributes, so that our final scores would be an honest estimate.

**Preparing the attributes.** kNN works by measuring distance between patients, so how each attribute is represented matters a lot. We handled them in three groups:

- *Categorical* attributes (`cp`, `restecg`, `slope`, `thal`) are labels rather than amounts. For example, `thal` uses 3, 6 and 7 for normal, fixed defect and reversible defect, and treating these as numbers would wrongly suggest 6 and 7 are "close". We one-hot encoded these so each category becomes its own 0/1 column.
- *Binary* attributes (`sex`, `fbs`, `exang`) are already 0/1, so we left them as they were.
- *Numerical* attributes (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`, `ca`) were standardized (subtract the mean, divide by the standard deviation). Without this, cholesterol, which is in the hundreds, would dominate every distance. The mean and standard deviation were calculated from the training set only and then applied to the test set.

**Our kNN model.** We wrote a small class, `MyKNN`, that stores the training patients in a `NearestNeighbors` object, finds the k closest patients for each new one, and predicts "disease" if more than half of them have it. We wrote it in the standard sklearn style (with `fit` and `predict`) so that we could still use `GridSearchCV` to search over k. We only tried odd values of k so that the vote could never tie.

**Baseline.** As a starting point, we used all 13 attributes and let `GridSearchCV` try every odd k from 1 to 41. For each k, it ran 10-fold cross-validation on the training set: train on 9 folds, compute F1 on the 10th, repeat for all 10 folds, and average.

**Choosing the attributes.** Using every attribute is not necessarily best, because weak attributes add noise to the distances. We used forward selection: start with no attributes, try adding each remaining attribute one at a time, run the same grid search over k for each candidate, and keep whichever one gives the highest cross-validated F1. We repeated this until all 13 attributes had been added, which gave us the best attribute set and best k for every size from 1 to 13. We then picked the size with the highest F1 overall.

**Final evaluation.** With k and the attributes fixed, we ran 10-fold cross-validation on the training set as described in the assignment. Each fold produced a model trained on 9/10 of the training set, and we tested every one of these 10 models on the held-out test set to get its precision, recall and F1. For this step we used a hand-written loop with `NearestNeighbors` and a majority vote, which does the same thing as `MyKNN`.

### Results

**Baseline.** With all 13 attributes, the best k was 7, with a cross-validated F1 of 0.800.

**Attribute selection.** The table below shows the best result for each number of attributes during forward selection (cross-validated F1 on the training set).

| # attributes | Attribute added | Best k | F1 |
|---|---|---|---|
| 1 | thal | 29 | 0.7431 |
| 2 | ca | 33 | 0.7884 |
| 3 | cp | 23 | 0.8093 |
| 4 | slope | 21 | 0.8316 |
| 5 | exang | 13 | 0.8445 |
| 6 | fbs | 13 | 0.8440 |
| **7** | **trestbps** | **41** | **0.8539** |
| 8 | chol | 15 | 0.8425 |
| 9 | age | 9 | 0.8330 |
| 10 | oldpeak | 15 | 0.8194 |
| 11 | sex | 11 | 0.8323 |
| 12 | thalach | 9 | 0.8317 |
| 13 | restecg | 7 | 0.7998 |

The score climbs quickly over the first few attributes, peaks at 7, and then drifts down as weaker attributes are added. The first attributes chosen (`thal`, `ca`, `cp`, `slope`, `exang`) are the stress test, fluoroscopy, chest pain and exercise results, which makes sense clinically.

**Final model.**

- **k = 41**
- **Attributes:** thal, ca, cp, slope, exang, fbs, trestbps

**Scores on the test set for each of the 10 fold models:**

| Fold | Precision | Recall | F1 |
|---|---|---|---|
| 1 | 0.7586 | 0.9565 | 0.8462 |
| 2 | 0.7857 | 0.9565 | 0.8627 |
| 3 | 0.8148 | 0.9565 | 0.8800 |
| 4 | 0.7419 | 1.0000 | 0.8519 |
| 5 | 0.7778 | 0.9130 | 0.8400 |
| 6 | 0.8077 | 0.9130 | 0.8571 |
| 7 | 0.8077 | 0.9130 | 0.8571 |
| 8 | 0.8462 | 0.9565 | 0.8980 |
| 9 | 0.7857 | 0.9565 | 0.8627 |
| 10 | 0.8148 | 0.9565 | 0.8800 |
| **Mean** | **0.7941** | **0.9478** | **0.8636** |

**Mean F1 score: 0.8636.**

**Discussion.** Choosing attributes clearly paid off: our 7 attributes beat the all-attribute baseline (0.854 vs 0.800 cross-validated F1), and the final model held up on the unseen test set with a mean F1 of 0.864. The model leans towards predicting disease. Recall is very high (it caught 21–23 of the 23 sick patients in the test set), while precision is lower because it also flags 4–8 healthy patients. For a screening tool, that is the better way to be wrong. The 10 scores are close to each other because every fold model is tested on the same 60 test patients and the models only differ slightly in their training data. Finally, k = 41 was the largest value we searched, so a somewhat larger k might perform as well or better.

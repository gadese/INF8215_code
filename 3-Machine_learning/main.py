import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from preprocessing import TransformationWrapper
from preprocessing import LabelEncoderP
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from SoftmaxClassifier import SoftmaxClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

# load dataset
data, target = load_iris().data, load_iris().target

# split data in train/test sets
X_train, X_test, y_train, y_test = train_test_split( data, target, test_size=0.33, random_state=42)

# standardize columns using normal distribution
# fit on X_train and not on X_test to avoid Data Leakage
s = StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.transform(X_test)

# import the custom classifier
cl = SoftmaxClassifier()

# train on X_train and not on X_test to avoid overfitting
train_p = cl.fit_predict(X_train, y_train)
test_p = cl.predict(X_test)

# display precision, recall and f1-score on train/test set
print("train : " + str(precision_recall_fscore_support(y_train, train_p, average="macro")))
print("test : " + str(precision_recall_fscore_support(y_test, test_p, average="macro")))

plt.plot(cl.losses_)
plt.show()

PATH = "data/"
X_train = pd.read_csv(PATH + "train.csv")
X_test = pd.read_csv(PATH + "test.csv")

X_train = X_train.drop(columns=["OutcomeSubtype", "AnimalID"])
X_test = X_test.drop(columns=["ID"])

X_train, y_train = X_train.drop(columns=["OutcomeType"]), X_train["OutcomeType"]

X_train.head()
X_test.head()
y_train.head()

X_train1 = pd.read_csv("data/train_preprocessed.csv")
X_test1 = pd.read_csv("data/test_preprocessed.csv")

X_train1.head()

X_train = X_train.drop(columns=["Color", "Name", "DateTime"])
X_test = X_test.drop(columns=["Color", "Name", "DateTime"])

X_train.head()


# Custom functions for parsing (pipelines)
def parse_reproduction_state(text):
    if text == 'Unknown':
        reproduction_state = 'Unknown'
    else:
        reproduction_state, _ = text.split(' ')
    return reproduction_state


def parse_sex(text):
    if text == 'Unknown':
        sex = 'Unknown'
    else:
        _, sex = text.split(' ')
    return sex


def parse_age(text):
    if isinstance(text, str) and text[0].isdigit():
        nbr, period = text.split(' ')
        nbr = int(nbr)
        if period[0] == 'y' or period[0] == 'Y':
            temps = nbr * 365
        elif period[0] == 'm' or period[0] == 'M':
            temps = nbr * 30
        elif period[0] == 'w' or period[0] == 'W':
            temps = nbr * 7
        elif period[0] == 'd' or period[0] == 'D':
            temps = nbr
        else:
            temps = np.nan
    else:
        temps = np.nan

    return temps


def parse_breed(text):
    text = text.replace(' ', '')
    text = text.upper()
    return text.replace('MIX', '')


def parse_mix(text):
    text = text.upper()
    if (text.find('MIX') >= 0) or (text.find('/') >= 0):
        return 1
    else:
        return 0


# Cat/Dog
pipeline_animal_type = Pipeline([('type', SimpleImputer(strategy='constant', fill_value='Unknown')),
                                 ('encode', LabelEncoderP()),
                                 ('onehot', OneHotEncoder(categories='auto', sparse = False))])

# Breed
pipeline_mix = Pipeline([('mix', TransformationWrapper(transformation= parse_mix)),
                         ('encode', LabelEncoderP())])

pipeline_breed = Pipeline([('breed', TransformationWrapper(transformation= parse_breed)),
                           ('encode', LabelEncoderP())])

pipeline_breed_mix = Pipeline([('mix_and_breed', SimpleImputer(strategy='constant', fill_value='Unknown')),
                               ('feats', FeatureUnion([
                                   ('mix', pipeline_mix),
                                   ('breed', pipeline_breed)
                               ])),
                               ('onehot', OneHotEncoder(categories='auto', sparse=False))])


# Sex and reproduction
pipeline_reproduction = Pipeline([('reproduction', TransformationWrapper(transformation = parse_reproduction_state)),
                                  ("encode", LabelEncoderP())])

pipeline_sex = Pipeline([('sex', TransformationWrapper(transformation = parse_sex)),
                         ("encode", LabelEncoderP())])


pipeline_sex_and_reproduction = Pipeline([("sex_and_reproduction", SimpleImputer(strategy='constant', fill_value='Unknown')),
                                          ('feats', FeatureUnion([
                                              ('reproduction', pipeline_reproduction),
                                              ('sex', pipeline_sex)
                                          ])),
                                          ("onehot", OneHotEncoder(categories='auto', sparse=False))])

# Age
pipeline_age = Pipeline([('age', TransformationWrapper(transformation=parse_age)),
                         ('age_imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
                         ('normalizer', StandardScaler())])

# Full pipeline
full_pipeline = ColumnTransformer([('type', pipeline_animal_type, ['AnimalType']),
                                   ('breed', pipeline_breed_mix, ['Breed']),
                                   ('sex_and_reproduction', pipeline_sex_and_reproduction, ['SexuponOutcome']),
                                   ('age', pipeline_age, ['AgeuponOutcome'])])

#Applies processing to data
X_train_prepared = pd.DataFrame(full_pipeline.fit_transform(X_train))
X_test_prepared = pd.DataFrame(full_pipeline.fit_transform(X_test))
X_train = pd.concat([X_train1,X_train_prepared], axis=1)
X_test = pd.concat([X_test1,X_test_prepared], axis=1)

target_label = LabelEncoder()
y_train_label = target_label.fit_transform(y_train)
print(target_label.classes_)


# Compares the different models
def compare(models, X_train, y_train_label, nb_runs, scoring):
    losses = []
    for model in models:
        cv_results = cross_validate(model, X_train, y_train_label, scoring=scoring, cv=nb_runs, return_train_score=True)
        losses.append([cv_results['train_neg_log_loss'], cv_results['train_precision_macro'], cv_results['train_recall_macro'], cv_results['train_f1_macro']])
    return losses

# Number of folds
nb_run = 3

#Models to compare
models = [SoftmaxClassifier(),
          DecisionTreeClassifier(random_state=0),
          GaussianNB()
         ]

scoring = ['neg_log_loss', 'precision_macro', 'recall_macro', 'f1_macro']
scores = compare(models, X_train, y_train_label, nb_run, scoring)

scores_mean = []
scores_std = []
for i in range(0, len(models)):
    scores_mean.append(np.mean(scores[i], axis=1))
    scores_std.append(np.std(scores[i], axis=1))

print('Average scores: ')
print(scores_mean)
print('Standard deviation of scores: ')
print(scores_std)

selected_model = DecisionTreeClassifier(random_state=0)
my_model = selected_model.fit(X_train, y_train_label)
y_pred = my_model.predict(X_train)

#Prints confusion matrix
print(pd.DataFrame(confusion_matrix(y_train_label, y_pred), columns=target_label.classes_, index=target_label.classes_))

#Prints histogram
print(target_label.classes_)
pd.Series(y_train_label).hist()
plt.show()

print("Adoption: ")
print(sum(y_train_label == 0))
print("Mort: ")
print(sum(y_train_label == 1))
print("Euthanasie: ")
print(sum(y_train_label == 2))
print("Retour: ")
print(sum(y_train_label == 3))
print("Transfert: ")
print(sum(y_train_label == 4))


#Exemple de code pour optimiser les paramètres
# from sklearn.model_selection import GridSearchCV
#
# parameter_grid = {'max_features': [1, 2, 3, 4, 5],
#                   'max_depth': range(150,1000, 10),
#                   }
#
# gs = GridSearchCV(DecisionTreeClassifier(),
#                   param_grid=parameter_grid,
#                   cv=3, return_train_score=True)
# gs_fitted = gs.fit(X_train, y_train)
#
# best_gs = gs_fitted.best_estimator_
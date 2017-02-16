"""
DecisionTemplates classifier.

This module contains a DecisionTemplates rule classifier for classification estimators.

"""
from collections import defaultdict, Counter
from functools import partial

import numpy as np

from sklearn.utils import check_array
from sklearn.utils.validation import check_consistent_length
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils.validation import has_fit_parameter, check_is_fitted
from sklearn.ensemble.voting_classifier import _parallel_fit_estimator
from typing import List, Tuple


class DecisionTemplatesClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    """DecisionTemplatesClassifier Rule classifier for unfitted estimators.

    Parameters
    ----------
    estimators : list of (string, estimator) tuples
        Invoking the ``fit`` method on the ``DecisionTemplatesClassifier`` will fit clones
        of those original estimators that will be stored in the class attribute
        `self.estimators_`.

    similarity_measure: string, optional (default='euclidean')
        Algorithm used to compute similarity between decision template
        and decision profile from sample in predict proba
            ‘euclidean’ will use euclidean norm

    template_creation: {'avg', 'med'} string, optional (default='avg')
        Algorithm used to create decision template for classs:
            ‘avg’ will use average
            ‘med’ will use median

    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for ``fit``.
        If -1, then the number of jobs is set to the number of cores.

    Attributes
    ----------
    estimators_ : list of classifiers
        The collection of fitted sub-estimators.

    classes_ : array-like, shape = [n_predictions]
        The classes labels.

    templates_: list of numpy nd array
        DecisionTemplates.
    """

    def __init__(self, estimators: List[Tuple[str, BaseEstimator]], similarity_measure: str = 'euclidean',
                 template_creation: str = 'avg', n_jobs: int     = 1):
        self._validate_parameters(estimators=estimators, similarity_measure=similarity_measure,
                                  template_creation=template_creation, n_jobs=n_jobs)
        self.estimators = estimators
        self.named_estimators = dict(estimators)
        self.similarity_measure = similarity_measure
        self.template_creation = template_creation
        self.n_jobs = n_jobs
        self.estimators_ = []
        self.classes_ = []
        self.le_ = LabelEncoder()

        similarity_measures = {'euclidean': self.eucklidean_similarity}
        template_creation_algorithms = {'avg': self._fit_one_template_for_each_class_by_avg,
                                        'med': self._fit_one_template_for_each_class_by_med}

        self._similarity_measure = similarity_measures[self.similarity_measure]
        self._fit_one_template_for_each_class = template_creation_algorithms[self.template_creation]

    def _validate_parameters(self, estimators: List[Tuple[str, BaseEstimator]], similarity_measure: [str],
                             template_creation: [str], n_jobs: [int]):

        if estimators is None or len(estimators) == 0:
            raise AttributeError('Invalid `estimators` attribute, `estimators`'
                                 ' should be a list of (string, estimator)'
                                 ' tuples')

        if not isinstance(n_jobs, int):
            raise AttributeError("Invalid `n_jobs` should be int")

        if template_creation not in ['avg', 'med']:
            raise ValueError("Unrecognized template_creation:".format(template_creation))

        if similarity_measure not in ['euclidean']:
            raise ValueError("Unrecognized similarity measure:".format(similarity_measure))

    def fit(self, X, y, sample_weight=None):
        """ Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

        Returns
        -------
        self : object
        """
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output'
                                      ' classification is not supported.')

        if sample_weight is not None:
            for name, step in self.estimators:
                if not has_fit_parameter(step, 'sample_weight'):
                    raise ValueError('Underlying estimator \'%s\' does not support'
                                     ' sample weights.' % name)

        self.le_.fit(y)
        self.classes_ = self.le_.classes_
        transformed_y = self.le_.transform(y)

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_parallel_fit_estimator)(clone(clf), X, transformed_y,
                    sample_weight)
                    for _, clf in self.estimators)

        self._fit_templates(X, transformed_y, sample_weight)
        return self

    def _fit_templates(self, X, y, sample_weight=None):
        if sample_weight is None:
            curr_sample_weight = np.ones((y.shape))
        else:
            curr_sample_weight = sample_weight.copy()

        self.templates_ = defaultdict(list)

        temp_templates = self._fit_one_template_for_each_class(X, y, curr_sample_weight)
        for label, template in temp_templates.items():
            self.templates_[label].append(template)

    def _fit_one_template_for_each_class_by_avg(self, X, y, sample_weight):
        templates = defaultdict(partial(np.zeros, shape=[len(self.estimators_), len(self.classes_)], dtype=np.float))
        count_weights = Counter()

        for sample_no, (label, weight) in enumerate(zip(y, sample_weight)):
            templates[label] += np.multiply(self._make_decision_profile(X[sample_no]), weight)
            count_weights.update({label: weight})

        for label, count in count_weights.items():
            if count != 0:
                templates[label] /= count

        return templates

    def _fit_one_template_for_each_class_by_med(self, X, y, sample_weight):
        templates = defaultdict(partial(np.zeros, shape=[len(self.estimators_), len(self.classes_)], dtype=np.float))
        templates_temp = defaultdict(list)

        for sample_no, (label, weight) in enumerate(zip(y, sample_weight)):
            DP = self._make_decision_profile(X[sample_no])
            for _ in range(int(weight)):
                templates_temp[label].append(DP)

        for label in templates_temp.keys():
            templates[label] = np.median(np.asarray(templates_temp[label]), axis=0)
        return templates

    def _make_decision_profile(self, x):
        check_is_fitted(self, 'estimators_')
        decision_profile = np.zeros([len(self.estimators_), len(self.classes_)], dtype=np.float)
        for i, estimator in enumerate(self.estimators_):
            assert np.array_equal(estimator.classes_, self.le_.transform(self.classes_))
            decision_profile[i] = estimator.predict_proba([x])
        return decision_profile

    def predict(self, X):
        """ Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        maj : array-like, shape = [n_samples]
            Predicted class labels.
        """

        check_is_fitted(self, 'estimators_')

        return self.le_.inverse_transform(np.argmax(self.predict_proba(X), axis=1))

    def _predict_proba(self, X):
        check_is_fitted(self, 'estimators_')
        X = check_array(X, accept_sparse='csr')
        return [self._collect_probas(feature_vector) for feature_vector in X]

    def _collect_probas(self, feature_vector):
        DP = self._make_decision_profile(feature_vector)

        similarities = defaultdict(list)

        for label in self.le_.transform(self.classes_):
            for template in self.templates_[label]:
                similarities[label].append((self._similarity_measure(template, DP)))

        return [max(similarities[label]) for label in self.le_.transform(self.classes_)]

    def eucklidean_similarity(self, DT, DP):
        check_consistent_length(DT, DP)
        return 1 - (np.sum(np.power(np.subtract(DT, DP), 2)) / (len(self.estimators_) * len(self.classes_)))

    @property
    def predict_proba(self):
        """Compute probabilities of possible outcomes for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        avg : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.
        """
        return self._predict_proba

    def transform(self, X):
        """Return class labels or probabilities for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        If `voting='soft'`:
          array-like = [n_classifiers, n_samples, n_classes]
            Class probabilities calculated by each classifier.
        If `voting='hard'`:
          array-like = [n_samples, n_classifiers]
            Class labels predicted by each classifier.
        """
        check_is_fitted(self, 'estimators_')

        return self._predict(X)

    def get_params(self, deep=True):
        """Return estimator parameter names for GridSearch support"""
        if not deep:
            return super(DecisionTemplatesClassifier, self).get_params(deep=False)
        else:
            out = super(DecisionTemplatesClassifier, self).get_params(deep=False)
            out.update(self.named_estimators.copy())
            for name, step in six.iteritems(self.named_estimators):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out

    def _predict(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict(X) for clf in self.estimators_]).T
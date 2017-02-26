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
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils.validation import has_fit_parameter, check_is_fitted
from sklearn.utils.fixes import bincount
from sklearn.utils import check_random_state, indices_to_mask
from sklearn.ensemble.voting_classifier import _parallel_fit_estimator
from sklearn.ensemble.bagging import _generate_bagging_indices, MAX_INT
from typing import List, Tuple, Any, Dict, DefaultDict


class DecisionTemplatesClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    """DecisionTemplatesClassifier Rule classifier for unfitted estimators.

    Parameters
    ----------
    estimators : list of (string, estimator) tuples
        Invoking the ``fit`` method on the ``DecisionTemplatesClassifier`` will fit clones
        of those original estimators that will be stored in the class attribute
        `self.estimators_`.

    groups_mapping : list of tuples, values in tuples determining groups which will include
                     estimators_ of corresponding index to tuple

    similarity_measure: string, optional (default='euclidean')
        Algorithm used to compute similarity between decision template
        and decision profile from sample in predict proba
            ‘euclidean’ will use euclidean norm

    template_creation: {'avg', 'med'} string, optional (default='avg')
        Algorithm used to create decision template for classs:
            ‘avg’ will use average
            ‘med’ will use median

    decision_similarity: {'separately', 'average_group', 'sum_group'} string, optional (default='separately')
        Algorithm used to produce similarities for decision function
            ‘separately’ calculation similarity measure for each template in group separately
            ‘average_group’ calculating similarity measure for each template in group separately
                            and then taking average from particular group
            ‘sum_group’ calculating similarity measure for each template in group separately
                            and then taking sum from particular group

    decision_strategy: {'max', 'k_nearest'} string, optional (default='max')
        Algorithm used to make decision and produce predict proba vector
            ‘max’ - taking class which has max decision similarity measure
            ‘k_nearest’ - taking k max decision similarity class and then majority voting

    k_similar_templates : int, optional (default=1)
        Used when``k_nearest`` decision_strategy strategy set.
        Determining k max similar classes

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

    def __init__(self, estimators: List[Tuple[str, BaseEstimator]], groups_mapping: List[Tuple[Any]] = None,
                 similarity_measure: str = 'euclidean', template_creation: str = 'avg',
                 decision_similarity: str = 'separately', decision_strategy: str = 'max', k_similar_templates: int = 1,
                 n_jobs: int = 1) -> None:

        self._validate_parameters(estimators=estimators, group_mapping=groups_mapping,
                                  similarity_measure=similarity_measure, template_creation=template_creation,
                                  decision_similarity=decision_similarity, decision_strategy=decision_strategy,
                                  k_similar_templates=k_similar_templates,
                                  n_jobs=n_jobs)

        self.estimators = estimators
        self.named_estimators = dict(estimators)
        self.similarity_measure = similarity_measure
        self.template_creation = template_creation
        self.decision_similarity = decision_similarity
        self.decision_strategy = decision_strategy
        self.k_similar_templates = k_similar_templates
        self.n_jobs = n_jobs
        self.estimators_ = []
        self.classes_ = []
        self.le_ = LabelEncoder()
        self.random_state = check_random_state(None)

        if groups_mapping is None:
            self.groups_mapping_ = [(1,) for _ in range(len(estimators))]
        else:
            self.groups_mapping_ = Counter()

        self.groups_ = Counter([group_name for group_tuple in self.groups_mapping_ for group_name in set(group_tuple)])

        similarity_measures = {'euclidean': self.euclidean_similarity}
        template_creation_algorithms = {'avg': self._fit_one_template_for_each_class_by_avg,
                                        'med': self._fit_one_template_for_each_class_by_med}
        decision_algorithms = {'max': self._max_similar_template,
                               'k_nearest': self._k_nearest_templates}

        decision_similarities_algorithms = {'separately': self._seperately,
                                            'average_group': self._average_group,
                                            'sum_group': self._sum_group}

        self._similarity_measure = similarity_measures[self.similarity_measure]
        self._fit_one_template_for_each_class = template_creation_algorithms[self.template_creation]
        self._decision = decision_algorithms[self.decision_strategy]
        self._make_decision_similarity = decision_similarities_algorithms[self.decision_similarity]

    def _validate_parameters(self, estimators: List[Tuple[str, BaseEstimator]], group_mapping: List[Tuple[Any]],
                             similarity_measure: str, template_creation: str, decision_similarity: str,
                             decision_strategy: str, k_similar_templates: int,n_jobs: int) -> None:

        if estimators is None or len(estimators) == 0:
            raise AttributeError('Invalid `estimators` attribute, `estimators`'
                                 ' should be a list of (string, estimator)'
                                 ' tuples')

        if group_mapping is not None and (not isinstance(group_mapping, list) or len(group_mapping) == 0):
            raise AttributeError('Invalid `groups` attribute, `estimators`'
                                 ' should be a non zero length list of tuples.')
        elif isinstance(group_mapping, list):
            if len(estimators) != len(group_mapping):
                raise AttributeError('Invalid `groups` attribute, should have the same len as estimators')
            else:
                for element in group_mapping:
                    if not isinstance(element, tuple):
                        raise AttributeError('Invalid `groups` attribute, should be list of tuples.')
                    elif len(element) == 0:
                        raise AttributeError('Invalid `groups` attribute, tuple inside groups can\'t be empty')

        if not isinstance(n_jobs, int):
            raise AttributeError("Invalid `n_jobs` should be int")

        if template_creation not in ['avg', 'med']:
            raise ValueError("Unrecognized template_creation:".format(template_creation))

        if similarity_measure not in ['euclidean']:
            raise ValueError("Unrecognized similarity measure:".format(similarity_measure))

        if decision_similarity not in ['separately', 'average_group', 'sum_group']:
            raise ValueError("Unrecognized decision_similarity:".format(decision_similarity))

        if decision_strategy not in ['max', 'k_nearest']:
            raise ValueError("Unrecognized decision_strategy:".format(decision_strategy))

        if not isinstance(k_similar_templates, int) or k_similar_templates <= 0:
            raise AttributeError("Invalid `k_similar_templates` attribute, should be int > 0")

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray= None):
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

    def _fit_templates(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None) -> None:
        if sample_weight is None:
            curr_sample_weight = np.ones((y.shape))
        else:
            curr_sample_weight = sample_weight.copy()

        self.templates_ = defaultdict(lambda: defaultdict(list))

        for group in self.groups_:
            temp_templates = self._fit_one_template_for_each_class(group, X, y, curr_sample_weight)
            for label, template in temp_templates.items():
                self.templates_[label][group].append(template)

    def _fit_one_template_for_each_class_by_avg(self, group: Any, X: np.ndarray, y: np.ndarray,
                                                sample_weight: np.ndarray) -> defaultdict:

        templates = defaultdict(partial(np.zeros, shape=[self.groups_[group], len(self.classes_)], dtype=np.float64))
        count_weights = Counter()

        for sample_no, (label, weight) in enumerate(zip(y, sample_weight)):
            templates[label] += np.multiply(self._make_decision_profile(group, X[sample_no]), weight)
            count_weights.update({label: weight})
        for label, count in count_weights.items():
            if count != 0:
                templates[label] = np.divide(templates[label], np.float64(count))

        return templates

    def _fit_one_template_for_each_class_by_med(self, group: Any, X: np.ndarray, y: np.ndarray,
                                                sample_weight: np.ndarray) -> defaultdict:

        templates = defaultdict(partial(np.zeros, shape=[self.groups_[group], len(self.classes_)], dtype=np.float64))
        templates_temp = defaultdict(list)

        for sample_no, (label, weight) in enumerate(zip(y, sample_weight)):
            DP = self._make_decision_profile(group, X[sample_no])
            for _ in range(int(weight)):
                templates_temp[label].append(DP)

        for label in templates_temp.keys():
            templates[label] = np.median(np.asarray(templates_temp[label], dtype=np.float64), axis=0)
        return templates

    def _make_decision_profile(self, group: Any, x: np.ndarray) -> np.ndarray:
        check_is_fitted(self, 'estimators_')
        decision_profile = np.zeros([self.groups_[group], len(self.classes_)], dtype=np.float64)
        group_estimators = (estimator for estimator, group_map_tupple in zip(self.estimators_, self.groups_mapping_)
                            if group in group_map_tupple)

        for i, estimator in enumerate(group_estimators):
            assert np.array_equal(estimator.classes_, self.le_.transform(self.classes_))
            decision_profile[i] = estimator.predict_proba([x])
        return decision_profile

    def _bootstrap(self, X: np.ndarray, sample_weight):
        seed = self.random_state.randint(MAX_INT)
        random_state = np.random.RandomState(seed)

        n_samples, n_features = X.shape

        features, indices = _generate_bagging_indices(random_state=random_state, bootstrap_features=False,
                                                      bootstrap_samples=True, n_features=n_features,
                                                      n_samples=n_samples, max_features=n_features,
                                                      max_samples=n_samples)
        curr_sample_weight = sample_weight.copy()
        sample_counts = bincount(indices, minlength=n_samples)
        curr_sample_weight *= sample_counts

        return curr_sample_weight


    def predict(self, X: np.ndarray):
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

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, 'estimators_')
        X = check_array(X, accept_sparse='csr')
        return np.asarray([self._collect_probas(feature_vector) for feature_vector in X], dtype=np.float64)

    def _collect_probas(self, feature_vector: np.ndarray) -> np.ndarray:
        similarities = self._make_decision_similarities(feature_vector)

        return self._decision(similarities)

    def euclidean_similarity(self, DT: np.ndarray, DP: np.ndarray) -> np.float64:
        check_consistent_length(DT, DP)
        return np.float64(np.subtract(1.0, np.divide(np.sum(np.power(np.subtract(DT, DP), 2)),
                                                     np.multiply(len(self.estimators_),len(self.classes_)))))

    def _make_decision_profiles_for_all_groups(self, feature_vector: np.ndarray) -> Dict[Any, np.ndarray]:
        decision_profiles = {}
        for group in self.groups_:
            decision_profiles[group] = self._make_decision_profile(group, feature_vector)

        return decision_profiles

    def _make_decision_similarities(self, feature_vector: np.ndarray) -> DefaultDict[Any, np.ndarray]:
        decision_profiles = self._make_decision_profiles_for_all_groups(feature_vector)
        similarities = defaultdict(list)

        for label in self.le_.transform(self.classes_):
            label_similarities = []
            for group in self.groups_:
                label_similarities.append([self._similarity_measure(template, decision_profiles[group])
                                           for template in self.templates_[label][group]])

            similarities[label] = self._make_decision_similarity(np.asarray(label_similarities, dtype=np.float64))

        return similarities

    def _max_similar_template(self, similarities: DefaultDict[Any, np.ndarray]) -> np.ndarray:
        return np.asarray([max(similarities[label]) for label in self.le_.transform(self.classes_)], dtype=np.float64)

    def _k_nearest_templates(self, similarities: DefaultDict[Any, np.ndarray]):
        raise NotImplementedError

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

    def _predict(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict(X) for clf in self.estimators_]).T

    def _seperately(self, label_similarities):
        return label_similarities.flatten()

    def _average_group(self, label_similarities):
        return np.average(label_similarities, axis=0)

    def _sum_group(self, label_similarities):
        return np.average(label_similarities, axis=0)
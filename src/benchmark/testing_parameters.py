import copy

TESTING_PARAMETERS = {'algorithm_1':
                        {'estimators': None, 'groups_mapping': None, 'similarity_measure': 'euclidean',
                         'template_construction': 'avg', 'template_fit_strategy': 'one_per_class',
                         'n_templates': 1, 'decision_strategy': 'most_similar_template',
                         'similarity_for_group': 'separately', 'k_similar_templates': 1, 'n_jobs': -1},
                    'algorithm_2':
                        {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                         'similarity_measure': 'euclidean',
                         'template_construction': 'avg', 'template_fit_strategy': 'one_per_class',
                         'n_templates': 1, 'decision_strategy': 'most_similar_template',
                         'similarity_for_group': 'separately', 'k_similar_templates': 1, 'n_jobs': -1},
                    'algorithm_3':
                        {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                         'similarity_measure': 'euclidean',
                         'template_construction': 'avg', 'template_fit_strategy': 'bootstrap',
                         'n_templates': 5, 'decision_strategy': 'most_similar_template',
                         'similarity_for_group': 'separately', 'k_similar_templates': 1, 'n_jobs': -1},
                    'algorithm_4':
                        {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                         'similarity_measure': 'euclidean',
                         'template_construction': 'avg', 'template_fit_strategy': 'bootstrap',
                         'n_templates': 10, 'decision_strategy': 'most_similar_template',
                         'similarity_for_group': 'separately', 'k_similar_templates': 1, 'n_jobs': -1},
                    'algorithm_5':
                        {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                         'similarity_measure': 'euclidean',
                         'template_construction': 'avg', 'template_fit_strategy': 'bootstrap',
                         'n_templates': 15, 'decision_strategy': 'most_similar_template',
                         'similarity_for_group': 'separately', 'k_similar_templates': 1, 'n_jobs': -1},
                    'algorithm_6':
                        {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                         'similarity_measure': 'euclidean',
                         'template_construction': 'avg', 'template_fit_strategy': 'bootstrap',
                         'n_templates': 5, 'decision_strategy': 'most_similar_template',
                         'similarity_for_group': 'average_group', 'k_similar_templates': 1, 'n_jobs': -1},
                    'algorithm_7':
                        {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                         'similarity_measure': 'euclidean',
                         'template_construction': 'avg', 'template_fit_strategy': 'bootstrap',
                         'n_templates': 10, 'decision_strategy': 'most_similar_template',
                         'similarity_for_group': 'average_group', 'k_similar_templates': 1, 'n_jobs': -1},
                    'algorithm_8':
                        {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                         'similarity_measure': 'euclidean',
                         'template_construction': 'avg', 'template_fit_strategy': 'bootstrap',
                         'n_templates': 15, 'decision_strategy': 'most_similar_template',
                         'similarity_for_group': 'average_group', 'k_similar_templates': 1, 'n_jobs': -1},
                    'algorithm_9':
                        {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                         'similarity_measure': 'euclidean',
                         'template_construction': 'avg', 'template_fit_strategy': 'bootstrap',
                         'n_templates': 5, 'decision_strategy': 'most_similar_template',
                         'similarity_for_group': 'sum_group', 'k_similar_templates': 1, 'n_jobs': -1},
                    'algorithm_10':
                        {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                         'similarity_measure': 'euclidean',
                         'template_construction': 'avg', 'template_fit_strategy': 'bootstrap',
                         'n_templates': 10, 'decision_strategy': 'most_similar_template',
                         'similarity_for_group': 'sum_group', 'k_similar_templates': 1, 'n_jobs': -1},
                    'algorithm_11':
                        {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                         'similarity_measure': 'euclidean',
                         'template_construction': 'avg', 'template_fit_strategy': 'bootstrap',
                         'n_templates': 15, 'decision_strategy': 'most_similar_template',
                         'similarity_for_group': 'sum_group', 'k_similar_templates': 1, 'n_jobs': -1},
                    'algorithm_12':
                        {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                         'similarity_measure': 'euclidean',
                         'template_construction': 'avg', 'template_fit_strategy': 'random_subspace',
                         'n_templates': 5, 'decision_strategy': 'most_similar_template',
                         'similarity_for_group': 'separately', 'k_similar_templates': 1, 'n_jobs': -1},
                    'algorithm_13':
                        {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                         'similarity_measure': 'euclidean',
                         'template_construction': 'avg', 'template_fit_strategy': 'random_subspace',
                         'n_templates': 10, 'decision_strategy': 'most_similar_template',
                         'similarity_for_group': 'separately', 'k_similar_templates': 1, 'n_jobs': -1},
                    'algorithm_14':
                        {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                         'similarity_measure': 'euclidean',
                         'template_construction': 'avg', 'template_fit_strategy': 'random_subspace',
                         'n_templates': 15, 'decision_strategy': 'most_similar_template',
                         'similarity_for_group': 'separately', 'k_similar_templates': 1, 'n_jobs': -1},
                    'algorithm_15':
                        {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                         'similarity_measure': 'euclidean',
                         'template_construction': 'avg', 'template_fit_strategy': 'random_subspace',
                         'n_templates': 5, 'decision_strategy': 'most_similar_template',
                         'similarity_for_group': 'average_group', 'k_similar_templates': 1, 'n_jobs': -1},
                    'algorithm_16':
                        {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                         'similarity_measure': 'euclidean',
                         'template_construction': 'avg', 'template_fit_strategy': 'random_subspace',
                         'n_templates': 10, 'decision_strategy': 'most_similar_template',
                         'similarity_for_group': 'average_group', 'k_similar_templates': 1, 'n_jobs': -1},
                    'algorithm_17':
                        {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                         'similarity_measure': 'euclidean',
                         'template_construction': 'avg', 'template_fit_strategy': 'random_subspace',
                         'n_templates': 15, 'decision_strategy': 'most_similar_template',
                         'similarity_for_group': 'average_group', 'k_similar_templates': 1, 'n_jobs': -1},
                    'algorithm_18':
                        {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                         'similarity_measure': 'euclidean',
                         'template_construction': 'avg', 'template_fit_strategy': 'random_subspace',
                         'n_templates': 5, 'decision_strategy': 'most_similar_template',
                         'similarity_for_group': 'sum_group', 'k_similar_templates': 1, 'n_jobs': -1},
                    'algorithm_19':
                        {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                         'similarity_measure': 'euclidean',
                         'template_construction': 'avg', 'template_fit_strategy': 'random_subspace',
                         'n_templates': 10, 'decision_strategy': 'most_similar_template',
                         'similarity_for_group': 'sum_group', 'k_similar_templates': 1, 'n_jobs': -1},
                    'algorithm_20':
                        {'estimators': None, 'groups_mapping': [(1,), (1,), (2,), (2,), (2,)],
                         'similarity_measure': 'euclidean',
                         'template_construction': 'avg', 'template_fit_strategy': 'random_subspace',
                         'n_templates': 15, 'decision_strategy': 'most_similar_template',
                         'similarity_for_group': 'sum_group', 'k_similar_templates': 1, 'n_jobs': -1}
                    }


def set_med_variant(parameters):
    med_parameters = copy.deepcopy(parameters)
    med_parameters['template_construction'] = 'med'
    return med_parameters
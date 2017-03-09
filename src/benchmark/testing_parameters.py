import copy

TESTING_PARAMETERS = {'algorithm_1':
                          {'groups_mapping': None, 'similarity_measure': 'euclidean',
                           'template_construction': 'avg', 'template_fit_strategy': 'one_per_class',
                           'n_templates': 1, 'decision_strategy': 'most_similar_template',
                           'similarity_for_group': 'separately', 'k_similar_templates': 1, 'n_jobs': -1},
                      'algorithm_2':
                          {'groups_mapping': [(1,), (1,), (2,), (2,), (2,)], 'similarity_measure': 'euclidean',
                           'template_construction': 'avg', 'template_fit_strategy': 'one_per_class',
                           'n_templates': 1, 'decision_strategy': 'most_similar_template',
                           'similarity_for_group': 'separately', 'k_similar_templates': 1, 'n_jobs': -1},
                      'algorithm_3':
                          {'groups_mapping': [(1,), (1,), (2,), (2,), (2,)], 'similarity_measure': 'euclidean',
                           'template_construction': 'avg', 'template_fit_strategy': 'bootstrap',
                           'n_templates': 5, 'decision_strategy': 'most_similar_template',
                           'similarity_for_group': 'separately', 'k_similar_templates': 1, 'n_jobs': -1},
                      'algorithm_4':
                          {'groups_mapping': [(1,), (1,), (2,), (2,), (2,)], 'similarity_measure': 'euclidean',
                           'template_construction': 'avg', 'template_fit_strategy': 'bootstrap',
                           'n_templates': 10, 'decision_strategy': 'most_similar_template',
                           'similarity_for_group': 'separately', 'k_similar_templates': 1, 'n_jobs': -1},
                      'algorithm_5':
                          {'groups_mapping': [(1,), (1,), (2,), (2,), (2,)], 'similarity_measure': 'euclidean',
                           'template_construction': 'avg', 'template_fit_strategy': 'bootstrap',
                           'n_templates': 15, 'decision_strategy': 'most_similar_template',
                           'similarity_for_group': 'separately', 'k_similar_templates': 1, 'n_jobs': -1},
                      'algorithm_6':
                          {'groups_mapping': [(1,), (1,), (2,), (2,), (2,)], 'similarity_measure': 'euclidean',
                           'template_construction': 'avg', 'template_fit_strategy': 'bootstrap', 'n_templates': 5,
                           'decision_strategy': 'most_similar_template', 'similarity_for_group': 'average_group',
                           'k_similar_templates': 1, 'n_jobs': -1},
                      'algorithm_7':
                          {'groups_mapping': [(1,), (1,), (2,), (2,), (2,)], 'similarity_measure': 'euclidean',
                           'template_construction': 'avg', 'template_fit_strategy': 'bootstrap',
                           'n_templates': 10, 'decision_strategy': 'most_similar_template',
                           'similarity_for_group': 'average_group', 'k_similar_templates': 1, 'n_jobs': -1},
                      'algorithm_8':
                          {'groups_mapping': [(1,), (1,), (2,), (2,), (2,)], 'similarity_measure': 'euclidean',
                           'template_construction': 'avg', 'template_fit_strategy': 'bootstrap',
                           'n_templates': 15, 'decision_strategy': 'most_similar_template',
                           'similarity_for_group': 'average_group', 'k_similar_templates': 1, 'n_jobs': -1},
                      'algorithm_9':
                          {'groups_mapping': [(1,), (1,), (2,), (2,), (2,)], 'similarity_measure': 'euclidean',
                           'template_construction': 'avg', 'template_fit_strategy': 'bootstrap', 'n_templates': 5,
                           'decision_strategy': 'most_similar_template', 'similarity_for_group': 'sum_group',
                           'k_similar_templates': 1, 'n_jobs': -1},
                      'algorithm_10':
                          {'groups_mapping': [(1,), (1,), (2,), (2,), (2,)], 'similarity_measure': 'euclidean',
                           'template_construction': 'avg', 'template_fit_strategy': 'bootstrap',
                           'n_templates': 10, 'decision_strategy': 'most_similar_template',
                           'similarity_for_group': 'sum_group', 'k_similar_templates': 1, 'n_jobs': -1},
                      'algorithm_11':
                          {'groups_mapping': [(1,), (1,), (2,), (2,), (2,)], 'similarity_measure': 'euclidean',
                           'template_construction': 'avg', 'template_fit_strategy': 'bootstrap',
                           'n_templates': 15, 'decision_strategy': 'most_similar_template',
                           'similarity_for_group': 'sum_group', 'k_similar_templates': 1, 'n_jobs': -1},
                      'algorithm_12':
                          {'groups_mapping': [(1,), (1,), (2,), (2,), (2,)], 'similarity_measure': 'euclidean',
                           'template_construction': 'avg', 'template_fit_strategy': 'random_subspace',
                           'n_templates': 5, 'decision_strategy': 'most_similar_template',
                           'similarity_for_group': 'separately', 'k_similar_templates': 1, 'n_jobs': -1},
                      'algorithm_13':
                          {'groups_mapping': [(1,), (1,), (2,), (2,), (2,)], 'similarity_measure': 'euclidean',
                           'template_construction': 'avg', 'template_fit_strategy': 'random_subspace',
                           'n_templates': 10, 'decision_strategy': 'most_similar_template',
                           'similarity_for_group': 'separately', 'k_similar_templates': 1, 'n_jobs': -1},
                      'algorithm_14':
                          {'groups_mapping': [(1,), (1,), (2,), (2,), (2,)], 'similarity_measure': 'euclidean',
                           'template_construction': 'avg', 'template_fit_strategy': 'random_subspace',
                           'n_templates': 15, 'decision_strategy': 'most_similar_template',
                           'similarity_for_group': 'separately', 'k_similar_templates': 1, 'n_jobs': -1},
                      'algorithm_15':
                          {'groups_mapping': [(1,), (1,), (2,), (2,), (2,)], 'similarity_measure': 'euclidean',
                           'template_construction': 'avg', 'template_fit_strategy': 'random_subspace',
                           'n_templates': 5, 'decision_strategy': 'most_similar_template',
                           'similarity_for_group': 'average_group', 'k_similar_templates': 1, 'n_jobs': -1},
                      'algorithm_16':
                          {'groups_mapping': [(1,), (1,), (2,), (2,), (2,)], 'similarity_measure': 'euclidean',
                           'template_construction': 'avg', 'template_fit_strategy': 'random_subspace',
                           'n_templates': 10, 'decision_strategy': 'most_similar_template',
                           'similarity_for_group': 'average_group', 'k_similar_templates': 1, 'n_jobs': -1},
                      'algorithm_17':
                          {'groups_mapping': [(1,), (1,), (2,), (2,), (2,)], 'similarity_measure': 'euclidean',
                           'template_construction': 'avg', 'template_fit_strategy': 'random_subspace',
                           'n_templates': 15, 'decision_strategy': 'most_similar_template',
                           'similarity_for_group': 'average_group', 'k_similar_templates': 1, 'n_jobs': -1},
                      'algorithm_18':
                          {'groups_mapping': [(1,), (1,), (2,), (2,), (2,)], 'similarity_measure': 'euclidean',
                           'template_construction': 'avg', 'template_fit_strategy': 'random_subspace',
                           'n_templates': 5, 'decision_strategy': 'most_similar_template',
                           'similarity_for_group': 'sum_group', 'k_similar_templates': 1, 'n_jobs': -1},
                      'algorithm_19':
                          {'groups_mapping': [(1,), (1,), (2,), (2,), (2,)], 'similarity_measure': 'euclidean',
                           'template_construction': 'avg', 'template_fit_strategy': 'random_subspace',
                           'n_templates': 10, 'decision_strategy': 'most_similar_template',
                           'similarity_for_group': 'sum_group', 'k_similar_templates': 1, 'n_jobs': -1},
                      'algorithm_20':
                          {'groups_mapping': [(1,), (1,), (2,), (2,), (2,)], 'similarity_measure': 'euclidean',
                           'template_construction': 'avg', 'template_fit_strategy': 'random_subspace',
                           'n_templates': 15, 'decision_strategy': 'most_similar_template',
                           'similarity_for_group': 'sum_group', 'k_similar_templates': 1, 'n_jobs': -1}
                      }


def set_med_variant(parameters):
    med_parameters = copy.deepcopy(parameters)
    med_parameters['template_construction'] = 'med'
    return med_parameters
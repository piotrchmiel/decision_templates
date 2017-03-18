TESTING_PARAMETERS = {'algorithm_no_group':
                          {'groups_mapping': None, 'similarity_measure': 'euclidean',
                           'template_construction': 'avg', 'template_fit_strategy': 'one_per_class',
                           'n_templates': 1, 'decision_strategy': 'most_similar_template',
                           'similarity_for_group': 'separately', 'k_similar_templates': 1, 'n_jobs': -1},
                      'algorithm_groups':
                          {'groups_mapping': [(1,), (1,), (2,), (2,), (2,)], 'similarity_measure': 'euclidean',
                           'template_construction': 'avg', 'template_fit_strategy': 'one_per_class',
                           'n_templates': 1, 'decision_strategy': 'most_similar_template',
                           'similarity_for_group': 'separately', 'k_similar_templates': 1, 'n_jobs': -1},
                      'algorithm_groups_multiple_temp':
                          {'groups_mapping': [(1,), (1,), (2,), (2,), (2,)], 'similarity_measure': 'euclidean',
                           'template_construction': 'avg', 'template_fit_strategy': 'bootstrap',
                           'n_templates': 5, 'decision_strategy': 'most_similar_template',
                           'similarity_for_group': 'separately', 'k_similar_templates': 1, 'n_jobs': -1}}

def make_algorithm_name(algorithm: str, parameters: dict) -> str:
    return algorithm + '_' + parameters['similarity_measure'] + '_' + parameters['template_construction'] + \
           '_' + parameters['template_fit_strategy'] + '_' + str(parameters['n_templates']) + '_' + \
           parameters['similarity_for_group']


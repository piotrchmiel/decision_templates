import os
import zipfile
import itertools
import urllib.parse
import urllib.request
from typing import Tuple, List, Callable, Any, Generator

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from src.learning_set import LearningSet
from src.settings import DATA_DIR


class KeelReader(object):

    def __init__(self, learning_set: LearningSet, filename: str, missing_values: bool, sparse: bool = False,
                 missing_values_strategy: str = 'mean', *args, **kwargs):
        self.learning_set = learning_set
        self.filename = filename
        self.missing_values = missing_values

        self._vectorizer = DictVectorizer(sparse=sparse, dtype=np.float64)
        self._imputer = Imputer(strategy=missing_values_strategy)
        self._processor = make_pipeline(self._vectorizer, self._imputer)
        self._keel_dir = os.path.join(DATA_DIR, 'keel')
        if not os.path.exists(self._keel_dir):
            os.mkdir(self._keel_dir)

        self._data, self._target, = self.read(filename, missing_values)
        self._processor.fit(self._data)

    def make_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        return self._processor.transform(self._data), np.asarray(self._target), self._vectorizer.get_feature_names()

    def make_k_fold_generator(self, cv: int) \
            -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Any, None]:

        for i in range(1, cv + 1):
            i_fold_filename = self.filename + '-' + str(cv) + '-' + str(i)
            train_data, train_labels, _ = self._make_k_fold_dataset(i_fold_filename + 'tra', cv)
            test_data, test_labels, _ = self._make_k_fold_dataset(i_fold_filename + 'tst', cv)
            yield train_data, train_labels, test_data, test_labels

    def _make_k_fold_dataset(self, fold_filename:str, cv: int):

        if not self._data_exists(fold_filename, 'dat'):
            k_fold_filename = fold_filename.split('-')[0] + '-' + str(cv) + '-fold'
            self.prepare_data(k_fold_filename, self.missing_values)

        data, target = self.read(fold_filename, self.missing_values)
        return self._processor.transform(data), np.asarray(target), self._vectorizer.get_feature_names()

    def read(self, filename: str, missing_values: bool) -> Tuple[List[dict], list]:
        self.prepare_data(filename=filename, missing_values=missing_values)
        attributes, start = self.read_header(filename=filename)
        return self.read_data(filename=filename, start=start, delimiter=',', attributes=attributes)

    def prepare_data(self, filename: str, missing_values: bool) -> None:
        if not self._data_exists(filename, 'dat'):
            if not self._data_exists(filename, 'zip'):
                self._download(filename, missing_values)
                self._unzip(filename)
                os.remove(os.path.join(self._keel_dir, filename + '.zip'))

    def read_header(self, filename: str) -> Tuple[List[Tuple[str, Callable]], int]:
        attributes = []
        data_start = 0

        with open(os.path.join(self._keel_dir, filename + '.dat')) as keel_file:
            for line_number, line in enumerate(keel_file):
                line = line.strip()
                if line.startswith("@attribute"):
                    attributes.append(self._parse_attribute(line))
                if line.startswith("@data"):
                    data_start = line_number + 1
                    break

        return attributes, data_start

    def read_data(self, filename: str, start: int, delimiter: str, attributes: List[Tuple[str, Callable]]) \
            -> Tuple[List[dict], List[Any]]:
        data = []
        target = []
        with open(os.path.join(self._keel_dir, filename + '.dat')) as keel_file:
            for line in itertools.islice(keel_file, start, None):
                line = line.strip()
                values = ([attribute.strip() for attribute in line.split(delimiter)])
                temp_dict = {}
                for attribute, (attribute_name, attribute_type) in zip(values[:-1], attributes[:-1]):
                    if any(lack_value == attribute for lack_value in ['<null>', '?']):
                        temp_dict[attribute_name] = np.nan
                    else:
                        temp_dict[attribute_name] = attribute_type(attribute)

                data.append(temp_dict)
                target.append(values[-1])
        return data, target

    def _parse_attribute(self, attribute_line: str) -> Tuple[str, Callable]:
        attribute_values = ([attribute.strip(",") for attribute in attribute_line.split()[1:]])
        type_name = str
        if 'real' in attribute_values[1]:
            type_name = float
        elif 'integer' in attribute_values[1]:
            type_name = int

        return attribute_values[0], type_name

    def _data_exists(self, filename: str, extension: str) -> bool:
        return os.path.exists(os.path.join(self._keel_dir, filename + '.' + extension))

    def _unzip(self, filename: str) -> None:
        with zipfile.ZipFile(os.path.join(self._keel_dir, filename + ".zip"), "r") as zip_ref:
            zip_ref.extractall(self._keel_dir)

    def _download(self, filename: str, missing_values: bool) -> None:
        urllib.request.urlretrieve(url=self._prepare_url(filename, missing_values),
                                   filename=os.path.join(self._keel_dir, filename + '.zip'))

    def _prepare_url(self, filename: str, missing_values: bool) -> str:
        url = 'http://sci2s.ugr.es/keel/dataset/data/'
        if missing_values:
            url += 'missing/'
        else:
            url += 'classification/'
        url += filename + '.zip'
        return url

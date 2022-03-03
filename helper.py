import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from attack import Attack
from target_model import TargetModel
from utils.parameters import Params
from dataset import Dataset
from attack import Attack
from utils.utils import get_all_scores

# this is the importance order required for partial knowledge LOMIA
importance_order_dict = {
    'Adult': ['work', 'sex', 'race', 'fnlwgt', 'occupation', 'education', 'hoursperweek', 'capitalgain', 'capitalloss'],
    'GSS': ['divorce', 'race', 'relig', 'sex', 'educ', 'age', 'year', 'childs', 'pornlaw']

}

class Helper:
    params: Params = None

    # we set dataset_sent_to_attacker's ground truth to zero to make sure attacker doesn't know the actual sensitive values
    def __init__(self, params):
        self.params = Params(**params)
        self.dataset = Dataset(self.params)
        self.target_model = TargetModel(self.params)
        dataset_sent_to_attacker = Dataset(self.params)
        dataset_sent_to_attacker.ground_truths={}
        self.attack = Attack(params=self.params, dataset=dataset_sent_to_attacker, target_model=self.target_model)

    # launches attack based on the category
    def test_attack(self):
        if self.params.attack_category is None or self.params.attack_category == 'disparate_vulnerability' or self.params.attack_category == 'distributional_privacy_leakage':
            self.attack.run_attack()
            self.calc_score()
        elif self.params.attack_category == 'part_know_att_incr_importance':
            if self.attack.name != 'LOMIA':
                raise ValueError(f'Please change attack category to something else or change attack type to LOMIA. CSMIA with {self.params.attack_category} will have combinatorial explosion')
            importance_order = importance_order_dict[self.dataset.name]
            for i in range(len(importance_order)):
                self.params.missing_nonsensitive_attributes = importance_order[:i+1]
                self.dataset = Dataset(self.params)
                self.target_model = TargetModel(self.params)
                dataset_sent_to_attacker = Dataset(self.params)
                dataset_sent_to_attacker.ground_truths={}
                self.attack = Attack(params=self.params, dataset=self.dataset, target_model=self.target_model)
                print(f'Performing partial knowledge attack with missing non-sensitive attributes: {self.dataset.missing_nonsensitive_attributes}')
                self.attack.run_attack()
                self.calc_score()

        elif self.params.attack_category == 'prepare_LOMIA_attack_dataset':
            if self.attack.name != 'LOMIA':
                raise ValueError('Please change attack type to LOMIA.')
            # only in this case we send the dataset with the ground truth to the attacker
            # the purpose is only to count the number of correct instances as in Table 3
            self.attack = Attack(params=self.params, dataset=self.dataset, target_model=self.target_model)
            self.attack.prepare_LOMIA_attack_dataset()

    # calculates the score as presented in the paper and displays in a tabular format
    # if the show_confusion_matrix_visually is set to True, it also plots the confusion matrix
    # also allows individual case reports for CSMIA if the flag is set to True
    def calc_score(self):
        for attribute in self.dataset.sensitive_attributes:
            actual = self.dataset.ground_truths[attribute]
            pred = self.attack.predicted_vals_by_attribute[attribute]

            print(f'\n\n\nOverall report for inferring sensitive attribute {attribute} of dataset {self.dataset.name} and target model type {self.target_model.model_type} when performed {self.attack.name}\n')
            if len(self.dataset.y_labels[attribute]) == 2:
                print(get_all_scores(actual, pred, labels=self.dataset.y_labels[attribute]))
            else:
                print('Confusion matrix')
                print(confusion_matrix(actual, pred, labels=self.dataset.y_labels[attribute]))
            if self.params.show_confusion_matrix_visually:
                cm_display = ConfusionMatrixDisplay(
                    confusion_matrix=confusion_matrix(actual, pred, labels=self.dataset.y_labels[attribute]),
                    display_labels=self.dataset.y_labels[attribute]
                )
                cm_display.plot()
                plt.show()

            if len(self.attack.predicted_case_by_attribute.keys())!=0 and self.params.show_case_reports:
                for case in range(1, 4):
                    case_indices = np.where(self.attack.predicted_case_by_attribute[attribute] == case)
                    actual = self.dataset.ground_truths[attribute][case_indices]
                    pred = self.attack.predicted_vals_by_attribute[attribute][case_indices]

                    print(f'\n\nCase {case} report\n')
                    if len(pred) == 0:
                        print(f'No case {case} samples')
                        continue
                    # print(confusion_matrix(actual, pred, labels=self.dataset.y_labels[attribute]))
                    print(get_all_scores(actual, pred, labels=self.dataset.y_labels[attribute]))

            if self.params.attack_category == 'disparate_vulnerability':
                group_column = self.dataset.data[self.params.extra_field_for_attack_category].to_numpy()
                list_of_groups = self.dataset.data[self.params.extra_field_for_attack_category].unique()

                for group in list_of_groups:
                    group_indices = np.where(group_column == group)
                    actual = self.dataset.ground_truths[attribute][group_indices]
                    pred = self.attack.predicted_vals_by_attribute[attribute][group_indices]

                    print(f'\n\nDisparate Vulnerability Report for Group {group}\n')
                    print(get_all_scores(actual, pred, labels=self.dataset.y_labels[attribute]))


import pandas as pd
import numpy as np
import random
from tqdm import tqdm

from bigml.ensemble import Ensemble
from bigml.api import BigML

from dataset import Dataset
from target_model import TargetModel
# from utils.utils import get_resource_id

# The list of resource ids for ensemble attacks when only one sensitive attribute is being inferred
# Requires 3 level of indexing: [Dataset_name, Target_model_type, Sensitive_attribute_name]
attack_models_dict = {
    'Adult': {
        'DT': {'marital': 'ensemble/621ffe4eaba2df5ee800053e'},
        'DNN': {'marital': 'ensemble/621ffeee8be2aa070600042c'}
    },
    'GSS': {
        # 'DT': {'xmovie': 'ensemble/61255727e4279b249c00502b'},
        'DT': {'xmovie': 'ensemble/621fff6a8f679a67b70004b1'},
        'DNN': {'xmovie': 'ensemble/621fffc68f679a67ab000573'}
    },
    'fivethirtyeight': {
        'DT': {'alcohol': 'ensemble/622000b28f679a67b900049b', 'age': 'ensemble/6220144baba2df5ee10004eb'},
        'DNN': {'alcohol': 'ensemble/622001338be2aa070600042f'}
    }
}

# The list of resource ids for ensemble attacks when two sensitive attribute is being inferred
# We only did it for 538 DT model hence only one entry
attack_models_for_multiple_missing_attribute_dict = {
    'fivethirtyeight': {
        'DT': {'alcohol': 'ensemble/622002ddaba2df5eef000517', 'age': 'ensemble/622002258be2aa071000048a'}
    }
}


y_values_dict = {
    'Adult': {'income': ['<=50K', '>50K']},
    'GSS': {'hapmar': ['nottoohappy', 'prettyhappy', 'veryhappy']},
    'fivethirtyeight' : {'steak_type': ['Medium', 'Medium Well', 'Medium rare', 'Rare', 'Well']}
}


class Attack:
    name: str = None
    dataset: Dataset = None
    target_model: TargetModel = None
    attack_models_dict: dict = None

    def __init__(self, params, dataset, target_model):
        self.name = params.attack_type
        self.dataset = dataset
        self.target_model = target_model
        self.api= BigML("usenixmiai", "d03f694d9f2e250a5b625ebe154a4f4159f6c338", domain="bigml.io")
        if self.name == 'LOMIA':
            if len(self.dataset.sensitive_attributes) == 2:
                self.attack_models_dict = attack_models_for_multiple_missing_attribute_dict
            else:
                self.attack_models_dict = attack_models_dict

    # runs the specified attack
    def run_attack(self):
        if self.name == 'CSMIA':
            self.run_CSMIA()
        elif self.name == 'LOMIA':
            self.run_LOMIA()

    # this only prepares the LOMIA attack dataset
    def prepare_LOMIA_attack_dataset(self):
        for attribute in self.dataset.sensitive_attributes:
            df = self.dataset.data.copy()
            sensitive_val = self.dataset.sensitive_vals[attribute]

            val_option=len(sensitive_val)
            size=df.shape[0]
            n = len(df)

            adv_queries_dict_by_sensitive_val = {}
            for val in sensitive_val:
                adv_query = df.copy()
                adv_query[attribute] = np.concatenate([np.repeat(val, n)])
                adv_queries_dict_by_sensitive_val[val] = adv_query
            list_of_sensitive_vals = list(adv_queries_dict_by_sensitive_val.keys())

            for sensitive_val in adv_queries_dict_by_sensitive_val.keys():
                adv_query = adv_queries_dict_by_sensitive_val[sensitive_val]
                y_attr = self.dataset.y_attr
                X_query=adv_query.copy().drop([y_attr], axis=1)
                y_query=adv_query[[y_attr]]
                y_list=y_query[y_attr].unique()
                predictions=[]
                print(f'Querying with sensitive values set to {sensitive_val}')
                for i in tqdm(range(X_query.shape[0])):
                    input_data = X_query.iloc[i]
                    prediction=self.target_model.model.predict(input_data, full=True)
                    predictions.append(prediction['prediction'])
                #arr=prediction
                df[f'prediction_{sensitive_val}'] = pd.Series(predictions, index = adv_query.index[:len(predictions)])



            def pred_sens_val(x):
                bits=np.zeros(len(list_of_sensitive_vals))
                for i in range(len(list_of_sensitive_vals)):
                    if x['prediction_{}'.format(list_of_sensitive_vals[i])] == x[y_attr]:
                        bits[i] = 1
                num_of_pred_match=len(np.argwhere(bits == 1))

                if num_of_pred_match == 0:
                    case = 3
                    pred = np.nan
                elif num_of_pred_match == 1:
                    case = 1
                    index = np.argwhere(bits == 1)[0][0]
                    pred = list_of_sensitive_vals[index]
                else:
                    case = 2
                    pred = np.nan
                # return (list_of_sensitive_vals[index], case)
                return pred, case

            prediction = []
            predicted_case = []
            for i in range(df.shape[0]):
                x = df.iloc[i]
                pred, case = pred_sens_val(x)
                prediction.append(pred)
                predicted_case.append(case)

            actual = self.dataset.ground_truths[attribute]
            prediction = np.array(prediction)
            predicted_case = np.array(predicted_case)

            case_1_index = np.where(predicted_case == 1)
            actual = actual[case_1_index]
            prediction = prediction[case_1_index]
            correct_count = 0
            for i in range(len(actual)):
                if actual[i] == prediction[i]:
                    correct_count += 1

            print(f'Total case 1 instances: {len(prediction)}')
            print(f'Number of correct instances: {correct_count}')
            

    # this loads the ensemble attack model using the resource id found from the dictionary
    # and then queries the ensemble for prediction of the sensitive attribute
    def run_LOMIA(self):
        self.predicted_vals_by_attribute = {}
        self.predicted_case_by_attribute = {}
        for attribute in self.dataset.sensitive_attributes:
            df = self.dataset.data.copy()

            ensemble = Ensemble(self.attack_models_dict[self.dataset.name][self.target_model.model_type][attribute], api=self.api)

            output = []
            print(f'Querying the ensemble attack model for prediction')
            for i in tqdm(range(df.shape[0])):
                input_data = df.iloc[i]
                prediction = ensemble.predict(input_data)
                output.append(prediction['prediction'])

            self.predicted_vals_by_attribute[attribute] = np.array(output)


    # this does both CSMIA and partial knowledge CSMIA
    def run_CSMIA(self):
        self.predicted_vals_by_attribute = {}
        self.predicted_case_by_attribute = {}
        for attribute in self.dataset.sensitive_attributes:
            df = self.dataset.data.copy()
            list_of_sensitive_vals = self.dataset.sensitive_vals[attribute]
            list_of_y_vals = df[self.dataset.y_attr].unique()
            n = len(df)

            adv_queries_dict_by_sensitive_val = {}
            for val in list_of_sensitive_vals:
                adv_query = df.copy()
                adv_query[attribute] = np.concatenate([np.repeat(val, n)])
                adv_queries_dict_by_sensitive_val[val] = adv_query

            list_of_prediction_count_for_sen_val = {}
            list_of_confidence_sum_for_sen_val = {}

            for sensitive_val in list_of_sensitive_vals:
                list_of_prediction_count_for_sen_val[sensitive_val] = []
                list_of_confidence_sum_for_sen_val[sensitive_val] = []
                adv_query = adv_queries_dict_by_sensitive_val[sensitive_val]
                y_attr = self.dataset.y_attr
                X_query=adv_query.copy().drop([y_attr], axis=1)
                y_query=adv_query[[y_attr]]
                y_list=y_query[y_attr].unique()

                print(f'Querying with sensitive values set to {sensitive_val}')
                for i in tqdm(range(X_query.shape[0])):
                    prediction_count_for_yval = {}
                    confidence_sum_yval = {}
                    for yval in list_of_y_vals:
                        prediction_count_for_yval[yval] = 0
                        confidence_sum_yval[yval] = 0.

                    input_data = X_query.iloc[i].copy()

                    # if there are missing nsa, then it is partial knowledge CSMIA
                    if len(self.dataset.missing_nonsensitive_attributes) != 0:
                        for nsa in self.dataset.missing_nonsensitive_attributes:
                            nsa_vals = self.dataset.missing_nsa_vals[nsa]

                            for nsa_val in nsa_vals:
                                input_data[nsa] = nsa_val
                                prediction=self.target_model.model.predict(input_data, full=True) 
                                predicted_yval = prediction['prediction']

                                prediction_count_for_yval[predicted_yval] += 1
                                if self.target_model.model_type == 'DNN':
                                    confidence_sum_yval[predicted_yval] += prediction['probability']
                                else:
                                    confidence_sum_yval[predicted_yval] += prediction['confidence']
                    # normal CSMIA
                    else:
                        prediction=self.target_model.model.predict(input_data, full=True) 
                        predicted_yval = prediction['prediction']

                        prediction_count_for_yval[predicted_yval] += 1
                        if self.target_model.model_type == 'DNN':
                            confidence_sum_yval[predicted_yval] += prediction['probability']
                        else:
                            confidence_sum_yval[predicted_yval] += prediction['confidence']                        
                    
                    list_of_prediction_count_for_sen_val[sensitive_val].append(prediction_count_for_yval)
                    list_of_confidence_sum_for_sen_val[sensitive_val].append(confidence_sum_yval)
                
            # print(list_of_prediction_count_for_sen_val)
            # print(list_of_confidence_sum_for_sen_val)

            actual_ys = df[self.dataset.y_attr]
            # print(actual_ys)
            predicted_sen_val = []
            case_of_pred = []
            for i in range(df.shape[0]):
                prediction_match = []
                confidence_score_for_correct_y = []
                total_confidence_scores = []

                for sensitive_val in list_of_sensitive_vals:
                    prediction_match.append(list_of_prediction_count_for_sen_val[sensitive_val][i][actual_ys[i]])
                    confidence_score_for_correct_y.append(list_of_confidence_sum_for_sen_val[sensitive_val][i][actual_ys[i]])
                    total_confidence_scores.append(sum([list_of_confidence_sum_for_sen_val[sensitive_val][i][y_val] for y_val in list_of_y_vals]))

                # print(prediction_match)
                # print(confidence_score_for_correct_y)
                prediction_match = np.array(prediction_match)
                # confidence_score_for_correct_y = np.array(confidence_score_for_correct_y)

                num_of_pred_match=len(np.argwhere(prediction_match > 0))
                # print(num_of_pred_match)

                if num_of_pred_match == 0:
                    case = 3
                    mins = np.argwhere(total_confidence_scores == np.min(total_confidence_scores))
                    index = random.choice(mins)[0]
                elif num_of_pred_match == 1:
                    case = 1
                    index = np.argwhere(prediction_match > 0)[0][0]
                else:
                    case = 2
                    confidence_score_for_correct_y = [confidence_score_for_correct_y[i]*(prediction_match[i]>0) for i in range(len(confidence_score_for_correct_y))]
                    maxes = np.argwhere(confidence_score_for_correct_y == np.max(confidence_score_for_correct_y))
                    index = random.choice(maxes)[0]      

                predicted_sen_val.append(list_of_sensitive_vals[index])
                case_of_pred.append(case)

            # print(predicted_sen_val)
            # print(case_of_pred)           

            self.predicted_vals_by_attribute[attribute] = np.array(predicted_sen_val)
            self.predicted_case_by_attribute[attribute] = np.array(case_of_pred)

    def run_FJRMIA(self, priors, cms):
        self.predicted_vals_by_attribute = {}
        self.predicted_case_by_attribute = {}
        for attribute in self.dataset.sensitive_attributes:
            prior = priors[attribute]
            cm = cms[attribute]
            df = self.dataset.data.copy()
            # adv_query = self.dataset.data.copy().drop([self.dataset.y_attr], axis=1)
            # actual_y = self.dataset.data[[self.dataset.y_attr]]
            # training_target=adv_query
            # training_target[y_attr]=y_train
            sensitive_val = self.dataset.sensitive_vals[attribute]
            # adv_query.reset_index(inplace = True, drop = True)

            val_option=len(sensitive_val)
            size=df.shape[0]
            #adv_query=adv_query.append(([adv_query]*(val_option-1)),ignore_index=True)
            #n = len(adv_query)/val_option
            n = len(df)

            adv_queries_dict_by_sensitive_val = {}
            for val in sensitive_val:
                adv_query = df.copy()
                adv_query[attribute] = np.concatenate([np.repeat(val, n)])
                adv_queries_dict_by_sensitive_val[val] = adv_query
            list_of_sensitive_vals = list(adv_queries_dict_by_sensitive_val.keys())

            for sensitive_val in adv_queries_dict_by_sensitive_val.keys():
                adv_query = adv_queries_dict_by_sensitive_val[sensitive_val]
                y_attr = self.dataset.y_attr
                X_query=adv_query.copy().drop([y_attr], axis=1)
                y_query=adv_query[[y_attr]]
                y_list=y_query[y_attr].unique()
                predictions=[]
                confidences=[]
                print(f'Querying with sensitive values set to {sensitive_val}')
                for i in tqdm(range(X_query.shape[0])):
                    #buil_query=X_query.iloc[i]
                    input_data = X_query.iloc[i]
                    # print(input_data)
                    prediction=self.target_model.model.predict(input_data, full=True)
                    #print(input_data, prediction)
                    #prediction=model.predict(input_data, full=False)
                    if self.target_model.model_type == 'DNN':
                        confidences.append(prediction['probability'])
                    else:
                        confidences.append(prediction['confidence'])
                    predictions.append(prediction['prediction'])
                #arr=prediction
                df[f'prediction_{sensitive_val}'] = pd.Series(predictions, index = adv_query.index[:len(predictions)])
                df[f'confidence_{sensitive_val}'.format(val[0])] = pd.Series(confidences, index = adv_query.index[:len(confidences)])

                # print(predictions)
                # print(confidences)

            y_val=y_values_dict[self.dataset.name][y_attr]
            predictions=[]
            y_list=self.dataset.data[y_attr].to_numpy()
            y_count=len(y_val)
            c=[[0]*y_count for i in range(y_count)]
            sensitive_val = self.dataset.sensitive_vals[attribute]
            for i in range(y_count):
                c[i][:]=cm[i][:]/sum(cm[i][:])
            for i in range(df.shape[0]):
                y=y_val.index(y_list[i])
                scores=[]
                for j in range(len(sensitive_val)):
                    yp=y_val.index(df.iloc[i]['prediction_{}'.format(sensitive_val[j])])
                    scores.append(c[y][yp]*prior[j])
                scores=np.array(scores)
                predictions.append(sensitive_val[np.argmax(scores)])

            self.predicted_vals_by_attribute[attribute] = np.array(predictions)

        

    # this was the old code used to run CSMIA. same functionality. but it cannot do partial knowledge
    def run_CSMIA_legacy(self):
        self.predicted_vals_by_attribute = {}
        self.predicted_case_by_attribute = {}
        for attribute in self.dataset.sensitive_attributes:
            df = self.dataset.data.copy()
            # adv_query = self.dataset.data.copy().drop([self.dataset.y_attr], axis=1)
            # actual_y = self.dataset.data[[self.dataset.y_attr]]
            # training_target=adv_query
            # training_target[y_attr]=y_train
            sensitive_val = self.dataset.sensitive_vals[attribute]
            # adv_query.reset_index(inplace = True, drop = True)

            val_option=len(sensitive_val)
            size=df.shape[0]
            #adv_query=adv_query.append(([adv_query]*(val_option-1)),ignore_index=True)
            #n = len(adv_query)/val_option
            n = len(df)

            adv_queries_dict_by_sensitive_val = {}
            for val in sensitive_val:
                adv_query = df.copy()
                adv_query[attribute] = np.concatenate([np.repeat(val, n)])
                adv_queries_dict_by_sensitive_val[val] = adv_query
            list_of_sensitive_vals = list(adv_queries_dict_by_sensitive_val.keys())

            for sensitive_val in adv_queries_dict_by_sensitive_val.keys():
                adv_query = adv_queries_dict_by_sensitive_val[sensitive_val]
                y_attr = self.dataset.y_attr
                X_query=adv_query.copy().drop([y_attr], axis=1)
                y_query=adv_query[[y_attr]]
                y_list=y_query[y_attr].unique()
                predictions=[]
                confidences=[]
                print(f'Querying with sensitive values set to {sensitive_val}')
                for i in tqdm(range(X_query.shape[0])):
                    #buil_query=X_query.iloc[i]
                    input_data = X_query.iloc[i]
                    # print(input_data)
                    prediction=self.target_model.model.predict(input_data, full=True)
                    #print(input_data, prediction)
                    #prediction=model.predict(input_data, full=False)
                    if self.target_model.model_type == 'DNN':
                        confidences.append(prediction['probability'])
                    else:
                        confidences.append(prediction['confidence'])
                    predictions.append(prediction['prediction'])
                #arr=prediction
                df[f'prediction_{sensitive_val}'] = pd.Series(predictions, index = adv_query.index[:len(predictions)])
                df[f'confidence_{sensitive_val}'.format(val[0])] = pd.Series(confidences, index = adv_query.index[:len(confidences)])

                # print(predictions)
                # print(confidences)

            def pred_sens_val(x):
                bits=np.zeros(len(list_of_sensitive_vals))
                confidences=np.zeros(len(list_of_sensitive_vals))
                for i in range(len(list_of_sensitive_vals)):
                    confidences[i] = x['confidence_{}'.format(list_of_sensitive_vals[i])]
                    if x['prediction_{}'.format(list_of_sensitive_vals[i])] == x[y_attr]:
                        bits[i] = 1
                num_of_pred_match=len(np.argwhere(bits == 1))

                if num_of_pred_match == 0:
                    case = 3
                    mins = np.argwhere(confidences == np.min(confidences))
                    index = random.choice(mins)[0]
                elif num_of_pred_match == 1:
                    case = 1
                    index = np.argwhere(bits == 1)[0][0]
                else:
                    case = 2
                    confidences = [confidences[i]*bits[i] for i in range(len(confidences))]
                    maxes = np.argwhere(confidences == np.max(confidences))
                    index = random.choice(maxes)[0]  
                # return (list_of_sensitive_vals[index], case)
                return list_of_sensitive_vals[index], case

            # dft=df.copy()
            # comp = dft.apply(pred_sens_val, axis=1).to_numpy()
            # comp, case = map(list,zip(*comp))
            # dft['case'] = case
            # dft['comp'] = comp
            # print(comp)
            # predicted_vals_by_attribute[attribute] = np.array(comp)
            # predicted_case_by_attribute[attribute] = np.array(case)
            prediction_by_CSMIA = []
            predicted_case_by_CSMIA = []
            for i in range(df.shape[0]):
                x = df.iloc[i]
                pred, case = pred_sens_val(x)
                prediction_by_CSMIA.append(pred)
                predicted_case_by_CSMIA.append(case)

            self.predicted_vals_by_attribute[attribute] = np.array(prediction_by_CSMIA)
            self.predicted_case_by_attribute[attribute] = np.array(predicted_case_by_CSMIA)



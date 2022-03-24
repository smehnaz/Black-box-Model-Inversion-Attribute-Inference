# black-boxMIAI


## Table of contents
 * [Basics](#basics)
 * [Installation](#installation)
 * [Reproducing Experiments](#reproducing-experiments)  
 * [How to use param files](#how-to-use-param-files)

# Basics
This contains the implementation of the Model Inversion Attack paper **"Are Your Sensitive Attributes Private? Novel Model Inversion Attribute Inference Attacks on Classification Models"**. The paper introduced two attacks - **CSMIA** and **LOMIA** (the later does not require confidence values from the prediction queries). Along with **CSMIA** and **LOMIA** this repo also contains the previous model inversion attack **FJRMIA** which was used as a baseline. The attacks are performed on three tabular datasets that are publicly available = **Adult**, **GSS**, and **538**. The goal of the attacks is to predict sensitive values missing from the query. Attacks are also performed in various settings -

* Inferring a single binary sensitive attribute
* Inferring a single multi-valued sensitive attribute
* Inferring multiple sensitive attributes
* Inferring sensitive attributes when one or more non-sensitive attributes are unknown
* Inferring sensitive attributes from data that was not originally on the training set (distributional privacy leakage)
* Analyzing disparate vulnerability of model inversion attack on different subgroups

All the attacks and details on how to run them are described in detail in later sections.


# Installation
### Installation using Anaconda
* Create a new conda environment using the following command: `conda create -n bboxmia python==3.7.11`
* Activate the environment using: `conda activate bboxmia`
* Install all dependencies: `pip install -r requirements.txt`


# Reproducing Experiments

For all the tables and figures we presented in the paper, we have written down a configuration file for each. To reproduce results of that particular experiment, all you have to do is load that configuration file while running the python code.

For example, to reproduce the results presented in table 13 or to see how the CSMIA and LOMIA attack performs on the Adult Dataset trained Decision Tree (DT) and Deep Neural Network (DNN) model, one can run the following commands -
* This will launch LOMIA attack on the Adult DT model: `python main.py --param configs/table_13/lomia_dt.yaml`
* This will launch LOMIA attack on the Adult DNN model: `python main.py --param configs/table_13/lomia_dnn.yaml`
* This will launch CSMIA attack on the Adult DT model: `python main.py --param configs/table_13/csmia_dt.yaml`
* This will launch CSMIA attack on the Adult DNN model: `python main.py --param configs/table_13/csmia_dnn.yaml`

### Comment

The results may slightly vary from the paper on **CSMIA** experiments when attacked on **GSS** or **538** dataset. In **CSMIA**, we query the target model for different sensitive attribute values using APIs and decide the instance's sensitive attribute value based on returned confidence scores. For some samples/instances, we obtain the same confidence scores with different sensitive attribute values. In some cases, the number of these samples for which the target model returns the same confidence value could be significant. Especially, in instances of case 3 (section 4.1), there are multiple candidates of sensitive attribute values with the lowest confidence scores. In those cases, we randomly select one of the possible values as the estimated sensitive attribute value. This results in selecting different sensitive attribute values for those instances in different runs. Therefore, while comparing with ground truth, it might result in slightly different overall performances in different metrics. The results we have shown in the paper represent the median of multiple runs.

Also, for **LOMIA** experiments, the attack models in this repo are trained using a different bigML account than the paper. BigML applies its own optimization techniques while generating the attack models (ensembles), there might be slight variation in results produced by this implementation.

### Binary valued single sensitive attribute inference (LOMIA, CSMIA and FJRMIA attack on Adult, GSS and 538 dataset trained DT and DNN models)

The following lists out all the configurations required to perform the binary valued single sensitive attribute inference attack on Adult and GSS dataset. To run the attack simply type in the terminal in the code directory: `python main.py --param configuration_file`

| Dataset | Target Model Type | Attack Type | configuration filename |
| ------- | ------- | ------- | ------- |
| Adult | DT | LOMIA | `configs/table_13/lomia_dt.yaml` |
| Adult | DT | CSMIA | `configs/table_13/csmia_dt.yaml` |
| Adult | DT | FJRMIA | `configs/table_13/fjrmia_dt.yaml` |
| Adult | DNN | LOMIA | `configs/table_13/lomia_dnn.yaml` |
| Adult | DNN | CSMIA | `configs/table_13/csmia_dnn.yaml` |
| Adult | DNN | FJRMIA | `configs/table_13/fjrmia_dnn.yaml` |
| GSS | DT | LOMIA | `configs/table_12/lomia_dt.yaml` |
| GSS | DT | CSMIA | `configs/table_12/csmia_dt.yaml` |
| GSS | DT | FJRMIA | `configs/table_12/fjrmia_dt.yaml` |
| GSS | DNN | LOMIA | `configs/table_12/lomia_dnn.yaml` |
| GSS | DNN | CSMIA | `configs/table_12/csmia_dnn.yaml` |
| GSS | DNN | FJRMIA | `configs/table_12/fjrmia_dnn.yaml` |

If LOMIA/CSMIA attack is to be performed on 538 dataset to infer Alcohol attribute, the following code snippet can be used as the parameter file after the `--param` tag while running the python code.

```
dataset: fivethirtyeight
attack_type: LOMIA/CSMIA # Depends on which attack you want to perform
target_model_type: DT/DNN # Depends on which target model you are attacking
sensitive_attributes: ['alcohol']
missing_nonsensitive_attributes: []
```

### Multi-valued single sensitive attribute inference (LOMIA and CSMIA attack on 538 dataset trained DT model to infer age attribute)

To infer the multi-valued age attribute, the following code snippet can be used as the parameter file.

```
dataset: fivethirtyeight
attack_type: LOMIA/CSMIA # Depends on which attack you want to perform
target_model_type: DT 
sensitive_attributes: ['age']
missing_nonsensitive_attributes: []
```

Also, the following built-in configuration files can be used for ease.

| Dataset | Target Model Type | Attack Type | configuration filename |
| ------- | ------- | ------- | ------- |
| 538 | DT | LOMIA | `configs/table_10/lomia.yaml` |
| 538 | DT | CSMIA | `configs/table_10/csmia.yaml` |


### Multiple sensitive attribute inference (LOMIA and CSMIA attack on 538 dataset trained DT model to infer both alcohol and age attribute)

The following built-in configurations file can already be used

| Dataset | Target Model Type | Attack Type | configuration filename |
| ------- | ------- | ------- | ------- |
| 538 | DT | LOMIA | `configs/table_15_16/LOMIA.yaml` |
| 538 | DT | CSMIA | `configs/table_15_16/CSMIA.yaml` |

Or, alternatively, the following code snippet,

```
dataset: fivethirtyeight
attack_type: LOMIA/CSMIA # Depends on which attack you want to perform
target_model_type: DT 
sensitive_attributes: ['alcohol', 'age']
missing_nonsensitive_attributes: []
```


### Partial Knowledge Attack - LOMIA (Attack on Adult and GSS trained models where increasingly more important non-sensitive attributes are missing)

The following built-in configurations file can already be used

| Dataset | Target Model Type | Attack Type | configuration filename |
| ------- | ------- | ------- | ------- |
| Adult | DT | LOMIA | `configs/figure_5/dt.yaml` |
| Adult | DNN | CSMIA | `configs/figure_5/dnn.yaml` |
| GSS | DT | LOMIA | `configs/figure_11/dt.yaml` |
| GSS | DNN | CSMIA | `configs/figure_12/dnn.yaml` |

Or, alternatively, the following code snippet,

```
dataset: Adult/GSS # Depends on which dataset being attacked
attack_type: LOMIA
target_model_type: DT/DNN # Depends on which target model is being attacked
sensitive_attributes: ['marital']/['xmovie'] # marital for Adult, xmovie for GSS
missing_nonsensitive_attributes: []
attack_category: 'part_know_att_incr_importance'
```

### Partial Knowledge Attack - CSMIA (One or two non-sensitive attributes are missing from Adult)

All the configurations in the directory `configs/figure_13/` can be used
To make the attributes `x` missing, the following configuration can be used `configs/figure_13/x.yaml`.

Alternatively, the following code snippet,

```
dataset: Adult
attack_type: CSMIA
target_model_type: DT
sensitive_attributes: ['marital']
missing_nonsensitive_attributes: [x, y] # where x, y are non-sensitive attributes that are missing
```

Warning: Please do not use more than 2 missing nsa in CSMIA as it may result in combinatorial explosion of queries.


### Distributional Privacy Leakage Attack (Adult and GSS dataset)

The following built-in configurations file can already be used.

Note: the first configuration queries using the DS<sub>D</sub> dataset and the second one queries using the DS<sub>T</sub> dataset for comparison.

| Dataset | Target Model Type | Attack Type | configuration filename |
| ------- | ------- | ------- | ------- |
| Adult | DT | CSMIA | `configs/figure_4a/csmia_on_DSd.yaml` `configs/figure_4a/csmia_on_DSt.yaml` |
| Adult | DT | LOMIA | `configs/figure_4a/lomia_on_DSd.yaml` `configs/figure_4a/lomia_on_DSt.yaml` |
| Adult | DNN | CSMIA | `configs/figure_8/adult_csmia_dnn_on_DSd.yaml` `configs/figure_8/adult_csmia_dnn_on_DSt.yaml` |
| Adult | DNN | LOMIA | `configs/figure_8/adult_lomia_dnn_on_DSd.yaml` `configs/figure_8/adult_lomia_dnn_on_DSt.yaml` |
| GSS | DT | CSMIA | `configs/figure_8/gss_csmia_dt_on_DSd.yaml` `configs/figure_8/gss_csmia_dt_on_DSt.yaml` |
| GSS | DT | LOMIA | `configs/figure_8/gss_lomia_dt_on_DSd.yaml` `configs/figure_8/gss_lomia_dt_on_DSt.yaml` |
| GSS | DNN | CSMIA | `configs/figure_8/gss_csmia_dnn_on_DSd.yaml` `configs/figure_8/gss_csmia_dnn_on_DSt.yaml` |
| GSS | DNN | LOMIA | `configs/figure_8/gss_lomia_dnn_on_DSd.yaml` `configs/figure_8/gss_lomia_dnn_on_DSt.yaml` |

Alternatively, the following code snippet,

```
dataset: Adult/GSS # Depends on which dataset being attacked
attack_type: LOMIA/CSMIA # Depends on which attack you want to perform
target_model_type: DT/DNN # Depends on which target model is being attacked
sensitive_attributes: ['marital']/['xmovie'] # marital for Adult, xmovie for GSS
missing_nonsensitive_attributes: []
attack_category: 'distributional_privacy_leakage'
```

### Disparate Vulnerability Attack (LOMIA)

The following built-in configuration files can be used

| Dataset | Target Model Type | Vulnerable Group | configuration filename |
| ------- | ------- | ------- | ------- |
| Adult | DNN | sex | `configs/figure_4b/sex.yaml` |
| Adult | DNN | race | `configs/figure_4b/race.yaml` |
| Adult | DNN | occupation | `configs/figure_10/occupation.yaml` |
| GSS | DT | religion | `configs/figure_9/religion.yaml` |

Alternatively, the following code snippet,

```
dataset: Adult/GSS # Depends on which dataset being attacked
attack_type: LOMIA
target_model_type: DT/DNN # Depends on which target model is being attacked
sensitive_attributes: ['marital']/['xmovie'] # marital for Adult, xmovie for GSS
missing_nonsensitive_attributes: []
attack_category: 'disparate_vulnerability'
extra_field_for_attack_category: x # The vulnerable subgroup (one of the fields on the dataset)
```

### LOMIA attack dataset preparation

For LOMIA attacks, the first step is to query the target model to build the attack dataset. Then, we train the ensemble attack model on BigML using the attack dataset. For experiments, we skip the first two steps and only query the ensemble attack model to predict the sensitive value. However, if one wants to try out the first step in LOMIA attack, it can easily be done in the following ways.

The following built-in configuration files can be used

| Dataset | Target Model Type |  Sensitive Attribute | configuration filename |
| ------- | ------- | ------- | ------- |
| Adult | DT | Marital | `configs/table_3/adult_dt.yaml` |
| Adult | DNN | Marital | `configs/table_3/adult_dnn.yaml` |
| GSS | DT | X-movie | `configs/table_3/gss_dt.yaml` |
| GSS | DNN | X-movie | `configs/table_3/gss_dnn.yaml` |
| 538 | DT | Age | `configs/table_3/538_age_dt.yaml` |
| 538 | DT | Alcohol | `configs/table_3/538_age_dt.yaml` |

Alternatively, the following code snippet,
```
dataset: Adult/GSS/fivethirtyeight # Depends on which dataset being attacked
attack_type: LOMIA
target_model_type: DT/DNN # Depends on which target model is being attacked
sensitive_attributes: ['marital']/['xmovie']/['alcohol']/['age'] # marital for Adult, xmovie for GSS, alcohol or age for 538
missing_nonsensitive_attributes: []
attack_category: 'prepare_LOMIA_attack_dataset'
```

# How to use param files
The `configs/default_params.yaml` file has detailed instructions on how to run custom experiments (not listed in the paper or above). It includes possible values to be used for each param field.

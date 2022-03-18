# Optimizing an ML Pipeline in Azure

## Overview

This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources

- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)

## Summary

**In 1-2 sentences, explain the problem statement: e.g "This dataset contains data about... we seek to predict..."**

The dataset consists of information about persons who were contacted during different bank marketing campaigns. We seek to predict if they will subscribe to a term deposit, based on historical data. Features include e.g. the age of the person, job type, maritial status, educational background, existing loans and features about how, when and how often the persons were contacted.

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**

The model with highest accuracy (91,7%) is a voting ensemble found by AutoML consisting of different classifiers. In comparison, the best found logistic regression model yields an accuracy of about 91,3% on a validation set.

## Scikit-learn Pipeline

**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

The Scikit-learn pipeline consists of two parts: The hyperparameter optimization using HyperDrive - as set up in a Jupyter notebook - and a `train.py` script which is called during the execution of HyperDrive.

The `ScriptRunConfig` specifies to use the `train.py` script for training a model, which compute resources to use (in this case a cluster of four compute nodes) and the conda environment (i.e. the packages necessary for training). The `HyperDriveConfig` specifies the kind of hyperparameter sampling (here, a random parameter sampling), an early stopping policy (here, a bandit policy), the primary metric used for evaluation (here, the accuracy, which should be maximized), the total number of models to be evaluated (20) and the maximum number of concurrent training runs (4).

In the `train.py` script, the hyperparameters `C` (regularization strength) and `max_iter` (maximum number of iterations) to be used for fitting the model are read. Then, the data is downloaded and preprocessed using the `clean_data` function. The preprocessing mainly consists of turning values of categorical variables into binary indicators for each category which can be beneficial for logistic regression models. Then, the data is divided into a training, validation and test set. The training data is used for fitting the model, the validation data for evaluating the model, given a hyperparameter combination, and the test set could be used later for comparisons with other models or assessing the best model's accuracy on an independent subset of the data (at this point, the test set is ignored). Using a fixed number for the random state guarantees the same split of data over all experiments. Especially, the same split of data is also used later for the AutoML experiment. A logistic regression model is fit to the training data and the model's accuracy, as determined on the validation set, is returned. Finally, a model is fitted on the union of the training and validation set, since using more data should yield an even better generalization performance (see also this [article](https://stats.stackexchange.com/questions/11602/training-on-the-full-dataset-after-cross-validation) on StackExchange). This model is also stored for later use.

After the HyperDrive run, the accuracy of the best performing logistic regression model on the validation set is output, as well as the parameters of the best model. The best model is downloaded to the `./output` directory.

**What are the benefits of the parameter sampler you chose?**

Since the random parameter sampler does not search exhaustively, it can be faster than a grid search. As demonstrated in the nanodegrees' previous lectures, it can still find good model parameters.

**What are the benefits of the early stopping policy you chose?**

The early stopping policy helps in saving compute resources if there are no significant further improvements in model performance for a certain number of iterations of the HyperDrive run. Here, the performance is evaluated for every second run and requires an improvement of at least 1% point over the best result from the previous two runs. 

## AutoML

**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

AutoML yields a voting ensemble model as the best model which combines the results of different classifiers (mostly gradient boosting based classifiers, sometimes also stochastic gradient descent) by soft voting. The chosen hyperparameters are different for each classifier and understanding their meaning would require also a deeper understanding of the particular classifiers involved.

## Pipeline comparison

**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

A comparison of the models found by HyperDrive and AutoML is difficult, since they should be evaluated on an independent test set and we'd need to perform an additional significance test (see my suggestions under "Future work").

However, a direct comparison of the best model's primary metric seems to suggest that AutoML yields a model with higher accuracy than the best logistic regression model found by HyperDrive. 

Ensembles of classifiers like they are used in AutoML are well-known to yield better performance on many datasets, since a combination of models reduces the chance of overfitting and thereby reduces variance. Furthermore, the ensemble includes classifiers based on methods like gradient boosting or stochastic gradient descent, which can be considered the current state-of-the-art in machine learning due to their proven superior performance on many datasets.

## Future work

**What are some areas of improvement for future experiments? Why might these improvements help the model?**

I see potential improvements in handling and splitting the data, validation, hyperparameter optimization and evaluation using an independent test set. Not all might improve the model, but rather the run-time of experiments.

**Data handling**

We could save time by downloading, cleaning and splitting the data only once at the beginning of the notebook and the resulting subsets could then be stored as tabular datasets in Azure blob storage. They could then be re-used by the `train.py` script as well as the AutoML part. I even tried that, but unfortunately I got an error message when accessing the workspace from the `train.py` script.

**Splitting of data and validation**

When comparing the performance of models, one must be careful to work with the same subsets of the data. I tried to ensure that by splitting the data into a train, validation and test set and providing the same data (train and validation set) for hyperparameter optimization to the Scikit-learn pipeline as well as to the AutoML run. However, AutoML seems to use cross-validation for hyperparameter optimization internally, whereas the Scikit-learn pipeline uses a fixed split of the data. Since cross-validation usually yields a better performance estimate, the comparison seems to be a bit unfair. I'd therefore suggest to use cross-validation for performance estimation also in the Scikit-learn pipeline.

**Imbalanced dataset**

The AutoML run suggests that the dataset is imbalanced. This means that one class is overrepresented which can lead to an overestimate of accuracy as the performance measure. There are several ways in which this problem can be tackled. For instance, one is to work on the data sample and use undersampling, oversampling or data augmentation to achieve a better balance of training examples. Another would be to use a performance metric for optimization which can take class imbalances into account, like the F1 score. 

**Hyperparameter optimization**

It might be possible to find a logistic regression model with better performance by increasing the number of iterations.

**Evaluation on an independent test set and testing for significance**

For a fair comparison, the best models should be evaluated on an independent test set or by an outer cross-validation, because performance on the validation set can be biased (see also [article](https://stats.stackexchange.com/questions/11602/training-on-the-full-dataset-after-cross-validation) on StackExchange ). Outer cross-validation may also enable statistical testing for significance of performance differences which should always be done. Since this final evaluation does not seem to be part of the project, I left it out and provided only a comparison based on the validation set. Also, I have to admit that I currently don't know exactly how to submit the pipeline derived by AutoML and apply it to the test set.

## Proof of cluster clean up

**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**

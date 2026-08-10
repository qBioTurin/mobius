import streamlit as st 
import pandas as pd, numpy as np
import graph_tool.all as gt 

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from typing import Dict, List, Tuple, Any, Union, Iterable, Optional
from collections import defaultdict, Counter
from itertools import chain, product
import enum

from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, 
    classification_report, matthews_corrcoef, precision_recall_fscore_support, 
    accuracy_score, balanced_accuracy_score, confusion_matrix)

from sklearn.feature_selection import RFECV, SequentialFeatureSelector, SelectKBest, f_classif, mutual_info_classif, SelectFromModel
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import RepeatedStratifiedKFold, BaseCrossValidator
from sklearn.linear_model import LassoLarsIC, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from imblearn.base import BaseSampler
import joblib


from statistics import geometric_mean, harmonic_mean
import zipfile
from io import BytesIO

from tools.graph_fs import FeatureGraphEnricher
import tools.ml_utils as mlu
import tools.utils as utils
from tools.gtools import OmicsGraphFilter
import tools.mongoml as mongoml

import warnings
import logging, time 


def build_performance_dict(fset_id: str, algo_id: str, performance_dict: Dict[str, float], key_prefix: str ):
    performance_dict = { f"{key_prefix}_{key}": value for key, value in performance_dict.items() }
    return dict( fset_id = fset_id, algo_id = algo_id, **performance_dict )


def build_probs_record(fset_id: str, algo_id: str, test_set: mlu.LabelledData, estimators ) -> mlu.ProbsRecord:
    probs_block = np.vstack([ trained_e.predict_proba( test_set.features )[:, 1] for trained_e in estimators ])
    return mlu.ProbsRecord( fset_id, algo_id, np.mean( probs_block, axis=0 ), probs_block )


def get_named_algorithm( algo_id: mlu.LearningAlgorithm ):

    algo_name = None    
    kwargs = dict()  ## for future use
    
    if ( algo := mlu.LearningAlgorithm.get_algorithm( algo_id, **kwargs) ) is not None:
        algo_name = str( algo )
        algo_name = algo_name[: algo_name.index("(")]
        
    return algo, algo_name


def get_tset_performances( 
        task: mlu.TaskParameters, 
        probs_data: List[mlu.ProbsRecord], 
        test_set_list: List[ mlu.LabelledData ] ) -> Dict[str, mlu.TestSetPerformance]: #-> Tuple[Dict[str, mlu.TestSetPerformance], List[Tuple[pd.DataFrame, pd.DataFrame]]]:
    
    tset_perf_dict = dict() 
                    
    for test_set in test_set_list: #st.session_state[ mlu.ML_SessionState.VALIDATION_SETS.value ]:
        probs_dict = {
            (probs_record.fset_id, probs_record.algo_id): probs_record
                for probs_record in probs_data[ test_set.name ]
        }


        performance_test_set = pd.concat([
            get_classification_report( test_set, probs_record, task )
                for probs_record in probs_dict.values()
        ])

        tset_perf_dict[ test_set.name ] = cp = mlu.TestSetPerformance(
            test_set_id = test_set.name,
            probs_record = probs_dict,
            raw_performances = performance_test_set, 
        )
        

        #     (curr_perfs.get_named_df("ci"), curr_perfs.get_named_df("raw") ))

    return tset_perf_dict#, df_list_outfile


def extract_feature_importances(models, feature_names) -> Optional[pd.DataFrame]:
    
    def get_model(model):
        """ Extract the last model from a Pipeline or return the model itself if it is not a Pipeline. """
        return model.steps[-1][1] if isinstance(model, Pipeline) else model 
    
    def get_importances(model) -> Optional[Dict[str, float]]:
        """ Extract feature importances from a model or a Pipeline, if they are available. """
        model = get_model(model)
        importances = None 
        if hasattr(model, 'coef_'):
            coefs = model.coef_
            if coefs.ndim == 2:
                coefs = coefs[0]
            importances = np.abs(coefs)  # or keep the sign if needed 
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_

        if importances is not None: 
            importances = importances / np.sum(importances)
            importances = dict( zip( feature_names, importances ) ) 
    
        return importances
    

    feature_importances = list() 

    for model in models:    
        if (imps := get_importances(model)) is None:
            feature_importances = None 
            break 
        
        feature_importances.append( imps )
    else:
        # if break was not hit, we have a list of dicts with feature importances -> build a DataFrame
        feature_importances = pd.DataFrame(feature_importances).T

    return feature_importances 


############################# EVALUATION MANAGER ##############################################


#         _te_sets: List[ mlu.LabelledData ] ) -> Tuple[ Dict[str, float], Dict[str, mlu.ProbsRecord ] ]:
        
        
#                     for cv_metric, cv_values in crossval_stats.items() if cv_metric not in time_stats #"_time_" not in cv_metric

#                 { ts.name: build_probs_record( fset_id, algo_id, ts, fitted_estimators ) for ts in _te_sets }


def get_stats_from_confusion_matrix( y_true, y_pred ) -> Dict[str, float]:
    def get_stat( dict_pair, metric_name ) -> Tuple[float, float]:
        return dict_pair[0][metric_name], dict_pair[1][metric_name] 

    clf_report = classification_report( y_true, y_pred, output_dict=True, zero_division=0.0 )
    tf_dicts = clf_report["True"], clf_report["False"]
    precision_neg, precision_pos = get_stat( tf_dicts, "precision" )
    recall_neg, recall_pos = get_stat( tf_dicts, "recall" )
    f1_neg, f1_pos = get_stat( tf_dicts, "f1-score" )

    return dict(
        accuracy = clf_report["accuracy"],
        balanced_accuracy = (recall_pos + recall_neg) / 2,
        recall_pos = recall_pos,
        recall_neg = recall_neg,
        precision_pos = precision_pos,
        precision_neg = precision_neg,
        f1_pos = f1_pos,
        f1_neg = f1_neg
    )


def _compute_stats( m: BaseEstimator, m_name: str, X: np.array, y: np.array, clf_threshold: float = 0.5 ) -> Dict[str, float]:
    return _compute_stats_from_probs( m_id=m_name, y_true=y, y_pred_probs=m.predict_proba( X )[:, 1], clf_threshold=clf_threshold ) 


def _compute_stats_from_probs( m_id: str, y_true: np.array, y_pred_probs, clf_threshold: float = 0.5 ) -> Dict[str, float]:
    y_pred = np.where( y_pred_probs > clf_threshold, 1, 0 )
    cm_stats = get_stats_from_confusion_matrix( y_true, y_pred )
    
    return dict(
        algo_id = m_id,
        roc_auc = roc_auc_score( y_true, y_pred_probs ), 
        pr_curve = average_precision_score( y_true, y_pred_probs ),
        mcc = matthews_corrcoef( y_true, y_pred ), 
        p_threshold = clf_threshold,
        **cm_stats
    )


def _make_pipeline(scaler: StandardScaler, model: BaseEstimator, X_train: np.array, y_train: np.array) -> Pipeline:
    """ Build a fitted pipeline from an already fitted scaler and fitting a model on the provided data  """
    
    model = clone(model).fit( X_train, y_train )
    return Pipeline( steps = [ ("scaler", scaler), ("clf", model) ])


def _cv_train_test_models(
    X: np.array, y: np.array, 
    i_train: np.array, i_test: np.array, 
    algos: Dict[str, BaseEstimator], 
    sampler: BaseSampler = None) -> Tuple[ Dict[str, Pipeline], Dict[str, Dict[str, float]] ]:

    ## scaling data and optionally applying sampling, once for each fold
    X_train, X_test = X[i_train], X[i_test]
    y_train, y_test = y[i_train], y[i_test]
    scaler = StandardScaler()
    X_train_sampled, y_train_sampled = scaler.fit_transform( X_train ), y_train
    if sampler is not None:
        X_train_sampled, y_train_sampled = sampler.fit_resample( X_train_sampled, y_train_sampled )

    
    ## fit a set of learning algorithms on the scaled and "sampled" training set. r
    ## Return a dictionary of pipelines (scaler + classifier) and a
    ## dict with key = model_name, value = list of metrics in each fold
    test_metrics = defaultdict(list)
    models = dict() ## dict: key = model_name, value = pipeline

    for name, clf in algos.items():
        models[name] = pipe = _make_pipeline( scaler, clf, X_train_sampled, y_train_sampled ) #make_pipeline( scaler, clone( model ).fit( X_train_sampled, y_train_sampled ) )
        test_metrics[name].append( _compute_stats( pipe, name, X_test, y_test ) )

    return models, test_metrics


def _cv_train_test_models__beta(
    X: np.array, y: np.array, 
    i_train: np.array, i_test: np.array, 
    algos: Dict[str, BaseEstimator], 
    sampler: BaseSampler = None) -> Tuple[ Dict[str, Pipeline], Dict[str, np.array] ]:

    ## scaling data and optionally applying sampling, once for each fold
    X_train, X_test = X[i_train], X[i_test]
    y_train, y_test = y[i_train], y[i_test]
    scaler = StandardScaler()
    X_train_sampled, y_train_sampled = scaler.fit_transform( X_train ), y_train
    if sampler is not None:
        X_train_sampled, y_train_sampled = sampler.fit_resample( X_train_sampled, y_train_sampled )

    
    ## fit a set of learning algorithms on the scaled and "sampled" training set. r
    ## Return a dictionary of pipelines (scaler + classifier) and a
    ## dict with key = model_name, value = list of metrics in each fold
    y_pred_probs = defaultdict(list)
    models = dict() #dict(y_true = y) ## dict: key = model_name, value = pipeline

    for name, clf in algos.items():
        models[name] = pipe = _make_pipeline( scaler, clf, X_train_sampled, y_train_sampled ) #make_pipeline( scaler, clone( model ).fit( X_train_sampled, y_train_sampled ) )
        # test_metrics[name].append( _compute_stats( pipe, name, X_test, y_test ) )
        y_pred_probs[name].append( pipe.predict_proba( X_test )[:, 1] )

    return models, y_pred_probs, y_test


def aggregate_feature_importances(model_per_fold: List[BaseEstimator], fset_list: List[str] ) -> pd.DataFrame:
    if ( importances := extract_feature_importances( model_per_fold, fset_list )) is not None:
        return importances.agg(["mean", "std"], axis=1).sort_values("mean", ascending=False) 
    return None 


def ensemble_search__joblib(
    X: pd.DataFrame, y: np.array, 
    range_nf: List[int],
    selectors ): 

    found_sets = list() 

    for nf, s_criterion in product(range_nf, selectors):
        crit = selectors[ s_criterion ]
        match crit:
            case BaseEstimator():
                selector = SelectFromModel( crit, max_features=nf )
            case _:
                selector = SelectKBest( crit, k=nf )
        found_sets.append(
            sorted( selector.fit( X, y ).get_feature_names_out() ) 
        )

    return found_sets


def cross_validate__sampling(
        X: np.array, y: np.array, 
        cv, 
        algos: Dict[str, BaseEstimator], 
        fset_list: List[str],
        sampling: BaseSampler = None ) -> Tuple[Dict[str, List[Pipeline]], pd.DataFrame ]:
    
    with joblib.Parallel(n_jobs=-1) as parallel:
        results = parallel(
            joblib.delayed( _cv_train_test_models )
                (X, y, i_train, i_test, algos, sampling) for i_train, i_test in cv.split(X, y)
        )
        

    sorted_pipelines = defaultdict(list)
    raw_stats = list() 

    for pipelines_fold, stats_fold in results:
        for m in algos.keys():
            # raw_stats[m].extend( stats_fold[m] )
            raw_stats.extend( stats_fold[m])
            sorted_pipelines[m].append( pipelines_fold[m] )

    f_importances = {
        m_id: importances #.agg(["mean", "std"], axis=1).sort_values("mean", ascending=False)
            for m_id, pipeline_list in sorted_pipelines.items()
                if ( importances := aggregate_feature_importances( pipeline_list, fset_list ) ) is not None
    }

    return sorted_pipelines, pd.DataFrame(raw_stats), f_importances # test_stats_df


def _optimize_threshold(
    m_id: str, y_true: List[np.array], y_probs: List[np.array]) -> Dict[str, float]:
    """ Optimize the threshold for a given model based on ROC curve and return the best threshold and its metrics. """

    y_true_long = np.concatenate(y_true)
    y_probs_long = np.concatenate(y_probs)


    ## Youden's method to find the optimal threshold based on ROC curve
    fpr, tpr, thresholds = roc_curve(y_true_long, y_probs_long)
    optimal_idx = np.argmax(tpr - fpr)  # You can also use other criteria for optimal threshold
    optimal_threshold = thresholds[optimal_idx]    
    optimal_threshold = optimal_threshold if optimal_threshold is not None and np.isfinite(optimal_threshold) else 0.5  


    return [
        _compute_stats_from_probs( m_id, y_true_curr, y_probs_curr, clf_threshold=optimal_threshold )
            for y_true_curr, y_probs_curr in zip(y_true, y_probs)
    ]

    
def cross_validate__sampling__beta(
        X: np.array, y: np.array, 
        cv, 
        algos: Dict[str, BaseEstimator], 
        fset_list: List[str],
        sampling: BaseSampler = None ) -> Tuple[Dict[str, List[Pipeline]], pd.DataFrame ]:
    

    with joblib.Parallel(n_jobs=-1) as parallel:
        cv_results = parallel(
            joblib.delayed( _cv_train_test_models__beta )
                (X, y, i_train, i_test, algos, sampling) for i_train, i_test in cv.split(X, y)
        )

        pipelines_fold, y_data_fold, y_true_fold = zip(*cv_results)
        y_predictions = defaultdict(list)

        for y_rec in y_data_fold:
            for m, y_data in y_rec.items():
                y_predictions[m].extend( y_data )
        
        opt_metrics = parallel(
            joblib.delayed( _optimize_threshold )
                (m_id, y_true_fold, y_probs) for m_id, y_probs in y_predictions.items()
        )
        
    sorted_pipelines = defaultdict(list)
    raw_stats = chain.from_iterable( opt_metrics )

    for pf in pipelines_fold:
        for m_id, model in pf.items():
            sorted_pipelines[m_id].append( model )

    if False:
        for pipelines_fold, stats_fold in results:
            for m in algos.keys():
                # raw_stats[m].extend( stats_fold[m] )
                raw_stats.extend( stats_fold[m])
                sorted_pipelines[m].append( pipelines_fold[m] )

    f_importances = {
        m_id: importances #.agg(["mean", "std"], axis=1).sort_values("mean", ascending=False)
            for m_id, pipeline_list in sorted_pipelines.items()
                if ( importances := aggregate_feature_importances( pipeline_list, fset_list ) ) is not None
    }

    return sorted_pipelines, pd.DataFrame(raw_stats), f_importances # test_stats_df


#########################################################################################################################


def fit_and_eval_model__loocv(
    algo_id: mlu.LearningAlgorithm, 
    fset_id: str,
    _tr_set: mlu.LabelledData, _te_sets: List[ mlu.LabelledData ], 
    clf_threshold: float = 0.5 ):

    algo_details = my_algo, algo_id = get_named_algorithm( algo_id )

    if None not in algo_details: 
        crossval_stats = cross_validate( 
            my_algo, 
            _tr_set.features, _tr_set.target, 
            cv = LeaveOneOut(), 
            return_estimator=True, return_indices=True )
        
        fitted_estimators = crossval_stats.pop("estimator")
        importances = aggregate_feature_importances( fitted_estimators, _tr_set.features.columns.tolist() )
        print(f"Feature importances for {algo_id} on {fset_id}: {importances}")
        
        y_pred, y_true__loocv = [
            np.hstack(arrays) for arrays in zip(*[
                (estimator.predict_proba(_tr_set.features.iloc[ idx_sample ])[:,1], _tr_set.target.iloc[ idx_sample ])
                    for estimator, idx_sample in zip( fitted_estimators, crossval_stats["indices"]["test"] )
        ])]
        

        metric_values = _compute_stats_from_probs( m_id=algo_id, y_true=y_true__loocv, y_pred_probs=y_pred )

        
        if False:
            cm_stats = get_stats_from_confusion_matrix( y_true__loocv, y_pred_labels )

            metric_values = dict( 
                roc_auc = roc_auc_score(y_true__loocv, y_pred ), 
                pr_curve = average_precision_score( y_true__loocv, y_pred ),
                mcc = matthews_corrcoef(y_true__loocv, y_pred_labels ), 
                **cm_stats
            )


        return (
            build_performance_dict( fset_id, algo_id, metric_values, key_prefix="loocv_" ),
            { ts.name: build_probs_record( fset_id, algo_id, ts, fitted_estimators ) for ts in _te_sets }, 
            { (algo_id, fset_id): importances }  # feature importances for the current algorithm - TODO: be aggregated
        )


class FeatureEvaluationManager:

    @classmethod
    def prepare_features( cls, input_data: List[mlu.LabelledData], feature_list: List[str] ):
        return [ d.select_features( feature_list ) for d in input_data ] 


    @classmethod
    def models_evaluation__sampling( 
            cls, 
            input_data: List[mlu.LabelledData], 
            fset_collection: Dict[ str, List[str]], 
            algos_id: List[mlu.LearningAlgorithm], cv_obj, 
            sampler: BaseSampler = None
        ) -> Tuple[ pd.DataFrame, Dict[str, List[mlu.ProbsRecord]], Dict[str, pd.DataFrame] ]:

        ## init results data structures 
        tset_results = { tset.name: list() for tset in input_data[1:] } 
        cv_results = list() 
        f_importance_collection = dict() ## fset_id -> feature importances DataFrame
        

        learning_algos = { 
            name: algo for algo, name in [ 
                get_named_algorithm(algo_id) for algo_id in algos_id ]}

        for f_id, fset in fset_collection.items():
            ## prepare data for training and testing
            curr_data = cls.prepare_features( input_data, fset )
            curr_training, curr_test = curr_data[0], curr_data[1:]
            ## get numpy arrays for training
            X_train, y_train = curr_training.features.to_numpy(), curr_training.target.to_numpy()
            models, cv_performances, f_importances = cross_validate__sampling__beta(X_train, y_train, cv_obj, learning_algos, fset, sampler )
            # cross_validate__sampling__beta

            f_importance_collection[ f_id ] = f_importances

            cv_performances.insert(0, "fset_id", f_id)
            cv_performances.insert(1, "n_features", len(fset))
            cv_results.append( cv_performances )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                for ts in curr_test:
                    tset_results[ ts.name ].extend([
                        build_probs_record( f_id, algo_id, ts, fitted_pipelines )
                            for algo_id, fitted_pipelines in models.items() 
                    ])

        return (
            pd.concat( cv_results ).set_index(["algo_id", "fset_id"]), 
            tset_results, 
            f_importance_collection
        )


# { ts.name: build_probs_record( fset_id, algo_id, ts, fitted_estimators ) for ts in _te_sets }

    if False:
        @classmethod
        def evaluation( cls, input_data: List[mlu.LabelledData], fset_id: str, feature_list: List[str], algos_id: List[mlu.LearningAlgorithm], cv_splits, cv_metrics ): ## ex evaluate 
            curr_data = cls.prepare_features( input_data, feature_list ) #self.__get_data( feature_list )
            curr_training, curr_test = curr_data[0], curr_data[1:]

            logging.info(f"\n\nEvaluate {fset_id} with {algos_id}\n\n")

            match cv_splits:
                case LeaveOneOut():
                    return [ 
                        fit_and_eval_model__loocv( algo_id, fset_id, curr_training, curr_test ) 
                            for algo_id in algos_id 
                    ]
                case _:
                    return [ 
                        fit_and_eval_model( algo_id, fset_id, cv_metrics, cv_splits, curr_training, curr_test ) 
                            for algo_id in algos_id 
                    ] 
        
    @classmethod
    def evaluation__loo( cls, input_data: List[mlu.LabelledData], fset_id: str, feature_list: List[str], algos_id: List[mlu.LearningAlgorithm] ): ## ex evaluate 
        curr_data = cls.prepare_features( input_data, feature_list ) #self.__get_data( feature_list )
        curr_training, curr_test = curr_data[0], curr_data[1:]

        logging.info(f"\n\nEvaluate {fset_id} with {algos_id}\n\n")
        to_apply = map( lambda algo_id: fit_and_eval_model__loocv( algo_id, fset_id, curr_training, curr_test ), algos_id)
        return list( to_apply )


    @classmethod
    def postprocess_results_evaluation( cls, fit_results: List ):
        results_training, results_test_sets, unsorted_f_importances = zip( *fit_results )
        f_importances = defaultdict( dict )
        for curr_dict in unsorted_f_importances:
            k_pair = algo_id, fset_id = list( curr_dict.keys() )[0]
            
            if ( curr_imps := curr_dict.get(k_pair) ) is not None:
                f_importances[fset_id][algo_id] = curr_imps
    
    
        ## reorganize test sets evaluation results 
        testsets_probs_record = defaultdict( list )
        for item in results_test_sets:
            for tset_id, probs_record in item.items():
                # Adding probs record ({probs_record.fset_id, probs_record.algo_id}) for {tset_id}
                testsets_probs_record[ tset_id ].append( probs_record )

        results_training = pd.DataFrame( results_training ).set_index(["algo_id", "fset_id"])
        return results_training, dict( testsets_probs_record ), f_importances

    
    @classmethod
    def models_evaluation_wrapper( 
        cls, 
        input_data: List[mlu.LabelledData], 
        fset_collection: Dict[ str, List[str]], 
        algos_id: List[mlu.LearningAlgorithm], 
        cv_splits, 
        # cv_metrics, ##TODO: remove this parameter
        sampler: BaseSampler = None 
    ) -> Tuple[ pd.DataFrame, Dict[str, List[mlu.ProbsRecord]] ]:
        

        #             for fset_id, fset_list in fset_collection.items()
        match cv_splits:
            case LeaveOneOut():
                ## TODO: scale data w.r.t. the training set 
                scaler = StandardScaler().fit( input_data[0].features )
                scaled_data = [ d.scale_data( scaler ) for d in input_data ]
                fit_results = chain.from_iterable([
                    cls.evaluation__loo( scaled_data, fset_id, fset_list, algos_id )
                        for fset_id, fset_list in fset_collection.items()
                ])
                return cls.postprocess_results_evaluation( list( fit_results ) )

            case _:
                perfs_df, probs_data, f_importances = cls.models_evaluation__sampling( input_data, fset_collection, algos_id, cv_splits, sampler ) 
                return perfs_df, probs_data, f_importances


    @classmethod
    def ensemble_evaluation( cls, input_data: List[mlu.LabelledData], ensemble_features: mlu.EnsembleFeatureSet, list_algos_id: List[ mlu.LearningAlgorithm ], k_folds: int = 5, n_repeats: int = 3 ):
        ensemble_collection = {
            # f"{ensemble_features.name}_{len(fset)}": fset 
            wrap_fset.uid: wrap_fset.feature_list
                for wrap_fset in ensemble_features.components.values()
        }
        kfold_cv = RepeatedStratifiedKFold( n_repeats=n_repeats, n_splits = k_folds, random_state=42 )
        perfs_df, probs_data, f_import = cls.models_evaluation__sampling( input_data, ensemble_collection, list_algos_id, kfold_cv ) 
        return perfs_df, probs_data

 
def build_roc_plot( title: str, y_true: np.array, y_pred: Dict[ str, np.array ], auc_thr: float = 0.5, pr_thr: float = 0.7, color_traces: Dict[str, str] = None ) -> go.Figure:
    fig = make_subplots( rows = 1, cols = 2, subplot_titles=("ROC Curve", "Precision-Recall Curve"))

    if color_traces is None:
        color_traces = dict()


    for trace_id in sorted( y_pred ):
        color = color_traces.get( trace_id, None )
        y_prob = y_pred[ trace_id ]
        fpr, tpr, _ = roc_curve( y_true, y_prob )
        auc_score = roc_auc_score( y_true, y_prob )
        trace_name = f"{trace_id} (AUROC={auc_score:.2f})"

        dash = "solid" if auc_score >= auc_thr else "dash"
        fig.add_trace( 
            go.Scatter( x = fpr, y = tpr, name = trace_name, mode = "lines", line = dict(dash = dash, width=5, color=color)  ), 
            row = 1, col = 1 )
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob )
        avg_score = average_precision_score( y_true, y_prob )
        trace_name = f"{trace_id} (AUPR={avg_score:.2f})"

        dash = "solid" if avg_score >= pr_thr else "dash"
        fig.add_trace(
            go.Scatter( x = recall, y = precision, name = trace_name, mode = "lines", line = dict(dash = dash, width=5, color=color) ), 
            row = 1, col = 2
        )


    ## PUT THE BASELINE
    fig.add_trace(
        go.Scatter(
            x = [0, 1], y = [0, 1], name = "baseline (AUC=0.5)", 
            mode = "lines", line = dict(dash="dash", color="black"), showlegend = False
        ), 
        row = 1, col = 1
    )
    pr_baseline = np.mean( y_true )
    fig.add_trace(go.Scatter(
        x = [0, 1], y=[pr_baseline, pr_baseline],
        mode='lines', line=dict(dash='dash', color='black'), showlegend=False,
        name=f'Baseline (pr={pr_baseline:.2f})'
    ), row =1, col=2)

    axes_info = (
        ( "False Positive Rate (1-Specificity)", "True Positive Rate (Sensitivity)" ),
        ( "Recall", "Precision" )
    )

    for i, (x_ax, y_ax) in enumerate( axes_info, 1 ):
        fig.update_xaxes( title_text = x_ax, range=[0,1], row = 1, col = i )
        fig.update_yaxes( title_text = y_ax, range=[0,1], row = 1, col = i )

    fig.update_layout( title_text = title, width=700, height=500 )
    
    return fig 


def get_classification_report( test_set: mlu.LabelledData, probs_record: mlu.ProbsRecord, task: mlu.TaskParameters ) -> pd.DataFrame:
   
    y_true = test_set.target
    rows = list()
    index_cols = [ "fset_id", "algo_id", "cv_id" ]
    class_metrics =  [ "precision", "recall", "f1-score" ]
    class_names = { False: "&".join(task.neg_class), True: "&".join(task.pos_class) }
    per_class_metric_names = [ f"{class_names[name]}_{cm}" for name in class_names for cm in class_metrics  ]
    

    for i, y_pred in enumerate( probs_record.raw_probs ):
        y_pred_labels = np.where( y_pred < 0.5, 0, 1 )

        clf_report = classification_report( y_true, y_pred_labels, output_dict=True, zero_division=0.)#, labels = ("Healthy", "CRC") )

        neg_stats, pos_stats = [clf_report[ str(boolean) ] for boolean in (False, True ) ]
        prec_rec_f1__neg = [ neg_stats[ metric_name ] for metric_name in class_metrics ]
        prec_rec_f1__pos = [ pos_stats[ metric_name ] for metric_name in class_metrics ]
        
        
        rows.append([
            probs_record.fset_id, probs_record.algo_id, i+1, 
            roc_auc_score( y_true, y_pred ), 
            average_precision_score( y_true, y_pred ),
            clf_report.get("accuracy"), 
            matthews_corrcoef( y_true, y_pred_labels ), 
            *prec_rec_f1__neg, 
            *prec_rec_f1__pos
        ])

    return pd.DataFrame(
        data = rows, 
        columns = [ *index_cols, "auc_score", "pr_auc", "accuracy", "mcc", *per_class_metric_names ]
    )#.set_index( index_cols )


class FeatureSelectorManager:

    @classmethod
    def apply_filter(cls, input_data: mlu.LabelledData, filter_method: str, num_features: int ) -> List[str]:
        filter_criterion = f_classif if filter_method == "ANOVA" else mutual_info_classif
        selector = SelectKBest( score_func=filter_criterion, k = num_features ).fit( input_data.features, input_data.target )
        return sorted( selector.get_feature_names_out() ) 

    @classmethod
    def apply_embedded(cls, input_data: mlu.LabelledData, embed_method: mlu.EmbeddedFeatureSelection, ub_num_features: int ) -> List[str]:
        hyperparams = mlu.EmbeddedFeatureSelection.get_hyperparameters( embed_method )
        embed = SelectFromModel( 
            estimator=mlu.EmbeddedFeatureSelection.get_algorithm( embed_method, **hyperparams ),
            threshold="median", max_features=ub_num_features)
        embed.fit( input_data.features, input_data.target )
        return sorted( embed.get_feature_names_out() )
 

    @classmethod
    def apply_wrapper(cls, input_data: mlu.LabelledData, wrapper_method: str, underlying_clf: mlu.LearningAlgorithm ) -> List[str]:
        clf = mlu.LearningAlgorithm.get_algorithm( underlying_clf )
        if wrapper_method == "RFE":
            fsel = RFECV( estimator=clf, cv = 5, min_features_to_select=2, verbose=2 )
        else:
            fsel = SequentialFeatureSelector( clf )

        fsel.fit( input_data.features, input_data.target )
        return sorted( fsel.get_feature_names_out() )
 

    @classmethod
    def apply_non_parametric(cls, input_data: mlu.LabelledData ) -> Dict[str, List[str]]:
        selected_list = dict()

        for criterion in ["aic", "bic"]:
            try:
                lasso_model = LassoLarsIC(criterion=criterion).fit(input_data.features, input_data.target)
            except ValueError: 
                ## avoiding case N_samples > N_features: compute estimated noise variance from linear regression model 
                reg = LinearRegression().fit( input_data.features, input_data.target )
                residuals = input_data.target - reg.predict(input_data.features)
                estimated_noise_variance = np.var(residuals)
                lasso_model = LassoLarsIC(
                    criterion=criterion, 
                    noise_variance=estimated_noise_variance ).fit(input_data.features, input_data.target)
            finally:
                coefs = lasso_model.coef_

            selected = [ feature for feature, coef in zip(input_data.features.columns.tolist(), coefs) if coef != 0 ]
            if len(selected) > 0:
                selected_list[ criterion.upper() ] = sorted( selected )
            else:
                logging.warning(f"We're sorry but LassoLars {criterion} results in zero features :()")
                
        return selected_list
        
 
    @classmethod
    def apply_greedy_search(cls, feature_graph: gt.Graph, score_list: List[str], feature_pool: List[str], min_nf: int, max_nf: int, target_cov: str ) -> Dict[int, List[str]]: 
        df_weights = FeatureGraphEnricher.get_dataframe_vertex_weights( feature_graph, target_cov )
        df_weights = df_weights[ score_list ].loc[ feature_pool ] 
        col_sums = df_weights.sum(axis="index")
        
        for col in df_weights.columns.tolist():
            df_weights[col] /= col_sums[col]
            
        mean_scores = [ harmonic_mean( row.to_numpy() ) for i, row  in df_weights.iterrows() ]
        df_weights["mean"] = mean_scores

        df_weights.sort_values(by="mean", ascending=False, inplace=True)    
        sorted_features = df_weights.index.tolist()
        fset_dict = { nf: sorted_features[:nf] for nf in range(min_nf, max_nf+1) }

        return fset_dict 


    @classmethod
    def apply_ensemble(cls, 
                       input_data: mlu.LabelledData, feature_set: List[str], 
                       min_nf: int, max_nf: int, 
                       n_reps: int = 3, 
                       st_stream = None ) -> Dict[int, List[str]]:# -> mlu.EnsembleFeatureSet:
            

        def stratified_bootstrap(df: pd.DataFrame, random_state: int) -> Tuple[pd.DataFrame, pd.Series]:
            """Return one stratified bootstrap sample (same size as df)."""
            
            target_col = "__target__"

            rep = (
                df.groupby(target_col, group_keys=False)
                .apply(lambda x: x.sample(n=len(x), replace=True, random_state=random_state))
                # .drop(columns=[target_col])
                # .sample(frac=1, random_state=random_state)  # optional shuffle
            )

            return rep.drop(columns=[target_col]), rep[target_col]

        def selection_step( dataset: mlu.LabelledData, min_nf: int, max_nf: int, selectors ):
            df = dataset.features.copy()
            df["__target__"] = dataset.target


            with joblib.Parallel(n_jobs=-1) as parallel:
                range_nf = list( range( min_nf, 1 + max_nf ) )

                results = parallel(
                    joblib.delayed(ensemble_search__joblib)(
                        *stratified_bootstrap( df, random_state=seed ), range_nf, selectors
                    ) for seed in range(n_reps)
                )

                fset_count = Counter([ 
                    tuple(fset) 
                        for seq in results 
                            for fset in seq 
                                if min_nf <= len(fset) <= max_nf ]) ## required condition to avoid weird bug for nf = 0

            return pd.DataFrame( [
                dict( nf=len(fset), supp=count, fset=fset ) for fset, count in fset_count.items()
            ]).sort_values( by = ["nf", "supp"], ascending=(True, False))


        def selection_step__crossval( dataset: mlu.LabelledData, min_nf: int, max_nf: int, selectors ):
            rep_kfold = RepeatedStratifiedKFold(n_splits = k_folds, n_repeats = n_reps, random_state=42)
            splits, _ = zip( *list( rep_kfold.split( dataset.features, dataset.target) ) ) 
            range_nf = list( range( min_nf, 1 + max_nf ) )
            fset_count = Counter()

            with joblib.Parallel(n_jobs=-1) as parallel:
                prep_data = map(
                    lambda a: mlu.LabelledData(
                        dataset.features.iloc[a], dataset.target.iloc[a],
                        dataset.metadata.iloc[a], dataset.cov_encoding
                    ).scale_data(), 
                    splits
                )
                results = parallel(
                    joblib.delayed(ensemble_search__joblib)(
                        wrapped_data.features, wrapped_data.target, range_nf, selectors
                    ) for wrapped_data in prep_data
                )

                for seq in results:
                    for fset in seq:
                        fset_count[ tuple( fset ) ] += 1

            return pd.DataFrame( [
                dict( nf=len(fset), supp=count, fset=fset ) for fset, count in fset_count.items()
            ]).sort_values( by = ["nf", "supp"], ascending=(True, False))


        def choice_step( df_ensemble: pd.DataFrame) -> dict:
            ensemble_res = dict()

        
            for curr_nf, subdf in df_ensemble.groupby("nf"):
                counter = Counter( chain.from_iterable( [ str_tuple for str_tuple in subdf.fset.values  ] )  )
                most_common = counter.most_common(curr_nf)
                try:
                    flist, _ = zip(*most_common)
                    ensemble_res[ len(flist) ] = sorted( flist )
                except ValueError:
                    logging.critical(f"Initial dataframe:\n{df_ensemble}")
                    logging.critical(f"Unexpected error in extracting most common features for nf={curr_nf} -- Counter: {counter} -- most common: {most_common}")

            return ensemble_res

        def write_in_stream( message: str ):
            if st_stream is not None:
                st_stream( message )


        my_selectors = {
            "anova": f_classif, 
            "mi": mutual_info_classif, 
            "lasso": mlu.LearningAlgorithm.get_algorithm( mlu.LearningAlgorithm.LOGISTIC_REGRESSION, penalty="l1", solver="saga"), 
            "logreg": mlu.LearningAlgorithm.get_algorithm( mlu.LearningAlgorithm.LOGISTIC_REGRESSION, penalty="l2" ),
            "rforest": mlu.LearningAlgorithm.get_algorithm( mlu.LearningAlgorithm.RANDOM_FOREST ), 
            "enet": mlu.EmbeddedFeatureSelection.get_algorithm( mlu.EmbeddedFeatureSelection.LOGISTIC_REGRESSION_ENET, **mlu.EmbeddedFeatureSelection.get_hyperparameters( mlu.EmbeddedFeatureSelection.LOGISTIC_REGRESSION_ENET ) ),
            "linearSVM_l1": mlu.EmbeddedFeatureSelection.get_algorithm( mlu.EmbeddedFeatureSelection.LINEAR_SVM, penalty="l1", dual=False), 
            "linearSVM": mlu.EmbeddedFeatureSelection.get_algorithm( mlu.EmbeddedFeatureSelection.LINEAR_SVM, penalty="l2", dual = False), 
            "lda": mlu.LearningAlgorithm.get_algorithm( mlu.LearningAlgorithm.LINEAR_DA )
        }


        write_in_stream( f"* Feature selection from a pool of {len(feature_set)} features combining {len(my_selectors)} selectors" )
        training_data = input_data.select_features( feature_set )
        write_in_stream( f"* Selecting feature sets with length in [{min_nf}, {max_nf}] using bootstrap resampling repeated {n_reps} times" )
        df = selection_step( training_data, min_nf, max_nf, my_selectors )
        write_in_stream( f"* Choosing feature sets through majority voting" )
        ensemble_dict = choice_step( df )
        return ensemble_dict
    

class FeatureSetManager:
    def __init__(self, case_study_id: str, feature_graph: gt.Graph, initial_sets: Dict[str, List[str]]):
        self.__graph = feature_graph
        self.__mdb = mongoml.connect_to_mongo( db_name="mobius" )
        self.__mdb_features = "features"
        

        self.__case_study_id = case_study_id
        self.__feature_groups = initial_sets
        self.__id_count = None 
        self.__initialize_starting_features()
        

    @property
    def ensembles(self) -> Dict[str, mlu.EnsembleFeatureSet]:
        query = {"ftype": mlu.FeatureType.ENSEMBLE_FEATURESET.value}
        ens_results = mongoml.find( self.__mdb, self.__mdb_features, self.__case_study_id, query )
        ens_collection = {
            ens_doc["uid"]: self.__create_ensemble_from_document( ens_doc )
                for ens_doc in ens_results
        }
        return ens_collection
    
    
    @property
    def ga_generations(self) -> Dict[str, mlu.GAGeneration]:
        ga_generations = dict()
        query = {"ftype": mlu.FeatureType.GA_GENERATION.value}

        for ga_run in mongoml.find( self.__mdb, self.__mdb_features, self.__case_study_id, query ):
            name = ga_run["uid"]
            ga_generations[name] = mlu.GAGeneration(
                name, 
                feature_pool = mlu.FeatureSubset.decode_fset( ga_run["feature_pool"] ),
                solutions = [ mlu.FeatureSubset.decode_fset( sol ) for sol in ga_run["solutions"] ]
            )

        return ga_generations
    

    def __initialize_starting_features(self):
        search_result = mongoml.find_one( 
            db = self.__mdb,
            collection_name=self.__mdb_features, 
            case_study_id=self.__case_study_id,
            query={"ftype": mlu.FeatureType.SIMPLE.value} )

        if search_result is None:
            starters = self.get_initial_featuresets()
            self.__id_count = 1 
            logging.critical(f"Inserting {len(starters)} initial feature sets in MongoDB")
            mongoml.insert_many( 
                db = self.__mdb, 
                collection_name = self.__mdb_features, 
                case_study_id = self.__case_study_id,
                records = [ obj.wrap_for_mongo() for obj in starters ] )
        else:
            num_fsets = mongoml.count_documents(
                db = self.__mdb, 
                collection_name = self.__mdb_features, 
                case_study_id = self.__case_study_id,
                query = {"ftype": mlu.FeatureType.SIMPLE.value} ) 
            self.__id_count = num_fsets - len(self.__feature_groups) + 1
            logging.critical(f"Initial feature sets already present in MongoDB")


    def get_initial_featuresets(self) -> List[mlu.FeatureSubset]:
        fset_objs = [
            mlu.FeatureSubset( flist, "user", 0, self.__graph, self.__feature_groups, uid = fset_uid )
                for fset_uid, flist in self.__feature_groups.items()
        ]

        return fset_objs
            

        ## insert the initial feature sets in the database
        mongoml.insert_many( 
            self.__mdb, 
            self.__mdb_features, 
            [ obj.wrap_for_mongo() for obj in fset_objs ] )

        
    def add_ga_generation(self, ga_gen: mlu.GAGeneration ):


        search_query = mongoml.find_one(
            self.__mdb, self.__mdb_features, self.__case_study_id, 
            {"uid": ga_gen.name, "ftype": mlu.FeatureType.GA_GENERATION.value} )

        logging.critical(f"Searching for GA generation {ga_gen.name} in MongoDB: {search_query}")

        if search_query is None:
            logging.critical(f"Saving GA generation {ga_gen.name} to MongoDB")
            mongoml.insert_one( self.__mdb, self.__mdb_features, self.__case_study_id, ga_gen.wrap_for_mongo() )
            return True 

        logging.critical(f"Failed to save GA generation {ga_gen.name} to MongoDB. But WHYYY???")
            # self.__ga_generations[ ga_gen.name ] = ga_gen


    def add_ensemble(self, ens_uid: str, ens_dict: Dict[int, List[str]] ):
        if False:
            assert ens_uid not in self.__ensemble_sets
            range_fset = min( ens_dict.keys() ), max( ens_dict.keys() )
            ens_dict = {
                set_size: self.wrap_feature_set( sorted(flist), "", f"{ens_uid}_{set_size}") 
                    for set_size, flist in ens_dict.items() 
            }
            self.__ensemble_sets[ ens_uid ] = ens_fset = mlu.EnsembleFeatureSet(
                ens_uid, range_fset, ens_dict
            )

        search_result = mongoml.find_one(
            db = self.__mdb, 
            collection_name = self.__mdb_features, 
            case_study_id = self.__case_study_id,
            query = {"uid": ens_uid, "ftype": mlu.FeatureType.ENSEMBLE_FEATURESET.value} )


        if search_result is None:   
            range_fset = min( ens_dict.keys() ), max( ens_dict.keys() ) 
            ens_dict = {
                set_size: self.wrap_feature_set( sorted(flist), "", f"{ens_uid}_{set_size}") 
                    for set_size, flist in ens_dict.items() 
            }
            ens_fset = mlu.EnsembleFeatureSet( ens_uid, range_fset, ens_dict )
            mongoml.insert_one( 
                db = self.__mdb, 
                collection_name = self.__mdb_features, 
                case_study_id=self.__case_study_id, 
                record = ens_fset.wrap_for_mongo() )
      
        # self.__ensemble_sets[ ens.name ] = ens


    def __create_ensemble_from_document(self, ens_doc: Dict[str, Any]) -> mlu.EnsembleFeatureSet:
        ens_id = ens_doc["uid"]
        lb, ub = ens_doc["lower_bound"], ens_doc["upper_bound"]

        fsets_from_ens = dict()

        for fset in ens_doc["components"]:
            raw_flist = mlu.FeatureSubset.decode_fset( fset["flist"] )
            fsets_from_ens[ len(raw_flist) ] = self.wrap_feature_set(
                raw_flist, fset["metadata"], fset["uid"]
            )

        return mlu.EnsembleFeatureSet(
            ens_id, 
            feature_range=(lb, ub),
            components=fsets_from_ens
        )


    def get_ensemble(self, ens_id: str) -> mlu.EnsembleFeatureSet:
        ens_mongo = mongoml.find_one( 
            db = self.__mdb, 
            collection_name = self.__mdb_features, 
            case_study_id=self.__case_study_id,
            query = {"uid": ens_id, "ftype": mlu.FeatureType.ENSEMBLE_FEATURESET.value}, 
            projection = {"_id": 0} )
        
        if ens_mongo is not None:
            return self.__create_ensemble_from_document( ens_mongo )
        

    @property
    def df_featuresets(self) -> pd.DataFrame:
        #     {"_id": 0, "ftype": 0})
        avail_fsets = mongoml.find( 
            db = self.__mdb, 
            collection_name = self.__mdb_features, 
            case_study_id = self.__case_study_id,
            query = {"ftype": mlu.FeatureType.SIMPLE.value}, 
            projection = {"_id": 0, "ftype": 0} )
        
        avail_fsets = list( avail_fsets )
        df = pd.DataFrame( data = avail_fsets).set_index( mlu.FeatureSubset.get_name_index_column() )
        df.flist = df.flist.apply( mlu.FeatureSubset.decode_fset )
        return df
    

    def add_feature_set(self, fset: List[str], metadata: Any) -> mlu.FeatureSubset:        
        ## check if the feature set is already present in the database
        fset = tuple( sorted( set( fset ) ) )
        fset_string = mlu.FeatureSubset.encode_fset( fset )

        if mongoml.find_one( self.__mdb, self.__mdb_features, self.__case_study_id, {"flist": fset_string} ) is None:
            fset_data = self.wrap_feature_set( fset, metadata ).wrap_for_mongo()
            mongoml.insert_one( self.__mdb, self.__mdb_features, self.__case_study_id, fset_data )
            self.__id_count += 1
    
    def remove_feature_set(self, fset_composition: List[str]) -> Optional[str]:
        doc = mongoml.find_one( self.__mdb, self.__mdb_features, self.__case_study_id, {"flist": mlu.FeatureSubset.encode_fset( fset_composition )} )
        if doc is not None:
            logging.critical(f"Removing feature set {doc['uid']} from MongoDB")


            try:
                mongoml.delete_one( self.__mdb, self.__mdb_features, self.__case_study_id, { "_id": doc["_id"] } )
                return doc.get("uid", None)  # Return the uid of the removed feature set
            except TypeError as e:
                logging.error(f"Error during deletion of feature set {doc.get('uid', 'unknown')}: {e}")
                return None


    def wrap_feature_set(self, sorted_fset: Tuple[str], metadata: Any, uid: str = None ) -> mlu.FeatureSubset:
        return mlu.FeatureSubset( sorted_fset, metadata, self.__id_count, self.__graph, self.__feature_groups )

    
class DataLoaderManager:

    def __init__(self, 
                input_zipfile: str, 
                task: mlu.TaskParameters, 
                user_covariates: List[str], 
                feature_graph_id: str ):
        
        logging.critical(f"Loading data from {input_zipfile}")
        data, features, encoding = utils.load_prepdata( input_zipfile )
        logging.critical(f"Reading the fucking graph...")
        
        self.__selected_covs = list( set( features.pop("covariates") ).intersection(
            user_covariates + [ task.target_cov ]
        ))

        ### remove covariates having just less than two values
        training_data = data["training_set"][ self.__selected_covs]
        value_counts = training_data[ self.__selected_covs ].nunique()
        one_val_covariates = set(value_counts[ value_counts <= 1 ].index.tolist())
        logging.critical(f"Removing covariates with n.values < 2: {one_val_covariates}")
        self.__selected_covs = list( filter( lambda cov: cov not in one_val_covariates, self.__selected_covs ) )

        self.__f_groups = { group_id: feature_list for group_id, feature_list in features.items() }
        self.__data = data 
        self.__task = task  
        self.__feature_graph = self.__prepare_feature_graph( feature_graph_id )

        
    def __prepare_feature_graph(self, graph_file_id: str ) -> Union[gt.Graph, gt.GraphView]:
        """ Load the feature graph from the input zipfile """ 

        def get_filtered_feature_graph( g: gt.Graph, pvalue_threshold: float ):
            """ Set the edge filter based on the pvalue threshold """

            final_bitmask = np.ones(shape=(g.num_edges(), ), dtype=bool)
            if pvalue_threshold is not None and pvalue_threshold < 1: # threshold < 1:
                final_bitmask = np.zeros(shape=(g.num_edges(), ), dtype=bool)
                for graph_stratification in g.gp[ mlu.GraphField.STRAT_LAYERS.value ].split("$"):
                    final_bitmask = np.bitwise_or( 
                        final_bitmask, 
                        g.ep[ mlu.GraphField.ADJ_PVALUE.value.format(graph_stratification) ].a < pvalue_threshold )
                    
        
                return gt.GraphView( g, efilt = final_bitmask )
        
            return g 

        if False:
            with zipfile.ZipFile( input_zipfile, "r" ) as main_zipfile:
                start_t = time.time()
                logging.critical(f"Reading feature graph from {input_zipfile}...")
                feature_graph = gt.load_graph( BytesIO( main_zipfile.read( "feature_graph.gt" ) ) )
                logging.critical(f"Feature graph loaded in {time.time() - start_t:.2f} seconds")
                return get_filtered_feature_graph( feature_graph, 1 ) 

        start_t = time.time()
        logging.critical(f"Retrieving feature graph w/ id={graph_file_id} from GridFS...")
        feature_graph = mongoml.retrieve_graph_from_mongo__with_id( graph_file_id )
        logging.critical(f"Feature graph loaded in {time.time() - start_t:.2f} seconds")
        return get_filtered_feature_graph( feature_graph, 1 ) 


    def __prepare_datasets(self) -> Tuple[ List[mlu.LabelledData], Dict[str, List[str]] ]:
        data_ids = [ "training_set", "test_set" ]
        data_ids.extend([ key for key in self.__data.keys() if key not in data_ids ])

        lab_data_names = [ "Discovery Set", "Validation Set" ]
        lab_data_names.extend( data_ids[2:])

        data_prep = mlu.DataPreparator(
            dataset_list = [ (name, self.__data.get(key)) for key, name in zip(data_ids, lab_data_names)], 
            task = self.__task, 
            covariates = self.__selected_covs, 
            features = list( chain.from_iterable( self.__f_groups.values() ) )
        )

        lab_data = data_prep.preprocessing() 

        return lab_data, data_prep.covariate_encoding
                

    def __init_feature_lists( self, st_session: Dict[str, Any]):
        def init_session_field( field: mlu.ML_SessionState, default_value: Any ):
            if ( _field := field.value ) not in st_session:
                st_session[ _field ] = default_value
            else: 
                logging.warning(f"Entry {_field} is already initialized.")
        
        init_session_field( mlu.ML_SessionState.PERFORMANCE_RECORD, [] )
        init_session_field( mlu.ML_SessionState.PERFORMANCE_RECORD_ENSEMBLES, [] )
        init_session_field( mlu.ML_SessionState.SELECTED_FEATURES, [] )


    def initialize_session(self, st_session: Dict[str, Any] ):
        datasets, cov_encoding = self.__prepare_datasets()
        task_info = mlu.TaskParameters( self.__task.target_cov, self.__task.pos_class, self.__task.neg_class)
        case_study_id = task_info.get_unique_study_identifier( self.__f_groups )
        #     (omic_id, ",".join(sorted( self.__f_groups[omic_id] )))
        #         for omic_id in sorted( self.__f_groups.keys() ) 
        ## get univocal identifier for current case study considering the task information and the set of initial features 

        fset_manager = FeatureSetManager(
            case_study_id=case_study_id, 
            feature_graph=self.__feature_graph, 
            initial_sets=self.__f_groups ) 

        
        ML_SessionState = mlu.ML_SessionState
        
        init_session_data = {
            ML_SessionState.INIT_SESSION: case_study_id,
            ML_SessionState.EVAL_MANAGER: MongoFEManager( case_study_id=case_study_id ),
            ML_SessionState.FEATURE_GROUPS: self.__f_groups, 

            ML_SessionState.DISCOVERY_SET: datasets[0],     ##training purposes
            ML_SessionState.VALIDATION_SETS: datasets[1:],  ##test & evaluation purposes
            ML_SessionState.ALL_DATA: datasets,             ##convenience purposes :)
            # ML_SessionState.VALIDATION_DATA: datasets[2:], ## TODO
            ML_SessionState.SELECTED_TASK: task_info,
            ML_SessionState.FEATURE_GRAPH: self.__feature_graph, 
            ML_SessionState.FEATURES: fset_manager, 
            ML_SessionState.FEATURE_FOUND: fset_manager.df_featuresets, 
            ML_SessionState.WX_TEST: mlu.WilcoxonTest( datasets[0], task_info )
        }
        for s_entry, value in init_session_data.items():
            st_session[ s_entry.value ] = value

        self.__init_feature_lists( st_session ) 
        st_session[ "cov_mapping" ] = cov_encoding 


class MongoFEManager:
    """ Manager for MongoDB-based feature evaluation. """

    class MongoFlag(enum.Enum):
        FEATURE_SET = "feature_set"
        ENSEMBLE_SET = "ensemble_set"
        GA_GENERATION = "ga_generation"


    def __init__(self, case_study_id: str):
        client = mongoml.connect_to_mongo( db_name="mobius" )
        self.__mdb = client
        self.__cvperf = "evaluations"
        self.__case_study_id = case_study_id 


    @staticmethod
    def encode_algorithms( algos: List[mlu.LearningAlgorithm] ) -> str:
        return "$".join(sorted([ algo.value for algo in algos ]))

    def models_evaluation_wrapper( 
        self, 
        task_info: mlu.TaskParameters,
        input_data: List[mlu.LabelledData], 
        fset_collection: Dict[ str, List[str]], 
        algos_id: List[mlu.LearningAlgorithm], 
        cv_splits, 
        flag_db: str, 
        sampler: BaseSampler = None, 
    ) -> Tuple[ pd.DataFrame, Dict[str, List[mlu.ProbsRecord]] ]:
        
        
        def get_cv_params( cv_obj ) -> Tuple[bool, Dict[str, Any]]:
            kfold_mode = False
            cv_params = dict()

            match cv_obj:
                case LeaveOneOut():
                    kfold_mode = False 
                    cv_params = { "cv_mode": "loocv" }
                case _:
                    kfold_mode, nreps = True, cv_obj.n_repeats
                    cv_params = { 
                        "cv_mode": "kfold", 
                        "k": cv_obj.get_n_splits() / nreps, 
                        "n_repeats": nreps 
                    }

            return kfold_mode, cv_params


        kfold_mode, cv_params = get_cv_params( cv_splits )
        cv_params.update({ "mongo_flag": flag_db })
        algos_map = { algo_id: get_named_algorithm(algo_id)[1] for algo_id in algos_id }

        search_query = {
            "type": mlu.FeatureType.SIMPLE.value,
            "fset_id": { "$in": list(fset_collection.keys()) }, 
            "algo_id": { "$in": list( algos_map.values()) },
            **cv_params
        }

        json_docs = mongoml.find( self.__mdb, self.__cvperf, self.__case_study_id, search_query )
        exp_ndocs = len(fset_collection) * len(algos_id)

        logging.critical(f"MongoDB: Found {len(json_docs)} cached results for {search_query} (expected {exp_ndocs})")

        if len(json_docs) == 0:
            logging.critical(f"MongoDB: No cached results found for {search_query}. Running evaluation...")

            eval_results = self.__run_evaluation(
                task_info= task_info,
                cv_params=cv_params,
                input_data=input_data,
                fset_collection=fset_collection,
                algos_id=algos_id,
                cv_splits=cv_splits,
                sampler=sampler
            )
            perf_df, f_importances, tset_perf_dict, df_list_outfile = eval_results.unwrap_results()
            return perf_df, f_importances, tset_perf_dict, df_list_outfile

        elif len(json_docs) == exp_ndocs:

            eval_results = self.__retrieve_from_docs( input_data, json_docs )
            perf_df, f_importances, tset_perf_dict, df_list_outfile = eval_results.unwrap_results()
            return ( perf_df, f_importances, tset_perf_dict, df_list_outfile )


        elif len(json_docs) < exp_ndocs:

            absence_map = np.zeros( shape=( len(search_query["fset_id"]["$in"]), len(search_query["algo_id"]["$in"]) ), dtype=bool )
            f_map = { fset: i for i, fset in enumerate(search_query["fset_id"]["$in"]) }
            a_map = { algo: i for i, algo in enumerate(search_query["algo_id"]["$in"]) }


            for rec in json_docs:
                fset_id, algo_id = rec["fset_id"], rec["algo_id"]
                absence_map[ f_map[fset_id], a_map[algo_id] ] = True

            cols_as_tuples = [tuple(absence_map[:, j]) for j in range(absence_map.shape[1])]
            groups = defaultdict(list)
            for j, col in enumerate(cols_as_tuples):
                if not all(col):
                    curr_fsets = tuple([ fset for fset, c in zip( fset_collection, col ) if not c ])
                    # groups[curr_fsets].append( search_query["algo_id"]["$in"][j] )
                    groups[curr_fsets].append( algos_id[j] )


            partial_result = self.__retrieve_from_docs( input_data, json_docs )

            for curr_fsets, curr_algos in groups.items():
                logging.info(f"** Computing {curr_algos} for feature set {curr_fsets}.")
                tmp_results = self.__run_evaluation(
                    task_info=task_info,
                    cv_params=cv_params,
                    input_data=input_data,
                    fset_collection={ k: fset_collection[k] for k in curr_fsets },
                    algos_id=curr_algos,
                    cv_splits=cv_splits,
                    sampler=sampler
                )

                partial_result.merge_results( tmp_results )

            return partial_result.unwrap_results()


    def __run_evaluation(
        self, 
        task_info: mlu.TaskParameters,
        cv_params: Dict[str, Any],
        input_data: List[mlu.LabelledData], 
        fset_collection: Dict[str, List[str]], 
        algos_id: List[mlu.LearningAlgorithm], 
        cv_splits,
        sampler: BaseSampler = None )-> mlu.EvaluationResultRecord:
        # -> Tuple[pd.DataFrame, Dict[str, List[mlu.ProbsRecord]], Dict[str, Dict[str, pd.DataFrame]]]:
        """ Run the evaluation of models on the given data and feature sets. """

        kfold_mode = cv_params["cv_mode"] == "kfold"

        perf_df, probs_data, f_importances = FeatureEvaluationManager.models_evaluation_wrapper(
                input_data, fset_collection, algos_id, cv_splits, sampler)
        tset_perf_dict = get_tset_performances(
            task=task_info,
            probs_data=probs_data, 
            test_set_list=input_data[1:] )
        
        if kfold_mode:
            perf_df.insert(0, "sampling", st.session_state.imbalance_adjust )
        else:
            ## quick fix for current bug 
            perf_df.drop(columns=["loocv__algo_id"], inplace=True, errors="ignore")
        

        self.__persist_models_evaluation(
            eval_params=cv_params,
            performance_cv=perf_df, 
            performance_tset=tset_perf_dict, 
            feature_importances=f_importances ) 

        return mlu.EvaluationResultRecord(
            cv_performances=perf_df,
            tsets_performance=tset_perf_dict,
            feature_importances=f_importances
        )


    def __persist_models_evaluation(
        self, 
        eval_params: Dict[str, Any],
        performance_cv: pd.DataFrame,
        performance_tset: Dict[str, mlu.TestSetPerformance],
        feature_importances: Dict[str, Dict[str, pd.DataFrame]]):


        df_ptsets = None 

        if len( performance_tset) > 0:
            df_ptsets = pd.concat(
                [ ptset.raw_performances for ptset in performance_tset.values() ], 
                keys=performance_tset.keys()
            ).reset_index() ## TODO: drop level_1 column?


        json_docs = list() ## prepare json docs to insert into MongoDB
        for (algo_id, fset_id), subdf in performance_cv.groupby(["algo_id", "fset_id"]):
            probs_tset = { 
                tset_name: tset_data.get_probsrecord(fset_id, algo_id).to_dict()
                    for tset_name, tset_data in performance_tset.items() 
            }

            obj = {
                "fset_id": fset_id,
                "algo_id": algo_id,
                "type": mlu.FeatureType.SIMPLE.value,
                **eval_params,
                "perfs": subdf.reset_index().to_dict(orient="records"), 
                "probs": probs_tset, 
                # "tset_stuff": curr_tset_data.to_dict(orient="records") if not curr_tset_data.empty else []
            }

            if df_ptsets is not None:
                curr_tset_data = df_ptsets[ 
                    (df_ptsets.fset_id == fset_id) & 
                    (df_ptsets.algo_id == algo_id) ]
                obj["tset_stuff"] = curr_tset_data.to_dict(orient="records") #if not curr_tset_data.empty else []

            if (f_imp := feature_importances.get(fset_id, {}).get(algo_id, None)) is not None:
                obj["f_importances"] = f_imp.reset_index().to_dict(orient="records")
            
            json_docs.append( obj )

        mongoml.insert_many( self.__mdb, self.__cvperf, self.__case_study_id, json_docs )


    def __retrieve_from_docs(
        self, 
        input_data: List[mlu.LabelledData],
        json_docs: List[Dict[str, Any]]) -> mlu.EvaluationResultRecord:
    

        performance_data = list() 
        feature_importances = defaultdict( dict ) ## { fset_id: { algo_id: pd.DataFrame } }
        probs_record = defaultdict( list )  ##keys: tset_id, List[probs_record]
        tset_id_list = [ tset.name for tset in input_data[1:] ] ## test & validation 

        tset_presence_flag = len(tset_id_list) > 0
        tset_data = list() 

        ## retrieve from json_docs
        for rec in json_docs:
            fset, algo = rec["fset_id"], rec["algo_id"]
            performance_data.extend( rec["perfs"] )

            if tset_presence_flag:
                ## get the predicted probabilities for each test set
                pr = rec["probs"]

                for tset_id, probs in pr.items():
                    probs_record[tset_id].append( mlu.ProbsRecord.decode_probs(fset, algo, probs) )

                ## get the performance records for each test set
                tset_data.extend( rec["tset_stuff"] )
            
            if (f_imp := rec.get("f_importances", None)) is not None:
                feature_importances[fset][algo] = pd.DataFrame.from_records( f_imp ).set_index("index")


        ## postprocessing
        perf_df = pd.DataFrame( performance_data ).set_index(["algo_id", "fset_id"])
        tset_data = pd.DataFrame.from_records( tset_data ) if tset_presence_flag else pd.DataFrame()

        perf_tsets = dict() ## { tset_name: mlu.TestSetPerformance }

        for tset_id in tset_id_list:
            curr_tset_data = tset_data[ tset_data.level_0 == tset_id ].drop(columns=["level_0", "level_1"])#.set_index(["algo_id", "fset_id"])
            curr_probs_records = { (pr.fset_id, pr.algo_id): pr for pr in probs_record[ tset_id ] }
            
            perf_tsets[tset_id] = mlu.TestSetPerformance(
                test_set_id=tset_id, 
                probs_record=curr_probs_records, 
                raw_performances=curr_tset_data )
            

        return mlu.EvaluationResultRecord(
            cv_performances=perf_df, 
            tsets_performance=perf_tsets, 
            feature_importances=feature_importances
        )

        #     df_list_outfile


    def ensemble_evaluation( 
        self, 
        input_data: List[mlu.LabelledData], 
        ensemble_features: mlu.EnsembleFeatureSet, 
        list_algos_id: List[ mlu.LearningAlgorithm ], 
        k_folds: int = 5, n_repeats: int = 3 ) -> pd.DataFrame:

        algos = self.encode_algorithms( list_algos_id )

        ### lookup query for current ensemble featureset, algorithms, and parameters 
        search_query = {
            "fset_id": ensemble_features.name,
            "type": mlu.FeatureType.ENSEMBLE_FEATURESET.value,
            "kcv": k_folds,
            "n_repeats": n_repeats,
            "algos": algos
        }

        if ( json_doc := mongoml.find_one( self.__mdb, self.__cvperf, self.__case_study_id, search_query ) ) is not None:
            perfs_df = pd.DataFrame( json_doc["perfs"] ).set_index(["algo_id", "fset_id"])
        else:
            perfs_df, _ = FeatureEvaluationManager.ensemble_evaluation(
                input_data, ensemble_features, list_algos_id,
                k_folds, n_repeats )
            
            json_doc = {
                **search_query,
                "perfs": [ r.to_dict() for _, r in perfs_df.reset_index().iterrows() ]
            }
            mongoml.insert_one( self.__mdb, self.__cvperf, self.__case_study_id, json_doc )

        return perfs_df



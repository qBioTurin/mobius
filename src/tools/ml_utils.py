from dataclasses import dataclass, InitVar, field
from sklearn.linear_model import LogisticRegression, LassoCV, ElasticNetCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer

from imblearn.base import BaseSampler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.combine import SMOTEENN, SMOTETomek

from collections import Counter
from itertools import chain
from graph_tool.all import Graph 

from tools.gtools import gt, OmicsGraphFilter
from tools.enums import GraphField
import plotly.express as px 
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
import logging


import fastcluster
import matplotlib.pyplot as plt 
from scipy.cluster import hierarchy
import scipy.stats as stats
import tempfile

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd, numpy as np 
import enum
import json, hashlib

import streamlit as st


def ml_guard(session_state):
    if not session_state.get( ML_SessionState.INIT_SESSION.value ):
        st.error("Please, go back to the main page and load the data before proceeding!")
        if st.button("Home"):
            st.switch_page("ML/load_ml.py")
        st.stop()


def algo_fset_grouper(key: str) -> str:
    ### nb. Grouper is a kind of fish (cernia) but also means "raggruppatore"
    options = "algorithm", "feature set"
    return st.radio("Group results w.r.t.", options=options, horizontal = True, key = key)


def update_feature_dataframe():
    fset_manager = st.session_state[ ML_SessionState.FEATURES.value ]
    st.session_state[ ML_SessionState.FEATURE_FOUND.value ] = fset_manager.df_featuresets

def get_case_study_id( case_study_data: Tuple[ Any ] ) -> str:
    return hashlib.md5( json.dumps(case_study_data ).encode("utf-8") ).hexdigest()


class CovariateTypeInferrer:

    class CovariateType(enum.Enum):
        BOOLEAN = "b"       # a binary variable 
        CATEGORICAL = "c"   # a categorical variable 
        ORDINAL = "o"       # a finite variable where the order matters 
        QUANTITATIVE = "q"  # a continuous variable 
        UNKNOWN = "u"       # idk 

    def __init__(self, data: pd.DataFrame):
        self.__data = data 


    def get_categoricals(self) -> Dict[str, CovariateType]:
        return {
            key: value
                for key, value in self.get_covariate_types().items()
                    if value == self.CovariateType.CATEGORICAL or value == self.CovariateType.BOOLEAN
                    # if value in ( self.CovariateType.CATEGORICAL, self.CovariateType.BOOLEAN )
        }
    
    def get_quantitative(self) -> Dict[str, CovariateType]:
        return {
            key: value
                for key, value in self.get_covariate_types().items()
                    if value == self.CovariateType.QUANTITATIVE
        }


    def get_covariate_types(self) -> Dict[str, CovariateType]:
        return {
            cov: self.__type_inference( self.__data[cov] )
                for cov in self.__data.columns.tolist()
        }

    def __type_inference(self, feature_values: pd.Series) -> CovariateType:
        assert feature_values is not None 
        starting_type = self.CovariateType.UNKNOWN

        try:
            feature_values.mean()
            starting_type = self.__categorical_check(self.CovariateType.QUANTITATIVE, feature_values)
        except (TypeError, ValueError):
            try:
                feature_values.median()
                starting_type = self.__categorical_check(self.CovariateType.ORDINAL, feature_values)
            except (TypeError, ValueError):
                starting_type = self.__categorical_check(self.CovariateType.CATEGORICAL, feature_values)      

        return starting_type
        
 
    def __categorical_check( self, starting_type: CovariateType, values: pd.Series ) -> CovariateType:
        elems = values.dropna().unique()

        if starting_type is self.CovariateType.QUANTITATIVE:
            if len(elems) < 13: #a random magic number 
                #check for a small number of integer values 
                if all( x == np.ceil(x) for x in elems ):   
                    starting_type =  self.CovariateType.CATEGORICAL #self.FeatureType.CATEGORICAL

        if starting_type is self.CovariateType.CATEGORICAL and len(elems) == 2:
            starting_type = self.CovariateType.BOOLEAN
                
        return starting_type


class MyOneHotEncoder:
    """ OneHotEncoder for categorical features:
     - encode boolean features as a single column
     - encode categorical features with N values as N boolean columns"""
    
    PLACEHOLDER_NAN = "NaN"


    def __init__(self, df: pd.DataFrame, cov_to_values = None):
        if cov_to_values is None:
            df_nans = df.fillna( self.PLACEHOLDER_NAN )
            cov_to_values = {
                col: sorted( df_nans[col].unique().tolist() )
                    for col in df.columns.tolist()
            }

        binary_cols = { cov for cov, values in cov_to_values.items() if len(values) == 2 }
        non_binary_cols = set(cov_to_values.keys()) - binary_cols

        boolean_mappings = { cov: self.__encode_boolean_covariate(cov, values) for cov, values in cov_to_values.items() if cov in binary_cols }
        categorical_mappings = { cov: self.__encode_categorical_covariate(cov, values) for cov, values in cov_to_values.items() if cov in non_binary_cols }
        self.__categorical_mappings = { **boolean_mappings, **categorical_mappings }


    @property
    def categorical_mappings(self) -> Dict[str, List[str]]:
        return {
            cov: sorted( enc.keys() ) 
                for cov, enc in self.__categorical_mappings.items()
        }
        

    def __encode_boolean_covariate(self, cov: str, values_pair):
        """ Build a dictionary of one function to encode a boolean covariate s.t. v1 < v2 (v1 = 0, v2 = 1) """
        
        assert len( values_pair) == 2
        return { f"{cov}_b": { values_pair[0]: 0, values_pair[1]: 1 } }

    def __encode_categorical_covariate(self, cov: str, values):
        """ Build a dictionary of functions to encode a categorical covariate with N values using dummy variables """

        assert len(values) > 2
        encoders = {
            f"{cov}_{value}": {x_val: int(x_val == value) for x_val in values } 
                for value in values 
        }

        return encoders

    def transform(self, df: pd.DataFrame):
        new_cols = dict() 

        for col in df.columns.tolist():
            if col in self.__categorical_mappings:
                curr_col = df[col].fillna( self.PLACEHOLDER_NAN )

                for new_col, encoding_func in self.__categorical_mappings[col].items():
                    try:
                        new_cols[ new_col ] = curr_col.apply( lambda x: encoding_func.get(x, 0))
                    except TypeError as e:
                        logging.error(f"Error while encoding {col} -> {new_col} using {encoding_func}")
        
        return pd.DataFrame(new_cols, index = df.index.tolist()) 


@dataclass
class LabelledData:
    features: pd.DataFrame
    target: pd.Series
    metadata: pd.DataFrame
    cov_encoding: Dict[str, List[str]] = None 
    name: str = None #= "placeholder at the moment"

    def select_features( self, feature_set ) -> "LabelledData":
        try:
            return LabelledData( self.features[ feature_set ].copy(), self.target, self.metadata, self.cov_encoding, self.name )
        except KeyError:
            missing_features = set(feature_set).difference( self.features.columns.tolist() )
            logging.error(f"Error while selecting features - missing: {missing_features}")
            available_features = list( set(feature_set).intersection( self.features.columns.tolist() ) )
            return LabelledData( self.features[ available_features ].copy(), self.target, self.metadata, self.cov_encoding, self.name )

    def get_full_matrix(self) -> pd.DataFrame:
        return pd.concat([ self.metadata, self.features ], axis = 1)
        

    def scale_data(self, fitted_scaler: StandardScaler = None) -> "LabelledData":
        scaler = StandardScaler().fit( self.features ) if fitted_scaler is None else fitted_scaler
        scaled_features = pd.DataFrame(
            data = scaler.transform( self.features ),
            columns = self.features.columns.tolist(),
            index = self.features.index.tolist() )
        return LabelledData( scaled_features, self.target, self.metadata, self.cov_encoding, self.name )

    def drop_samples(self, sample_ids: List[str]) -> "LabelledData":
        """ Drop samples with the given sample ids from the dataset """
        if not sample_ids:
            return self
        
        new_features = self.features.drop( index = sample_ids, errors = "ignore" )
        new_target = self.target.drop( index = sample_ids, errors = "ignore" )
        new_metadata = self.metadata.drop( index = sample_ids, errors = "ignore" )

        return LabelledData( new_features, new_target, new_metadata, self.cov_encoding, self.name )

    def get_class_distribution(self):
        return self.target.value_counts()


@dataclass
class TaskParameters:
    target_cov: str 
    pos_class: Tuple[str]
    neg_class: Tuple[str]

    def __post_init__(self):
        self.pos_class = (self.pos_class,) if isinstance(self.pos_class, str) else tuple( self.pos_class )
        self.neg_class = (self.neg_class,) if isinstance(self.neg_class, str) else tuple( self.neg_class )

    def get_class(self, which_one: bool) -> str:
        the_class = self.pos_class if which_one else self.neg_class
        return "+".join( sorted(the_class) ) 

    def get_triple(self) -> Tuple[str, Tuple[str], Tuple[str]]:
        return self.target_cov, self.pos_class, self.neg_class
    

    def get_unique_study_identifier(self, feature_groups: Dict[str, List[str]]) -> str:
        whole_featureset = tuple(
            (omic_id, ",".join(sorted( feature_groups[omic_id] )))
                for omic_id in sorted( feature_groups.keys() ) 
        ) 
        return get_case_study_id( ( self.get_triple(), whole_featureset ) )


class DataPreparator:
    def __init__( self, 
            dataset_list: List[Tuple[str, pd.DataFrame]], #List[pd.DataFrame],
            task: TaskParameters, 
            covariates: List[str], 
            features: List[str] ):
        
        self.__dataset_names, self.__datasets = zip( *dataset_list )
        self.__task = task
        self.__features = features
        self.__covariates = covariates
        self.__target_values = self.__datasets[0][ task.target_cov ]

        tr_data = self.__datasets[0]

        self.__inferrer = CovariateTypeInferrer( tr_data[ covariates ] )
        self.__cat_list = self.__inferrer.get_categoricals()
        self.__quant_list = self.__inferrer.get_quantitative() 
        self.__q_mapping = { col: f"{col}_q" for col in self.__quant_list.keys() }
        self.__scaler = StandardScaler()
        self.__imputer = None #SimpleImputer(strategy="median")

        self.__cov_encoding = { cov: [ q_cov ] for cov, q_cov in self.__q_mapping.items() }
        self.__encoder = None

        if self.__cat_list:
            self.__encoder = MyOneHotEncoder( tr_data[ list( self.__cat_list.keys() ) ] ) 
            self.__cov_encoding.update( self.__encoder.categorical_mappings )


    @property
    def scaler(self) -> StandardScaler:
        return self.__scaler
    
    @property
    def covariate_encoding(self) -> Dict[str, List[str]]:
        return self.__cov_encoding


    def __encode_covariates(self, index: int) -> Optional[ pd.DataFrame ]:
        if self.__encoder is not None:
            df = self.__datasets[ index ]
            cat_list = list( self.__cat_list.keys() )
            return self.__encoder.transform( df[ cat_list ] )
        return None 
    
    def __encode_quantitatives(self, index: int, data_stats: pd.DataFrame ) -> Optional[ pd.DataFrame ]:
        if self.__quant_list:
            quant_list = list( self.__quant_list.keys() )
            q_df = self.__datasets[ index ][ quant_list ].rename( columns = self.__q_mapping )
            return self.__fillna( index, q_df, data_stats )
            
        return None 


    def __get_means(self) -> pd.DataFrame:
        q_covs = self.__quant_list.keys() 
        q_data = self.__datasets[0][ q_covs ].copy()
        q_stats = pd.DataFrame( index = q_data.columns.tolist() )
        q_data["target"] = self.__target_values #== self.__task.pos_class

        q_stats = q_data.groupby(by="target").median()
        q_stats.columns = [ f"{col}_q" for col in q_stats.columns.tolist() ]

        return q_stats

    def __fillna(self, index: int, df: pd.DataFrame, df_stats: pd.DataFrame ) -> pd.DataFrame:

        tc, pc, nc = self.__task.target_cov, self.__task.pos_class, self.__task.neg_class
        my_cols = set( df_stats.columns.tolist() )
        df[ tc ] = self.__datasets[index][ tc ]  #== pc

        label_list = pc + nc 

        if ( columns_with_nan := df.columns[df.isna().any()].tolist() ):

            for col in my_cols.intersection( columns_with_nan ):
                for label in label_list:
                    df.loc[ (df[tc] == label) & (df[col].isna() ), col ] = df_stats.loc[ label, col ]
                # df.loc[( df[tc] == pc ) & ( df[col].isna() ), col ] = vp
                # df.loc[( df[tc] == nc ) & ( df[col].isna() ), col ] = vn

        return df.drop(columns=[tc])

    
    def __preprocessing(self, index: int, cov_encoding: Dict[str, List[str]], data_stats: pd.DataFrame ) -> LabelledData: 
        
        cat_covs = self.__encode_covariates( index )
        qua_covs = self.__encode_quantitatives( index, data_stats )
        omics_features = self.__datasets[ index ][ self.__features ]
        df_series = [ cat_covs, qua_covs, omics_features ]

        full_df = pd.concat( [ df for df in df_series if df is not None ], axis = 1 ) 

        if False:
            if index == 0:
                self.__scaler.fit( full_df )

            scaled_features = pd.DataFrame(
                data = self.__scaler.transform( full_df ), 
                columns = full_df.columns.tolist(),
                index = full_df.index.tolist() )
        
        if self.__imputer is None:
            self.__imputer = SimpleImputer(strategy="median")
            self.__imputer.fit( full_df )

        scaled_features = pd.DataFrame(
            self.__imputer.transform( full_df ),
            columns = full_df.columns.tolist(),
            index = full_df.index.tolist()
        )

        raw_data = self.__datasets[ index ]

        return LabelledData(
            features = scaled_features, 
            target = raw_data[ self.__task.target_cov ].isin( self.__task.pos_class ), #== self.__task.pos_class, 
            metadata = raw_data[ self.__covariates ].copy(),
            cov_encoding = cov_encoding 

        )

    def preprocessing(self) -> List[ LabelledData ]:
        data_stats = self.__get_means()
        datasets = list() 

        for i in range( len( self.__datasets ) ):
            if self.__datasets[i] is not None:
                d = self.__preprocessing( i, self.__cov_encoding, data_stats ) 
                d.name = self.__dataset_names[i]
                datasets.append( d )
        #         for i in range( len( self.__datasets ) ) 

        return datasets


class FeatureType(enum.Enum):
    SIMPLE = enum.auto()
    ENSEMBLE_FEATURESET = enum.auto()
    GA_GENERATION = enum.auto()
    

@dataclass
class FeatureSubset:
    feature_list: List[str]
    metadata: str 
    id_count: InitVar[ int ]
    feature_graph: InitVar[ gt.Graph ]
    feature_groups: InitVar[ Dict[ str, List[str] ]]
    uid: Optional[str] = None 
    found_count: int = 1 
    composition: str = field(init=False)
    n_features: int = field(init=False)
    n_edges: int = field(init=False)
    n_edges_05: int = field(init=False)
    n_edges_01: int = field(init=False)
    n_edges_001: int = field(init=False)
    
    
    def __post_init__(self, 
                      id_count: int, 
                      feature_graph: gt.Graph, 
                      feature_groups: Dict[str, List[str]] ):
        
        subgraph = OmicsGraphFilter.get_subgraph( feature_graph, self.feature_list, get_graphview=True )
        self.n_features = subgraph.num_vertices()
        self.n_edges = subgraph.num_edges() 

        p_thresholds = (0.05, 0.01, 0.001)
        self.n_edges_05, self.n_edges_01, self.n_edges_001 = [
            int( np.sum( subgraph.ep["WholeData"].fa < p ) ) for p in p_thresholds
        ]

        self.omic_composition = [
            ( fset_id, len( set(feature_groups[fset_id] ).intersection( self.feature_list )) )
                for fset_id in sorted( feature_groups.keys() )
        ]
        self.composition = "_".join( [
            f"{n}-{fset_id}" for fset_id, n in self.omic_composition if n > 0
        ]) 

        ## get set composition 
        #     f"{n}-{fset_id}" 
        #         for fset_id in sorted( feature_groups.keys() )

        if self.uid is None:
            metadata_str = f"_{self.metadata}" if self.metadata else ""
            self.uid = f"#{id_count}__{self.composition}" 


    def __len__(self):
        return len( self.feature_list )

    def increment_counter(self):
        self.found_count += 1 

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            uid = self.uid, 
            n_features = self.n_features, 
            n_edges = self.n_edges, 
            n_edges_05 = self.n_edges_05, 
            n_edges_01 = self.n_edges_01, 
            n_edges_001 = self.n_edges_001, 
            composition = self.composition, 
            metadata = str(self.metadata),
            flist = self.feature_list
        )
    
    @classmethod
    def get_name_index_column(cls) -> str:
        return "uid"


    @staticmethod
    def encode_fset( fset: List[ str ]):
        return "$".join( sorted( fset ) )
    
    @staticmethod
    def decode_fset( fset_str: str ) -> List[str]:
        return fset_str.split("$") 
    
    def wrap_for_mongo(self) -> Dict[str, Any]:
        d = self.to_dict()
        d["flist"] = self.encode_fset( self.feature_list )
        d["ftype"] = FeatureType.SIMPLE.value
        return d 
    

@dataclass
class EnsembleFeatureSet:
    name: str
    feature_range: tuple  #pair (nf_min, nf_max)
    # components: list    # list of lists: [ fl_{nf_min}, fl_{nf_min+1}, ..., fl_{nf_max} ]
    components: Dict[int, FeatureSubset]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame( data = [
            c.to_dict() for c in self.components.values()
        ]).set_index( FeatureSubset.get_name_index_column() )
    
    def wrap_for_mongo(self) -> Dict[str, Any]:
        feature_sequence = [ c.wrap_for_mongo() for c in self.components.values() ]

        return {
            "uid": self.name,
            "ftype": FeatureType.ENSEMBLE_FEATURESET.value,
            "lower_bound": self.feature_range[0],
            "upper_bound": self.feature_range[1],
            "components": feature_sequence
        }


@dataclass
class GAGeneration:
    name: str 
    feature_pool: list 
    solutions: list

    def add_solution( self, fset: List[str] ):
        self.solutions.append( fset )

    def wrap_for_mongo(self) -> Dict[str, Any]:
        return {
            "uid": self.name,
            "ftype": FeatureType.GA_GENERATION.value,
            "feature_pool": FeatureSubset.encode_fset( self.feature_pool ),
            "solutions": [ FeatureSubset.encode_fset( fset ) for fset in self.solutions ]
        }


@dataclass
class ProbsRecord:
    fset_id: str 
    algo_id: str
    y_probs: np.array
    raw_probs: np.array 


    def to_dict(self) -> Dict[str, Any]:
        return {
            "y_probs": self.y_probs.tolist(), ##
            "raw_probs": self.raw_probs.tolist(), 
        }
    
    @staticmethod
    def decode_probs( fset_id: str, algo_id: str, probs_data: Dict[str, List[float]]) -> "ProbsRecord":
        return ProbsRecord(
            fset_id=fset_id,
            algo_id=algo_id,
            y_probs=np.array(probs_data["y_probs"]),
            raw_probs=np.array(probs_data["raw_probs"])
        )

@dataclass
class TestSetPerformance:
    test_set_id: str 
    probs_record: Dict[Tuple[str, str], ProbsRecord]  ## ex List[ProbsRecord]
    raw_performances: pd.DataFrame
    ci_performances: pd.DataFrame = field(init=False)


    def __post_init__(self):
        self.ci_performances = ConfidenceIntervalManager.compute_intervals( 
            self.raw_performances, ["fset_id", "algo_id"] )


    def get_named_df(self, field: str) -> pd.DataFrame:
        match field:
            case "ci":
                target_df = self.ci_performances.copy() 
            case "raw":
                target_df = self.raw_performances.copy()
            case _:
                raise ValueError(f"Unknown dataframe type: {field}")

        target_df.insert(0, "test_set_id", self.test_set_id)
        return target_df
    
    def get_outfiles(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ Get the dataframes to be saved as output files """
        return (self.get_named_df("ci"), self.get_named_df("raw"))
    
    def get_probsrecord(self, fset_id: str, algo_id: str) -> Optional[ProbsRecord]:
        """ Get the probabilities record for the given feature set and algorithm IDs """
        return self.probs_record.get( (fset_id, algo_id), None )

    def merge_results(self, other: "TestSetPerformance" ):
        if other.test_set_id == self.test_set_id:
            self.probs_record.update(other.probs_record)
            self.raw_performances = pd.concat([self.raw_performances, other.raw_performances])
            self.ci_performances = pd.concat([self.ci_performances, other.ci_performances])


@dataclass
class EvaluationResultRecord:
    cv_performances: pd.DataFrame
    tsets_performance: Dict[str, TestSetPerformance]
    feature_importances: Dict[Tuple[str, str], pd.DataFrame] 
    df_outfiles: List[Tuple[pd.DataFrame, pd.DataFrame]] = field(init=False)

    def __post_init__(self):
        self.df_outfiles = [
            tset_perf.get_outfiles() 
                for tset_perf in self.tsets_performance.values()
        ]

    def unwrap_results(self) -> Tuple[
            pd.DataFrame, 
            Dict[Tuple[str, str], pd.DataFrame], 
            Dict[str, TestSetPerformance], 
            List[Tuple[pd.DataFrame, pd.DataFrame]]]:
        """ Unwrap the results into a tuple of (cv_performances, tsets_performance, feature_importances) """
        return self.cv_performances, self.feature_importances, self.tsets_performance, self.df_outfiles


    def merge_results(self, other: "EvaluationResultRecord"):
        """ Merge the results of another EvaluationResultRecord into this one """

        self.cv_performances = pd.concat([self.cv_performances, other.cv_performances])
        self.feature_importances.update(other.feature_importances)

        for tset_id, tset_perf in other.tsets_performance.items():
            if tset_id in self.tsets_performance:
                self.tsets_performance[tset_id].merge_results( tset_perf )
            else:
                logging.critical(f"Strange!!! This should not happen! Test set {tset_id} not found in the current record!")
                self.tsets_performance[tset_id] = tset_perf

        self.df_outfiles = [ t_performance.get_outfiles() for t_performance in self.tsets_performance.values() ]


class ML_SessionState( enum.Enum ):
    FEATURE_GROUPS = "f_groups"
    FEATURE_FOUND = "my_fsets"
    FEATURE_GRAPH = "ml_graph"
    FEATURES = "fmanager"


    ALL_DATA = "all_data"
    DISCOVERY_SET = "discovery_set"
    VALIDATION_SETS = "validation_sets"


    SELECTED_TASK = "curr_task"

    INIT_SESSION = "ml_session"

    PERFORMANCE_RECORD = "perfo"
    PERFORMANCE_RECORD_ENSEMBLES = "ens_perfo"
    SELECTED_FEATURES = "my_features"

    EVAL_MANAGER = "m_eval"
    FSEL_MANAGER = "m_fsel"

    WX_TEST = "wilcoxon_test"


class LearningAlgorithm(enum.Enum):
    LOGISTIC_REGRESSION = "log_reg"
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "r_forest"
    XGBOOST = "xgboost"
    SVM = "svm"
    LINEAR_SVM = "linear_svm"
    NAIVE_BAYES = "n_bayes"
    KNN = "knn"
    LINEAR_DA = "lda"
    BASELINE = "baseline"

    @classmethod
    def ensure_enum(cls, a_id: "LearningAlgorithm"):
        match a_id:
            case str():
                return cls(a_id)
            case enum.Enum():
                return cls(a_id.value)

        raise RuntimeError(f"Unknown {cls}: {a_id}")

    @classmethod
    def get_algorithm( cls, a_id: "LearningAlgorithm", **kwargs ):
        a = None 

        
        a_id = cls.ensure_enum(a_id)
        

        match a_id:
            case cls.LOGISTIC_REGRESSION:
                a = LogisticRegression(**kwargs)
            case cls.DECISION_TREE:
                a = DecisionTreeClassifier(**kwargs)
            case cls.RANDOM_FOREST:
                a = RandomForestClassifier(**kwargs)
            case cls.XGBOOST:
                a = XGBClassifier(**kwargs)
            case cls.SVM:
                a = SVC(probability = True, **kwargs)
            case cls.LINEAR_SVM:
                a = CalibratedClassifierCV(LinearSVC(**kwargs))
            case cls.NAIVE_BAYES:
                a = GaussianNB(**kwargs)
            case cls.KNN:
                a = KNeighborsClassifier(**kwargs)
            case cls.LINEAR_DA:
                a = LinearDiscriminantAnalysis(**kwargs)
            case cls.BASELINE:
                a = DummyClassifier(strategy="stratified", **kwargs)
            case _:
                raise ValueError(f"Unknown algorithm: {a_id} (type: {type(a_id)})")

        return a


class EmbeddedFeatureSelection(enum.Enum):
    LOGISTIC_REGRESSION_L1 = "LASSO"
    LOGISTIC_REGRESSION_ENET = "ElasticNet"
    LOGISTIC_REGRESSION = "LogisticRegression"
    DECISION_TREE = "DecisionTree"
    LINEAR_SVM = "LinearSVM"

    @classmethod
    def get_algorithm( cls, a_id: "EmbeddedFeatureSelection", **kwargs ) :
        a = None 

        match a_id:
            case cls.LOGISTIC_REGRESSION_L1:
                a = LogisticRegression
            case cls.LOGISTIC_REGRESSION_ENET:
                a = LogisticRegression
            case cls.LOGISTIC_REGRESSION:
                a = LogisticRegression
            case cls.DECISION_TREE:
                a = DecisionTreeClassifier
            case cls.LINEAR_SVM:
                a = LinearSVC
            case _:
                raise ValueError(f"Unknown algorithm: {a_id}")
            
        return a(**kwargs)
    
    @classmethod
    def get_hyperparameters(cls, a_id: "EmbeddedFeatureSelection") -> Dict[str, Any]:
        params = dict()

        match a_id:
            case cls.LOGISTIC_REGRESSION_L1:
                params["penalty"] = "l1"
                params["solver"] = "liblinear"
            case cls.LOGISTIC_REGRESSION_ENET:
                params["penalty"] = "elasticnet"
                params["solver"] = "saga"
                params["l1_ratio"] = 0.5
            case cls.LOGISTIC_REGRESSION:
                params["penalty"] = "l2"
                params["solver"] = "lbfgs"
            case cls.DECISION_TREE:
                params["criterion"] = "entropy"
                params["max_depth"] = 5
            case cls.LINEAR_SVM:
                pass 
            case _:
                raise ValueError(f"Unknown algorithm: {a_id}")
            
        return params


class ImbalancedTechnique(enum.Enum):
    SMOTE = "smote"
    ADASYN = "adasyn"
    RANDOM_UNDER_SAMPLER = "random_under"
    NEAR_MISS = "near_miss"
    SMOTE_TOMEK = "smote_tomek"
    SMOTE_ENN = "smote_enn"


def get_imbalanced_sampler( sampler_id: str, **kwargs ) -> Optional[ BaseSampler ]:
    try:
        sampler_id = ImbalancedTechnique( sampler_id )
        match sampler_id:
            case ImbalancedTechnique.SMOTE:
                sampler = SMOTE( **kwargs )
            case ImbalancedTechnique.ADASYN:
                sampler = ADASYN( **kwargs )
            case ImbalancedTechnique.RANDOM_UNDER_SAMPLER:
                sampler = RandomUnderSampler( **kwargs )
            case ImbalancedTechnique.NEAR_MISS:
                sampler = NearMiss( **kwargs )
            case ImbalancedTechnique.SMOTE_TOMEK:
                sampler = SMOTETomek( **kwargs )
            case ImbalancedTechnique.SMOTE_ENN:
                sampler = SMOTEENN( **kwargs )

    except ValueError:
        sampler = None 

    return sampler


def init_session( session, field: ML_SessionState, default_value ):
    if field.value not in session:
        session[ field.value ] = default_value


def prepare_features( initial_feature_sets, session_state: Dict ):
    df_fset = session_state[ ML_SessionState.FEATURE_FOUND.value ] 
    chained_fset = chain.from_iterable( [ df_fset.loc[ fset ].flist for fset in initial_feature_sets ] )
    return list( set( chained_fset ) )


def histogram_feature_counts(
        feature_counts: Counter, 
        feature_graph: gt.Graph, 
        comm_id: pd.Series,
        color_map: Dict[str,str] = None, 
        min_freq: float = 0.1, 
        return_df: bool = False ):
    
    my_features = list( feature_counts.keys() )
    feature_omics = {
        name: omic for name, omic in zip( feature_graph.vp[ GraphField.FEATURE_NAME.value], feature_graph.vp[ GraphField.FEATURE_OMIC.value ])
            if name in my_features
    }


    if comm_id is None:
        comm_id = { feature: 1 for feature in my_features }
    df = pd.DataFrame( 
        data = [
            ( feature, count, feature_omics[ feature ], comm_id[ feature ])
                for feature, count in feature_counts.items()], 
        columns=["feature", "count", "omic", "id_comm"])#.sort_values(by="count")
    df.insert(2, "frequency", df["count"] / df["count"].max() )
    

    if False:
        barplot = px.bar(
            df,
            x = "feature", y = "count", color = "omic", 
            color_discrete_map = color_map,
            orientation="v")
        
        ## put horizontal line corresponding to min_occ 
        barplot.add_shape(type="line", x0=-0.5, x1=len(df)-0.5, y0=min_occ, y1=min_occ,
                        line=dict(color="red", width=2))

    n_comms = df.id_comm.nunique()
    df.sort_values(by=["id_comm", "count"], ascending=[True, False], inplace=True)
    bp_per_row = ncols = min(5, n_comms)
    nrows = (n_comms // bp_per_row) + 1
    

    fig = make_subplots(
        rows=nrows, cols=ncols,
        shared_yaxes=True, 
        subplot_titles=[f"id_comm={x}" for x in df.id_comm.unique()]
    )
    flag_omic_legend = set()

    for i, (c_id, subdf) in enumerate( df.groupby(by="id_comm")):
        row_index = 1 + (i // bp_per_row)
        col_index = 1 + (i % bp_per_row)
        for omic in subdf.omic.unique():
            omic_df = subdf[ subdf.omic == omic ]
            put_in_legend = omic not in flag_omic_legend
            fig.add_trace(
                go.Bar( 
                    x=omic_df["feature"], 
                    y = omic_df["frequency"], 
                    name=omic, legendgroup=omic, marker_color=color_map[omic],
                    showlegend=put_in_legend
                ), row=row_index, col=col_index
            )
            flag_omic_legend.add( omic )
        fig.add_hline(y=min_freq, line=dict(color="red", width=2),
                      row=row_index, col=col_index)

    fig.update_layout(
        height=400 * nrows,
        width=300 * n_comms,
        showlegend=True,
        barmode="group",
        title_text="Feature frequency by community",
        xaxis_title="Feature",
        yaxis_title="Frequency", 
    )
    fig.update_xaxes(tickangle=45)


    return_value = (fig, df) if return_df else fig 
    return return_value


def hierarchical_clustering( pairwise_dist_m: np.array ):
    def save_fig_to_temp( fig: plt.Figure ) -> str:
        temp_name = tempfile.NamedTemporaryFile(mode="wb", prefix="HierClust_", suffix=".png", delete=False)
        fig.savefig( temp_name.name, bbox_inches = "tight" )
        plt.close( fig )
        return temp_name.name

    fig_size = (20, 8)

    clusters_ravasz = fastcluster.linkage( pairwise_dist_m, method = "average" )
    fig_ravasz = plt.figure( figsize=fig_size )
    hierarchy.dendrogram( Z = clusters_ravasz, ax = fig_ravasz.add_subplot(1,1,1) )

    columns = dict() 
    for t_value in list( clusters_ravasz[:,2] )[::-1][:10]:
        prediction = hierarchy.fcluster( clusters_ravasz, t_value, criterion = "distance" )
        nclusters = len( set( prediction ) )
        if nclusters > 1:
            columns[ f"{nclusters}" ] = prediction - 1 ##forcing cluster id to start from zero

    ravasz_filename = save_fig_to_temp( fig_ravasz )
    df_hc = pd.DataFrame( columns )
    return df_hc, (ravasz_filename, )
    #     [ ravasz_spy_filename, ravasz_filename ] )


class ConfidenceIntervalManager:

    @classmethod
    def compute_intervals( cls, data: pd.DataFrame, groupby_features: List[str], method: str = "b" ):
        func = cls.compute_ci__boostrapping if method == "b" else cls.compute_ci__t_student
        ci_data = list()
        
        
        metrics_cols = data.columns.tolist()[3:]
        future_cols = chain.from_iterable([ ( f"{m_name}_mean", f"{m_name}__CI_95%" ) for m_name in metrics_cols ])
        future_cols = [ *groupby_features ] + list( future_cols )

        for group_id, subdf in data.groupby( groupby_features ):
            
            ci_data.append( list( group_id ) )
            current_row = ci_data[-1]

            for col in metrics_cols: #subdf.columns.tolist()[3:]:
                mean, cil, ciu = func( subdf[col].to_numpy() )
                current_row.extend([ mean, (cil, ciu) ])

            #         *group_id, col,*func( subdf[col].to_numpy() )   #mean, cil, ciu

        return pd.DataFrame( data = ci_data, columns = future_cols ).set_index( groupby_features )


    @classmethod
    def compute_ci__t_student( cls, data: np.array, confidence_lvl: float = 0.95 ):
        mean_score, n = np.mean( data ), len( data )
        std_err = np.std( data, ddof=1 ) / (n ** .5)

        # Valore critico di t
        t_crit = stats.t.ppf((1 + confidence_lvl) / 2, df=n-1)
        margin_of_error = t_crit * std_err

        ci_lower = mean_score - margin_of_error
        ci_upper = mean_score + margin_of_error

        return mean_score, ci_lower, ci_upper


    @classmethod 
    def compute_ci__boostrapping( cls, data: np.array, n_iter: int = 1000 ):
        bootstrap_means = [
            np.mean( np.random.choice( data, size = len(data), replace = True ) )
                for _ in range( n_iter )
        ] 
        ci_lower = np.percentile( bootstrap_means, 2.5 )
        ci_upper = np.percentile( bootstrap_means, 97.5 )
        return np.mean(data), ci_lower, ci_upper
        

# Functions
def train_test_split__stratified(df, strata, size=None, seed=None):#, keep_index= True):
    '''
    It samples data from a pandas dataframe using strata. These functions use
    proportionate stratification:
    n1 = (N1/N) * n
    where:
        - n1 is the sample size of stratum 1
        - N1 is the population size of stratum 1
        - N is the total population size
        - n is the sampling size
    Parameters
    ----------
    :df: pandas dataframe from which data will be sampled.
    :strata: list containing columns that will be used in the stratified sampling.
    :size: sampling size. If not informed, a sampling size will be calculated
        using Cochran adjusted sampling formula:
        cochran_n = (Z**2 * p * q) /e**2
        where:
            - Z is the z-value. In this case we use 1.96 representing 95%
            - p is the estimated proportion of the population which has an
                attribute. In this case we use 0.5
            - q is 1-p
            - e is the margin of error
        This formula is adjusted as follows:
        adjusted_cochran = cochran_n / 1+((cochran_n -1)/N)
        where:
            - cochran_n = result of the previous formula
            - N is the population size
    :seed: sampling seed
    :keep_index: if True, it keeps a column with the original population index indicator
    
    Returns
    -------
    A sampled pandas dataframe based in a set of strata.
    Examples
    --------
    >> df.head()
    	id  sex age city 
    0	123 M   20  XYZ
    1	456 M   25  XYZ
    2	789 M   21  YZX
    3	987 F   40  ZXY
    4	654 M   45  ZXY
    ...
    # This returns a sample stratified by sex and city containing 30% of the size of
    # the original data
    >> stratified = stratified_sample(df=df, strata=['sex', 'city'], size=0.3)
    Requirements
    ------------
    - pandas
    - numpy
    '''


    def discretize_covariate(df, cov) -> str:
        def quartile_assignment(cov_value):
            if cov_value < q[0.25]:
                return f"Q1_{cov}"
            elif cov_value < q[0.5]:
                return f"Q2_{cov}"
            elif cov_value < q[0.75]:
                return f"Q3_{cov}"
            else:
                return f"Q4_{cov}"


        q = df[cov].quantile([0.25,0.5,0.75])
        discrete_cov_name = f"Q_{cov}"
        df[ discrete_cov_name ] = df[cov].apply( quartile_assignment )
        return discrete_cov_name
    

    #### get CATEGORICAL strata 
    aux_cols = [] 

    numeric_columns = set( df.select_dtypes(include=['number']).columns ).intersection(strata)
    my_strata = [ s for s in strata if s not in numeric_columns ]

    for nc in numeric_columns:
        my_strata.append( discretize_covariate(df, nc) )
        aux_cols.append( my_strata[-1] )


    population = len(df)
    size = round( population * size )
    tmp = df[my_strata]

    tmp['size'] = 1
    tmp_grpd = tmp.groupby(my_strata).count().reset_index()
    tmp_grpd['samp_size'] = round(size/population * tmp_grpd["size"]).astype(int)
    stratified = list() 


    for i in range(len(tmp_grpd)):
        # query generator for each iteration
        qry = ""
        
        for stratum in my_strata:
            value = tmp_grpd.iloc[i][stratum]
            n = tmp_grpd.iloc[i]['samp_size']

            if type(value) == str:
                value = f"'{value}'"
            
            qry += f" `{stratum}` == {value} &"
        
        # final dataframe
        qry = qry[:-1] ##remove final '&' character
        curr_samples = df.query(qry).sample(n=n, random_state=seed)
        stratified.extend( curr_samples.index.tolist() )

    if aux_cols:
        df.drop(columns=aux_cols, inplace = True )

    whole_samples = set( df.index.tolist() )
    diff = list( whole_samples.difference( stratified ) )

    return df.loc[ diff ].copy(), df.loc[ stratified ].copy()


class WilcoxonTest:
    def __init__( self, data: LabelledData, task: TaskParameters ):
        self.__my_data = data 
        self.__scaled_data = pd.DataFrame( 
            data = StandardScaler().fit_transform( self.__my_data.features ),
            columns = self.__my_data.features.columns, 
            index = self.__my_data.features.index 
        )
        self.__task = task 
        self.__samples = {
            task.get_class(True): data.metadata[ data.metadata[ task.target_cov ].isin(task.pos_class) ].index.tolist(),
            task.get_class(False): data.metadata[ data.metadata[ task.target_cov ].isin(task.neg_class) ].index.tolist()
        }
        self.__posneg_dict = {
            sample_type: data.features.loc[ sample_list ]
                for sample_type, sample_list in self.__samples.items() 
        }
        l1, l2 = self.__posneg_dict.keys() 
        wx_data = list()

        for feature in data.features.columns.tolist():
            w_test, w_p = stats.ranksums(
                self.__posneg_dict[l1][feature].to_numpy(),
                self.__posneg_dict[l2][feature].to_numpy() )   
            wx_data.append( (feature, w_test, w_p) )

        self.__df_wx = pd.DataFrame(
            data = wx_data,
            columns = ["feature", "w_stat", "p_value"]
        ).set_index("feature").sort_values("p_value", ascending=True)
        
            
    def get_wilcoxon_results( self, feature_list: List[ str ] = None ) -> pd.DataFrame:
        if feature_list is None:
            feature_list = self.__df_wx.index.tolist()
        return self.__df_wx.loc[ feature_list ]
    

    def get_boxplot_trends( self, features: FeatureSubset ) -> go.Figure: 
        boxplot_kwargs = dict(
            boxpoints='all',  # represent all points
            jitter=0.3,
            pointpos=-1.8,
            hovertemplate="<b>%{text}</b><br>Value: %{y}<extra></extra>", 
        )

        df = self.__scaled_data[ features ].reset_index().melt(id_vars=["index"])

        fig = go.Figure()

        for sample_type, sample_list in self.__samples.items():
            subdf = df[ df["index"].isin( sample_list ) ]
            self.__posneg_dict[sample_type] = self.__my_data.features[ features ].loc[ sample_list ]

            fig.add_trace( go.Box(
                x = subdf["variable"], 
                y = subdf["value"], 
                name=sample_type,
                text=subdf["index"],
                **boxplot_kwargs
            ))
   
        fig.update_layout( 
            title=f"Feature trends w.r.t. {self.__task.target_cov} covariate", 
            xaxis_title="Feature",
            yaxis_title="Z-score",
            boxmode='group', showlegend=True )

        return fig  


import pandas as pd 
import numpy as np 
import graph_tool.all as gt 

from typing import List, Tuple, Callable, Dict, Literal

from collections import defaultdict, Counter
from itertools import chain

from pymoo.core.problem import ElementwiseProblem, Problem
from multiprocessing.pool import ThreadPool
from pymoo.core.result import Result 
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.expx import ExponentialCrossover

from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
from pymoo.termination.robust import RobustTermination


from pymoo.algorithms.moo.nsga2 import NSGA2
from dataclasses import dataclass, field, asdict

import statsmodels.api as sm 
import statsmodels.formula.api as smf 
from scipy.stats import pointbiserialr

from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, clone

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, make_scorer, average_precision_score, f1_score, matthews_corrcoef

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tools.enums import enum, GraphField
from tools.ml_utils import LabelledData, FeatureSubset
from tools.gtools import new_string_gp, new_numeric_vp
import joblib, time, tempfile, logging, os 

import shap 


class GuideMetric(enum.Enum):
    ROC_AUC = "roc_auc"
    ROC_PR = "roc_pr"
    F1_SCORE = "f1-score"
    MCC = "mcc"

    @classmethod
    def get_function(cls, metric: str) -> Callable:
        match metric: 
            case cls.ROC_AUC.value:
                return roc_auc_score
            case cls.ROC_PR.value:
                return average_precision_score
            case cls.F1_SCORE.value:
                return f1_score
            case cls.MCC.value:
                return lambda y_true, y_pred: matthews_corrcoef(y_true, np.where(y_pred > 0.5, 1, 0))
            case _:
                return roc_auc_score 
            

@dataclass
class CommunityInfo:
    id: int
    indices: np.array
    adj: np.array
    relevances: np.array
    shape: Tuple[int, int] = field(init=False)
    avg_relevance: float = field(init=False)
    w_avg_redundancy: float = field(init=False)
    cluster_coeff: float = field(init=False)
    density: float = field(init=False)
    # ub_density: float = field(init=False)
    diff_density: float = field(init=False)
    norm_relevances: float = field(init=False)
    

    def __post_init__(self):
        n_features = self.adj.shape[0]
        n_edges = np.count_nonzero( self.adj ) #/2 
        self.shape = (n_features, n_edges)


        if False:
            ### average relevance score in the community
            self.norm_relevances = np.linalg.norm( self.relevances ) / (n_features**.5)
            self.avg_relevance = np.mean( self.relevances )
            ### sum of the absolute values of the edges in the community
            edge_sum = np.sum( np.abs( self.adj ) )


            self.density = edge_sum / n_edges if n_edges > 0 else 0.
            self.diff_density = 1 - self.density 
            self.w_avg_redundancy = self.density
            self.cluster_coeff = self.norm_relevances / self.diff_density

        self.norm_relevances = np.linalg.norm( self.relevances )  
        self.avg_relevance = np.mean( self.relevances )
        self.density = np.sqrt( np.sum( np.abs( self.adj ) ) / ( n_features * (n_features - 1) ) ) if n_features > 1 else 0. 
        self.diff_density = 1 - self.density
        self.w_avg_redundancy = self.density
        self.cluster_coeff = (1. / self.diff_density) if  self.diff_density > 0 else 1e-06


    def asdict(self):
        return {
            "c_id": self.id,
            "num_features": self.shape[0],
            "num_edges": self.shape[1],
            "avg_relevance": self.avg_relevance,
            "norm_relevances": self.norm_relevances,
            "redundancy": self.w_avg_redundancy,
            "density": self.density,
            "cluster_coeff": self.cluster_coeff,
            # "imp_score": self.avg_relevance / (1+self.density)
        }


class InfoCommunityDetectionMethod:
    def __init__(self, 
                 adj_matrix: np.array,
                 df_communities: pd.DataFrame, 
                 cd_method: str, 
                 col_feature_index: str, 
                 relevance_col: str = "score" ):
        
        self.__community_info = list()    

        for c_id, subdf in df_communities.groupby( cd_method ):
            indices_comm = subdf[col_feature_index].to_numpy() #subdf.int_id.to_numpy()
            self.__community_info.append( 
                CommunityInfo(
                    id = c_id,
                    indices = indices_comm,
                    adj = adj_matrix[ np.ix_(indices_comm, indices_comm) ],
                    relevances = subdf[relevance_col].to_numpy()
                )
            )
        else:
            ## normalize cluster coefficients in range [0, 1]
            total_sum = 1 + sum([ c_info.cluster_coeff for c_info in self.community_info ])
            for c_info in self.community_info:
                c_info.cluster_coeff /= total_sum
    
    @property
    def community_info(self):
        return self.__community_info    


if False:
    class WholeGraphBasedFeatureSelectionProblem( ElementwiseProblem ):
        def __init__(self, g: gt.Graph, score: str, **kwargs): 
            super().__init__( n_var = g.num_vertices(), n_obj = 2, xl = 0, xu = 1, var_type=int, **kwargs)
            self.feature_graph = g 
            self.scores = self.scores = self.get_relevance_scores( score )
            self.adj = np.triu( np.abs( gt.adjacency( g, weight = g.ep.WholeData  ).toarray() ) )


        def get_relevance_scores(self, score: str):
            scores = np.nan_to_num( self.feature_graph.vp[ score ].a, nan = 0.0 ) 
            return ( scores - scores.min() ) / (scores.max() - scores.min())
        

        def _evaluate( self, x, out, *args,  **kwargs ):
            relevance, redundancy = 0., np.inf 

            if (nf := np.sum( x )):
                relevance = np.linalg.norm( np.multiply( x, self.scores ) ) * np.log10(1+nf)/nf

                sub_x = np.nonzero( x )[0]
                sub_A = self.adj[ np.ix_(sub_x, sub_x)]

                n_edges = np.nonzero(sub_A)[0].shape[0]
                redundancy = 0. if n_edges == 0 else np.linalg.norm( sub_A )  * (n_edges**.5)

            out["F"] = [ -relevance, redundancy ]


    class AbstractGraphBasedFeatureSelectionProblem( ElementwiseProblem ):
        def __init__(self, g: gt.Graph, df_communities: pd.DataFrame, cd_method: str, score: str, **kwargs):
            super().__init__( 
                n_var = g.num_vertices(), n_obj = 2, 
                xl = 0, xu = 1, var_type=int, 
                **kwargs)
            self.feature_names = np.array( list( g.vp.feature ) )
            score = score or GraphField.VERTEX_WEIGHT.value.format("avg_vscore")
            self.scores = np.nan_to_num( g.vp[ score ].a, nan = 0.0 ) 
            self.adj = np.triu( np.abs( gt.adjacency( g, weight = g.ep.WholeData  ).toarray() ) )

            _df_communities = df_communities.copy()
            diff = set( self.feature_names).difference(_df_communities.index.tolist())
            assert len(diff) == 0

            _df_communities["score"] = self.scores
            ## assign integers to features following lexicographic order
            _df_communities["int_id"] = range( g.num_vertices() ) #enumerate features from 0 to N-1
            _df_communities.sort_values(by=[cd_method, "int_id"], inplace=True)
            ## assign integers to features following community ordering 
            _df_communities["gene_pos"] = range( g.num_vertices() )
            ## reordering features based in the original ordering
            _df_communities.sort_values(by="int_id", inplace=True)
            ## get the new ordering of features
            new_ordering = _df_communities.gene_pos.to_numpy() 
            ## reordering features based on community detection method
            self.feature_names = self.feature_names[ new_ordering ]
            self.scores = self.scores[ new_ordering ]
            self.adj = self.adj[ np.ix_( new_ordering, new_ordering ) ]
            
            info_communities = InfoCommunityDetectionMethod(
                adj_matrix = self.adj, 
                df_communities = _df_communities, 
                cd_method = cd_method, 
                col_feature_index="gene_pos", 
                relevance_col = "score" )
            self.community_info = info_communities.community_info


        def compute_relevance(self, x):
            relevance = 0. 
            weighted_x = np.multiply( x, self.scores )

            for comm_info in self.community_info:
                if (nf := np.count_nonzero( (x_curr := weighted_x[comm_info.indices]) )):
                    relevance += np.log1p(nf) / nf * np.linalg.norm( x_curr )  

            return relevance
        
        
        def compute_redundancy(self, x):
            redundancy = 0. 

            for comm_info in self.community_info:

                if np.any( x_curr := x[comm_info.indices]):
                    sub_x = np.nonzero( x_curr )[0]
                    sub_A = comm_info.adj[ np.ix_(sub_x, sub_x)].ravel()
                    if sub_A.size > 0: #np.any( sub_A ):
                        redundancy += np.linalg.norm( sub_A ) * comm_info.cluster_coeff
                        
            return redundancy 

            
    class HybridGraphBasedFeatureSelectionProblem( AbstractGraphBasedFeatureSelectionProblem ):
        def __init__(self, 
                    g: gt.Graph, 
                    dataset: LabelledData, 
                    algo: BaseEstimator, 
                    df_communities: pd.DataFrame, 
                    cd_method: str, 
                    guide_metric: GuideMetric, **kwargs    ):
            
            super().__init__( g, df_communities, cd_method, score=None, **kwargs)
            self.data = dataset
            self.guide_metric = ( guide_metric.value, GuideMetric.get_function( guide_metric.value ) )
            self.algo = Pipeline([("scaler", StandardScaler() ),("clf", algo)])


        def _evaluate( self, x, out, *args,  **kwargs ):
            relevance, redundancy = 0., np.inf

            if np.any(x):
                relevance = self.__compute_average_guide_metric(x)
                redundancy = self.compute_redundancy(x)


            out["F"] = [ -relevance, redundancy ]


        def __compute_average_guide_metric(self, x ):
            chromo_features = self.feature_names[ np.nonzero( x )[0] ]
            my_data = self.data.select_features( chromo_features )
            metric_value = cross_val_score(
                estimator = self.algo, 
                X = my_data.features, 
                y = my_data.target, 
                cv = 5,  
                scoring = make_scorer( self.guide_metric[1], response_method="predict_proba"), # needs_proba=True), 
                n_jobs=-1
            )
            
            return metric_value.mean()


    class PenalizedCommunityBasedFeatureSelectionproblem( AbstractGraphBasedFeatureSelectionProblem ):
        def __init__(self, g: gt.Graph, df_communities: pd.DataFrame, cd_method: str, score: str, **kwargs):
            super().__init__( g, df_communities, cd_method, score, **kwargs )


        def _evaluate( self, x, out, *args,  **kwargs ):
            relevance = self.compute_relevance(x)
            redundancy = self.compute_redundancy(x)
            

            out["F"] = [ -relevance, redundancy ]


    class CommunityBasedFeatureSelectionProblem( AbstractGraphBasedFeatureSelectionProblem ):
        def __init__(self, g: gt.Graph, df_communities: pd.DataFrame, cd_method: str, score: str, **kwargs ):
            super().__init__( g, df_communities, cd_method, score, **kwargs )


        def _evaluate( self, x, out, *args,  **kwargs ):
            redundancy = .0


            for comm_info in self.community_info:
                
                if np.count_nonzero( (x_curr := x[comm_info.indices]) ):
                    sub_x = np.nonzero( x_curr )[0]
                    sub_A = comm_info.adj[ np.ix_(sub_x, sub_x)]

                    if (n_edges := np.count_nonzero(sub_A)) > 0:
                        redundancy += np.sum(sub_A) * (n_edges**.5) # np.log10(1+n_edges)
            
            out["F"] = [ -self.compute_relevance(x), redundancy ]


#         series_fsets ) )
    

def get_solutions_from_generation( feature_graph: gt.Graph, gen_id: int, solutions: List, f_values: np.array ) -> Tuple[ Dict, Dict ]:
    id_curr_gen = f"GA_gen_{gen_id+1}"
    series_fsets = [ 
        FeatureSubset( flist, "", None, feature_graph, dict(), uid = f"{id_curr_gen}__sol_{j}" )#.to_dict()
            for j, flist in enumerate( solutions )
    ]
    fsets_id, n_nodes, n_edges = zip(* map( 
        lambda wf: ( wf.uid, wf.n_features, wf.n_edges ), #( wf["fset_id"], wf["n_features"], wf["n_edges"] ), 
        series_fsets ) )
    
    solutions_dict = dict(
        fsets_id = fsets_id, 
        iteration = gen_id, 
        n_nodes = n_nodes, 
        n_edges = n_edges, 
    )
    fvalues_dict = dict(
        iteration = gen_id, 
        relevance = np.abs(f_values[:,0]), 
        redundancy = f_values[:,1]
    )
    return solutions_dict, fvalues_dict 


### Problem Level Abstraction

def compute_partial_redundancy( 
        x_file: str, x_shape, x_dtype, 
        i_begin, i_end, 
        adj_file: str, adj_shape: tuple, adj_dtype,
        c_w: float ) -> np.array:
    
    # Load the shared array for the current worker
    x_shared = np.memmap(x_file, dtype=x_dtype, mode='r', shape=x_shape)
    x = x_shared[:, i_begin: i_end]

    adj = np.memmap(adj_file, dtype=adj_dtype, mode='r', shape=adj_shape)
    red_c = np.zeros( shape = ( x.shape[0], ), dtype = float )
    memoiz = dict() 

    for i in range( x.shape[0] ):
        idx = np.flatnonzero( x[i, :] )
        
        if idx.size > 0:
            if (tup := tuple( idx )) in memoiz:
                red_c[i] = memoiz[ tup ]
                continue

            memoiz[ tup ] = red_c[i] = np.linalg.norm( adj[ np.ix_( idx, idx ) ] ) 
            

    return red_c * c_w


class SHAPFeatureImportance:
    def __init__(self, X: pd.DataFrame, y, model: BaseEstimator, n_reps: int = 10 ):
        splitter = StratifiedKFold(n_splits=n_reps, shuffle=True, random_state=42)
        feature_scores = [
            self.__get_shap( X.iloc[train_idx], y.iloc[train_idx], model )
                for train_idx, _ in splitter.split(X, y) 
        ]
        feature_scores = np.mean( np.array( feature_scores ), axis=0 )
        self.__shap_values = pd.DataFrame(
            data = list( zip( X.columns.tolist(), feature_scores ) ),
            columns = ["feature", "shap_value"]
        ).set_index("feature")

    @property
    def shap_values(self) -> pd.DataFrame:
        return self.__shap_values
    
    @staticmethod
    def __get_shap(X, y, model: BaseEstimator) -> np.ndarray:

        X_scaled = pd.DataFrame(
            data = StandardScaler().fit_transform( X ),
            columns = X.columns,
            index=X.index
        )
        fitted_model = clone(model).fit( X_scaled, y )

        match fitted_model:
            case LogisticRegression():
                explainer = shap.LinearExplainer( fitted_model, X_scaled )
                return (np.abs( explainer.shap_values( X_scaled ) ).mean( axis=0 ))
                
            case RandomForestClassifier() | XGBClassifier():
                explainer = shap.TreeExplainer( fitted_model )
                mean_values = np.abs( explainer.shap_values( X_scaled ) ).mean( axis=0 )
                if len(mean_values.shape) > 1:
                    return mean_values[:,0]     ## RandomForest
                else:
                    return mean_values          ## XGBoost 
            case _:
                raise NotImplementedError(f"SHAP values not implemented for model type: {type(fitted_model)}")


class InputDataMOEA:

    def __init__(self, g: gt.Graph, df_communities: pd.DataFrame, cd_method: str, v_score: str):
        self.df_communities = df_communities
        self.cd_method = cd_method
        self.v_score = v_score

        self.feature_names = np.array( list( g.vp.feature ) )
        score = v_score or GraphField.VERTEX_WEIGHT.value.format("avg_vscore")
        self.scores = np.nan_to_num( g.vp[ score ].a, nan = 0.0 ) 
        self.adj = np.triu( np.abs( gt.adjacency( g, weight = g.ep.WholeData  ).toarray() ) )

        _df_communities = df_communities.copy()
        diff = set( self.feature_names).difference(_df_communities.index.tolist())
        assert len(diff) == 0

        _df_communities["score"] = self.scores
        ## assign integers to features following lexicographic order
        _df_communities["int_id"] = range( g.num_vertices() ) #enumerate features from 0 to N-1
        _df_communities.sort_values(by=[cd_method, "int_id"], inplace=True)
        ## assign integers to features following community ordering 
        _df_communities["gene_pos"] = range( g.num_vertices() )
        ## reordering features based in the original ordering
        _df_communities.sort_values(by="int_id", inplace=True)
        ## get the new ordering of features
        new_ordering = _df_communities.gene_pos.to_numpy() 
        ## reordering features based on community detection method
        self.feature_names = self.feature_names[ new_ordering ]
        self.scores = self.scores[ new_ordering ]
        self.adj = self.adj[ np.ix_( new_ordering, new_ordering ) ]
        
        info_communities = InfoCommunityDetectionMethod(
            adj_matrix = self.adj, 
            df_communities = _df_communities, 
            cd_method = cd_method, 
            col_feature_index="gene_pos", 
            relevance_col = "score" )
        self.community_info = info_communities.community_info


class NetworkMOEAProblem( Problem ):
    def __init__(self, g: gt.Graph, df_communities: pd.DataFrame, cd_method: str, score: str, num_procs: int = 4, **kwargs):

        super().__init__(
            n_var = g.num_vertices(),
            n_obj=2, xl = 0, xu = 1,
            var_type=int, 
            **kwargs )
        
        self.input_data = InputDataMOEA( g, df_communities, cd_method, score )
        self.feature_names = self.input_data.feature_names

        self.temp_dir = tempfile.TemporaryDirectory()
        logging.critical(f"Creating temporary directory: {self.temp_dir.name}")
        self.__memmap_info = dict() 


        for c in self.input_data.community_info:
            adj_filename = f"{self.temp_dir.name}/comm_{c.id}_adj.mmap"
            ## store memmap info => ordering is the same as the parameters of compute_partial_redundancy
            self.__memmap_info[ c.id ] = ( adj_filename, c.adj.shape, c.adj.dtype )
            mem = np.memmap(adj_filename, dtype=c.adj.dtype, mode='w+', shape=c.adj.shape)
            mem[:] = c.adj[:]
            mem.flush()


        self.__relevance_function = None  #self.compute_relevance_by_score
        self.__relevance_kwargs = dict()
        self.__n_proc = min(num_procs, len( self.input_data.community_info ) )


    def set_relevance_function(self, 
        func: Literal["score", "model", "shap"], 
        score: str = None,
        clf: BaseEstimator = None,
        metric: str = None,
        ncv: int = None, 
        tr_set: LabelledData = None ) :


        match func:
            case "score":
                self.__relevance_function = self.compute_relevance_by_score
                assert score, "Score parameter must be provided for 'score' relevance function."
                self.__relevance_kwargs = {"score": score}

            case "model":
                self.__relevance_function = self.compute_relevance_by_model
                assert clf, "Model parameter must be provided for 'model' relevance function."
                assert isinstance(clf, BaseEstimator), "Classifier must be a scikit-learn estimator."
                assert metric, "Metric parameter must be provided for 'model' relevance function."
                assert ncv, "Number of cross-validation folds must be provided for 'model' relevance function."
                assert tr_set, "Training set parameter must be provided for 'model' relevance function."
                
                logging.critical(f"########################\nYou choose a model but we don't care-- Using LogisticRegression as default classifier.\n########################")
                self.__relevance_kwargs = {
                    "clf": Pipeline([("scaler", StandardScaler() ), ("clf", clf)]),
                    "metric": GuideMetric.get_function( metric ), 
                    "ncv": ncv, 
                    "data": tr_set
                }

                logging.critical(f"Model-based relevance function.\nClassifier: {clf}\nMetric: {metric}\nTraining set: {tr_set.features.shape}, {tr_set.target.shape}")
            case "shap":
                self.__relevance_function = self.compute_relevance_by_score
                assert tr_set, "Training set parameter must be provided for 'shap' relevance function"  
                assert ncv, "Number of repetitions must be provided for 'shap' relevance function."
                assert clf is not None, "Model parameter must be provided for 'model' relevance function."
                assert isinstance(clf, BaseEstimator), "Classifier must be a scikit-learn estimator."
                
                desired_feature_ordering = self.input_data.feature_names.tolist()
                shap_importances = SHAPFeatureImportance(
                    X = tr_set.features,   #[ desired_feature_ordering ], ##quite stupid idea, maybe use the original order and then reorder
                    y = tr_set.target, 
                    model = clf, 
                    n_reps = ncv ) 
                
                self.input_data.scores = shap_importances.shap_values.loc[ desired_feature_ordering , "shap_value" ].values
                self.__relevance_kwargs = { "score": f"SHAP_{metric}" }

        
        return self 
            
    
    def _evaluate(self, x, out, *args, **kwargs):

        y = np.zeros( shape = (x.shape[0], 2), dtype=float )
        y[:, 0] = (-1) * self.__relevance_function( x ) 
        y[:, 1] = self.compute_redundancy( x )
        out["F"] = y


    def compute_relevance_by_score(self, x) -> np.array:

        y = np.zeros( shape = ( x.shape[0], ), dtype = float )
        x_weighted = np.multiply( x, self.input_data.scores )

        for c in self.input_data.community_info:
            ## get the number of features selected in the community for each solution
            nf = np.sum( x[:, c.indices ], axis=1 ) 
            ## compute a scaling factor based on the number of features selected in the community
            coeff = np.divide( np.log1p( nf ), nf + 1 )
            ## accumulate the weighted relevance score for the community
            y += np.multiply( coeff, np.linalg.norm( x_weighted[:, c.indices ], axis=1 ) )

        ## we want to maximize relevance, so we return the negative value
        return y 
    

    def compute_relevance_by_model(self, x ):
        y = np.zeros( shape = ( x.shape[0], ), dtype = float )
        algo = self.__relevance_kwargs.get("clf")
        training_data = self.__relevance_kwargs.get("data")
        ncv = self.__relevance_kwargs.get("ncv")

        scorer = make_scorer( 
            score_func=self.__relevance_kwargs.get("metric"), 
            response_method="predict_proba" )

        for i, x_i in enumerate(x):
            chromo_features = self.feature_names[ np.nonzero( x_i )[0] ]
            my_data = training_data.select_features( chromo_features )
            metric_value = cross_val_score(
                estimator = algo, 
                X = my_data.features, 
                y = my_data.target, 
                cv = ncv,  
                scoring = scorer, 
                n_jobs=-1 )
            y[i] = np.mean( metric_value )
        
        return y 


    def compute_redundancy(self, x ) -> np.array:

        
        y = np.zeros( shape = ( x.shape[0], ), dtype = float )
        ## create a temporary memmap file for the current population
        ## TODO: optimize this step to avoid creating/deleting the file at each call
        temp_file = f"{self.temp_dir.name}/current_pop.mmap"
        x_shared = np.memmap(temp_file, dtype=x.dtype, mode='w+', shape=x.shape)
        x_shared[:] = x[:]  # Copy data once
        x_shared.flush()
        
        with joblib.Parallel(n_jobs=self.__n_proc, backend="loky") as parallel:
            results = parallel(
                joblib.delayed( compute_partial_redundancy )( 
                    ## args for current population
                    temp_file, x.shape, x.dtype,
                    ## indices corresponding to the community
                    c.indices[0], c.indices[-1] + 1,
                    ## args for community adjacency matrix
                    *self.__memmap_info[ c.id ],
                    c.cluster_coeff )
                for c in self.input_data.community_info
            )

        del x_shared  # Clean up the memmap
        os.unlink( temp_file )  # Remove the temporary file

        return np.sum( results, axis=0 )

###############################


class MOEAHistory:
    def __init__(self, moea_result: Result, feature_graph: gt.Graph, feature_groups: Dict[str, List[str]] ):
        self.__result = moea_result
        self.__feature_names = moea_result.problem.feature_names
        self.__history = moea_result.history
        self.__feature_pools = list()


        self.__feature_graph = feature_graph
        self.__feature_groups = feature_groups

        list_of_records = list()
        feature_freqs = np.zeros( shape = ( len(self.__feature_names), len(self.__history) ), dtype = float )

        for i, h in enumerate( self.__history, 1):
            res = h.result()
            mean_rel, mean_red = res.F.mean(axis=0)
            
            pool_mask = np.bitwise_or.reduce([x.X for x in res.pop], axis=0)
            feature_freqs[:, i-1] = np.sum( [ x.X for x in res.pop ], axis=0) / len(res.pop)

            feature_pool = FeatureSubset(
                feature_list = self.__get_feature_names( pool_mask ),
                metadata = f"GA_pool_{i}", 
                id_count=None, 
                feature_graph=feature_graph, 
                feature_groups=feature_groups 
            )
            self.__feature_pools.append( feature_pool )
            composition = { omic: number for omic, number in feature_pool.omic_composition }

            list_of_records.append({
                "generation": i,
                **composition,
                "n_features": len(feature_pool),
                "n_edges": feature_pool.n_edges,
                "n_edges_0.05": feature_pool.n_edges_05,
                "n_edges_0.01": feature_pool.n_edges_01,
                "n_edges_0.001": feature_pool.n_edges_001,
                "mean_relevance": mean_rel * (-1),
                "mean_redundancy": mean_red
            }) 

        self.__df_evolution = pd.DataFrame( list_of_records ).set_index("generation")
        self.__df_feature_freqs = pd.DataFrame(
            data = feature_freqs, 
            index = self.__feature_names,
            columns = [ f"gen_{i+1}" for i in range( feature_freqs.shape[1] ) ]
        )

    @property
    def population_size(self) -> int:
        return len( self.__history[0].result().pop )
    
    @property
    def n_generations(self) -> int:
        return len( self.__history )

    @property
    def df_evolution(self) -> pd.DataFrame:
        return self.__df_evolution
    
    @property
    def df_feature_frequencies(self) -> pd.DataFrame:
        return self.__df_feature_freqs
    

    def __get_feature_names(self, mask: np.array) -> List[str]:
        return [ f for i, f in enumerate(self.__feature_names) if mask[i] ]

    def get_population(self, gen_id: int) -> List[FeatureSubset]:
        pop = self.__history[ gen_id - 1 ].result().pop
        feature_sets = [ self.__get_feature_names( x.X ) for x in pop ]

        return [
            FeatureSubset(
                feature_list = flist,
                metadata = f"GA_gen_{gen_id}__sol_{j}",
                id_count=None,
                feature_graph=self.__feature_graph,
                feature_groups=self.__feature_groups ) 
                
            for j, flist in enumerate(feature_sets)
        ]
    
    def get_feature_pool(self, gen_id: int) -> FeatureSubset:
        return self.__feature_pools[ gen_id - 1 ]   
    
    def multiomics_convergence_plot(self, omic_palette: Dict[str, str]) -> go.Figure: 
        x_axis = self.__df_evolution.index.tolist() 
        fig = go.Figure()
        scatter_args = dict( x = x_axis, mode="lines+markers" )

        fig.add_trace(
            go.Scatter(
                name = "All features",
                y = self.__df_evolution["n_features"].tolist(),
                marker = dict( color = "gray" ),
                line = dict( color = "gray", width=4 ),
                **scatter_args
            )
        )

        for omic in self.__feature_groups.keys():
            y_axis = self.__df_evolution[ omic ].tolist()
            fig.add_trace(
                go.Scatter(
                    name = omic,
                    y = y_axis,
                    marker = dict( color = omic_palette.get( omic, "gray" ) ),
                    line = dict( color = omic_palette.get( omic, "gray" ), width=2 ),
                    **scatter_args
                )
            )

        fig.update_xaxes(title_text="Generation")
        fig.update_yaxes(title_text="Number of features")
        ##set name of the legend 
        fig.update_layout(
            title="MOEA Convergence Plot: number of features per generation", 
            legend_title_text="Omic Layer",
            legend=dict(
                orientation="v",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        return fig

    def complexity_reduction_plots(self) -> go.Figure:
        palette_name = "agsunset"


        df = self.__df_evolution


        fig = make_subplots(rows=1, cols=2, subplot_titles=("Pareto Front", "Solutions complexity"))
        fig.add_trace(
            go.Scatter( 
                x = df.mean_redundancy,
                y = df.mean_relevance, 
                text = df.index,
                mode = "markers", 
                marker = dict( 
                    color = df.index, 
                    colorscale = palette_name, 
                    showscale = True, 
                    colorbar= dict(title="generation") )
            ), 
            row = 1, col = 1
        )

        fig.add_trace(
            go.Scatter( 
                x = df.n_features, 
                y = df.n_edges,
                text = df.index ,
                mode = "markers", 
                marker = dict( 
                    color = df.index, 
                    colorscale = palette_name, 
                    showscale = False )
            ), 
            row = 1, col = 2
        )

        fig.update_xaxes(title_text="Redundancy",  row=1, col=1)
        fig.update_yaxes(title_text="Relevance", row=1, col=1)

        fig.update_xaxes(title_text="Num. nodes",  row=1, col=2)
        fig.update_yaxes(title_text="Num. edges",  row=1, col=2)

        fig.update_layout(
            showlegend=False,
            coloraxis_colorbar=dict(
                yanchor="top",
                y=1,
                x=0,
                ticks="outside"
            )
        )
        return fig  


    def heatmap_evolution_plot(self) -> go.Figure:
        fig = go.Figure(
            data=go.Heatmap(
                z=self.__df_feature_freqs.values,
                x=np.arange(1, self.__df_feature_freqs.shape[1]+1),
                y=self.__df_feature_freqs.index,
                colorscale='Viridis'
            )
        )

        fig.update_layout(
            title="Feature Selection Frequency Across Generations",
            xaxis_title="Generation",
            yaxis_title="Feature"
        )

        return fig

    
class GraphBasedFeatureSelection:

    @classmethod
    def run_optimization_algorithm( 
        cls, 
        problem: Problem, 
        pop_size: int, 
        max_n_gen: int = 1000, 
        seed: int = 42, 
        seeded_pop: bool = False,
        ostream = None ) -> Result:


        initial_sample = BinaryRandomSampling()

        if seeded_pop:
            seeded_prop = 0.1
            ostream(f"Seeded population enabled with p={seeded_prop} -> {int(pop_size*seeded_prop)} out of {pop_size}...")
            ## define distribution based on vertex scores 
            probs = problem.feature_graph.vp[ GraphField.VERTEX_WEIGHT.value.format("avg_vscore") ].a
            probs /= np.sum(probs)

            ## initial random population 
            initial_sample = (np.random.random((pop_size, problem.n_var)) < 0.5).astype(bool)

            ## select random row indices to be seeded
            for i in np.random.choice( pop_size, size=int(pop_size * seeded_prop), replace=False ):
                num_ones = np.nonzero( initial_sample[i, :] )[0] 
                chosen_indices = np.random.choice( problem.n_var, size=len(num_ones), replace=False, p=probs )
                initial_sample[i, chosen_indices ] = True


        selected_ga_algoritm = NSGA2
        ga_args = dict(
            pop_size=pop_size,
            sampling=initial_sample,
            crossover= ExponentialCrossover(), #HalfUniformCrossover(),#UniformCrossover(),#TwoPointCrossover(),
            mutation=BitflipMutation(prob = 0.1),
            eliminate_duplicates=True )
        

        algorithm = selected_ga_algoritm( **ga_args, seed = seed )
        

        termination = RobustTermination(
            MultiObjectiveSpaceTermination(
                n_skip=5
            ), 
        )

        # prepare the algorithm to solve the specific problem (same arguments as for the minimize function)
        algorithm.setup(
            problem, 
            termination=termination, 
            seed=seed, 
            verbose=False, 
            save_history=True)
        
        time_spent = time_gen = time.time()

        while algorithm.has_next():
            # ask the algorithm for the next solution to be evaluated
            pop = algorithm.ask()

            # evaluate the individuals using the algorithm's evaluator (necessary to count evaluations for termination)
            algorithm.evaluator.eval(problem, pop)

            # returned the evaluated individuals which have been evaluated or even modified
            algorithm.tell(infills=pop)

            
            # do same more things, printing, logging, storing or even modifying the algorithm object
            if ostream is not None and algorithm.n_gen % 50 == 0:
                mean_rel, mean_red = np.mean(list(map(lambda x: x._F, pop)), axis=0)
                ostream(f"Generation {algorithm.n_gen} -- n_eval: {algorithm.evaluator.n_eval} -- avg. rel: {mean_rel:.3f} -- avg. red: {mean_red:.3f} -- req. time: {(time.time() - time_gen):.2f} seconds")
                time_gen = time.time()

            if algorithm.n_gen > max_n_gen:
                break
        
        if ostream is not None:
            mean_rel, mean_red = np.mean(list(map(lambda x: x._F, pop)), axis=0)
            time_spent = time.time() - time_spent
            ostream(f"Final generation {algorithm.n_gen} -- n_eval: {algorithm.evaluator.n_eval} -- avg. rel: {mean_rel:.3f} -- avg. red: {mean_red:.3f}")
            ostream(f"Total time: {time_spent:.2f} seconds -- avg. time per generation: {(time.time() - time_gen)/algorithm.n_gen:.2f} seconds")

        return algorithm.result()

    @classmethod
    def get_solutions( cls, result: Result ) -> List[List[str]]:
        solutions = [
            [ result.problem.feature_names[i] for i, b in enumerate( x.X ) if b ]
                for x in result.pop
        ]
        return solutions, result.F 
        

    @classmethod
    def get_features_through_generations(cls, graph, history ) -> pd.DataFrame:
        ## map from features to integers
        map_fnames = { name: i for i, name in enumerate(list(graph.vp[GraphField.FEATURE_NAME.value])) }
        n_features = len(map_fnames)
        my_data = list()


        for gen_data in history :
            a_curr = np.zeros(shape=(n_features,))
            my_data.append( a_curr )
            solutions, _ = cls.get_solutions( gen_data.result()  )
            for f, count in Counter( chain.from_iterable( solutions )).items():
                a_curr[ map_fnames[ f ] ] = count 

        my_data = np.vstack( my_data ).T
        # my_data[ my_data == 0 ] = -1

        return pd.DataFrame(
            data = my_data / np.max( my_data ),
            index=list(graph.vp[GraphField.FEATURE_NAME.value]) 
        )


    @classmethod
    def get_feature_pool( cls, solutions: List[List[str]] ) -> List[str]:
        return sorted( set( chain.from_iterable( solutions ) ) )
    

    @classmethod
    def get_plot(cls, graph: gt.Graph, history ) -> np.array:
        " restituisce array per y-axis: conteggio di quante feature ci sono alla i-esima generazione"
        omics_counts = defaultdict( list )

        my_props = [ graph.vp[ key.value ] for key in (GraphField.FEATURE_NAME, GraphField.FEATURE_OMIC )]
        omics_map = { feature: omic for feature, omic in zip( *my_props )}
        omics_set = set( my_props[1] )


        for algo_instance in history:
            solutions_set, _ = cls.get_solutions( algo_instance.result() )
            feature_pool = cls.get_feature_pool( solutions_set )
            # Countqre omiche 
            curr_counts = Counter([ omics_map[feature] for feature in feature_pool ])
            for omic_type in omics_set:
                omics_counts[ omic_type ].append( curr_counts[omic_type] )
            omics_counts[ "y_axis" ].append( len(feature_pool) )

        y_axis = omics_counts.pop("y_axis")
        omics_np_arrays = { key: np.array( value ) for key, value in omics_counts.items() }

        return np.array( y_axis ), omics_np_arrays


    @classmethod
    def build_ga_history_dataframe( cls, ga_history, feature_graph: gt.Graph ) -> pd.DataFrame:
        def concat_dicts( dicts: List[ Dict ] ) -> pd.DataFrame:
            return pd.concat( [pd.DataFrame(data=d) for d in dicts] )

        results = list()
        solution_results = map( lambda gen_data: cls.get_solutions( gen_data.result() ), ga_history )
        for i, pair in enumerate( solution_results ):
            results.append( get_solutions_from_generation( feature_graph, i, pair[0], pair[1] ) )


        if False:

            with joblib.Parallel(n_jobs=-1) as parallel:
                #         for generation_data in ga_history
                solution_results = map( lambda gen_data: cls.get_solutions( gen_data.result() ), ga_history )
                results = parallel(
                    joblib.delayed( cls.get_fast_solutions_from_generation )( i, pair[0], pair[1] )
                        for i, pair in enumerate( solution_results )  #map( lambda gen_data: cls.get_solutions( gen_data.results() )))
                )
        solution_data, fvalues_data = zip( *results )
        return (
            concat_dicts( solution_data ), 
            concat_dicts( fvalues_data )
        )


    @classmethod
    def build_ga_history_dataframe__old( cls, ga_history, feature_graph: gt.Graph ) -> pd.DataFrame:
        df_solutions = list()
        df_f_values = list()

        for i, generation_data in enumerate( ga_history ):
            solutions, f_values = cls.get_solutions( generation_data.result() )
            id_curr_gen = f"GA_gen_{i+1}"
            series_fsets = [ 
                FeatureSubset( flist, "", None, feature_graph, dict(), uid = f"{id_curr_gen}__sol_{j}" )#.to_dict()
                    for j, flist in enumerate( solutions )
            ]
            fsets_id, n_nodes, n_edges = zip(* map( 
                lambda wf: ( wf.uid, wf.n_features, wf.n_edges ), #( wf["fset_id"], wf["n_features"], wf["n_edges"] ), 
                series_fsets ) )
            
            df_solutions.append( pd.DataFrame( data = dict(
                fsets_id = fsets_id, 
                iteration = i, 
                n_nodes = n_nodes, 
                n_edges = n_edges, 
            )))
            df_f_values.append( pd.DataFrame( data = dict(
                iteration = i, 
                relevance = np.abs(f_values[:,0]), 
                redundancy = f_values[:,1]
            )
            ))
        
        return pd.concat( df_solutions ), pd.concat( df_f_values )


def compute_feature_scores__joblib( X: pd.DataFrame, y: pd.Series ):
    perf_models = {
        "lr_lasso": LogisticRegression(penalty="l1", solver="liblinear"),
        "lr_ridge": LogisticRegression(penalty="l2", solver="liblinear"), 
        "nb": GaussianNB(), 
        "knn": KNeighborsClassifier(n_neighbors=5, weights="distance"), 
        "dt_entropy": DecisionTreeClassifier(criterion="entropy", max_depth=3, max_features="log2"), 
        "dt_gini": DecisionTreeClassifier(criterion="gini", max_depth=3, max_features="log2"), 
        # "gpc": GaussianProcessClassifier()
    }

    mean_aucs = {
        f"AUC_{m_id}": np.mean( cross_validate( m, X, y, cv = 3, scoring = "roc_auc" ).get("test_score") )
            for m_id, m in perf_models.items() 
    }
    ## geometric mean of AUCs
    mean_aucs["AUC"] = np.prod( list(mean_aucs.values()) ) ** (1. / len(mean_aucs))
    return mean_aucs


class FeatureGraphEnricher:

    ## possibili score per le features:
    # - PBC: massimizzare somma |coefficienti| => rendi negativa
    # - MI: massimizzare mutua informazione feature-target => rendi negativa 
    # - pvalues ANOVA -> da minimizzare => già ok
    # - AUC classificatore semplice in CV sul training set => rendi negativa
    # - MCC dei clf 

    @classmethod
    def __compute_logistic( cls, X: pd.DataFrame, y):
        X_tmp = X.copy() 
        X_tmp.columns = [ f"f_{i}" for i in range( len(X_tmp.columns) ) ]
        X_tmp[ "target" ] = y 
        scores = list()

        for i, feature in enumerate( X.columns ):
            result = smf.glm( formula = f"target ~ f_{i}", data = X_tmp, family = sm.families.Binomial() ).fit()
            scores.append( ( result.params[f"f_{i}"],  list(result.pvalues)[1] ) )

        return scores 

    @classmethod
    def __scale_pvalues( cls, pvalues ):
        return np.log10( pvalues)


    @classmethod
    def compute_AUC_vertex_weights(cls, g: gt.Graph, data: LabelledData, n_jobs: int = -1 ) -> pd.DataFrame:
        feature_ordering = list( g.vp[ GraphField.FEATURE_NAME.value ] )
        with joblib.Parallel(n_jobs=n_jobs) as parallel:
            results = parallel(
                joblib.delayed( compute_feature_scores__joblib )( data.features[[feature]], data.target )
                    for feature in feature_ordering
            )
            return pd.DataFrame( data = results, index = feature_ordering )
        

    @classmethod
    def compute_vertex_weights(cls, g: gt.Graph, data: LabelledData ):

        feature_ordering = list( g.vp[ GraphField.FEATURE_NAME.value ] )
        feature_omics = list( g.vp[ GraphField.FEATURE_OMIC.value ] )
        X = data.features[ feature_ordering ].copy()
        y = data.target 


        pbc_scores = [ pointbiserialr( y, X[ feature ] ) for feature in feature_ordering ]
        pbc, p_pbc = zip( *pbc_scores )
        _, p_logreg = zip( * cls.__compute_logistic( X, y ) )


        my_data = dict(
            mi = mutual_info_classif( X, y ),    
            pbc = np.abs( pbc ),                                ### force values to positive values 
            log_p_logreg = -cls.__scale_pvalues( p_logreg ),    ### switch to positive values
            # **feature_performances
        )
        my_data = pd.DataFrame( data = my_data, index = feature_ordering )
        my_data.insert( 0, "omic", feature_omics )
        return my_data


    @classmethod
    def set_vertex_weights(cls, g: gt.Graph, target_cov: str, df_set_vweights: pd.DataFrame ):
        for col_name, series in df_set_vweights.items():
            new_numeric_vp( 
                g, GraphField.VERTEX_WEIGHT, "double", series.to_numpy(), fmt = col_name )

        stringify_scoreset = "$".join( sorted( df_set_vweights.columns.tolist() ) )
        new_string_gp( g, GraphField.FEATURE_SCORES, stringify_scoreset, fmt = target_cov )
    

    @classmethod
    def get_vertex_weights(cls, g: gt.Graph, target_cov: str ) -> List[str]:# -> pd.DataFrame:
        scores = list()
        if cls.is_weighted_graph(g, target_cov):
            scores = g.gp[ GraphField.FEATURE_SCORES.value.format( target_cov ) ].split("$")
        return scores
    

    @classmethod
    def get_dataframe_vertex_weights(cls, g: gt.Graph, target_cov: str ) -> pd.DataFrame:
        graph_data = dict( feature = g.vp[ GraphField.FEATURE_NAME.value ] )

        ## collect raw vertex weights from graph
        graph_data.update({
            score_id: np.array( g.vp[ GraphField.VERTEX_WEIGHT.value.format( score_id ) ].a )
                for score_id in cls.get_vertex_weights( g, target_cov )
        })

        #     graph_data[ score_id ] = np.array( g.vp[ GraphField.VERTEX_WEIGHT.value.format( score_id ) ].a )

        df = pd.DataFrame( graph_data ).set_index( GraphField.FEATURE_NAME.value )
        df.drop(columns=list(filter(lambda c: c.startswith("AUC_"), df.columns)), inplace=True)
        
        ## divide each column by its maximum value
        df.insert(0, "avg_vscore", df.apply( lambda x: x / np.sum(x), axis=0).mean(axis=1))
        df.insert( 0, GraphField.FEATURE_OMIC.value, list( g.vp[ GraphField.FEATURE_OMIC.value ] ) )
        # df[ GraphField.FEATURE_OMIC.value ] = g.vp[ ]
        return df


    @classmethod
    def is_weighted_graph(cls, g: gt.Graph, target_cov: str ) -> bool:
        return bool( GraphField.FEATURE_SCORES.value.format( target_cov ) in g.gp )


    @classmethod
    def update_graph_weights(cls, g: gt.Graph, df_w: pd.DataFrame):
        vprops = set( g.vp.keys() )
        cols2skip = { GraphField.FEATURE_NAME.value, GraphField.FEATURE_OMIC.value }

        for col_name, series in df_w.items():
            if col_name in cols2skip:
                continue 

            if GraphField.FEATURE_SCORES.value.format( col_name ) in vprops:
                print(f"Updating graph weights for {col_name}")
                g.vp[ GraphField.VERTEX_WEIGHT.value.format( col_name ) ].a = series.to_numpy()
            else:
                print(f"Creating new graph weights for {col_name}")
                new_numeric_vp( g, GraphField.VERTEX_WEIGHT, "double", series.to_numpy(), fmt = col_name )
        
        return g 


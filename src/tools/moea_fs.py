import pandas as pd 
import numpy as np 
import graph_tool.all as gt 

from typing import List, Tuple, Callable, Dict, Literal

from collections import defaultdict, Counter
from itertools import chain

from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.core.result import Result 
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.crossover.pntx import PointCrossover, SinglePointCrossover, TwoPointCrossover
from pymoo.operators.crossover.hux import HalfUniformCrossover
from pymoo.operators.crossover.expx import ExponentialCrossover

from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
from pymoo.termination.robust import RobustTermination


from pymoo.algorithms.moo.nsga2 import NSGA2
from dataclasses import dataclass, field, asdict

import statsmodels.api as sm 
import statsmodels.formula.api as smf 
from scipy.stats import pointbiserialr
from scipy.special import softmax 

from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, make_scorer, average_precision_score, f1_score, matthews_corrcoef


import enum 
import joblib, time, tempfile, os 

import logging
import timeit


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
            "cluster_coeff": self.cluster_coeff
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


class InputDataMOEA:

    def __init__(self, g: gt.Graph, df_communities: pd.DataFrame, cd_method: str, v_score: str):
        self.df_communities = df_communities
        self.cd_method = cd_method
        self.v_score = v_score

        self.feature_names = np.array( list( g.vp.feature ) )
        score = v_score #or GraphField.VERTEX_WEIGHT.value.format("avg_vscore")
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


class NetworkMOEAProblem( Problem ):
    def __init__(self, g: gt.Graph, df_communities: pd.DataFrame, cd_method: str, score: str, num_procs: int = 4, **kwargs):

        super().__init__(
            n_var = g.num_vertices(),
            n_obj=2, xl = 0, xu = 1,
            var_type=int, 
            **kwargs )
        
        self.input_data = InputDataMOEA( g, df_communities, cd_method, score )

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
        func: Literal["score", "model"], 
        score: str = None,
        clf: BaseEstimator = None,
        metric: str = None,
        tr_set = None ) :


        match func:
            case "score":
                self.__relevance_function = self.compute_relevance_by_score
                assert score, "Score parameter must be provided for 'score' relevance function."
                self.__relevance_kwargs = {"score": score}

            case "model":
                self.__relevance_function = self.compute_relevance_by_model
                assert clf, "Model parameter must be provided for 'model' relevance function."
                assert metric, "Metric parameter must be provided for 'model' relevance function."
                assert tr_set, "Training set parameter must be provided for 'model' relevance function."
                
                self.__relevance_kwargs = {
                    "clf": Pipeline([("scaler", StandardScaler() ),("clf", LogisticRegression())]), 
                    "metric": GuideMetric.get_function( metric ), 
                    "data": tr_set
                }
            case _:
                raise ValueError(f"Unknown relevance function: {func}")
        
            
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
        my_data = self.__relevance_kwargs.get("data")
        scorer = make_scorer( 
            score_func=self.__relevance_kwargs.get("metric"), 
            response_method="predict_proba" )

        for i, x_i in enumerate(x):
            chromo_features = self.feature_names[ np.nonzero( x_i )[0] ]
            my_data = self.data.select_features( chromo_features )
            metric_value = cross_val_score(
                estimator = algo, 
                X = my_data.features, 
                y = my_data.target, 
                cv = 5,  
                scoring = scorer, 
                n_jobs=-1
            )
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



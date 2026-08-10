import os
import streamlit as st
import graph_tool.all as gt  
import zipfile, tempfile
import pandas as pd 
from typing import Dict, List
from dataclasses import dataclass
from functools import reduce

import plotly.express as px 
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import json
from io import BytesIO
from tools.helpers import HelpMessage

tab_files_extensions = ["txt", "csv", "tsv", "xlsx"]


def nearZeroVar(df: pd.DataFrame, freqCut: float = 95/5, uniqueCut: int = 10):

    def nzv( col: pd.Series) -> bool:
        n_samples = len(col)
        n_unique = col.nunique()
        percentUnique = n_unique / n_samples * 100 

        if n_unique == 1 or percentUnique < uniqueCut:
            return True
        
        unique_set = col.value_counts()
        if len(unique_set) < 2:
            return True
        
        freqRatio = unique_set.iloc[0] / unique_set.iloc[1:].sum()

        if freqRatio >= freqCut:
            return True

        return False
    

    nzv_cols = [ col for col in df.columns if nzv( df[col] ) ]
    return nzv_cols


def generate_colors(n, input_palette = plt.cm.viridis):
    colors = input_palette([i / n for i in range(n)])
    return [tuple(int(c*255) for c in color[:3]) for color in colors]


def generate_hex_palette(n, input_palette = plt.cm.viridis):
    return [ f"#{r:02x}{g:02x}{b:02x}" for (r,g,b) in generate_colors(n, input_palette=input_palette)]


@st.cache_data
def load_prepdata(input_zipfile):
    if input_zipfile is not None:
        with zipfile.ZipFile( input_zipfile, 'r') as my_zip:
            jsons_to_read = ["covariates_and_features.json", "encodings.json"]
            feature_groups, encodings = [ json.load( my_zip.open( json_file ) ) for json_file in jsons_to_read ]
            
            if ( data_list := encodings.get("data") ):
                data_keys = [ "training_set", "test_set", "validation_set" ]
                id_train, id_test, id_valids = [ data_list.get( key ) for key in data_keys ]
                assert id_train and id_test, "Training and test set must be present in the encodings file"
                read_csv_kwargs = dict( sep='\t', index_col=0, header=0 )
                ## loading validation set if present
                try:
                    validation_sets = { 
                        id_val: pd.read_csv( my_zip.open(f"{id_val}.tsv"), **read_csv_kwargs) 
                            for id_val in id_valids }
                except TypeError as e:
                    validation_sets = dict()
                    
                ## loading training and test set
                data = dict( **validation_sets )

                for id_set in [ id_train, id_test ]:
                    try:
                        data[id_set] = pd.read_csv( my_zip.open(f"{id_set}.tsv"), **read_csv_kwargs )
                    except KeyError:
                        print(f"File {id_set}.tsv not found in the zip archive. Skipping...") #, file=sys.stderr)

                    #     validation_sets[ id_set ] = pd.read_csv( my_zip.open(f"{id_set}.tsv"), **read_csv_kwargs )


                #     ** validation_sets
            else:
                ## assuming we're reading prepared (but not preprocessed) data
                data = dict( 
                    omics_data = pd.read_csv( 
                        my_zip.open("omics_data.tsv"), sep='\t', index_col=0, header=0),
                )
            
            return data, feature_groups, encodings
        

def write_prepdata( 
        tsv_data: Dict[ str, pd.DataFrame ],
        feature_sets: Dict[str, List[str]], 
        encodings: Dict[str, Dict[str, int]], 
        zip_handle_append: str = None):
    
    def save_data_in_zip(zip_handle):
        """ Utility function to save data in a zip file """
        
        for name, df in tsv_data.items():
            if df.empty:
                st.warning(f"DataFrame {name} is empty. Skipping...")
            else:
                write_dataframe_in_zip( zip_handle, df, name, "tsv" )
        for json_file, json_content in json_files.items():
            zip_handle.writestr( json_file, json.dumps(json_content, indent=4) )

        if False and feature_graph is not None:
            write_graph_in_zip( zip_handle, feature_graph, "feature_graph.gt" )
            
        return zip_handle.filename 
    

    json_files = {
        "covariates_and_features.json": feature_sets,
        "encodings.json": encodings
    }

    if zip_handle_append is not None:
        with zipfile.ZipFile( zip_handle_append, "a" ) as my_zip:
            return save_data_in_zip( my_zip )
    else:
        with zipfile.ZipFile( tempfile.NamedTemporaryFile(suffix=".zip", delete=False), 'w') as my_zip:
            return save_data_in_zip( my_zip )


@dataclass
class SamplesMetadata:
    raw_data: pd.DataFrame
    filt_data: pd.DataFrame
    covariates: List

@dataclass
class SamplesOmicsData:
    raw_data: Dict[ str, pd.DataFrame ]
    filt_data: Dict[ str, pd.DataFrame ]
    omics_alias: Dict[ str, str ]


def write_plotly_in_zip(zip_name, figure: go.Figure, filename: str):
    with zipfile.ZipFile(zip_name, "a", compression=zipfile.ZIP_DEFLATED) as zip_out: 
        with tempfile.NamedTemporaryFile(prefix="plotly_figure__", suffix=".html" ) as tmp_file:
            figure.write_html( tmp_file.name )
            zip_out.write( tmp_file.name, f"{filename}.html" )


class OmicsDataHandler:
    #### TODO RIMUOVERE CLASSE: SOSTITUIRE CON FUNZIONE FILTERING E STOP
    def __init__(self, metadata: pd.DataFrame) -> None:
        self.__data = metadata.copy()
        self.__omics = None 

    def filtering(self, omics_data: pd.DataFrame, strat_by: str, thr_median: float = 0) -> List[str]:
        """ Return the list of columns (aka list of strings) s.t. pass the checks """

        omics_cols = omics_data.columns.tolist()
        self.__omics = pd.merge( self.__data[ strat_by ], omics_data, left_index=True, right_index=True )
        cols = list( filter( 
            lambda c: not self.__nearZeroVar( c ), 
            omics_cols ) 
        )
        
        if thr_median == 0:
            return cols 
        
        cols.insert( 0, strat_by )
        self.__omics = self.__omics[ cols ]

        median_per_class = {
            classvalue: self.__apply_median_threshold( subdf, thr_median ) #subdf.median( numeric_only = True )
                for classvalue, subdf in self.__omics.groupby( strat_by )
        } 
        medians_df, bitmasks = zip( *list( median_per_class.values() ) )
        ored_bitmask = reduce( lambda x, y: x | y, bitmasks )
        filtered = medians_df[0][ored_bitmask].index.tolist()
        return filtered


    def __nearZeroVar(self, col, uniqueCut: int = 10 ) -> bool:
        """  
        """

        data = self.__omics[ col ]
        n_samples = len(data)
        n_unique = data.nunique()
        percentUnique = n_unique / n_samples * 100 

        if n_unique == 1 or percentUnique < uniqueCut:
            return True
        
        
        return False 


    def __apply_median_threshold(self, df: pd.DataFrame, t: float):
        medians = df.median( numeric_only = True )
        bitmask = medians >= t
        return medians, bitmask
    

@st.cache_data
def load_dataframe( filename: str ):
    sep = "," if filename.endswith(".csv") else "\t"
    return pd.read_csv( filename, sep = sep, index_col = 0, header = 0)


@st.cache_data
def load_zip_archive( zip_filename: str ):
    with zipfile.ZipFile( zip_filename, 'r') as my_zip:
        dfs = {
            filename.replace(".tsv", ""): pd.read_csv(
                    my_zip.open(filename), sep='\t', index_col=0,header=0)
                for filename in my_zip.namelist()
        }
        return dfs 
    

def build_graph_zip( graph_collection: Dict[ str, gt.Graph ], tmp_filename: str = None) -> str: 
    if tmp_filename is None:
        tmp_filename = tempfile.NamedTemporaryFile(prefix="graph_archive__", suffix=".zip", delete=False).name

    with zipfile.ZipFile( tmp_filename, "w" ) as zip_out:
        for name, g in graph_collection.items():
            write_graph_in_zip( zip_out, g, f"graph_{name}.gt" )

    return zip_out.filename


def read_graph_from_zip( zip_handle, graph_filename: str ) -> gt.Graph:
    
    def wrapper_gt_load_graph( zip_file_handle, graph_filename: str ) -> gt.Graph:
        """ Utility function to read a graph from a zip file handle 
        looking whether the file is actually present in the archive and picking the compressed version (extension .xz) if it is present """
        actual_filename = [ 
            f for f in zip_file_handle.namelist() if f.startswith(graph_filename) ]
        if not actual_filename:
            raise ValueError(f"Graph file {graph_filename} not found in the zip archive.")
        
        actual_filename = actual_filename.pop()

        with tempfile.TemporaryDirectory() as dir_tmp:
            zip_file_handle.extract( actual_filename, path=dir_tmp )
            return gt.load_graph( f"{dir_tmp}/{actual_filename}" )


    match zip_handle: 
        case str() | BytesIO():
            with zipfile.ZipFile( zip_handle, "r" )  as _zip_handle:
                return wrapper_gt_load_graph( _zip_handle, graph_filename )
        case zipfile.ZipFile():
            return wrapper_gt_load_graph( zip_handle, graph_filename )

    raise ValueError(f"zip_handle must be either a file path (str) or a zipfile.ZipFile object. Now is {type(zip_handle)}")


def write_graph_in_zip( zip_out, g: gt.Graph, arcname: str ):
    gt_xz_extension = ".gt.xz"
    if arcname.endswith(".gt"):
        arcname = arcname.replace(".gt", gt_xz_extension )

    with tempfile.NamedTemporaryFile( suffix=gt_xz_extension ) as tmp:
        g.save( tmp.name ) #, fmt = "gt" )
        zip_out.write( tmp.name, arcname = arcname )


def build_tsv_archive( **kwargs ):
    with zipfile.ZipFile(tempfile.NamedTemporaryFile(suffix=".zip", delete=False), 'w') as zip_out:
        for name, df in kwargs.items():
            write_dataframe_in_zip( zip_out, df, name, "tsv" )
            
    return zip_out.filename


def write_dataframe_in_zip( zip_file, df: pd.DataFrame, out_filename: str, extension: str = "tsv" ):
    sep = "," if extension == "csv" else "\t"
    zip_file.writestr( 
        f"{out_filename}.{extension}", 
        df.to_csv( sep = sep, index = True, header = True )
    )


def write_figure_in_zip( zip_handle: zipfile.ZipFile, plotly_figure: go.Figure, filename: str, prefix: str = "figure", fmt: str ="pdf" ):
        with tempfile.NamedTemporaryFile(prefix=f"{prefix}__", suffix=f".{fmt}" ) as tmp_file:
            plotly_figure.write_image( tmp_file.name, format=fmt )
            zip_handle.write( tmp_file.name, f"{filename}.{fmt}" )

def write_figure_collection( zip_file: str, figure_set: Dict[str, go.Figure], prefix: str = "figure", fmt: str ="pdf" ) -> str:
    with tempfile.TemporaryDirectory(delete=False) as tmp_dir:
        with zipfile.ZipFile( os.path.join(tmp_dir, zip_file), "a", compression=zipfile.ZIP_DEFLATED) as zip_out:
            for name, fig in figure_set.items():
                write_figure_in_zip( zip_out, fig, name, prefix=prefix, fmt = fmt )
        return zip_out.filename
       
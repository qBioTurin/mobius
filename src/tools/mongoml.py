import streamlit as st
import pymongo, gridfs
import graph_tool.all as gt
import tempfile
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError 
from bson.objectid import ObjectId
from typing import List, Dict, Any, Tuple, Any, TypeAlias, Optional, Union
import logging
import os 

MONGO_URI = "mongodb://mongo:27017"
LOCAL_URI = "mongodb://localhost:27017"
YOUR_URI = MONGO_URI if os.path.exists("/.dockerenv") else LOCAL_URI


JsonDoc: TypeAlias = Dict[str, Any]
JsonCollection: TypeAlias = List[JsonDoc]


def init_mongo(uri: str = YOUR_URI, db_name: str = "mobius"):
    if ( mongo_uri := os.getenv("MONGO_URI") ) is not None:
        uri = mongo_uri
    else:
        uri = LOCAL_URI
    client = MongoClient(uri)
    client.drop_database(db_name)

@st.cache_resource
def connect_to_mongo(uri: str = YOUR_URI, db_name: str = None) -> Database:
    """
    Connect to a MongoDB database.

    :param uri: MongoDB connection URI
    :param db_name: Name of the database to connect to
    :return: pymongo.database.Database object
    """

    if ( mongo_uri := os.getenv("MONGO_URI") ) is not None:
        uri = mongo_uri
    else:
        uri = LOCAL_URI

    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    assert db_name is not None, "Database name must be provided"
    return client[db_name]


def encode_case_study_id(case_study_id: str, records: Union[JsonDoc, JsonCollection]) -> Union[JsonDoc, JsonCollection]:
    match records:
        case list():
            return [ { "case_study_id": case_study_id, **json_doc } for json_doc in records ]
        case dict():
            return { "case_study_id": case_study_id, **records }
        case None:
            return None
        case _:
            raise ValueError(f"Records must be a list of dictionaries or a single dictionary. Obtained: {type(records)}")


def insert_many(db: Database, collection_name: str, case_study_id: str, records: JsonCollection) -> List[Any]:
    """
    Insert multiple records into a MongoDB collection.

    :param db: pymongo.database.Database object
    :param collection_name: Name of the collection to insert records into
    :param records: List of records to insert (dictionaries)
    """
    try:
        collection = db[collection_name]
        result = collection.insert_many( encode_case_study_id(case_study_id, records) )
    except ServerSelectionTimeoutError as e:
        logging.error(f"Could not connect to MongoDB: {e}")
        raise RuntimeError(f"Could not connect to MongoDB: {e}")
    else:
        return result.inserted_ids
    
def insert_one(db: Database, collection_name: str, case_study_id: str, record: JsonDoc) -> Any:
    """
    Insert a single record into a MongoDB collection.

    :param db: pymongo.database.Database object
    :param collection_name: Name of the collection to insert the record into
    :param record: The record to insert (dictionary)
    :return: The inserted record's ID
    """
    try:
        collection = db[collection_name]
        result = collection.insert_one( encode_case_study_id(case_study_id, record) )
    except ServerSelectionTimeoutError as e:
        logging.error(f"Could not connect to MongoDB: {e}")
        raise RuntimeError(f"Could not connect to MongoDB: {e}")
    else:
        return result.inserted_id

def find(db: Database, collection_name: str, case_study_id: str, query: JsonDoc, projection: JsonDoc = None) -> JsonCollection:
    """
    Find records in a MongoDB collection.

    :param db: pymongo.database.Database object
    :param collection_name: Name of the collection to search
    :param query: Query to filter the records
    :return: List of found records
    """
    try:
        collection = db[collection_name]
        query = encode_case_study_id(case_study_id, query)
        projection = {**projection, "case_study_id": 0} if projection else None
        
        results = collection.find(query, projection)
    except ServerSelectionTimeoutError as e:
        logging.error(f"Could not connect to MongoDB: {e}")
        raise RuntimeError(f"Could not connect to MongoDB: {e}")
    else:
        return list(results)

def find_one(db: Database, collection_name: str, case_study_id: str, query: JsonDoc, projection: JsonDoc = None) -> JsonDoc:
    """
    Find a single record in a MongoDB collection.

    :param db: pymongo.database.Database object
    :param collection_name: Name of the collection to search
    :param query: Query to filter the records
    :return: The found record or None if not found
    """
    try:
        collection = db[collection_name]
        query = encode_case_study_id(case_study_id, query)
        projection = {**projection, "case_study_id": 0} if projection else None
        
        result = collection.find_one(query, projection)
    except ServerSelectionTimeoutError as e:
        logging.error(f"Could not connect to MongoDB: {e}")
        raise RuntimeError(f"Could not connect to MongoDB: {e}")
    else:
        return result
    

def delete_one(db: Database, collection_name: str, case_study_id: str, query: JsonDoc) -> int:
    """
    Delete a single record from a MongoDB collection.

    :param db: pymongo.database.Database object
    :param collection_name: Name of the collection to delete the record from
    :param query: Query to identify the record to delete
    :return: The number of records deleted (0 or 1)
    """
    try:
        collection = db[collection_name]
        result = collection.delete_one( encode_case_study_id(case_study_id, query) )
    except ServerSelectionTimeoutError as e:
        logging.error(f"Could not connect to MongoDB: {e}")
        raise RuntimeError(f"Could not connect to MongoDB: {e}")
    else:
        return result.deleted_count


def count_documents(db: Database, collection_name: str, case_study_id: str, query: JsonDoc) -> int:
    """
    Count documents in a MongoDB collection that match a query.

    :param db: pymongo.database.Database object
    :param collection_name: Name of the collection to count documents in
    :param query: Query to filter the documents
    :return: The count of matching documents
    """
    try:
        collection = db[collection_name]
        count = collection.count_documents( encode_case_study_id(case_study_id, query) )
    except ServerSelectionTimeoutError as e:
        logging.error(f"Could not connect to MongoDB: {e}")
        raise RuntimeError(f"Could not connect to MongoDB: {e}")
    else:
        return count


@st.cache_resource
def get_gridfs_bucket(chunk_size_bytes = 8 * 1024 * 1024) -> gridfs.GridFSBucket:

    db = connect_to_mongo(db_name="mobius")
    fs = gridfs.GridFSBucket(db, bucket_name="feature_nets", chunk_size_bytes=chunk_size_bytes)
    return fs 


def check_graph_in_mongo( g_id: str) -> bool:
    """ Check if a graph-tool Graph exists in MongoDB GridFS using the given identifier."""

    cursor = get_gridfs_bucket().find({"filename": g_id}, limit=1)
    return bool( list( cursor ) )


def search_graph_with_metadata( g_id: str, p_threshold: float):
    """ Retrieve a graph-tool Graph from MongoDB GridFS using the given identifier.
    Returns the graph with the best p-value equal or above the given threshold. """

    best_found = dict( _id = None, p = 1.0 )

    for record in get_gridfs_bucket().find({"filename": g_id}):
        if (curr_p := record.metadata.get("p")):
            if curr_p == p_threshold:
                best_found = dict( _id = record._id, p = curr_p )
                break
            elif curr_p > p_threshold and curr_p < best_found["p"]:
                best_found = dict( _id = record._id, p = curr_p )

    return ( best_found["_id"], best_found["p"] ) if best_found["_id"] else None 
        
        
def save_graph_in_mongo( g_id: str, g: gt.Graph, metadata: Dict[str, Any]):
    """ Save a graph-tool Graph into MongoDB GridFS with given identifier and metadata."""

    with tempfile.NamedTemporaryFile(suffix=".gt.xz", mode="wb", delete_on_close=False) as tmp_file:
        tmp_file.close()  # Close the file so graph-tool can write to it without issues 
        g.save(tmp_file.name)
        with open( tmp_file.name, "rb") as f:
            get_gridfs_bucket().upload_from_stream(
                filename=g_id,
                source=f,
                metadata=metadata )
            logging.critical(f"Uploaded graph {g_id} to MongoDB GridFS.")


def retrieve_graph_from_mongo__with_id( file_id: str ) -> gt.Graph:
    """ Retrieve a graph-tool Graph from MongoDB GridFS using the given file identifier."""


    with tempfile.NamedTemporaryFile(suffix=".gt.xz", mode="b+w", delete_on_close=False) as tmp_file:
        fs = get_gridfs_bucket()
        fs.download_to_stream(file_id=ObjectId(file_id), destination=tmp_file)
        tmp_file.close()
        return gt.load_graph(tmp_file.name)

def retrieve_graph_from_mongo( g_id: str) -> gt.Graph:
    """ Retrieve a graph-tool Graph from MongoDB GridFS using the given identifier."""

    if check_graph_in_mongo( g_id ):
        with tempfile.NamedTemporaryFile(suffix=".gt.xz", mode="b+w", delete_on_close=False) as tmp_file:
            get_gridfs_bucket().download_to_stream_by_name(filename=g_id, destination=tmp_file)
            tmp_file.close()
            return gt.load_graph(tmp_file.name)
        

def retrieve_graph_metadata_from_mongo( g_id: str) -> List[Dict[str, Any]]:
    """ Retrieve metadata of a graph-tool Graph from MongoDB GridFS using the given identifier."""
    
    file_doc = list( get_gridfs_bucket().find({"filename": g_id}) )
    g_metadata = [ { "_id": str(g_file._id), **g_file.metadata} for g_file in file_doc ]
    return g_metadata

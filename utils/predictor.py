import pandas as pd
import networkx as nx
from pathlib import Path
import json
import os

class Predictor():
    def __init__(self):
        # Data Ingestion Variables
        self.json_data = []  # the input data in list form
        self.json_folder = []  # the input data files if provided as a data directory

        # Graph Specific Variables
        self.nodes = []  # the list node types requested by the user
        self.edges = []  # the list of edge types requested by the user
        self.TrainGraph = nx.Graph()  # the Graph
        self.TestGraph=nx.Graph()
        self.keymap = []  # a list that holds the keys for each node
        self.edgestatus = True  # a boolean variable to ensure and edge can or cannot exist
        self.edgeattr = []  # the attributes of an edge in the form of node1--node2
        self.traindb = pd.DataFrame()  # a Pandas DataFrame with the graph data
        self.testdb = pd.DataFrame()  # a Pandas DataFrame with the graph data

        # Splunk Variables
        self.credentials = {}
        self.baseurl = ""
        self.splunk_search = {}

        # Node2Vec Variables
        self.walklen = 5  # walk length for node2vec random walks
        self.numwalk = 100  # number of walks
        self.p = 0.125  # return parameter
        self.q = 2.0  # in-out parameter

        # Link Prediction Variables
        self.threshold = 0.20  # the link prediction threshold, default is 0.2
        self.hits = []  # the edges that are below a certain threshold for link prediction
        self.edge_predictions = []  # list of probabilites for each test edge embedding

    # Ingest file for graph creation using JSON loads and Pandas DataFrames
    # Input: filepath - absolute location of the file path
    # Ouput: None
    def ingest_file(self, filepath,type):
        print(f"Beginning ingestion of data at file {filepath}")
        for l in open(filepath).readlines():
            self.json_data.append(json.loads(l))
        if type=="train":
            self.traindb = pd.DataFrame(self.json_data)
        else:
            self.testdb = pd.DataFrame(self.json_data)



    # Ingest folder containing files for graph creation
    # Input: folderpath - absolute location of the folder
    # Output: None
    def ingest_folder(self, folderpath,type):
        print(f"Beginning ingestion of data at folder {folderpath}")
        for (root, directory, files) in sorted(os.walk(folderpath)):
            if files:
                for f in files:
                    self.json_folder.append(u'%s' % os.path.join(str(root), str(f)))
        print(f"Done walking folder {folderpath}")
        for j in self.json_folder:
            self.ingest_file(j,type)

        # If the folder is really just a single file
        print(f'{folderpath} is a single file.')
        if len(self.json_folder) == 0:
            self.ingest_file(folderpath,type)

    def set_walk_length(self,input_walklen):
        self.walklen=input_walklen

    def set_walk_num(self,input_walknum):
        self.numwalk=input_walknum

    def set_threshold(self,input_threshold):
        self.threshold=input_threshold

    
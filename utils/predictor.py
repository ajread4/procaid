import pandas as pd
import networkx as nx
import json
import os
import numpy as np
import random
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.model_selection import train_test_split

class Predictor():
    def __init__(self,ret,in_out,numwalks,walklength,threshold_value,verbose):
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
        self.traindb = pd.DataFrame()  # a Pandas DataFrame with the graph data
        self.testdb = pd.DataFrame()  # a Pandas DataFrame with the graph data
        self.trainedgesattr=[] # training edge attributes
        self.testedgesattr=[] # test edge attributes

        # New Edge Variables
        self.affirmed_testedge_keymap = [] # keymap for test edges that are confirmed to exist in the training graph as well
        self.newedgeattr = []  # a list of attributes of edges that exist in the test graph but do not exist in the training graph, in the form of node1--node2
        self.affirmed_testedge = []  # list of test edges that do exist in the training graph

        '''
        # Splunk Variables - to be implemented later 
        self.credentials = {}
        self.baseurl = ""
        self.splunk_search = {}
        '''

        # Node2Vec Variables
        self.walklen = walklength  # walk length for node2vec random walks
        self.numwalk = numwalks  # number of walks
        self.p = ret  # return parameter
        self.q = in_out  # in-out parameter

        # Link Prediction Variables
        self.threshold = threshold_value  # the link prediction threshold, default is 0.2
        self.hits = []  # the edges that are below a certain threshold for link prediction
        self.edge_predictions = []  # list of probabilites for each test edge embedding

        # Graph Embedding Variables
        self.pos_edge_embeddings = []  # list of edge embeddings that exist in the training graph
        self.pos_edge_label = []  # labels corresponding to positive edge embeddings
        self.neg_edge_embeddings = []  # list of edge embeddings that don't exist in the training graph
        self.neg_edge_label = []  # labels of corresponding negative edges
        self.test_edge_embeddings = []  # list of edge embeddings in the testing graph
        self.test_edge_labels = []  # labels corresponding to the test edges

        # Utility variables
        self.verbose=verbose

    # Ingest file for graph creation using JSON loads and Pandas DataFrames
    # Input: filepath - absolute location of the file path
    # Ouput: None
    def ingest_file(self, filepath,type):
        if self.verbose:
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
        if self.verbose:
            print(f"Beginning ingestion of data at folder {folderpath}")
        for (root, directory, files) in sorted(os.walk(folderpath)):
            if files:
                for f in files:
                    self.json_folder.append(u'%s' % os.path.join(str(root), str(f)))
        if self.verbose:
            print(f"Done walking folder {folderpath}")
        for j in self.json_folder:
            self.ingest_file(j,type)
        if len(self.json_folder) == 0:
            # If the folder is really just a single file
            if self.verbose:
                print(f'{folderpath} is a single file.')
            self.ingest_file(folderpath,type)

    # Add weights to the nodes [Unused right now]
    # Input: input_Graph - NetworkX Graph to add node degrees to
    # Output: None
    def add_weights(self, input_Graph):
        for n in input_Graph.nodes():
            input_Graph.nodes[n]['weight'] = 1 / input_Graph.degree(n)
        if self.verbose:
            print(f'Done adding weights to node in graph')

    # Combine two entities
    # Input: first - first entity, second - second entity
    # Output: first+second - addition of the two input entities
    def combine(self, first, second):
        return first + second

    # Return the node attribute from an ID in keymap
    # Input: input_id - integer that represents the requested node ID
    # Output: the node that corresponds to that ID
    def return_nodeattr_fromid(self, input_id):
        return self.keymap[input_id]

    # Create a string that represents the node attributes placed together
    # Input: firstnode - attributes of the first node, secondnode - attributes of the second node
    # Output: string that respresnts the node attributes combined with a "--" between them
    def create_edge(self, firstnode, secondnode):
        return str(firstnode) + "--" + str(secondnode)

    # Return an array of node degrees
    # Input: input_Graph - NetworkX Graph to find node degrees for
    # Output: the node degrees for NetworkX type Graph, input_Graph
    def return_node_degrees(self, input_Graph):
        return np.array([input_Graph.degree(node) for node in list(input_Graph.nodes())])

    # Instantiate Node2Vec with the training graph, specificed, walk length, number of walks, p and q values
    # Input: None
    # Output: None
    def createnode2vec(self):
        self.node2vec = Node2Vec(self.TrainGraph, walk_length=self.walklen, num_walks=self.numwalk, p=self.p,q=self.q)  # https://arxiv.org/pdf/1607.00653.pdf
        if self.verbose:
            print(f'Node2Vec instantiated with p={self.p}, q={self.q}, walklength={self.walklen}, numwalks={self.numwalk}')
        self.model = self.node2vec.fit(window=2, sample=0.1)
        self.hadamard = HadamardEmbedder(keyed_vectors=self.model.wv)  # embed edges as the product of the node embeddings
        if self.verbose:
            print("Done with node2vec instantiation")

    # Instantiate the Logistic Regression classifier
    # Input: None
    # Output: None
    def createlogreg(self):
        self.edge_classifier = LogisticRegression(random_state=42, max_iter=100000)
        if self.verbose:
            print(f'Logistic regression classifier instantiated')

    # Embed edges using Hadamard embedding
    # Input: edges - list of edges formatted Node1--Node2, label - specific value to assign the edge (either 0 or 1)
    # Output: embedded_edges - list of edges embedded in the feature space, embedded_labels - the labels that correspond to each embedded edge
    def embed_edges(self, edges, label):
        embedded_edges = []
        embedded_labels = []
        for n1, n2 in edges:
            embedded_edges.append(self.hadamard[str(n1), str(n2)])
            embedded_labels.append(label)
        return embedded_edges, embedded_labels

    # Embed negative edges in the Graph
    # Input: None
    # Output: None
    def negative_embeds(self):
        normalized_probs = self.return_node_degrees(self.TrainGraph) / np.linalg.norm(
            self.return_node_degrees(self.TrainGraph), ord=1)
        while len(self.neg_edge_embeddings) < len(self.pos_edge_embeddings) * 10:
            self.random_edge = np.random.choice(list(self.TrainGraph.nodes()), 2, replace=False, p=normalized_probs)
            random_node1 = random.choice(list(self.TrainGraph.nodes()))
            random_node2 = random.choice(list(self.TrainGraph.nodes()))
            random_edge = self.create_edge_from_ids(random_node1, random_node2)
            random_edge_rev = self.create_edge_from_ids(random_node2, random_node1)
            if random_edge not in self.TrainGraph.edges() and random_edge_rev not in self.TrainGraph.edges():
                self.neg_edge_embeddings.append(self.hadamard[str(random_node1), str(random_node2)])
                self.neg_edge_label.append(0.)

    # Evalute the ROC and AP of the link prediction algorithm, not used yet
    # Input: None
    # Output: ROC and AP
    def method_evaluation(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.combine(self.pos_edge_embeddings, self.neg_edge_embeddings),
            self.combine(self.pos_edge_label, self.neg_edge_label), test_size=0.20, random_state=42)
        self.fitpredict(X_train, y_train)
        y_preds = self.edge_classifier.predict_proba(X_test)[:, 1]
        self.roc = roc_auc_score(y_test, y_preds)
        self.ap = average_precision_score(y_test, y_preds)
        return self.roc, self.ap

    # Fit the edge classifier using Logistic Regression
    # Input: Xlabels - the edge embeddings to train on, Ylabels - the labels for each edge embedding
    # Output: None
    def fitpredict(self, Xlabels, Ylabels):
        self.edge_classifier.fit(Xlabels, Ylabels)

    # Make predictions using the trained edge classifier
    # Input: embeds - embedded edges to test the edge classifier
    # Output: probabilities of existence for each embeds provided
    def fitpredictproba(self, embeds):
        # print('Predicting')
        return self.edge_classifier.predict_proba(embeds)[:, 1]

    # Embed edges from the Training graph, create negative edge embeddings, and fit the edge classifier
    # Input: None
    # Output: None
    def linkpredict(self):
        if self.verbose:
            print('Beginning Link Prediction')
        self.pos_edge_embeddings, self.pos_edge_label = self.embed_edges(self.TrainGraph.edges(), 1)
        self.negative_embeds()
        self.fitpredict(self.combine(self.pos_edge_embeddings, self.neg_edge_embeddings),
                        self.combine(self.pos_edge_label, self.neg_edge_label))
        if self.verbose:
            print('Beginning Anomalous Edge Detection')
        self.detect()

    # Conduct link prediction on the test edges
    # Input: None
    # Output: None
    def detect(self):
        self.newedge()  # check for new edges
        if self.verbose:
            print(f'Confirmed edges: {self.affirmed_testedge_keymap}')
        self.test_edge_embeddings, _ = self.embed_edges(self.affirmed_testedge_keymap, 0)
        self.edge_predictions = self.fitpredictproba(self.test_edge_embeddings)
        pred_dictionary = {self.affirmed_testedge_keymap[i]: self.edge_predictions[i] for i in
                           range(len(self.affirmed_testedge_keymap))}
        for k, v in pred_dictionary.items():
            self.threshold_analysis(k, v)

    # Return the edge predictions
    # Input: None
    # Output: edge_predictions
    def return_edge_predictions(self):
        return self.edge_predictions

    # Select only edges that are below the link prediction threshold
    # Input: input_edge - edge formatted by Node1--Node2, input_pred - decimal value between 0 and 1
    # Output: None
    def threshold_analysis(self, input_edge, input_pred):
        if input_pred < self.threshold:
            final_edge = self.create_edge(self.return_nodeattr_fromid(int(input_edge[0])),
                                          self.return_nodeattr_fromid(int(input_edge[1])))
            print(f'Edge: {final_edge} below threshold with link prediction:{input_pred}')

    # Ensure the request edges make sense, are not missing, and are not empty
    # Input: error_check - list of edges to check
    # Output: None
    def checkerrors_edge(self, error_check):
        for e in error_check:
            if str(e) == "" or str(e).split("-")[-1] == "":
                print(f'ERROR: Mission edge in input')
                quit()
            elif len(str(e).split("-")) == 1:
                print(f'ERROR: Missing node within edge in input')
                quit()

    # Ensure the nodes are columns in the Pandas DataFrame of data
    # Input: testnode - requested node to test
    # Output: None
    def checkerrors_node(self, testnode):
        # print(f'Checking the dataset and requested node to ensure compliance.')
        if testnode in list(self.traindb) and testnode in list(self.testdb):
            if self.verbose:
                print(f'Found {testnode} in dataset')
            return True
        else:
            print(f'ERROR: Did not find node {testnode} in dataset')
            return False

    # Run error checking mechanisms on input edges
    # Input: input_edges - list of input edges in the form of Node1--Node2,Node2--Node3
    # Output: None
    def feature_check(self,input_edges):
        edgelist=input_edges.split(",")
        self.checkerrors_edge(edgelist)
        for edge in edgelist:
            n1,n2=self.split_edge(edge)
            if (self.checkerrors_node(n1)) and (self.checkerrors_node(n2)):
                self.save_edge(edge)
                if self.verbose:
                    print(f'Feature check complete')
            else:
                quit()
    # Save the edges for the graph
    # Input: good_edge - edge to save
    # Output: None
    def save_edge(self, good_edge):
        self.edges.append(good_edge)

    # Split an edge based on "-"
    # Input: input_edge - input edge
    # Output: node1 - first node in edge, node2 - second node in edge
    def split_edge(self, input_edge):
        node1 = str(input_edge).split("--")[0]
        node2 = str(input_edge).split("--")[1]
        return node1, node2

    # Create edge from two indices of nodes
    # Input: n1 - first node id, n2 - second node id
    # Output: (n1,n2) - edge created with two node ids
    def create_edge_from_ids(self, n1, n2):
        return (n1, n2)

    # Find the attributes for the node
    # Input: r - row within the Pandas DataFrame of data, requested_node - node to find attributes for
    # Output: attributes of a specific column in the Pandas DataFrame
    def return_nodevalues(self, r, requested_node):
        return getattr(r, str(requested_node))

    # Retrieve the index of the attributes of a node if already created. If not already created, add the attributes to the keymap.
    # Input: node - the specific node, attr - attributes of the node
    # Output: keymap index of the node attributes
    def add_nodehandler(self, node, attr,input_graph):
        if attr not in self.keymap:
            input_graph.add_node(len(self.keymap), props={str(node): str(attr)})
            self.keymap.append(str(attr))
        return self.keymap.index(str(attr))

    # Create Train Graph using TrainDB
    # Input: None
    # Output: None
    def process_train(self):
        for row in self.traindb.itertuples():
            for e in self.edges:
                n1, n2 = self.split_edge(e)
                firstnode = self.return_nodevalues(row, str(n1))
                secondnode = self.return_nodevalues(row, str(n2))
                nodeid_firstnode = self.add_nodehandler(n1, str(firstnode),self.TrainGraph)
                nodeid_secondnode = self.add_nodehandler(n2, str(secondnode),self.TrainGraph)
                #print(f'Node: {firstnode} and id: {nodeid_firstnode}')
                #print(f'Node: {secondnode} and id: {nodeid_secondnode}')
                self.trainedgesattr.append(self.create_edge(firstnode, secondnode))
                self.TrainGraph.add_edge(nodeid_firstnode, nodeid_secondnode)
        if self.verbose:
            print(f"Number of Training Nodes: {len(self.TrainGraph.nodes())}")
            print(f"Number of Training Edges: {len(self.TrainGraph.edges())}")

    # Create Test Graph using TestDB
    # Input: None
    # Output: None
    def process_test(self):
        for row in self.testdb.itertuples():
            for e in self.edges:
                n1, n2 = self.split_edge(e)
                firstnode = self.return_nodevalues(row, str(n1))
                secondnode = self.return_nodevalues(row, str(n2))
                nodeid_firstnode = self.add_nodehandler(n1, str(firstnode),self.TestGraph)
                nodeid_secondnode = self.add_nodehandler(n2, str(secondnode),self.TestGraph)
                #print(f'Node: {firstnode} and id: {nodeid_firstnode}')
                #print(f'Node: {secondnode} and id: {nodeid_secondnode}')
                self.testedgesattr.append(self.create_edge(firstnode, secondnode))
                self.TestGraph.add_edge(nodeid_firstnode, nodeid_secondnode)
        if self.verbose:
            print(f"Number of Testing Nodes: {len(self.TestGraph.nodes())}")
            print(f"Number of Testing Edges: {len(self.TestGraph.edges())}")

    # Find the new edges
    # Input: None
    # Output: None
    def newedge(self):
        for edg in self.testedgesattr:
            if edg not in self.trainedgesattr and self.reverse_edge(edg) not in self.trainedgesattr:
                if self.verbose:
                    print(f'Test Edge: {edg} not found in training graph')
                self.newedgeattr.append(edg)
            else:
                self.affirmed_testedge.append(edg)
                n1, n2 = self.split_edge(edg)
                self.affirmed_testedge_keymap.append((self.get_keymap(str(n1)), self.get_keymap(str(n2))))

    # Flip the supplied edge
    # Input: forward_edge - the original edge
    # Output: the forward_edge flipped so that node1--node2 becomes node2--node1
    def reverse_edge(self, forward_edge):
        return list(forward_edge).reverse()

    # Return the ID of a node based on its attributes
    # Input: node - attribute of the node requested
    # Output: integer that represents the ID of the node requested
    def get_keymap(self, node):
        return self.keymap.index(node)

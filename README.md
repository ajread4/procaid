## ProcAID Stage One 

ProcAID stands for "Process Anomaly-based Intrusion Detection." The capability is made of two stages: 

1. Stage One: Unsupervised link prediction on a process creation log graph 
2. Stage Two: Inverse leadership and inverse density analysis  

Full explanation of ProcAID can be found here [ProcAID](https://www.proquest.com/openview/e4ce5ff777fc5943a8b4624677b3cad1/1.pdf?pq-origsite=gscholar&cbl=18750&diss=y).

For convenience and use-case purposes, this repository contains only Stage One of ProcAID. Stage Two of ProcAID will be located in a seperate repository.

## Install
```
git clone https://github.com/ajread4/procaid_stage1.git
cd procaid_stage1
pip3 install -r requirements.txt
```
## Usage
```
$ python3 linkpredict.py -h
usage: linkpredict.py [-h] [-numwalk int] [-walklen int] [-threshold int] [-ret int] [-in_out int] [-v] [-train traindata] [-test testdata] edges

linkpredict - a capability to conduct unsupervised link prediction on information security data.

positional arguments:
  edges             specify the edges for the graph to create in the form of NodeX--NodeY,NodeY--NodeZ,...

optional arguments:
  -h, --help        show this help message and exit
  -numwalk int      specify the number of walks for Node2Vec, default is 100
  -walklen int      specify the walk length for Node2Vec, default is 5
  -threshold int    specify the link prediction threshold, default is 0.2
  -ret int          specify the return parameter (p) for Node2Vec, default is 0.125
  -in_out int       specify the in out parameter (q) for Node2Vec, default is 2.0
  -v, --verbose     run linkpredict in verbose mode

FILE OR FOLDER INPUT ARGUMENTS:
  -train traindata  specify the location of the json formatted training data, can be file or folder
  -test testdata    specify the location of the json formatted testing data, can be file or folder
```

## Example Usage

1. Conduct unsupervised link prediction on host logs where the edges are between ```UserName``` and ```EventID```. 
```
$ python3 linkpredict.py -train train_data.json -test test_data.json UserName--EventID
Computing transition probabilities: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 21595.39it/s]
Generating walks (CPU: 1): 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 10898.26it/s]
Edge: Comp067947$--4688 below threshold with link prediction:0.1120337033179818
Edge: Comp067947$--4634 below threshold with link prediction:0.10267681339152787
Edge: Comp067947$--4624 below threshold with link prediction:0.10192561404736276
Edge: Comp447172$--4624 below threshold with link prediction:0.15070581500844746
Edge: User529192--4624 below threshold with link prediction:0.1497220737896318
Edge: User529192--4634 below threshold with link prediction:0.14582165695484503
Edge: Comp715254$--4634 below threshold with link prediction:0.16846520887399277
```
2. Conduct unsupervised link prediction on host logs where the edges are between ```EventID``` and ```LogHost```, and ```UserName``` and ```DomainName``` in verbose mode. 
```
$ python3 linkpredict.py -train train_data.json -test test_data.json EventID--LogHost,UserName--DomainName -v
Beginning ingestion of data at folder ../octopus_graph/tests/json_files/xaa.json
Done walking folder ../octopus_graph/tests/json_files/xaa.json
../octopus_graph/tests/json_files/xaa.json is a single file.
Beginning ingestion of data at file ../octopus_graph/tests/json_files/xaa.json
Beginning ingestion of data at folder ../octopus_graph/tests/json_files/xab.json
Done walking folder ../octopus_graph/tests/json_files/xab.json
../octopus_graph/tests/json_files/xab.json is a single file.
Beginning ingestion of data at file ../octopus_graph/tests/json_files/xab.json
Found EventID in dataset
Found LogHost in dataset
Feature check complete
Found UserName in dataset
Found DomainName in dataset
Feature check complete
Number of Training Nodes: 12
Number of Training Edges: 9
Number of Testing Nodes: 14
Number of Testing Edges: 11
Computing transition probabilities: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 24256.22it/s]
Generating walks (CPU: 1): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 8104.15it/s]
Node2Vec instantiated with p=0.125, q=2.0, walklength=5, numwalks=100
Done with node2vec instantiation
Logistic regression classifier instantiated
Beginning Link Prediction
Beginning Anomalous Edge Detection
Test Edge: User124533--Domain001 not found in training graph
Test Edge: Comp649388$--Domain001 not found in training graph
Confirmed edges: [(0, 1), (2, 3), (4, 5), (2, 3), (6, 5), (2, 3), (6, 5), (7, 3), (8, 5), (9, 3), (6, 5), (10, 3), (4, 5), (10, 3), (4, 5), (11, 3), (6, 5), (10, 3), (4, 5), (10, 3), (6, 5), (10, 3), (4, 5), (10, 3), (6, 5), (10, 3), (6, 5), (6, 5), (10, 3), (4, 5), (10, 3), (6, 5), (10, 3), (4, 5), (10, 3), (6, 5), (10, 3), (4, 5), (10, 3), (0, 1), (4, 5), (2, 3), (6, 5), (2, 3), (6, 5), (7, 3), (8, 5), (9, 3), (6, 5), (10, 3), (4, 5), (10, 3), (4, 5), (11, 3), (6, 5), (10, 3), (4, 5), (10, 3), (6, 5), (10, 3)]
Edge: 4688--Comp649388 below threshold with link prediction:0.17901565968693298
Edge: Comp067947$--Domain001 below threshold with link prediction:0.17345416850670756
Edge: 4634--ActiveDirectory below threshold with link prediction:0.16135788234850051
Edge: 4624--ActiveDirectory below threshold with link prediction:0.17153132747530125
Edge: Comp447172$--Domain001 below threshold with link prediction:0.18359886499072067
Edge: 4672--ActiveDirectory below threshold with link prediction:0.15038469704623472
Edge: Comp916004$--Domain001 below threshold with link prediction:0.17698751619104305
Edge: User529192--Domain001 below threshold with link prediction:0.18670064192212737
```
3. Conduct unsupervised link prediction on host logs where the edges are between ```LogHost``` and ```DomainName```, and ```DomainName``` and ```UserName```, with a link prediction threshold of ```0.3``` and where Node2Vec does```1000``` random walks to learn the graph.
```
$ python3 linkpredict.py -train train_data.json -test test_data.json LogHost--DomainName,DomainName--UserName -numwalk 1000 -threshold 0.3
Computing transition probabilities: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 14382.53it/s]
Generating walks (CPU: 1): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:00<00:00, 12671.69it/s]
Edge: Comp649388--Domain001 below threshold with link prediction:0.14623839372876862
Edge: Domain001--Comp067947$ below threshold with link prediction:0.15351358767086112
Edge: ActiveDirectory--Domain001 below threshold with link prediction:0.14794925818583163
Edge: Domain001--Comp447172$ below threshold with link prediction:0.15363190325064735
Edge: Domain001--Comp916004$ below threshold with link prediction:0.1502812291174214
Edge: Domain001--User529192 below threshold with link prediction:0.15083770391715126
Edge: Domain001--Comp715254$ below threshold with link prediction:0.14616425824667778
```
## File and Directory Information 

- ```linkpredict.py``` 
  - This is the main Python script that runs the unsupervised anomaly detection algorithm. 
- ```utils```
  - This directory holds the ```Predictor``` class within ```predictor.py``` which is used by ```linkpredict.py``` to create graphs, analyze them, and conduct link prediction.
- ```requirements.txt```
  - This contains the requirements for running this script. 

## Dependencies 
This script uses Python 3.8.10 for operation. It has not been fully tested on other versions of Python. 

## Publication
The full ProcAID publication is located here: [ProcAID](https://www.proquest.com/openview/e4ce5ff777fc5943a8b4624677b3cad1/1.pdf?pq-origsite=gscholar&cbl=18750&diss=y)

## Author
All of the code was written by me, AJ Read, for my thesis at GW. 

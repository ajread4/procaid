## ProcAID Stage One 

ProcAID stands for "Process Anomaly-based Intrusion Detection." The capability is made of two stages: 

1. Stage One: Unsupervised link prediction on a process creation log graph 
2. Stage Two: Inverse leadership and inverse density analysis  

The full explanation of ProcAID can be found here [ProcAID](https://www.proquest.com/openview/e4ce5ff777fc5943a8b4624677b3cad1/1.pdf?pq-origsite=gscholar&cbl=18750&diss=y).

For convenience and use-case purposes, this repository contains the framework and algorithm to run Stage One of ProcAID with varying nodes, edges, thresholds, or parameters.

Stage Two of ProcAID can be found [here](https://github.com/ajread4/procaid_stage2).

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
Computing transition probabilities: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 21147.75it/s]
Generating walks (CPU: 1): 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 10361.93it/s]
Edge: Comp067947$--4688 below threshold with link prediction: 0.14468678519025702
Edge: Comp067947$--4634 below threshold with link prediction: 0.11758853401959393
Edge: Comp067947$--4624 below threshold with link prediction: 0.10847653073588186
Edge: Comp447172$--4624 below threshold with link prediction: 0.16362865738986088
Edge: Comp916004$--4672 below threshold with link prediction: 0.12773455618206841
Edge: User529192--4624 below threshold with link prediction: 0.1482870988505544
Edge: User529192--4634 below threshold with link prediction: 0.13785672081109693
Edge: Comp715254$--4634 below threshold with link prediction: 0.14500785386691073
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
Computing transition probabilities: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 19730.16it/s]
Generating walks (CPU: 1): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 8590.13it/s]
Node2Vec instantiated with p=0.125, q=2.0, walklength=5, numwalks=100
Done with node2vec instantiation
Logistic regression classifier instantiated
Beginning Link Prediction
Beginning Anomalous Edge Detection
Test Edge: User124533--Domain001 not found in training graph
Test Edge: Comp649388$--Domain001 not found in training graph
Confirmed edges: [(0, 1), (2, 3), (4, 5), (2, 3), (6, 5), (2, 3), (6, 5), (7, 3), (8, 5), (9, 3), (6, 5), (10, 3), (4, 5), (10, 3), (4, 5), (11, 3), (6, 5), (10, 3), (4, 5), (10, 3), (6, 5), (10, 3), (4, 5), (10, 3), (6, 5), (10, 3), (6, 5), (6, 5), (10, 3), (4, 5), (10, 3), (6, 5), (10, 3), (4, 5), (10, 3), (6, 5), (10, 3), (4, 5), (10, 3), (0, 1), (4, 5), (2, 3), (6, 5), (2, 3), (6, 5), (7, 3), (8, 5), (9, 3), (6, 5), (10, 3), (4, 5), (10, 3), (4, 5), (11, 3), (6, 5), (10, 3), (4, 5), (10, 3), (6, 5), (10, 3)]
Edge: 4688--Comp649388 below threshold with link prediction: 0.14436852254332727
Edge: Comp067947$--Domain001 below threshold with link prediction: 0.17076364809377953
Edge: 4634--ActiveDirectory below threshold with link prediction: 0.1670822829444345
Edge: 4624--ActiveDirectory below threshold with link prediction: 0.16492791138344146
Edge: Comp447172$--Domain001 below threshold with link prediction: 0.16929482039785573
Edge: 4672--ActiveDirectory below threshold with link prediction: 0.1716846109743066
Edge: Comp916004$--Domain001 below threshold with link prediction: 0.1563144122549408
Edge: User529192--Domain001 below threshold with link prediction: 0.16312944749064529
Edge: Comp715254$--Domain001 below threshold with link prediction: 0.16773827716922093
```
3. Conduct unsupervised link prediction on host logs where the edges are between ```LogHost``` and ```DomainName```, and ```DomainName``` and ```UserName```, with a link prediction threshold of ```0.3``` and where Node2Vec does```1000``` random walks to learn the graph.
```
$ python3 linkpredict.py -train train_data.json -test test_data.json LogHost--DomainName,DomainName--UserName -numwalk 1000 -threshold 0.3
Computing transition probabilities: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 12441.39it/s]
Generating walks (CPU: 1): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:00<00:00, 12549.53it/s]
Edge: Comp649388--Domain001 below threshold with link prediction: 0.1545750896334678
Edge: Domain001--Comp067947$ below threshold with link prediction: 0.15869929853152523
Edge: ActiveDirectory--Domain001 below threshold with link prediction: 0.15460793703248177
Edge: Domain001--Comp447172$ below threshold with link prediction: 0.15618177292848773
Edge: Domain001--Comp916004$ below threshold with link prediction: 0.1516173728908642
Edge: Domain001--User529192 below threshold with link prediction: 0.1533558378087372
Edge: Domain001--Comp715254$ below threshold with link prediction: 0.15618532668322663
```

## ProcAID Stage One Parameters 
To fully emulate Stage One of ProcAID with Process Creation logs, use the following edges and parameters (some of which are set by default): 
```
Edges: User--ProcessPath,ProcessPath--ParentProcessPath,User--ParentProcessPath
ret=0.125
in_out=2.0
numwalk=100
walklen=5
threshold=0.2
```
## File and Directory Information 

- ```linkpredict.py``` 
  - This is the main Python script that runs the unsupervised anomaly detection algorithm. 
- ```utils```
  - This directory contains```predictor.py``` and the ```Predictor``` class which is used by ```linkpredict.py``` to create graphs, analyze them, and conduct link prediction.
- ```requirements.txt```
  - This file contains the requirements for running this script. 

## Dependencies 
This script uses Python 3.8.10 for operation. It has not been fully tested on other versions of Python.

## Planned Features
There are a few features to be added in the future: 

- Output anomalous edges in various formats to include file, database, etc
- Full Splunk integration 
- Options for link prediction classifier

## Publication
The full ProcAID publication is located here: [ProcAID](https://www.proquest.com/openview/e4ce5ff777fc5943a8b4624677b3cad1/1.pdf?pq-origsite=gscholar&cbl=18750&diss=y)

## Author
All of the code was written by me, AJ Read, for my thesis at GW. 
- Twitter: [ajread3](https://twitter.com/ajread3)
- Github: [ajread4](https://github.com/ajread4)
- LinkedIn: [Austin Read](https://www.linkedin.com/in/austin-read-88953b189/)

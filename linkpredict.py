import argparse
from utils.predictor import Predictor

def main():
	"""
	Main function for linkpredict
	"""
	parser = argparse.ArgumentParser(description='linkpredict - a capability to conduct unsupervised link prediction on information security data.')
	parser.add_argument('edges', action='store', help='specify the edges for the graph to create in the form of NodeX--NodeY,NodeY--NodeZ,...',metavar='edges')
	parser.add_argument('-numwalk',type=int,action='store',help='specify the number of walks for Node2Vec, default is 100',metavar='int',default=100)
	parser.add_argument('-walklen',type=int,action='store',help='specify the walk length for Node2Vec, default is 5',metavar='int',default=5)
	parser.add_argument('-threshold',type=float,action='store',help='specify the link prediction threshold, default is 0.2',metavar='int',default=0.2)
	parser.add_argument('-ret', type=float, action='store',help='specify the return parameter (p) for Node2Vec, default is 0.125', metavar='int', default=0.125)
	parser.add_argument('-in_out', type=float, action='store',help='specify the in out parameter (q) for Node2Vec, default is 2.0', metavar='int',default=2.0)
	parser.add_argument("-v","--verbose", help="run linkpredict in verbose mode",action="store_true")
	#parser.add_argument("-o","--output",type=str,action="store",help="specify file location to output results",metavar="file") # to be added

	SimpleIngestType=parser.add_argument_group("FILE OR FOLDER INPUT ARGUMENTS")
	SimpleIngestType.add_argument('-train',dest='train',action='store',help='specify the location of the json formatted training data, can be file or folder',metavar='traindata')
	SimpleIngestType.add_argument('-test',dest='test',action='store',help='specify the location of the json formatted testing data, can be file or folder',metavar='testdata')

	'''
	# To be added 
	SplunkIngestType =parser.add_argument_group('SPLUNK INPUT ARGUMENTS')
	SplunkIngestType.add_argument('-url', dest='baseurl', type=str, action='store',
								  help='specify the url for the Splunk REST API (normally port 8089)',
								  metavar='https://[SPLUNK IP]:port')
	SplunkIngestType.add_argument('-user', dest='user_name', type=str, action='store',
								  help='specify the username for the splunk user', metavar='username')
	SplunkIngestType.add_argument('-pass', dest='user_pass', type=str, action='store',
								  help='specify the password for the splunk user', metavar='password')
	SplunkIngestType.add_argument('-trainquery', dest='trainquery', type=str, action='store',
								  help='specify splunk query for training time, use \'earliest=\' and \'latest=\' to specify timeframe',
								  metavar='trainquery')
	SplunkIngestType.add_argument('-testquery', dest='testquery', type=str, action='store',
								  help='specify splunk query for testing time, use \'earliest=\' and \'latest=\' to specify timeframe',
								  metavar='testquery')
	'''
	args=parser.parse_args()

	# Instantiate the Predictor class
	graph=Predictor(args.ret,args.in_out,args.numwalk,args.walklen,args.threshold,args.verbose)

	# Ingest data
	graph.ingest_folder(args.train,"train")
	graph.ingest_folder(args.test,"test")

	# Check for input errors
	graph.feature_check(args.edges)

	# Process and Create Graphs
	graph.process_train()
	graph.process_test()

	# Conduct link prediction
	graph.createnode2vec()
	graph.createlogreg()
	graph.linkpredict()

if __name__=="__main__":
	try:
		main()
	except Exception as err:
		print(repr(err))

import argparse
from utils.predictor import Predictor

def main():
	"""
	Main function for linkpredict
	"""
	parser = argparse.ArgumentParser(description='linkpredict - a capability to conduct unsupervised link prediction on information security data.')
	parser.add_argument('edges', action='store', help='specify the edges for the graph to create',metavar='edges')

	SimpleIngestType=parser.add_argument_group("FILE OR FOLDER INPUT ARGUMENTS")
	SimpleIngestType.add_argument('-train',dest='train',action='store',help='specify the location of the json formatted training data',metavar='traindata')
	SimpleIngestType.add_argument('-test',dest='test',action='store',help='specify the location of the json formatted testing data',metavar='testdata')

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
	args=parser.parse_args()

	print(args.edges)
	graph=Predictor()
	graph.ingest_file(args.train,"train")
	graph.ingest_file(args.test,"test")
	graph.feature_check(args.edges)
	graph.process_train()
	graph.process_test()
	graph.createnode2vec()
	graph.createlogreg()
	graph.linkpredict()
if __name__=="__main__":
	try:
		main()
	except Exception as err:
		print(repr(err))

# Script to build and visualize a knn-graph from test data

from sklearn.neighbors import KNeighborsTransformer
import networkx

from helpers.readers import Reader

reader = Reader('/Users/dianamindroc/Desktop/PhD/Data/suresim_simulations')
reader.get_folders('crossing')
reader.get_files_from_folder(0)
reader.set_file(reader.files[0])
reader.read_txt(4)
data = reader.extract_xyz([0,1,2])

transformer = KNeighborsTransformer(n_neighbors=10, mode = 'distance')
transformer.fit_transform(data)

graph = transformer.kneighbors_graph()
graph_arr = graph.toarray()

xgraph = networkx.from_scipy_sparse_matrix(graph)
nxgraph = networkx.davis_southern_women_graph()


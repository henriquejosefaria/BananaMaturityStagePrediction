from math import sqrt
import numpy.linalg as la
import numpy as np
import os
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
import matplotlib.pyplot as plt
import warnings
from mpl_toolkits.mplot3d import Axes3D
import time
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq, kmeans, whiten
from scipy.spatial.distance import mahalanobis, euclidean
warnings.warn("deprecated", DeprecationWarning)
warnings.simplefilter("ignore")
#-------------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------------#

#                                                     DATA LOADING                                                  #

#-------------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------------#


start = time.time()
'''
Guarda um array com os caminhos de todos os canais por caso de treino

channels_cases = [0] * 4
for i in range(4):
    sentence1 = "Channel"
    dirStr = 'Training/Case_'+str(i+1)+'/'
    channels_cases[i] = [x[0] for x in os.walk(dirStr) if sentence1 in x[0]]
'''

'''
Guarda um array com os caminhos dos canais  103, 104 e 108 por caso de treino
'''
channels_cases = [0] * 4
for i in range(4):
    sentence1 = "Channel_103"
    sentence2 = "Channel_104"
    sentence3 = "Channel_108"
    dirStr = 'Training/Case_'+str(i+1)+'/'
    channels_cases[i] = [x[0] for x in os.walk(dirStr) if sentence1 in x[0] or sentence2 in x[0] or sentence3 in x[0]]


'''
carrega o dataset .npz de cada canal de treino
'''
train_cases = [0] * len(channels_cases)
for i in range(len(channels_cases)):
    #caso i
    train_cases[i] = []
    for channel in channels_cases[i]:
        data = np.load(channel + '/temp_cam_data.npz')
        train_cases[i].append(data['arr_0'])





'''
Guarda um array com os caminhos de todos os canais por caso de teste

channels_cases2 = [0]
sentence = "Channel"
dirStr = 'Testing/Case_5/'
channels_cases2[0] = [x[0] for x in os.walk(dirStr) if len(x[0]) < 27 and len(x[0]) > 20]
'''

'''
Guarda um array com os caminhos dos canais escolhidos por caso de teste
'''
channels_cases2 = [0]
dirStr = 'Testing/Case_5/'
sentence1 = "Channel_103"
sentence2 = "Channel_104"
sentence3 = "Channel_108"
channels_cases2[0] = [x[0] for x in os.walk(dirStr) if (sentence1 in x[0] or sentence2 in x[0] or sentence3 in x[0]) and len(x[0]) < 27 and len(x[0]) > 20]



'''
carrega o dataset .npz de cada canal de teste
'''
test_cases = [[]] * len(channels_cases)
for channel in channels_cases2[0]:
    data = np.load(channel + '/temp_cam_data.npz')
    test_cases[0].append(data['arr_0'])



'''
Primeiro numpy array guardado de treino

caso 1 channel_0

'''
#print("Train data: case_1, Channel_0 = \n",train_cases[0][0])
train_cases = np.array(train_cases)
print(train_cases.shape)


#Primeiro numpy array guardado de teste

#print("Test data: case_1, Channel_0 = \n",test_cases[0][0])

print("------------------        DATASET  LOADED       ----------------------\n\n")
end = time.time()
print("Took ",end - start, " seconds to collect the data!\n\n")


#-------------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------------#

#                                                        PCA                                                        #

#-------------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------------#


def calculate_wcss(data):
    wcss = []
    for n in range(2, 21):
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(X=data)
        wcss.append(kmeans.inertia_)
    return wcss

def optimal_number_of_clusters(wcss):
    x1, y1 = 2, wcss[0]
    x2, y2 = 20, wcss[len(wcss)-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    
    return distances.index(max(distances)) + 2

start = time.time()
'''
Número de componentes a manter no PCA
'''
pca = PCA(n_components=11)

'''
Esta função coloca por ordem o que antes eram os ts {t1,...,t11} => {t1,t11,t2,t10,t3,t9,...,t6}
'''
def organize(array,length):
	new_array = []
	for i in range(length//2):
		new_array.append(array[i])
		new_array.append(array[length-i-1])
	new_array.append(array[length//2 + 1])
	return new_array
'''
Criação de etiquetas para identificar estágios
Implementaççao do PCA sobre cada canal de cada caso
'''
debug = 0
organized = 1
print("-*-  DATA WILL BE ORGANIZED  -*- "*4)
def data_parser():
	print("\n\n-------------         COLLECTING TRAIN DATA          -----------------")
	photos = []
	new_train_cases = []
	for i in range(len(train_cases)):
		print("------------------ COLLECTING DATA FROM CASE ",i+1, "----------------------")
		for j in range(len(train_cases[i])):
			if debug == 1:
				soma = 0
				mat = []
				dif = (train_cases[i][j][11] - train_cases[i][j][0])
				dif = np.array(dif) #(216,12)
				print(dif.shape)
				for x in dif:
					mat = np.concatenate((mat, x))
				#print(np.array(mat).shape)
				#soma = [soma += x for x in mat]
				#print(soma)
				for x in mat:
					soma += x
			#12ª figura eliminada
			X = train_cases[i][j].reshape((12,-1))[:-1]
			if organized == 1:
				X = organize(X,len(X))
			pca.fit(X)
			transformation = pca.transform(X)
			new_train_cases.append(transformation)
			# optimal kernels
			photos.append(transformation)

	print("-------------         TRAIN DATA COLLECTED          ------------------")
	print("\n\n-------------         COLLECTING TEST DATA          ------------------")

	new_test_cases = []
	for i in range(len(test_cases[0])):
		X = test_cases[0][i].reshape((12,-1))[:-1]
		if organized == 1:
			X = organize(X,len(X))
		pca.fit(X)
		new_test_cases.append(pca.transform(X))

	print("-------------         TEST DATA COLLECTED           ------------------")
	return new_train_cases, new_test_cases, photos

new_train_cases, new_test_cases, photos = data_parser()

def plotting():
	size = len(new_train_cases)/4

	plt.figure(1)
	#plt.figure(1).add_subplot(111, projection='3d')
	plt.xlabel("X-axis")
	plt.ylabel("Y-axis")
	plt.title("The train graph")

	'''
	Adding dots to the plot
	'''
	for i in range(len(new_train_cases)):
		if i < size:
			plt.plot(new_train_cases[i][0],new_train_cases[i][1], 'bo')
			#plt.annotate(labeled_train_cases[i],(new_train_cases[i][0],new_train_cases[i][1]),(new_train_cases[i][0]-0.07,new_train_cases[i][1]+0.05))
		elif i < 2*size:
			plt.plot(new_train_cases[i][0],new_train_cases[i][1], 'ro')
			#plt.annotate(labeled_train_cases[i],(new_train_cases[i][0],new_train_cases[i][1]),(new_train_cases[i][0],new_train_cases[i][1]+0.05))
		elif i < 3*size:
			plt.plot(new_train_cases[i][0],new_train_cases[i][1], 'go')
			#plt.annotate(labeled_train_cases[i],(new_train_cases[i][0],new_train_cases[i][1]),(new_train_cases[i][0],new_train_cases[i][1]+0.05))
		else:
			plt.plot(new_train_cases[i][0],new_train_cases[i][1], 'yo')
			#plt.annotate(labeled_train_cases[i],(new_train_cases[i][0],new_train_cases[i][1]),(new_train_cases[i][0],new_train_cases[i][1]+0.05))

	plt.figure(2)
	#plt.figure(2).add_subplot(111, projection='3d')
	plt.xlabel("X-axis")
	plt.ylabel("Y-axis")
	plt.title("The test graph")
	'''
	Adding dots to the plot
	'''
	for i in range(len(new_test_cases)):
		if i < size:
			plt.plot(new_test_cases[i][0],new_test_cases[i][1], 'bo')
			#plt.annotate(labeled_train_cases[i],(new_train_cases[i][0],new_train_cases[i][1]),(new_train_cases[i][0]-0.07,new_train_cases[i][1]+0.05))
		elif i < 2*size:
			plt.plot(new_test_cases[i][0],new_test_cases[i][1], 'ro')
			#plt.annotate(labeled_train_cases[i],(new_train_cases[i][0],new_train_cases[i][1]),(new_train_cases[i][0],new_train_cases[i][1]+0.05))
		elif i < 3*size:
			plt.plot(new_test_cases[i][0],new_test_cases[i][1], 'go')
			#plt.annotate(labeled_train_cases[i],(new_train_cases[i][0],new_train_cases[i][1]),(new_train_cases[i][0],new_train_cases[i][1]+0.05))
		else:
			plt.plot(new_test_cases[i][0],new_test_cases[i][1], 'yo')
			#plt.annotate(labeled_train_cases[i],(new_train_cases[i][0],new_train_cases[i][1]),(new_train_cases[i][0],new_train_cases[i][1]+0.05))

	print("------------------          PLOTTING           -----------------------")
	end = time.time()
	print("Took ",end - start, " seconds to perform PCA to 12 componets and plotting them!")

#plotting()

# calculando a soma dos quadrados para as 19 quantidade de clusters
print('\n-*- EXTRA: VAMOS DESCOBRIR O NÚMERO IDEAL DE KERNELS E VERIFICAR QUAIS OS TEMPOS DE CASOS DIFERENTES COM MENOS DISTÂNCIA EUCLIDEANA -*-\n'*2)

def get_optimal_clusters(photos):
	sum_of_squares = calculate_wcss(np.array(photos).reshape((-1,11)))
	optimal_clusters = optimal_number_of_clusters(sum_of_squares)
	print("OPTIMAL NUMBER OF CLUSTERS => ",optimal_clusters)
	plt.xlabel("Número de clusters")
	plt.ylabel("Soma dos quadrados intra-cluster")
	plt.title("Gráfico dos Kernels")
	plt.plot([ i for i in range(len(sum_of_squares)) ],
		[ sum_of_squares[i] for i in range(len(sum_of_squares)) ])
	#plt.show()
	return optimal_clusters

optimal_clusters = get_optimal_clusters(photos)

print("\n    -*- VAMOS DESCOBRIR PARA CADA DADO CASO E CANAL QUAIS OS Ts DOS RESTANTES CASOS E MESMO CANAL QUE MAIS SE ASSEMELHAM -*-\n"*2)

photos = np.asarray(photos).reshape(132, 11)
covar_mat = np.matmul(np.transpose(photos), photos)
inv_cov = la.pinv(covar_mat)


#Passa array2D com n_clusters para array2D com n_clusters-1
#Devolve novo array2D com menos 1 dimensão e posição a que o elemento removido se juntou
def redefine_clusters(array2D,current_clusters,arrayClusters,arrayLabels):
	indexes = []
	values = []
	count = 0

	#Colecionamos os menores valores e respetivos indices
	for array in array2D:
		i = min(array)
		pos_i = array.index(i)
		indexes.append(pos_i)
		values.append(i)
		count += 1

	#Indexes -> guarda em cada posição o t (t1,...,t12) a que corresponde o menor valor para cada array
	#Values  -> guarda o menor valor de entre todos os t (t1,...,t12) para cada array

	#Escolhemos o menor valor e obtemos os ts que vão ser fundidos
	minimum_value = min(values)
	chosen_merge1 = values.index(minimum_value)    # Index de menor valor            (linha)
	chosen_merge2 = indexes[chosen_merge1]         # t correspondente ao menor valor (coluna)
	minor_index = min(chosen_merge1,chosen_merge2)
	major_index = max(chosen_merge1,chosen_merge2)

	new_array = []
	flag_array_merged = 1
	size_old_array = len(array2D)
	#Vamos percorrer todos os arrays um a um e vamos verificar se são os escolhidos para fazer merge
	for i in range(size_old_array):
		flag_column_merge = 1
		if i != chosen_merge1 and i != chosen_merge2:
			array1 = []
			for index in range(len(array2D[i])):
				if index == minor_index:
					# merge das colunas correspondentes aos ts que farão merge
					array1.append(min(array2D[i][minor_index],array2D[i][major_index]))
					flag_column_merge = 0
				elif index == major_index:
					pass
				else:
					array1.append(array2D[i][index])
			new_array.append(array1)
		elif flag_array_merged == 1:
			#Merge dos 2 arrays mais próximos
			array1 = array2D[minor_index]
			array2 = array2D[major_index]
			array3 = []
			#Guardamos a menor distância entre um dos 2 ts e qualquer outro
			for pos in range(len(array1)):
				if pos == minor_index:
					array3.append(max(array1[pos],array2[pos])) # o cluster + próximo de um cluster x é ele próprio
				elif pos == major_index:
					pass                                       # eliminamos a coluna desnecessária
				else:
					array3.append(min(array1[pos],array2[pos]))
			new_array.append(array3)
			flag_array_merged = 0
		else:
			pass

	#Vamos agora atualizar a label do maior array mudando-a para ser igual á do menor
	new_arrayLabels = []
	for i in range(len(arrayLabels)):
		if i == minor_index:
			new_arrayLabels.append(arrayLabels[minor_index] + arrayLabels[major_index])  # igualamos as labels do menor indice e do maior
		elif i == major_index:
			pass
		else:
			new_arrayLabels.append(arrayLabels[i])


	#Neste ponto temos um novo array formado, falta indicar de forma escrita o merge feito
	new_arrayClusters = []
	for i in range(len(arrayClusters)):
		if i == minor_index:
			new_arrayClusters.append(arrayClusters[i] + arrayClusters[major_index])
		elif i == major_index:
			pass
		else:
			new_arrayClusters.append(arrayClusters[i])

	# Vamos imprimir o que fizemos na função
	if debug == 1:
		print('Fez-se merge do cluster {', arrayClusters[minor_index] ,"} com o cluster ",arrayClusters[major_index])
	return new_array, new_arrayClusters, new_arrayLabels


def calculate_new_clusters(euc,optimal_clusters,current_clusters,clusters_per_case,cases):
	clusters_to_success = current_clusters - optimal_clusters
	arrayClusters = []
	for i in range(cases):
		for i2 in range(clusters_per_case):
			arrayClusters += [['t'+str(i2+1)+'_'+str(i+1)]]
	arrayLabels = [[i] for i in range(current_clusters)]
	#junção de 2 clusters por iteração até chegarmos ao número total de clusters a serem agregados
	for i in range(clusters_to_success):
		if debug == 1:
			print("\n\n-*-GETTING RID OF 1 CLUSTER PER CASE!!-*-\n\n")
		size = len(euc)
		channel_steps = size // current_clusters
		#iteração com todos os
		euc, arrayClusters, arrayLabels = redefine_clusters(euc, current_clusters, arrayClusters,arrayLabels)
		current_clusters -= 1
	if debug == 1:
		print('\n\nClusters criados:\n')
		print(', '.join(str(v) for v in arrayClusters) + "\n\n")

	#Lets redo labels to get ordered numbers
	response = [[0]]*clusters_per_case*cases
	count = 1
	for labels in arrayLabels:
		labels.sort()
		if debug == 1:
			print("{" + ', '.join('t'+str(int(l/clusters_per_case) + 1) + '_' + str(int(l%clusters_per_case) + 1) for l in labels) + "} => " + str(count))
		for l in labels:
			response[l] = count
		count += 1
	'''
	for c in range(4):
		print("\n\nPara o caso " + str(c+1) + " teremos as seguintes correspondencias entre ts e labels:")
		count = 1
			print("{" + ', '.join(str(v) for v in arrayLabels) + "} => " + str(count))
			for number in arrayLabels:
				response[c][number-1] = count
			count += 1
	'''
	return response

	
channels = 3
cases = 4
clusters_per_case = 11
current_clusters = clusters_per_case * cases

def clustering(channels,cases,clusters_per_case,current_clusters,photos):
	response = [0]*channels
	#itera canal (0-2)
	for channel in range(channels):
		aux = []
		mal = [] #distância de Mahalanobis
		euc = [] #distância Euclidiana
		distances = [] # array de distancias originais
		# aux é preenchido com todos os casos de 1 canal (caso 1 canal 0 -> 12 fotos,...,caso 4 canal 0 -> 12 fotos)
		# No total temos 48 fotos por aux
		for case in range(cases):
			base = 0
			if case == 0:
				base = channel * clusters_per_case
			else: 
				base = case * channels * clusters_per_case + channel * clusters_per_case
			aux += [i for i in range(base,base+clusters_per_case)]
		
		# para cada canal vamos comparar cada foto com todas as dos restantes canais
		#1º ciclo -> faz rotate 4 vezes de 12 fotografias
		for i in range(cases):
			# percorre as 12 primeiras fotos de um canal
			for pos1 in range(clusters_per_case):
				aux2 = []
				for pos2 in range(0,cases*clusters_per_case):
					#aux2.append(mahalanobis(photos[aux[pos1]],photos[aux[pos2]], inv_cov)) # mahalanobis distances
					aux2.append(int(euclidean(photos[aux[pos1]],photos[aux[pos2]])))        #euclidian distances
				#Aumento da distãncia de um cluster para si mesmo igualando-o á maior distância
				aux2[pos1] = max(aux2[:clusters_per_case])
				aux2 = aux2[clusters_per_case*(cases-i):] + aux2[:clusters_per_case*(cases-i)] #necessário para termos uma matriz quadrada com colunas comuns
				distances.append(aux2)
				#mal.append(aux2)
				euc.append(aux2)
			aux = aux[clusters_per_case:] + aux[:clusters_per_case]

		#A distância euclidiana passada é uma lista de inteiros
		# O array response tem todos os dados 
		response[channel] = calculate_new_clusters(euc,optimal_clusters,current_clusters,clusters_per_case,cases)
		# Neste ponto o array mal é composto por distancias de malhalanobis entre um tx e 
		# cada um dos ty de cada um dos restantes casos 
		# Por questões de eficiencia e como a distancia de t1 do caso 1 ao t1 do caso 2 é o mesmo que a distancia do
		# t1 do caso 2 ao t1 do caso 1 estas não foram recalculadas
	labels_train_cases = []
	for case in range(cases):
		for channel in range(channels):
			labels_train_cases += response[channel][case*11:(case+1)*11]

	return distances, labels_train_cases, photos

distances, labels_train_cases, photos_train_cases = clustering(channels,cases,clusters_per_case,current_clusters,photos)



if organized == 0:
	#Print for Euclidean distance -> Dataset original
	print("\ncaso1/caso2 |   t1    |   t2    |    t3   |   t4    |   t5    |   t6    |   t7    |    t8   |   t9    |   t10   |   t11   |")
	print("    t1      |",round(distances[0][0+11],0),"  |",round(distances[0][1+11],0),"  |",round(distances[0][2+11],0),"  |",round(distances[0][3+11],0),"  |",round(distances[0][4+11],0),"   |",round(distances[0][5+11],0),"   |",round(distances[0][6+11],0),"   |",round(distances[0][7+11],0),"  |",round(distances[0][8+11],0),"  |",round(distances[0][9+11],0),"  |",round(distances[0][10+11],0),"  |" )
	print("    t2      |",round(distances[1][0+11],0),"  |",round(distances[1][1+11],0),"  |",round(distances[1][2+11],0),"  |",round(distances[1][3+11],0),"  |",round(distances[1][4+11],0),"  |",round(distances[1][5+11],0),"   |",round(distances[1][6+11],0),"   |",round(distances[1][7+11],0),"  |",round(distances[1][8+11],0),"  |",round(distances[1][9+11],0),"  |",round(distances[1][10+11],0),"  |" )
	print("    t3      |",round(distances[2][0+11],0),"  |",round(distances[2][1+11],0),"  |",round(distances[2][2+11],0),"  |",round(distances[2][3+11],0),"  |",round(distances[2][4+11],0),"  |",round(distances[2][5+11],0),"  |",round(distances[2][6+11],0),"  |",round(distances[2][7+11],0),"  |",round(distances[2][8+11],0),"  |",round(distances[2][9+11],0),"  |",round(distances[2][10+11],0),"  |" )
	print("    t4      |",round(distances[3][0+11],0),"  |",round(distances[3][1]+11,0),"  |",round(distances[3][2+11],0),"  |",round(distances[3][3+11],0),"  |",round(distances[3][4+11],0),"  |",round(distances[3][5+11],0),"  |",round(distances[3][6+11],0),"  |",round(distances[3][7+11],0),"  |",round(distances[3][8+11],0),"  |",round(distances[3][9+11],0),"  |",round(distances[3][10+11],0),"  |" )
	print("    t5      |",round(distances[4][0+11],0),"  |",round(distances[4][1]+11,0),"  |",round(distances[4][2+11],0),"  |",round(distances[4][3+11],0),"  |",round(distances[4][4+11],0),"  |",round(distances[4][5+11],0),"  |",round(distances[4][6+11],0),"  |",round(distances[4][7+11],0),"  |",round(distances[4][8+11],0),"   |",round(distances[4][9+11],0),"  |",round(distances[4][10+11],0),"  |" )
	print("    t6      |",round(distances[5][0+11],0),"  |",round(distances[5][1]+11,0),"  |",round(distances[5][2+11],0),"  |",round(distances[5][3+11],0),"  |",round(distances[5][4+11],0),"  |",round(distances[5][5+11],0),"  |",round(distances[5][6+11],0),"  |",round(distances[5][7+11],0),"  |",round(distances[5][8+11],0),"  |",round(distances[5][9+11],0),"  |",round(distances[5][10+11],0),"  |" )
	print("    t7      |",round(distances[6][0+11],0),"  |",round(distances[6][1]+11,0),"  |",round(distances[6][2+11],0),"   |",round(distances[6][3+11],0),"  |",round(distances[6][4+11],0),"  |",round(distances[6][5+11],0),"  |",round(distances[6][6+11],0),"  |",round(distances[6][7+11],0),"  |",round(distances[6][8+11],0),"  |",round(distances[6][9+11],0),"  |",round(distances[6][10+11],0),"  |" )
	print("    t8      |",round(distances[7][0+11],0),"  |",round(distances[7][1]+11,0),"  |",round(distances[7][2+11],0),"  |",round(distances[7][3+11],0),"  |",round(distances[7][4+11],0),"  |",round(distances[7][5+11],0),"  |",round(distances[7][6+11],0),"  |",round(distances[7][7+11],0),"  |",round(distances[7][8+11],0),"  |",round(distances[7][9+11],0),"  |",round(distances[7][10+11],0),"  |" )
	print("    t9      |",round(distances[8][0+11],0),"  |",round(distances[8][1]+11,0),"  |",round(distances[8][2+11],0),"  |",round(distances[8][3+11],0),"  |",round(distances[8][4+11],0),"  |",round(distances[8][5+11],0),"  |",round(distances[8][6+11],0),"  |",round(distances[8][7+11],0),"  |",round(distances[8][8+11],0),"  |",round(distances[8][9+11],0),"  |",round(distances[8][10+11],0),"  |" )
	print("    t10     |",round(distances[9][0+11],0),"  |",round(distances[9][1]+11,0),"  |",round(distances[9][2+11],0),"  |",round(distances[9][3+11],0),"  |",round(distances[9][4+11],0),"   |",round(distances[9][5+11],0),"   |",round(distances[9][6+11],0),"  |",round(distances[9][7+11],0),"  |",round(distances[9][8+11],0),"  |",round(distances[9][9+11],0),"  |",round(distances[9][10+11],0),"  |" )
	print("    t11     |",round(distances[10][0+11],0),"  |",round(distances[10][1+11],0),"  |",round(distances[10][2+11],0),"  |",round(distances[10][3+11],0),"  |",round(distances[10][4+11],0),"   |",round(distances[10][5+11],0),"   |",round(distances[10][6+11],0),"   |",round(distances[10][7+11],0),"  |",round(distances[10][8+11],0),"  |",round(distances[10][9+11],0),"  |",round(distances[10][10+11],0),"  |\n\n" )
else:
	#Print for Euclidean distance -> Dataset reorganizado
	print("\ncaso1/caso2 |   t1    |   t2    |    t3   |   t4    |   t5    |   t6    |   t7    |    t8   |   t9    |   t10   |   t11   |")
	print("    t1      |",round(distances[0][0+11],0),"  |",round(distances[0][1+11],0),"  |",round(distances[0][2+11],0),"  |",round(distances[0][3+11],0),"  |",round(distances[0][4+11],0),"  |",round(distances[0][5+11],0),"  |",round(distances[0][6+11],0),"  |",round(distances[0][7+11],0),"  |",round(distances[0][8+11],0),"   |",round(distances[0][9+11],0),"   |",round(distances[0][10+11],0),"   |" )
	print("    t2      |",round(distances[1][0+11],0),"  |",round(distances[1][1+11],0),"  |",round(distances[1][2+11],0),"  |",round(distances[1][3+11],0),"  |",round(distances[1][4+11],0),"  |",round(distances[1][5+11],0),"  |",round(distances[1][6+11],0),"  |",round(distances[1][7+11],0),"  |",round(distances[1][8+11],0),"   |",round(distances[1][9+11],0),"   |",round(distances[1][10+11],0),"   |" )
	print("    t3      |",round(distances[2][0+11],0),"  |",round(distances[2][1+11],0),"  |",round(distances[2][2+11],0),"  |",round(distances[2][3+11],0),"  |",round(distances[2][4+11],0),"  |",round(distances[2][5+11],0),"  |",round(distances[2][6+11],0),"  |",round(distances[2][7+11],0),"  |",round(distances[2][8+11],0),"   |",round(distances[2][9+11],0),"  |",round(distances[2][10+11],0),"  |" )
	print("    t4      |",round(distances[3][0+11],0),"  |",round(distances[3][1]+11,0),"  |",round(distances[3][2+11],0),"  |",round(distances[3][3+11],0),"  |",round(distances[3][4+11],0),"  |",round(distances[3][5+11],0),"  |",round(distances[3][6+11],0),"  |",round(distances[3][7+11],0),"  |",round(distances[3][8+11],0),"  |",round(distances[3][9+11],0),"  |",round(distances[3][10+11],0),"  |" )
	print("    t5      |",round(distances[4][0+11],0),"  |",round(distances[4][1]+11,0),"  |",round(distances[4][2+11],0),"  |",round(distances[4][3+11],0),"  |",round(distances[4][4+11],0),"  |",round(distances[4][5+11],0),"  |",round(distances[4][6+11],0),"  |",round(distances[4][7+11],0),"  |",round(distances[4][8+11],0),"  |",round(distances[4][9+11],0),"  |",round(distances[4][10+11],0),"  |" )
	print("    t6      |",round(distances[5][0+11],0),"  |",round(distances[5][1]+11,0),"  |",round(distances[5][2+11],0),"  |",round(distances[5][3+11],0),"  |",round(distances[5][4+11],0),"  |",round(distances[5][5+11],0),"  |",round(distances[5][6+11],0),"   |",round(distances[5][7+11],0),"  |",round(distances[5][8+11],0),"  |",round(distances[5][9+11],0),"  |",round(distances[5][10+11],0),"  |" )
	print("    t7      |",round(distances[6][0+11],0),"  |",round(distances[6][1]+11,0),"  |",round(distances[6][2+11],0),"  |",round(distances[6][3+11],0),"  |",round(distances[6][4+11],0),"  |",round(distances[6][5+11],0),"  |",round(distances[6][6+11],0),"  |",round(distances[6][7+11],0),"  |",round(distances[6][8+11],0),"  |",round(distances[6][9+11],0),"  |",round(distances[6][10+11],0),"  |" )
	print("    t8      |",round(distances[7][0+11],0),"  |",round(distances[7][1]+11,0),"  |",round(distances[7][2+11],0),"  |",round(distances[7][3+11],0),"  |",round(distances[7][4+11],0),"  |",round(distances[7][5+11],0),"  |",round(distances[7][6+11],0),"  |",round(distances[7][7+11],0),"   |",round(distances[7][8+11],0),"  |",round(distances[7][9+11],0),"  |",round(distances[7][10+11],0),"  |" )
	print("    t9      |",round(distances[8][0+11],0),"  |",round(distances[8][1]+11,0),"  |",round(distances[8][2+11],0),"  |",round(distances[8][3+11],0),"  |",round(distances[8][4+11],0),"  |",round(distances[8][5+11],0),"  |",round(distances[8][6+11],0),"  |",round(distances[8][7+11],0),"  |",round(distances[8][8+11],0),"  |",round(distances[8][9+11],0),"  |",round(distances[8][10+11],0),"  |" )
	print("    t10     |",round(distances[9][0+11],0),"  |",round(distances[9][1]+11,0),"  |",round(distances[9][2+11],0),"  |",round(distances[9][3+11],0),"  |",round(distances[9][4+11],0),"  |",round(distances[9][5+11],0),"  |",round(distances[9][6+11],0),"  |",round(distances[9][7+11],0),"  |",round(distances[9][8+11],0),"  |",round(distances[9][9+11],0),"  |",round(distances[9][10+11],0),"  |" )
	print("    t11     |",round(distances[10][0+11],0),"  |",round(distances[10][1+11],0),"  |",round(distances[10][2+11],0),"  |",round(distances[10][3+11],0),"  |",round(distances[10][4+11],0),"  |",round(distances[10][5+11],0),"  |",round(distances[10][6+11],0),"  |",round(distances[10][7+11],0),"  |",round(distances[10][8+11],0),"  |",round(distances[10][9+11],0),"  |",round(distances[10][10+11],0),"  |\n\n" )




#A +11distância euclidiana pass+11ada é uma lista de inteir+11os
# O array response tem+11 todos os dados 
#respons+11e = calculate_new_cluster+11s(euc,optimal_clusters)


print('\n-*- EURECA DESCOBRIMOS -*-'*2)

print('\n-*- ENDED EXTRA: VAMOS DESCOBRIR O NÚMERO IDEAL DE KERNELS E VERIFICAR QUAIS OS TEMPOS DE CASOS DIFERENTES COM MENOS DISTÂNCIA EUCLIDEANA -*-\n'*2)



#plt.show()


#print(new_test_cases)

#-------------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------------#

#                                                    PREDICTION                                                     #

#-------------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------------#


'''
PREVISÕES COM HIPÓTESE DE USAR SVMs LINEARES, EXPONENCIAIS E SIGMOID
'''
def predictions(photos_train_cases,labels_train_cases,case):
	start = time.time()
	with warnings.catch_warnings():
		# ignore all caught warnings
		warnings.filterwarnings("ignore")
		# execute code that will generate warnings
		switcher = {
        	0: 'linear',     # SVM LINEAR
        	1: 'rbf',        # SVM EXPONENCIAL
        	2: 'sigmoid',    # SVM SIGMOID
        	#3: 'polynomial' # SVM POLINOMIAL -> DÁ ERRO AO USAR
    	}
		'''
		training SVM
		'''
		trained = svm.SVC(kernel=switcher.get(case, "No such case exists!")).fit(np.array(photos_train_cases).reshape(-1,clusters_per_case), labels_train_cases)

		'''
		predicting with SVM and printing results
		'''
		print("Predictions with SVM: ",switcher.get(case, "No such case exists!"))
		for i in range(len(new_test_cases)):
			print("Canal: ",i+1)
			predictedStage = trained.predict(np.array(new_test_cases[i]).reshape(-1,clusters_per_case))
			print(predictedStage)

	end = time.time()
	#print("Took ",end - start, " seconds to predict the test case over 11 componets!")




#-------------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------------#

#                                                   EXPERIENCES                                                     #

#-------------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------------#

def combinations(cases):
	#lista dos casos de teste base
	tests = [[c] for c in range(cases)]
	composicoes = len(tests) # composicoes == 4
	for comps in range(composicoes):
		next_comp = 1
		for index in range(comps+1,len(tests)):
			next_index = 1
			#não podemos ter combinações com valores repetidos
			for number in tests[comps]:
				if number in tests[index]:
					next_index = 0
					break
			#não podemos ter 2 vezes a mesma combinação
			pre_add = tests[index]+tests[comps]
			pre_add.sort()
			if pre_add in tests:
				next_index = 0
			if next_index == 0:
				pass
			else:
				tests.append(pre_add)
	return tests

def experiences():
	tests = combinations(cases)
	#cada t será um conjunto de 1/2/3/4 casos de treino
	for t in tests:
		experience_array = np.array(photos[t[0]*clusters_per_case*channels:(t[0]+1)*clusters_per_case*channels])
		for case in t:
			if case != t[0]:
				experience_array = np.concatenate((experience_array,np.array(photos[case*clusters_per_case*channels:(case+1)*clusters_per_case*channels])), axis=0)
		distances, labels_train_cases, photos_train_cases = clustering(channels,len(t),clusters_per_case,len(t)*clusters_per_case,experience_array)
		print("\n Using the following train cases: ", [number +1 for number in t], "we obtained the following predictions:\n")
		for svmtype in range(3):
			predictions(photos_train_cases, labels_train_cases,svmtype)




experiences()

























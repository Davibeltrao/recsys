from surprise import BaselineOnly
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
import csv
import math
import sklearn.metrics.pairwise
import os
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
import numpy as np
accepts = {}
cuisine = {}
parking = {}
geoplace = {}
usercuisine = {}
userpay = {}
usergeneral = {}
ratings = {}
bestRest = {}


def GetPredictedFeature(forcollab,userId):
	predictedFeatures = {}
	largest = 0
	largestId = ""
	for  u in forcollab:
		actual = 0
		print u
		actual = similarity(forcollab[userId],forcollab[u])
		if actual > largest:
			largest = actual
			largestId = u
	print largestId
	print " most close " 
	print usergeneral[u]
	print "most close cuisine"
	print usercuisine[u]
	print "user"
	print usergeneral[userId]
	print "user cuisine"
	print usercuisine[userId]
	preference = {}
	preference["ambience"] = {}
	preference["cuisine"] = {}
	preference ["accepts"] = {}
	preference ["style"] = {}
	preference["price"] = {}
	preference["parking"] = {}
	preference["alcohol"] = {}
	preference["smoking"] = {}
	greatestValues = {}
	for i in forcollab:
		for a in forcollab[i]:
			if a["rating"] >=2:
				if i not in bestRest:
					bestRest[i] = []
				else:
					bestRest[i].append(a["itemId"])
	for qw in bestRest[userId]:
		if qw in accepts:
			for i in accepts[qw]:
				if i not in preference["accepts"]:
					preference["accepts"][i] = 0
				preference["accepts"][i]+= 1
		if qw in cuisine:
			for i in cuisine[qw]:
				if i not in preference["cuisine"]:
					preference["cuisine"][i] = 0
				preference["cuisine"][i]+= 1
		if qw in parking:
			for i in parking[qw]:
				if i not in preference["parking"]:
					preference["parking"][i] = 0
				preference["parking"][i]+= 1
			if qw in geoplace:
				#print geoplace[qw]
				if geoplace[qw][12] not in preference["ambience"]:
					preference["ambience"][geoplace[qw][12]] = 0
				preference["ambience"][geoplace[qw][12]] +=1
				if geoplace[qw][16] not in preference["style"]:
					preference["style"][geoplace[qw][16]] = 0
				preference["style"][geoplace[qw][16]] += 1
				if geoplace[qw][14] not in preference["price"]:
					preference["price"][geoplace[qw][14]] = 0
				preference["price"][geoplace[qw][14]] +=1
				if geoplace[qw][10] not in preference["alcohol"]:
					preference["alcohol"][geoplace[qw][10]] = 0
				preference["alcohol"][geoplace[qw][10]] +=1
				if geoplace[qw][10] not in preference["smoking"]:
					preference["smoking"][geoplace[qw][11]] = 0
				preference["smoking"][geoplace[qw][11]] +=1			
	for i in preference:
		greatestValues[i] = getBiggest(preference[i])
	print greatestValues
	print UserSimilarity(usergeneral[userId],usergeneral[u])

def UserSimilarity(userVec,mostSimilarVec):
	#we can weight each data and use the two weighted info(from te restaurants user like and from users that a similar sometimes user similarity is way more accurate than user preferences)
	confiability = 0.0
	numberEqual = 0.0;
	featureNum = 0.0;
	for i in range(0,len(userVec)):
		if userVec[i] == mostSimilarVec[i]:
			print userVec[i] + "jsjsjsj" + mostSimilarVec[i]
			numberEqual += 1
			print numberEqual
		featureNum += 1
	confiability = numberEqual/featureNum
	return confiability


def getBiggest(mapId):
	greatest = 0;
	greatestId = "dummy"
	for i in mapId:
		if mapId[i] > greatest:
			greatest = mapId[i]
			greatestId = i
	return (greatestId,greatest)

def similarity(x,y):
	acumup = 0
	acumdown = 0
	A = 0
	B = 0
	for i in x:
		for j in y:
			item1 = i["itemId"]
			item2 = j["itemId"]
			item1ra = i["rating"]
			item2ra = j["rating"]
			if item1 == item2:
				acumup += item1ra * item2ra
				A += item1ra ** 2
				B += item2ra ** 2
	acumdown = math.sqrt(A) * math.sqrt(B);
	if acumdown == 0 or acumup == 0:
		print "not similar at all"
		return
	similarity = acumup/acumdown;
	return  similarity
file_path = os.path.expanduser('restaurant-data-with-consumer-ratings/rating_final.csv')
#reader = Reader(line_format='userId placeId rating food_rating service_rating', sep=',')
#data = Dataset.load_from_file(file_path, reader=reader)
data = []
onlyRatings = []
forcollab = {}
train_data_matrix = np.zeros((138, 130))
with open('restaurant-data-with-consumer-ratings/chefmozaccepts.csv') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
	for row in spamreader:
		if row[0] in accepts:
			accepts[row[0]].append(row[1])
		else:
			accepts[row[0]] = []
			accepts[row[0]].append(row[1])
with open('restaurant-data-with-consumer-ratings/chefmozcuisine.csv') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
	for row in spamreader:
		if row[0] in cuisine:
			cuisine[row[0]].append(row[1])
		else:
			cuisine[row[0]] = []
			cuisine[row[0]].append(row[1])
with open('restaurant-data-with-consumer-ratings/chefmozparking.csv') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
	for row in spamreader:
		if row[0] in parking:
			parking[row[0]].append(row[1])
		else:
			parking[row[0]] = []
			parking[row[0]].append(row[1])
with open('restaurant-data-with-consumer-ratings/usercuisine.csv') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
	for row in spamreader:
		if row[0] in usercuisine:
			usercuisine[row[0]].append(row[1])
		else:
			usercuisine[row[0]] = []
			usercuisine[row[0]].append(row[1])

with open('restaurant-data-with-consumer-ratings/userpayment.csv') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
	for row in spamreader:
		if row[0] in userpay:
			userpay[row[0]].append(row[1])
		else:
			userpay[row[0]] = []
			userpay[row[0]].append(row[1])
with open('restaurant-data-with-consumer-ratings/userprofile.csv') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
	for row in spamreader:
			iterrows = iter(row)
			next(iterrows)
			usergeneral[row[0]] = []	
			for p in iterrows:
				usergeneral[row[0]].append(p)
with open('restaurant-data-with-consumer-ratings/geoplaces2.csv') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
	for row in spamreader:
			iterrows = iter(row)
			next(iterrows)
			geoplace[row[0]] = []	
			for p in iterrows:
				geoplace[row[0]].append(p)
#############3

with open('restaurant-data-with-consumer-ratings/rating_final.csv') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
	for row in spamreader:
			iterrows = iter(row)
			next(iterrows)
			next(iterrows)
			ratings[row[0] + "_" + row[1]] = []	
			for p in iterrows:
				ratings[row[0] + "_" + row[1]].append(p)
for g in usergeneral:
	ga = usergeneral[g]
	for a in ratings:
		if g in a:
			data.append({"latitude":str(ga[0]),"longitude":str(ga[1]),"smoker":str(ga[2]),"drink_level":str(ga[3]),"dress_level":str(ga[4]),"ambience":str(ga[5]),"transport": str(ga[6]),"m_status": str(ga[7]),"restId": a.replace(g + "_","")})
			onlyRatings.append(float(ratings[a][0]))
			if g not in forcollab:
				forcollab[g] = []
			forcollab[g].append({"itemId":a.replace(g + "_",""), "rating":float(ratings[a][0]) })
			
v = DictVectorizer()
dataTrain = data[:len(data)/2]
dataTest = data[len(data)/2:]
ratingsTrain = onlyRatings[:len(onlyRatings)/2]
ratingsTest = onlyRatings[len(onlyRatings)/2:]
X_train = v.fit_transform(dataTrain)
y_train = np.array(ratingsTrain)
fm = pylibfm.FM(num_factors=10, num_iter=100, verbose=True, task="regression", initial_learning_rate=0.001, learning_rate_schedule="optimal")
fm.fit(X_train, y_train)
X_test = v.transform(dataTest)
preds = fm.predict(X_test)
from sklearn.metrics import mean_squared_error
print "FM MSE: %.4f" % math.sqrt(mean_squared_error(ratingsTest,preds))
vectorizer = DictVectorizer()
X = vectorizer.fit_transform(forcollab["U1003"])
Y = vectorizer.transform(forcollab["U1003"])
GetPredictedFeature(forcollab,"U1003")
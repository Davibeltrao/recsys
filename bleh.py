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

file_path = os.path.expanduser('restaurant-data-with-consumer-ratings/rating_final.csv')
#reader = Reader(line_format='userId placeId rating food_rating service_rating', sep=',')
#data = Dataset.load_from_file(file_path, reader=reader)
accepts = {}
cuisine = {}
parking = {}
geoplace = {}
usercuisine = {}
userpay = {}
usergeneral = {}
ratings = {}
bestRest = {}
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
#print vectorizer.get_feature_names()
la = sklearn.metrics.pairwise.cosine_similarity(X,Y)
#print la
#print forcollab
#for i in forcollab:
#	if i["rating"] >=2:
#		if i["userId"] not in bestRest:
#			bestRest[i["userId"]] = []
#		else:
#			bestRest[i["userId"]].append(i["itemId"])
#for qw in bestRest["U1061"]:
#	if qw in accepts:
#		print "accepts " + ''.join(accepts[qw])
#	if qw in cuisine:
#		print "cuisine " + ''.join(cuisine[qw])
#	if qw in parking:
#		print "parking " + ''.join(parking[qw])
	#print geoplace[qw]
#print forcollab
#print "\n"
print la
print vectorizer.get_feature_names()
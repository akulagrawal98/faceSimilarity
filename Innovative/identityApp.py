from PIL import Image
import numpy as np
import pickle
import glob
import os
import json
import cv2
from flask import Flask, request, render_template,request,redirect,url_for,g,make_response
import numpy as np
import predict_sim as selfscript
from feature_extractor import FeatureExtractor
import operator
import sqlite3
import takeImage as ti

app = Flask(__name__,static_url_path = "")


#DATABASE 
DATABASE='database.db'
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

def make_dicts(cursor, row):
    return dict((cursor.description[idx][0], value)
                for idx, value in enumerate(row))

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


#CREATE MODEL
fe = FeatureExtractor()

#CREATE ALL FEATURES DICTIONARY
def load_features():
	allFeatures={}
	for nameFolder in glob.glob("testImage/*"):
		if(nameFolder!="testImage/features"):
			name=nameFolder.split("/")[1]
			allFeatures[name]=np.zeros((1000,))

	for file in glob.glob("testImage/features/*"):
		personName=(file.split("/")[2]).split(".")[0]
		pickle_in = open(file,"rb")
		loaded_feature = pickle.load(pickle_in)
		allFeatures[personName]=loaded_feature
	return allFeatures
# print(len(allFeatures))


@app.route('/' , methods=['GET','POST'])
def home():
	added = (request.method == 'POST')
	if added:
		email = request.form['email']
		name = request.form['name']
		mobile = request.form['phone']
		ti.TakeImages(1,name,fe)
		try:
			cur=get_db().cursor()
			cur.execute('insert into candidate(email,name,mobile) values(?,?,?)',(email,name,int(mobile)))
			get_db().commit()
			return render_template("home.html",err=False)
		except:
			return render_template('home.html',file_error=True,detail_upload=True,err=True)
		
	return render_template("home.html")

@app.route('/search' , methods=['GET','POST'])
def search():
	searched = (request.method == 'POST')
	if searched:
		imgPath = request.form['imgsrc']
		img=selfscript.imagePreprocess(imgPath)
		feature=fe.extract(img)
		allFeatures=load_features()
		result_dict=selfscript.comparePickle(feature,allFeatures)
		clone_dict=result_dict.copy()
		result_list=[]
		detail_list=[]
		cur=get_db().cursor()
		print("I am clone DICT",clone_dict)
		similarity_list=[]
		for i in range(5):
			similar=max(clone_dict.items(), key=operator.itemgetter(1))[0]
			
			similarity_list.append(round((clone_dict[similar]*100),2))
			result_list.append(similar.capitalize())
			clone_dict[similar]=0
			# print("SIMILAR VALUE",similar)
			detail = cur.execute('select * from candidate where name="{}"'.format(similar))
			detail_list.append(detail.fetchall())
			# print(detail.fetchall())
		# print(result_list)
		# print(imgPath)
		print("THIS IS THE PATH",imgPath)
		return render_template('search.html',similarity_list=similarity_list,file_error=True,result_list=result_list,query_img=imgPath,detail_list=detail_list)
	return render_template('search.html',file_error=True)
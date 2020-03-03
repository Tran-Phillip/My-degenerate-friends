import json 
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
# data = None
import nltk
from nltk.corpus import stopwords
import datetime
import scipy.stats as stats
from nltk.corpus import sentiwordnet as swn
from textblob import TextBlob
import random
# file = 1

# counter = defaultdict(lambda:0)
# while(file <= 5):
# 	print("file: ", file)
# 	with open("message_"+str(file)+".json", "r") as read_file:
# 		data = json.load(read_file)

# 	for msg in data['messages']:
# 		counter[msg['sender_name']] += 1
# 	file += 1

# values = [counter[x] for x in counter]
# labels = list(counter)
# plt.pie(values, labels=labels,autopct='%1.1f%%',)
# plt.show()


# data = None
# file = 1
# counter = defaultdict(lambda:0)
# stop_words = set(stopwords.words('english'))
# stop_words.add("i")
# stop_words.add("lo")
# stop_words.add("u")
# stop_words.add("im")
# stop_words.add("i'm")
# stop_words.add("")

# with open('stop_words.txt', 'r') as f:
# 	for word in f:
# 		stop_words.add(word[0:-1])




# while(file <= 5):
# 	print("file: ", file)
# 	with open("message_"+str(file)+".json", "r") as read_file:
# 		data = json.load(read_file)

# 	for msg in data['messages']:
# 		try:
# 			content = msg['content']
# 			split_content = content.split(" ")
# 			split_content[-1] = split_content[-1][0:-1]

# 			for word in split_content:
# 				if(word.lower() not in stop_words and len(word) > 2 and msg['sender_name'] == 'Phillip Tran'):
# 					counter[word.lower()] += 1
# 		except: 
# 			pass

# 	file += 1


# l = []
# for word in counter:
# 	l.append( (word, counter[word]))

# sorted_words = (sorted(l, key=lambda l:l[1]))



# for i in range(1,41):
# 	print(sorted_words[-1 * i])

# file = 5
# counter = defaultdict(lambda:0)
# while(file >= 1):
# 	print("file: ", file)
# 	with open("message_"+str(file)+".json", "r") as read_file:
# 		data = json.load(read_file)
# 	for msg in data['messages']:
# 		ms = msg['timestamp_ms']
# 		date = (datetime.datetime.fromtimestamp(ms/1000).strftime('%Y-%m-%d'))
# 		counter[date] += 1
# 	file -=1


# years = mdates.YearLocator()   # every year
# months = mdates.MonthLocator()  # every month
# years_fmt = mdates.DateFormatter('%m')

# tups = []
# for key in counter:
# 	tups.append((key, counter[key]))

# sorted_dates = (sorted(tups, key=lambda tups:tups[0]))
# fig, ax = plt.subplots()
# dates = [datetime.datetime.strptime(x[0], "%Y-%m-%d") for x in sorted_dates]
# values = [x[1] for x in sorted_dates]
# ax.plot(dates, values)

# # format the ticks
# ax.xaxis.set_major_locator(months)
# ax.xaxis.set_major_formatter(years_fmt)
# ax.xaxis.set_minor_locator(months)

# # format the coords message box
# ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
# ax.format_ydata = lambda x: '$%1.2f' % x  # format the price.

# # rotates and right aligns the x labels, and moves the bottom of the
# # axes up to make room for them
# fig.autofmt_xdate()
# # plt.show()

# print(np.percentile(values, [25, 50, 75]))

# fit = stats.norm.pdf(values, np.mean(values), np.std(values))  #this is a fitting indeed
# print(fit)

# file = 5
# counter = defaultdict(lambda:0)
# active_convo = []
# last_msg = defaultdict(lambda:0)


# while(file >= 1):
# 	print("file: ", file)
# 	with open("message_"+str(file)+".json", "r") as read_file:
# 		data = json.load(read_file)
# 	for msg in data['messages']:
# 		ms = int(msg['timestamp_ms'])
# 		name = msg['sender_name']

# 		# check who leaves the convo 

# 		for idx,person in enumerate(active_convo):
# 			if(person in last_msg and  abs(last_msg[person] - ms) > 300000 ): # 5 minutes
# 				active_convo.pop(idx)

# 		if(name not in active_convo):
# 			active_convo.append(name)
# 			last_msg[name] = ms
# 		if(len(active_convo) > 1):
# 			active_convo.sort()
# 			counter["->".join(active_convo)] += 1

# 	file -=1



# convos = []

# for convo in counter:
# 	convos.append((convo, counter[convo]))


# filter_convos = []



# sorted_convos = (sorted(convos, key=lambda convos:convos[1]))

# for convo in sorted_convos:
# 	print(convo)




# file = 5
# counter = []
# while(file >= 1):
# 	print("file: ", file)
# 	with open("message_"+str(file)+".json", "r") as read_file:
# 		data = json.load(read_file)
# 	for msg in data['messages']:
# 		try:
# 			content = msg['content']
# 			sender = msg['sender_name']

# 			if(sender == 'Bryan Chiu'):
# 				blob = TextBlob(content)

# 				for sentence in blob.sentences:
# 					pol = sentence.sentiment.polarity
# 					if(pol):
# 						counter.append(pol)
# 		except:
# 			pass

# 	file -=1

# plt.hist(counter)
# plt.show()

file = 5
all_tokens = []
while(file >= 1):
	print("file: ", file)
	with open("message_"+str(file)+".json", "r") as read_file:
		data = json.load(read_file)
	for msg in data['messages']:
		try:
			tokens = nltk.word_tokenize(msg['content'].lower())
			all_tokens += tokens
		except: 
			pass

	file -= 1
text = nltk.Text(all_tokens)
text.generate(length=500, random_seed=random.randint(0,1000))

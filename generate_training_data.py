# Create training data set from given url links.
# Example seed url : https://www.washingtonpost.com/business/technology/?utm_term=.60dd8ebdfa1f
# 
# Seed url :
#	 --> Get all article links from the url 
# 	 --> For each article link extract the text and title

import urllib.request
from bs4 import BeautifulSoup
from urllib.error import HTTPError
import csv

# Load stopwords into list
stopwords = nltk.corpus.stopwords.words('english')

def get_all_article_links(seed_url_str):
	"""
		Given a url, fetch all article links from the page 
		using beautiful soup
	"""

	page_source = urllib.request.urlopen(seed_url_str).read().decode('utf-8')
	article_links_arr = []
	soup_obj = BeautifulSoup(page_source)
	for anchor_tag in soup_obj.find_all("a"):
		if anchor_tag.has_key('href'):
			if '2017' in anchor_tag['href']:
				article_links_arr.append(anchor_tag['href'])

	return article_links_arr


def get_all_article_title_text_wp(link_str):
	"""
		Given an article link from washington post 
		returns the text & title of the article
	"""

	page_source = urllib.request.urlopen(link_str).read().decode('utf-8')
	paragraph_tag = []
	paragraph_text = []
	print(">> Processing for link : " + link_str)
	soup_obj1 = BeautifulSoup(page_source,'lxml')
	article = soup_obj1.find("article")
	
	soup_obj2 = BeautifulSoup(article.encode('utf-8'),'lxml')
	paragraph_tag.extend(soup_obj2.find_all("p"))
	
	for paragraph in paragraph_tag:
		paragraph_text.append(paragraph.text)

	return [' '.join(paragraph_text), soup_obj1.title.text]


def get_all_article_title_text_np(link_str):
	"""
		Given an article link from newyork post 
		returns the text & title of the article
	"""

	page_source = urllib.request.urlopen(link_str).read().decode('utf-8')
	soup = BeautifulSoup(page_source)

	print(">> Processing for link : " + link_str)

	p_text = []
	for div in soup.find_all('div'):
		if div.has_key('class'):
			if 'entry-content' in div['class']:
				div_data = div

	soup_obj2 = BeautifulSoup(div_data.encode('utf-8'),'lxml')
	for p in soup_obj2.find_all('p'):
		p_text.append(p.text)

	return [' '.join(p_text), soup.title.text]

def text_tokenization(text):

	'''
		Tokenize sentences
		Remove punctuation
		Remove stopwords
		Lowercase all words
	'''
	words = nltk.word_tokenize(text)
	cleanedwords = []
	for word in words:
		if word.lower() not in stopwords:
			cleanedletters = []
			for letter in word.lower():
				if letter not in punctuation:
					cleanedletters.append(letter)
			
			if len(cleanedletters) > 0:
				cleanedwords.append(''.join(cleanedletters))

	return cleanedwords


if __name__ == "__main__":

	seed_urls = ['http://nypost.com/sports/',
				 'http://nypost.com/tech/',
				 'https://www.washingtonpost.com/business/technology/?utm_term=.2b425cc358b7',
				 'https://www.washingtonpost.com/sports/?utm_term=.aa7699967dcb']

	all_article_links = []
	for seed_url in seed_urls:
		all_article_links.extend(get_all_article_links(seed_url))

	url_text_title = {}
	for link in all_article_links:
		try:
			if 'nypost' in link:
				url_text_title[link] = get_all_article_title_text_np(link)
				if 'sports' in link:
					url_text_title[link].append('non-tech')
				else:
					url_text_title[link].append('tech')
			else:
				url_text_title[link] = get_all_article_title_text_wp(link)
				if 'technology' in link:
					url_text_title[link].append('tech')
				else:
					url_text_title[link].append('non-tech')
		except:
			print(" Forbidden error ")
		break

	print("Identified text and title for : " + str(len(url_text_title)) + " articles.")
    
    
	# Write data to a text file
	csvfile = open('trainingdataset.csv','w')
	writer = csv.writer(csvfile,delimiter = ',')
	writer.writerow(['LINK','TITLE','TEXT','LABEL'])

	for link in url_text_title:
		writer.writerow([link,url_text_title[link][1].encode('utf-8'),url_text_title[link][0].encode('utf-8'),url_text_title[link][2].encode('utf-8')])
	
	for link in url_text_title:
		print(url_text_title[link][0].encode('utf-8'))
		text_tokens = text_tokenization(url_text_title[link][0])
		print(text_tokens)

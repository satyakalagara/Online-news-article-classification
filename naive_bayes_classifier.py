"""
	Train & test using Naive Bayes classifier 
	I am using sklearn implementation of naive bayes algorithm along with feature extraction 
	techniques provided by sklearn
"""

import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from pandas import DataFrame
from sklearn.metrics import f1_score,accuracy_score

def read_data_to_matrix(file_name):
	"""
		Read data from file and return a pandas dataframe. 
		Pandas data frame is much more easier to slice & dice. 
	"""
	fp = open(file_name,'r')
	csv_reader = csv.reader(fp)

	file_records = []
	for line in csv_reader:
		if len(line) > 1:
			file_records.append({'text':line[0], 'class':line[1]})

	data_frame = DataFrame(file_records)
	return data_frame

def get_test_data():

	return [
			"Ad sales boost Time Warner profit Quarterly profits at US media giant TimeWarner jumped 76% to $1.13bn (£600m) for the three months to December,from $639m year-earlier.The firm, which is now one of the biggest investors in Google, benefited from sales of high-speed internet connections and higher advert sales. TimeWarner said fourth quarter sales rose 2% to $11.1bn from $10.9bn. Its profits were buoyed by one-off gains which offset a profit dip at Warner Bros, and less users for AOL. Time Warner said on Friday that it now owns 8% of search-engine Google. But its own internet business, AOL, had has mixed fortunes. It lost 464,000 subscribers in the fourth quarter profits were lower than in the preceding three quarters. However, the company said AOL's underlying profit before exceptional items rose 8% on the back of stronger internet advertising revenues. It hopes to increase subscribers by offering the online service free to TimeWarner internet customers and will try to sign up AOL's existing customers for high-speed broadband. TimeWarner also has to restate 2000 and 2003 results following a probe by the US Securities Exchange Commission (SEC), which is close to concluding.Time Warners fourth quarter profits were slightly better than analysts' expectations. But its film division saw profits slump 27% to $284m, helped by box-office flops Alexander and Catwoman, a sharp contrast to year-earlier, when the third and final film in the Lord of the Rings trilogy boosted results. For the full-year, TimeWarner posted a profit of $3.36bn, up 27% from its 2003 performance, while revenues grew 6.4% to $42.09bn.Our financial performance was strong, meeting or exceeding all of our full-year objectives and greatly enhancing our flexibility,chairman and chief executive Richard Parsons said. For 2005, TimeWarner is projecting operating earnings growth of around 5%, and also expects higher revenue and wider profit margins.TimeWarner is to restate its accounts as part of efforts to resolve an inquiry into AOL by US market regulators. It has already offered to pay $300m to settle charges, in a deal that is under review by the SEC. The company said it was unable to estimate the amount it needed to set aside for legal reserves, which it previously set at $500m. It intends to adjust the way it accounts for a deal with German music publisher Bertelsmann's purchase of a stake in AOL Europe, which it had reported as advertising revenue. It will now book the sale of its stake in AOL Europe as a loss on the value of that stake.",
			"Dollar gains on Greenspan speech.The dollar has hit its highest level against the euro in almost three months after the Federal Reserve head said the US trade deficit is set to stabilise.And Alan Greenspan highlighted the US government's willingness to curb spending and rising household savings as factors which may help to reduce it. In late trading in New York, the dollar reached $1.2871 against the euro, from $1.2974 on Thursday. Market concerns about the deficit has hit the greenback in recent months. On Friday, Federal Reserve chairman Mr Greenspan's speech in London ahead of the meeting of G7 finance ministers sent the dollar higher after it had earlier tumbled on the back of worse-than-expected US jobs data. I think the chairman's taking a much more sanguine view on the current account deficit than he's taken for some time, said Robert Sinche, head of currency strategy at Bank of America in New York. He's taking a longer-term view, laying out a set of conditions under which the current account deficit can improve this year and next.Worries about the deficit concerns about China do, however, remain. China's currency remains pegged to the dollar and the US currency's sharp falls in recent months have therefore made Chinese export prices highly competitive. But calls for a shift in Beijing's policy have fallen on deaf ears, despite recent comments in a major Chinese newspaper that the time is ripe for a loosening of the peg. The G7 meeting is thought unlikely to produce any meaningful movement in Chinese policy. In the meantime, the US Federal Reserve's decision on 2 February to boost interest rates by a quarter of a point - the sixth such move in as many months - has opened up a differential with European rates. The half-point window, some believe, could be enough to keep US assets looking more attractive, and could help prop up the dollar. The recent falls have partly been the result of big budget deficits, as well as the US's yawning current account gap, both of which need to be funded by the buying of US bonds and assets by foreign firms and governments. The White House will announce its budget on Monday, and many commentators believe the deficit will remain at close to half a trillion dollars.",
			"Yukos unit buyer faces loan claim The owners of embattled Russian oil giant Yukos are to ask the buyer of its former production unit to pay back a $900m (£479m) loan.State-owned Rosneft bought the Yugansk unit for $9.3bn in a sale forced by Russia to part settle a $27.5bn tax claim against Yukos. Yukos' owner Menatep Group says it will ask Rosneft to repay a loan that Yugansk had secured on its assets. Rosneft already faces a similar $540m repayment demand from foreign banks. Legal experts said Rosneft's purchase of Yugansk would include such obligations.The pledged assets are with Rosneft, so it will have to pay real money to the creditors to avoid seizure of Yugansk assets,said Moscow-based US lawyer Jamie Firestone, who is not connected to the case. Menatep Group's managing director Tim Osborne told the Reuters news agency:If they default, we will fight them where the rule of law exists under the international arbitration clauses of the credit.Rosneft officials were unavailable for comment. But the company has said it intends to take action against Menatep to recover some of the tax claims and debts owed by Yugansk. Yukos had filed for bankruptcy protection in a US court in an attempt to prevent the forced sale of its main production arm. The sale went ahead in December and Yugansk was sold to a little-known shell company which in turn was bought by Rosneft. Yukos claims its downfall was punishment for the political ambitions of its founder Mikhail Khodorkovsky and has vowed to sue any participant in the sale."
		]

if __name__ == "__main__":

	file_records = read_data_to_matrix('training_dataset.csv')

	# Get Frequency Count 
	count_vect = CountVectorizer(stop_words="english",decode_error='ignore')
	train_counts = count_vect.fit_transform(file_records['text'].values)

	# Train NB classifier
	classifier = MultinomialNB()
	targets = file_records['class'].values
	classifier.fit(train_counts, targets)

	test_data = get_test_data()
	test_counts = count_vect.transform(test_data)
	predictions = classifier.predict(test_counts)
	predict_matrix = []
	for i in predictions:
		if i == 'non-tech':
			i = 0
		else:
			i = 1
		predict_matrix.append(i)
	print(predict_matrix)

	actual_class = [0,0,0]
	print("f1: %f\n" %f1_score(actual_class,predict_matrix,average = None))
	print("accuracy: %f\n" %accuracy_score(actual_class,predict_matrix))


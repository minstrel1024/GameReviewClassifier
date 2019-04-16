import argparse
import sys
import os.path
import numpy as np

from nltk import ngrams

from sklearn.naive_bayes import BernoulliNB

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Dictionary(object):
	def __init__(self, min_count = 2):
		self.min_count = min_count
		self.word2idx = {} 
		self.idx2word = [] 
		self.word_count = {}
		self.idx2word.append('<bias>')
		self.word2idx['<bias>'] = 0
		self.word_count['<bias>'] = 0
		self.idx2word.append('<unk>')
		self.word2idx['<unk>'] = 1
		self.word_count['<unk>'] = 1
		self.idx2word.append('<s>')
		self.word2idx['<s>'] = 2
		self.word_count['<s>'] = 0
		self.idx2word.append('</s>')
		self.word2idx['</s>'] = 3
		self.word_count['</s>'] = 0
		self.length = 4
		
	def add_word(self, word):
		self.word_count[word] = self.word_count.get(word, 0)+1
		if(word not in self.word2idx)and(self.word_count[word] >= self.min_count):
			self.idx2word.append(word) 
			self.word2idx[word] = self.length
			self.length += 1 
			return self.word2idx[word]
			
	def __len__(self):
		return len(self.idx2word) 
		
	def onehot_encoded(self, word):
		if(word not in self.word2idx):
			word = '<unk>'
		vec = np.zeros(self.length) 
		vec[self.word2idx[word]] = 1 
		return vec
		
	def get_index(self, word):
		if(word not in self.word2idx):
			word = '<unk>'
		return self.word2idx[word]
		
class SkipGram(nn.Module):
	#From https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html#sphx-glr-beginner-nlp-word-embeddings-tutorial-py
	def __init__(self, vocab_size, embedding_dim, context_size, hidden_layer_size = 64):
		super(SkipGram, self).__init__()
		self.embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.linear1 = nn.Linear(2 * context_size * embedding_dim, hidden_layer_size)
		self.linear2 = nn.Linear(hidden_layer_size, vocab_size)

	def forward(self, inputs):
		embeds = self.embeddings(inputs).view((1, -1))
		out = F.relu(self.linear1(embeds))
		out = self.linear2(out)
		log_probs = F.log_softmax(out, dim=1)
		return log_probs

def Embedding(train_string, unigram, context_size = 2, embedding_dim = 16, max_iter = 10):
	#From https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html#sphx-glr-beginner-nlp-word-embeddings-tutorial-py
	data = []
	for s in train_string:
		for i in range(0, len(s)-0):
			context = []
			for j in range(i-context_size, i+context_size+1):
				if i != j:
					if j < 0:
						context.append('<s>')
					elif j >= len(s):
						context.append('</s>')
					else:
						context.append(s[j])
			target = s[i]
			data.append((context, target))
	losses = []
	loss_function = nn.NLLLoss()
	model = SkipGram(unigram.length, embedding_dim, context_size).cuda()
	optimizer = optim.SGD(model.parameters(), lr=0.001)
	cuda0 = torch.device('cuda:0')
	for epoch in range(max_iter):
		total_loss = 0
		for context, target in data:
			
			context_idxs = torch.tensor([unigram.get_index(w) for w in context], dtype=torch.long, device = cuda0)
			model.zero_grad()
			log_probs = model(context_idxs)
			loss = loss_function(log_probs, torch.tensor([unigram.get_index(target)], dtype=torch.long).cuda())
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
		losses.append(total_loss)
		print("Embedding Training Iteration: " + str(epoch) + ", Loss: " + str(total_loss))
		#if epoch > 0 and losses[epoch-1]-losses[epoch] < losses[epoch]*1E-4:
		#	break
	return model.embeddings
	
class LSTM(nn.Module):
	#From https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#sphx-glr-beginner-nlp-sequence-models-tutorial-py
	def __init__(self, vocab_size, tagset_size, embedding_dim = 16, hidden_dim = 64):
		super(LSTM, self).__init__()
		self.hidden_dim = hidden_dim

		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

		# The LSTM takes word embeddings as inputs, and outputs hidden states
		# with dimensionality hidden_dim.
		self.lstm = nn.LSTM(embedding_dim, hidden_dim)

		# The linear layer that maps from hidden state space to tag space
		self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
		self.hidden = self.init_hidden()

	def init_hidden(self):
		return (torch.zeros(1, 1, self.hidden_dim).cuda(),
				torch.zeros(1, 1, self.hidden_dim).cuda())
	def to_cpu(self):
		self.hidden = (self.hidden[0].cpu(), self.hidden[1].cpu())
		return self.cpu()

	def forward(self, sentence):
		embeds = self.word_embeddings(sentence)
		lstm_out, self.hidden = self.lstm(
			embeds.view(len(sentence), 1, -1), self.hidden)
		tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
		tag_scores = F.log_softmax(tag_space, dim=1)
		return tag_scores[len(sentence)-1].view(1, -1)
		
def LSTMTraining(train_string, train_label, unigram, embedding_dim = 16, hidden_dim = 64, max_iter = 20):
	#From https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#sphx-glr-beginner-nlp-sequence-models-tutorial-py
	assert(len(train_string) != 0 and len(train_string) == len(train_label))
	model = LSTM(unigram.length, 2, embedding_dim, hidden_dim).cuda()
	loss_function = nn.NLLLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.1)
	train_data = []
	for i in range(len(train_string)):
		train_data.append((train_string[i], train_label[i]))
	cuda0 = torch.device('cuda:0')


	for epoch in range(max_iter):
		total_loss = 0
		for sentence, label in train_data:
			# Step 1. Remember that Pytorch accumulates gradients.
			# We need to clear them out before each instance
			model.zero_grad()

			# Also, we need to clear out the hidden state of the LSTM,
			# detaching it from its history on the last instance.
			model.hidden = model.init_hidden()

			# Step 2. Get our inputs ready for the network, that is, turn them into
			# Tensors of word indices.

			sentence_in = torch.tensor([unigram.get_index(w) for w in sentence], dtype=torch.long, device=cuda0)
			itarget = 0
			if label == "1":
				itarget = 1
			target = torch.tensor([itarget], dtype=torch.long, device=cuda0)
			# Step 3. Run our forward pass.
			tag_scores = model(sentence_in)
			# Step 4. Compute the loss, gradients, and update the parameters by
			#  calling optimizer.step()
			loss = loss_function(tag_scores, target)
			loss.backward()
			total_loss += loss.item()
			optimizer.step()
		print("LSTM Training Iteration: " + str(epoch) + ", Loss: " + str(total_loss))
	return model

def eval(predict, eval, s = "Evaluation:"):
	assert(len(predict) != 0 and len(predict) == len(eval))
	tp = tn = fp = fn = 0.0
	l = len(predict)
	for i in range(l):
		if predict[i] == '1':
			if eval[i] == '1':
				tp += 1
			else:
				fp += 1
		else:
			if eval[i] == '1':
				fn += 1
			else:
				tn += 1
	P = tp/(tp+fp)
	R = tp/(tp+fn)
	F1 = 2*P*R/(P+R)
	Acc = (tp+tn)/(tp+tn+fp+fn)
	print()
	print(s)
	print("Acc: " + str(Acc))
	print("Precision: " + str(P))
	print("Recall: " + str(R))
	print("F1: " + str(F1))
	return P, R, F1, Acc

		
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="NONE", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	#addonoffarg(parser, 'debug', help="debug mode", default=False)
	parser.add_argument("--train", "-t", type=str, default="data/train_comments.csv", help="train file")
	#parser.add_argument("--predict", "-p", type=str, nargs='+', default=["data/eval_comments.csv"], help="evaluation files")
	parser.add_argument("--predict", "-p", type=str, default="data/eval_comments.csv", help="evaluation file")
	#parser.add_argument("--outdir", "-o", type=str, default=".", help="location of evaluation files")
	parser.add_argument("--classifier", "-c", default="trigram_bayes", choices=["bow_bayes", "bigram_bayes", "trigram_bayes", "lstm", "embedding"], help="classifier")
	parser.add_argument("--embedding", "-e", default="auto", help="location of embedding file, use 'self' to calculate embedding itself, use 'auto' to train embedding by pytorch")

	try:
		args = parser.parse_args()
	except IOError as msg:
		parser.error(str(msg))
	
	train_string = []
	train_label = []
	unigram = Dictionary(5)
	bigram = Dictionary(10)
	trigram = Dictionary(10)
	with open(args.train, 'r', encoding='windows-1252') as f:
		for line in f.readlines():
			nowcomment = line.split(',', 5)
			if len(nowcomment) < 5:
				sys.stderr.write("Wrong train_data format")
				sys.exit(1)
			train_label.append(nowcomment[1])
			s = nowcomment[5].strip().lower()
			ls = s.split()
			train_string.append(ls)
			for tok in ls:
				unigram.add_word(tok)
				bigram.add_word(tok)
				trigram.add_word(tok)
			for tok in list(ngrams(ls, 2)):
				bigram.add_word(tok)
				trigram.add_word(tok)
			for tok in list(ngrams(ls, 3)):
				trigram.add_word(tok)
	print(unigram.length)
	#print(unigram.word2idx)
	#print(unigram.word_count)
	print(bigram.length)
	#print(bigram.word2idx)
	#print(bigram.word_count)
	print(trigram.length)
	
	predict_string = []
	real_label = []
	with open(args.predict, 'r', encoding='windows-1252') as f:
		for line in f.readlines():
			nowcomment = line.split(',', 5)
			if len(nowcomment) < 5:
				sys.stderr.write("Wrong predict_data format")
				sys.exit(1)
			real_label.append(nowcomment[1])
			s = nowcomment[5].strip().lower()
			ls = s.split()
			predict_string.append(ls)
				
	if(args.classifier == 'bow_bayes'):
		train_data = []
		predict_data = []
		predict_label = []
	
		for s in train_string:
			nvec = unigram.onehot_encoded('<bias>')
			for tok in s:
				nvec += unigram.onehot_encoded(tok)
			train_data.append(nvec)
		for s in predict_string:
			nvec = unigram.onehot_encoded('<bias>')
			for tok in s:
				nvec += unigram.onehot_encoded(tok)
			predict_data.append(nvec)		
		
		clf = BernoulliNB()
		clf.fit(train_data, train_label)
		predict_label = clf.predict(predict_data)
		with open(args.predict+'.bow_bayes.pred', 'w') as f:
			l = len(predict_string)
			for i in range(l):
				f.write(predict_label[i] + ',' + real_label[i] + ',' + ' '.join(predict_string[i])+'\n')
		predict_train_label = clf.predict(train_data)
		P, R, F1, Acc = eval(predict_train_label, train_label, "Bag-of-words Bayes(Train):")
		P, R, F1, Acc = eval(predict_label, real_label, "Bag-of-words Naive Bayes:")
	
	if(args.classifier == 'bigram_bayes'):
		train_data = []
		predict_data = []
		predict_label = []
		for s in train_string:
			nvec = bigram.onehot_encoded('<bias>')
			for tok in list(ngrams(s, 2)):
				nvec += bigram.onehot_encoded(tok)
			train_data.append(nvec)
		for s in predict_string:
			nvec = bigram.onehot_encoded('<bias>')
			for tok in list(ngrams(s, 2)):
				nvec += bigram.onehot_encoded(tok)
			predict_data.append(nvec)		

		clf = BernoulliNB()
		clf.fit(train_data, train_label)
		predict_label = clf.predict(predict_data)
		with open(args.predict+'.bigram_bayes.pred', 'w') as f:
			l = len(predict_string)
			for i in range(l):
				f.write(predict_label[i] + ',' + real_label[i] + ',' + ' '.join(predict_string[i])+'\n')
		predict_train_label = clf.predict(train_data)
		P, R, F1, Acc = eval(predict_train_label, train_label, "Bigrams Naive Bayes(Train):")
		P, R, F1, Acc = eval(predict_label, real_label, "Bigrams Naive Bayes:")
		
	if(args.classifier == 'trigram_bayes'):
		train_data = []
		predict_data = []
		predict_label = []
		for s in train_string:
			nvec = trigram.onehot_encoded('<bias>')
			for tok in list(ngrams(s, 3)):
				nvec += trigram.onehot_encoded(tok)
			train_data.append(nvec)
		for s in predict_string:
			nvec = trigram.onehot_encoded('<bias>')
			for tok in list(ngrams(s, 3)):
				nvec += trigram.onehot_encoded(tok)
			predict_data.append(nvec)		

		clf = BernoulliNB()
		clf.fit(train_data, train_label)
		predict_label = clf.predict(predict_data)
		predict_train_label = clf.predict(train_data)
		with open(args.predict+'.trigram_bayes.pred', 'w') as f:
			l = len(predict_string)
			for i in range(l):
				f.write(predict_label[i] + ',' + real_label[i] + ',' + ' '.join(predict_string[i])+'\n')
		predict_train_label = clf.predict(train_data)
		P, R, F1, Acc = eval(predict_train_label, train_label, "Trigrams Naive Bayes(Train):")
		P, R, F1, Acc = eval(predict_label, real_label, "Trigrams Naive Bayes:")

	
	dic = {}
	cuda0 = torch.device('cuda:0')
	if(args.embedding != 'auto' and (args.classifier in ['embedding', 'lstm'])):
		torch.manual_seed(1117)
		if args.embedding == 'self':
			embedding = Embedding(train_string = train_string, unigram = unigram, context_size = 2, embedding_dim = 16, max_iter = 5)
			with open('data/embedding.csv', 'w') as f:
				for i in range(unigram.length):
					ll = dic[unigram.idx2word[i]] = embedding(torch.tensor([i], dtype=torch.long, device = cuda0)).detach().cpu().numpy()[0]
					f.write(unigram.idx2word[i])
					for j in range(len(ll)):
						f.write(' '+str(ll[j]))
					f.write('\n')
		else:
			import distsim
			dic = distsim.load_word2vec(args.embedding)
			avg = np.zeros(len(dic['the']))
			for w in dic:
				avg += dic[w]
			avg /= len(dic)
			with open('data/embedding_tmp.csv', 'w') as f:
				for i in range(unigram.length):
					if(unigram.idx2word[i] not in dic):
						dic[unigram.idx2word[i]] = numpy.zeros(len(avg))
					ll = dic[unigram.idx2word[i]]
					f.write(unigram.idx2word[i])
					for j in range(len(ll)):
						f.write(' '+str(ll[j]))
					f.write('\n')
		
	if(args.classifier == 'lstm'):
		torch.manual_seed(1117)
		predict_label = []
		predict_train_label = []
		lstm = LSTMTraining(train_string, train_label, unigram, embedding_dim = 32, hidden_dim = 64, max_iter = 40)
		with open(args.predict+'.lstm.pred', 'w') as f:
			l = len(predict_string)
			for i in range(l):
				s = lstm(torch.tensor([unigram.get_index(w) for w in predict_string[i]], dtype=torch.long, device = cuda0)).detach().cpu().numpy()[0]
				lnow = np.argmax(s)
				if lnow == 0:
					lnow = "-1"
				else:
					lnow = "1"
				predict_label.append(lnow)
				f.write(predict_label[i] + ',' + real_label[i] +','+ ' '.join(predict_string[i])+'\n')
			l = len(train_string)
			for i in range(l):
				s = lstm(torch.tensor([unigram.get_index(w) for w in train_string[i]], dtype=torch.long, device = cuda0)).detach().cpu().numpy()[0]
				lnow = np.argmax(s)
				if lnow == 0:
					lnow = "-1"
				else:
					lnow = "1"
				predict_train_label.append(lnow)
		P, R, F1, Acc = eval(predict_train_label, train_label, "LSTM(Train):")
		P, R, F1, Acc = eval(predict_label, real_label, "LSTM:")
			
		
		
		
		
		
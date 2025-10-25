#include "word2vec.hpp"

//-------------canonical form----------------

Word2vec::Word2vec(vector<vector<string> > tokenized_corpus, unordered_map<string, ll> vocab, unsigned int sliding_window) {
	this->vocab = vocab;
	this->sliding_window = sliding_window;
	this->tokenized_corpus = tokenized_corpus;
}

Word2vec::Word2vec(const Word2vec & other) {
	this->vocab = other.vocab;
	this->tokenized_corpus = other.tokenized_corpus;
	this->sliding_window = other.sliding_window;
	this->training_pairs = other.training_pairs;
}

const Word2vec & Word2vec::operator=(const Word2vec & other) {
	if (this != &other) {
		this->vocab = other.vocab;
		this->tokenized_corpus = other.tokenized_corpus;
		this->sliding_window = other.sliding_window;
		this->training_pairs = other.training_pairs;
	}

	return *this;
}

Word2vec::~Word2vec() {

}

//------------------main functions-----------------------

void	Word2vec::make_training_pairs() {
	for (auto & vec : this->tokenized_corpus) {
		for (auto & chunk : vec) {
			
		}
	}
}
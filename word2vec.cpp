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
	this->dec_training_pairs = other.dec_training_pairs;
}

const Word2vec & Word2vec::operator=(const Word2vec & other) {
	if (this != &other) {
		this->vocab = other.vocab;
		this->tokenized_corpus = other.tokenized_corpus;
		this->sliding_window = other.sliding_window;
		this->training_pairs = other.training_pairs;
		this->dec_training_pairs = other.dec_training_pairs;
	}

	return *this;
}

Word2vec::~Word2vec() {

}

//-----------------helper functions---------------------


void	Word2vec::print_training_pairs() {
	cout << "-------training pairs-----------" << endl;

	for (const auto & pair : this->training_pairs) {
		cout << pair.first << " -> " <<  pair.second << endl;
		cout << "-------------------------------" << endl;
	}
}



//------------------main functions-----------------------

void	Word2vec::make_training_pairs() {

	cout << "starting to make training pairs" << endl;
	cout << "-------------------------------" << endl;

	// pre calculate the number of pairs :

	size_t total_pairs = 0;
    const ll side_size = this->sliding_window / 2;
    if (side_size <= 0)
		return;

	for (const auto & sentance : this->tokenized_corpus) {
		const ll sentance_size = sentance.size();

		if (sentance_size <= 1)
			continue ;

		for (ll i = 0; i < sentance_size; ++i) {
			ll lower_bound = max((ll)0, i - side_size);
			ll upper_bound = min(sentance_size - 1, i + side_size);
			total_pairs += (upper_bound - lower_bound);
		}
	}

	// doing this save us from using .push_back() which is expensive
	this->training_pairs.reserve(total_pairs);
	this->dec_training_pairs.reserve(total_pairs);

	// now filling the traing pairs
	for (const auto & sentance : this->tokenized_corpus) {
		const ll sentance_size = sentance.size();

		if (sentance_size <= 1)
			continue ;

		for (ll i = 0; i < sentance_size; ++i) {
			ll start = max((ll)0, i - side_size);
			ll end = min(sentance_size, i + side_size + 1);
			
			for (ll j = start; j < end; ++j) {
				if (j == i)
					continue ;
				this->training_pairs.emplace_back(sentance[i], sentance[j]);
				this->dec_training_pairs.emplace_back(this->vocab[sentance[i]],
													this->vocab[sentance[j]]);
			}
		}
	}

	cout <<"finished making training pairs" << endl;
}


void	Word2vec::one_hot_encoder() {
	
}
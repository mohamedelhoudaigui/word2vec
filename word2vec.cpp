#include "word2vec.hpp"

//-------------canonical form----------------

Word2vec::Word2vec(vector<vector<string> > tokenized_corpus,
					unordered_map<string, ll> vocab,
					unsigned int sliding_window,
					unsigned int embedding_size) {
	this->vocab = vocab;
	this->sliding_window = sliding_window;
	this->tokenized_corpus = tokenized_corpus;
	this->embedding_size = embedding_size;

	initialize_matrix(this->embedding_matrix, vocab.size(), embedding_size);
	initialize_matrix(this->output_matrix, this->embedding_size, this->vocab.size());
}

Word2vec::Word2vec(const Word2vec & other) {
	this->vocab = other.vocab;
	this->tokenized_corpus = other.tokenized_corpus;
	this->sliding_window = other.sliding_window;
	this->training_pairs = other.training_pairs;
	this->dec_training_pairs = other.dec_training_pairs;
	this->embedding_size = other.embedding_size;
	this->embedding_matrix = other.embedding_matrix;
	this->output_matrix = other.output_matrix;
}

const Word2vec & Word2vec::operator=(const Word2vec & other) {
	if (this != &other) {
		this->vocab = other.vocab;
		this->tokenized_corpus = other.tokenized_corpus;
		this->sliding_window = other.sliding_window;
		this->training_pairs = other.training_pairs;
		this->dec_training_pairs = other.dec_training_pairs;
		this->embedding_size = other.embedding_size;
		this->embedding_matrix = other.embedding_matrix;
		this->output_matrix = other.output_matrix;
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

double		Word2vec::get_random_uniform() {

	static mt19937 engine(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    uniform_real_distribution<double> distribution(-0.5, 0.5);

    return distribution(engine);
}

void	Word2vec::print_embedding_matrix() {
	cout << "-----------embedding matrix -----------" << endl;
	for (const auto& word : this->embedding_matrix) {
		for (const auto& p : word) {
			cout << fixed  << setprecision(3) << p << '*';
		}
		cout << endl;
	}
	cout << "------------------------------------" << endl;
}

vector<double> Word2vec::softmax(vector<double>& input_vector) {

    vector<double> output_vector;
    output_vector.reserve(input_vector.size());

    double sum_exp = 0.0;

    for (double value : input_vector) {
        sum_exp += std::exp(value);
    }

    for (double value : input_vector) {
        output_vector.push_back(std::exp(value) / sum_exp);
    }

    return output_vector;
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

	cout <<"finished making training pairs, total : " << this->training_pairs.size() << endl;
}


void	Word2vec::verify() {
	for (auto & sentance : this->tokenized_corpus) {
		for (auto & token : sentance) {
			if (!this->vocab.count(token)) {
				abort();
			}
		}
	}
}


void	Word2vec::initialize_matrix(vector<vector<double> >& matrix, unsigned int row, unsigned int col) {
	// a row for each word
	matrix.resize(row);

	for (size_t i = 0; i < row; ++i) {
		matrix[i].resize(col);

		for (size_t j = 0; j < col; ++j) {
			matrix[i][j] = get_random_uniform();
		}
	}

}

void	Word2vec::matrix_mult(const vector<double> & vec, const vector<vector<double> > & matrix, vector<double> & output_scores) {

	if (matrix.empty() || matrix[0].empty() || vec.empty()) {
        cerr << "error: matrix or vector are empty" << endl;
		output_scores.clear();
        return ;
    }

	const size_t embedding_size = vec.size();
    const size_t vocab_size = matrix[0].size();


    if (embedding_size != matrix.size()) {
        cerr << "error: dimension mismatch for matrix multiplication" << endl;
        cerr << "vector size is " << vec.size() << ", but Matrix rows are " << matrix.size() << endl;
		output_scores.clear();
        return ;
    }

	output_scores.assign(vocab_size, 0.0);

	for (size_t k = 0; k < embedding_size; ++k) {
        const double v1_k = vec[k]; 
        for (size_t j = 0; j < vocab_size; ++j) {
            output_scores[j] += v1_k * matrix[k][j];
        }
    }
}

void	Word2vec::training_loop(unsigned int epochs) {

	for (unsigned int i = 0; i < epochs; ++i) {
		for (auto & p : this->dec_training_pairs) {
			vector<double> result;
			matrix_mult(this->embedding_matrix[p.first - 1], this->output_matrix, result);
			softmax(result);
		}
	}
}
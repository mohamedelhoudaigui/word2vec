#pragma once

#include <iostream>
#include <unordered_map>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cctype>
#include <utility>
#include <chrono>
#include <random>
#include <iomanip>


using namespace std;

#define ll long long

#define SLIDING_WINDOW 4
#define EMBEDDING_SIZE 50
#define VOCAB_SIZE 200


class Word2vec {

	public:
		Word2vec(vector<vector<string> > tokenized_corpus,
				unordered_map<string, ll> vocab,
				unsigned int sliding_window,
				unsigned int embedding_size);

		Word2vec(const Word2vec & other);
		const Word2vec & operator=(const Word2vec & other);
		~Word2vec();

		void			print_training_pairs();
		void			print_embedding_matrix();
		double			get_random_uniform();
		vector<double>	softmax(vector<double>& input_vector);
		void			matrix_mult(const vector<double> & vec, const vector<vector<double> > & matrix, vector<double> & output_scores);

		void			make_training_pairs();
		void			initialize_matrix(vector<vector<double> >& matrix, unsigned int row, unsigned int col);
		void			training_loop(unsigned int epochs);

	private:
		vector<pair<string, string> >	training_pairs;
		vector<pair<double, double> >	dec_training_pairs;
		unordered_map<string, ll>		vocab;
		vector<vector<string> >			tokenized_corpus;
		vector<vector<double> >			embedding_matrix;
		vector<vector<double> >			output_matrix;
		unsigned int					sliding_window;
		unsigned int					embedding_size;

};
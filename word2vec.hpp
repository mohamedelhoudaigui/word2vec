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

		void	print_training_pairs();
		void	print_embedding_matrix();
		double	get_random_uniform();

		void	make_training_pairs();
		void	initialize_embedding_matrix();
		void	one_hot_encoder();

	private:
		vector<pair<string, string> >	training_pairs;
		vector<pair<ll, ll> >			dec_training_pairs;
		unordered_map<string, ll>		vocab;
		vector<vector<string> >			tokenized_corpus;
		vector<vector<double> >			embedding_matrix;
		unsigned int					sliding_window;
		unsigned int					embedding_size;

};
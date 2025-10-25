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


using namespace std;

#define ll long long


class Word2vec {

	public:
		Word2vec(vector<vector<string> > tokenized_corpus,
				unordered_map<string, ll> vocab,
				unsigned int sliding_window);

		Word2vec(const Word2vec & other);
		const Word2vec & operator=(const Word2vec & other);
		~Word2vec();

		void	make_training_pairs();

	private:
		vector<pair<string, string> >	training_pairs;
		unordered_map<string, ll>		vocab;
		unsigned int					sliding_window;
		vector<vector<string> >			tokenized_corpus;

};
#include "./word2vec.hpp"
#include "./tokenizer/BPE.hpp"

int main (int argc , char *argv[]) {

	if (argc != 2) {
		cerr << "usage : word2vec <file path>" << endl;
		return 1;
	}

	BPE tokenizer(argv[1], ":.;\n\r");
	tokenizer.divide_corpus();
    tokenizer.train(100);

	Word2vec embedder(tokenizer.get_tokenized_corpus(),
						tokenizer.get_vocab(), 
						4);

	embedder.make_training_pairs();

	return 0;
}
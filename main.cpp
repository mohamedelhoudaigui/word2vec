#include "./word2vec.hpp"
#include "./tokenizer/BPE.hpp"

int main (int argc , char *argv[]) {

	if (argc != 2) {
		cerr << "usage : word2vec <file path>" << endl;
		return 1;
	}

	BPE tokenizer(argv[1], ":.;\n\r");
	tokenizer.divide_corpus();
    tokenizer.train(VOCAB_SIZE);

	Word2vec embedder(tokenizer.get_tokenized_corpus(),
					tokenizer.get_vocab(), 
					SLIDING_WINDOW,
					EMBEDDING_SIZE);

	//tokenizer.print_token_corpus();

	embedder.make_training_pairs();
	//embedder.print_training_pairs();
	embedder.print_embedding_matrix();

	return 0;
}
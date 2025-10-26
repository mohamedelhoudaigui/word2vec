SRCS = main.cpp word2vec.cpp

HEADERS = word2vec.hpp ./tokenizer/BPE.hpp

OBJS = $(SRCS:.cpp=.o)

CC = c++

CFLAGS = -Wall -Wextra -Werror -std=c++11 -fsanitize=address

TOK = make -C tokenizer

NAME = word2vec

TOK_PATH = -Ltokenizer
TOK_NAME = -lbpe


all: lib $(NAME)

$(NAME): $(OBJS) ./tokenizer/libbpe.a
	$(CC) $(CFLAGS) -o $(NAME) $(OBJS) $(TOK_PATH) $(TOK_NAME)

lib:
	$(TOK)

%.o: %.cpp $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) $(OBJS)
	$(TOK) clean

fclean: clean
	$(RM) $(NAME)
	$(TOK) fclean

re: fclean all

.PHONY: clean
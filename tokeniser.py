import re
from collections import Counter
import statistics as stats

class Tokeniser:
    def __init__(self):
        self.__check_train = False
        self.__vocabulary = []
        self.__vocab_counts = {}
        self.__vocab_sum_reciprocal = 0
        self.__vocab_freq = []

    def tokenise_on_punctuation(self, text):
        """
        Tokenises a provided string of text on preset punctuation list and white space. Returns a list of tokens.
        """
        words = re.sub(r"[!#%&'()*,-./:;?@[\]_{}¡§«¶·»¿‘’“”–—\"]|\s"," ", text)
        words = words.split()
        return words

    def train(self, text):
        """
        Takes a corpus and prepares the tokeniser for use on new texts with the methods: tokenise, tokenise_with_count_threshold and tokenise_with_freq_threshold.
        Collects a number of metrics for use in these later methods.
        """
        words = re.sub(r"[!#%&'()*,-./:;?@[\]_{}¡§«¶·»¿‘’“”–—\"]|\s"," ", text)
        self.__check_train = True
        self.__vocabulary = words.split()
        self.__vocab_counts = dict(Counter(self.__vocabulary))
        self.__vocab_sum_reciprocal = 1/sum(self.__vocab_counts.values())
        self.__vocab_freq = list(map(lambda x: x * self.__vocab_sum_reciprocal, list(self.__vocab_counts.values())))

    def tokenise(self, text, use_unk=False):
        """
        Splits the input then goes over the tokens. If a token is in the trained vocabulary, it's
        added to the output, otherwise unknown tokens are dealt with depending on the use_unk argument.
        Raises an error if the train method has not yet been invoked.
        """
        if self.__check_train == True:
            words = re.sub(r"[!#%&'()*,-./:;?@[\]_{}¡§«¶·»¿‘’“”–—\"]|\s"," ", text)
            words = words.split()
            output = []
            for token in words:
                if token in self.__vocabulary:
                    output.append(token)
                elif token not in self.__vocabulary and use_unk == True:
                    output.append('UNK')
                else:
                    for letter in token:
                        output.append(letter)
            return output
        else:
            raise RuntimeError(f'The tokeniser has not been trained yet.')

    def tokenise_with_count_threshold(self, text, threshold, use_unk=False):
        """
        Splits the input then goes over the tokens. Tokens are only added to the output if they appeared x times
        or more, with x defined by the threshold argument. The use_unk argument is used as for the tokenise method.
        Raises an error if the train method has not yet been invoked.
        """
        if self.__check_train == True:
            words = re.sub(r"[!#%&'()*,-./:;?@[\]_{}¡§«¶·»¿‘’“”–—\"]|\s"," ", text)
            words = words.split()

            threshold_vocab = {}
            for token, count in self.__vocab_counts.items():
              if count >= threshold:
                  threshold_vocab[token] = count

            output = []
            for token in words:
                if token in list(threshold_vocab.keys()):
                    output.append(token)
                elif token not in list(threshold_vocab.keys()) and use_unk == True:
                    output.append('UNK')
                else:
                    for letter in token:
                        output.append(letter)
            return output
        else:
            raise RuntimeError(f'The tokeniser has not been trained yet.')

    def tokenise_with_freq_threshold(self, text, threshold, use_unk=False):
        """
        Splits the input then goes over the tokens. Tokens are only added to the output if they appeared at or above
        the frequency threshold (between 0 and 1). The use_unk argument is used as for the tokenise method.
        Raises an error if the train method has not yet been invoked.
        """
        if not 0 <= threshold <=1:
            raise ValueError("The choice of threshold value must be between 0 and 1 inclusive.")
        if self.__check_train == True:
            words = re.sub(r"[!#%&'()*,-./:;?@[\]_{}¡§«¶·»¿‘’“”–—\"]|\s"," ", text)
            words = words.split()

            count_freq = dict(zip(self.__vocab_counts.keys(),self.__vocab_freq))
            freq_vocab = {}
            for token, freq in count_freq.items():
                if freq >= threshold:
                    freq_vocab[token] = freq

            output = []
            for token in words:
                if token in list(freq_vocab.keys()):
                    output.append(token)
                elif token not in list(freq_vocab.keys()) and use_unk == True:
                    output.append('UNK')
                else:
                    for letter in token:
                        output.append(letter)
            return output
        else:
            raise RuntimeError(f'The tokeniser has not been trained yet.')
    
    def __str__(self):
        return f"Welcome to the tokeniser."

def get_stats(tokenised_text):
    """
    Takes a tokenised corpus and returns a dictionary with the fields shown.
    """
    counts = Counter(tokenised_text)
    type_count = len(counts.keys())
    token_count = sum(counts.values())
    chars = [len(token) for token in tokenised_text]
    char_counts = Counter(chars)
    d = {}
    d["type_count"] = type_count # number of different tokens in the corpus
    d["token_count"] = token_count # total number of tokens in the corpus
    d["type_token_ratio"] = type_count/token_count # ratio between type_count and token_count
    d["token_count_by_length"] = dict(sorted(char_counts.items())) # number of tokens of different lengths, measured in characters
    d["average_token_length"] = stats.mean(chars) # mean length in characters of all tokens
    d["token_length_std"] = stats.stdev(chars) # standard deviation length in characters of all tokens
    return d

def main():
    """
    Test that runs if the file is run in the command line as a script.
    Script will not run if the file is imported as a module.
    """
    text = input("Please provide your string of text: ")
    test_tokenise = Tokeniser()
    print('Here is your text, tokenised on punctuation and whitespace:\n', test_tokenise.tokenise_on_punctuation(text))
    
if __name__ == "__main__":
    main()

import nltk as ntlk
from nltk.corpus.reader.plaintext import PlaintextCorpusReader

corpus_root = 'C:\\Users\\burha\\Documents\\Projecten\\NLP\\bbc'
corpus_bbc = PlaintextCorpusReader(corpus_root, r'.*\.txt')

print("Corpus imported")
print(corpus_bbc)
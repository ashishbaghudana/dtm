import logging
import os
from gensim import corpora, utils
from gensim.models.wrappers.dtmmodel import DtmModel
import numpy as np
import glob
from nltk.corpus import stopwords

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug('test')

class DocumentReader:
    """
    Class to read documents from directories classified as business, sports,
    technology and so on.
    """
    def __init__(self):
        self.documents = []
        self.time_seq = []

    def read(self, input_directory):
        input_files = glob.glob(os.path.join(input_directory, '*.txt'))
        for article in input_files:
            with open(article) as f:
                text = f.read()
                document = self.parse(text)
                self.documents.append(document)
        self.time_seq.append(len(input_files))

    def parse(self, text):
        all_words = text.split()
        stop_words = stopwords.words('english')
        stoplist = set('for a of the and to in'.split())
        words = [word.decode('UTF-8', 'ignore').lower() for word in all_words if word.lower() not in stop_words]
        return words


class DTMcorpus(corpora.textcorpus.TextCorpus):
    """
    Subclassing the TextCorpus class and adding methods get_texts and
    overriding __len__
    """
    def get_texts(self):
        return self.input

    def __len__(self):
        return len(self.input)


class DTM:
    """
    Implement DtmModel and print topics
    """
    def __init__(self, dtm_path, corpus, time_seq):
        self.dtm_path = dtm_path
        self.corpus = corpus
        self.time_seq = time_seq
        self.model = None

    def create_model(self):
        self.model = DtmModel(self.dtm_path, self.corpus, self.time_seq,
            num_topics=5, id2word=self.corpus.dictionary, initialize_lda=True)
        return self.model

def main():
    # read input directories
    reader = DocumentReader()
    reader.read('../corpus/month1')
    reader.read('../corpus/month2')
    reader.read('../corpus/month3')

    # create corpus from Documents
    corpus = DTMcorpus(reader.documents)

    # set DTM path
    dtm_path = '../bin/dtm'
    dtm = DTM(dtm_path, corpus, reader.time_seq)
    model = dtm.create_model()

    # print topics
    topics = model.show_topic(topicid=1, time=1, topn=10)
    print topics

if __name__ == '__main__':
    main()

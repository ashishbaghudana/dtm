import logging
import os
from gensim import corpora, utils
from gensim.models.wrappers.dtmmodel import DtmModel
import numpy as np
import glob
from nltk.corpus import stopwords
import xml.etree.ElementTree as ET
import pickle
from progress.bar import Bar

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

    def read_dir(self, input_directory):
        """
        The directory consists of XML files that are annotated specifically for
        speeches. Each XML file corresponds to the list of speeches in one
        month.

        Every speech is contained in the XML tag <speech id='...'><p></p><speech>
        This information is extracted out of the XML tags by using the builtin
        module 'xml.etree.ElementTree'
        """
        input_files = glob.glob(os.path.join(input_directory, '*.xml'))
        bar = Bar('Parsing XML Files', max=len(input_files), suffix='%(percent)d%%')
        for debates in input_files:
            bar.next()
            self.parse(debates)
        bar.finish()

    def parse(self, debates):
        """
        Parse the content of each XML file to separate each speech and process
        it individually. Only if a month has more than 100 speeches it is
        recorded.
        """
        tree = ET.parse(debates)
        root = tree.getroot()
        count = 0
        documents = []
        for child in root:
            if child.tag == 'speech':
                if len(child)>0:
                    text = child[0].text
                    if text is not None:
                        count += 1
                        document = self.tokenize(text)
                        documents.append(document)
        if count>100:
            self.documents += documents
            self.time_seq.append(count)

    def tokenize(self, content):
        """
        Tokenization according to Wikipedia corpus, where any token less than 2
        characters long and greater than 15 characters long is ignored. The
        token must not start with '_'.
        """
        return [token.encode('utf8') for token in utils.tokenize(content, lower=True, errors='ignore')
                if 2 <= len(token) <= 15 and not token.startswith('_')]

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
            num_topics=15, id2word=self.corpus.dictionary, initialize_lda=True)
        return self.model

def main():
    # read input directories
    reader = DocumentReader()
    reader.read_dir('/home/ashish/Projects/data')

    # create corpus from Documents
    corpus = DTMcorpus(reader.documents)

    # remove extremes
    corpus.dictionary.filter_extremes(no_below=20, no_above=0.05, keep_n=100000)

    # set output file and save corpus to file - do I need this?
    # outp = 'bnc_corpus'
    # corpora.MmCorpus.serialize(outp + '_bow.mm', corpus, progress_cnt=10000) # another ~9h
    # wiki.dictionary.save_as_text(outp + '_wordids.txt.bz2')

    # set DTM path
    dtm_path = '/home/ashish/Projects/dtm'
    dtm = DTM(dtm_path, corpus, reader.time_seq)
    model = dtm.create_model()

    with open('/home/ashish/Projects/model', 'wb') as fout:
        pickle.dump(model, fout)

if __name__ == '__main__':
    main()

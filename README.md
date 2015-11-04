# Dynamic Topic Modeling using Gensim

The corpus is currently derived from [here](https://github.com/derekgreene/dynamic-nmf/tree/master/data) and consists of documents across 5 categories - business, entertainment, football, politics and technology. The data is currently is minimally preprocessed, just to convert all strings to unicode literals, and simple stop word filtering from the nltk stopwords corpus.

### Preliminary Results
```text
topic1   |   topic2   |   topic3   |   topic4   |   topic5
--------------------------------------------------------------
-        |   said     |   -        |   mr       |   said
best     |   -        |   would    |   said     |   people
music    |   us       |   "i       |   -        |   would
one      |   also     |   club     |   film     |   mr
also     |   new      |   game     |   would    |   could
new      |   people   |   chelsea  |   blair    |   government
top      |   mobile   |   united   |   also     |   -
number   |   last     |   league   |   labour   |   said
games    |   would    |   players  |   told     |   new
first    |   could    |   said     |   best     |   also
```

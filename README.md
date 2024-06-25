# LLM Zoomcamp pre-course workshop 2: Implement a Search Engine

As the title suggests, this repo captures the content as well as code walkthrough from the workshop conducted by [datatalks.club](https://datatalks.club/). This session is more of a taster into the actual course [llm-zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) for those interested in Large Language Models (LLMs). Actually, this would be pretty useful in the first week of the course as the instructor goes through the materials quite briefly. Hence, it would be advisable to go through the pre-course thoroughly. Enough chit-chat, let's dive into the content, and the video link for the workshop is [here](https://www.youtube.com/watch?v=nMrGK5QgPVE).

## 1. Introduction to LLM and RAG

**1.1 LLM**

LLM stands for Large Language Models, and Generative Pre-Trained Transformers or simply GPTs are an example of a LLM. And yes, OpenAI's flagship product the ChatGPT-3/4 is an example of a LLM. So what exactly is an LLM?

LLMs are an instance of a foundation model, i.e. models that are pre-trained on large amounts of unlabelled and self-supervised data. The foundation model learns from patterns in the data in a way that produces generalizable and adaptable output. And LLMs are instances of foundation models applied specifically to text and text-like things.

LLMs are also among the biggest models when it comes to parameter count. For example, OpenAI ChatGPT-3 model has about a 175 billion parameters. That's insane but necessary for making the product more adaptable. Parameter is a value the model can change independently as it learns, and the more parameters a model has, the more complex it can be.

So how do they work? - LLMs can be said to be made of three things:

1. Data - Large amounts of text data used as inputs into LLMs
2. Architecture - As for architecture this is a neural network, and for GPT that is a `transformer` (transformer architecture enables the model to handle sequences of data as they are designed to understand the context of each word in a sentence)
3. Training - The aforementioned transformer architecture is trained on the large amounts of data used as input, and consquentially the model learns to predict the next word in a sentence

![image](https://github.com/peterchettiar/llm-search-engine/assets/89821181/a917fa0d-4b5d-40ef-ab95-5a4a214b2b69)

The image above is a good representation of how the ChatGPT-3 operates; you input a prompt and having gone through the transformer process, it gives a text response as well. The key concept here is to understand how the transformer architecture works but that is not the main objective for today. Hence, read this [article](https://www.datacamp.com/tutorial/how-transformers-work) to understand more about the transformer architecture in detail.

> Note: A transformer is a type of artificial intelligence model that learns to understand and generate human-like text by analyzing patterns in large amounts of text data.

**1.2 RAG**

RAG stands for Retrieval-Augmentation Generation which is a technique that supplements text generation with information from private or proprietary data sources. The main purpose of having a RAG model in place together with a LLM is so that the relevance of the search experience can be improved. The RAG model adds context from various data sources to complement the original knowledge base of the LLM. This method allows the responses from the LLM to be more accurate and a generally faster response.

Following is a good visual representation of the implementation and orchestration of RAG:

![image](https://github.com/peterchettiar/llm-search-engine/assets/89821181/1df01240-c487-4ef3-99f4-d16157b8175c)

## 2. Preparing the environment

In the workshop, we'll use Github Codespaces, but you can use any env

We need to install the following libraries:

```bash
pip install requests pandas scikit-learn jupyter
```

Start jupyter:

```bash
jupyter notebook
```

Download the data:

```python
import requests 

docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)
```

Creating the dataframe:

```python
import pandas as pd

df = pd.DataFrame(documents, columns=['course', 'section', 'question', 'text'])
df.head()
```

## 3. Basics of Text Search

- **Information Retrieval** - The process of obtaining relevant information from large datasets based on user queries.
- **Vector Spaces** - A mathematical representation where text is converted into vectors (points in space) allowing for quantitative comparison.
- **Bag of Words** - A simple text representation model treating each document as a collection of words disregarding grammar and word order but keeping multiplicity.
- **TF-IDF (Term Frequency-Inverse Document Frequency)** - A statistical measure used to evaluate how important a word is to a document in a collection or corpus. It increases with the number of times a word appears in the document but is offset by the frequency of the word in the corpus.


## 3. Implementing Basic Text Search

Let's implement it ourselves.


### Keyword filtering

First, keyword filtering:

```python
df[df.course == 'data-engineering-zoomcamp'].head()
```

### Vectorization

For Count Vectorizer and TF-IDF we will first use a simple example

```python
documents = [
    "Course starts on 15th Jan 2024",
    "Prerequisites listed on GitHub",
    "Submit homeworks after start date",
    "Registration not required for participation",
    "Setup Google Cloud and Python before course"
]
```

Let's use a count vectorizer first:

```python
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words='english')
X = cv.fit_transform(docs_example)

names = cv.get_feature_names_out()

df_docs = pd.DataFrame(X.toarray(), columns=names).T
df_docs
```

This representation is called "bag of words" - here we ignore the order of words, just focus on the words themselves. In many cases this is sufficient and gives pretty good results already.

Now let's replace it with `TfidfVectorizer`:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

cv = TfidfVectorizer(stop_words='english')
X = cv.fit_transform(docs_example)

names = cv.get_feature_names_out()

df_docs = pd.DataFrame(X.toarray(), columns=names).T
df_docs.round(2)
```

### Query-Document Similarity

We represent the query in the same vector space - i.e. using the same vectorizer:


```python
query = "Do I need to know python to sign up for the January course?"

q = cv.transform([query])
q.toarray()
```

We can see the words of the query and the words of some document:

```python
query_dict = dict(zip(names, q.toarray()[0]))
query_dict

doc_dict = dict(zip(names, X.toarray()[1]))
doc_dict
```

The more words in common - the better the matching score. Let's calculate it:

```python
df_qd = pd.DataFrame([query_dict, doc_dict], index=['query', 'doc']).T

(df_qd['query'] * df_qd['doc']).sum()
```

This is a dot-product. So we can use matrix multiplication to compute the score:


```python
X.dot(q.T).toarray()
```

Watch [this linear algebra refresher](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/01-intro/08-linear-algebra.md) if you're a bit rusty on matrix multiplication (don't worry - it's developer friendly)

Bottom line: it's a very fast and effective method of computing similarities


In practice, we usually use cosine similarity:

```python
cosine_similarity(X, q)
```

The TF-IDF vectorizer already outputs a normalized vectors, so the results are identical. We won't go into details of how it works, but you can check "Introduction to Infromation Retrieval" if you want to learn more. 

### Vectorizing all the documents

Let's now do it for all the documents:

```python
fields = ['section', 'question', 'text']
transformers = {}
matrices = {}

for field in fields:
    cv = TfidfVectorizer(stop_words='english', min_df=3)
    X = cv.fit_transform(df[field])

    transformers[field] = cv
    matrices[field] = X

transformers['text'].get_feature_names_out()
matrices['text']
```

### Search

Let's now do search with the text field:

```python
query = "I just singned up. Is it too late to join the course?"

q = transformers['text'].transform([query])
score = cosine_similarity(matrices['text'], q).flatten()
```

Let's do it only for the data engineering course:

```python
mask = (df.course == 'data-engineering-zoomcamp').values
score = score * mask
```

And get the top results:

```python
import numpy as np

idx = np.argsort(-score)[:10]
```

Note: [np.argpartition](https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html) is a more efficient way of doing the same thing

Get the docs:

```python
df.iloc[idx].text
```

### Search with all the fields & boosting + filtering

We can do it for all the fields. Let's also boost one of the fields - `question` - to give it more importance than to others 

```python
boost = {'question': 3.0}

score = np.zeros(len(df))

for f in fields:
    b = boost.get(f, 1.0)
    q = transformers[f].transform([query])
    s = cosine_similarity(matrices[f], q).flatten()
    score = score + b * s
```

And add filters (in this case, only one):

```python
filters = {
    'course': 'data-engineering-zoomcamp'
}

for field, value in filters.items():
    mask = (df[field] == value).values
    score = score * mask
```

Getting the results:

```python
idx = np.argsort(-score)[:10]
results = df.iloc[idx]
results.to_dict(orient='records')
```

### Putting it all together 

Let's create a class for us to use:

```python
class TextSearch:

    def __init__(self, text_fields):
        self.text_fields = text_fields
        self.matrices = {}
        self.vectorizers = {}

    def fit(self, records, vectorizer_params={}):
        self.df = pd.DataFrame(records)

        for f in self.text_fields:
            cv = TfidfVectorizer(**vectorizer_params)
            X = cv.fit_transform(self.df[f])
            self.matrices[f] = X
            self.vectorizers[f] = cv

    def search(self, query, n_results=10, boost={}, filters={}):
        score = np.zeros(len(self.df))

        for f in self.text_fields:
            b = boost.get(f, 1.0)
            q = self.vectorizers[f].transform([query])
            s = cosine_similarity(self.matrices[f], q).flatten()
            score = score + b * s

        for field, value in filters.items():
            mask = (self.df[field] == value).values
            score = score * mask

        idx = np.argsort(-score)[:n_results]
        results = self.df.iloc[idx]
        return results.to_dict(orient='records')
```

Using it:

```python
index = TextSearch(
    text_fields=['section', 'question', 'text']
)
index.fit(documents)

index.search(
    query='I just singned up. Is it too late to join the course?',
    n_results=5,
    boost={'question': 3.0},
    filters={'course': 'data-engineering-zoomcamp'}
)
```

You can fild the implementation here too if you want to use it: https://github.com/alexeygrigorev/minsearch


**Note**: this is a toy example for illustrating how relevance search works. It's not meant to be used in production.

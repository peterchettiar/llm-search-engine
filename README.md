# LLM Zoomcamp pre-course workshop 2: Implement a Search Engine

As the title suggests, this repo captures the content as well as code walkthrough from the workshop conducted by [datatalks.club](https://datatalks.club/). This session is more of a taster into the actual course [llm-zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) for those interested in Large Language Models (LLMs). Actually, this would be pretty useful in the first week of the course as the instructor goes through the materials quite briefly. Hence, it would be advisable to go through the pre-course thoroughly. Enough chit-chat, let's dive into the content and the video link for the workshop is [here](https://www.youtube.com/watch?v=nMrGK5QgPVE).

## 1. Introduction to LLM and RAG

**1.1 LLM**

LLM stands for Large Language Models, and Generative Pre-Trained Transformers or simply GPTs are an example of a LLM. And yes, OpenAI's flagship product the ChatGPT-3/4 is an example of a LLM. So what exactly is an LLM?

LLMs are an instance of a foundation model, i.e. models that are pre-trained on large amounts of unlabelled and self-supervised data. The foundation model learns from patterns in the data in a way that produces generalizable and adaptable output. And LLMs are instances of foundation models applied specifically to text and text-like things.

LLMs are also among the biggest models when it comes to parameter count. For example, OpenAI ChatGPT-3 model has about a 175 billion parameters. That's insane but necessary for making the product more adaptable. Parameter is a value the model can change independently as it learns, and the more parameters a model has, the more complex it can be.

So how do they work? - LLMs can be said to be made of three things:

1. Data - Large amounts of text data used as inputs into LLMs
2. Architecture - As for architecture this is a neural network and for GPT that is a `transformer` (transformer architecture enables the model to handle sequences of data as they are designed to understand the context of each word in a sentence)
3. Training - The aforementioned transformer architecture is trained on the large amounts of data used as input, and consquentially the model learns to predict the next word in a sentence

![image](https://github.com/peterchettiar/llm-search-engine/assets/89821181/a917fa0d-4b5d-40ef-ab95-5a4a214b2b69)

The image above is a good representation of how the ChatGPT-3 operates; you input a prompt and having gone through the transformer process, it gives a text response as well. The key concept here is to understand how the transformer architecture works but that is not the main objective for today. Hence, read this [article](https://www.datacamp.com/tutorial/how-transformers-work) to understand more about the transformer architecture in detail.

> Note: A transformer is a type of artificial intelligence model that learns to understand and generate human-like text by analyzing patterns in large amounts of text data.

**1.2 RAG**

RAG stands for Retrieval-Augmentation Generation

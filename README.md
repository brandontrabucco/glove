# GloVe Word Embeddings

## Preface

In this repository, we implement a python framework using NumPy and NLTK to load and process the GloVe word embeddings into a vocabulary to be used in machine learning projects. Below is the original homepage for GloVe with links to other sources and data files that may be interesting to read.

https://nlp.stanford.edu/projects/glove/

## Installation and Setup

First, clone this repository onto your computer using the following command.

```
brandon@btrabucco.com:~$ git clone http://gitub.com/brandontrabucco/glove
```

Second, install the repository as a user library in Python3 using the following command.

```
brandon@btrabucco.com:~$ cd glove/
brandon@btrabucco.com:~/glove$ python3 pip install --user -r requirements.txt
```

This command may require multiple minutes to complete, as multiple computing libraries such as NumPy and NLTK and required by this repository. Additional word data for NLTK will also be downloaded.

Finally, we must download and extract the glove word embeddings and vocabulary files from their online hosts. This has been coded into a bash script *embeddings/download_and_extract.sh* which you should execute.

```
brandon@btrabucco.com:~/glove$ cd embeddings/
brandon@btrabucco.com:~/glove/embeddings$ chmod u+x download_and_extract.sh
brandon@btrabucco.com:~/glove/embeddings$ download_and_extract.sh
```

We can now run the tests to ensure the repository has downloaded all appropriate files and has been installed to your system. Run the following python file containing test cases.

```
brandon@btrabucco.com:~/glove/embeddings$ cd ../
brandon@btrabucco.com:~/glove$ python3 tests.py
All test cases passed.
```

If you see this message, you are good to go.

## Usage

Congratulations, you are now prepared to run your natural language experiments using one of the most powerful word embeddings. You can load the glove vocabulary and embedding matrix using the following command.

```
brandon@btrabucco.com:~/glove$ python3
>>> import glove
>>> import glove.configuration
>>> config = glove.configuration.Configuration(...)
>>> vocab, embeddings = glove.load(config)
Loading embeddings from some/directory/to/embeddings.txt.
Vocab initialized with ... words.
```

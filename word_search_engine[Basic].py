import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk import ngrams
import scipy
import numpy as np
from tqdm import tqdm
import time

class wse:
    def clean_punc(self,sent):
        cleaned = re.sub(r'[?|!|\'|"|#]',r'',sent)
        cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
        return  cleaned

    def text_preprossesing(self,text):
        stop=set(stopwords.words('english'))

        from tqdm import tqdm
        cleaned_data = []
        for word in self.clean_punc(text.lower()).split(' '):
            if word not in stop:
                cleaned_data.append(word)

        new_text = ' '.join(cleaned_data)

        new_text = new_text.replace(':','').replace('-','').replace("“",'').replace("”",'').replace("\n",'').replace('’','').replace('—','').replace('  ',' ')
        
        return new_text

    # function to create lookup dictionary and count vectorized of the complete corpus
    def create_lookup(self,new_text):
        # getting all the words from the corpus 
        cv = CountVectorizer()
        cv.fit([new_text])
        word = cv.get_feature_names()
        new_word = []
        
        # neglecting any numerical value [based on requirement]
        for i in word:
            try:
                int(i)
            except:
                new_word.append(i)
        
        # for creating vectorizer of ngrams 
        ngram_list_cv = []
        # index of words based on the presence of ngram for optimization
        ngram_list = {}
        
        for i in range(len(new_word)):
            ngram = []
            for j in ngrams(new_word[i],3):
                ngram.append(''.join(j))
            ngram_list_cv.append(' '.join(ngram)) 
            ngram_list[new_word[i]] = ngram
        
        all_ngram = []
        for i in ngram_list:
            for j in ngram_list[i]:
                if j not in all_ngram:
                    all_ngram.append(j)
                    
        ngram_word_lookup = {}
        for i in all_ngram:
            for j in ngram_list:
                if i in ngram_list[j]:
                    if i in ngram_word_lookup:
                        ngram_word_lookup[i].append(new_word.index(j))
                    else:
                        ngram_word_lookup[i] = [new_word.index(j)]
                        
        cv_new = CountVectorizer()
        count = cv_new.fit_transform(ngram_list_cv)
        
        # count vectors based on the presence of ngrams for each word
        ngram_look = {}
        import numpy as np
        for i in range(count.shape[0]):
            ngram_look[new_word[i]] = np.asarray(count.todense())[i]
            
        print("Total Words::",len(new_word))
        print("Total ngrams::",len(all_ngram))
        # ngram_look --> vectors of each word based on there ngrams
        # ngram_word_lookup --> mapping of ngrams to there respected words index in list
        # new_word --> all words that are present in new_text
        # cv_new --> fitted vectorized on text_data
        return ngram_look,ngram_word_lookup,new_word,cv_new

    # function that returs the most similar word wiht similarity value
    def givesim(self,term,ngram_look,ngram_word_lookup,new_word,cv_new):
        test_list = []
        ngram = []
        # creating ngram vector for the test text 
        for j in ngrams(term,3):
            ngram.append(''.join(j))
        
        test_list_search = ngram
        test_list.append(' '.join(ngram))
        k = cv_new.transform(test_list)
        test_vect = (np.asarray(k.todense()))[0]
        
        # getting the related terms based on ngrams of the testing text
        searching_terms = []
        for i in test_list_search:
            if i in ngram_word_lookup:
                for j in ngram_word_lookup[i]:
                    if j not in searching_terms:
                        searching_terms.append(new_word[j])
        
        # calculating similarity only on related terms that have common ngram
        sim_dict = {}
        for ngram_term in searching_terms:
            sim = 1-scipy.spatial.distance.cosine(ngram_look[ngram_term].reshape(-1,1), test_vect.reshape(-1,1))
            sim_dict[ngram_term] = sim 
        
        # sorting dictionary based on similatiy value and returing the highest similarity text 
        final_dict = {k: v for k, v in sorted(sim_dict.items(), key=lambda item: item[1],reverse = True)}
        best_term = max(final_dict, key=final_dict.get)
        return {'input':term,'predicted':{best_term:final_dict[best_term]}}
        

searcher = wse()
start_time = time.time()
new_text = searcher.text_preprossesing('natural language processing (nlp) is a subfield of linguistics, computer science, information engineering, and artificial intelligence concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data.\nchallenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation.\nthe history of natural language processing (nlp) generally started in the 1950s, although work can be found from earlier periods.\nin 1950, alan turing published an article titled "computing machinery and intelligence" which proposed what is now called the turing test as a criterion of intelligence[clarification needed].\nthe georgetown experiment in 1954 involved fully automatic translation of more than sixty russian sentences into english. the authors claimed that within three or five years, machine translation would be a solved problem.[2]  however, real progress was much slower, and after the alpac report in 1966, which found that ten-year-long research had failed to fulfill the expectations, funding for machine translation was dramatically reduced.  little further research in machine translation was conducted until the late 1980s when the first statistical machine translation systems were developed.\nsome notably successful natural language processing systems developed in the 1960s were shrdlu, a natural language system working in restricted "blocks worlds" with restricted vocabularies, and eliza, a simulation of a rogerian psychotherapist, written by joseph weizenbaum between 1964 and 1966.  using almost no information about human thought or emotion, eliza sometimes provided a startlingly human-like interaction. when the "patient" exceeded the very small knowledge base, eliza might provide a generic response, for example, responding to "my head hurts" with "why do you say your head hurts?".\nduring the 1970s, many programmers began to write "conceptual ontologies", which structured real-world information into computer-understandable data.  examples are margie (schank, 1975), sam (cullingford, 1978), pam (wilensky, 1978), talespin (meehan, 1976), qualm (lehnert, 1977), politics (carbonell, 1979), and plot units (lehnert 1981).  during this time, many chatterbots were written including parry, racter, and jabberwacky.\nup to the 1980s, most natural language processing systems were based on complex sets of hand-written rules.  starting in the late 1980s, however, there was a revolution in natural language processing with the introduction of machine learning algorithms for language processing.  this was due to both the steady increase in computational power (see moore\'s law) and the gradual lessening of the dominance of chomskyan theories of linguistics (e.g. transformational grammar), whose theoretical underpinnings discouraged the sort of corpus linguistics that underlies the machine-learning approach to language processing.[3] some of the earliest-used machine learning algorithms, such as decision trees, produced systems of hard if-then rules similar to existing hand-written rules.  however, part-of-speech tagging introduced the use of hidden markov models to natural language processing, and increasingly, research has focused on statistical models, which make soft, probabilistic decisions based on attaching real-valued weights to the features making up the input data. the cache language models upon which many speech recognition systems now rely are examples of such statistical models.  such models are generally more robust when given unfamiliar input, especially input that contains errors (as is very common for real-world data), and produce more reliable results when integrated into a larger system comprising multiple subtasks.\nmany of the notable early successes occurred in the field of machine translation, due especially to work at ibm research, where successively more complicated statistical models were developed.  these systems were able to take advantage of existing multilingual textual corpora that had been produced by the parliament of canada and the european union as a result of laws calling for the translation of all governmental proceedings into all official languages of the corresponding systems of government.  however, most other systems depended on corpora specifically developed for the tasks implemented by these systems, which was (and often continues to be) a major limitation in the success of these systems. as a result, a great deal of research has gone into methods of more effectively learning from limited amounts of data.\nrecent research has increasingly focused on unsupervised and semi-supervised learning algorithms.  such algorithms can learn from data that has not been hand-annotated with the desired answers or using a combination of annotated and non-annotated data.  generally, this task is much more difficult than supervised learning, and typically produces less accurate results for a given amount of input data.  however, there is an enormous amount of non-annotated data available (including, among other things, the entire content of the world wide web), which can often make up for the inferior results if the algorithm used has a low enough time complexity to be practical.\nin the 2010s, representation learning and deep neural network-style machine learning methods became widespread in natural language processing, due in part to a flurry of results showing that such techniques[4][5] can achieve state-of-the-art results in many natural language tasks, for example in language modeling,[6] parsing,[7][8] and many others. popular techniques include the use of word embeddings to capture semantic properties of words, and an increase in end-to-end learning of a higher-level task (e.g., question answering) instead of relying on a pipeline of separate intermediate tasks (e.g., part-of-speech tagging and dependency parsing). in some areas, this shift has entailed substantial changes in how nlp systems are designed, such that deep neural network-based approaches may be viewed as a new paradigm distinct from statistical natural language processing. for instance, the term neural machine translation (nmt) emphasizes the fact that deep learning-based approaches to machine translation directly learn sequence-to-sequence transformations, obviating the need for intermediate steps such as word alignment and language modeling that was used in statistical machine translation (smt).\nin the early days, many language-processing systems were designed by hand-coding a set of rules:[9][10] such as by writing grammars or devising heuristic rules for stemming. \nsince the so-called "statistical revolution"[11][12] in the late 1980s and mid-1990s, much natural language processing research has relied heavily on machine learning. the machine-learning paradigm calls instead for using statistical inference to automatically learn such rules through the analysis of large corpora (the plural form of corpus, is a set of documents, possibly with human or computer annotations) of typical real-world examples.\nmany different classes of machine-learning algorithms have been applied to natural-language-processing tasks. these algorithms take as input a large set of "features" that are generated from the input data. some of the earliest-used algorithms, such as decision trees, produced systems of hard if-then rules similar to the systems of handwritten rules that were then common. increasingly, however, research has focused on statistical models, which make soft, probabilistic decisions based on attaching real-valued weights to each input feature. such models have the advantage that they can express the relative certainty of many different possible answers rather than only one, producing more reliable results when such a model is included as a component of a larger system.\nsystems based on machine-learning algorithms have many advantages over hand-produced rules:\nthe following is a list of some of the most commonly researched tasks in natural language processing. some of these tasks have direct real-world applications, while others more commonly serve as subtasks that are used to aid in solving larger tasks.\nthough natural language processing tasks are closely intertwined, they are frequently subdivided into categories for convenience. a coarse division is given below.\nthe first published work by an artificial intelligence was published in 2018, 1 the road, marketed as a novel, contains sixty million words.\n')
#new_text = text_preprossesing(text)
ngram_look, ngram_word_lookup ,new_word, cv_new = searcher.create_lookup(new_text)
print("DONE --- CREATING_LOOKUP_DICT --- %s seconds ---" % (time.time() - start_time))

# checking with some wrong words 
start_time = time.time()
print(searcher.givesim('languge',ngram_look,ngram_word_lookup,new_word,cv_new))
print(searcher.givesim('procecing',ngram_look,ngram_word_lookup,new_word,cv_new))
print(searcher.givesim('lingistics',ngram_look,ngram_word_lookup,new_word,cv_new))
print(searcher.givesim('informetion',ngram_look,ngram_word_lookup,new_word,cv_new))
print("DONE --- PREDICTING_SIMILAR_WORD --- %s seconds ---" % (time.time() - start_time))

"""
=================OUTPUT==================
Total Words:: 451
Total ngrams:: 1093
DONE --- CREATING_LOOKUP_DICT --- 1.066976547241211 seconds ---
{'input': 'languge', 'predicted': {'language': 0.7071067811865475}}
{'input': 'procecing', 'predicted': {'processing': 0.5773502691896258}}
{'input': 'lingistics', 'predicted': {'linguistics': 0.7559289460184545}}
{'input': 'informetion', 'predicted': {'information': 0.6666666666666667}}
DONE --- PREDICTING_SIMILAR_WORD --- 0.08927679061889648 seconds ---
"""

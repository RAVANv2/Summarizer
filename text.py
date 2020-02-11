from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

from flask import Flask,request,jsonify

app = Flask(__name__)

@app.route('/question')
def general():
    def read_article(file_name):
        file = open(file_name,'r')
        filedata = file.readlines()
        article = filedata[0].split(".")
        sentences = []
        
        for sentence in article:
        # print(sentence)
            sentences.append(sentence.replace("^a-zA-Z]"," ").split(" "))
        sentences.pop()
        
        return sentences
    #Now it returns list of words which appear in sentence

    def build_sililarity_matrix(sentences,stopwords):
        #Empty silmilarity matrix
        similarity_matrix = np.zeros((len(sentences),len(sentences)))
        
        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1==idx2:
                    continue
                similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1],sentences[idx2],stopwords)
        return similarity_matrix


    def sentence_similarity(sent1,sent2,stopwords=None):
        if stopwords is None:
            stopwords = []
            
        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]
        
        all_words = list(set(sent1 + sent2))
        
        vector1 = [0]*len(all_words)
        vector2 = [0]*len(all_words)
        
        #build the vector for first sequence
        for w in sent1:
            if w in stopwords:
                continue
            vector1[all_words.index(w)] += 1
            
        #build the vector for second sequence
        for w in sent2:
            if w in stopwords:
                continue
            vector2[all_words.index(w)] += 1
            
        return 1 - cosine_distance(vector1,vector2)

    #Doubt 1: Why they find the similarity between counts of words
    #Doubt 2: 1-cosine_distance if cosine_distance<0 then ??

    def generate_summary(file_name,top_n=4):
        stop_words = stopwords.words('english')
        summarize_text = []
        
        #step-1 -> Read Text and split it
        sentences = read_article(file_name)
        
        #step-2 -> Generate similarity matrix accross sentences
        sentence_similarity_matrix = build_sililarity_matrix(sentences,stop_words)
        
        #step-3 -> Rank Sentences in Similarity Matrix
        sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
        scores = nx.pagerank(sentence_similarity_graph)
        
        #step-4 -> sort the rank and pick top sentences
        ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
        #print("Indexes of top ranked_sentence order are ", ranked_sentence) 
        
        for i in range(top_n):
            summarize_text.append(" ".join(ranked_sentence[i][1]))
    #     print(summarize_text)
        
        # Step 5 - output summerizer
        print("Summarize Text: \n", ".".join(summarize_text))
        data = request.get_json(force=True)

    return jsonify(generate_summary(data))

if __name__ == "__main__":
    app.run(port=8000,debug=True)


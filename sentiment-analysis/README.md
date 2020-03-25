---


---

<h1 id="sentiment-analysis">Sentiment Analysis</h1>
<p>Training a stacked LSTM - MLP system to predict numerical ratings of movie reviews.<br>
<strong>Language:</strong> Julia<br>
<strong>Libraries:</strong> <a href="https://fluxml.ai/Flux.jl/stable/" title="Flux">Flux</a><br>
<strong>Dataset:</strong> <a href="https://nlp.stanford.edu/sentiment/code.html" title="Stanford Sentiment Treebank">Stanford Sentiment Treebank</a></p>
<h2 id="structure">Structure</h2>
<ul>
<li>Stopwords are removed from the reviews</li>
<li>Words are encoded into vectors of length 300 using word2vec from the python <a href="https://pypi.org/project/embeddings/">embeddings</a> package</li>
<li>Each word is fed into a LSTM (input 300, output 120), starting with the first word in the review and ending with the last</li>
<li>The final output of the LSTM is fed into a MLP (layer sizes 120-120-5), to produce a rating from 1 to 5 for that review</li>
</ul>
<p>The loss function is defined for this structure, so both the LSTM and MLP are trained at once:</p>
<pre><code>function loss(xbatch, ybatch)
    n_words = size(xbatch)[3]
    Flux.reset!(accumulator)
    for w in 1:n_words-1
        accumulator(xbatch[:,:,w])
    end
    l = Flux.crossentropy(classifier(accumulator(xbatch[:,:,n_words])), ybatch)
    return l/size(xbatch)[2]
    end
</code></pre>
<h2 id="results">Results</h2>
<p>The final accuracy of the system was 41%. It was effective at determining the ranking of clearly positive or negative reviews, as shown by the sample results below:</p>
<p><code>Dunkirk: One of the most captivating and compelling films of the year so far .</code><br>
<strong><code>Predicted rating: 5</code></strong></p>
<p><code>Max Max: Fury Road: Arty , gorgeous , exciting , compelling , and poignant all at once .</code><br>
<strong><code>Predicted rating: 5</code></strong></p>
<p><code>Thor: Ragnarok: A great film that will definitely entertain you and keep a smile on your face .</code><br>
<strong><code>Predicted rating: 4</code></strong></p>
<p><code>Meet the Spartans: This was the worst movie I 've ever seen , so bad that I hesitate to label it a ' movie ' and thus reflect shame upon the entire medium of film .</code><br>
<strong><code>Predicted rating: 1</code></strong></p>
<p><code>Santa Claus Conquers the Martians: The plot , such as it is , proves it is possible to insult the intelligence of a three - year - old</code><br>
<strong><code>Predicted rating: 2</code></strong></p>


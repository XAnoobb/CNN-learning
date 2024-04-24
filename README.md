# CNN-learning
Learning how to lighten CNN neural networks and other attempts to manipulate them

<img width="446" alt="ff25cdbddacd56192695258f33224d28" src="https://github.com/XAnoobb/CNN-learning/assets/63050109/dcc3a7ab-da41-4b50-af60-6be960b4b878">

The following four functions provided by pytorch in structural and non-structural pruning were used to evaluate the performance after pruning for digit recognition of MNIST datasets implemented using CNN networks

random_unstructured(module, name, amount)
cnn_randunstrc_performance.py 
--> cnn_randunstrc_result

l1_unstructured(module, name, amount, importance_scores=None)
cnn_l1prune_performance.py 
--> cnn_l1prune_result

random_structured(module, name, amount, dim)
cnn_randstrc_performance.py 
--> cnn_randstrc_result

ln_structured(module, name, amount, n, dim, importance_scores=None)
cnn_lnstrc_performance.py 
--> cnn_lnstrc_result

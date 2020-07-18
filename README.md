# List of Resources
A collaborative list of resources for Computational Neuroscience


## Interesting Papers/ Articles/ Blog Posts:

### Information Theory

* Foundational paper in the field of information theory by Claude Shannon in 1948 [A Mathematical Theory of Communication](http://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)
>>This work developed the concepts of information entropy and redundancy, and introduced the term bit (which Shannon credited to John Tukey) as a unit of information. It was also in this paper that the Shannon–Fano coding technique was proposed – a technique developed in conjunction with Robert Fano. <br>
Shannon's article laid out the basic elements of communication:
>> * An information source that produces a message
>> * A transmitter that operates on the message to create a signal which can be sent through a channel
>> * A channel, which is the medium over which the signal, carrying the information that composes the message, is sent
>> * A receiver, which transforms the signal back into the message intended for delivery
>> * A destination, which can be a person or a machine, for whom or which the message is intended

### Entropy
* [Entropy explained](https://towardsdatascience.com/the-intuition-behind-shannons-entropy-e74820fe9800) with some python implimentation.

### Noise
* [Noise in the nervous system by A. Aldo Faisal et al](http://learning.eng.cam.ac.uk/pub/Public/Wolpert/Publications/FaiSelWol08.pdf). Looks like a really interesting and has good explainations on understanding what noise actually is.

### Brain Oscillations
* Traveling waves in the brain and understanding their propagation. Seems to focus on alpha (10 Hz). [The Hidden Spatial Dimension of Alpha: 10-Hz Perceptual Echoes Propagate as Periodic Traveling Waves in the Human Brain](https://www.cell.com/cell-reports/fulltext/S2211-1247(18)32003-5?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS2211124718320035%3Fshowall%3Dtrue)


### Dimensionality
* [Towards the neural population doctrine](https://stat.columbia.edu/~cunningham/pdf/SaxenaCONB2019.pdf)
>> We detail four areas of the field where the joint analysis of neural populations has significantly furthered our understanding of computation in the brain: correlated variability, decoding, neural dynamics, and artificial neural networks.

* [Perform non-linear dimensionality reduction with Isomap and LLE in Python from scratch](https://towardsdatascience.com/step-by-step-signal-processing-with-machine-learning-manifold-learning-8e1bb192461c)

* [Isomap tutorial](https://jakevdp.github.io/PythonDataScienceHandbook/05.02-introducing-scikit-learn.html)

* Looking at different non-linear dimensionality reductions methods: [Iterative Non-linear Dimensionality Reduction with Manifold Sculpting.](https://papers.nips.cc/paper/3241-iterative-non-linear-dimensionality-reduction-with-manifold-sculpting.pdf)
>> Many algorithms have been recently developed for reducing dimensionality by projecting data onto an intrinsic non-linear manifold. Unfortunately, existing algorithms often lose significant precision in this transformation. Manifold Sculpting is a new algorithm that iteratively reduces dimensionality by simulating surface tension in local neighborhoods. We present several experiments that show Manifold Sculpting yields more accurate results than existing algorithms with both generated and natural data-sets. Manifold Sculpting is also able to benefit from both prior dimensionality reduction efforts.

* Using manifolds/ dimensionality reduction on sleep data. [The intrinsic attractor manifold and population dynamics of a canonical cognitive circuit across waking and sleep](https://www.nature.com/articles/s41593-019-0460-x)
>> We characterize and directly visualize manifold structure in the mammalian head direction circuit, revealing that the states form a topologically nontrivial one-dimensional ring. The ring exhibits isometry and is invariant across waking and rapid eye movement sleep. This result directly demonstrates that there are continuous attractor dynamics and enables powerful inference about mechanism. 

* [A Global Geometric Framework for Nonlinear Dimensionality Reduction](https://science.sciencemag.org/content/290/5500/2319)
>> Here we describe an approach to solving dimensionality reduction problems that uses easily measured local metric information to learn the underlying global geometry of a data set. Unlike classical techniques such as principal component analysis (PCA) and multidimensional scaling (MDS), our approach is capable of discovering the nonlinear degrees of freedom that underlie complex natural observations, such as human handwriting or images of a face under different viewing conditions. In contrast to previous algorithms for nonlinear dimensionality reduction, ours efficiently computes a globally optimal solution, and, for an important class of data manifolds, is guaranteed to converge asymptotically to the true structure.

### Modeling

* A guide for applying [Machine learning for neural decoding](https://arxiv.org/ftp/arxiv/papers/1708/1708.00909.pdf). 
>> Description: This	 tutorial	 describes	 how	 to effectively	 apply	 these	 algorithms	 for	 typical	 decoding	 problems.	 We	 provide	 descriptions,	 best practices,	and	code	for	applying	common	machine	learning	methods,	including	neural	networks	and	gradient	boosting.	We	also	provide	detailed	comparisons	of	the	performance	of	various	methods	at the	 task	 of	 decoding	 spiking	 activity	 in	 motor	 cortex,	 somatosensory	 cortex,	 and	 hippocampus.


* [A How-to-Model Guide for Neuroscience](https://www.eneuro.org/content/7/1/ENEURO.0352-19.2019). Steps on how to go about posing questions that models can answer. 

* [Direct Fit to Nature: An Evolutionary Perspective on Biological and Artificial Neural Networks](https://www.sciencedirect.com/science/article/abs/pii/S089662731931044X)

* Preprint on [Neural Network Poisson Models for Behavioural and Neural Spike Train Data](https://www.biorxiv.org/content/10.1101/2020.07.13.201673v1.abstract) by Dayan's group. 

* Maximum likelihood estimation for neural data [slide deck](http://pillowlab.princeton.edu/teaching/statneuro2018/slides/slides07_encodingmodels.pdf) by Jonathan Pillow. A walk through of the concept and derivation.


#### Optimization
* **BOOK**: [Resources by Boyd and Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/) covering convex optimization and appproaches.

### Machine Learning 

* [Reconciling modern machine-learning practice and the classical bias–variance trade-off](https://www.pnas.org/content/116/32/15849)



## Books 

* [Mathematics for Machine Learning by A. Aldo Faisal, Cheng Soon Ong, and Marc Peter Deisenroth](https://mml-book.github.io/book/mml-book.pdf)

* [Information Theory, Inference, and Learning Algorithms](https://www.inference.org.uk/itprnn/book.pdf). See chapter 28 for Model Comparison and Occam’s Razor for model simplicity. 

* [Pattern Recognition and Machine Learning by Bishop](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf)

* [Theoretical Neuroscience](http://www.gatsby.ucl.ac.uk/~lmate/biblio/dayanabbott.pdf) by Dayan and Abbott.

* [Introduction to Applied Linear Algebra Vectors, Matrices, and Least Squares](http://vmls-book.stanford.edu/vmls.pdf) by Boyd and Vandenberghe.

## Datasets

* A list of open [datasets](https://github.com/openlists/ElectrophysiologyData) that span EEG, MEG, ECoG, and LFP. 

* A large list of [BCI resources](https://github.com/NeuroTechX/awesome-bci#brain-databases) including datasets, tutorials, papers, books etc. 

* The [TUH EEG Corpus](https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml), a list of several EEG dataset with several resources. Requies filling out form to download the data.  

* [Project Tycho named after Tycho Brache](). The project aims to share reliable massive neural and behavioral data for understanding brain mechanism.
>>**[Tycho Brahe](https://en.wikipedia.org/wiki/Tycho_Brahe)** was a Danish nobleman, astronomer, and writer known for his accurate and comprehensive astronomical observations. He was born in the then Danish peninsula of Scania. Tycho was well known in his lifetime as an astronomer, astrologer, and alchemist.

## Videos

* [Gradients of Brain Organization Workshop](https://www.mcgill.ca/neuro/channels/event/virtual-gradients-brain-organization-workshop-zoom-302746). 
>> Description: Recent years have seen a rise of new methods and applications to study smooth spatial transitions — or gradients — of brain organization. Identification and analysis of cortical gradients provides a framework to study brain organization across species, to examine changes in brain development and aging, and to more generally study the interrelation between brain structure, function and cognition. We will bring together outstanding junior and senior scientists to discuss the challenges and opportunities afforded by this emerging perspective.


## Memes

<img src="https://memegenerator.net/img/instances/26848932/occams-razor-the-simplest-explanation-is-almost-always-somebody-screwed-up.jpg" width="200" height="200" />

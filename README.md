# List of Resources
A collaborative list of resources for Computational Neuroscience






## Interesting Papers/ Articles/ Blog Posts:



### Information Theory


* Foundational paper in the field of information theory by Claude Shannon in 1948 [A Mathematical Theory of Communication](http://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf). Might be helpful to [watch this video by Kan Academy](https://www.khanacademy.org/computing/computer-science/informationtheory/moderninfotheory/v/a-mathematical-theory-of-communication) describing the work (from Markov Chain perspective) before diving into the paper. 

  <details>
  <summary> Details!  </summary>
  
    This work developed the concepts of information entropy and redundancy, and introduced the term bit (which Shannon credited to John Tukey) as a unit of information. It was also in this paper that the Shannon–Fano coding technique was proposed – a technique developed in conjunction with Robert Fano. <br>
    Shannon's article laid out the basic elements of communication:
    * An information source that produces a message
    * A transmitter that operates on the message to create a signal which can be sent through a channel
    * A channel, which is the medium over which the signal, carrying the information that composes the message, is sent
    * A receiver, which transforms the signal back into the message intended for delivery
    * A destination, which can be a person or a machine, for whom or which the message is intended
    
    [More on Shannon](https://thebitplayer.com/#more-information) and his contributions to the world of Computer sci, entropy, info theory, signal detection etc.

  </details>
  
* Ian Goodfellow's (developed GANs) [Book Chapter](https://www.deeplearningbook.org/contents/prob.html) on Information Theory from a Deep Learning Perspective
  <details>
  <summary> Details!  </summary>
  
  Goodfellow is best known for inventing **generative adversarial networks (GANs)**. He is also the lead author of the textbook Deep Learning. At Google, he developed a system enabling Google Maps to automatically transcribe addresses from photos taken by Street View cars and demonstrated security vulnerabilities of machine learning systems.
  </details>



### Entropy
 
* [Entropy explained](https://towardsdatascience.com/the-intuition-behind-shannons-entropy-e74820fe9800) (or randomness) with some python implimentation.




### Noise
* [Noise in the nervous system by A. Aldo Faisal et al](http://learning.eng.cam.ac.uk/pub/Public/Wolpert/Publications/FaiSelWol08.pdf). Looks like a really interesting and has good explainations on understanding what noise actually is.

### Brain Oscillations
* Traveling waves in the brain and understanding their propagation. Seems to focus on alpha (10 Hz). [The Hidden Spatial Dimension of Alpha: 10-Hz Perceptual Echoes Propagate as Periodic Traveling Waves in the Human Brain](https://www.cell.com/cell-reports/fulltext/S2211-1247(18)32003-5?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS2211124718320035%3Fshowall%3Dtrue)




### Dimensionality
#### _*General Dimnesionality*_
* [Towards the neural population doctrine](https://stat.columbia.edu/~cunningham/pdf/SaxenaCONB2019.pdf)
  <details>
    <summary> Details!  </summary>

    We detail four areas of the field where the joint analysis of neural populations has significantly furthered our understanding of computation in the brain: correlated variability, decoding, neural dynamics, and artificial neural networks.

  </details>

* [SVD and PCA explained](https://www.cns.nyu.edu/~david/handouts/svd.pdf). Handout walking through the math behind both and a few other topics (regression,covariance etc.). 
  <details>
    <summary> Details!  </summary>

    This handout is a review of some basic concepts in linear algebra. For a detailed introduction, consult a linear algebra text. Linear Algebra and its Applications by Gilbert Strang (Harcourt, Brace, Jovanovich, 1988) is excellent.

  </details>

#### _*Non-Linear Dimensionality Reduction*_
* [Using t-SNE](https://distill.pub/2016/misread-tsne/). An interactive guide on how to use t-SNE effectively
  <details>
    <summary> Details!  </summary>
    Although extremely useful for visualizing high-dimensional data, t-SNE plots can sometimes be mysterious or misleading. By exploring how it behaves in simple cases, we can learn to use it more effectively.
  </details>

* [Perform non-linear dimensionality reduction with Isomap and LLE in Python from scratch](https://towardsdatascience.com/step-by-step-signal-processing-with-machine-learning-manifold-learning-8e1bb192461c)

* [Isomap tutorial in Python](https://jakevdp.github.io/PythonDataScienceHandbook/05.02-introducing-scikit-learn.html)

* Looking at different non-linear dimensionality reductions methods: [Iterative Non-linear Dimensionality Reduction with Manifold Sculpting.](https://papers.nips.cc/paper/3241-iterative-non-linear-dimensionality-reduction-with-manifold-sculpting.pdf)
  <details>
    <summary> Details!  </summary>

    Many algorithms have been recently developed for reducing dimensionality by projecting data onto an intrinsic non-linear manifold. Unfortunately, existing algorithms often lose significant precision in this transformation. Manifold Sculpting is a new algorithm that iteratively reduces dimensionality by simulating surface tension in local neighborhoods. We present several experiments that show Manifold Sculpting yields more accurate results than existing algorithms with both generated and natural data-sets. Manifold Sculpting is also able to benefit from both prior dimensionality reduction efforts.
  </details>

* Using manifolds/ dimensionality reduction on sleep data. [The intrinsic attractor manifold and population dynamics of a canonical cognitive circuit across waking and sleep](https://www.nature.com/articles/s41593-019-0460-x)
  <details>
    <summary> Details!  </summary>

    We characterize and directly visualize manifold structure in the mammalian head direction circuit, revealing that the states form a topologically nontrivial one-dimensional ring. The ring exhibits isometry and is invariant across waking and rapid eye movement sleep. This result directly demonstrates that there are continuous attractor dynamics and enables powerful inference about mechanism. 
  </details>

* [A Global Geometric Framework for Nonlinear Dimensionality Reduction](https://science.sciencemag.org/content/290/5500/2319)
  <details>
    <summary> Details!  </summary>

    Here we describe an approach to solving dimensionality reduction problems that uses easily measured local metric information to learn the underlying global geometry of a data set. Unlike classical techniques such as principal component analysis (PCA) and multidimensional scaling (MDS), our approach is capable of discovering the nonlinear degrees of freedom that underlie complex natural observations, such as human handwriting or images of a face under different viewing conditions. In contrast to previous algorithms for nonlinear dimensionality reduction, ours efficiently computes a globally optimal solution, and, for an important class of data manifolds, is guaranteed to converge asymptotically to the true structure.
  </details>





### Modeling
#### _*General Modeling*_
* A guide for applying [Machine learning for neural decoding](https://arxiv.org/ftp/arxiv/papers/1708/1708.00909.pdf). 
  <details>
    <summary> Details!  </summary> 

    Description: This	 tutorial	 describes	 how	 to effectively	 apply	 these	 algorithms	 for	 typical	 decoding	 problems.	 We	 provide	 descriptions,	 best practices,	and	code	for	applying	common	machine	learning	methods,	including	neural	networks	and	gradient	boosting.	We	also	provide	detailed	comparisons	of	the	performance	of	various	methods	at the	 task	 of	 decoding	 spiking	 activity	 in	 motor	 cortex,	 somatosensory	 cortex,	 and	 hippocampus.
  </details>
  
* [Stochastic dynamics as a principle of brain function](https://www.oxcns.org/papers/463_Deco+Rolls+09StochasticDynamics.pdf)
  <details>
    <summary> Details!  </summary> 

    We show that in a finite-sized cortical attractor network, this can be an advantage, for it leads to probabilistic behavior that is advantageous in decision-making, by preventing deadlock, and is important in signal detectability. We show how computations can be performed through stochastic dynamical effects, including the role of noise in enabling probabilistic jumping across barriers in the energy landscape describing the flow of the dynamics in attractor networks. The results obtained in neurophysiological studies of decision-making and signal detectability are modelled by the stochastical neurodynamics of integrate-and-fire networks of neurons with probabilistic neuronal spiking. We describe how these stochastic neurodynamical effects can be analyzed, and their importance in many aspects of brain function, including decision-making, memory recall, short-term memory, and attention.
  </details>



* [A How-to-Model Guide for Neuroscience](https://www.eneuro.org/content/7/1/ENEURO.0352-19.2019). Steps on how to go about posing questions that models can answer. 

* [Direct Fit to Nature: An Evolutionary Perspective on Biological and Artificial Neural Networks](https://www.sciencedirect.com/science/article/abs/pii/S089662731931044X)

* Preprint on [Neural Network Poisson Models for Behavioural and Neural Spike Train Data](https://www.biorxiv.org/content/10.1101/2020.07.13.201673v1.abstract) by Dayan's group. 

* Maximum likelihood estimation for neural data [slide deck](http://pillowlab.princeton.edu/teaching/statneuro2018/slides/slides07_encodingmodels.pdf) by Jonathan Pillow. A walk through of the concept and derivation.


#### _*Optimization for Modeling*_
* **BOOK**: [Resources by Boyd and Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/) covering convex optimization and appproaches.


#### _*Bayesian Modeling*_

* [Probabilistic population codes for Bayesian decision making](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2742921/)

  <details>
    <summary> Details!  </summary> 
    We present a neural model of decision making that can perform both evidence accumulation and action selection optimally. More specifically, we show that, given a Poisson-like distribution of spike counts, biological neural networks can accumulate evidence without loss of information through linear integration of neural activity, and can select the most likely action through attractor dynamics. This holds for arbitrary correlations, any tuning curves, continuous and discrete variables, and sensory evidence whose reliability varies over time. Our model predicts that the neurons in the lateral intraparietal cortex involved in evidence accumulation encode, on every trial, a probability distribution which predicts the animal’s performance. 

  </details>

#### _**_ 






#### _*Non/Linear Systems*_

* [Nonlinear Dynamics and Chaos by Steven Strogatz](http://arslanranjha.weebly.com/uploads/4/8/9/3/4893701/nonlinear-dynamics-and-chaos-strogatz.pdf) applied to physics, bio, chem, and engineering. Recommended by Bingni Brutton and Maria Geffen. 

* A [video lecture series](https://www.youtube.com/playlist?list=PLbN57C5Zdl6j_qJA-pARJnKsmROzPnO9V) by Steven Strongatz on Nonlinear Dynamics and Chaos: With Applications to Physics, Biology, Chemistry, and Engineering

* A [video lecture series](https://www.youtube.com/playlist?list=PLF0b3ThojznQ9xUDm-EbgFAnzdbeDVuSz) by Liz Bradley on Nonlinear Dynamics: Mathematical and Computational Approaches. [Course page.](https://www.complexityexplorer.org/courses/94-nonlinear-dynamics-mathematical-and-computational-approaches-winter-spring-2019)

### Machine Learning 
#### _*General Machine Learning*_
* [Reconciling modern machine-learning practice and the classical bias–variance trade-off](https://www.pnas.org/content/116/32/15849)

#### _*Autoencoders*_
* Variational autoencoders used with dimensionality reduction. [VAE-SNE: a deep generative model for simultaneous dimensionality reduction and clustering](https://www.biorxiv.org/content/10.1101/2020.07.17.207993v1)
  <details>
    <summary> Details!  </summary>  

    Description: We introduce a method for both dimension reduction and clustering called VAE-SNE (variational autoencoder stochastic neighbor embedding). Our model combines elements from deep learning, probabilistic inference, and manifold learning to produce interpretable compressed representations while also readily scaling to tens-of-millions of observations. Unlike existing methods, VAE-SNE simultaneously compresses high-dimensional data and automatically learns a distribution of clusters within the data --- without the need to manually select the number of clusters. This naturally creates a multi-scale representation, which makes it straightforward to generate coarse-grained descriptions for large subsets of related observations and select specific regions of interest for further analysis.
  </details>



## General Neuroscience

* [Classic must-read neuroscience papers](https://www.sfn.org/about/history-of-neuroscience/classic-papers#Learning) suggested by SfN (Society for Neuroscience). Broken down by topic. 




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
  <details>
    <summary> Details!  </summary>  

    **[Tycho Brahe](https://en.wikipedia.org/wiki/Tycho_Brahe)** was a Danish nobleman, astronomer, and writer known for his accurate and comprehensive astronomical observations. He was born in the then Danish peninsula of Scania. Tycho was well known in his lifetime as an astronomer, astrologer, and alchemist.
  </details>


* [PhysioNet](https://physionet.org/) is a large database of different types of data and most can be easily downloaded.

* [Open Neuro](https://openneuro.org/public/datasets/) an initiative to encourage the sharing of neuro data.


## Videos

* [Gradients of Brain Organization Workshop](https://www.mcgill.ca/neuro/channels/event/virtual-gradients-brain-organization-workshop-zoom-302746). 

  <details>
    <summary> Details!  </summary>  

    Description: Recent years have seen a rise of new methods and applications to study smooth spatial transitions — or gradients — of brain organization. Identification and analysis of cortical gradients provides a framework to study brain organization across species, to examine changes in brain development and aging, and to more generally study the interrelation between brain structure, function and cognition. We will bring together outstanding junior and senior scientists to discuss the challenges and opportunities afforded by this emerging perspective.

  </details>


## Jobs
List of job boards that update often and have neuro related jobs. 

* [Neuromodec](https://neuromodec.com/jobs/) gathers job in the fields of neuromodulation, engineering, neurosicence, and mental health. 

* [Researchgate](https://www.researchgate.net/jobs?page=1&regions=) job board usualy jobs in academia around the world.

* For vision and vision related jobs and posts (industry and academia) sign up for the [Vision List Mailing List](http://visionscience.com/mailman/listinfo/visionlist_visionscience.com). Note that the job board on their main site is not updated often but researchers do send out job notifications often through the mailing list. 


## Memes

<img src="https://memegenerator.net/img/instances/26848932/occams-razor-the-simplest-explanation-is-almost-always-somebody-screwed-up.jpg" width="200" height="200" />

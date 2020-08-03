# List of Resources
A collaborative list of resources for Computational Neuroscience. 






## Interesting Papers/ Articles/ Blog Posts:

## Contents

- [Information Theory](#information-theory)
- [Entropy](#entropy)
- [Noise](#noise)
- [Brain Oscillations](#brain-oscillations)
- [Causality](#causality)
- [Dimensionality](#dimensionality)
  * [General Dimnesionality](#general-dimnesionality)
  * [Non-Linear Dimensionality Reduction](#non-linear-dimensionality-reduction)
- [Modeling](#dimensionality)
  * [General Modeling](#general-modeling)
  * [Optimization for Modeling](#optimization-for-modeling)
  * [Bayesian Modeling](#bayesian-modeling)
  * [Linear and Non-Linear Systems](#linear-and-non-linear-systems)
  * [Markov Processes](#markov-processes)
  * [Control Theory](#control-theory)  
- [Machine Learning](#machine-learning)
  * [General Machine Learning](#general-machine-learning)
  * [Autoencoders](#autoencoders)
  * [Reinforcement Learning](#reinforcement-learning)
- [General Neuroscience](#general-neuroscience) 
- [Books](#books) 
- [Datasets](#datasets) 
- [Videos](#videos) 
- [Jobs](#jobs) 
- [Memes](#memes) 

<!-- toc -->





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





### Causality 

* [Quasi-experimental causality in neuroscience and behavioural research](https://www.nature.com/articles/s41562-018-0466-5)
  <details>
    <summary> Details!  </summary>

    In many scientific domains, causality is the key question. For example, in neuroscience, we might ask whether a medication affects perception, cognition or action. Randomized controlled trials are the gold standard to establish causality, but they are not always practical. The field of empirical economics has developed rigorous methods to establish causality even when randomized controlled trials are not available. Here we review these quasi-experimental methods and highlight how neuroscience and behavioural researchers can use them to do research that can credibly demonstrate causal effects.

  </details>


* [A Causal Network Analysis of Neuromodulation in the Cortico-Subcortical Limbic Network](https://www.cell.com/neuron/fulltext/S0896-6273(20)30466-9#.XwXvXEHciHI.twitter) applied to neurons. 
  <details>
    <summary> Details!  </summary>

    Neural decoding and neuromodulation technologies hold great promise for treating mood and other brain disorders in next-generation therapies that manipulate functional brain networks. Here we perform a novel causal network analysis to decode multiregional communication in the primate mood processing network and determine how neuromodulation, short-burst tetanic microstimulation (sbTetMS), alters multiregional network communication. The causal network analysis revealed a mechanism of network excitability that regulates when a sender stimulation site communicates with receiver sites. Decoding network excitability from neural activity at modulator sites predicted sender-receiver communication, whereas sbTetMS neuromodulation temporarily disrupted sender-receiver communication. These results reveal specific network mechanisms of multiregional communication and suggest a new generation of brain therapies that combine neural decoding to predict multiregional communication with neuromodulation to disrupt multiregional communication.

  </details>


* [Advancing functional connectivity research from association to causation](https://www.nature.com/articles/s41593-019-0510-4)
  <details>
    <summary> Details!  </summary>

    Cognition and behavior emerge from brain network interactions, such that investigating causal interactions should be central to the study of brain function. Approaches that characterize statistical associations among neural time series—functional connectivity (FC) methods—are likely a good starting point for estimating brain network interactions. Yet only a subset of FC methods (‘effective connectivity’) is explicitly designed to infer causal interactions from statistical associations. Here we incorporate best practices from diverse areas of FC research to illustrate how FC methods can be refined to improve inferences about neural mechanisms, with properties of causal neural interactions as a common ontology to facilitate cumulative progress across FC approaches. We further demonstrate how the most common FC measures (correlation and coherence) reduce the set of likely causal models, facilitating causal inferences despite major limitations. Alternative FC measures are suggested to immediately start improving causal inferences beyond these common FC measures.

  </details>




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

* [A Short Introduction to Bayesian Neural Networks](https://davidstutz.de/a-short-introduction-to-bayesian-neural-networks/)
  <details>
    <summary> Details!  </summary> 

    With the rising success of deep neural networks, their reliability in terms of robustness (for example, against various kinds of adversarial examples) and confidence estimates becomes increasingly important. Bayesian neural networks promise to address these issues by directly modeling the uncertainty of the estimated network weights. In this article, I want to give a short introduction of training Bayesian neural networks, covering three recent approaches.
  </details>



#### _*Optimization for Modeling*_
* **BOOK**: [Resources by Boyd and Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/) covering convex optimization and appproaches.






#### _*Bayesian Modeling*_

* [Probabilistic population codes for Bayesian decision making](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2742921/)

  <details>
    <summary> Details!  </summary> 
    We present a neural model of decision making that can perform both evidence accumulation and action selection optimally. More specifically, we show that, given a Poisson-like distribution of spike counts, biological neural networks can accumulate evidence without loss of information through linear integration of neural activity, and can select the most likely action through attractor dynamics. This holds for arbitrary correlations, any tuning curves, continuous and discrete variables, and sensory evidence whose reliability varies over time. Our model predicts that the neurons in the lateral intraparietal cortex involved in evidence accumulation encode, on every trial, a probability distribution which predicts the animal’s performance. 

  </details>







#### _*Linear and Non-Linear Systems*_

* [Nonlinear Dynamics and Chaos by Steven Strogatz](http://arslanranjha.weebly.com/uploads/4/8/9/3/4893701/nonlinear-dynamics-and-chaos-strogatz.pdf) applied to physics, bio, chem, and engineering. Recommended by Bingni Brutton and Maria Geffen. 

* A [video lecture series](https://www.youtube.com/playlist?list=PLbN57C5Zdl6j_qJA-pARJnKsmROzPnO9V) by Steven Strongatz on Nonlinear Dynamics and Chaos: With Applications to Physics, Biology, Chemistry, and Engineering

* A [video lecture series](https://www.youtube.com/playlist?list=PLF0b3ThojznQ9xUDm-EbgFAnzdbeDVuSz) by Liz Bradley on Nonlinear Dynamics: Mathematical and Computational Approaches. [Course page.](https://www.complexityexplorer.org/courses/94-nonlinear-dynamics-mathematical-and-computational-approaches-winter-spring-2019)





#### _*Markov Processes*_

* [Light intro to Markov Chains](https://towardsdatascience.com/introduction-to-markov-chains-50da3645a50d)

* [Deeper intro to Hidden Markov Models](https://towardsdatascience.com/introduction-to-hidden-markov-models-cd2c93e6b781) with python implimentation

* [Hidden Markov Model](http://www.blackarbs.com/blog/introduction-hidden-markov-models-python-networkx-sklearn/2/9/2017) example with python implimentation using sklearn






#### _*Control Theory*_
* [The coordination of movement: optimal feedback control and beyond](http://www.diedrichsenlab.org/pubs/TICS_2010.pdf)
  <details>
    <summary> Details!  </summary> 
    Optimal control theory and its more recent extension, optimal feedback control theory, provide valuable insights into the flexible and task-dependent control of movements. Here, we focus on the problem of coordination, defined as movements that involve multiple effectors (muscles, joints or limbs). Optimal control theory makes quantitative predictions concerning the distribution of work across multiple effectors. Optimal feedback control theory further predicts variation in feedback control with changes in task demands and the correlation structure between different effectors. We highlight two crucial areas of research, hierarchical control and the problem of movement initiation, that need to be developed for an optimal feedback control theory framework to characterise movement coordination more fully and to serve as a basis for studying the neural mechanisms involved in voluntary motor control.

  </details>









### Machine Learning 
#### _*General Machine Learning*_

* Recomended learning [Relational inductive biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261)
  <details>
    <summary> Details!  </summary>  

    Artificial intelligence (AI) has undergone a renaissance recently, making major progress in key domains such as vision, language, control, and decision-making. This has been due, in part, to cheap data and cheap compute resources, which have fit the natural strengths of deep learning. However, many defining characteristics of human intelligence, which developed under much different pressures, remain out of reach for current approaches. In particular, generalizing beyond one's experiences--a hallmark of human intelligence from infancy--remains a formidable challenge for modern AI. The following is part position paper, part review, and part unification. We argue that combinatorial generalization must be a top priority for AI to achieve human-like abilities, and that structured representations and computations are key to realizing this objective. Just as biology uses nature and nurture cooperatively, we reject the false choice between "hand-engineering" and "end-to-end" learning, and instead advocate for an approach which benefits from their complementary strengths. We explore how using relational inductive biases within deep learning architectures can facilitate learning about entities, relations, and rules for composing them. We present a new building block for the AI toolkit with a strong relational inductive bias--the graph network--which generalizes and extends various approaches for neural networks that operate on graphs, and provides a straightforward interface for manipulating structured knowledge and producing structured behaviors. We discuss how graph networks can support relational reasoning and combinatorial generalization, laying the foundation for more sophisticated, interpretable, and flexible patterns of reasoning. As a companion to this paper, we have released an open-source software library for building graph networks, with demonstrations of how to use them in practice.
  </details>




* [Proto-value Functions: A Laplacian Framework for Learning Representation and Control in Markov Decision Processes](https://jmlr.csail.mit.edu/papers/volume8/mahadevan07a/mahadevan07a.pdf)




* [The why, how, and when of representations for complex systems](https://arxiv.org/abs/2006.02870)
  <details>
    <summary> Details!  </summary>  

    Complex systems thinking is applied to a wide variety of domains, from neuroscience to computer science and economics. The wide variety of implementations has resulted in two key challenges: the progenation of many domain-specific strategies that are seldom revisited or questioned, and the siloing of ideas within a domain due to inconsistency of complex systems language. In this work we offer basic, domain-agnostic language in order to advance towards a more cohesive vocabulary. We use this language to evaluate each step of the complex systems analysis pipeline, beginning with the system and data collected, then moving through different mathematical formalisms for encoding the observed data (i.e. graphs, simplicial complexes, and hypergraphs), and relevant computational methods for each formalism. At each step we consider different types of \emph{dependencies}; these are properties of the system that describe how the existence of one relation among the parts of a system may influence the existence of another relation. We discuss how dependencies may arise and how they may alter interpretation of results or the entirety of the analysis pipeline. We close with two real-world examples using coauthorship data and email communications data that illustrate how the system under study, the dependencies therein, the research question, and choice of mathematical representation influence the results. We hope this work can serve as an opportunity of reflection for experienced complexity scientists, as well as an introductory resource for new researchers.
  </details>





* [Reconciling modern machine-learning practice and the classical bias–variance trade-off](https://www.pnas.org/content/116/32/15849)



* [A deep learning framework for neuroscience](https://www.nature.com/articles/s41593-019-0520-2)
  <details>
    <summary> Details!  </summary>  

    Systems neuroscience seeks explanations for how the brain implements a wide variety of perceptual, cognitive and motor tasks. Conversely, artificial intelligence attempts to design computational systems based on the tasks they will have to solve. In artificial neural networks, the three components specified by design are the objective functions, the learning rules and the architectures. With the growing success of deep learning, which utilizes brain-inspired architectures, these three designed components have increasingly become central to how we model, engineer and optimize complex artificial learning systems. Here we argue that a greater focus on these components would also benefit systems neuroscience. We give examples of how this optimization-based framework can drive theoretical and experimental progress in neuroscience. We contend that this principled perspective on systems neuroscience will help to generate more rapid progress.
  </details>







#### _*Autoencoders*_
* Variational autoencoders used with dimensionality reduction. [VAE-SNE: a deep generative model for simultaneous dimensionality reduction and clustering](https://www.biorxiv.org/content/10.1101/2020.07.17.207993v1)
  <details>
    <summary> Details!  </summary>  

    Description: We introduce a method for both dimension reduction and clustering called VAE-SNE (variational autoencoder stochastic neighbor embedding). Our model combines elements from deep learning, probabilistic inference, and manifold learning to produce interpretable compressed representations while also readily scaling to tens-of-millions of observations. Unlike existing methods, VAE-SNE simultaneously compresses high-dimensional data and automatically learns a distribution of clusters within the data --- without the need to manually select the number of clusters. This naturally creates a multi-scale representation, which makes it straightforward to generate coarse-grained descriptions for large subsets of related observations and select specific regions of interest for further analysis.
  </details>
  
  
* Encoders for timeseries. [Deep reconstruction of strange attractors from time series](https://arxiv.org/pdf/2002.05909.pdf) 
  <details>
    <summary> Details!  </summary>  

    Experimental measurements of physical systems often have a limited number of independent channels, causing essential dynamical variables to remain unobserved. However, many popular methods for unsupervised inference of latent dynamics from experimental data implicitly assume that the measurements have higher intrinsic dimensionality than the underlying system---making coordinate identification a dimensionality reduction problem. Here, we study the opposite limit, in which hidden governing coordinates must be inferred from only a low-dimensional time series of measurements. Inspired by classical techniques for studying the strange attractors of chaotic systems, we introduce a general embedding technique for time series, consisting of an autoencoder trained with a novel latent-space loss function. We show that our technique reconstructs the strange attractors of synthetic and real-world systems better than existing techniques, and that it creates consistent, predictive representations of even stochastic systems. We conclude by using our technique to discover dynamical attractors in diverse systems such as patient electrocardiograms, household electricity usage, and eruptions of the Old Faithful geyser---demonstrating diverse applications of our technique for exploratory data analysis.
  </details>


* [Inferring single-trial neural population dynamics using sequential auto-encoders](https://www.nature.com/articles/s41592-018-0109-9)
  <details>
    <summary> Details!  </summary>  

    Neuroscience is experiencing a revolution in which simultaneous recording of thousands of neurons is revealing population dynamics that are not apparent from single-neuron responses. This structure is typically extracted from data averaged across many trials, but deeper understanding requires studying phenomena detected in single trials, which is challenging due to incomplete sampling of the neural population, trial-to-trial variability, and fluctuations in action potential timing. We introduce latent factor analysis via dynamical systems, a deep learning method to infer latent dynamics from single-trial neural spiking data. When applied to a variety of macaque and human motor cortical datasets, latent factor analysis via dynamical systems accurately predicts observed behavioral variables, extracts precise firing rate estimates of neural dynamics on single trials, infers perturbations to those dynamics that correlate with behavioral choices, and combines data from non-overlapping recording sessions spanning months to improve inference of underlying dynamics.
  </details>



#### _*Reinforcement Learning*_

* [See book by Reinforcement Learning by Sutton and Barto](http://incompleteideas.net/book/RLbook2020.pdf)

* [How teaching AI to be curious helps machines learn for themselves](https://www.theverge.com/2018/11/1/18051196/ai-artificial-intelligence-curiosity-openai-montezumas-revenge-noisy-tv-problem)

 * [Deep active inference as variational policy gradients](https://arxiv.org/pdf/1907.03876.pdf)
    <details>
      <summary> Details!  </summary>  

      Active Inference is a theory arising from theoretical neuroscience which casts action and planning as Bayesian inference problems to be solved by minimizing a single quantity — the variational free energy. The theory promises a unifying account of action and perception coupled with a biologically plausible process theory. However, despite these potential advantages, current implementations of Active Inference can only handle small policy and state–spaces and typically require the environmental dynamics to be known. In this paper we propose a novel deep Active Inference algorithm that approximates key densities using deep neural networks as flexible function approximators, which enables our approach to scale to significantly larger and more complex tasks than any before attempted in the literature. We demonstrate our method on a suite of OpenAIGym benchmark tasks and obtain performance comparable with common reinforcement learning baselines. Moreover, our algorithm evokes similarities with maximum-entropy reinforcement learning and the policy gradients algorithm, which reveals interesting connections between the Active Inference framework and reinforcement learning.
    </details>

* [Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition](https://arxiv.org/pdf/cs/9905014.pdf)


* [Deep Reinforcement Learning and Its Neuroscientific Implications](https://www.sciencedirect.com/science/article/abs/pii/S0896627320304682)
    <details>
      <summary> Details!  </summary>  

      The emergence of powerful artificial intelligence (AI) is defining new research directions in neuroscience. To date, this research has focused largely on deep neural networks trained using supervised learning in tasks such as image classification. However, there is another area of recent AI work that has so far received less attention from neuroscientists but that may have profound neuroscientific implications: deep reinforcement learning (RL). Deep RL offers a comprehensive framework for studying the interplay among learning, representation, and decision making, offering to the brain sciences a new set of research tools and a wide range of novel hypotheses. In the present review, we provide a high-level introduction to deep RL, discuss some of its initial applications to neuroscience, and survey its wider implications for research on brain and behavior, concluding with a list of opportunities for next-stage research.
    </details>
















## General Neuroscience

* [Classic must-read neuroscience papers](https://www.sfn.org/about/history-of-neuroscience/classic-papers#Learning) suggested by SfN (Society for Neuroscience). Broken down by topic. 




## Books 

* [Mathematics for Machine Learning by A. Aldo Faisal, Cheng Soon Ong, and Marc Peter Deisenroth](https://mml-book.github.io/book/mml-book.pdf)

* [Information Theory, Inference, and Learning Algorithms](https://www.inference.org.uk/itprnn/book.pdf). See chapter 28 for Model Comparison and Occam’s Razor for model simplicity. 

* [Pattern Recognition and Machine Learning by Bishop](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf)

* [Theoretical Neuroscience](http://www.gatsby.ucl.ac.uk/~lmate/biblio/dayanabbott.pdf) by Dayan and Abbott.

* [Introduction to Applied Linear Algebra Vectors, Matrices, and Least Squares](http://vmls-book.stanford.edu/vmls.pdf) by Boyd and Vandenberghe.

* [Reinforcement Learning by Sutton and Barto](http://incompleteideas.net/book/RLbook2020.pdf) is the main book on reinforcement learning that takes a deep dive into the topic.




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

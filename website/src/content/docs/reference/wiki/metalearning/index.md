---
title: "Meta Learning"
description: "All 266 public types in the AiDotNet.metalearning namespace, organized by kind."
section: "API Reference"
---

**266** public types in this namespace, organized by kind.

## Models & Types (140)

| Type | Summary |
|:-----|:--------|
| [`ACLAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/aclalgorithm/) | Implementation of ACL: Adaptive Continual Learning with task-specific parameter importance masks. |
| [`ANILAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/anilalgorithm/) | Implementation of Almost No Inner Loop (ANIL) meta-learning algorithm. |
| [`ANILModel<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/anilmodel/) | ANIL model for few-shot classification with head-only adaptation. |
| [`ANPAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/anpalgorithm/) | Implementation of Attentive Neural Process (ANP) (Kim et al., ICLR 2019). |
| [`ATAMLAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/atamlalgorithm/) | Implementation of ATAML: Attention-based Task-Adaptive Meta-Learning. |
| [`ActiveTransFSLAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/activetransfslalgorithm/) | Implementation of ActiveTransFSL: Active Transductive Few-Shot Learning. |
| [`AdaptedMetaModel<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/adaptedmetamodel/) | Generic adapted model wrapper for meta-learning algorithms that use gradient-based inner-loop adaptation. |
| [`AutoLoRAAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/autoloraalgorithm/) | Implementation of AutoLoRA: Automatically Tuning Matrix Ranks in Low-Rank Adaptation Based on Meta Learning (Zhang et al., NAACL 2024). |
| [`BMAMLAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/bmamlalgorithm/) | Implementation of BMAML: Bayesian Model-Agnostic Meta-Learning (Yoon et al., NeurIPS 2018). |
| [`BOILAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/boilalgorithm/) | Implementation of Body Only Inner Loop (BOIL) meta-learning algorithm. |
| [`BOILModel<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/boilmodel/) | BOIL model for few-shot classification with body-only adaptation. |
| [`BalancedTaskSampler<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/balancedtasksampler/) | Samples tasks while ensuring that all classes in the meta-dataset appear equally often across the sampled episodes over time. |
| [`BatchEpisodeSampler<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/batchepisodesampler/) | Efficiently samples batches of episodes with optional prefetching and caching. |
| [`BayProNetAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/baypronetalgorithm/) | Implementation of BayProNet: Bayesian Prototypical Networks with uncertainty-aware prototype distributions for few-shot learning. |
| [`BayTransProtoAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/baytransprotoalgorithm/) | Implementation of BayTransProto: Bayesian Transductive Prototypical Networks. |
| [`CAMLAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/camlalgorithm/) | Implementation of CAML (Context-Aware Meta-Learning) (Fifty et al., NeurIPS 2023). |
| [`CAVIAAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/caviaalgorithm/) | Implementation of CAVIA (Fast Context Adaptation via Meta-Learning) for few-shot learning. |
| [`CAVIAModel<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/caviamodel/) | CAVIA inference model that uses adapted context parameters for predictions. |
| [`CNAPAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/cnapalgorithm/) | Implementation of Conditional Neural Adaptive Processes (CNAP) for meta-learning. |
| [`CNPAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/cnpalgorithm/) | Implementation of Conditional Neural Process (CNP) (Garnelo et al., ICML 2018). |
| [`ConstellationNetAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/constellationnetalgorithm/) | Implementation of ConstellationNet (structured part-based few-shot learning) (Xu et al., ICLR 2021). |
| [`ContextMetaRLAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/contextmetarlalgorithm/) | Implementation of Context Meta-RL: context-conditioned meta-reinforcement learning with multi-head attention-based aggregation. |
| [`ConvCNPAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/convcnpalgorithm/) | Implementation of Convolutional Conditional Neural Process (Gordon et al., ICLR 2020). |
| [`ConvNPAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/convnpalgorithm/) | Implementation of Convolutional Neural Process (Foong et al., 2020). |
| [`CurriculumTaskSampler<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/curriculumtasksampler/) | Samples tasks following a difficulty-based curriculum: starts with easy tasks and gradually increases difficulty as training progresses. |
| [`DKTAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/dktalgorithm/) | Implementation of DKT (Deep Kernel Transfer) (Patacchiola et al., ICLR 2020). |
| [`DPGNAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/dpgnalgorithm/) | Implementation of DPGN (Distribution Propagation Graph Network) (Yang et al., CVPR 2020). |
| [`DREAMAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/dreamalgorithm/) | Implementation of DREAM: Directed REward Augmented Meta-learning. |
| [`DeepEMDAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/deepemdalgorithm/) | Implementation of DeepEMD (Earth Mover's Distance for Few-Shot Learning). |
| [`DiscoRLAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/discorlalgorithm/) | Implementation of DiscoRL: Discovery-based meta-RL with reusable skill discovery. |
| [`DynamicTaskSampler<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/dynamictasksampler/) | Samples tasks with probability proportional to the loss observed on previous evaluations. |
| [`DynamicTaskSamplingAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/dynamictasksamplingalgorithm/) | Implementation of Dynamic Task Sampling: difficulty-aware gradient reweighting for meta-learning. |
| [`EPNetAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/epnetalgorithm/) | Implementation of EPNet (Embedding Propagation Network) (Rodriguez et al., CVPR 2020). |
| [`ETPNAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/etpnalgorithm/) | Implementation of ETPN: Embedding-Transformed Prototypical Networks. |
| [`EpisodeCache<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/episodecache/) | Caches sampled episodes for reuse, reducing the cost of repeated episode generation. |
| [`Episode<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/episode/) | Concrete implementation of `IEpisode` that wraps a meta-learning task with episode-level metadata. |
| [`EpisodicDataLoaderTaskSamplerAdapter<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/episodicdataloadertasksampleradapter/) | Adapter that wraps an existing `IEpisodicDataLoader` to implement the `ITaskSampler` interface. |
| [`EquivCNPAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/equivcnpalgorithm/) | Implementation of Equivariant Conditional Neural Process (Kawano et al., 2021). |
| [`ExternalMemory<T>`](/docs/reference/wiki/metalearning/externalmemory/) | External memory matrix for MANN. |
| [`FEATAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/featalgorithm/) | Implementation of FEAT (Few-shot Embedding Adaptation with Transformer). |
| [`FRNAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/frnalgorithm/) | Implementation of FRN (Few-shot Classification via Feature Map Reconstruction) (Wertheimer et al., CVPR 2021). |
| [`FewTUREAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/fewturealgorithm/) | Implementation of FewTURE (Few-shot Transformer with Uncertainty and Reliable Estimation) (Hiller et al., ECCV 2022). |
| [`FlexPACBayesAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/flexpacbayesalgorithm/) | Implementation of Flex-PAC-Bayes: Flexible PAC-Bayesian Meta-Learning with data-dependent prior construction. |
| [`FreqPriorAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/freqprioralgorithm/) | Implementation of FreqPrior: Frequency-based prior for cross-domain few-shot learning. |
| [`FreqPromptAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/freqpromptalgorithm/) | Implementation of FreqPrompt: Frequency-domain prompt tuning for few-shot learning. |
| [`GCDPLNetAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/gcdplnetalgorithm/) | Implementation of GCDPLNet: Graph-based Cross-Domain Prototype Learning Network. |
| [`GNNMetaAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/gnnmetaalgorithm/) | Implementation of Graph Neural Network-based Meta-learning. |
| [`GaussianClassificationMetaDataset<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/gaussianclassificationmetadataset/) | A synthetic meta-dataset for classification where each class is a Gaussian blob in feature space. |
| [`HyperCLIPAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/hyperclipalgorithm/) | Implementation of HyperCLIP: Contrastive Learning for Hypernetwork-based Meta-Learning. |
| [`HyperMAMLAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/hypermamlalgorithm/) | Implementation of HyperMAML (hypernetwork-based MAML initialization). |
| [`HyperNeRFMetaAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/hypernerfmetaalgorithm/) | Implementation of HyperNeRF Meta: Positional-Encoding-Conditioned Meta-Learning. |
| [`HyperNetMetaRLAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/hypernetmetarlalgorithm/) | Implementation of HyperNet Meta-RL: hypernetwork-based task-specific parameter generation. |
| [`HyperShotAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/hypershotalgorithm/) | Implementation of HyperShot (kernel hypernetwork for few-shot learning). |
| [`ICMFusionAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/icmfusionalgorithm/) | Implementation of ICM-Fusion: In-Context Meta-Optimized LoRA Fusion (2025). |
| [`InContextRLAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/incontextrlalgorithm/) | Implementation of In-Context RL: meta-RL via in-context learning without explicit gradient updates at test time. |
| [`JMPAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/jmpalgorithm/) | Implementation of JMP: Joint Multi-Phase meta-learning. |
| [`LBANPAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/lbanpalgorithm/) | Implementation of Latent Bottleneck Attentive Neural Process (Feng et al., ICML 2023). |
| [`LEOAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/leoalgorithm/) | Implementation of Latent Embedding Optimization (LEO) meta-learning algorithm. |
| [`LEOModel<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/leomodel/) | LEO model for few-shot classification with latent space optimization. |
| [`LSTMNTMController<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/lstmntmcontroller/) | LSTM-based NTM controller implementation with learnable parameters. |
| [`LaplacianShotAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/laplacianshotalgorithm/) | Implementation of LaplacianShot (Laplacian Regularized Few-Shot Learning). |
| [`LinearVectorModel`](/docs/reference/wiki/metalearning/linearvectormodel/) | A simple linear model mapping Matrix input to Vector output, useful for meta-learning examples and testing. |
| [`LoRARecycleAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/lorarecyclealgorithm/) | Implementation of LoRA-Recycle (Hu et al., CVPR 2025). |
| [`MAMLAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/mamlalgorithm/) | Implementation of the MAML (Model-Agnostic Meta-Learning) algorithm. |
| [`MAMLPlusPlusAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/mamlplusplusalgorithm/) | Implementation of MAML++ (How to Train Your MAML) for few-shot learning. |
| [`MANNAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/mannalgorithm/) | Implementation of Memory-Augmented Neural Networks (MANN) for meta-learning. |
| [`MANNMemoryStatistics`](/docs/reference/wiki/metalearning/mannmemorystatistics/) | Memory statistics tracking for MANN. |
| [`MANNModel<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/mannmodel/) | MANN model for inference with external memory. |
| [`MCLAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/mclalgorithm/) | Implementation of MCL (Meta-learning with Contrastive Learning). |
| [`MLPNTMController<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/mlpntmcontroller/) | MLP-based NTM controller implementation with learnable parameters. |
| [`MOCAAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/mocaalgorithm/) | Implementation of MOCA: Meta-learning with Online Complementary Augmentation. |
| [`MPTSAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/mptsalgorithm/) | Implementation of MPTS: Meta-learning with Progressive Task-Specific adaptation. |
| [`MatchingNetworksAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/matchingnetworksalgorithm/) | Implementation of Matching Networks for few-shot learning. |
| [`MatchingNetworksModel<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/matchingnetworksmodel/) | Matching Networks model for inference. |
| [`MePoAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/mepoalgorithm/) | Implementation of MePo: Memory Prototypes for continual few-shot meta-learning. |
| [`MetaBaselineAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/metabaselinealgorithm/) | Implementation of Meta-Baseline (simple pre-train then meta-train with cosine classifier). |
| [`MetaCollaborativeAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/metacollaborativealgorithm/) | Implementation of Meta-Collaborative Learning for cross-domain few-shot learning. |
| [`MetaContinualALAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/metacontinualalalgorithm/) | Implementation of MetaContinualAL: Meta-Continual Active Learning with uncertainty-guided parameter-selective adaptation. |
| [`MetaDDPMAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/metaddpmalgorithm/) | Implementation of Meta-DDPM: Meta-Learning with Denoising Diffusion Probabilistic Models. |
| [`MetaDMAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/metadmalgorithm/) | Implementation of Meta-DM: Applications of Diffusion Models on Few-Shot Learning (Hu et al., ICIP 2024). |
| [`MetaDatasetFormat<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/metadatasetformat/) | Supports the Meta-Dataset benchmark format (Triantafillou et al., 2020): a multi-domain evaluation protocol with variable-way variable-shot task sampling. |
| [`MetaDiffAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/metadiffalgorithm/) | Implementation of MetaDiff: Meta-Learning with Conditional Diffusion for Few-Shot Learning (Zhang et al., AAAI 2024). |
| [`MetaFDMixupAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/metafdmixupalgorithm/) | Implementation of Meta-FDMixup: Feature-Distribution Mixup for cross-domain few-shot learning (Xu et al., CVPR 2021). |
| [`MetaLearnerOptionsBase<T>`](/docs/reference/wiki/metalearning/metalearneroptionsbase/) | Base implementation of IMetaLearnerOptions with industry-standard defaults. |
| [`MetaLoRAAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/metaloraalgorithm/) | Implementation of Meta-LoRA: Low-Rank Adaptation for Meta-Learning (2024). |
| [`MetaLoRABankAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/metalorabankalgorithm/) | Implementation of Meta-LoRA Bank (2024). |
| [`MetaOptNetAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/metaoptnetalgorithm/) | Implementation of Meta-learning with Differentiable Convex Optimization (MetaOptNet) algorithm. |
| [`MetaOptNetModel<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/metaoptnetmodel/) | MetaOptNet model for few-shot classification with convex optimization. |
| [`MetaPACOHAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/metapacohalgorithm/) | Implementation of Meta-PACOH: Hierarchical PAC-Bayesian Meta-Learning with per-group prior variances. |
| [`MetaSGDAdaptedModel<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/metasgdadaptedmodel/) | Wrapper model for Meta-SGD adapted models that includes the per-parameter optimizer. |
| [`MetaSGDAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/metasgdalgorithm/) | Implementation of Meta-SGD (Meta Stochastic Gradient Descent) algorithm. |
| [`MetaTaskAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/metataskalgorithm/) | Implementation of MetaTask: Meta-learned Task Augmentation via gradient interpolation. |
| [`ModelPredictiveTaskSampler<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/modelpredictivetasksampler/) | Model Predictive Task Sampling (MPTS): predicts which tasks will yield the greatest learning signal by maintaining a posterior estimate of per-task adaptation risk, then sampling tasks that balance exploration and exploitation. |
| [`NPAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/npalgorithm/) | Implementation of Neural Process (NP) (Garnelo et al., 2018). |
| [`NPBMLAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/npbmlalgorithm/) | Implementation of NPBML (Neural Process-Based Meta-Learning). |
| [`NTMAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/ntmalgorithm/) | Implementation of Neural Turing Machine (NTM) for meta-learning. |
| [`NTMMemory<T>`](/docs/reference/wiki/metalearning/ntmmemory/) | External memory matrix for Neural Turing Machine. |
| [`NTMModel<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/ntmmodel/) | NTM model for inference with persistent memory. |
| [`NTMReadHead<T>`](/docs/reference/wiki/metalearning/ntmreadhead/) | NTM read head for content-based addressing. |
| [`NTMWriteHead<T>`](/docs/reference/wiki/metalearning/ntmwritehead/) | NTM write head for content-based addressing. |
| [`NeuralProcessModel<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/neuralprocessmodel/) | Adapted model for Neural Process family algorithms. |
| [`OMLAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/omlalgorithm/) | Implementation of OML: Online Meta-Learning (Javed & White, 2019). |
| [`OpenMAMLAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/openmamlalgorithm/) | Implementation of Open-MAML (open-set MAML with out-of-distribution detection). |
| [`OpenMAMLPlusAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/openmamlplusalgorithm/) | Implementation of Open-MAML++: MAML with per-parameter learning rates and open-set novelty detection. |
| [`PACOHAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/pacohalgorithm/) | Implementation of PACOH: PAC-Bayesian Meta-Learning with Optimal Hyperparameters (Rothfuss et al., ICLR 2021). |
| [`PEARLAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/pearlalgorithm/) | Implementation of PEARL: Probabilistic Embeddings for Actor-critic RL (Rakelly et al., ICML 2019). |
| [`PMFAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/pmfalgorithm/) | Implementation of PMF (P>M>F: Pre-training, Meta-training, Fine-tuning). |
| [`PTMAPAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/ptmapalgorithm/) | Implementation of PT+MAP (Power Transform + Maximum A Posteriori) (Hu et al., ICLR 2021). |
| [`PerParameterOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/perparameteroptimizer/) | Per-parameter optimizer for Meta-SGD that learns individual optimization coefficients. |
| [`ProtoNetsAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/protonetsalgorithm/) | Implementation of Prototypical Networks (ProtoNets) algorithm for few-shot learning. |
| [`PrototypicalModel<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/prototypicalmodel/) | Prototypical model for few-shot classification. |
| [`R2D2Algorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/r2d2algorithm/) | Implementation of R2-D2 (Meta-learning with Differentiable Closed-form Solvers) for few-shot learning. |
| [`RCNPAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/rcnpalgorithm/) | Implementation of Recurrent Conditional Neural Process (2024). |
| [`RecurrentHyperNetAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/recurrenthypernetalgorithm/) | Implementation of Recurrent HyperNetwork for meta-learning. |
| [`RelationModule<T>`](/docs/reference/wiki/metalearning/relationmodule/) | Relation module that computes similarity between feature pairs for Relation Networks. |
| [`RelationNetworkAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/relationnetworkalgorithm/) | Implementation of Relation Networks algorithm for few-shot learning. |
| [`RelationNetworkModel<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/relationnetworkmodel/) | Relation Network model for few-shot classification. |
| [`ReptileAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/reptilealgorithm/) | Implementation of the Reptile meta-learning algorithm. |
| [`RotatedDigitsMetaDataset<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/rotateddigitsmetadataset/) | A synthetic meta-dataset for image-like classification where each class is a rotated "digit" pattern (a simple 2D feature vector derived from an angle). |
| [`SDCLAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/sdclalgorithm/) | Implementation of SDCL: Self-Distillation Collaborative Learning for meta-learning. |
| [`SEALAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/sealalgorithm/) | Implementation of the SEAL (Sample-Efficient Adaptive Learning) meta-learning algorithm. |
| [`SIBAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/sibalgorithm/) | Implementation of SIB (Sequential Information Bottleneck) for transductive few-shot learning. |
| [`SNAILAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/snailalgorithm/) | Implementation of SNAIL (Simple Neural Attentive Meta-Learner) for few-shot learning. |
| [`SetFeatAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/setfeatalgorithm/) | Implementation of SetFeat (set-feature based few-shot learning) (Afrasiyabi et al., CVPR 2022). |
| [`SimpleShotAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/simpleshotalgorithm/) | Implementation of SimpleShot for few-shot learning via nearest-centroid classification. |
| [`SineWaveMetaDataset<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/sinewavemetadataset/) | A synthetic meta-dataset for regression where each task is a sinusoidal function with random amplitude and phase. |
| [`SteerCNPAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/steercnpalgorithm/) | Implementation of Steerable Conditional Neural Process (Holderrieth et al., 2021). |
| [`SwinTNPAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/swintnpalgorithm/) | Implementation of Swin Transformer Neural Process (2024). |
| [`TADAMAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/tadamalgorithm/) | Implementation of Task-Dependent Adaptive Metric (TADAM) algorithm for few-shot learning. |
| [`TADAMModel<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/tadammodel/) | TADAM model for few-shot classification with task conditioning and metric scaling. |
| [`TETNPAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/tetnpalgorithm/) | Translation-Equivariant Transformer Neural Process (TE-TNP, 2024). |
| [`TIMAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/timalgorithm/) | Implementation of TIM (Transductive Information Maximization) for few-shot learning. |
| [`TNPAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/tnpalgorithm/) |  |
| [`TaskCondHyperNetAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/taskcondhypernetalgorithm/) | Implementation of Task-Conditioned HyperNetwork for meta-learning. |
| [`UniformTaskSampler<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/uniformtasksampler/) | Samples tasks uniformly at random from a meta-dataset. |
| [`UnsupervisedMetaLearnAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/unsupervisedmetalearnalgorithm/) | Implementation of Unsupervised Meta-Learning (Hsu et al., 2019). |
| [`VERSAAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/versaalgorithm/) | Implementation of VERSA (Versatile and Efficient Few-shot Learning) for few-shot learning. |
| [`WarpGradAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/warpgradalgorithm/) | Implementation of WarpGrad (Meta-Learning with Warped Gradient Descent) for few-shot learning. |
| [`iMAMLAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/imamlalgorithm/) | Implementation of the iMAML (Implicit Model-Agnostic Meta-Learning) algorithm. |
| [`iTAMLAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/itamlalgorithm/) | Implementation of iTAML: incremental Task-Agnostic Meta-Learning (Rajasegaran et al., 2020). |

## Base Classes (4)

| Type | Summary |
|:-----|:--------|
| [`MetaDatasetBase<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/metadatasetbase/) | Abstract base class for meta-datasets that generate episodes on-the-fly. |
| [`MetaLearnerBase<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/metalearnerbase/) | Unified base class for all meta-learning algorithms, providing both training infrastructure and shared algorithm utilities. |
| [`MetaLearningModelBase<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/metalearningmodelbase/) | Abstract base class for meta-learning adapted models that wrap a base model with task-specific parameters. |
| [`NeuralProcessBase<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/neuralprocessbase/) | Base class for the Neural Process family of meta-learning algorithms. |

## Interfaces (3)

| Type | Summary |
|:-----|:--------|
| [`IEpisodicDataset<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/iepisodicdataset/) |  |
| [`IMetaLearningAlgorithm<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/imetalearningalgorithm/) |  |
| [`INTMController<T>`](/docs/reference/wiki/metalearning/intmcontroller/) | Interface for NTM controller. |

## Enums (15)

| Type | Summary |
|:-----|:--------|
| [`CAVIAContextInjectionMode`](/docs/reference/wiki/metalearning/caviacontextinjectionmode/) | Specifies how context parameters are injected into the model's computation. |
| [`DatasetSplit`](/docs/reference/wiki/metalearning/datasetsplit/) | Represents the type of dataset split. |
| [`FastWeightApplicationMode`](/docs/reference/wiki/metalearning/fastweightapplicationmode/) | Specifies how fast weights are applied to modify the base model. |
| [`FewTUREUncertaintyMethod`](/docs/reference/wiki/metalearning/fewtureuncertaintymethod/) | Uncertainty estimation method for FewTURE. |
| [`GNNAggregationType`](/docs/reference/wiki/metalearning/gnnaggregationtype/) | Specifies how nodes are aggregated to form graph-level representations. |
| [`MatchingNetworksAttentionFunction`](/docs/reference/wiki/metalearning/matchingnetworksattentionfunction/) | Attention function types for Matching Networks. |
| [`MetaLearningAlgorithmType`](/docs/reference/wiki/metalearning/metalearningalgorithmtype/) | Specifies the type of meta-learning algorithm used for few-shot learning and quick adaptation. |
| [`MetaSGDLearningRateInitialization`](/docs/reference/wiki/metalearning/metasgdlearningrateinitialization/) | Learning rate initialization strategies for Meta-SGD. |
| [`MetaSGDLearningRateScheduleType`](/docs/reference/wiki/metalearning/metasgdlearningratescheduletype/) | Learning rate schedule types for Meta-SGD meta-training. |
| [`MetaSGDUpdateRuleType`](/docs/reference/wiki/metalearning/metasgdupdateruletype/) | Update rule types for Meta-SGD per-parameter optimization. |
| [`NTMControllerType`](/docs/reference/wiki/metalearning/ntmcontrollertype/) | Controller type for Neural Turing Machine. |
| [`NTMMemoryInitialization`](/docs/reference/wiki/metalearning/ntmmemoryinitialization/) | Memory initialization strategies for NTM. |
| [`ProtoNetsDistanceFunction`](/docs/reference/wiki/metalearning/protonetsdistancefunction/) | Distance functions supported by Prototypical Networks for measuring similarity between embeddings. |
| [`SEALAdaptiveLearningRateMode`](/docs/reference/wiki/metalearning/sealadaptivelearningratemode/) | Specifies the mode for computing adaptive learning rates in SEAL. |
| [`TaskSimilarityMetric`](/docs/reference/wiki/metalearning/tasksimilaritymetric/) | Specifies how task similarity is computed for building the task graph. |

## Options & Configuration (102)

| Type | Summary |
|:-----|:--------|
| [`ACLOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/acloptions/) | Configuration options for the ACL (Adaptive Continual Learning) algorithm. |
| [`ANILOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/aniloptions/) | Configuration options for Almost No Inner Loop (ANIL) algorithm. |
| [`ANPOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/anpoptions/) | Configuration options for Attentive Neural Process (ANP) (Kim et al., ICLR 2019). |
| [`ATAMLOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/atamloptions/) | Configuration options for the ATAML (Attention-based Task-Adaptive Meta-Learning) algorithm. |
| [`ActiveTransFSLOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/activetransfsloptions/) | Configuration options for ActiveTransFSL (Active Transductive Few-Shot Learning). |
| [`AutoLoRAOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/autoloraoptions/) | Configuration options for AutoLoRA (Zhang et al., NAACL 2024). |
| [`BMAMLOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/bmamloptions/) | Configuration options for BMAML: Bayesian Model-Agnostic Meta-Learning (Yoon et al., NeurIPS 2018). |
| [`BOILOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/boiloptions/) | Configuration options for Body Only Inner Loop (BOIL) algorithm. |
| [`BayProNetOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/baypronetoptions/) | Configuration options for BayProNet: Bayesian Prototypical Networks for few-shot learning with uncertainty estimation. |
| [`BayTransProtoOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/baytransprotooptions/) | Configuration options for BayTransProto (Bayesian Transductive Prototypical Networks). |
| [`CAMLOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/camloptions/) | Configuration options for CAML (Context-Aware Meta-Learning) (Fifty et al., NeurIPS 2023). |
| [`CAVIAOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/caviaoptions/) | Configuration options for the CAVIA (Fast Context Adaptation via Meta-Learning) algorithm. |
| [`CNAPOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/cnapoptions/) | Configuration options for the Conditional Neural Adaptive Processes (CNAP) algorithm. |
| [`CNPOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/cnpoptions/) | Configuration options for Conditional Neural Process (CNP) (Garnelo et al., ICML 2018). |
| [`ConstellationNetOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/constellationnetoptions/) | Configuration options for ConstellationNet (Xu et al., ICLR 2021). |
| [`ContextMetaRLOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/contextmetarloptions/) | Configuration options for Context Meta-RL: context-conditioned meta-reinforcement learning with attention-based aggregation. |
| [`ConvCNPOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/convcnpoptions/) | Configuration options for Convolutional Conditional Neural Process (ConvCNP) (Gordon et al., ICLR 2020). |
| [`ConvNPOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/convnpoptions/) | Configuration options for ConvNP. |
| [`DKTOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/dktoptions/) | Configuration options for DKT (Deep Kernel Transfer) (Patacchiola et al., ICLR 2020). |
| [`DPGNOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/dpgnoptions/) | Configuration options for DPGN (Distribution Propagation Graph Network) (Yang et al., CVPR 2020). |
| [`DREAMOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/dreamoptions/) | Configuration options for DREAM: Directed REward Augmented Meta-learning. |
| [`DeepEMDOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/deepemdoptions/) | Configuration options for DeepEMD (Zhang et al., CVPR 2020) few-shot learning. |
| [`DiscoRLOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/discorloptions/) | Configuration options for DiscoRL: Discovery-based meta-RL with skill discovery. |
| [`DynamicTaskSamplingOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/dynamictasksamplingoptions/) | Configuration options for the DynamicTaskSampling algorithm. |
| [`EPNetOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/epnetoptions/) | Configuration options for EPNet (Embedding Propagation Network) (Rodriguez et al., CVPR 2020). |
| [`ETPNOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/etpnoptions/) | Configuration options for ETPN (Embedding-Transformed Prototypical Networks). |
| [`EquivCNPOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/equivcnpoptions/) | Configuration options for EquivCNP. |
| [`FEATOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/featoptions/) | Configuration options for FEAT (Few-shot Embedding Adaptation with Transformer) (Ye et al., CVPR 2020). |
| [`FRNOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/frnoptions/) | Configuration options for FRN (Few-shot Classification via Feature Map Reconstruction) (Wertheimer et al., CVPR 2021). |
| [`FewTUREOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/fewtureoptions/) | Configuration options for FewTURE (Few-shot Transformer with Uncertainty and Reliable Estimation) (Hiller et al., ECCV 2022). |
| [`FlexPACBayesOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/flexpacbayesoptions/) | Configuration options for Flex-PAC-Bayes: Flexible PAC-Bayesian Meta-Learning with data-dependent prior construction. |
| [`FreqPriorOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/freqprioroptions/) | Configuration options for FreqPrior: Frequency-based prior for cross-domain few-shot learning. |
| [`FreqPromptOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/freqpromptoptions/) | Configuration options for FreqPrompt: Frequency-domain prompt tuning for few-shot learning. |
| [`GCDPLNetOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/gcdplnetoptions/) | Configuration options for GCDPLNet (Graph-based Cross-Domain Prototype Learning Network). |
| [`GNNMetaOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/gnnmetaoptions/) | Configuration options for the Graph Neural Network Meta-learning algorithm. |
| [`HyperCLIPOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/hyperclipoptions/) | Configuration options for HyperCLIP meta-learning. |
| [`HyperMAMLOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/hypermamloptions/) | Configuration options for HyperMAML (hypernetwork-based MAML initialization). |
| [`HyperNeRFMetaOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/hypernerfmetaoptions/) | Configuration options for HyperNeRF Meta-learning. |
| [`HyperNetMetaRLOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/hypernetmetarloptions/) | Configuration options for HyperNet Meta-RL: hypernetwork-based policy generation for meta-reinforcement learning. |
| [`HyperShotOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/hypershotoptions/) | Configuration options for HyperShot (Sendera et al., NeurIPS 2023). |
| [`ICMFusionOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/icmfusionoptions/) | Configuration options for ICM-Fusion (In-Context Meta-Optimized LoRA Fusion, 2025). |
| [`InContextRLOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/incontextrloptions/) | Configuration options for In-Context RL: meta-RL via in-context adaptation without explicit gradient updates at test time. |
| [`JMPOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/jmpoptions/) | Configuration options for JMP (Joint Multi-Phase meta-learning). |
| [`LBANPOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/lbanpoptions/) | Configuration options for LBANP. |
| [`LEOOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/leooptions/) | Configuration options for Latent Embedding Optimization (LEO) algorithm. |
| [`LaplacianShotOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/laplacianshotoptions/) | Configuration options for LaplacianShot (Ziko et al., ICML 2020) few-shot learning. |
| [`LoRARecycleOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/lorarecycleoptions/) | Configuration options for LoRA-Recycle (Hu et al., CVPR 2025). |
| [`MAMLOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/mamloptions/) | Configuration options for MAML (Model-Agnostic Meta-Learning) algorithm. |
| [`MAMLPlusPlusOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/mamlplusplusoptions/) | Configuration options for MAML++ (How to Train Your MAML) meta-learning algorithm. |
| [`MANNOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/mannoptions/) | Configuration options for Memory-Augmented Neural Networks (MANN) algorithm. |
| [`MCLOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/mcloptions/) | Configuration options for MCL (Meta-learning with Contrastive Learning) few-shot method. |
| [`MOCAOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/mocaoptions/) | Configuration options for the MOCA (Meta-learning with Online Complementary Augmentation) algorithm. |
| [`MPTSOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/mptsoptions/) | Configuration options for the MPTS (Meta-learning with Progressive Task-Specific adaptation) algorithm. |
| [`MatchingNetworksOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/matchingnetworksoptions/) | Configuration options for Matching Networks algorithm. |
| [`MePoOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/mepooptions/) | Configuration options for the MePo (Memory Prototypes) meta-learning algorithm. |
| [`MetaBaselineOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/metabaselineoptions/) | Configuration options for Meta-Baseline (Chen et al., ICLR 2021). |
| [`MetaCollaborativeOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/metacollaborativeoptions/) | Configuration options for Meta-Collaborative Learning across multiple task domains. |
| [`MetaContinualALOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/metacontinualaloptions/) | Configuration options for the MetaContinualAL (Meta-Continual Active Learning) algorithm. |
| [`MetaDDPMOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/metaddpmoptions/) | Configuration options for Meta-DDPM: Meta-Learning with Denoising Diffusion Probabilistic Models. |
| [`MetaDMOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/metadmoptions/) | Configuration options for Meta-DM: Applications of Diffusion Models on Few-Shot Learning (Hu et al., ICIP 2024). |
| [`MetaDiffOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/metadiffoptions/) | Configuration options for MetaDiff: Meta-Learning with Conditional Diffusion for Few-Shot Learning (Zhang et al., AAAI 2024). |
| [`MetaFDMixupOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/metafdmixupoptions/) | Configuration options for Meta-FDMixup: Feature-Distribution Mixup for cross-domain few-shot learning (Xu et al., CVPR 2021). |
| [`MetaLoRABankOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/metalorabankoptions/) | Configuration options for Meta-LoRA Bank (2024). |
| [`MetaLoRAOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/metaloraoptions/) | Configuration options for Meta-LoRA (Low-Rank Adaptation for Meta-Learning). |
| [`MetaOptNetOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/metaoptnetoptions/) | Configuration options for Meta-learning with Differentiable Convex Optimization (MetaOptNet) algorithm. |
| [`MetaPACOHOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/metapacohoptions/) | Configuration options for Meta-PACOH: Hierarchical PAC-Bayesian Meta-Learning with per-group prior variances. |
| [`MetaSGDOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/metasgdoptions/) | Configuration options for the Meta-SGD (Meta Stochastic Gradient Descent) algorithm. |
| [`MetaTaskOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/metataskoptions/) | Configuration options for the MetaTask (Meta-learned Task Augmentation) algorithm. |
| [`NPBMLOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/npbmloptions/) | Configuration options for NPBML (Neural Process-Based Meta-Learning). |
| [`NPOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/npoptions/) | Configuration options for Neural Process (NP) (Garnelo et al., 2018). |
| [`NTMOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/ntmoptions/) | Configuration options for Neural Turing Machine (NTM) algorithm. |
| [`OMLOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/omloptions/) | Configuration options for the OML (Online Meta-Learning) algorithm. |
| [`OpenMAMLOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/openmamloptions/) | Configuration options for Open-MAML (open-set MAML for open-world few-shot learning). |
| [`OpenMAMLPlusOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/openmamlplusoptions/) | Configuration options for Open-MAML++: MAML extended for open-set recognition with novelty detection. |
| [`PACOHOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/pacohoptions/) | Configuration options for PACOH: PAC-Bayesian Meta-Learning with Optimal Hyperparameters (Rothfuss et al., ICLR 2021). |
| [`PEARLOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/pearloptions/) | Configuration options for PEARL: Probabilistic Embeddings for Actor-critic RL (Rakelly et al., ICML 2019). |
| [`PMFOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/pmfoptions/) | Configuration options for PMF (P>M>F: Pre-training, Meta-training, Fine-tuning) (Hu et al., ICLR 2022). |
| [`PTMAPOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/ptmapoptions/) | Configuration options for PT+MAP (Power Transform + Maximum A Posteriori) (Hu et al., ICLR 2021). |
| [`ProtoNetsOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/protonetsoptions/) | Configuration options for Prototypical Networks (ProtoNets) algorithm. |
| [`R2D2Options<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/r2d2options/) | Configuration options for R2-D2 (Meta-learning with Differentiable Closed-form Solvers) algorithm. |
| [`RCNPOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/rcnpoptions/) | Configuration options for RCNP. |
| [`RecurrentHyperNetOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/recurrenthypernetoptions/) | Configuration options for Recurrent HyperNetwork meta-learning. |
| [`RelationNetworkOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/relationnetworkoptions/) | Configuration options for Relation Networks algorithm. |
| [`ReptileOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/reptileoptions/) | Configuration options for the Reptile meta-learning algorithm. |
| [`SDCLOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/sdcloptions/) | Configuration options for SDCL: Self-Distillation Collaborative Learning for meta-learning. |
| [`SEALOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/sealoptions/) | Configuration options for the SEAL (Sample-Efficient Adaptive Learning) meta-learning algorithm. |
| [`SIBOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/siboptions/) | Configuration options for SIB (Sequential Information Bottleneck) (Hu et al., 2020) few-shot learning. |
| [`SNAILOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/snailoptions/) | Configuration options for SNAIL (Simple Neural Attentive Meta-Learner) algorithm. |
| [`SetFeatOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/setfeatoptions/) | Configuration options for SetFeat (set-feature based few-shot learning) (Afrasiyabi et al., CVPR 2022). |
| [`SimpleShotOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/simpleshotoptions/) | Configuration options for SimpleShot (Wang et al., 2019) few-shot classification. |
| [`SteerCNPOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/steercnpoptions/) | Configuration options for SteerCNP. |
| [`SwinTNPOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/swintnpoptions/) | Configuration options for SwinTNP. |
| [`TADAMOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/tadamoptions/) | Configuration options for Task-Dependent Adaptive Metric (TADAM) algorithm. |
| [`TETNPOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/tetnpoptions/) | Configuration options for Translation-Equivariant Transformer Neural Process (TE-TNP). |
| [`TIMOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/timoptions/) | Configuration options for TIM (Transductive Information Maximization) (Boudiaf et al., NeurIPS 2020). |
| [`TNPOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/tnpoptions/) | Configuration options for TNP. |
| [`TaskCondHyperNetOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/taskcondhypernetoptions/) | Configuration options for Task-Conditioned HyperNetwork. |
| [`UnsupervisedMetaLearnOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/unsupervisedmetalearnoptions/) | Configuration options for Unsupervised Meta-Learning (Hsu et al., 2019). |
| [`VERSAOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/versaoptions/) | Configuration options for VERSA (Versatile and Efficient Few-shot Learning) algorithm. |
| [`WarpGradOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/warpgradoptions/) | Configuration options for the WarpGrad (Warped Gradient Descent) meta-learning algorithm. |
| [`iMAMLOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/imamloptions/) | Configuration options for iMAML (Implicit Model-Agnostic Meta-Learning) algorithm. |
| [`iTAMLOptions<T, TInput, TOutput>`](/docs/reference/wiki/metalearning/itamloptions/) | Configuration options for the iTAML (incremental Task-Agnostic Meta-Learning) algorithm. |

## Helpers & Utilities (2)

| Type | Summary |
|:-----|:--------|
| [`MetaLearnerOptionsBuilder<T>`](/docs/reference/wiki/metalearning/metalearneroptionsbuilder/) | Fluent builder for MetaLearnerOptionsBase. |
| [`TaskDifficultyEstimator<T>`](/docs/reference/wiki/metalearning/taskdifficultyestimator/) | Estimates the difficulty of a meta-learning task based on geometric properties of the data: inter-class separation, intra-class variance, and support/query alignment. |


---
title: "Federated Learning"
description: "All 266 public types in the AiDotNet.federatedlearning namespace, organized by kind."
section: "API Reference"
---

**266** public types in this namespace, organized by kind.

## Models & Types (207)

| Type | Summary |
|:-----|:--------|
| [`AdaptiveCompressor<T>`](/docs/reference/wiki/federatedlearning/adaptivecompressor/) | Adaptive compressor: dynamically adjusts compression ratio per client based on bandwidth, gradient importance, and staleness. |
| [`AgnosticFairnessObjective<T>`](/docs/reference/wiki/federatedlearning/agnosticfairnessobjective/) | Implements AFL (Agnostic Federated Learning) — minimax fairness optimization. |
| [`AmdSevSnpTeeProvider<T>`](/docs/reference/wiki/federatedlearning/amdsevsnpteeprovider/) | TEE provider for AMD SEV-SNP (Secure Encrypted Virtualization - Secure Nested Paging). |
| [`ArithmeticSecretSharing<T>`](/docs/reference/wiki/federatedlearning/arithmeticsecretsharing/) | Implements additive secret sharing over an arithmetic field for efficient linear operations. |
| [`ArmCcaTeeProvider<T>`](/docs/reference/wiki/federatedlearning/armccateeprovider/) | TEE provider for ARM CCA (Confidential Compute Architecture) / Realm Management Extension. |
| [`AsyncFedEDTrainer<T>`](/docs/reference/wiki/federatedlearning/asyncfededtrainer/) | Implements AsyncFedED — Asynchronous FL with Entropy-Driven client scheduling. |
| [`AvailabilityAwareClientSelectionStrategy`](/docs/reference/wiki/federatedlearning/availabilityawareclientselectionstrategy/) | Availability-aware client selection using per-client online probabilities. |
| [`BaseObliviousTransfer`](/docs/reference/wiki/federatedlearning/baseoblivioustransfer/) | Implements base 1-out-of-2 oblivious transfer using symmetric cryptography. |
| [`BasicCompositionPrivacyAccountant`](/docs/reference/wiki/federatedlearning/basiccompositionprivacyaccountant/) | Privacy accountant using basic (naive) composition. |
| [`BloomFilterPsi`](/docs/reference/wiki/federatedlearning/bloomfilterpsi/) | Implements Bloom filter based probabilistic Private Set Intersection. |
| [`BobaAggregationStrategy<T>`](/docs/reference/wiki/federatedlearning/bobaaggregationstrategy/) | Implements BOBA (Bayesian Optimal Byzantine-robust Aggregation) strategy. |
| [`BooleanSecretSharing`](/docs/reference/wiki/federatedlearning/booleansecretsharing/) | Implements XOR-based boolean secret sharing for bitwise operations. |
| [`BooleanTriple`](/docs/reference/wiki/federatedlearning/booleantriple/) | Represents a pre-shared AND triple for boolean secret sharing. |
| [`BucketingAggregationStrategy<T>`](/docs/reference/wiki/federatedlearning/bucketingaggregationstrategy/) | Implements the Bucketing meta-strategy for Byzantine-robust federated learning. |
| [`BufferedAsyncFederatedTrainer<T>`](/docs/reference/wiki/federatedlearning/bufferedasyncfederatedtrainer/) | Implements FedBuff — Buffered asynchronous federated aggregation. |
| [`BulyanFullModelAggregationStrategy<T, TInput, TOutput>`](/docs/reference/wiki/federatedlearning/bulyanfullmodelaggregationstrategy/) | Bulyan aggregation for `IFullModel` (Multi-Krum selection + trimmed aggregation). |
| [`CircuitBasedPsi`](/docs/reference/wiki/federatedlearning/circuitbasedpsi/) | Implements circuit-based Private Set Intersection using garbled circuit evaluation. |
| [`CircuitGate`](/docs/reference/wiki/federatedlearning/circuitgate/) | Represents a single gate in a boolean circuit. |
| [`ClientDriftResult`](/docs/reference/wiki/federatedlearning/clientdriftresult/) | Per-client drift analysis result. |
| [`ClientGraphStats`](/docs/reference/wiki/federatedlearning/clientgraphstats/) | Statistics about a client's local subgraph, used for graph-aware aggregation weighting. |
| [`ClientProofBundle`](/docs/reference/wiki/federatedlearning/clientproofbundle/) | Contains all proofs that a client must provide for a given verification level. |
| [`ClientVerificationResult`](/docs/reference/wiki/federatedlearning/clientverificationresult/) | Contains the result of verifying a single client's update. |
| [`ClusteredClientSelectionStrategy`](/docs/reference/wiki/federatedlearning/clusteredclientselectionstrategy/) | Cluster-based client selection using simple k-means over per-client embeddings. |
| [`CompressedTree<T>`](/docs/reference/wiki/federatedlearning/compressedtree/) | Represents a compressed decision-tree encoding of a parameter update. |
| [`ComputationIntegrityProof<T>`](/docs/reference/wiki/federatedlearning/computationintegrityproof/) | Generates proofs of local training computation integrity (research-stage). |
| [`ContributionBasedIncentive<T>`](/docs/reference/wiki/federatedlearning/contributionbasedincentive/) | Incentive mechanism that rewards clients proportional to their evaluated contribution. |
| [`DFedAvgMProtocol<T>`](/docs/reference/wiki/federatedlearning/dfedavgmprotocol/) | Implements DFedAvgM — Decentralized FedAvg with Momentum for peer-to-peer FL. |
| [`DPFedLoRA<T>`](/docs/reference/wiki/federatedlearning/dpfedlora/) | Implements DP-FedLoRA — Differentially Private Federated LoRA with per-layer noise calibration. |
| [`DataFreeFCL<T>`](/docs/reference/wiki/federatedlearning/datafreefcl/) | Implements Data-Free Federated Continual Learning — prevents forgetting without storing real data. |
| [`DataShapleyEvaluator<T>`](/docs/reference/wiki/federatedlearning/datashapleyevaluator/) | Data Shapley evaluator: efficient Monte Carlo approximation of Shapley values. |
| [`DeTAGProtocol<T>`](/docs/reference/wiki/federatedlearning/detagprotocol/) | Implements DeTAG — Decentralized gradient Tracking for exact convergence. |
| [`DecentralizedAggregator<T>`](/docs/reference/wiki/federatedlearning/decentralizedaggregator/) | Decentralized aggregator — performs local mixing of model parameters based on topology. |
| [`DiffieHellmanPsi`](/docs/reference/wiki/federatedlearning/diffiehellmanpsi/) | Implements Diffie-Hellman based Private Set Intersection using commutative encryption. |
| [`DiffusiveNoiseUnlearner<T>`](/docs/reference/wiki/federatedlearning/diffusivenoiseunlearner/) | Unlearning via structured diffusive noise injection targeting memorized samples. |
| [`DirectionAlignmentInspector<T>`](/docs/reference/wiki/federatedlearning/directionalignmentinspector/) | Direction Alignment Inspector — detects backdoor attacks via gradient direction analysis. |
| [`DivideAndConquerAggregationStrategy<T>`](/docs/reference/wiki/federatedlearning/divideandconqueraggregationstrategy/) | Implements DnC (Divide and Conquer) aggregation strategy for Byzantine-robust FL. |
| [`DriftAdaptiveAggregator<T>`](/docs/reference/wiki/federatedlearning/driftadaptiveaggregator/) | Drift-adaptive aggregator: wraps any aggregation strategy and adjusts weights based on drift. |
| [`DriftReport`](/docs/reference/wiki/federatedlearning/driftreport/) | Comprehensive report of drift analysis across all federated clients. |
| [`EntityAligner`](/docs/reference/wiki/federatedlearning/entityaligner/) | High-level orchestrator for entity alignment in vertical federated learning. |
| [`EntityAlignmentResult`](/docs/reference/wiki/federatedlearning/entityalignmentresult/) | Contains the results of an entity alignment operation, including the PSI result, party sizes, and diagnostic information. |
| [`ErrorFeedbackCompressor<T>`](/docs/reference/wiki/federatedlearning/errorfeedbackcompressor/) | Error feedback compressor: wraps any compression method with residual accumulation. |
| [`ExactRetrainingUnlearner<T>`](/docs/reference/wiki/federatedlearning/exactretrainingunlearner/) | Gold-standard unlearning: retrains the model from scratch excluding the target client. |
| [`ExpandedSubgraph<T>`](/docs/reference/wiki/federatedlearning/expandedsubgraph/) | Represents a subgraph expanded with pseudo-nodes. |
| [`ExtendedObliviousTransfer`](/docs/reference/wiki/federatedlearning/extendedoblivioustransfer/) | Implements OT extension — amortizes a small number of base OTs into many cheap OTs. |
| [`FLTrustAggregationStrategy<T>`](/docs/reference/wiki/federatedlearning/fltrustaggregationstrategy/) | Implements the FLTrust aggregation strategy for Byzantine-robust federated learning. |
| [`FLoRA<T>`](/docs/reference/wiki/federatedlearning/flora/) | Implements FLoRA — Federated Low-Rank Adaptation with stacked lossless aggregation. |
| [`FedAGCContinualLearning<T>`](/docs/reference/wiki/federatedlearning/fedagccontinuallearning/) | Implements FedAGC — Adaptive Gradient Correction for federated continual learning. |
| [`FedAGHNPersonalization<T>`](/docs/reference/wiki/federatedlearning/fedaghnpersonalization/) | Implements FedAGHN (Adaptive Gradient-based Heterogeneous Networks) personalization. |
| [`FedAaAggregationStrategy<T>`](/docs/reference/wiki/federatedlearning/fedaaaggregationstrategy/) | Implements FedAA (Federated Adaptive Aggregation) strategy. |
| [`FedAdagradServerOptimizer<T>`](/docs/reference/wiki/federatedlearning/fedadagradserveroptimizer/) | FedAdagrad server optimizer — adaptive learning rates using accumulated squared gradients. |
| [`FedAdamServerOptimizer<T>`](/docs/reference/wiki/federatedlearning/fedadamserveroptimizer/) | FedAdam server optimizer — adaptive learning rates with momentum and second-moment estimation. |
| [`FedAlignAggregationStrategy<T>`](/docs/reference/wiki/federatedlearning/fedalignaggregationstrategy/) | Implements FedAlign (Feature Alignment) aggregation strategy. |
| [`FedAvgAggregationStrategy<T>`](/docs/reference/wiki/federatedlearning/fedavgaggregationstrategy/) | Implements the Federated Averaging (FedAvg) aggregation strategy. |
| [`FedAvgFullModelAggregationStrategy<T, TInput, TOutput>`](/docs/reference/wiki/federatedlearning/fedavgfullmodelaggregationstrategy/) | FedAvg aggregation for `IFullModel` using vector-based parameters. |
| [`FedAvgMServerOptimizer<T>`](/docs/reference/wiki/federatedlearning/fedavgmserveroptimizer/) | FedAvgM server optimizer — server-side momentum for stabilized federated averaging. |
| [`FedBABUPersonalization<T>`](/docs/reference/wiki/federatedlearning/fedbabupersonalization/) | Implements FedBABU (Body And Bottom Update) personalization strategy. |
| [`FedBNAggregationStrategy<T>`](/docs/reference/wiki/federatedlearning/fedbnaggregationstrategy/) | Implements the Federated Batch Normalization (FedBN) aggregation strategy. |
| [`FedBNFullModelAggregationStrategy<T, TInput, TOutput>`](/docs/reference/wiki/federatedlearning/fedbnfullmodelaggregationstrategy/) | FedBN aggregation for `IFullModel` when the model is a `NeuralNetworkBase`. |
| [`FedCILContinualLearning<T>`](/docs/reference/wiki/federatedlearning/fedcilcontinuallearning/) | Implements FedCIL — Federated Class-Incremental Learning with prototype consolidation. |
| [`FedCPPersonalization<T>`](/docs/reference/wiki/federatedlearning/fedcppersonalization/) | Implements FedCP (Conditional Policy) personalization with input-dependent routing. |
| [`FedDFDistillation<T>`](/docs/reference/wiki/federatedlearning/feddfdistillation/) | FedDF — Federated ensemble distillation using model averaging on unlabeled public data. |
| [`FedDTCompressor<T>`](/docs/reference/wiki/federatedlearning/feddtcompressor/) | Implements FedDT — Decision-tree-based compression for heterogeneous federated architectures. |
| [`FedDynHeterogeneityCorrection<T>`](/docs/reference/wiki/federatedlearning/feddynheterogeneitycorrection/) | FedDyn-style dynamic regularization using a per-client drift accumulator. |
| [`FedFairOptimizer<T>`](/docs/reference/wiki/federatedlearning/fedfairoptimizer/) | Implements FedFair — multi-objective optimization balancing accuracy, fairness, and efficiency. |
| [`FedGENDistillation<T>`](/docs/reference/wiki/federatedlearning/fedgendistillation/) | FedGEN — Data-free federated distillation using a lightweight generator on the server. |
| [`FedGnnAggregationStrategy<T>`](/docs/reference/wiki/federatedlearning/fedgnnaggregationstrategy/) | GNN-aware federated aggregation that weights contributions by subgraph topology characteristics. |
| [`FedKDCompressor<T>`](/docs/reference/wiki/federatedlearning/fedkdcompressor/) | Implements FedKD — Knowledge Distillation-based communication for federated learning. |
| [`FedLcAggregationStrategy<T>`](/docs/reference/wiki/federatedlearning/fedlcaggregationstrategy/) | Implements FedLC (Logit Calibration) aggregation strategy. |
| [`FedMDDistillation<T>`](/docs/reference/wiki/federatedlearning/fedmddistillation/) | FedMD — Model-agnostic federated learning via mutual distillation on a public dataset. |
| [`FedMaAggregationStrategy<T>`](/docs/reference/wiki/federatedlearning/fedmaaggregationstrategy/) | Implements FedMA (Federated Matched Averaging) aggregation strategy. |
| [`FedMeZO<T>`](/docs/reference/wiki/federatedlearning/fedmezo/) | Implements FedMeZO — Memory-efficient Zeroth-Order optimization for federated LLM fine-tuning. |
| [`FedNovaHeterogeneityCorrection<T>`](/docs/reference/wiki/federatedlearning/fednovaheterogeneitycorrection/) | FedNova-style normalization of client updates by local steps. |
| [`FedPACPersonalization<T>`](/docs/reference/wiki/federatedlearning/fedpacpersonalization/) | Implements FedPAC (Personalization via Aggregation and Calibration) with prototype alignment. |
| [`FedPETuning<T>`](/docs/reference/wiki/federatedlearning/fedpetuning/) | Implements FedPETuning — a unified framework for parameter-efficient fine-tuning (PEFT) in federated learning that supports multiple PEFT methods under one API. |
| [`FedProxAggregationStrategy<T>`](/docs/reference/wiki/federatedlearning/fedproxaggregationstrategy/) | Implements the Federated Proximal (FedProx) aggregation strategy. |
| [`FedProxFullModelAggregationStrategy<T, TInput, TOutput>`](/docs/reference/wiki/federatedlearning/fedproxfullmodelaggregationstrategy/) | FedProx aggregation for `IFullModel`. |
| [`FedRoDPersonalization<T>`](/docs/reference/wiki/federatedlearning/fedrodpersonalization/) | Implements FedRoD (Representation on Demand) personalization with dual classifiers. |
| [`FedSamAggregationStrategy<T>`](/docs/reference/wiki/federatedlearning/fedsamaggregationstrategy/) | Implements FedSAM (Sharpness-Aware Minimization for Federated Learning) aggregation strategy. |
| [`FedSelectPersonalization<T>`](/docs/reference/wiki/federatedlearning/fedselectpersonalization/) | Implements FedSelect — learned sparse binary masks for personalization. |
| [`FedYogiServerOptimizer<T>`](/docs/reference/wiki/federatedlearning/fedyogiserveroptimizer/) | FedYogi server optimizer — adaptive learning rates with controlled second-moment growth. |
| [`FederatedAdapterTuning<T>`](/docs/reference/wiki/federatedlearning/federatedadaptertuning/) | Implements FedAdapter — federated bottleneck adapter tuning where small adapter modules are inserted into each transformer block and only these are communicated. |
| [`FederatedDPO<T>`](/docs/reference/wiki/federatedlearning/federateddpo/) | Implements Federated DPO (Direct Preference Optimization) for reward-model-free LLM alignment. |
| [`FederatedEWC<T>`](/docs/reference/wiki/federatedlearning/federatedewc/) | Federated Elastic Weight Consolidation (EWC) — prevents forgetting by penalizing changes to parameters that are important for previously learned tasks. |
| [`FederatedExperienceReplay<T>`](/docs/reference/wiki/federatedlearning/federatedexperiencereplay/) | Implements Federated Experience Replay for continual learning. |
| [`FederatedGraphPartitioner<T>`](/docs/reference/wiki/federatedlearning/federatedgraphpartitioner/) | Partitions a graph across federated clients using various strategies. |
| [`FederatedLoRA<T>`](/docs/reference/wiki/federatedlearning/federatedlora/) | Federated LoRA — Low-Rank Adaptation for parameter-efficient federated fine-tuning. |
| [`FederatedOrthogonalProjection<T>`](/docs/reference/wiki/federatedlearning/federatedorthogonalprojection/) | Federated Orthogonal Projection — prevents forgetting by projecting gradients to be orthogonal to the subspace of previously important parameter directions. |
| [`FederatedPromptTuning<T>`](/docs/reference/wiki/federatedlearning/federatedprompttuning/) | Federated Prompt Tuning — soft prompt aggregation for foundation model personalization. |
| [`FederatedRLHF<T>`](/docs/reference/wiki/federatedlearning/federatedrlhf/) | Configuration and orchestration for Federated RLHF (Reinforcement Learning from Human Feedback). |
| [`FetchSGDCompressor<T>`](/docs/reference/wiki/federatedlearning/fetchsgdcompressor/) | Implements FetchSGD — Count-Sketch + Top-k hybrid compression for massive models. |
| [`FlameAggregationStrategy<T>`](/docs/reference/wiki/federatedlearning/flameaggregationstrategy/) | Implements FLAME (Filtering via cosine similarity + Adaptive clipping + Noise) for Byzantine-robust federated learning with backdoor resistance. |
| [`FuzzyPsi`](/docs/reference/wiki/federatedlearning/fuzzypsi/) | Implements approximate entity matching using multiple similarity strategies. |
| [`GarbledCircuitData`](/docs/reference/wiki/federatedlearning/garbledcircuitdata/) | Contains the garbled circuit data produced by the garbler. |
| [`GarbledCircuitEvaluator`](/docs/reference/wiki/federatedlearning/garbledcircuitevaluator/) | Evaluates garbled circuits produced by `GarbledCircuitGenerator`. |
| [`GarbledCircuitGenerator`](/docs/reference/wiki/federatedlearning/garbledcircuitgenerator/) | Implements Yao's garbled circuit generation with point-and-permute, free XOR, and half-gates optimizations. |
| [`GaussianDifferentialPrivacyVector<T>`](/docs/reference/wiki/federatedlearning/gaussiandifferentialprivacyvector/) | Implements Gaussian differential privacy for vector-based model updates. |
| [`GaussianDifferentialPrivacy<T>`](/docs/reference/wiki/federatedlearning/gaussiandifferentialprivacy/) |  |
| [`GossipProtocol`](/docs/reference/wiki/federatedlearning/gossipprotocol/) | Gossip Protocol — randomized peer-to-peer model exchange for decentralized FL. |
| [`GradientAscentUnlearner<T>`](/docs/reference/wiki/federatedlearning/gradientascentunlearner/) | Approximate unlearning via gradient ascent: reverses learning by ascending the loss on target data. |
| [`GradientBoundednessProof<T>`](/docs/reference/wiki/federatedlearning/gradientboundednessproof/) | Proves that each gradient component is within [-B, B] (element-wise range proof). |
| [`GradientCommitmentData<T>`](/docs/reference/wiki/federatedlearning/gradientcommitmentdata/) | Contains the data for a gradient commitment. |
| [`GradientNormRangeProof<T>`](/docs/reference/wiki/federatedlearning/gradientnormrangeproof/) | Generates and verifies proofs that a gradient's L2 norm is within a declared bound. |
| [`GradientSketchCompressor<T>`](/docs/reference/wiki/federatedlearning/gradientsketchcompressor/) | Count Sketch-based gradient compression for federated learning. |
| [`GraphNeighborhoodPrivacy<T>`](/docs/reference/wiki/federatedlearning/graphneighborhoodprivacy/) | Applies local differential privacy (LDP) to neighborhood queries to prevent topology leakage. |
| [`GraphNodeGenerator<T>`](/docs/reference/wiki/federatedlearning/graphnodegenerator/) | Generates pseudo-node features for missing cross-client neighbors using a learned model. |
| [`GroupFairnessConstraint<T>`](/docs/reference/wiki/federatedlearning/groupfairnessconstraint/) | Enforces group fairness constraints during federated learning aggregation. |
| [`HashCommitmentScheme<T>`](/docs/reference/wiki/federatedlearning/hashcommitmentscheme/) | Implements a hash-based commitment scheme using SHA-256. |
| [`HeterogeneousLoRA<T>`](/docs/reference/wiki/federatedlearning/heterogeneouslora/) | Heterogeneous LoRA — supports different LoRA ranks per client with SVD-based aggregation. |
| [`HierarchicalFedLoRA<T>`](/docs/reference/wiki/federatedlearning/hierarchicalfedlora/) | Implements HierFedLoRA — Hierarchical LoRA aggregation for edge-cloud federated topologies. |
| [`HybridMpcProtocol<T>`](/docs/reference/wiki/federatedlearning/hybridmpcprotocol/) | Combines arithmetic secret sharing (for linear operations) with garbled circuits (for non-linear operations) into a single hybrid MPC protocol. |
| [`InMemoryFederatedTrainer<T, TInput, TOutput>`](/docs/reference/wiki/federatedlearning/inmemoryfederatedtrainer/) | In-memory federated learning trainer for local simulation and tests. |
| [`InfluenceFunctionUnlearner<T>`](/docs/reference/wiki/federatedlearning/influencefunctionunlearner/) | Influence function-based unlearning: mathematically estimates and subtracts a client's contribution. |
| [`IntelSgxTeeProvider<T>`](/docs/reference/wiki/federatedlearning/intelsgxteeprovider/) | TEE provider for Intel SGX (Software Guard Extensions) process-level enclaves. |
| [`IntelTdxTeeProvider<T>`](/docs/reference/wiki/federatedlearning/inteltdxteeprovider/) | TEE provider for Intel TDX (Trust Domain Extensions) confidential VMs. |
| [`KNNPersonalization<T>`](/docs/reference/wiki/federatedlearning/knnpersonalization/) | Implements kNN-Per — kNN-based personalization at inference time with zero extra training cost. |
| [`KrumFullModelAggregationStrategy<T, TInput, TOutput>`](/docs/reference/wiki/federatedlearning/krumfullmodelaggregationstrategy/) | Krum aggregation for `IFullModel` (Byzantine-robust selection by distance). |
| [`LabelDifferentialPrivacy<T>`](/docs/reference/wiki/federatedlearning/labeldifferentialprivacy/) | Implements differential privacy protection for label holder gradients in VFL. |
| [`LeafFederatedDatasetLoader<T>`](/docs/reference/wiki/federatedlearning/leaffederateddatasetloader/) | Loads LEAF benchmark JSON files into per-client datasets suitable for federated learning simulation. |
| [`LeafFederatedDataset<TInput, TOutput>`](/docs/reference/wiki/federatedlearning/leaffederateddataset/) | Represents a LEAF dataset with optional train/test splits. |
| [`LeafFederatedSplit<TInput, TOutput>`](/docs/reference/wiki/federatedlearning/leaffederatedsplit/) | Represents a single LEAF split (train/test) as per-client datasets. |
| [`LeafRedditFederatedDatasetLoader`](/docs/reference/wiki/federatedlearning/leafredditfederateddatasetloader/) | Loads the LEAF Reddit benchmark JSON files into per-client token-sequence datasets. |
| [`LeafSent140FederatedDatasetLoader`](/docs/reference/wiki/federatedlearning/leafsent140federateddatasetloader/) | Loads the LEAF Sent140 benchmark JSON files into per-client datasets. |
| [`LeafShakespeareFederatedDatasetLoader`](/docs/reference/wiki/federatedlearning/leafshakespearefederateddatasetloader/) | Loads the LEAF Shakespeare benchmark JSON files into per-client datasets. |
| [`LeafTokenSequenceFederatedDatasetLoader`](/docs/reference/wiki/federatedlearning/leaftokensequencefederateddatasetloader/) | Loads LEAF-style JSON files that store token sequences (`x`) and next-token labels (`y`). |
| [`LightweightShapleyEvaluator<T>`](/docs/reference/wiki/federatedlearning/lightweightshapleyevaluator/) | Implements Lightweight Shapley — O(n) Shapley value approximation using gradient similarity. |
| [`LossThresholdProof<T>`](/docs/reference/wiki/federatedlearning/lossthresholdproof/) | Proves that a client's local training loss is below a threshold without revealing the actual value. |
| [`MedianFullModelAggregationStrategy<T, TInput, TOutput>`](/docs/reference/wiki/federatedlearning/medianfullmodelaggregationstrategy/) | Coordinate-wise median aggregation for `IFullModel`. |
| [`MissingFeatureHandler<T>`](/docs/reference/wiki/federatedlearning/missingfeaturehandler/) | Handles missing features in vertical FL when not all parties have data for all entities. |
| [`ModelDriftDetector<T>`](/docs/reference/wiki/federatedlearning/modeldriftdetector/) | Model-based drift detector: uses gradient direction and weight divergence to detect drift. |
| [`ModelUpdateVerifier<T>`](/docs/reference/wiki/federatedlearning/modelupdateverifier/) | Server-side verification engine that checks all proofs from clients before aggregation. |
| [`MoonAggregationStrategy<T>`](/docs/reference/wiki/federatedlearning/moonaggregationstrategy/) | Implements MOON (Model-COntrastive Learning) aggregation strategy. |
| [`MultiKrumFullModelAggregationStrategy<T, TInput, TOutput>`](/docs/reference/wiki/federatedlearning/multikrumfullmodelaggregationstrategy/) | Multi-Krum aggregation for `IFullModel` (select m central updates, then average). |
| [`MultiPartyPsi`](/docs/reference/wiki/federatedlearning/multipartypsi/) | Implements multi-party Private Set Intersection for 3 or more parties. |
| [`MultiPartyPsiResult`](/docs/reference/wiki/federatedlearning/multipartypsiresult/) | Contains results of a multi-party PSI computation. |
| [`NeuralCleanseDetector<T>`](/docs/reference/wiki/federatedlearning/neuralcleansedetector/) | Neural Cleanse — post-hoc backdoor detection by reverse-engineering potential triggers. |
| [`ObliviousTransferPsi`](/docs/reference/wiki/federatedlearning/oblivioustransferpsi/) | Implements Oblivious Transfer based Private Set Intersection using cuckoo hashing. |
| [`OpenFedLLMPipeline<T>`](/docs/reference/wiki/federatedlearning/openfedllmpipeline/) | Implements OpenFedLLM pipeline patterns for federated LLM training, alignment, and serving. |
| [`OptiGradTrustAggregationStrategy<T>`](/docs/reference/wiki/federatedlearning/optigradtrustaggregationstrategy/) | Implements OptiGradTrust (Optimized Gradient Trust) aggregation strategy with historical reputation tracking. |
| [`OptimizedPrivateSetAnalytics<T>`](/docs/reference/wiki/federatedlearning/optimizedprivatesetanalytics/) | Implements Optimized Private Set Analytics (OPSA) beyond basic intersection. |
| [`PFedGatePersonalization<T>`](/docs/reference/wiki/federatedlearning/pfedgatepersonalization/) | Implements pFedGate — gated layer-wise mixture of local and global parameters. |
| [`PedersenCommitment<T>`](/docs/reference/wiki/federatedlearning/pedersencommitment/) | Implements Pedersen commitment scheme — additively homomorphic for verifiable aggregation. |
| [`PerformanceAwareClientSelectionStrategy`](/docs/reference/wiki/federatedlearning/performanceawareclientselectionstrategy/) | Performance-aware client selection using an explore/exploit policy over historical scores. |
| [`PersonalizedFederatedLearning<T>`](/docs/reference/wiki/federatedlearning/personalizedfederatedlearning/) | Implements personalized federated learning where each client maintains some client-specific parameters. |
| [`PowerSGDCompressor<T>`](/docs/reference/wiki/federatedlearning/powersgdcompressor/) | PowerSGD: low-rank gradient compression using randomized SVD approximation. |
| [`PrototypeFederatedGraphLearning<T>`](/docs/reference/wiki/federatedlearning/prototypefederatedgraphlearning/) | Prototype-based federated graph learning: clients share class prototypes instead of full model parameters. |
| [`PrototypicalContributionEvaluator<T>`](/docs/reference/wiki/federatedlearning/prototypicalcontributionevaluator/) | Prototypical contribution evaluator: measures client value using prototype representations. |
| [`ProxyCertificate`](/docs/reference/wiki/federatedlearning/proxycertificate/) | A signed certificate issued by the proxy after verifying a client update. |
| [`ProxyVerificationResult`](/docs/reference/wiki/federatedlearning/proxyverificationresult/) | Result of a proxy ZKP verification. |
| [`ProxyZKPVerifier<T>`](/docs/reference/wiki/federatedlearning/proxyzkpverifier/) | Implements Proxy-based Zero-Knowledge Proof verification for federated learning. |
| [`PsiResult`](/docs/reference/wiki/federatedlearning/psiresult/) | Contains the results of a Private Set Intersection computation. |
| [`QFairFederatedLearning<T>`](/docs/reference/wiki/federatedlearning/qfairfederatedlearning/) | Implements q-FFL (q-Fair Federated Learning) — parameterized fairness via power-mean. |
| [`RdpPrivacyAccountant`](/docs/reference/wiki/federatedlearning/rdpprivacyaccountant/) | A simple RĂ©nyi Differential Privacy (RDP) accountant for repeated Gaussian mechanisms. |
| [`RemoteAttestationResult`](/docs/reference/wiki/federatedlearning/remoteattestationresult/) | Represents the result of verifying a remote attestation quote from a TEE. |
| [`RfaFullModelAggregationStrategy<T, TInput, TOutput>`](/docs/reference/wiki/federatedlearning/rfafullmodelaggregationstrategy/) | Robust Federated Aggregation (RFA) via geometric median (Weiszfeld iterations). |
| [`RingAllReduceProtocol`](/docs/reference/wiki/federatedlearning/ringallreduceprotocol/) | Ring AllReduce — communication-efficient decentralized averaging using a ring topology. |
| [`SampledSubgraph<T>`](/docs/reference/wiki/federatedlearning/sampledsubgraph/) | Represents a sampled subgraph from neighborhood sampling. |
| [`ScaffoldHeterogeneityCorrection<T>`](/docs/reference/wiki/federatedlearning/scaffoldheterogeneitycorrection/) | SCAFFOLD-style heterogeneity correction using control variates. |
| [`SecureAggregationVector<T>`](/docs/reference/wiki/federatedlearning/secureaggregationvector/) | Implements secure aggregation for vector-based model updates. |
| [`SecureAggregation<T>`](/docs/reference/wiki/federatedlearning/secureaggregation/) | Implements secure aggregation for federated learning using cryptographic techniques. |
| [`SecureClippingProtocol<T>`](/docs/reference/wiki/federatedlearning/secureclippingprotocol/) | Implements secure gradient clipping without revealing gradient norms. |
| [`SecureComparisonProtocol<T>`](/docs/reference/wiki/federatedlearning/securecomparisonprotocol/) | Implements secure greater-than comparison on secret-shared values. |
| [`SecureCrossClientEdgeDiscovery<T>`](/docs/reference/wiki/federatedlearning/securecrossclientedgediscovery/) | PSI-based secure discovery of cross-client edges without revealing full adjacency lists. |
| [`SecureGradientExchange<T>`](/docs/reference/wiki/federatedlearning/securegradientexchange/) | Provides encryption for gradient tensors exchanged between parties in vertical FL. |
| [`SegmentedGossipProtocol<T>`](/docs/reference/wiki/federatedlearning/segmentedgossipprotocol/) | Implements Segmented Gossip — communication-efficient gossip that exchanges model segments. |
| [`SemiAsyncFederatedTrainer<T>`](/docs/reference/wiki/federatedlearning/semiasyncfederatedtrainer/) | Implements Semi-Asynchronous Federated Learning — hybrid sync/async with periodic barriers. |
| [`ShapleyValueEvaluator<T>`](/docs/reference/wiki/federatedlearning/shapleyvalueevaluator/) | Exact Shapley value evaluator: computes each client's marginal contribution across all coalitions. |
| [`ShuffleModelDP<T>`](/docs/reference/wiki/federatedlearning/shufflemodeldp/) | Implements Shuffle Model Differential Privacy for federated learning. |
| [`SimulatedTeeProvider<T>`](/docs/reference/wiki/federatedlearning/simulatedteeprovider/) | Software-simulated TEE provider for testing and development without hardware. |
| [`SparseLoRA<T>`](/docs/reference/wiki/federatedlearning/sparselora/) | Implements SLoRA — Sparse LoRA for communication-efficient federated fine-tuning. |
| [`SparseUpdate<T>`](/docs/reference/wiki/federatedlearning/sparseupdate/) | Represents a sparse update: only a subset of indices have non-zero values. |
| [`SplitNeuralNetwork<T>`](/docs/reference/wiki/federatedlearning/splitneuralnetwork/) | Implements a split neural network for vertical federated learning. |
| [`StatisticalDriftDetector<T>`](/docs/reference/wiki/federatedlearning/statisticaldriftdetector/) | Statistical drift detector: uses Page-Hinkley, ADWIN, or DDM tests on client metrics. |
| [`StratifiedClientSelectionStrategy`](/docs/reference/wiki/federatedlearning/stratifiedclientselectionstrategy/) | Stratified client selection using a client-to-group mapping. |
| [`SubgraphExpander<T>`](/docs/reference/wiki/federatedlearning/subgraphexpander/) | Expands a local subgraph with pseudo-nodes to approximate missing cross-client neighbors. |
| [`SubgraphFederatedTrainer<T>`](/docs/reference/wiki/federatedlearning/subgraphfederatedtrainer/) | Main coordinator for subgraph-level federated GNN training. |
| [`TeeSecureAggregation<T>`](/docs/reference/wiki/federatedlearning/teesecureaggregation/) | Performs weighted model aggregation inside a TEE enclave boundary. |
| [`ThresholdSecureAggregationVector<T>`](/docs/reference/wiki/federatedlearning/thresholdsecureaggregationvector/) | Implements dropout-resilient secure aggregation for vector-based model updates. |
| [`ThresholdSecureAggregation<T>`](/docs/reference/wiki/federatedlearning/thresholdsecureaggregation/) | Implements dropout-resilient secure aggregation for structured (layered) model updates. |
| [`TiltedERMFairness<T>`](/docs/reference/wiki/federatedlearning/tiltedermfairness/) | Implements TERM (Tilted Empirical Risk Minimization) for fairness in FL. |
| [`TimeVaryingTopology<T>`](/docs/reference/wiki/federatedlearning/timevaryingtopology/) | Implements time-varying topology for decentralized federated learning. |
| [`TopKSparsificationCompressor<T>`](/docs/reference/wiki/federatedlearning/topksparsificationcompressor/) | Implements Top-k Sparsification — send only the k largest gradient elements. |
| [`TrainingStep`](/docs/reference/wiki/federatedlearning/trainingstep/) | Represents a single training step in the integrity log. |
| [`TrainingStepLog`](/docs/reference/wiki/federatedlearning/trainingsteplog/) | Records training steps for computation integrity verification. |
| [`TrimmedMeanFullModelAggregationStrategy<T, TInput, TOutput>`](/docs/reference/wiki/federatedlearning/trimmedmeanfullmodelaggregationstrategy/) | Coordinate-wise trimmed mean aggregation for `IFullModel`. |
| [`UnbalancedPsi`](/docs/reference/wiki/federatedlearning/unbalancedpsi/) | Implements PSI optimized for asymmetric (unbalanced) set sizes. |
| [`UniformRandomClientSelectionStrategy`](/docs/reference/wiki/federatedlearning/uniformrandomclientselectionstrategy/) | Uniform random client selection (fractional participation). |
| [`UnlearningCertificate`](/docs/reference/wiki/federatedlearning/unlearningcertificate/) | Certificate proving that a client's data has been unlearned from the federated model. |
| [`UnlearningVerification`](/docs/reference/wiki/federatedlearning/unlearningverification/) | Contains the results of an unlearning verification check. |
| [`VerifiableAggregationStrategy<TModel>`](/docs/reference/wiki/federatedlearning/verifiableaggregationstrategy/) | Decorator that wraps any `IAggregationStrategy` with proof verification. |
| [`VerificationConstraint`](/docs/reference/wiki/federatedlearning/verificationconstraint/) | Describes a constraint to be proven in zero-knowledge. |
| [`VerificationProof`](/docs/reference/wiki/federatedlearning/verificationproof/) | Represents a zero-knowledge proof of a computational property. |
| [`VerificationRecord`](/docs/reference/wiki/federatedlearning/verificationrecord/) | Records a verification event for auditing. |
| [`VerticalDataPartitioner<T>`](/docs/reference/wiki/federatedlearning/verticaldatapartitioner/) | Partitions features (columns) across parties for vertical federated learning simulation. |
| [`VerticalFederatedBenchmark<T>`](/docs/reference/wiki/federatedlearning/verticalfederatedbenchmark/) | Provides benchmarking utilities for evaluating VFL implementations. |
| [`VerticalFederatedTrainer<T>`](/docs/reference/wiki/federatedlearning/verticalfederatedtrainer/) | Main orchestrator for vertical federated learning training. |
| [`VerticalFederatedUnlearner<T>`](/docs/reference/wiki/federatedlearning/verticalfederatedunlearner/) | Implements GDPR-compliant entity unlearning for vertical federated learning models. |
| [`VerticalPartyClient<T>`](/docs/reference/wiki/federatedlearning/verticalpartyclient/) | Represents a feature-holding party in vertical federated learning. |
| [`VerticalPartyLabelHolder<T>`](/docs/reference/wiki/federatedlearning/verticalpartylabelholder/) | Represents the label-holding party in vertical federated learning. |
| [`VflAlignmentSummary`](/docs/reference/wiki/federatedlearning/vflalignmentsummary/) | Contains summary statistics from the entity alignment phase of VFL. |
| [`VflBenchmarkDataset<T>`](/docs/reference/wiki/federatedlearning/vflbenchmarkdataset/) | Contains a complete benchmark dataset with vertically-partitioned data. |
| [`VflBenchmarkResult`](/docs/reference/wiki/federatedlearning/vflbenchmarkresult/) | Contains results from a VFL benchmark run. |
| [`VflEpochResult<T>`](/docs/reference/wiki/federatedlearning/vflepochresult/) | Contains metrics from a single VFL training epoch. |
| [`VflPartyDataset<T>`](/docs/reference/wiki/federatedlearning/vflpartydataset/) | Contains data for a single party in a benchmark dataset. |
| [`VflTrainingResult<T>`](/docs/reference/wiki/federatedlearning/vfltrainingresult/) | Contains the complete results from a VFL training run. |
| [`WeightedRandomClientSelectionStrategy`](/docs/reference/wiki/federatedlearning/weightedrandomclientselectionstrategy/) | Weighted random selection without replacement (typically weighted by sample count). |
| [`WinsorizedMeanFullModelAggregationStrategy<T, TInput, TOutput>`](/docs/reference/wiki/federatedlearning/winsorizedmeanfullmodelaggregationstrategy/) | Coordinate-wise winsorized mean aggregation for `IFullModel`. |
| [`ZOClientMessage`](/docs/reference/wiki/federatedlearning/zoclientmessage/) | Compact message from a client in the FedMeZO protocol. |

## Base Classes (13)

| Type | Summary |
|:-----|:--------|
| [`AggregationStrategyBase<TModel, T>`](/docs/reference/wiki/federatedlearning/aggregationstrategybase/) | Base class for federated aggregation strategies. |
| [`ClientSelectionStrategyBase`](/docs/reference/wiki/federatedlearning/clientselectionstrategybase/) | Base class for client selection strategies. |
| [`FederatedHeterogeneityCorrectionBase<T>`](/docs/reference/wiki/federatedlearning/federatedheterogeneitycorrectionbase/) | Base class for heterogeneity correction implementations. |
| [`FederatedLearningComponentBase<T>`](/docs/reference/wiki/federatedlearning/federatedlearningcomponentbase/) | Base class for federated learning components that need numeric operations for a generic numeric type. |
| [`FederatedServerOptimizerBase<T>`](/docs/reference/wiki/federatedlearning/federatedserveroptimizerbase/) | Base class for server-side federated optimizers. |
| [`FederatedTrainerBase<TModel, TData, TMetadata, T>`](/docs/reference/wiki/federatedlearning/federatedtrainerbase/) | Base class for federated learning trainers. |
| [`HomomorphicEncryptionProviderBase<T>`](/docs/reference/wiki/federatedlearning/homomorphicencryptionproviderbase/) | Base class for homomorphic encryption providers. |
| [`ParameterDictionaryAggregationStrategyBase<T>`](/docs/reference/wiki/federatedlearning/parameterdictionaryaggregationstrategybase/) | Base class for aggregation strategies operating on parameter dictionaries. |
| [`PrivacyAccountantBase`](/docs/reference/wiki/federatedlearning/privacyaccountantbase/) | Base class for privacy accountants. |
| [`PrivacyMechanismBase<TModel, T>`](/docs/reference/wiki/federatedlearning/privacymechanismbase/) | Base class for privacy mechanisms in federated learning. |
| [`PsiBase`](/docs/reference/wiki/federatedlearning/psibase/) | Base class providing shared functionality for PSI protocol implementations. |
| [`RobustFullModelAggregationStrategyBase<T, TInput, TOutput>`](/docs/reference/wiki/federatedlearning/robustfullmodelaggregationstrategybase/) | Base class for robust aggregation strategies operating on `IFullModel` parameters. |
| [`TeeProviderBase<T>`](/docs/reference/wiki/federatedlearning/teeproviderbase/) | Base class for TEE providers with common enclave lifecycle, sealing, and attestation logic. |

## Interfaces (30)

| Type | Summary |
|:-----|:--------|
| [`IBackdoorDetector<T>`](/docs/reference/wiki/federatedlearning/ibackdoordetector/) | Interface for detecting backdoor attacks in federated learning updates. |
| [`IClientContributionEvaluator<T>`](/docs/reference/wiki/federatedlearning/iclientcontributionevaluator/) | Evaluates how much each client contributed to the federated global model. |
| [`ICrossClientEdgeHandler<T>`](/docs/reference/wiki/federatedlearning/icrossclientedgehandler/) | Handles secure discovery and management of edges that cross client boundaries. |
| [`IDecentralizedTopology`](/docs/reference/wiki/federatedlearning/idecentralizedtopology/) | Interface for decentralized peer-to-peer network topologies in serverless federated learning. |
| [`IFairnessConstraint<T>`](/docs/reference/wiki/federatedlearning/ifairnessconstraint/) | Defines and enforces fairness constraints during federated learning aggregation. |
| [`IFederatedAdapterStrategy<T>`](/docs/reference/wiki/federatedlearning/ifederatedadapterstrategy/) | Interface for federated adapter strategies that enable parameter-efficient fine-tuning (PEFT) in FL. |
| [`IFederatedContinualLearningStrategy<T>`](/docs/reference/wiki/federatedlearning/ifederatedcontinuallearningstrategy/) | Interface for federated continual learning strategies that prevent catastrophic forgetting. |
| [`IFederatedDistillationStrategy<T>`](/docs/reference/wiki/federatedlearning/ifederateddistillationstrategy/) | Interface for federated knowledge distillation strategies that enable model-heterogeneous FL. |
| [`IFederatedDriftDetector<T>`](/docs/reference/wiki/federatedlearning/ifederateddriftdetector/) | Detects concept drift in federated learning by monitoring client model updates over time. |
| [`IFederatedGraphTrainer<T>`](/docs/reference/wiki/federatedlearning/ifederatedgraphtrainer/) | Orchestrates federated learning across clients holding subgraphs of a larger graph. |
| [`IFederatedUnlearner<T>`](/docs/reference/wiki/federatedlearning/ifederatedunlearner/) | Core interface for federated unlearning: removes a client's contribution from the global model. |
| [`IFuzzyMatcher`](/docs/reference/wiki/federatedlearning/ifuzzymatcher/) | Defines the interface for approximate entity matching in PSI. |
| [`IGarbledCircuit`](/docs/reference/wiki/federatedlearning/igarbledcircuit/) | Defines the contract for garbled circuit generation and evaluation. |
| [`IGradientCommitment<T>`](/docs/reference/wiki/federatedlearning/igradientcommitment/) | Defines the contract for committing to gradient values before revealing them. |
| [`IGraphAggregationStrategy<T>`](/docs/reference/wiki/federatedlearning/igraphaggregationstrategy/) | Graph-aware model aggregation strategy for federated GNN training. |
| [`IIncentiveMechanism<T>`](/docs/reference/wiki/federatedlearning/iincentivemechanism/) | Computes incentive rewards for federated learning participants based on their contributions. |
| [`ILabelProtector<T>`](/docs/reference/wiki/federatedlearning/ilabelprotector/) | Protects label holder information from being inferred by feature-holding parties. |
| [`IObliviousTransfer`](/docs/reference/wiki/federatedlearning/ioblivioustransfer/) | Defines the contract for an oblivious transfer (OT) protocol. |
| [`IPrivateSetIntersection`](/docs/reference/wiki/federatedlearning/iprivatesetintersection/) | Defines the interface for Private Set Intersection protocols. |
| [`IRemoteAttestationVerifier`](/docs/reference/wiki/federatedlearning/iremoteattestationverifier/) | Verifies remote attestation quotes from TEE enclaves. |
| [`ISecretSharingScheme<T>`](/docs/reference/wiki/federatedlearning/isecretsharingscheme/) | Defines the contract for a secret sharing scheme that splits and recombines tensor values. |
| [`ISecureComputationProtocol<T>`](/docs/reference/wiki/federatedlearning/isecurecomputationprotocol/) | Defines the contract for a multi-party computation protocol that can perform secure arithmetic and comparison operations on secret-shared values. |
| [`ISplitModel<T>`](/docs/reference/wiki/federatedlearning/isplitmodel/) | Represents a split neural network for vertical federated learning. |
| [`ISubgraphSampler<T>`](/docs/reference/wiki/federatedlearning/isubgraphsampler/) | Samples neighborhoods from a client's local subgraph for mini-batch GNN training. |
| [`ITeeProvider<T>`](/docs/reference/wiki/federatedlearning/iteeprovider/) | Abstracts a Trusted Execution Environment backend for enclave lifecycle, data sealing, and attestation quote generation. |
| [`ITeeSecureAggregator<T>`](/docs/reference/wiki/federatedlearning/iteesecureaggregator/) | Performs model aggregation inside a TEE enclave boundary. |
| [`IVerifiableComputation`](/docs/reference/wiki/federatedlearning/iverifiablecomputation/) | Defines the contract for generating and verifying proofs of computation correctness. |
| [`IVerticalFederatedTrainer<T>`](/docs/reference/wiki/federatedlearning/iverticalfederatedtrainer/) | Orchestrates the vertical federated learning training process. |
| [`IVerticalParty<T>`](/docs/reference/wiki/federatedlearning/iverticalparty/) | Represents a party in vertical federated learning that holds a subset of features. |
| [`IZkProofSystem`](/docs/reference/wiki/federatedlearning/izkproofsystem/) | Defines the abstract interface for a zero-knowledge proof backend. |

## Enums (12)

| Type | Summary |
|:-----|:--------|
| [`AlignmentDistanceMetric`](/docs/reference/wiki/federatedlearning/alignmentdistancemetric/) | Distance metric for feature alignment in FedAlign. |
| [`BlockSelectionStrategy`](/docs/reference/wiki/federatedlearning/blockselectionstrategy/) | Strategy for selecting which block to synchronize each round. |
| [`ConstraintType`](/docs/reference/wiki/federatedlearning/constrainttype/) | Types of constraints that can be proven in zero-knowledge. |
| [`DriftAction`](/docs/reference/wiki/federatedlearning/driftaction/) | Recommended action for a drifting client. |
| [`DriftType`](/docs/reference/wiki/federatedlearning/drifttype/) | Drift classification for a client's data distribution. |
| [`FedLLMStage`](/docs/reference/wiki/federatedlearning/fedllmstage/) | Specifies the training stage in the OpenFedLLM pipeline. |
| [`FedSamVariant`](/docs/reference/wiki/federatedlearning/fedsamvariant/) | Specifies the FedSAM variant to use. |
| [`GateType`](/docs/reference/wiki/federatedlearning/gatetype/) | Types of boolean gates supported in garbled circuits. |
| [`OneShotAggregationMode`](/docs/reference/wiki/federatedlearning/oneshotaggregationmode/) | Aggregation modes for One-Shot Federated Learning. |
| [`PEFTMethod`](/docs/reference/wiki/federatedlearning/peftmethod/) | Specifies the PEFT method used by FedPETuning. |
| [`PersonalizedLayerSelectionStrategy`](/docs/reference/wiki/federatedlearning/personalizedlayerselectionstrategy/) | Strategy for selecting which layers to personalize in federated learning. |
| [`TopologyStrategy<T>`](/docs/reference/wiki/federatedlearning/topologystrategy/) | Topology generation strategy. |

## Options & Configuration (4)

| Type | Summary |
|:-----|:--------|
| [`FederatedDPOOptions`](/docs/reference/wiki/federatedlearning/federateddpooptions/) | Configuration options for Federated DPO. |
| [`FederatedRLHFOptions`](/docs/reference/wiki/federatedlearning/federatedrlhfoptions/) | Configuration options for Federated RLHF. |
| [`LeafFederatedDatasetLoadOptions`](/docs/reference/wiki/federatedlearning/leaffederateddatasetloadoptions/) | Options controlling how LEAF federated benchmark JSON files are loaded. |
| [`OpenFedLLMOptions`](/docs/reference/wiki/federatedlearning/openfedllmoptions/) | Configuration options for the OpenFedLLM pipeline. |


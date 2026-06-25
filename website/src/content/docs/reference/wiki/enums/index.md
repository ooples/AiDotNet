---
title: "Enums"
description: "All 253 public types in the AiDotNet.enums namespace, organized by kind."
section: "API Reference"
---

**253** public types in this namespace, organized by kind.

## Enums (251)

| Type | Summary |
|:-----|:--------|
| [`AcquisitionFunctionType`](/docs/reference/wiki/enums/acquisitionfunctiontype/) | Represents different types of acquisition functions used in Bayesian optimization. |
| [`ActivationCategory`](/docs/reference/wiki/enums/activationcategory/) | Categories of activation functions based on their architectural role and behavior. |
| [`ActivationFunction`](/docs/reference/wiki/enums/activationfunction/) | Represents different activation functions used in neural networks and deep learning. |
| [`ActivationFunctionRole`](/docs/reference/wiki/enums/activationfunctionrole/) | Defines the functional roles of activation functions in neural networks. |
| [`ActivationTask`](/docs/reference/wiki/enums/activationtask/) | Tasks or architectural positions where an activation function is commonly used. |
| [`AdamAnomalyGuardMode`](/docs/reference/wiki/enums/adamanomalyguardmode/) | Policy for the PyTorch GradScaler-style anomaly guard on the Adam optimizer's tape-based `Step`. |
| [`AdditiveDecompositionAlgorithmType`](/docs/reference/wiki/enums/additivedecompositionalgorithmtype/) | Represents different algorithm types for additive decomposition of time series data. |
| [`AnomalyDetectorType`](/docs/reference/wiki/enums/anomalydetectortype/) | Specifies the type of anomaly detection algorithm to use. |
| [`AssignmentStrategy`](/docs/reference/wiki/enums/assignmentstrategy/) | Strategy for assigning requests to model versions during A/B testing. |
| [`AutoMLBudgetPreset`](/docs/reference/wiki/enums/automlbudgetpreset/) | Defines compute budget presets for AutoML runs. |
| [`AutoMLFinalModelSelectionPolicy`](/docs/reference/wiki/enums/automlfinalmodelselectionpolicy/) | Defines how AutoML chooses the final model to return after the search completes. |
| [`AutoMLSearchStrategy`](/docs/reference/wiki/enums/automlsearchstrategy/) | Defines the search strategy used to explore AutoML candidate configurations. |
| [`AutoMLStatus`](/docs/reference/wiki/enums/automlstatus/) | Represents the current status of an AutoML search process. |
| [`AutoMLTaskFamily`](/docs/reference/wiki/enums/automltaskfamily/) | Defines the high-level task family for an AutoML run. |
| [`BenchmarkExecutionStatus`](/docs/reference/wiki/enums/benchmarkexecutionstatus/) | Represents the execution outcome for a benchmark suite run. |
| [`BenchmarkFailurePolicy`](/docs/reference/wiki/enums/benchmarkfailurepolicy/) | Controls how the benchmark runner should behave when one or more suites fail. |
| [`BenchmarkMetric`](/docs/reference/wiki/enums/benchmarkmetric/) | Defines standardized metrics used in benchmark reports. |
| [`BenchmarkReportDetailLevel`](/docs/reference/wiki/enums/benchmarkreportdetaillevel/) | Specifies how much detail should be included in benchmark reports. |
| [`BenchmarkSuite`](/docs/reference/wiki/enums/benchmarksuite/) | Defines the supported benchmark suites available through the AiDotNet facade. |
| [`BenchmarkSuiteKind`](/docs/reference/wiki/enums/benchmarksuitekind/) | Categorizes benchmark suites by their evaluation style and infrastructure requirements. |
| [`BetaSchedule`](/docs/reference/wiki/enums/betaschedule/) | Defines the types of beta (noise variance) schedules available for diffusion models. |
| [`BeveridgeNelsonAlgorithmType`](/docs/reference/wiki/enums/beveridgenelsonalgorithmtype/) | Represents different algorithm types for Beveridge-Nelson decomposition of time series data. |
| [`BidiagonalAlgorithmType`](/docs/reference/wiki/enums/bidiagonalalgorithmtype/) | Represents different algorithm types for bidiagonal matrix decomposition. |
| [`BoundaryHandlingMethod`](/docs/reference/wiki/enums/boundaryhandlingmethod/) | Specifies how to handle boundaries when processing data that extends beyond the available range. |
| [`CLIPVariant`](/docs/reference/wiki/enums/clipvariant/) | Variants for CLIP text encoder models. |
| [`CacheEvictionPolicy`](/docs/reference/wiki/enums/cacheevictionpolicy/) | Cache eviction policies for managing limited cache memory. |
| [`CalibrationMethod`](/docs/reference/wiki/enums/calibrationmethod/) | Calibration methods for quantization - techniques to determine optimal scaling factors when converting high-precision models to low-precision formats. |
| [`CausalDiscoveryAlgorithmType`](/docs/reference/wiki/enums/causaldiscoveryalgorithmtype/) | Specifies the algorithm to use for causal structure learning (DAG discovery). |
| [`CausalDiscoveryCategory`](/docs/reference/wiki/enums/causaldiscoverycategory/) | Categories of causal discovery algorithms based on their methodology. |
| [`ChainType`](/docs/reference/wiki/enums/chaintype/) | Represents different types of chains for composing language model operations. |
| [`ChatGLM3Variant`](/docs/reference/wiki/enums/chatglm3variant/) | Variants for the ChatGLM3 text encoder. |
| [`CheckpointMetricType`](/docs/reference/wiki/enums/checkpointmetrictype/) | Standard metrics for checkpoint selection and early stopping. |
| [`CholeskyAlgorithmType`](/docs/reference/wiki/enums/choleskyalgorithmtype/) | Represents different algorithm types for Cholesky decomposition of matrices. |
| [`ClassificationTaskType`](/docs/reference/wiki/enums/classificationtasktype/) | Specifies the type of classification task being performed. |
| [`ClusteringMetricType`](/docs/reference/wiki/enums/clusteringmetrictype/) | Defines the types of metrics used to evaluate clustering quality. |
| [`ComponentType`](/docs/reference/wiki/enums/componenttype/) | Defines the type of an AI pipeline component (Tier 2 metadata). |
| [`CompressionType`](/docs/reference/wiki/enums/compressiontype/) | Defines the types of model compression strategies available in the AiDotNet library. |
| [`ComputationalBudget`](/docs/reference/wiki/enums/computationalbudget/) | Describes the computational budget available for model training and search. |
| [`ComputeCost`](/docs/reference/wiki/enums/computecost/) | Relative computational cost of an operation. |
| [`ConcertoModelSize`](/docs/reference/wiki/enums/concertomodelsize/) | Defines the backbone size variants for Concerto (joint 2D-3D point cloud segmentation). |
| [`ConditionNumberMethod`](/docs/reference/wiki/enums/conditionnumbermethod/) | Specifies different methods for calculating the condition number of a matrix. |
| [`ConformalPredictionMode`](/docs/reference/wiki/enums/conformalpredictionmode/) | Defines conformal prediction calibration modes. |
| [`ContrastiveLossType`](/docs/reference/wiki/enums/contrastivelosstype/) | Types of contrastive loss functions for knowledge distillation. |
| [`ConvexSolverType`](/docs/reference/wiki/enums/convexsolvertype/) | Types of convex solvers available for MetaOptNet. |
| [`CrossValidationType`](/docs/reference/wiki/enums/crossvalidationtype/) | Defines the types of cross-validation strategies available. |
| [`DEVAModelSize`](/docs/reference/wiki/enums/devamodelsize/) | Defines the backbone size variants for DEVA (Decoupled Video Segmentation). |
| [`DataComplexity`](/docs/reference/wiki/enums/datacomplexity/) | Represents the level of complexity in a dataset, which helps determine appropriate model selection and preprocessing. |
| [`DataSetType`](/docs/reference/wiki/enums/datasettype/) | Represents the different types of datasets used in machine learning workflows. |
| [`DecompositionComponentType`](/docs/reference/wiki/enums/decompositioncomponenttype/) | Represents the different components that can be extracted when decomposing a time series. |
| [`DenseNetVariant`](/docs/reference/wiki/enums/densenetvariant/) | Specifies the DenseNet model variant. |
| [`DiffusionPredictionType`](/docs/reference/wiki/enums/diffusionpredictiontype/) | Defines what the diffusion model predicts during the denoising process. |
| [`DistanceMetricType`](/docs/reference/wiki/enums/distancemetrictype/) | Represents different methods for measuring the distance or similarity between data points. |
| [`DistillationStrategyType`](/docs/reference/wiki/enums/distillationstrategytype/) | Specifies the type of knowledge distillation strategy to use for transferring knowledge from teacher to student models. |
| [`DistilledT5Variant`](/docs/reference/wiki/enums/distilledt5variant/) | Variants for distilled T5 text encoder models. |
| [`DistributedStrategy`](/docs/reference/wiki/enums/distributedstrategy/) | Defines the distributed training strategy to use. |
| [`DistributionType`](/docs/reference/wiki/enums/distributiontype/) | Represents different probability distributions used in statistical modeling and machine learning. |
| [`EMDAlgorithmType`](/docs/reference/wiki/enums/emdalgorithmtype/) | Represents different algorithm types for Empirical Mode Decomposition (EMD). |
| [`EdgeDeviceType`](/docs/reference/wiki/enums/edgedevicetype/) | Types of edge devices for optimization targeting. |
| [`EdgeDirection`](/docs/reference/wiki/enums/edgedirection/) | Specifies the direction of edges to retrieve when querying a knowledge graph. |
| [`EfficientNetVariant`](/docs/reference/wiki/enums/efficientnetvariant/) | Specifies the EfficientNet model variant. |
| [`EfficientTAMModelSize`](/docs/reference/wiki/enums/efficienttammodelsize/) | Defines the backbone size variants for EfficientTAM (Track Anything Model). |
| [`EigenAlgorithmType`](/docs/reference/wiki/enums/eigenalgorithmtype/) | Represents different algorithm types for computing eigenvalues and eigenvectors of matrices. |
| [`EnvelopeType`](/docs/reference/wiki/enums/envelopetype/) | Specifies whether to use an upper or lower envelope in signal processing and data analysis operations. |
| [`EoMTModelSize`](/docs/reference/wiki/enums/eomtmodelsize/) | Defines the backbone size variants for EoMT (Encoder-only Mask Transformer). |
| [`ExpressionNodeType`](/docs/reference/wiki/enums/expressionnodetype/) | Defines the different types of nodes that can exist in a computational graph. |
| [`FeatureExtractionStrategy`](/docs/reference/wiki/enums/featureextractionstrategy/) | Defines strategies for extracting features from higher-dimensional tensors. |
| [`FederatedPartitioningStrategy`](/docs/reference/wiki/enums/federatedpartitioningstrategy/) | Defines how a centralized dataset should be partitioned into per-client datasets for federated simulations. |
| [`FewShotSelectionStrategy`](/docs/reference/wiki/enums/fewshotselectionstrategy/) | Represents strategies for selecting few-shot examples in prompt templates. |
| [`FitType`](/docs/reference/wiki/enums/fittype/) | Represents different types of model fit quality and common issues in machine learning models. |
| [`FitnessCalculatorType`](/docs/reference/wiki/enums/fitnesscalculatortype/) | Specifies different loss functions and fitness calculators for evaluating model performance. |
| [`FluxPredictorVariant`](/docs/reference/wiki/enums/fluxpredictorvariant/) | Variants for FLUX noise predictor architectures. |
| [`FluxVariant`](/docs/reference/wiki/enums/fluxvariant/) | Variants for FLUX model. |
| [`FoundationModelSize`](/docs/reference/wiki/enums/foundationmodelsize/) | Defines the size variants available for foundation models. |
| [`GemmaVariant`](/docs/reference/wiki/enums/gemmavariant/) | Variants for Gemma text encoder models. |
| [`GeneticNodeType`](/docs/reference/wiki/enums/geneticnodetype/) | Types of nodes in a genetic programming tree. |
| [`GradientType`](/docs/reference/wiki/enums/gradienttype/) | Specifies different types of gradient descent optimization algorithms used in machine learning. |
| [`GramSchmidtAlgorithmType`](/docs/reference/wiki/enums/gramschmidtalgorithmtype/) | Represents different algorithm types for the Gram-Schmidt orthogonalization process. |
| [`GraphGenerationType`](/docs/reference/wiki/enums/graphgenerationtype/) | Type of graph generation approach. |
| [`GraphRAGMode`](/docs/reference/wiki/enums/graphragmode/) | Specifies the retrieval mode for enhanced GraphRAG. |
| [`GuidanceType`](/docs/reference/wiki/enums/guidancetype/) | Types of guidance methods for diffusion model inference. |
| [`HessenbergAlgorithmType`](/docs/reference/wiki/enums/hessenbergalgorithmtype/) | Represents different algorithm types for Hessenberg decomposition of matrices. |
| [`HiDreamVariant`](/docs/reference/wiki/enums/hidreamvariant/) | Variants for HiDream-I1 model. |
| [`HodrickPrescottAlgorithmType`](/docs/reference/wiki/enums/hodrickprescottalgorithmtype/) | Represents different algorithm types for implementing the Hodrick-Prescott filter. |
| [`ImportanceThresholdStrategy`](/docs/reference/wiki/enums/importancethresholdstrategy/) | Defines strategies for setting the importance threshold in feature selection. |
| [`InfraType`](/docs/reference/wiki/enums/infratype/) | Defines the type of infrastructure component (Tier 3 metadata). |
| [`InitializationMethod`](/docs/reference/wiki/enums/initializationmethod/) | Methods for initializing a population. |
| [`InputType`](/docs/reference/wiki/enums/inputtype/) | Specifies the dimensionality of input data for machine learning models. |
| [`InternImageModelSize`](/docs/reference/wiki/enums/internimagemodelsize/) | Defines the model size variants for InternImage (DCNv3-based CNN backbone). |
| [`Interpolation2DType`](/docs/reference/wiki/enums/interpolation2dtype/) | Specifies different methods for interpolating 2D data points to create a continuous surface. |
| [`InverseType`](/docs/reference/wiki/enums/inversetype/) | Specifies different algorithms for calculating matrix inverses in mathematical operations. |
| [`KGEmbeddingType`](/docs/reference/wiki/enums/kgembeddingtype/) | Specifies the type of knowledge graph embedding model to use. |
| [`KMaXDeepLabModelSize`](/docs/reference/wiki/enums/kmaxdeeplabmodelsize/) | Defines the backbone size variants for kMaX-DeepLab panoptic segmentation. |
| [`KandinskyVersion`](/docs/reference/wiki/enums/kandinskyversion/) | Versions for the Kandinsky model family. |
| [`KernelType`](/docs/reference/wiki/enums/kerneltype/) | Specifies different kernel functions used in machine learning algorithms like Support Vector Machines (SVMs). |
| [`LLMProvider`](/docs/reference/wiki/enums/llmprovider/) | Defines the large language model (LLM) providers available for AI agent assistance during model building and inference. |
| [`LanguageModelBackbone`](/docs/reference/wiki/enums/languagemodelbackbone/) | Defines the language model backbone types used in multimodal neural networks. |
| [`LayerApiShape`](/docs/reference/wiki/enums/layerapishape/) | Describes the Forward method signature shape a layer uses. |
| [`LayerTask`](/docs/reference/wiki/enums/layertask/) | Tasks or architectural roles that a neural network layer performs. |
| [`LayerType`](/docs/reference/wiki/enums/layertype/) | Specifies different types of layers used in neural networks, particularly in deep learning models. |
| [`LdlAlgorithmType`](/docs/reference/wiki/enums/ldlalgorithmtype/) | Represents different algorithm types for LDL decomposition of matrices. |
| [`LicenseKeyStatus`](/docs/reference/wiki/enums/licensekeystatus/) | Represents the current status of a license key after validation. |
| [`LossApiShape`](/docs/reference/wiki/enums/lossapishape/) | Describes the method signature shape a loss function uses for its primary calculation. |
| [`LossCategory`](/docs/reference/wiki/enums/losscategory/) | Categories of loss functions based on the type of learning task they serve. |
| [`LossTask`](/docs/reference/wiki/enums/losstask/) | Specific tasks a loss function is designed or commonly used for. |
| [`LossTestInputFormat`](/docs/reference/wiki/enums/losstestinputformat/) | Describes what kind of test data a loss function expects. |
| [`LossType`](/docs/reference/wiki/enums/losstype/) | Defines different loss functions used to measure how well a model's predictions match the actual values. |
| [`LqAlgorithmType`](/docs/reference/wiki/enums/lqalgorithmtype/) | Represents different algorithm types for LQ decomposition of matrices. |
| [`LuAlgorithmType`](/docs/reference/wiki/enums/lualgorithmtype/) | Represents different algorithm types for LU decomposition of matrices. |
| [`MMDiTXVariant`](/docs/reference/wiki/enums/mmditxvariant/) | Variants for MMDiT noise predictor architectures. |
| [`Mask2FormerModelSize`](/docs/reference/wiki/enums/mask2formermodelsize/) | Defines the backbone size variants for Mask2Former. |
| [`MaskDINOModelSize`](/docs/reference/wiki/enums/maskdinomodelsize/) | Defines the backbone size variants for Mask DINO. |
| [`MatrixDecompositionType`](/docs/reference/wiki/enums/matrixdecompositiontype/) | Specifies different methods for breaking down (decomposing) matrices into simpler components. |
| [`MatrixLayout`](/docs/reference/wiki/enums/matrixlayout/) | Specifies how data is organized in matrices when working with arrays of data. |
| [`MatrixType`](/docs/reference/wiki/enums/matrixtype/) | Defines the different types of matrices that can be used in mathematical operations. |
| [`MedNeXtModelSize`](/docs/reference/wiki/enums/mednextmodelsize/) | Defines the backbone size variants for MedNeXt medical segmentation. |
| [`MedSAM2ModelSize`](/docs/reference/wiki/enums/medsam2modelsize/) | Defines the backbone size variants for MedSAM 2 (3D Medical SAM). |
| [`MedSAMModelSize`](/docs/reference/wiki/enums/medsammodelsize/) | Defines the backbone size variants for MedSAM (Medical SAM). |
| [`MetricOptimizationDirection`](/docs/reference/wiki/enums/metricoptimizationdirection/) | Specifies the direction for metric optimization (whether lower or higher values are better). |
| [`MetricType`](/docs/reference/wiki/enums/metrictype/) | Defines the types of metrics used to evaluate machine learning models. |
| [`MixedPrecisionType`](/docs/reference/wiki/enums/mixedprecisiontype/) | Types of mixed precision training data types. |
| [`MobileNetV2WidthMultiplier`](/docs/reference/wiki/enums/mobilenetv2widthmultiplier/) | Specifies the width multiplier for MobileNetV2. |
| [`MobileNetV3Variant`](/docs/reference/wiki/enums/mobilenetv3variant/) | Specifies the MobileNetV3 model variant. |
| [`MobileNetV3WidthMultiplier`](/docs/reference/wiki/enums/mobilenetv3widthmultiplier/) | Specifies the width multiplier for MobileNetV3. |
| [`ModelCategory`](/docs/reference/wiki/enums/modelcategory/) | Defines the algorithm family or category that a machine learning model belongs to. |
| [`ModelComplexity`](/docs/reference/wiki/enums/modelcomplexity/) | Indicates the computational complexity and resource requirements of a machine learning model. |
| [`ModelCompressionMode`](/docs/reference/wiki/enums/modelcompressionmode/) | Defines the mode of model compression to apply during serialization. |
| [`ModelDomain`](/docs/reference/wiki/enums/modeldomain/) | Defines the application domain or field that a machine learning model is designed for. |
| [`ModelPerformance`](/docs/reference/wiki/enums/modelperformance/) | Represents the overall performance quality of a machine learning model. |
| [`ModelTask`](/docs/reference/wiki/enums/modeltask/) | Defines the specific task or capability that a machine learning model performs. |
| [`ModificationType`](/docs/reference/wiki/enums/modificationtype/) | Represents the types of modifications that can be applied to a model structure. |
| [`MultiplicativeAlgorithmType`](/docs/reference/wiki/enums/multiplicativealgorithmtype/) | Represents different multiplicative algorithm types for time series analysis and forecasting. |
| [`NERModelVariant`](/docs/reference/wiki/enums/nermodelvariant/) | Defines common model size variants for Named Entity Recognition (NER) models. |
| [`NetworkComplexity`](/docs/reference/wiki/enums/networkcomplexity/) | Defines the complexity level of a neural network architecture. |
| [`NeuralArchitectureSearchStrategy`](/docs/reference/wiki/enums/neuralarchitecturesearchstrategy/) | Specifies the strategy used for neural architecture search. |
| [`NeuralNetworkTaskType`](/docs/reference/wiki/enums/neuralnetworktasktype/) | Defines the different types of tasks that a neural network can be designed to perform. |
| [`NnUNetModelSize`](/docs/reference/wiki/enums/nnunetmodelsize/) | Defines the architecture variants for nnU-Net v2 medical segmentation. |
| [`NormalizationMethod`](/docs/reference/wiki/enums/normalizationmethod/) | Defines different methods for normalizing data before processing in machine learning algorithms. |
| [`ODISEModelSize`](/docs/reference/wiki/enums/odisemodelsize/) | Defines the backbone size variants for ODISE (Open-vocabulary DIffusion-based panoptic SEgmentation). |
| [`OMGSegModelSize`](/docs/reference/wiki/enums/omgsegmodelsize/) | Defines the backbone size variants for OMG-Seg. |
| [`OneFormerModelSize`](/docs/reference/wiki/enums/oneformermodelsize/) | Defines the backbone size variants for OneFormer. |
| [`OperationType`](/docs/reference/wiki/enums/operationtype/) | Represents different operation types in computation graphs for JIT compilation and automatic differentiation. |
| [`OptimizationMode`](/docs/reference/wiki/enums/optimizationmode/) | Specifies the mode of optimization for an optimizer. |
| [`OptimizationPassType`](/docs/reference/wiki/enums/optimizationpasstype/) | Represents the type of optimization pass applied to the computation graph. |
| [`OptimizerType`](/docs/reference/wiki/enums/optimizertype/) | Defines different optimization algorithms used to train machine learning models. |
| [`OutlierDetectionMethod`](/docs/reference/wiki/enums/outlierdetectionmethod/) | Defines different methods for detecting outliers in datasets. |
| [`OutputDistribution`](/docs/reference/wiki/enums/outputdistribution/) | Specifies the target distribution for quantile transformation. |
| [`OutputType`](/docs/reference/wiki/enums/outputtype/) | The expected format of model outputs that a loss function operates on. |
| [`PIDNetModelSize`](/docs/reference/wiki/enums/pidnetmodelsize/) | Defines the backbone size variants for PIDNet real-time segmentation. |
| [`PNAAggregator`](/docs/reference/wiki/enums/pnaaggregator/) | Aggregation function types for Principal Neighbourhood Aggregation (PNA). |
| [`PNAScaler`](/docs/reference/wiki/enums/pnascaler/) | Scaler function types for Principal Neighbourhood Aggregation (PNA). |
| [`ParameterType`](/docs/reference/wiki/enums/parametertype/) | Defines the types of parameters that can be used in hyperparameter search |
| [`PartitionStrategy`](/docs/reference/wiki/enums/partitionstrategy/) | Strategies for partitioning models between cloud and edge devices. |
| [`PayloadEncryptionScheme`](/docs/reference/wiki/enums/payloadencryptionscheme/) | Specifies the encryption scheme applied to the model payload within an AIMF envelope. |
| [`PipelineStage`](/docs/reference/wiki/enums/pipelinestage/) | Defines which stage of an AI pipeline a component operates in. |
| [`PixArtVariant`](/docs/reference/wiki/enums/pixartvariant/) | Variants for the PixArt model family. |
| [`PointTransformerV3ModelSize`](/docs/reference/wiki/enums/pointtransformerv3modelsize/) | Defines the backbone size variants for Point Transformer V3 (3D point cloud segmentation). |
| [`PolarAlgorithmType`](/docs/reference/wiki/enums/polaralgorithmtype/) | Represents different algorithm types for computing the polar decomposition of matrices. |
| [`PoolingType`](/docs/reference/wiki/enums/poolingtype/) | Defines different methods for pooling (downsampling) data in neural networks, particularly in convolutional neural networks. |
| [`PositionalEncodingType`](/docs/reference/wiki/enums/positionalencodingtype/) | Represents the type of positional encoding used in transformer attention layers. |
| [`PrecisionMode`](/docs/reference/wiki/enums/precisionmode/) | Defines the numeric precision mode for neural network training and computation. |
| [`PredictionType`](/docs/reference/wiki/enums/predictiontype/) | Specifies the type of prediction task that a machine learning model performs. |
| [`PreferenceLossType`](/docs/reference/wiki/enums/preferencelosstype/) | Types of loss functions for preference optimization methods. |
| [`ProbabilityCalibrationMethod`](/docs/reference/wiki/enums/probabilitycalibrationmethod/) | Defines probability calibration strategies for classification-like outputs. |
| [`PromptOptimizationStrategy`](/docs/reference/wiki/enums/promptoptimizationstrategy/) | Represents strategies for optimizing prompts to improve language model performance. |
| [`PromptTemplateType`](/docs/reference/wiki/enums/prompttemplatetype/) | Represents different types of prompt templates for language model interactions. |
| [`QATMethod`](/docs/reference/wiki/enums/qatmethod/) | Specifies the Quantization-Aware Training (QAT) method to use during model training. |
| [`QrAlgorithmType`](/docs/reference/wiki/enums/qralgorithmtype/) | Represents different algorithm types for computing the QR decomposition of matrices. |
| [`QualityLevel`](/docs/reference/wiki/enums/qualitylevel/) | Quality levels for adaptive inference on resource-constrained devices. |
| [`QuantizationGranularity`](/docs/reference/wiki/enums/quantizationgranularity/) | Specifies the granularity level for quantization scaling factors. |
| [`QuantizationMode`](/docs/reference/wiki/enums/quantizationmode/) | Specifies the quantization mode for model optimization and export. |
| [`QuantizationStrategy`](/docs/reference/wiki/enums/quantizationstrategy/) | Specifies the quantization strategy (algorithm) to use for model compression. |
| [`QueryMeldNetModelSize`](/docs/reference/wiki/enums/querymeldnetmodelsize/) | Defines the backbone size variants for QueryMeldNet (MQ-Former). |
| [`Qwen2Variant`](/docs/reference/wiki/enums/qwen2variant/) | Variants for Qwen2 text encoder models. |
| [`RLAutoMLAgentType`](/docs/reference/wiki/enums/rlautomlagenttype/) | Defines which reinforcement learning agent families can be explored by AutoML. |
| [`RedactionStrategy`](/docs/reference/wiki/enums/redactionstrategy/) | Strategy for redacting detected PII from text. |
| [`RegularizationType`](/docs/reference/wiki/enums/regularizationtype/) | Specifies the type of regularization to apply to a machine learning model. |
| [`RelationAggregationMethod`](/docs/reference/wiki/enums/relationaggregationmethod/) | Methods for aggregating multiple relation scores in Relation Networks. |
| [`RelationModuleType`](/docs/reference/wiki/enums/relationmoduletype/) | Types of relation module architectures for Relation Networks. |
| [`ResNetVariant`](/docs/reference/wiki/enums/resnetvariant/) | Defines the available ResNet (Residual Network) architecture variants. |
| [`SAGEAggregatorType`](/docs/reference/wiki/enums/sageaggregatortype/) | Aggregation function type for GraphSAGE. |
| [`SAM21ModelSize`](/docs/reference/wiki/enums/sam21modelsize/) | Defines the backbone size variants for SAM 2.1 (Segment Anything Model 2.1). |
| [`SAMHQModelSize`](/docs/reference/wiki/enums/samhqmodelsize/) | Defines the backbone size variants for SAM-HQ (High-Quality Segment Anything Model). |
| [`SAMModelSize`](/docs/reference/wiki/enums/sammodelsize/) | Defines the backbone size variants for SAM (Segment Anything Model). |
| [`SANAVariant`](/docs/reference/wiki/enums/sanavariant/) | Variants for the SANA model family. |
| [`SD3Variant`](/docs/reference/wiki/enums/sd3variant/) | Variants for the Stable Diffusion 3 model family. |
| [`SEATSAlgorithmType`](/docs/reference/wiki/enums/seatsalgorithmtype/) | Represents different algorithm types for SEATS (Seasonal Extraction in ARIMA Time Series) decomposition. |
| [`SEEMModelSize`](/docs/reference/wiki/enums/seemmodelsize/) | Defines the backbone size variants for SEEM (Segment Everything Everywhere All at Once). |
| [`SSAAlgorithmType`](/docs/reference/wiki/enums/ssaalgorithmtype/) | Represents different algorithm types for Singular Spectrum Analysis (SSA). |
| [`SSLMethodCategory`](/docs/reference/wiki/enums/sslmethodcategory/) | Categorizes self-supervised learning methods by their learning paradigm. |
| [`SSLMethodType`](/docs/reference/wiki/enums/sslmethodtype/) | Specifies the type of self-supervised learning method to use for representation learning. |
| [`STLAlgorithmType`](/docs/reference/wiki/enums/stlalgorithmtype/) | Represents different algorithm types for Seasonal-Trend decomposition using LOESS (STL). |
| [`SafetyAction`](/docs/reference/wiki/enums/safetyaction/) | Defines the action to take when a safety violation is detected. |
| [`SafetyCategory`](/docs/reference/wiki/enums/safetycategory/) | Comprehensive taxonomy of safety and harm categories for content classification. |
| [`SafetySeverity`](/docs/reference/wiki/enums/safetyseverity/) | Indicates the severity level of a safety finding. |
| [`SamplingType`](/docs/reference/wiki/enums/samplingtype/) | Specifies the method used to sample or combine values when reducing data dimensions. |
| [`SchurAlgorithmType`](/docs/reference/wiki/enums/schuralgorithmtype/) | Represents different algorithm types for computing the Schur decomposition of matrices. |
| [`SegFormerModelSize`](/docs/reference/wiki/enums/segformermodelsize/) | Defines the model size variants for SegFormer (Mix Transformer backbone). |
| [`SegGPTModelSize`](/docs/reference/wiki/enums/seggptmodelsize/) | Defines the backbone size variants for SegGPT (interactive/in-context segmentation). |
| [`SegNeXtModelSize`](/docs/reference/wiki/enums/segnextmodelsize/) | Defines the model size variants for SegNeXt (Multi-Scale Convolutional Attention backbone). |
| [`SelectionMethod`](/docs/reference/wiki/enums/selectionmethod/) | Methods for selecting individuals for reproduction. |
| [`SequencePoolingMode`](/docs/reference/wiki/enums/sequencepoolingmode/) | Strategy for collapsing a transformer encoder's `[batch, seq, dim]` hidden states into a single `[batch, dim]` vector before the classification head, when the task is single-label per sequence. |
| [`SequentialFeatureSelectionDirection`](/docs/reference/wiki/enums/sequentialfeatureselectiondirection/) | Defines the direction of sequential feature selection. |
| [`SerializationFormat`](/docs/reference/wiki/enums/serializationformat/) | Specifies the serialization format used for the model payload within an AIMF envelope. |
| [`SigLIP2Variant`](/docs/reference/wiki/enums/siglip2variant/) | Variants for SigLIP 2 text encoder models. |
| [`SigLIPVariant`](/docs/reference/wiki/enums/siglipvariant/) | Variants for SigLIP text encoder models. |
| [`SimCSEType`](/docs/reference/wiki/enums/simcsetype/) | Defines the training paradigms for SimCSE (Simple Contrastive Learning of Sentence Embeddings). |
| [`SonataModelSize`](/docs/reference/wiki/enums/sonatamodelsize/) | Defines the backbone size variants for Sonata (3D point cloud segmentation with self-distillation). |
| [`SpikingNeuronType`](/docs/reference/wiki/enums/spikingneurontype/) | Specifies the type of spiking neuron model to use in neuromorphic computing simulations. |
| [`SplitCriterion`](/docs/reference/wiki/enums/splitcriterion/) | Specifies the criterion used to determine the best way to split data in decision trees and other tree-based models. |
| [`StepwiseMethod`](/docs/reference/wiki/enums/stepwisemethod/) | Specifies the direction of feature selection in stepwise regression and other statistical models. |
| [`StreamingTrainingMode`](/docs/reference/wiki/enums/streamingtrainingmode/) | Controls the memory-bounded streaming training path (optimizer-in-backward with 8-bit Adam state and topological-min gradient release). |
| [`SvdAlgorithmType`](/docs/reference/wiki/enums/svdalgorithmtype/) | Represents different algorithm types for Singular Value Decomposition (SVD). |
| [`SwinUNETRModelSize`](/docs/reference/wiki/enums/swinunetrmodelsize/) | Defines the backbone size variants for Swin-UNETR medical segmentation. |
| [`SyntheticTabularTaskType`](/docs/reference/wiki/enums/synthetictabulartasktype/) | Defines the task type for the synthetic federated tabular benchmark suite. |
| [`T5Variant`](/docs/reference/wiki/enums/t5variant/) | Variants for the T5 text encoder used in SD3 / FLUX / Imagen pipelines. |
| [`TakagiAlgorithmType`](/docs/reference/wiki/enums/takagialgorithmtype/) | Represents different algorithm types for Takagi factorization of complex symmetric matrices. |
| [`TargetPlatform`](/docs/reference/wiki/enums/targetplatform/) | Target hardware platforms for model deployment and optimization. |
| [`TeacherModelType`](/docs/reference/wiki/enums/teachermodeltype/) | Specifies the type of teacher model to use for knowledge distillation. |
| [`TemporalAggregationType`](/docs/reference/wiki/enums/temporalaggregationtype/) | Specifies how frame-level features are aggregated into a single video-level representation. |
| [`TestStatisticType`](/docs/reference/wiki/enums/teststatistictype/) | Represents different types of statistical tests used to evaluate hypotheses and determine significance in data analysis. |
| [`TimeSeriesFoundationModelTask`](/docs/reference/wiki/enums/timeseriesfoundationmodeltask/) | Defines the tasks that a time series foundation model can perform. |
| [`TimeSeriesModelType`](/docs/reference/wiki/enums/timeseriesmodeltype/) | Represents different types of time series forecasting models used for analyzing and predicting sequential data over time. |
| [`TimeSeriesTokenizationStrategy`](/docs/reference/wiki/enums/timeseriestokenizationstrategy/) | Tokenization strategies for time series foundation models. |
| [`TrainingStageType`](/docs/reference/wiki/enums/trainingstagetype/) | Types of training stages in a multi-stage training pipeline. |
| [`TransUNetModelSize`](/docs/reference/wiki/enums/transunetmodelsize/) | Defines the backbone size variants for TransUNet medical segmentation. |
| [`TransformerTaskType`](/docs/reference/wiki/enums/transformertasktype/) | Defines the different types of tasks that transformer-based AI models can perform. |
| [`TreeSearchStrategy`](/docs/reference/wiki/enums/treesearchstrategy/) | Tree search strategies for exploring the reasoning space in Tree-of-Thoughts. |
| [`TridiagonalAlgorithmType`](/docs/reference/wiki/enums/tridiagonalalgorithmtype/) | Represents different algorithm types for converting a matrix to tridiagonal form. |
| [`UNINEXTModelSize`](/docs/reference/wiki/enums/uninextmodelsize/) | Defines the backbone size variants for UNINEXT. |
| [`UduAlgorithmType`](/docs/reference/wiki/enums/udualgorithmtype/) | Represents different algorithm types for UDU' decomposition of matrices. |
| [`UncertaintyQuantificationMethod`](/docs/reference/wiki/enums/uncertaintyquantificationmethod/) | Defines the supported uncertainty quantification strategies for inference. |
| [`UniVSModelSize`](/docs/reference/wiki/enums/univsmodelsize/) | Defines the backbone size variants for UniVS (Universal Video Segmentation). |
| [`UnivariateScoringFunction`](/docs/reference/wiki/enums/univariatescoringfunction/) | Defines the scoring functions available for univariate feature selection. |
| [`VGGVariant`](/docs/reference/wiki/enums/vggvariant/) | Defines the available VGG network architecture variants. |
| [`VMambaModelSize`](/docs/reference/wiki/enums/vmambamodelsize/) | Defines the backbone size variants for VMamba visual state space model. |
| [`ViTAdapterModelSize`](/docs/reference/wiki/enums/vitadaptermodelsize/) | Defines the model size variants for ViT-Adapter. |
| [`ViTCoMerModelSize`](/docs/reference/wiki/enums/vitcomermodelsize/) | Defines the model size variants for ViT-CoMer (hybrid CNN-Transformer). |
| [`VideoModelVariant`](/docs/reference/wiki/enums/videomodelvariant/) | Defines common model size variants for video processing models (SR, interpolation, flow, etc.). |
| [`VisionMambaModelSize`](/docs/reference/wiki/enums/visionmambamodelsize/) | Defines the backbone size variants for Vision Mamba (Vim) segmentation. |
| [`WaveletAlgorithmType`](/docs/reference/wiki/enums/waveletalgorithmtype/) | Represents different types of wavelet transform algorithms for signal processing. |
| [`WaveletType`](/docs/reference/wiki/enums/wavelettype/) | Defines the different types of biorthogonal wavelets that can be used for signal processing and analysis. |
| [`WeightFunction`](/docs/reference/wiki/enums/weightfunction/) | Defines different weight functions used in robust statistical methods and machine learning algorithms. |
| [`WindowFunctionType`](/docs/reference/wiki/enums/windowfunctiontype/) | Defines different window functions used in signal processing and data analysis. |
| [`Word2VecType`](/docs/reference/wiki/enums/word2vectype/) | Specifies the architecture type for Word2Vec models. |
| [`X11AlgorithmType`](/docs/reference/wiki/enums/x11algorithmtype/) | Represents different variants of the X-11 seasonal adjustment algorithm used in time series analysis. |
| [`XDecoderModelSize`](/docs/reference/wiki/enums/xdecodermodelsize/) | Defines the backbone size variants for X-Decoder. |
| [`YOLO11SegModelSize`](/docs/reference/wiki/enums/yolo11segmodelsize/) | Defines the model size variants for YOLO11-Seg instance segmentation. |
| [`YOLO26SegModelSize`](/docs/reference/wiki/enums/yolo26segmodelsize/) | Defines the model size variants for YOLO26-Seg instance segmentation. |
| [`YOLOv12SegModelSize`](/docs/reference/wiki/enums/yolov12segmodelsize/) | Defines the model size variants for YOLOv12-Seg instance segmentation. |
| [`YOLOv8SegModelSize`](/docs/reference/wiki/enums/yolov8segmodelsize/) | Defines the size variants for YOLOv8-Seg instance segmentation models. |
| [`YOLOv9SegModelSize`](/docs/reference/wiki/enums/yolov9segmodelsize/) | Defines the model size variants for YOLOv9-Seg instance segmentation. |

## Helpers & Utilities (1)

| Type | Summary |
|:-----|:--------|
| [`ClusteringMetricTypeExtensions`](/docs/reference/wiki/enums/clusteringmetrictypeextensions/) | Extension methods for `ClusteringMetricType`. |

## Attributes (1)

| Type | Summary |
|:-----|:--------|
| [`ClusteringMetricInfoAttribute`](/docs/reference/wiki/enums/clusteringmetricinfoattribute/) | Attribute to annotate clustering metric enum values with their optimization direction. |


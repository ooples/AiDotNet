---
title: "Models"
description: "All 659 public types in the AiDotNet.models namespace, organized by kind."
section: "API Reference"
---

**659** public types in this namespace, organized by kind.

## Models & Types (104)

| Type | Summary |
|:-----|:--------|
| [`AiDotNetLicenseKey`](/docs/reference/wiki/models/aidotnetlicensekey/) | Represents a license key for AiDotNet model encryption and online validation. |
| [`AiModelResult<T, TInput, TOutput>`](/docs/reference/wiki/models/aimodelresult/) | Partial class containing Test-Time Augmentation (TTA) prediction methods. |
| [`AlignmentEvaluationData<T>`](/docs/reference/wiki/models/alignmentevaluationdata/) | Contains test cases for evaluating AI alignment. |
| [`AlignmentFeedbackData<T>`](/docs/reference/wiki/models/alignmentfeedbackdata/) | Contains human feedback data for AI alignment. |
| [`AlignmentMetrics<T>`](/docs/reference/wiki/models/alignmentmetrics/) | Contains metrics for evaluating AI alignment with human values. |
| [`AutoMLRunSummary`](/docs/reference/wiki/models/automlrunsummary/) | Represents a redacted (safe-to-share) summary of an AutoML run. |
| [`AutoMLTrialSummary`](/docs/reference/wiki/models/automltrialsummary/) | Represents a redacted (safe-to-share) summary of a single AutoML trial. |
| [`BootstrapResult<T>`](/docs/reference/wiki/models/bootstrapresult/) | Represents the results of bootstrap validation for a machine learning model, containing R² metrics for training, validation, and test datasets. |
| [`CategoricalColumnStats`](/docs/reference/wiki/models/categoricalcolumnstats/) | Statistics for a categorical column. |
| [`CategoricalDistribution`](/docs/reference/wiki/models/categoricaldistribution/) | Represents a categorical (discrete choice) parameter distribution. |
| [`CertifiedAccuracyMetrics<T>`](/docs/reference/wiki/models/certifiedaccuracymetrics/) | Contains metrics for certified accuracy evaluation. |
| [`CertifiedPrediction<T>`](/docs/reference/wiki/models/certifiedprediction/) | Represents a certified prediction with robustness guarantees. |
| [`CheckpointMetadata<T>`](/docs/reference/wiki/models/checkpointmetadata/) | Contains metadata about a checkpoint without loading the full checkpoint data. |
| [`Checkpoint<T, TInput, TOutput>`](/docs/reference/wiki/models/checkpoint/) | Represents a saved checkpoint of model training state. |
| [`ChiSquareTestResult<T>`](/docs/reference/wiki/models/chisquaretestresult/) | Represents the results of a Chi-Square statistical test, which is used to determine whether there is a significant  association between two categorical variables. |
| [`ClassificationConformalPredictionSet`](/docs/reference/wiki/models/classificationconformalpredictionset/) | Represents a conformal prediction set for classification tasks. |
| [`ClientSelectionRequest`](/docs/reference/wiki/models/clientselectionrequest/) | Represents a request to select participating clients for a federated learning round. |
| [`ClusteringMetrics<T>`](/docs/reference/wiki/models/clusteringmetrics/) | Represents clustering quality metrics for evaluating the performance of clustering algorithms. |
| [`ContinuousDistribution`](/docs/reference/wiki/models/continuousdistribution/) | Represents a continuous (real-valued) parameter distribution. |
| [`CrossValidationResult<T, TInput, TOutput>`](/docs/reference/wiki/models/crossvalidationresult/) | Aggregates results from all folds in a cross-validation procedure. |
| [`DataSetStats<T, TInput, TOutput>`](/docs/reference/wiki/models/datasetstats/) |  |
| [`DatasetComparison<T>`](/docs/reference/wiki/models/datasetcomparison/) | Comparison results between two dataset versions. |
| [`DatasetLineage`](/docs/reference/wiki/models/datasetlineage/) | Lineage information for a dataset showing its origin and transformations. |
| [`DatasetResult<T, TInput, TOutput>`](/docs/reference/wiki/models/datasetresult/) | Represents detailed results and statistics for a specific dataset (training, validation, or test). |
| [`DatasetSnapshot`](/docs/reference/wiki/models/datasetsnapshot/) | A snapshot of a dataset at a point in time. |
| [`DatasetStatistics<T>`](/docs/reference/wiki/models/datasetstatistics/) | Statistical summary of a dataset. |
| [`DatasetVersionInfo<T>`](/docs/reference/wiki/models/datasetversioninfo/) | Information about a dataset version. |
| [`DatasetVersion<T>`](/docs/reference/wiki/models/datasetversion/) | Represents a versioned dataset with integrity verification. |
| [`DistributionFitResult<T>`](/docs/reference/wiki/models/distributionfitresult/) | Represents the result of fitting a statistical distribution to a dataset, including the distribution type, goodness of fit measure, and estimated parameters. |
| [`DynamicShapeInfo`](/docs/reference/wiki/models/dynamicshapeinfo/) | Describes which dimensions of a model's input/output shapes are dynamic (variable at runtime). |
| [`EpochHistory`](/docs/reference/wiki/models/epochhistory/) | Represents the training history for a single epoch or iteration. |
| [`Experiment`](/docs/reference/wiki/models/experiment/) | Represents a machine learning experiment that groups related training runs. |
| [`ExperimentInfo<T>`](/docs/reference/wiki/models/experimentinfo/) | Contains structured experiment tracking information from a trained model. |
| [`ExperimentRun<T>`](/docs/reference/wiki/models/experimentrun/) | Represents a single training run within an experiment. |
| [`FTestResult<T>`](/docs/reference/wiki/models/ftestresult/) | Represents the results of an F-test, which is used to compare the variances of two populations. |
| [`FederatedClientDataset<TInput, TOutput>`](/docs/reference/wiki/models/federatedclientdataset/) | Represents a single client's local dataset for federated learning. |
| [`FederatedLearningMetadata`](/docs/reference/wiki/models/federatedlearningmetadata/) | Contains metadata and metrics about federated learning training progress and results. |
| [`FilterAction`](/docs/reference/wiki/models/filteraction/) | Represents a filtering action taken on the output. |
| [`FineTuningData<T, TInput, TOutput>`](/docs/reference/wiki/models/finetuningdata/) | Container for fine-tuning training and evaluation data. |
| [`FineTuningMetrics<T>`](/docs/reference/wiki/models/finetuningmetrics/) | Metrics for evaluating fine-tuning quality. |
| [`FitDetectorResult<T>`](/docs/reference/wiki/models/fitdetectorresult/) | Represents the result of a model fit detection analysis, which evaluates how well a model fits the data and provides recommendations for improvement. |
| [`FoldResult<T, TInput, TOutput>`](/docs/reference/wiki/models/foldresult/) | Represents the results of a single fold in cross-validation. |
| [`GradientModel<T>`](/docs/reference/wiki/models/gradientmodel/) | Default implementation of a gradient model. |
| [`HarmfulContentFinding`](/docs/reference/wiki/models/harmfulcontentfinding/) | Represents a specific harmful content finding. |
| [`HarmfulContentResult<T>`](/docs/reference/wiki/models/harmfulcontentresult/) | Result of harmful content identification. |
| [`HyperparameterApplicationResult`](/docs/reference/wiki/models/hyperparameterapplicationresult/) | Contains the results of applying agent-recommended hyperparameters to a model's options. |
| [`HyperparameterOptimizationResult<T>`](/docs/reference/wiki/models/hyperparameteroptimizationresult/) | Contains the results of a hyperparameter optimization process. |
| [`HyperparameterSearchSpace`](/docs/reference/wiki/models/hyperparametersearchspace/) | Defines the search space for hyperparameter optimization. |
| [`HyperparameterTrial<T>`](/docs/reference/wiki/models/hyperparametertrial/) | Represents a single trial in hyperparameter optimization. |
| [`InferenceSequence<T, TInput, TOutput>`](/docs/reference/wiki/models/inferencesequence/) | Represents one independent, stateful inference sequence (e.g., one chat/generation stream). |
| [`InferenceSession<T, TInput, TOutput>`](/docs/reference/wiki/models/inferencesession/) | Facade-friendly inference session that owns stateful inference internals. |
| [`IntegerDistribution`](/docs/reference/wiki/models/integerdistribution/) | Represents an integer parameter distribution. |
| [`InterventionEffect<T>`](/docs/reference/wiki/models/interventioneffect/) | Represents the effect of an intervention in a time series or sequential data, capturing the starting point, duration, and magnitude of the effect. |
| [`InterventionInfo`](/docs/reference/wiki/models/interventioninfo/) | Represents information about an intervention in a time series or sequential data, specifying when it started and how long it lasted. |
| [`JailbreakDetectionResult<T>`](/docs/reference/wiki/models/jailbreakdetectionresult/) | Result of jailbreak attempt detection. |
| [`JailbreakIndicator`](/docs/reference/wiki/models/jailbreakindicator/) | Represents a specific jailbreak indicator. |
| [`KnowledgeGraphResult<T>`](/docs/reference/wiki/models/knowledgegraphresult/) | Contains the results of knowledge graph processing, including trained embeddings, community structure, and link prediction evaluation. |
| [`LayerQuantizationInfo`](/docs/reference/wiki/models/layerquantizationinfo/) | Contains quantization information for a specific layer. |
| [`LicenseValidationResult`](/docs/reference/wiki/models/licensevalidationresult/) | Contains the result of a license key validation attempt. |
| [`MannWhitneyUTestResult<T>`](/docs/reference/wiki/models/mannwhitneyutestresult/) | Represents the results of a Mann-Whitney U test, which is a non-parametric statistical test used to determine  whether two independent samples come from the same distribution. |
| [`MetaAdaptationResult<T>`](/docs/reference/wiki/models/metaadaptationresult/) | Results from adapting a meta-learner to a single task. |
| [`MetaEvaluationResult<T>`](/docs/reference/wiki/models/metaevaluationresult/) | Results from evaluating a meta-learner across multiple tasks. |
| [`MetaTrainingResult<T>`](/docs/reference/wiki/models/metatrainingresult/) | Results from a complete meta-training run with history tracking. |
| [`MetaTrainingStepResult<T>`](/docs/reference/wiki/models/metatrainingstepresult/) | Results from a single meta-training step (one outer loop update). |
| [`ModelComparison<T>`](/docs/reference/wiki/models/modelcomparison/) | Comparison results between two model versions. |
| [`ModelEvaluationData<T, TInput, TOutput>`](/docs/reference/wiki/models/modelevaluationdata/) | Represents a comprehensive collection of evaluation data for a model across training, validation, and test datasets. |
| [`ModelEvaluationInput<T, TInput, TOutput>`](/docs/reference/wiki/models/modelevaluationinput/) | Represents the input data required for evaluating a machine learning model. |
| [`ModelLineage`](/docs/reference/wiki/models/modellineage/) | Lineage information for a model showing its origin and dependencies. |
| [`ModelMetadata<T>`](/docs/reference/wiki/models/modelmetadata/) | Represents metadata about a machine learning model, including its type, complexity, and additional descriptive information. |
| [`ModelRegistryInfo<T, TInput, TOutput>`](/docs/reference/wiki/models/modelregistryinfo/) | Contains structured model registry information from a trained model. |
| [`ModelSearchCriteria<T>`](/docs/reference/wiki/models/modelsearchcriteria/) | Criteria for searching models in the registry. |
| [`ModelStatsInputs<T, TInput, TOutput>`](/docs/reference/wiki/models/modelstatsinputs/) | Represents a container for inputs needed to calculate various statistics and metrics for a model. |
| [`ModelVersionInfo<T>`](/docs/reference/wiki/models/modelversioninfo/) | Information about a specific model version. |
| [`NASResultSummary`](/docs/reference/wiki/models/nasresultsummary/) | Represents a redacted summary of Neural Architecture Search (NAS) results. |
| [`NormalizationParameters<T>`](/docs/reference/wiki/models/normalizationparameters/) | Represents the parameters used for normalizing a single feature or target variable in a machine learning model. |
| [`NumericColumnStats<T>`](/docs/reference/wiki/models/numericcolumnstats/) | Statistics for a numeric column. |
| [`OFASubnetSummary`](/docs/reference/wiki/models/ofasubnetsummary/) | Represents OnceForAll subnet configuration summary. |
| [`OptimizationInputData<T, TInput, TOutput>`](/docs/reference/wiki/models/optimizationinputdata/) | Represents the input data for optimization processes, including training, validation, and test datasets. |
| [`OptimizationIterationInfo<T>`](/docs/reference/wiki/models/optimizationiterationinfo/) | Represents information about a single iteration in an optimization process, including fitness and overfitting detection results. |
| [`OptimizationResult<T, TInput, TOutput>`](/docs/reference/wiki/models/optimizationresult/) | Represents the comprehensive results of an optimization process for a symbolic model, including the best solution found, performance metrics, feature selection results, and detailed statistics for different datasets. |
| [`OptimizationStepData<T, TInput, TOutput>`](/docs/reference/wiki/models/optimizationstepdata/) |  |
| [`ParameterIndexRange`](/docs/reference/wiki/models/parameterindexrange/) | Represents a contiguous parameter index range. |
| [`PermutationTestResult<T>`](/docs/reference/wiki/models/permutationtestresult/) | Represents the results of a permutation test, which is a non-parametric statistical significance test that determines whether the observed difference between two groups is statistically significant. |
| [`QuantizationInfo`](/docs/reference/wiki/models/quantizationinfo/) | Contains information about model quantization applied during or after training. |
| [`RedTeamingResults<T>`](/docs/reference/wiki/models/redteamingresults/) | Contains results from red teaming adversarial testing. |
| [`RegisteredModel<T, TInput, TOutput>`](/docs/reference/wiki/models/registeredmodel/) | Represents a registered model in the model registry with its metadata and versioning information. |
| [`RegressionConformalInterval<TOutput>`](/docs/reference/wiki/models/regressionconformalinterval/) | Represents a conformal prediction interval for regression-style outputs. |
| [`ResourceUsageStats`](/docs/reference/wiki/models/resourceusagestats/) | Contains statistics about system resource usage during training. |
| [`RobustnessMetrics<T>`](/docs/reference/wiki/models/robustnessmetrics/) | Contains metrics for evaluating adversarial robustness of models. |
| [`RobustnessStats<T>`](/docs/reference/wiki/models/robustnessstats/) | Represents adversarial robustness diagnostics aggregated over a dataset. |
| [`RoundMetadata`](/docs/reference/wiki/models/roundmetadata/) | Contains detailed metrics for a single federated learning round. |
| [`SafetyFilterResult<T>`](/docs/reference/wiki/models/safetyfilterresult/) | Result of safety filtering on model output. |
| [`SafetyValidationResult<T>`](/docs/reference/wiki/models/safetyvalidationresult/) | Result of safety validation for an input. |
| [`StageCallbacks<T, TInput, TOutput>`](/docs/reference/wiki/models/stagecallbacks/) | Callbacks for training stage events. |
| [`TTestResult<T>`](/docs/reference/wiki/models/ttestresult/) | Represents the results of a t-test, which is a statistical hypothesis test used to determine if there is a significant  difference between the means of two groups. |
| [`TrainingSpeedStats`](/docs/reference/wiki/models/trainingspeedstats/) | Contains statistics about training speed and progress. |
| [`TrainingStageResult<T, TInput, TOutput>`](/docs/reference/wiki/models/trainingstageresult/) | Result of executing a training stage. |
| [`TrainingStage<T, TInput, TOutput>`](/docs/reference/wiki/models/trainingstage/) | Represents a single stage in a training pipeline with comprehensive configuration options. |
| [`UncertaintyCalibrationData<TInput, TOutput>`](/docs/reference/wiki/models/uncertaintycalibrationdata/) | Provides optional calibration data for uncertainty quantification features. |
| [`UncertaintyPredictionResult<T, TOutput>`](/docs/reference/wiki/models/uncertaintypredictionresult/) | Represents a prediction result augmented with uncertainty information. |
| [`UncertaintyStats<T>`](/docs/reference/wiki/models/uncertaintystats/) | Represents uncertainty-quantification diagnostics aggregated over a dataset. |
| [`ValidationIssue`](/docs/reference/wiki/models/validationissue/) | Represents a specific validation issue. |
| [`VectorModel<T>`](/docs/reference/wiki/models/vectormodel/) | Represents a linear model that uses a vector of coefficients to make predictions. |
| [`VulnerabilityReport`](/docs/reference/wiki/models/vulnerabilityreport/) | Detailed report of a specific vulnerability found during red teaming. |

## Base Classes (3)

| Type | Summary |
|:-----|:--------|
| [`ModelBase<T, TInput, TOutput>`](/docs/reference/wiki/models/modelbase/) | Abstract base class for standalone models that directly implement `IFullModel`. |
| [`ModelWrapperBase<T, TInput, TOutput>`](/docs/reference/wiki/models/modelwrapperbase/) | Abstract base class for model wrappers that delegate to an underlying `IFullModel`. |
| [`ParameterDistribution`](/docs/reference/wiki/models/parameterdistribution/) | Base class for parameter distributions. |

## Enums (95)

| Type | Summary |
|:-----|:--------|
| [`ActiveLearningStrategyType`](/docs/reference/wiki/models/activelearningstrategytype/) | Specifies the active learning strategy to use for sample selection. |
| [`AdaBoostAlgorithm`](/docs/reference/wiki/models/adaboostalgorithm/) | AdaBoost algorithm variants. |
| [`AdvancedCompressionStrategy`](/docs/reference/wiki/models/advancedcompressionstrategy/) | Advanced gradient compression strategies for federated learning communication efficiency. |
| [`AnomalyFeatures`](/docs/reference/wiki/models/anomalyfeatures/) | Flags for selecting which anomaly detection features to calculate. |
| [`AttestationPolicy`](/docs/reference/wiki/models/attestationpolicy/) | Specifies the attestation policy for TEE remote attestation verification. |
| [`BackboneType`](/docs/reference/wiki/models/backbonetype/) | Backbone network types for feature extraction. |
| [`BackdoorDetectionStrategy`](/docs/reference/wiki/models/backdoordetectionstrategy/) | Specifies the backdoor detection strategy for federated learning. |
| [`BetaLinkFunction`](/docs/reference/wiki/models/betalinkfunction/) | Link functions for Beta Regression mean model. |
| [`BlockSelectionMode`](/docs/reference/wiki/models/blockselectionmode/) | Block selection mode for DFedBCA protocol. |
| [`ClassificationSplitCriterion`](/docs/reference/wiki/models/classificationsplitcriterion/) | Criterion used to measure the quality of a split in classification decision trees. |
| [`ContinualLearningStrategyType`](/docs/reference/wiki/models/continuallearningstrategytype/) | Specifies the continual learning strategy to use for preventing catastrophic forgetting. |
| [`ContributionMethod`](/docs/reference/wiki/models/contributionmethod/) | Specifies the method used to evaluate client contributions in federated learning. |
| [`DARTDropoutMode`](/docs/reference/wiki/models/dartdropoutmode/) | Dropout modes for DART. |
| [`DARTDropoutType`](/docs/reference/wiki/models/dartdropouttype/) | Types of dropout selection for DART. |
| [`DARTNormalization`](/docs/reference/wiki/models/dartnormalization/) | Normalization strategies for DART. |
| [`DARTNormalizationType`](/docs/reference/wiki/models/dartnormalizationtype/) | Types of normalization after dropout in DART. |
| [`DecentralizedTopologyType`](/docs/reference/wiki/models/decentralizedtopologytype/) | Specifies the decentralized topology for peer-to-peer federated learning. |
| [`DetectionArchitecture`](/docs/reference/wiki/models/detectionarchitecture/) | Detection model architectures. |
| [`DifferencingFeatures`](/docs/reference/wiki/models/differencingfeatures/) | Flags for selecting which differencing and stationarity features to compute. |
| [`DifferentialPrivacyMode`](/docs/reference/wiki/models/differentialprivacymode/) | Specifies where differential privacy noise is applied in the federated learning pipeline. |
| [`DistanceMetric`](/docs/reference/wiki/models/distancemetric/) | Distance metrics for measuring similarity between samples. |
| [`EdgeHandling`](/docs/reference/wiki/models/edgehandling/) | How to handle edge cases where the full window is not available. |
| [`FairnessConstraintType`](/docs/reference/wiki/models/fairnessconstrainttype/) | Specifies the type of fairness constraint to enforce during federated learning. |
| [`FederatedAdapterType`](/docs/reference/wiki/models/federatedadaptertype/) | Specifies the federated adapter type for parameter-efficient fine-tuning. |
| [`FederatedAggregationStrategy`](/docs/reference/wiki/models/federatedaggregationstrategy/) | Specifies which federated aggregation strategy to use. |
| [`FederatedAsyncMode`](/docs/reference/wiki/models/federatedasyncmode/) | Specifies the asynchronous federated learning mode. |
| [`FederatedClientSelectionStrategy`](/docs/reference/wiki/models/federatedclientselectionstrategy/) | Specifies how clients are selected to participate in a federated training round. |
| [`FederatedCompressionStrategy`](/docs/reference/wiki/models/federatedcompressionstrategy/) | Specifies the compression strategy for federated model updates. |
| [`FederatedContinualLearningStrategy`](/docs/reference/wiki/models/federatedcontinuallearningstrategy/) | Specifies the federated continual learning strategy for preventing catastrophic forgetting across rounds. |
| [`FederatedDistillationStrategy`](/docs/reference/wiki/models/federateddistillationstrategy/) | Specifies the federated knowledge distillation strategy. |
| [`FederatedDriftMethod`](/docs/reference/wiki/models/federateddriftmethod/) | Specifies the drift detection method used in federated learning. |
| [`FederatedHeterogeneityCorrection`](/docs/reference/wiki/models/federatedheterogeneitycorrection/) | Specifies which heterogeneity correction algorithm to use. |
| [`FederatedLearningMode`](/docs/reference/wiki/models/federatedlearningmode/) | Specifies the federated learning paradigm to use. |
| [`FederatedMetaLearningStrategy`](/docs/reference/wiki/models/federatedmetalearningstrategy/) | Specifies the federated meta-learning strategy. |
| [`FederatedPersonalizationStrategy`](/docs/reference/wiki/models/federatedpersonalizationstrategy/) | Specifies the personalization strategy for federated learning. |
| [`FederatedPrivacyAccountant`](/docs/reference/wiki/models/federatedprivacyaccountant/) | Specifies which privacy accountant to use for reporting privacy spend in federated learning. |
| [`FederatedServerOptimizer`](/docs/reference/wiki/models/federatedserveroptimizer/) | Specifies which server-side federated optimizer (FedOpt family) to use. |
| [`FederatedStalenessWeighting`](/docs/reference/wiki/models/federatedstalenessweighting/) | Specifies how to down-weight stale updates in asynchronous federated learning. |
| [`FrequencyAttentionType`](/docs/reference/wiki/models/frequencyattentiontype/) | Specifies the type of frequency attention to use in FEDformer. |
| [`FuzzyMatchStrategy`](/docs/reference/wiki/models/fuzzymatchstrategy/) | Specifies the similarity strategy used for fuzzy entity matching in PSI. |
| [`GAMLSSDistributionFamily`](/docs/reference/wiki/models/gamlssdistributionfamily/) | Distribution families supported by GAMLSS. |
| [`GAMLSSModelType`](/docs/reference/wiki/models/gamlssmodeltype/) | Types of sub-models available for modeling distribution parameters. |
| [`GLMMEstimationMethod`](/docs/reference/wiki/models/glmmestimationmethod/) | Estimation methods for GLMM. |
| [`GLMMFamily`](/docs/reference/wiki/models/glmmfamily/) | Response distribution families for GLMM. |
| [`GLMMLinkFunction`](/docs/reference/wiki/models/glmmlinkfunction/) | Link functions for GLMM. |
| [`GammaLinkFunction`](/docs/reference/wiki/models/gammalinkfunction/) | Link functions for Gamma regression. |
| [`GradientBoostingLoss`](/docs/reference/wiki/models/gradientboostingloss/) | Loss functions for Gradient Boosting classifier. |
| [`GradientClippingMethod`](/docs/reference/wiki/models/gradientclippingmethod/) | Specifies the method used for gradient clipping. |
| [`GraphFLMode`](/docs/reference/wiki/models/graphflmode/) | Specifies the federated graph learning task type. |
| [`GraphPartitionStrategy`](/docs/reference/wiki/models/graphpartitionstrategy/) | Specifies how a graph is partitioned across federated clients. |
| [`HomomorphicEncryptionMode`](/docs/reference/wiki/models/homomorphicencryptionmode/) | Specifies how homomorphic encryption is applied during federated aggregation. |
| [`HomomorphicEncryptionScheme`](/docs/reference/wiki/models/homomorphicencryptionscheme/) | Specifies which homomorphic encryption scheme to use. |
| [`InverseGaussianLinkFunction`](/docs/reference/wiki/models/inversegaussianlinkfunction/) | Link functions for Inverse Gaussian regression. |
| [`KNNAlgorithm`](/docs/reference/wiki/models/knnalgorithm/) | Algorithms for finding nearest neighbors. |
| [`LinearLoss`](/docs/reference/wiki/models/linearloss/) | Loss functions for linear classifiers. |
| [`LinearPenalty`](/docs/reference/wiki/models/linearpenalty/) | Penalty types for linear classifiers. |
| [`MissingFeatureStrategy`](/docs/reference/wiki/models/missingfeaturestrategy/) | Specifies how to handle missing feature blocks when not all parties have data for all entities. |
| [`MixedEffectsCovarianceStructure`](/docs/reference/wiki/models/mixedeffectscovariancestructure/) | Covariance structures for random effects. |
| [`MixedEffectsEstimationMethod`](/docs/reference/wiki/models/mixedeffectsestimationmethod/) | Estimation methods for mixed-effects models. |
| [`MixedEffectsOptimization`](/docs/reference/wiki/models/mixedeffectsoptimization/) | Optimization methods for mixed-effects models. |
| [`MixedEffectsOptimizer`](/docs/reference/wiki/models/mixedeffectsoptimizer/) | Optimization algorithms for mixed-effects model fitting. |
| [`ModelSize`](/docs/reference/wiki/models/modelsize/) | Model size variants. |
| [`MpcProtocol`](/docs/reference/wiki/models/mpcprotocol/) | Specifies the multi-party computation protocol to use. |
| [`MpcSecurityModel`](/docs/reference/wiki/models/mpcsecuritymodel/) | Specifies the adversary model assumed by the MPC protocol. |
| [`NGBoostDistributionType`](/docs/reference/wiki/models/ngboostdistributiontype/) | Types of distributions supported by NGBoost. |
| [`NGBoostScoringRuleType`](/docs/reference/wiki/models/ngboostscoringruletype/) | Types of scoring rules supported by NGBoost. |
| [`NHiTSInterpolationMode`](/docs/reference/wiki/models/nhitsinterpolationmode/) | Interpolation method for upsampling forecasts in N-HiTS stacks. |
| [`NHiTSPoolingMode`](/docs/reference/wiki/models/nhitspoolingmode/) | Pooling strategy for downsampling in N-HiTS stacks. |
| [`NeckType`](/docs/reference/wiki/models/necktype/) | Neck architecture types for multi-scale feature fusion. |
| [`NmsType`](/docs/reference/wiki/models/nmstype/) | NMS (Non-Maximum Suppression) algorithm variants. |
| [`OrdinalLinkFunction`](/docs/reference/wiki/models/ordinallinkfunction/) | Link functions for ordinal regression. |
| [`PseudoNodeStrategy`](/docs/reference/wiki/models/pseudonodestrategy/) | Specifies how to handle missing cross-client neighbor nodes in subgraph-level FL. |
| [`PsiProtocol`](/docs/reference/wiki/models/psiprotocol/) | Specifies the cryptographic protocol used for Private Set Intersection. |
| [`ReplayBufferStrategy`](/docs/reference/wiki/models/replaybufferstrategy/) | Specifies the buffer management strategy for Experience Replay. |
| [`RollingRegressionFeatures`](/docs/reference/wiki/models/rollingregressionfeatures/) | Flags for selecting which rolling regression features to calculate. |
| [`RollingStatistics`](/docs/reference/wiki/models/rollingstatistics/) | Flags for selecting which rolling statistics to calculate. |
| [`SeasonalityFeatures`](/docs/reference/wiki/models/seasonalityfeatures/) | Flags for selecting which seasonality and calendar features to generate. |
| [`SecureAggregationMode`](/docs/reference/wiki/models/secureaggregationmode/) | Determines which secure aggregation protocol variant is used. |
| [`SplitPointStrategy`](/docs/reference/wiki/models/splitpointstrategy/) | Specifies how to choose the split point in a split neural network for vertical FL. |
| [`SuperLearnerMetaLearner`](/docs/reference/wiki/models/superlearnermetalearner/) | Meta-learner types for Super Learner. |
| [`TechnicalIndicators`](/docs/reference/wiki/models/technicalindicators/) | Flags for selecting which technical indicators to calculate. |
| [`TeeProviderType`](/docs/reference/wiki/models/teeprovidertype/) | Specifies the Trusted Execution Environment hardware provider. |
| [`TrialStatus`](/docs/reference/wiki/models/trialstatus/) | Represents the status of a hyperparameter trial. |
| [`TweedieLinkFunction`](/docs/reference/wiki/models/tweedielinkfunction/) | Link functions for Tweedie regression. |
| [`UnlearningMethod`](/docs/reference/wiki/models/unlearningmethod/) | Specifies the federated unlearning method to use when a client requests data removal. |
| [`VerificationLevel`](/docs/reference/wiki/models/verificationlevel/) | Specifies the level of zero-knowledge verification applied to client updates. |
| [`VflAggregationMode`](/docs/reference/wiki/models/vflaggregationmode/) | Specifies how embeddings from multiple parties are combined in vertical federated learning. |
| [`VflUnlearningMethod`](/docs/reference/wiki/models/vflunlearningmethod/) | Specifies the method used to remove an entity's influence from a trained VFL model. |
| [`VolatilityMeasures`](/docs/reference/wiki/models/volatilitymeasures/) | Flags for selecting which volatility measures to calculate. |
| [`WeightingScheme`](/docs/reference/wiki/models/weightingscheme/) | Weighting schemes for neighbor voting. |
| [`WindowAutoDetectionMethod`](/docs/reference/wiki/models/windowautodetectionmethod/) | Methods for auto-detecting optimal window sizes. |
| [`ZeroInflatedCountLink`](/docs/reference/wiki/models/zeroinflatedcountlink/) | Link functions for the count component of zero-inflated models. |
| [`ZeroInflatedDistributionFamily`](/docs/reference/wiki/models/zeroinflateddistributionfamily/) | Base count distributions for zero-inflated models. |
| [`ZeroInflatedZeroLink`](/docs/reference/wiki/models/zeroinflatedzerolink/) | Link functions for the zero-inflation component. |
| [`ZkProofSystem`](/docs/reference/wiki/models/zkproofsystem/) | Specifies the zero-knowledge proof system to use for verifiable federated learning. |

## Structs (1)

| Type | Summary |
|:-----|:--------|
| [`ModelResult<T, TInput, TOutput>`](/docs/reference/wiki/models/modelresult/) | Represents the complete results of a model-building process, including the model solution, fitness metrics, fit detection results, evaluation data, and selected features. |

## Options & Configuration (456)

| Type | Summary |
|:-----|:--------|
| [`A2COptions<T>`](/docs/reference/wiki/models/a2coptions/) | Configuration options for Advantage Actor-Critic (A2C) agents. |
| [`A3COptions<T>`](/docs/reference/wiki/models/a3coptions/) | Configuration options for Asynchronous Advantage Actor-Critic (A3C) agents. |
| [`ADMMOptimizerOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/admmoptimizeroptions/) | Configuration options for the Alternating Direction Method of Multipliers (ADMM) optimization algorithm, which is particularly effective for problems with complex regularization requirements. |
| [`AIMOptions<T>`](/docs/reference/wiki/models/aimoptions/) | Configuration options for AIM (Adaptive Iterative Mechanism), a marginal-based differentially private synthetic data generation method. |
| [`AMSGradOptimizerOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/amsgradoptimizeroptions/) | Configuration options for the AMSGrad optimization algorithm, which is an improved variant of the Adam optimizer that addresses potential convergence issues by maintaining the maximum of past squared gradients. |
| [`ARIMAOptions<T>`](/docs/reference/wiki/models/arimaoptions/) | Configuration options for the ARIMA (AutoRegressive Integrated Moving Average) time series forecasting model. |
| [`ARIMAXModelOptions<T>`](/docs/reference/wiki/models/arimaxmodeloptions/) | Configuration options for the ARIMAX (AutoRegressive Integrated Moving Average with eXogenous variables) time series forecasting model. |
| [`ARMAOptions<T>`](/docs/reference/wiki/models/armaoptions/) | Configuration options for the ARMA (AutoRegressive Moving Average) time series forecasting model. |
| [`ARModelOptions<T>`](/docs/reference/wiki/models/armodeloptions/) | Configuration options for the AR (AutoRegressive) time series forecasting model. |
| [`ASTModelOptions`](/docs/reference/wiki/models/astmodeloptions/) | Configuration options for AST (Audio Spectrogram Transformer) models (Gong et al. |
| [`ActiveLearningOptions`](/docs/reference/wiki/models/activelearningoptions/) | Represents configuration options for active learning. |
| [`AdaBoostClassifierOptions<T>`](/docs/reference/wiki/models/adaboostclassifieroptions/) | Configuration options for AdaBoost classifier. |
| [`AdaBoostR2RegressionOptions`](/docs/reference/wiki/models/adaboostr2regressionoptions/) | Configuration options for the AdaBoost R2 regression algorithm. |
| [`AdaDeltaOptimizerOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/adadeltaoptimizeroptions/) | Configuration options for the AdaDelta optimization algorithm, which is an extension of AdaGrad that adapts learning rates based on a moving window of gradient updates. |
| [`AdaMaxOptimizerOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/adamaxoptimizeroptions/) | Configuration options for the AdaMax optimization algorithm, a variant of Adam that uses the infinity norm. |
| [`AdagradOptimizerOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/adagradoptimizeroptions/) | Configuration options for the Adagrad optimization algorithm, which adapts the learning rate for each parameter based on historical gradient information. |
| [`Adam8BitOptimizerOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/adam8bitoptimizeroptions/) | Configuration options for the 8-bit Adam optimization algorithm, which reduces memory usage by quantizing optimizer states. |
| [`AdamOptimizerOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/adamoptimizeroptions/) | Configuration options for the Adam optimization algorithm, which combines the benefits of AdaGrad and RMSProp. |
| [`AdamWOptimizerOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/adamwoptimizeroptions/) | Configuration options for the AdamW optimization algorithm with decoupled weight decay. |
| [`AdaptiveFitDetectorOptions`](/docs/reference/wiki/models/adaptivefitdetectoroptions/) | Configuration options for the Adaptive Fit Detector, which automatically selects the most appropriate method to detect overfitting and underfitting in machine learning models. |
| [`AdaptiveRandomForestOptions<T>`](/docs/reference/wiki/models/adaptiverandomforestoptions/) | Configuration options for Adaptive Random Forest classifier. |
| [`AdvancedCompressionOptions`](/docs/reference/wiki/models/advancedcompressionoptions/) | Configuration for advanced gradient compression methods (PowerSGD, sketching, adaptive). |
| [`AdversarialAttackOptions<T>`](/docs/reference/wiki/models/adversarialattackoptions/) | Configuration options for adversarial attack algorithms. |
| [`AdversarialDefenseOptions<T>`](/docs/reference/wiki/models/adversarialdefenseoptions/) | Configuration options for adversarial defense mechanisms. |
| [`AdversarialRobustnessConfiguration<T>`](/docs/reference/wiki/models/adversarialrobustnessconfiguration/) | Non-generic version for backward compatibility and simpler use cases. |
| [`AdversarialRobustnessConfiguration<T, TInput, TOutput>`](/docs/reference/wiki/models/adversarialrobustnessconfiguration-2/) | Configuration for adversarial robustness and AI safety during model building and inference. |
| [`AdversarialRobustnessOptions<T>`](/docs/reference/wiki/models/adversarialrobustnessoptions/) | Unified configuration options for adversarial robustness and AI safety. |
| [`AiModelResultOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/aimodelresultoptions/) | Represents the configuration options for creating a AiModelResult. |
| [`AlignmentMethodOptions<T>`](/docs/reference/wiki/models/alignmentmethodoptions/) | Configuration options for AI alignment methods. |
| [`AlphaFactorOptions<T>`](/docs/reference/wiki/models/alphafactoroptions/) | Configuration options for the AlphaFactorModel. |
| [`AntColonyOptimizationOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/antcolonyoptimizationoptions/) | Configuration options for the Ant Colony Optimization algorithm, which is inspired by the foraging behavior of ants to find optimal paths through a search space. |
| [`AsyncFederatedLearningOptions`](/docs/reference/wiki/models/asyncfederatedlearningoptions/) | Configuration options for asynchronous federated learning (FedAsync / FedBuff). |
| [`AttentionAllocationOptions<T>`](/docs/reference/wiki/models/attentionallocationoptions/) | Configuration options for Attention Allocation. |
| [`AudioNeuralNetworkOptions`](/docs/reference/wiki/models/audioneuralnetworkoptions/) | Base configuration options for audio neural network models. |
| [`AutoDiffTabOptions<T>`](/docs/reference/wiki/models/autodifftaboptions/) | Configuration options for AutoDiff-Tab, an automated diffusion model for tabular data that searches over diffusion configurations and noise schedules to find optimal settings. |
| [`AutoIntOptions<T>`](/docs/reference/wiki/models/autointoptions/) | Configuration options for AutoInt (Automatic Feature Interaction Learning). |
| [`AutocorrelationFitDetectorOptions`](/docs/reference/wiki/models/autocorrelationfitdetectoroptions/) | Configuration options for detecting autocorrelation in time series data and regression residuals. |
| [`AutoformerOptions<T>`](/docs/reference/wiki/models/autoformeroptions/) | Configuration options for the Autoformer model (Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting). |
| [`BFGSOptimizerOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/bfgsoptimizeroptions/) | Configuration options for the BFGS (Broyden-Fletcher-Goldfarb-Shanno) optimization algorithm. |
| [`BackdoorDefenseOptions`](/docs/reference/wiki/models/backdoordefenseoptions/) | Configuration options for backdoor attack detection and mitigation in federated learning. |
| [`BayesianFitDetectorOptions`](/docs/reference/wiki/models/bayesianfitdetectoroptions/) | Configuration options for the Bayesian model fit detector, which evaluates how well a model fits the data. |
| [`BayesianNetworkSynthOptions<T>`](/docs/reference/wiki/models/bayesiannetworksynthoptions/) | Configuration options for Bayesian Network Synthesis, a statistical approach that learns a directed acyclic graph (DAG) structure and conditional probability tables to generate synthetic tabular data via ancestral sampling. |
| [`BayesianOptimizerOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/bayesianoptimizeroptions/) | Configuration options for the Bayesian optimization algorithm. |
| [`BayesianRegressionOptions<T>`](/docs/reference/wiki/models/bayesianregressionoptions/) | Configuration options for Bayesian regression algorithms. |
| [`BayesianStructuralTimeSeriesOptions<T>`](/docs/reference/wiki/models/bayesianstructuraltimeseriesoptions/) | Configuration options for Bayesian Structural Time Series models. |
| [`BetaRegressionOptions`](/docs/reference/wiki/models/betaregressionoptions/) | Configuration options for Beta Regression models. |
| [`BlackLittermanNeuralOptions<T>`](/docs/reference/wiki/models/blacklittermanneuraloptions/) | Configuration options for Neural Black-Litterman. |
| [`BloombergGPTOptions<T>`](/docs/reference/wiki/models/bloomberggptoptions/) | Configuration options for BloombergGPT-style financial language model. |
| [`BootstrapFitDetectorOptions`](/docs/reference/wiki/models/bootstrapfitdetectoroptions/) | Configuration options for the Bootstrap Fit Detector, which evaluates model fit quality using bootstrap resampling. |
| [`CCDMOptions<T>`](/docs/reference/wiki/models/ccdmoptions/) | Configuration options for CCDM (Conditional Continuous Diffusion Model for Time Series). |
| [`CLAPModelOptions`](/docs/reference/wiki/models/clapmodeloptions/) | Configuration options for CLAP (Contrastive Language-Audio Pretraining) models. |
| [`CMAESOptimizerOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/cmaesoptimizeroptions/) | Configuration options for the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimization algorithm. |
| [`CQLOptions<T>`](/docs/reference/wiki/models/cqloptions/) | Configuration options for Conservative Q-Learning (CQL) agent. |
| [`CSDIOptions<T>`](/docs/reference/wiki/models/csdioptions/) | Configuration options for CSDI (Conditional Score-based Diffusion model for Imputation). |
| [`CTABGANPlusOptions<T>`](/docs/reference/wiki/models/ctabganplusoptions/) | Configuration options for CTAB-GAN+, an enhanced conditional tabular GAN with auxiliary classifier discriminator and mixed-type encoder for high-quality synthetic data generation. |
| [`CTGANOptions<T>`](/docs/reference/wiki/models/ctganoptions/) | Configuration options for CTGAN (Conditional Tabular GAN), a generative adversarial network specifically designed for generating realistic synthetic tabular data. |
| [`CalibratedClassifierOptions<T>`](/docs/reference/wiki/models/calibratedclassifieroptions/) | Configuration options for calibrated classifiers. |
| [`CalibratedProbabilityFitDetectorOptions`](/docs/reference/wiki/models/calibratedprobabilityfitdetectoroptions/) | Configuration options for the Calibrated Probability Fit Detector, which evaluates how well a model's  predicted probabilities match actual outcomes. |
| [`CausalDiscoveryOptions`](/docs/reference/wiki/models/causaldiscoveryoptions/) | Configuration options for causal structure discovery. |
| [`CausalGANOptions<T>`](/docs/reference/wiki/models/causalganoptions/) | Configuration options for Causal-GAN, a GAN that learns causal graph structure and generates data respecting causal relationships between features. |
| [`CertifiedDefenseOptions<T>`](/docs/reference/wiki/models/certifieddefenseoptions/) | Configuration options for certified defense mechanisms. |
| [`ChronosBoltOptions<T>`](/docs/reference/wiki/models/chronosboltoptions/) | Configuration options for Chronos-Bolt (Fast Non-Autoregressive Time Series Forecasting). |
| [`ChronosFinanceOptions<T>`](/docs/reference/wiki/models/chronosfinanceoptions/) | Configuration options for Chronos Finance (Amazon's time series foundation model for financial forecasting). |
| [`ClassifierOptions<T>`](/docs/reference/wiki/models/classifieroptions/) | Configuration options for classification models, which are machine learning methods used to predict categorical outcomes (discrete classes) rather than continuous values. |
| [`ClientSelectionOptions`](/docs/reference/wiki/models/clientselectionoptions/) | Configuration options for client selection in federated learning. |
| [`CommitmentOptions`](/docs/reference/wiki/models/commitmentoptions/) | Configuration options for cryptographic commitment schemes. |
| [`ConditionalInferenceTreeOptions`](/docs/reference/wiki/models/conditionalinferencetreeoptions/) | Configuration options for Conditional Inference Trees, a statistically-driven approach to decision tree learning. |
| [`ConfusionMatrixFitDetectorOptions`](/docs/reference/wiki/models/confusionmatrixfitdetectoroptions/) | Configuration options for the Confusion Matrix Fit Detector, which evaluates how well a classification model performs. |
| [`ConjugateGradientOptimizerOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/conjugategradientoptimizeroptions/) | Configuration options for the Conjugate Gradient optimization algorithm, which is used to train machine learning models. |
| [`ContinualLearningOptions`](/docs/reference/wiki/models/continuallearningoptions/) | Represents configuration options for continual learning. |
| [`ContributionEvaluationOptions`](/docs/reference/wiki/models/contributionevaluationoptions/) | Configuration options for client contribution evaluation in federated learning. |
| [`ConvTasNetOptions`](/docs/reference/wiki/models/convtasnetoptions/) | Configuration options for ConvTasNet audio source separation models. |
| [`CookDistanceFitDetectorOptions`](/docs/reference/wiki/models/cookdistancefitdetectoroptions/) | Configuration options for the Cook's Distance fit detector, which helps identify influential data points and detect potential overfitting or underfitting in regression models. |
| [`CoordinateDescentOptimizerOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/coordinatedescentoptimizeroptions/) | Configuration options for the Coordinate Descent optimization algorithm, which optimizes a function by solving for one variable at a time while holding others constant. |
| [`CopulaGANOptions<T>`](/docs/reference/wiki/models/copulaganoptions/) | Configuration options for CopulaGAN, a synthetic tabular data generator that combines Gaussian copula transformations with the CTGAN training pipeline. |
| [`CopulaSynthOptions<T>`](/docs/reference/wiki/models/copulasynthoptions/) | Configuration options for Copula-Based Synthesis, a statistical method that models the joint distribution of features by fitting marginal distributions individually and coupling them with a copula function. |
| [`CrossClientEdgeOptions`](/docs/reference/wiki/models/crossclientedgeoptions/) | Configuration for handling cross-client edges in federated graph learning. |
| [`CrossValidationFitDetectorOptions`](/docs/reference/wiki/models/crossvalidationfitdetectoroptions/) | Configuration options for detecting overfitting, underfitting, and good fitting in machine learning models using cross-validation techniques. |
| [`CrossValidationOptions`](/docs/reference/wiki/models/crossvalidationoptions/) | Represents the configuration options for cross-validation in machine learning models. |
| [`CrossformerOptions<T>`](/docs/reference/wiki/models/crossformeroptions/) | Configuration options for the Crossformer model. |
| [`DARTClassifierOptions<T>`](/docs/reference/wiki/models/dartclassifieroptions/) | Configuration options for DART (Dropouts meet Multiple Additive Regression Trees) classifier. |
| [`DARTOptions`](/docs/reference/wiki/models/dartoptions/) | Configuration options for DART (Dropouts meet Multiple Additive Regression Trees). |
| [`DCCRNOptions`](/docs/reference/wiki/models/dccrnoptions/) | Configuration options for DCCRN (Deep Complex Convolution Recurrent Network) audio enhancement models. |
| [`DCRNNOptions<T>`](/docs/reference/wiki/models/dcrnnoptions/) | Configuration options for DCRNN (Diffusion Convolutional Recurrent Neural Network). |
| [`DDPGOptions<T>`](/docs/reference/wiki/models/ddpgoptions/) | Configuration options for DDPG agent. |
| [`DFPOptimizerOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/dfpoptimizeroptions/) | Configuration options for the Davidon-Fletcher-Powell (DFP) optimization algorithm, which is a quasi-Newton method used for finding local minima of functions. |
| [`DGCNNOptions`](/docs/reference/wiki/models/dgcnnoptions/) | Configuration options for DGCNN models. |
| [`DLinearOptions<T>`](/docs/reference/wiki/models/dlinearoptions/) | Options for `DLinearModel` — the decomposition-linear forecaster (Zeng et al., "Are Transformers Effective for Time Series Forecasting?", AAAI 2023). |
| [`DPCTGANOptions<T>`](/docs/reference/wiki/models/dpctganoptions/) | Configuration options for DP-CTGAN, a differentially private version of CTGAN that provides formal privacy guarantees while generating synthetic tabular data. |
| [`DQNOptions<T>`](/docs/reference/wiki/models/dqnoptions/) | Configuration options for Deep Q-Network (DQN) agents. |
| [`DecentralizedFederatedOptions`](/docs/reference/wiki/models/decentralizedfederatedoptions/) | Configuration options for decentralized (serverless) federated learning. |
| [`DecisionTransformerOptions<T>`](/docs/reference/wiki/models/decisiontransformeroptions/) | Configuration options for Decision Transformer agents. |
| [`DecisionTreeClassifierOptions<T>`](/docs/reference/wiki/models/decisiontreeclassifieroptions/) | Configuration options for decision tree classifiers. |
| [`DecisionTreeOptions`](/docs/reference/wiki/models/decisiontreeoptions/) | Configuration options for decision tree algorithms. |
| [`DeepAROptions<T>`](/docs/reference/wiki/models/deeparoptions/) | Configuration options for the DeepAR (Deep Autoregressive) model. |
| [`DeepFactorOptions<T>`](/docs/reference/wiki/models/deepfactoroptions/) | Configuration options for the DeepFactor (Deep Factor Model) for time series forecasting. |
| [`DeepFilterNetOptions`](/docs/reference/wiki/models/deepfilternetoptions/) | Configuration options for DeepFilterNet audio enhancement models. |
| [`DeepHitOptions<T>`](/docs/reference/wiki/models/deephitoptions/) | Configuration options for DeepHit survival analysis model. |
| [`DeepPortfolioManagerOptions<T>`](/docs/reference/wiki/models/deepportfoliomanageroptions/) | Configuration options for the DeepPortfolioManager model. |
| [`DeepStateOptions<T>`](/docs/reference/wiki/models/deepstateoptions/) | Configuration options for the DeepState (Deep State Space) model. |
| [`DeepSurvOptions<T>`](/docs/reference/wiki/models/deepsurvoptions/) | Configuration options for DeepSurv survival analysis model. |
| [`DifferentialEvolutionOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/differentialevolutionoptions/) | Configuration options for Differential Evolution optimization, a powerful variant of genetic algorithms that is particularly effective for continuous optimization problems. |
| [`DiffusionModelOptions<T>`](/docs/reference/wiki/models/diffusionmodeloptions/) | Configuration options for diffusion-based generative models. |
| [`DiffusionTSOptions<T>`](/docs/reference/wiki/models/diffusiontsoptions/) | Configuration options for DiffusionTS (Interpretable Diffusion for Time Series). |
| [`DocumentNeuralNetworkOptions`](/docs/reference/wiki/models/documentneuralnetworkoptions/) | Base configuration options for document neural network models. |
| [`DoubleDQNOptions<T>`](/docs/reference/wiki/models/doubledqnoptions/) | Configuration options for Double DQN agent. |
| [`DoubleQLearningOptions<T>`](/docs/reference/wiki/models/doubleqlearningoptions/) | Configuration options for Double Q-Learning agents. |
| [`DreamerOptions<T>`](/docs/reference/wiki/models/dreameroptions/) | Configuration options for Dreamer agents. |
| [`DuelingDQNOptions<T>`](/docs/reference/wiki/models/duelingdqnoptions/) | Configuration options for Dueling DQN agent. |
| [`DynaQOptions<T>`](/docs/reference/wiki/models/dynaqoptions/) | Configuration options for Dyna-Q reinforcement learning agents. |
| [`DynaQPlusOptions<T>`](/docs/reference/wiki/models/dynaqplusoptions/) |  |
| [`DynamicRegressionWithARIMAErrorsOptions<T>`](/docs/reference/wiki/models/dynamicregressionwitharimaerrorsoptions/) | Configuration options for Dynamic Regression with ARIMA Errors, a powerful time series forecasting method that combines regression with time series error correction. |
| [`ETSformerOptions<T>`](/docs/reference/wiki/models/etsformeroptions/) | Configuration options for the ETSformer model. |
| [`EarlyStoppingConfig`](/docs/reference/wiki/models/earlystoppingconfig/) | Early stopping configuration for a training stage. |
| [`ElasticNetRegressionOptions<T>`](/docs/reference/wiki/models/elasticnetregressionoptions/) | Configuration options for Elastic Net Regression (combined L1 and L2 regularization). |
| [`EnsembleFitDetectorOptions`](/docs/reference/wiki/models/ensemblefitdetectoroptions/) | Configuration options for the Ensemble Fit Detector, which combines multiple model fitness detectors to provide more robust and accurate recommendations for algorithm selection. |
| [`EpsilonGreedyBanditOptions<T>`](/docs/reference/wiki/models/epsilongreedybanditoptions/) |  |
| [`ExpectedSARSAOptions<T>`](/docs/reference/wiki/models/expectedsarsaoptions/) | Configuration options for Expected SARSA agents. |
| [`ExplainableBoostingClassifierOptions<T>`](/docs/reference/wiki/models/explainableboostingclassifieroptions/) | Configuration options for Explainable Boosting Machine (EBM) classifier. |
| [`ExplainableBoostingMachineOptions`](/docs/reference/wiki/models/explainableboostingmachineoptions/) | Configuration options for Explainable Boosting Machine (EBM) models. |
| [`ExponentialSmoothingOptions<T>`](/docs/reference/wiki/models/exponentialsmoothingoptions/) | Configuration options for Exponential Smoothing, a time series forecasting method that gives exponentially decreasing weights to older observations. |
| [`ExtraTreesClassifierOptions<T>`](/docs/reference/wiki/models/extratreesclassifieroptions/) | Configuration options for Extra Trees (Extremely Randomized Trees) classifier. |
| [`ExtremelyRandomizedTreesRegressionOptions`](/docs/reference/wiki/models/extremelyrandomizedtreesregressionoptions/) | Configuration options for Extremely Randomized Trees regression, an ensemble learning method that builds multiple decision trees with additional randomization for improved prediction accuracy. |
| [`FEDformerOptions<T>`](/docs/reference/wiki/models/fedformeroptions/) | Configuration options for the FEDformer (Frequency Enhanced Decomposed Transformer) model. |
| [`FTRLOptimizerOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/ftrloptimizeroptions/) | Configuration options for the Follow-The-Regularized-Leader (FTRL) optimizer, an advanced gradient-based optimization algorithm particularly effective for sparse datasets and online learning. |
| [`FTTransformerOptions<T>`](/docs/reference/wiki/models/fttransformeroptions/) | Configuration options for FT-Transformer, a Feature Tokenizer + Transformer for tabular data. |
| [`FactorTransformerOptions<T>`](/docs/reference/wiki/models/factortransformeroptions/) | Configuration options for the FactorTransformer model. |
| [`FactorVAEOptions<T>`](/docs/reference/wiki/models/factorvaeoptions/) | Configuration options for the FactorVAE model. |
| [`FeatureImportanceFitDetectorOptions`](/docs/reference/wiki/models/featureimportancefitdetectoroptions/) | Configuration options for the Feature Importance Fit Detector, which analyzes how different input features contribute to a model's predictions and evaluates potential issues with model fit. |
| [`FederatedAdapterOptions`](/docs/reference/wiki/models/federatedadapteroptions/) | Configuration options for federated parameter-efficient fine-tuning (PEFT). |
| [`FederatedCompressionOptions`](/docs/reference/wiki/models/federatedcompressionoptions/) | Configuration options for federated update compression (quantization, sparsification, and error feedback). |
| [`FederatedContinualLearningOptions`](/docs/reference/wiki/models/federatedcontinuallearningoptions/) | Configuration options for federated continual learning (preventing catastrophic forgetting in FL). |
| [`FederatedDistillationOptions`](/docs/reference/wiki/models/federateddistillationoptions/) | Configuration options for federated knowledge distillation, enabling model-heterogeneous FL. |
| [`FederatedDriftOptions`](/docs/reference/wiki/models/federateddriftoptions/) | Configuration options for federated concept drift detection and adaptation. |
| [`FederatedFairnessOptions`](/docs/reference/wiki/models/federatedfairnessoptions/) | Configuration options for fairness constraints in federated learning. |
| [`FederatedGraphOptions`](/docs/reference/wiki/models/federatedgraphoptions/) | Configuration options for federated graph learning. |
| [`FederatedHeterogeneityCorrectionOptions`](/docs/reference/wiki/models/federatedheterogeneitycorrectionoptions/) | Configuration options for federated heterogeneity correction algorithms. |
| [`FederatedLearningOptions`](/docs/reference/wiki/models/federatedlearningoptions/) | Configuration options for federated learning training. |
| [`FederatedMetaLearningOptions`](/docs/reference/wiki/models/federatedmetalearningoptions/) | Configuration options for federated meta-learning. |
| [`FederatedPersonalizationOptions`](/docs/reference/wiki/models/federatedpersonalizationoptions/) | Configuration options for personalized federated learning (PFL). |
| [`FederatedServerOptimizerOptions`](/docs/reference/wiki/models/federatedserveroptimizeroptions/) | Configuration options for server-side federated optimizers (FedOpt family). |
| [`FederatedUnlearningOptions`](/docs/reference/wiki/models/federatedunlearningoptions/) | Configuration options for federated unlearning (right to be forgotten). |
| [`FinBERTOptions<T>`](/docs/reference/wiki/models/finbertoptions/) | Configuration options for FinBERT (Financial BERT) model. |
| [`FinBERTToneOptions<T>`](/docs/reference/wiki/models/finberttoneoptions/) | Configuration options for FinBERT-Tone model for fine-grained financial sentiment/tone analysis. |
| [`FinDiffOptions<T>`](/docs/reference/wiki/models/findiffoptions/) | Configuration options for FinDiff, a diffusion model specialized for generating realistic financial tabular data with temporal correlation preservation. |
| [`FinGPTOptions<T>`](/docs/reference/wiki/models/fingptoptions/) | Configuration options for FinGPT (Financial GPT) model. |
| [`FinMAOptions<T>`](/docs/reference/wiki/models/finmaoptions/) | Configuration options for FinMA (Financial Multi-Agent) model. |
| [`FinRLAgentOptions<T>`](/docs/reference/wiki/models/finrlagentoptions/) | Configuration options for the FinRL unified trading agent. |
| [`FinancialA2CAgentOptions<T>`](/docs/reference/wiki/models/financiala2cagentoptions/) | Configuration options for the Financial A2C (Advantage Actor-Critic) trading agent. |
| [`FinancialAutoMLOptions<T>`](/docs/reference/wiki/models/financialautomloptions/) | Configuration options for financial AutoML runs. |
| [`FinancialBERTOptions<T>`](/docs/reference/wiki/models/financialbertoptions/) | Configuration options for FinancialBERT model - a domain-adapted BERT for comprehensive financial analysis. |
| [`FinancialDQNAgentOptions<T>`](/docs/reference/wiki/models/financialdqnagentoptions/) | Configuration options for the Financial DQN (Deep Q-Network) trading agent. |
| [`FinancialNLPOptions`](/docs/reference/wiki/models/financialnlpoptions/) | Base configuration options for financial NLP models. |
| [`FinancialNeuralNetworkOptions`](/docs/reference/wiki/models/financialneuralnetworkoptions/) | Base configuration options for financial neural network models. |
| [`FinancialPPOAgentOptions<T>`](/docs/reference/wiki/models/financialppoagentoptions/) | Configuration options for the Financial PPO (Proximal Policy Optimization) trading agent. |
| [`FinancialSACAgentOptions<T>`](/docs/reference/wiki/models/financialsacagentoptions/) | Configuration options for the Financial SAC (Soft Actor-Critic) trading agent. |
| [`FineTuningConfiguration<T, TInput, TOutput>`](/docs/reference/wiki/models/finetuningconfiguration/) | Configuration for fine-tuning during model building. |
| [`FineTuningOptions<T>`](/docs/reference/wiki/models/finetuningoptions/) | Configuration options for fine-tuning methods. |
| [`FitnessCalculatorOptions`](/docs/reference/wiki/models/fitnesscalculatoroptions/) | Configuration options for the fitness calculator, which determines how model performance is evaluated. |
| [`FlowStateOptions<T>`](/docs/reference/wiki/models/flowstateoptions/) | Configuration options for FlowState (IBM's SSM-based Time Series Foundation Model). |
| [`ForecastingModelOptions`](/docs/reference/wiki/models/forecastingmodeloptions/) | Base configuration options for financial forecasting models. |
| [`FuzzyMatchOptions`](/docs/reference/wiki/models/fuzzymatchoptions/) | Configuration options for fuzzy entity matching in Private Set Intersection. |
| [`GAMLSSOptions`](/docs/reference/wiki/models/gamlssoptions/) | Configuration options for GAMLSS (Generalized Additive Models for Location, Scale, and Shape). |
| [`GANDALFOptions<T>`](/docs/reference/wiki/models/gandalfoptions/) | Configuration options for GANDALF (Gated Additive Neural Decision Forest). |
| [`GARCHModelOptions<T>`](/docs/reference/wiki/models/garchmodeloptions/) | Configuration options for the Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model, which is used for analyzing and forecasting volatility in time series data. |
| [`GLMMOptions<T>`](/docs/reference/wiki/models/glmmoptions/) | Configuration options for Generalized Linear Mixed-Effects Models (GLMM). |
| [`GOGGLEOptions<T>`](/docs/reference/wiki/models/goggleoptions/) | Configuration options for GOGGLE (Generative mOdelling for tabular data by learninG reLational structurE), a graph-based VAE that learns feature dependency structure for generating realistic tabular data. |
| [`GPT4TSOptions<T>`](/docs/reference/wiki/models/gpt4tsoptions/) | Configuration options for GPT4TS (One Fits All: Power General Time Series Analysis by Pretrained LM). |
| [`GammaRegressionOptions<T>`](/docs/reference/wiki/models/gammaregressionoptions/) | Configuration options for Gamma Regression, a generalized linear model for positive continuous data. |
| [`GaussianProcessFitDetectorOptions`](/docs/reference/wiki/models/gaussianprocessfitdetectoroptions/) | Configuration options for the Gaussian Process Fit Detector, which analyzes model fit quality using Gaussian Process regression to detect overfitting, underfitting, and uncertainty issues. |
| [`GaussianProcessRegressionOptions`](/docs/reference/wiki/models/gaussianprocessregressionoptions/) | Configuration options for Gaussian Process Regression, a flexible non-parametric approach to regression that provides uncertainty estimates along with predictions. |
| [`GaussianSplattingOptions`](/docs/reference/wiki/models/gaussiansplattingoptions/) | Configuration options for Gaussian Splatting models. |
| [`GeneralizedAdditiveModelOptions<T>`](/docs/reference/wiki/models/generalizedadditivemodeloptions/) | Configuration options for Generalized Additive Models (GAMs), which are flexible regression models that combine multiple simple functions to model complex relationships. |
| [`GeneticAlgorithmOptimizerOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/geneticalgorithmoptimizeroptions/) | Configuration options for genetic algorithm optimization, which uses principles inspired by natural selection to find optimal solutions to complex problems. |
| [`GlobalEarlyStoppingConfig`](/docs/reference/wiki/models/globalearlystoppingconfig/) | Global early stopping configuration that spans multiple training stages. |
| [`GradientBanditOptions<T>`](/docs/reference/wiki/models/gradientbanditoptions/) |  |
| [`GradientBasedOptimizerOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/gradientbasedoptimizeroptions/) | Configuration options for gradient-based optimization algorithms. |
| [`GradientBoostingClassifierOptions<T>`](/docs/reference/wiki/models/gradientboostingclassifieroptions/) | Configuration options for Gradient Boosting classifier. |
| [`GradientBoostingFitDetectorOptions`](/docs/reference/wiki/models/gradientboostingfitdetectoroptions/) | Configuration options for the Gradient Boosting Fit Detector, which analyzes model fit quality to detect overfitting in gradient boosting models. |
| [`GradientBoostingRegressionOptions`](/docs/reference/wiki/models/gradientboostingregressionoptions/) | Configuration options for Gradient Boosting Regression, an ensemble learning technique that combines multiple decision trees to create a powerful regression model. |
| [`GradientDescentOptimizerOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/gradientdescentoptimizeroptions/) | Configuration options for the Gradient Descent optimizer, which is a fundamental algorithm for finding the minimum of a function by iteratively moving in the direction of steepest descent. |
| [`GraphWaveNetOptions<T>`](/docs/reference/wiki/models/graphwavenetoptions/) | Configuration options for GraphWaveNet (Graph WaveNet for Deep Spatial-Temporal Modeling). |
| [`HeteroscedasticityFitDetectorOptions`](/docs/reference/wiki/models/heteroscedasticityfitdetectoroptions/) | Configuration options for the Heteroscedasticity Fit Detector, which analyzes whether a model's prediction errors have constant variance across all prediction values. |
| [`HierarchicalRiskParityOptions<T>`](/docs/reference/wiki/models/hierarchicalriskparityoptions/) | Configuration options for Hierarchical Risk Parity. |
| [`HippoOptions<T>`](/docs/reference/wiki/models/hippooptions/) | Configuration options for HiPPO (High-order Polynomial Projection Operators) model. |
| [`HistGradientBoostingOptions`](/docs/reference/wiki/models/histgradientboostingoptions/) | Configuration options for Histogram-based Gradient Boosting, a fast ensemble learning technique that uses binned features for efficient tree building. |
| [`HoeffdingTreeOptions<T>`](/docs/reference/wiki/models/hoeffdingtreeoptions/) | Configuration options for Hoeffding Tree classifier. |
| [`HoldoutValidationFitDetectorOptions`](/docs/reference/wiki/models/holdoutvalidationfitdetectoroptions/) | Configuration options for the Holdout Validation Fit Detector, which analyzes model performance on separate training and validation datasets to identify overfitting, underfitting, and other model quality issues. |
| [`HomomorphicEncryptionOptions`](/docs/reference/wiki/models/homomorphicencryptionoptions/) | Configuration options for homomorphic encryption (HE) in federated learning. |
| [`HybridFitDetectorOptions`](/docs/reference/wiki/models/hybridfitdetectoroptions/) | Configuration options for the Hybrid Fit Detector, which combines multiple model evaluation techniques to provide a comprehensive assessment of model quality. |
| [`IQLOptions<T>`](/docs/reference/wiki/models/iqloptions/) | Configuration options for Implicit Q-Learning (IQL) agent. |
| [`ITransformerOptions<T>`](/docs/reference/wiki/models/itransformeroptions/) | Configuration options for the iTransformer (Inverted Transformer) model. |
| [`InformationCriteriaFitDetectorOptions`](/docs/reference/wiki/models/informationcriteriafitdetectoroptions/) | Configuration options for the Information Criteria Fit Detector, which uses statistical information criteria like AIC and BIC to evaluate model quality and complexity trade-offs. |
| [`InformerOptions<T>`](/docs/reference/wiki/models/informeroptions/) | Configuration options for the Informer model (Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting). |
| [`InstantNGPOptions<T>`](/docs/reference/wiki/models/instantngpoptions/) | Configuration options for Instant-NGP models. |
| [`InterpretabilityOptions`](/docs/reference/wiki/models/interpretabilityoptions/) | Configuration options for model interpretability and explainability features. |
| [`InterventionAnalysisOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/interventionanalysisoptions/) | Configuration options for Intervention Analysis, which is a time series modeling technique used to assess the impact of specific events or interventions on a time series. |
| [`InverseGaussianRegressionOptions<T>`](/docs/reference/wiki/models/inversegaussianregressionoptions/) | Configuration options for Inverse Gaussian Regression, a generalized linear model for positive continuous data with variance proportional to the cube of the mean. |
| [`InvestLMOptions<T>`](/docs/reference/wiki/models/investlmoptions/) | Configuration options for InvestLM (Investment Language Model). |
| [`JackknifeFitDetectorOptions`](/docs/reference/wiki/models/jackknifefitdetectoroptions/) | Configuration options for the Jackknife Fit Detector, which uses the jackknife resampling technique to evaluate model stability and detect overfitting or underfitting. |
| [`KFoldCrossValidationFitDetectorOptions`](/docs/reference/wiki/models/kfoldcrossvalidationfitdetectoroptions/) | Configuration options for the K-Fold Cross Validation Fit Detector, which evaluates model quality by analyzing performance across multiple data partitions. |
| [`KNearestNeighborsOptions`](/docs/reference/wiki/models/knearestneighborsoptions/) | Configuration options for the K-Nearest Neighbors algorithm, which makes predictions based on the values of the K closest data points in the training set. |
| [`KNeighborsOptions<T>`](/docs/reference/wiki/models/kneighborsoptions/) | Configuration options for K-Nearest Neighbors classifiers. |
| [`KairosOptions<T>`](/docs/reference/wiki/models/kairosoptions/) | Configuration options for Kairos (Adaptive and Generalizable Time Series Foundation Model). |
| [`KernelRidgeRegressionOptions`](/docs/reference/wiki/models/kernelridgeregressionoptions/) | Configuration options for Kernel Ridge Regression, which combines ridge regression with the kernel trick to model non-linear relationships in data. |
| [`KnowledgeDistillationOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/knowledgedistillationoptions/) | Configuration options for knowledge distillation training. |
| [`KronosOptions<T>`](/docs/reference/wiki/models/kronosoptions/) | Configuration options for Kronos (Foundation Model for the Language of Financial Markets). |
| [`LAMBOptimizerOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/lamboptimizeroptions/) | Configuration options for the LAMB (Layer-wise Adaptive Moments for Batch training) optimization algorithm. |
| [`LARSOptimizerOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/larsoptimizeroptions/) | Configuration options for the LARS (Layer-wise Adaptive Rate Scaling) optimization algorithm. |
| [`LBFGSOptimizerOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/lbfgsoptimizeroptions/) | Configuration options for the Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimizer, which is an efficient optimization algorithm for training machine learning models. |
| [`LLMTimeOptions<T>`](/docs/reference/wiki/models/llmtimeoptions/) | Configuration options for LLM-Time (Zero-Shot Time Series Forecasting via LLM Tokenization). |
| [`LSPIOptions<T>`](/docs/reference/wiki/models/lspioptions/) | Configuration options for LSPI (Least-Squares Policy Iteration) agents. |
| [`LSTDOptions<T>`](/docs/reference/wiki/models/lstdoptions/) | Configuration options for LSTD (Least-Squares Temporal Difference) agents. |
| [`LSTNetOptions<T>`](/docs/reference/wiki/models/lstnetoptions/) | Configuration options for the LSTNet (Long Short-Term Time-series Network) model. |
| [`LagLlamaOptions<T>`](/docs/reference/wiki/models/lagllamaoptions/) | Configuration options for Lag-Llama (Large Language Model for time series forecasting). |
| [`LassoRegressionOptions<T>`](/docs/reference/wiki/models/lassoregressionoptions/) | Configuration options for Lasso Regression (L1 regularized linear regression). |
| [`LearningCurveFitDetectorOptions`](/docs/reference/wiki/models/learningcurvefitdetectoroptions/) | Configuration options for the Learning Curve Fit Detector, which analyzes training progress to determine when a model has converged or is unlikely to improve further. |
| [`LevenbergMarquardtOptimizerOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/levenbergmarquardtoptimizeroptions/) | Configuration options for the Levenberg-Marquardt optimization algorithm, which is used for non-linear least squares optimization in machine learning and AI models. |
| [`LinearClassifierOptions<T>`](/docs/reference/wiki/models/linearclassifieroptions/) | Configuration options for linear classifiers. |
| [`LinearMixedModelOptions<T>`](/docs/reference/wiki/models/linearmixedmodeloptions/) | Configuration options for mixed-effects (hierarchical/multilevel) models. |
| [`LinearQLearningOptions<T>`](/docs/reference/wiki/models/linearqlearningoptions/) | Configuration options for Linear Q-Learning agents. |
| [`LinearSARSAOptions<T>`](/docs/reference/wiki/models/linearsarsaoptions/) | Configuration options for Linear SARSA agents. |
| [`LionOptimizerOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/lionoptimizeroptions/) | Configuration options for the Lion (Evolved Sign Momentum) optimization algorithm. |
| [`LoRAConfiguration`](/docs/reference/wiki/models/loraconfiguration/) | LoRA configuration for parameter-efficient fine-tuning. |
| [`LocallyWeightedRegressionOptions`](/docs/reference/wiki/models/locallyweightedregressionoptions/) | Configuration options for Locally Weighted Regression, a non-parametric method that creates a model by fitting simple models to localized subsets of data. |
| [`LogisticRegressionOptions<T>`](/docs/reference/wiki/models/logisticregressionoptions/) | Configuration options for Logistic Regression, a statistical method used for binary classification problems in machine learning. |
| [`M5ModelTreeOptions`](/docs/reference/wiki/models/m5modeltreeoptions/) | Configuration options for the M5 Model Tree algorithm, which combines decision trees with linear regression models at the leaf nodes. |
| [`MADDPGOptions<T>`](/docs/reference/wiki/models/maddpgoptions/) | Configuration options for Multi-Agent DDPG (MADDPG) agents. |
| [`MAModelOptions<T>`](/docs/reference/wiki/models/mamodeloptions/) | Configuration options for Moving Average (MA) models, which are used to analyze time series data by modeling the error terms as a linear combination of previous error terms. |
| [`MGTSDOptions<T>`](/docs/reference/wiki/models/mgtsdoptions/) | Configuration options for MG-TSD (Multi-Granularity Time Series Diffusion). |
| [`MLkNNOptions<T>`](/docs/reference/wiki/models/mlknnoptions/) | Configuration options for ML-kNN (Multi-Label k-Nearest Neighbors) classifier. |
| [`MOIRAIOptions<T>`](/docs/reference/wiki/models/moiraioptions/) | Configuration options for MOIRAI (Salesforce's Universal Time Series Foundation Model). |
| [`MOMENTOptions<T>`](/docs/reference/wiki/models/momentoptions/) | Configuration options for MOMENT (Multi-task Optimization through Masked Encoding for Time series) foundation model. |
| [`MQCNNOptions<T>`](/docs/reference/wiki/models/mqcnnoptions/) | Configuration options for the MQCNN (Multi-Quantile Convolutional Neural Network) model. |
| [`MTGNNOptions<T>`](/docs/reference/wiki/models/mtgnnoptions/) | Configuration options for MTGNN (Multivariate Time-series Graph Neural Network). |
| [`Mamba2Options<T>`](/docs/reference/wiki/models/mamba2options/) | Configuration options for Mamba-2 (Structured State Space Duality) forecasting. |
| [`MambaOptions<T>`](/docs/reference/wiki/models/mambaoptions/) | Configuration options for Mamba (Selective State Space Model). |
| [`MambularOptions<T>`](/docs/reference/wiki/models/mambularoptions/) | Configuration options for Mambular (State Space Models for Tabular Data). |
| [`MarketMakingOptions<T>`](/docs/reference/wiki/models/marketmakingoptions/) | Configuration options for the MarketMakingAgent. |
| [`MedSynthOptions<T>`](/docs/reference/wiki/models/medsynthoptions/) | Configuration options for MedSynth, a privacy-preserving medical tabular data synthesis model combining a VAE/GAN hybrid with clinical validity constraints. |
| [`MeshCNNOptions`](/docs/reference/wiki/models/meshcnnoptions/) | Configuration options for the MeshCNN neural network. |
| [`MiniBatchGradientDescentOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/minibatchgradientdescentoptions/) | Configuration options for Mini-Batch Gradient Descent, an optimization algorithm that updates model parameters using the average gradient computed from small random subsets of training data. |
| [`MiniRocketOptions<T>`](/docs/reference/wiki/models/minirocketoptions/) | Configuration options for the MiniRocket time series classifier. |
| [`MisGANOptions<T>`](/docs/reference/wiki/models/misganoptions/) | Configuration options for MisGAN, a GAN for learning from incomplete data with dual generator/discriminator pairs for data and missingness patterns. |
| [`MissingFeatureOptions`](/docs/reference/wiki/models/missingfeatureoptions/) | Configuration for handling missing features in vertical federated learning. |
| [`MixedEffectsModelOptions`](/docs/reference/wiki/models/mixedeffectsmodeloptions/) | Configuration options for Mixed-Effects (Hierarchical/Multilevel) Models. |
| [`MixtureOfExpertsOptions<T>`](/docs/reference/wiki/models/mixtureofexpertsoptions/) | Configuration options for the Mixture-of-Experts (MoE) neural network model. |
| [`ModelOptions`](/docs/reference/wiki/models/modeloptions/) |  |
| [`ModelStatsOptions`](/docs/reference/wiki/models/modelstatsoptions/) | Configuration options for model statistics and diagnostics calculations, which help evaluate the quality, reliability, and performance of machine learning models. |
| [`ModifiedPolicyIterationOptions<T>`](/docs/reference/wiki/models/modifiedpolicyiterationoptions/) | Configuration options for Modified Policy Iteration agents. |
| [`MomentumOptimizerOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/momentumoptimizeroptions/) | Configuration options for the Momentum Optimizer, which enhances gradient descent by adding a fraction of the previous update direction to the current update. |
| [`MonteCarloExploringStartsOptions<T>`](/docs/reference/wiki/models/montecarloexploringstartsoptions/) | Configuration options for Monte Carlo Exploring Starts agents. |
| [`MonteCarloOptions<T>`](/docs/reference/wiki/models/montecarlooptions/) | Configuration options for Monte Carlo agents. |
| [`MonteCarloValidationOptions`](/docs/reference/wiki/models/montecarlovalidationoptions/) | Represents the options for Monte Carlo cross-validation. |
| [`MpcOptions`](/docs/reference/wiki/models/mpcoptions/) | Configuration options for multi-party computation protocols in federated learning. |
| [`MuZeroOptions<T>`](/docs/reference/wiki/models/muzerooptions/) | Configuration options for MuZero agents. |
| [`MultilayerPerceptronOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/multilayerperceptronoptions/) | Configuration options for Multilayer Perceptron (MLP), a type of feedforward artificial neural network that consists of multiple layers of neurons. |
| [`MultinomialLogisticRegressionOptions<T>`](/docs/reference/wiki/models/multinomiallogisticregressionoptions/) | Configuration options for Multinomial Logistic Regression, a classification method that generalizes logistic regression to multiclass problems with more than two possible discrete outcomes. |
| [`NASOptions<T>`](/docs/reference/wiki/models/nasoptions/) | Configuration options for Neural Architecture Search (NAS). |
| [`NBEATSModelOptions<T>`](/docs/reference/wiki/models/nbeatsmodeloptions/) | Configuration options for the N-BEATS (Neural Basis Expansion Analysis for Time Series) model. |
| [`NGBoostClassifierOptions<T>`](/docs/reference/wiki/models/ngboostclassifieroptions/) | Configuration options for NGBoost (Natural Gradient Boosting) classifier models. |
| [`NGBoostRegressionOptions`](/docs/reference/wiki/models/ngboostregressionoptions/) | Configuration options for NGBoost (Natural Gradient Boosting) regression models. |
| [`NHiTSOptions<T>`](/docs/reference/wiki/models/nhitsoptions/) | Configuration options for the N-HiTS (Neural Hierarchical Interpolation for Time Series) model. |
| [`NLinearOptions<T>`](/docs/reference/wiki/models/nlinearoptions/) | Options for `NLinearModel` — the normalization-linear forecaster (Zeng et al., AAAI 2023). |
| [`NODEOptions<T>`](/docs/reference/wiki/models/nodeoptions/) | Configuration options for NODE (Neural Oblivious Decision Ensembles). |
| [`NStepQLearningOptions<T>`](/docs/reference/wiki/models/nstepqlearningoptions/) |  |
| [`NStepSARSAOptions<T>`](/docs/reference/wiki/models/nstepsarsaoptions/) | Configuration options for N-step SARSA agents. |
| [`NadamOptimizerOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/nadamoptimizeroptions/) | Configuration options for the Nadam optimizer, which combines Nesterov momentum with Adam's adaptive learning rates for efficient training of neural networks and other gradient-based models. |
| [`NaiveBayesOptions<T>`](/docs/reference/wiki/models/naivebayesoptions/) | Configuration options for Naive Bayes classifiers. |
| [`NeRFOptions`](/docs/reference/wiki/models/nerfoptions/) | Configuration options for NeRF models. |
| [`NegativeBinomialRegressionOptions<T>`](/docs/reference/wiki/models/negativebinomialregressionoptions/) | Configuration options for Negative Binomial Regression, a statistical model used for count data that exhibits overdispersion (variance exceeding the mean). |
| [`NelderMeadOptimizerOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/neldermeadoptimizeroptions/) | Configuration options for the Nelder-Mead optimization algorithm, a derivative-free method for finding the minimum of an objective function in a multidimensional space. |
| [`NesterovAcceleratedGradientOptimizerOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/nesterovacceleratedgradientoptimizeroptions/) | Configuration options for the Nesterov Accelerated Gradient optimization algorithm, a momentum-based technique that improves convergence speed in gradient descent optimization. |
| [`NeuralCVaROptions<T>`](/docs/reference/wiki/models/neuralcvaroptions/) | Configuration options for the NeuralCVaR model. |
| [`NeuralGARCHOptions<T>`](/docs/reference/wiki/models/neuralgarchoptions/) | Configuration options for the Neural GARCH volatility model. |
| [`NeuralNetworkARIMAOptions<T>`](/docs/reference/wiki/models/neuralnetworkarimaoptions/) | Configuration options for Neural Network ARIMA (AutoRegressive Integrated Moving Average) models,  which combine traditional statistical time series methods with neural networks for improved forecasting. |
| [`NeuralNetworkFitDetectorOptions`](/docs/reference/wiki/models/neuralnetworkfitdetectoroptions/) | Configuration options for the Neural Network Fit Detector, which evaluates the quality of a neural network's fit to data by analyzing performance metrics and detecting issues like underfitting and overfitting. |
| [`NeuralNetworkOptions`](/docs/reference/wiki/models/neuralnetworkoptions/) | Base configuration options for all neural network models. |
| [`NeuralNetworkRegressionOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/neuralnetworkregressionoptions/) | Configuration options for neural network regression models, providing fine-grained control over network architecture, training parameters, activation functions, and optimization strategies. |
| [`NeuralNoiseReducerOptions`](/docs/reference/wiki/models/neuralnoisereduceroptions/) | Configuration options for neural noise reduction models. |
| [`NeuralStressTestOptions<T>`](/docs/reference/wiki/models/neuralstresstestoptions/) | Configuration options for the NeuralStressTest model. |
| [`NeuralVaROptions<T>`](/docs/reference/wiki/models/neuralvaroptions/) | Configuration options for the NeuralVaR model. |
| [`NewtonMethodOptimizerOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/newtonmethodoptimizeroptions/) | Configuration options for Newton's Method optimizer, an advanced second-order optimization technique that uses both gradient and Hessian information to accelerate convergence in optimization problems. |
| [`NonLinearRegressionOptions`](/docs/reference/wiki/models/nonlinearregressionoptions/) | Configuration options for nonlinear regression models, which capture complex, nonlinear relationships between input features and output variables using kernel functions and iterative optimization. |
| [`NonStationaryTransformerOptions<T>`](/docs/reference/wiki/models/nonstationarytransformeroptions/) | Configuration options for the Non-stationary Transformer model. |
| [`OCTGANOptions<T>`](/docs/reference/wiki/models/octganoptions/) | Configuration options for OCT-GAN (One-Class Tabular GAN), designed for generating synthetic data with a focus on minority/imbalanced classes. |
| [`ObjectDetectionOptions<T>`](/docs/reference/wiki/models/objectdetectionoptions/) | Configuration options for object detection models. |
| [`OffPolicyMonteCarloOptions<T>`](/docs/reference/wiki/models/offpolicymontecarlooptions/) | Configuration options for Off-Policy Monte Carlo Control agents with importance sampling. |
| [`OhlcColumnConfig`](/docs/reference/wiki/models/ohlccolumnconfig/) | Configuration for OHLC (Open, High, Low, Close) column indices. |
| [`OnPolicyMonteCarloOptions<T>`](/docs/reference/wiki/models/onpolicymontecarlooptions/) | Configuration options for On-Policy Monte Carlo Control agents. |
| [`OnlineClassifierOptions<T>`](/docs/reference/wiki/models/onlineclassifieroptions/) | Base configuration options for online (incremental) classifiers. |
| [`OnlineNaiveBayesOptions<T>`](/docs/reference/wiki/models/onlinenaivebayesoptions/) | Configuration options for Online Naive Bayes classifier. |
| [`OptimizationAlgorithmOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/optimizationalgorithmoptions/) | Configuration options for optimization algorithms used in machine learning models. |
| [`OrdinalRegressionOptions<T>`](/docs/reference/wiki/models/ordinalregressionoptions/) | Configuration options for Ordinal Regression (Proportional Odds Model), a classification method for predicting ordered categorical outcomes. |
| [`OrthogonalRegressionOptions<T>`](/docs/reference/wiki/models/orthogonalregressionoptions/) | Configuration options for Orthogonal Regression (also known as Total Least Squares), which minimizes  the perpendicular distances from data points to the fitted model, accounting for errors in both  dependent and independent variables. |
| [`PANNsModelOptions`](/docs/reference/wiki/models/pannsmodeloptions/) | Configuration options for PANNs (Pretrained Audio Neural Networks) models (Kong et al. |
| [`PATEGANOptions<T>`](/docs/reference/wiki/models/pateganoptions/) | Configuration options for PATE-GAN, a differentially private GAN that uses the Private Aggregation of Teacher Ensembles (PATE) framework for privacy-preserving synthetic data generation. |
| [`PPOOptions<T>`](/docs/reference/wiki/models/ppooptions/) | Configuration options for Proximal Policy Optimization (PPO) agents. |
| [`PartialDependencePlotFitDetectorOptions`](/docs/reference/wiki/models/partialdependenceplotfitdetectoroptions/) | Configuration options for the Partial Dependence Plot Fit Detector, which uses partial dependence plots to evaluate model fit quality and detect overfitting or underfitting in machine learning models. |
| [`PartialLeastSquaresRegressionOptions<T>`](/docs/reference/wiki/models/partialleastsquaresregressionoptions/) | Configuration options for Partial Least Squares Regression (PLS), a technique that combines features of principal component analysis and multiple regression to handle multicollinearity and high-dimensional data. |
| [`ParticleSwarmOptimizationOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/particleswarmoptimizationoptions/) | Configuration options for Particle Swarm Optimization (PSO), a population-based stochastic optimization technique inspired by social behavior of bird flocking or fish schooling. |
| [`PatchTSTOptions<T>`](/docs/reference/wiki/models/patchtstoptions/) | Configuration options for the PatchTST (Patch Time Series Transformer) model. |
| [`PermutationTestFitDetectorOptions`](/docs/reference/wiki/models/permutationtestfitdetectoroptions/) | Configuration options for the permutation test fit detector, which helps identify overfitting, underfitting, and high variance in machine learning models. |
| [`PhysicsInformedOptions`](/docs/reference/wiki/models/physicsinformedoptions/) | Base configuration options for physics-informed neural network models. |
| [`PointNetOptions`](/docs/reference/wiki/models/pointnetoptions/) | Configuration options for PointNet models. |
| [`PointNetPlusPlusOptions`](/docs/reference/wiki/models/pointnetplusplusoptions/) | Configuration options for PointNet++ models. |
| [`PoissonRegressionOptions<T>`](/docs/reference/wiki/models/poissonregressionoptions/) | Configuration options for Poisson Regression, a specialized form of regression analysis used for modeling count data and contingency tables where the dependent variable consists of non-negative integers. |
| [`PolicyIterationOptions<T>`](/docs/reference/wiki/models/policyiterationoptions/) | Configuration options for Policy Iteration agents. |
| [`PolynomialRegressionOptions<T>`](/docs/reference/wiki/models/polynomialregressionoptions/) | Configuration options for Polynomial Regression, an extension of linear regression that models the relationship between variables using polynomial functions to capture non-linear relationships in data. |
| [`PortfolioOptimizerOptions<T>`](/docs/reference/wiki/models/portfoliooptimizeroptions/) | Base options for portfolio optimizers. |
| [`PortfolioOptions<T>`](/docs/reference/wiki/models/portfoliooptions/) | Configuration options for portfolio optimization models. |
| [`PowellOptimizerOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/powelloptimizeroptions/) | Configuration options for Powell's method, a derivative-free optimization algorithm used for finding the minimum of a function without requiring gradient information. |
| [`PrecisionRecallCurveFitDetectorOptions`](/docs/reference/wiki/models/precisionrecallcurvefitdetectoroptions/) | Configuration options for the Precision-Recall Curve Fit Detector, which evaluates model quality using precision-recall metrics particularly valuable for imbalanced classification problems. |
| [`PredictionStatsOptions`](/docs/reference/wiki/models/predictionstatsoptions/) | Configuration options for prediction statistics generation, which provides statistical analysis and reporting for model predictions including confidence intervals and learning curve analysis. |
| [`PrincipalComponentRegressionOptions<T>`](/docs/reference/wiki/models/principalcomponentregressionoptions/) | Configuration options for Principal Component Regression (PCR), which combines principal component analysis with linear regression to address multicollinearity and dimensionality issues in regression problems. |
| [`PrioritizedSweepingOptions<T>`](/docs/reference/wiki/models/prioritizedsweepingoptions/) |  |
| [`ProbabilityCalibratorOptions`](/docs/reference/wiki/models/probabilitycalibratoroptions/) | Configuration options for probability calibrator. |
| [`ProphetOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/prophetoptions/) | Configuration options for Prophet, a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. |
| [`ProximalGradientDescentOptimizerOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/proximalgradientdescentoptimizeroptions/) | Configuration options for the Proximal Gradient Descent optimizer, an advanced optimization algorithm that combines traditional gradient descent with proximal operators to handle regularization effectively. |
| [`PsiOptions`](/docs/reference/wiki/models/psioptions/) | Configuration options for Private Set Intersection (PSI) protocols. |
| [`QLambdaOptions<T>`](/docs/reference/wiki/models/qlambdaoptions/) |  |
| [`QMIXOptions<T>`](/docs/reference/wiki/models/qmixoptions/) | Configuration options for QMIX agents. |
| [`QuantileRegressionForestsOptions`](/docs/reference/wiki/models/quantileregressionforestsoptions/) | Configuration options for Quantile Regression Forests, an extension of Random Forests that enables prediction of conditional quantiles rather than just the conditional mean. |
| [`QuantileRegressionOptions<T>`](/docs/reference/wiki/models/quantileregressionoptions/) | Configuration options for Quantile Regression, a technique that enables prediction of specific quantiles of the conditional distribution rather than just the conditional mean. |
| [`RAkELOptions<T>`](/docs/reference/wiki/models/rakeloptions/) | Configuration options for RAkEL (Random k-Labelsets) classifier. |
| [`REINFORCEOptions<T>`](/docs/reference/wiki/models/reinforceoptions/) | Configuration options for REINFORCE agents. |
| [`REaLTabFormerOptions<T>`](/docs/reference/wiki/models/realtabformeroptions/) | Configuration options for REaLTabFormer, a GPT-2 style autoregressive transformer for generating realistic tabular data by treating columns as a sequence of tokens. |
| [`ROCCurveFitDetectorOptions`](/docs/reference/wiki/models/roccurvefitdetectoroptions/) | Configuration options for the ROC Curve Fit Detector, which evaluates classification model quality using Receiver Operating Characteristic (ROC) curve analysis. |
| [`RWKV7LanguageModelOptions<T>`](/docs/reference/wiki/models/rwkv7languagemodeloptions/) | Configuration options for the RWKV-7 "Goose" language model. |
| [`RWKVForecastingOptions<T>`](/docs/reference/wiki/models/rwkvforecastingoptions/) | Configuration options for RWKV-based time series forecasting. |
| [`RadialBasisFunctionOptions`](/docs/reference/wiki/models/radialbasisfunctionoptions/) | Configuration options for Radial Basis Function (RBF) models, a type of artificial neural network that uses radial basis functions as activation functions for approximating complex non-linear relationships. |
| [`RainbowDQNOptions<T>`](/docs/reference/wiki/models/rainbowdqnoptions/) | Configuration options for Rainbow DQN agent. |
| [`RandomForestClassifierOptions<T>`](/docs/reference/wiki/models/randomforestclassifieroptions/) | Configuration options for Random Forest classifiers. |
| [`RandomForestRegressionOptions`](/docs/reference/wiki/models/randomforestregressionoptions/) | Configuration options for Random Forest Regression, an ensemble learning method that combines multiple decision trees to improve prediction accuracy and control overfitting. |
| [`RealizedVolatilityTransformerOptions<T>`](/docs/reference/wiki/models/realizedvolatilitytransformeroptions/) | Configuration options for the Realized Volatility Transformer model. |
| [`RegressionOptions<T>`](/docs/reference/wiki/models/regressionoptions/) | Configuration options for regression models, which are statistical methods used to estimate  relationships between variables and make predictions. |
| [`RegularizationOptions`](/docs/reference/wiki/models/regularizationoptions/) | Configuration options for regularization techniques used to prevent overfitting in machine learning models. |
| [`RelationalGCNOptions<T>`](/docs/reference/wiki/models/relationalgcnoptions/) | Configuration options for RelationalGCN (Relational Graph Convolutional Network). |
| [`ResidualAnalysisFitDetectorOptions`](/docs/reference/wiki/models/residualanalysisfitdetectoroptions/) | Configuration options for the Residual Analysis Fit Detector, which evaluates model fit quality by analyzing prediction residuals against various statistical thresholds. |
| [`ResidualBootstrapFitDetectorOptions`](/docs/reference/wiki/models/residualbootstrapfitdetectoroptions/) | Configuration options for the Residual Bootstrap Fit Detector, which uses bootstrap resampling of residuals to assess model fit quality and detect overfitting or underfitting. |
| [`RidgeRegressionOptions<T>`](/docs/reference/wiki/models/ridgeregressionoptions/) | Configuration options for Ridge Regression (L2 regularized linear regression). |
| [`RiskModelOptions<T>`](/docs/reference/wiki/models/riskmodeloptions/) | Base options for risk models. |
| [`RobustAggregationOptions`](/docs/reference/wiki/models/robustaggregationoptions/) | Configuration options for robust aggregation strategies in federated learning. |
| [`RobustRegressionOptions<T>`](/docs/reference/wiki/models/robustregressionoptions/) | Configuration options for robust regression models, which are designed to be less sensitive to outliers and violations of standard regression assumptions. |
| [`RootMeanSquarePropagationOptimizerOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/rootmeansquarepropagationoptimizeroptions/) | Configuration options for the Root Mean Square Propagation (RMSProp) optimizer, an adaptive learning rate optimization algorithm commonly used in training neural networks. |
| [`S4Options<T>`](/docs/reference/wiki/models/s4options/) | Configuration options for S4 (Structured State Space Sequence Model). |
| [`SACOptions<T>`](/docs/reference/wiki/models/sacoptions/) | Configuration options for Soft Actor-Critic (SAC) agents. |
| [`SAINTOptions<T>`](/docs/reference/wiki/models/saintoptions/) | Configuration options for SAINT (Self-Attention and Intersample Attention Transformer). |
| [`SARIMAOptions<T>`](/docs/reference/wiki/models/sarimaoptions/) | Configuration options for Seasonal Autoregressive Integrated Moving Average (SARIMA) models, which extend ARIMA models to incorporate seasonal components in time series data. |
| [`SARSALambdaOptions<T>`](/docs/reference/wiki/models/sarsalambdaoptions/) |  |
| [`SARSAOptions<T>`](/docs/reference/wiki/models/sarsaoptions/) | Configuration options for SARSA agents. |
| [`SECBERTOptions<T>`](/docs/reference/wiki/models/secbertoptions/) | Configuration options for SEC-BERT model specialized for SEC filings analysis. |
| [`SMOTENCOptions<T>`](/docs/reference/wiki/models/smotencoptions/) | Configuration options for SMOTE-NC (Synthetic Minority Over-sampling Technique for Nominal and Continuous features), a k-NN based oversampling method that generates synthetic minority samples by interpolating between existing ones. |
| [`STGNNOptions<T>`](/docs/reference/wiki/models/stgnnoptions/) | Configuration options for STGNN (Spatio-Temporal Graph Neural Network). |
| [`STLDecompositionOptions<T>`](/docs/reference/wiki/models/stldecompositionoptions/) | Configuration options for Seasonal-Trend-Loess (STL) decomposition, a versatile method for decomposing time series into seasonal, trend, and residual components. |
| [`SVMOptions<T>`](/docs/reference/wiki/models/svmoptions/) | Configuration options for Support Vector Machine classifiers. |
| [`SafetyFilterConfiguration<T>`](/docs/reference/wiki/models/safetyfilterconfiguration/) | Configuration for safety filtering during inference. |
| [`SafetyFilterOptions<T>`](/docs/reference/wiki/models/safetyfilteroptions/) | Configuration options for safety filtering mechanisms. |
| [`ScoreGradOptions<T>`](/docs/reference/wiki/models/scoregradoptions/) | Configuration options for ScoreGrad (Score-based Gradient Models for Time Series). |
| [`SecureAggregationOptions`](/docs/reference/wiki/models/secureaggregationoptions/) | Configuration options for secure aggregation in federated learning. |
| [`ShapleyValueFitDetectorOptions`](/docs/reference/wiki/models/shapleyvaluefitdetectoroptions/) | Configuration options for the Shapley Value Fit Detector, which evaluates model fit quality by analyzing feature importance using Shapley values. |
| [`SileroVadOptions`](/docs/reference/wiki/models/silerovadoptions/) | Configuration options for Silero VAD (Voice Activity Detection) models. |
| [`SimMTMOptions<T>`](/docs/reference/wiki/models/simmtmoptions/) | Configuration options for SimMTM (Simple Pre-Training Framework for Masked Time-Series Modeling). |
| [`SimulatedAnnealingOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/simulatedannealingoptions/) | Configuration options for the Simulated Annealing optimization algorithm, a probabilistic technique for approximating the global optimum of a given function. |
| [`SpectralAnalysisOptions<T>`](/docs/reference/wiki/models/spectralanalysisoptions/) | Configuration options for spectral analysis of time series data, which transforms time-domain signals into the frequency domain to identify periodic components and patterns. |
| [`SpiralNetOptions`](/docs/reference/wiki/models/spiralnetoptions/) | Configuration options for SpiralNet++ mesh neural network. |
| [`SplineRegressionOptions`](/docs/reference/wiki/models/splineregressionoptions/) | Configuration options for spline regression models, which fit piecewise polynomial functions to data for flexible nonlinear modeling. |
| [`SplitModelOptions`](/docs/reference/wiki/models/splitmodeloptions/) | Configuration for split neural network architecture in vertical federated learning. |
| [`StateSpaceModelOptions<T>`](/docs/reference/wiki/models/statespacemodeloptions/) | Configuration options for State Space Models, which represent time series data through hidden states and observable outputs for forecasting and analysis. |
| [`StepwiseRegressionOptions<T>`](/docs/reference/wiki/models/stepwiseregressionoptions/) | Configuration options for Stepwise Regression, an automated feature selection approach that iteratively adds or removes predictors based on their statistical significance. |
| [`StochasticGradientDescentOptimizerOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/stochasticgradientdescentoptimizeroptions/) | Configuration options for Stochastic Gradient Descent (SGD) optimization, a widely used algorithm for training machine learning models with large datasets. |
| [`StratifiedKFoldCrossValidationFitDetectorOptions`](/docs/reference/wiki/models/stratifiedkfoldcrossvalidationfitdetectoroptions/) | Configuration options for detecting overfitting, underfitting, and model stability using stratified k-fold cross-validation. |
| [`SubgraphSamplingOptions`](/docs/reference/wiki/models/subgraphsamplingoptions/) | Configuration for subgraph neighborhood sampling during federated GNN training. |
| [`SundialOptions<T>`](/docs/reference/wiki/models/sundialoptions/) | Configuration options for Sundial (A Family of Highly Capable Time Series Foundation Models). |
| [`SuperLearnerOptions`](/docs/reference/wiki/models/superlearneroptions/) | Configuration options for Super Learner (Stacking) ensemble. |
| [`SupportVectorRegressionOptions`](/docs/reference/wiki/models/supportvectorregressionoptions/) | Configuration options for Support Vector Regression (SVR), a powerful regression technique that uses support vector machines to predict continuous values. |
| [`SymbolicRegressionOptions`](/docs/reference/wiki/models/symbolicregressionoptions/) | Configuration options for Symbolic Regression, an evolutionary approach to finding mathematical expressions that best fit a dataset. |
| [`TBATSModelOptions<T>`](/docs/reference/wiki/models/tbatsmodeloptions/) | Configuration options for the TBATS (Trigonometric seasonality, Box-Cox transformation, ARMA errors,  Trend, and Seasonal components) time series forecasting model. |
| [`TCNOptions<T>`](/docs/reference/wiki/models/tcnoptions/) | Configuration options for the TCN (Temporal Convolutional Network) model. |
| [`TD3Options<T>`](/docs/reference/wiki/models/td3options/) | Configuration options for TD3 agent. |
| [`TESTOptions<T>`](/docs/reference/wiki/models/testoptions/) | Configuration options for TEST (Text Embedding for Seasonality and Trend — Generating Text-Aligned Embeddings for Time Series). |
| [`TFCOptions<T>`](/docs/reference/wiki/models/tfcoptions/) | Configuration options for TF-C (Time-Frequency Consistency for Self-Supervised Time Series). |
| [`TOTEMOptions<T>`](/docs/reference/wiki/models/totemoptions/) | Configuration options for TOTEM (TOkenized Time Series EMbeddings). |
| [`TOTOOptions<T>`](/docs/reference/wiki/models/totooptions/) | Configuration options for TOTO (Datadog's Time Series Foundation Model for Observability). |
| [`TRPOOptions<T>`](/docs/reference/wiki/models/trpooptions/) | Configuration options for Trust Region Policy Optimization (TRPO) agents. |
| [`TS2VecOptions<T>`](/docs/reference/wiki/models/ts2vecoptions/) | Configuration options for TS2Vec (Contrastive Learning of Universal Time Series Representations). |
| [`TSDiffOptions<T>`](/docs/reference/wiki/models/tsdiffoptions/) | Configuration options for TSDiff (Time Series Diffusion for unconditional/conditional generation). |
| [`TSMixerOptions<T>`](/docs/reference/wiki/models/tsmixeroptions/) | Configuration options for the TSMixer model. |
| [`TVAEOptions<T>`](/docs/reference/wiki/models/tvaeoptions/) | Configuration options for TVAE (Tabular Variational Autoencoder), a VAE-based model for generating realistic synthetic tabular data. |
| [`TabDDPMOptions<T>`](/docs/reference/wiki/models/tabddpmoptions/) | Configuration options for TabDDPM (Tabular Denoising Diffusion Probabilistic Model), a diffusion-based model for generating realistic synthetic tabular data. |
| [`TabDPTOptions<T>`](/docs/reference/wiki/models/tabdptoptions/) | Configuration options for TabDPT (Tabular Data Pre-Training). |
| [`TabFlowOptions<T>`](/docs/reference/wiki/models/tabflowoptions/) | Configuration options for TabFlow, a flow matching model for generating synthetic tabular data using continuous normalizing flows with optimal transport paths. |
| [`TabLLMGenOptions<T>`](/docs/reference/wiki/models/tabllmgenoptions/) | Configuration options for TabLLM-Gen, an LLM-style tabular data generator that uses schema-aware tokenization and autoregressive transformers with column prompts. |
| [`TabMOptions<T>`](/docs/reference/wiki/models/tabmoptions/) | Configuration options for TabM, a parameter-efficient ensemble model for tabular data. |
| [`TabNetOptions<T>`](/docs/reference/wiki/models/tabnetoptions/) | Configuration options for TabNet, an attention-based interpretable deep learning model for tabular data. |
| [`TabPFNOptions<T>`](/docs/reference/wiki/models/tabpfnoptions/) | Configuration options for TabPFN (Prior-Fitted Networks for Tabular Data). |
| [`TabROptions<T>`](/docs/reference/wiki/models/tabroptions/) | Configuration options for TabR, a retrieval-augmented model for tabular data. |
| [`TabSynOptions<T>`](/docs/reference/wiki/models/tabsynoptions/) | Configuration options for TabSyn, a state-of-the-art synthetic tabular data generator that combines a VAE with latent diffusion for high-quality generation. |
| [`TabTransformerGenOptions<T>`](/docs/reference/wiki/models/tabtransformergenoptions/) | Configuration options for TabTransformer-Gen, a generative model that uses contextual embeddings from multi-head self-attention over categorical columns to generate realistic tabular data. |
| [`TabTransformerOptions<T>`](/docs/reference/wiki/models/tabtransformeroptions/) | Configuration options for TabTransformer. |
| [`TableGANOptions<T>`](/docs/reference/wiki/models/tableganoptions/) | Configuration options for TableGAN, a DCGAN-style generative adversarial network for synthesizing tabular data with classification and information loss regularization. |
| [`TabuSearchOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/tabusearchoptions/) | Configuration options for Tabu Search, a metaheuristic optimization algorithm that enhances local search by using memory structures to avoid revisiting previously explored solutions. |
| [`TabularActorCriticOptions<T>`](/docs/reference/wiki/models/tabularactorcriticoptions/) | Configuration options for Tabular Actor-Critic agents. |
| [`TabularQLearningOptions<T>`](/docs/reference/wiki/models/tabularqlearningoptions/) | Configuration options for Tabular Q-Learning agents. |
| [`Tacotron2ModelOptions`](/docs/reference/wiki/models/tacotron2modeloptions/) | Configuration options for Tacotron 2 text-to-speech models. |
| [`TeeAttestationOptions`](/docs/reference/wiki/models/teeattestationoptions/) | Configuration options for TEE remote attestation. |
| [`TeeOptions`](/docs/reference/wiki/models/teeoptions/) | Configuration options for Trusted Execution Environment integration in federated learning. |
| [`TemporalFusionTransformerOptions<T>`](/docs/reference/wiki/models/temporalfusiontransformeroptions/) | Configuration options for the Temporal Fusion Transformer (TFT) model. |
| [`TemporalGCNOptions<T>`](/docs/reference/wiki/models/temporalgcnoptions/) | Configuration options for TemporalGCN (Temporal Graph Convolutional Network). |
| [`ThompsonSamplingOptions<T>`](/docs/reference/wiki/models/thompsonsamplingoptions/) |  |
| [`TiDEOptions<T>`](/docs/reference/wiki/models/tideoptions/) | Options for `TiDEModel` — Time-series Dense Encoder (Das et al., "Long-term Forecasting with TiDE: Time-series Dense Encoder", TMLR 2023). |
| [`TimeBridgeOptions<T>`](/docs/reference/wiki/models/timebridgeoptions/) | Configuration options for TimeBridge (Non-Stationarity Matters for Time Series Foundation Models). |
| [`TimeDiffOptions<T>`](/docs/reference/wiki/models/timediffoptions/) | Configuration options for TimeDiff (Non-autoregressive Diffusion-based Time Series Forecasting). |
| [`TimeGANOptions<T>`](/docs/reference/wiki/models/timeganoptions/) | Configuration options for TimeGAN, a generative adversarial network designed specifically for generating realistic time-series data while preserving temporal dynamics. |
| [`TimeGPTOptions<T>`](/docs/reference/wiki/models/timegptoptions/) | Configuration options for TimeGPT-style time series forecasting model. |
| [`TimeGradOptions<T>`](/docs/reference/wiki/models/timegradoptions/) | Configuration options for TimeGrad (Autoregressive Denoising Diffusion Model for Time Series). |
| [`TimeLLMOptions<T>`](/docs/reference/wiki/models/timellmoptions/) | Configuration options for Time-LLM (Large Language Model Reprogramming for Time Series). |
| [`TimeMAEOptions<T>`](/docs/reference/wiki/models/timemaeoptions/) | Configuration options for TimeMAE (Masked Autoencoder for Time Series). |
| [`TimeMachineOptions<T>`](/docs/reference/wiki/models/timemachineoptions/) | Configuration options for TimeMachine (Time Series State Space Model). |
| [`TimeMoEOptions<T>`](/docs/reference/wiki/models/timemoeoptions/) | Configuration options for Time-MoE (Billion-Scale Time Series Foundation Models with Mixture of Experts). |
| [`TimeSeriesClassifierOptions<T>`](/docs/reference/wiki/models/timeseriesclassifieroptions/) | Configuration options for time series classifiers. |
| [`TimeSeriesCrossValidationFitDetectorOptions`](/docs/reference/wiki/models/timeseriescrossvalidationfitdetectoroptions/) | Configuration options for detecting overfitting, underfitting, and model stability in time series models using cross-validation techniques. |
| [`TimeSeriesFeatureOptions`](/docs/reference/wiki/models/timeseriesfeatureoptions/) | Configuration options for time series feature extraction transformers. |
| [`TimeSeriesForestOptions<T>`](/docs/reference/wiki/models/timeseriesforestoptions/) | Configuration options for the Time Series Forest classifier. |
| [`TimeSeriesIsolationForestOptions<T>`](/docs/reference/wiki/models/timeseriesisolationforestoptions/) | Configuration options for Time Series Isolation Forest anomaly detection. |
| [`TimeSeriesRegressionOptions<T>`](/docs/reference/wiki/models/timeseriesregressionoptions/) | Configuration options for time series regression models, which analyze data collected over time to identify patterns and make predictions. |
| [`TimerOptions<T>`](/docs/reference/wiki/models/timeroptions/) | Configuration options for Timer (Generative Pre-Training for Time Series). |
| [`TimesFMOptions<T>`](/docs/reference/wiki/models/timesfmoptions/) | Configuration options for the TimesFM (Time Series Foundation Model). |
| [`TimesNetOptions<T>`](/docs/reference/wiki/models/timesnetoptions/) | Configuration options for the TimesNet model. |
| [`TinyTimeMixersOptions<T>`](/docs/reference/wiki/models/tinytimemixersoptions/) | Configuration options for Tiny Time Mixers (TTM) foundation model. |
| [`TradingAgentOptions<T>`](/docs/reference/wiki/models/tradingagentoptions/) | Configuration options for financial trading agents. |
| [`TrainingPipelineConfiguration<T, TInput, TOutput>`](/docs/reference/wiki/models/trainingpipelineconfiguration/) | Configuration for a multi-step training pipeline with customizable stages. |
| [`TransferFunctionOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/transferfunctionoptions/) | Configuration options for Transfer Function models, which model the dynamic relationship between input (exogenous) and output (endogenous) time series. |
| [`TrustRegionOptimizerOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/trustregionoptimizeroptions/) | Configuration options for Trust Region optimization algorithms, which are robust methods for solving nonlinear optimization problems. |
| [`TweedieRegressionOptions<T>`](/docs/reference/wiki/models/tweedieregressionoptions/) | Configuration options for Tweedie Regression, a flexible generalized linear model that encompasses several distributions (Poisson, Gamma, Inverse Gaussian) as special cases. |
| [`UCBBanditOptions<T>`](/docs/reference/wiki/models/ucbbanditoptions/) | Configuration options for the UCB bandit agent. |
| [`UncertaintyQuantificationOptions`](/docs/reference/wiki/models/uncertaintyquantificationoptions/) | Configuration options for enabling uncertainty quantification during inference. |
| [`UniTSOptions<T>`](/docs/reference/wiki/models/unitsoptions/) | Configuration options for UniTS (Unified Time Series Model). |
| [`UnobservedComponentsOptions<T, TInput, TOutput>`](/docs/reference/wiki/models/unobservedcomponentsoptions/) | Configuration options for Unobserved Components Models (UCM), which decompose time series into trend, seasonal, cycle, and irregular components. |
| [`VARMAModelOptions<T>`](/docs/reference/wiki/models/varmamodeloptions/) | Configuration options for Vector Autoregressive Moving Average (VARMA) models, which extend VAR models by incorporating moving average terms. |
| [`VARModelOptions<T>`](/docs/reference/wiki/models/varmodeloptions/) | Configuration options for Vector Autoregressive (VAR) models, which model the linear interdependencies among multiple time series. |
| [`VIFFitDetectorOptions`](/docs/reference/wiki/models/viffitdetectoroptions/) | Configuration options for detecting multicollinearity in regression models using Variance Inflation Factor (VIF) analysis. |
| [`VITSModelOptions`](/docs/reference/wiki/models/vitsmodeloptions/) | Configuration options for VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech) models. |
| [`VaROptions<T>`](/docs/reference/wiki/models/varoptions/) | Configuration options for Value-at-Risk (VaR) models. |
| [`ValueIterationOptions<T>`](/docs/reference/wiki/models/valueiterationoptions/) | Configuration options for Value Iteration agents. |
| [`VerificationOptions`](/docs/reference/wiki/models/verificationoptions/) | Configuration options for zero-knowledge verification in federated learning. |
| [`VerticalFederatedLearningOptions`](/docs/reference/wiki/models/verticalfederatedlearningoptions/) | Top-level configuration for vertical federated learning (VFL). |
| [`VflUnlearningOptions`](/docs/reference/wiki/models/vflunlearningoptions/) | Configuration for GDPR-compliant entity unlearning in vertical federated learning. |
| [`VisionTSOptions<T>`](/docs/reference/wiki/models/visiontsoptions/) | Configuration options for VisionTS (Visual Masked Autoencoders as Zero-Shot Time Series Forecasters). |
| [`WatkinsQLambdaOptions<T>`](/docs/reference/wiki/models/watkinsqlambdaoptions/) |  |
| [`Wav2Vec2ModelOptions`](/docs/reference/wiki/models/wav2vec2modeloptions/) | Configuration options for Wav2Vec2 speech recognition models. |
| [`WaveNetOptions<T>`](/docs/reference/wiki/models/wavenetoptions/) | Configuration options for the WaveNet model adapted for time series forecasting. |
| [`WeightedRegressionOptions<T>`](/docs/reference/wiki/models/weightedregressionoptions/) | Configuration options for weighted regression models, which assign different importance to different observations. |
| [`WorldModelsOptions<T>`](/docs/reference/wiki/models/worldmodelsoptions/) | Configuration options for World Models agents. |
| [`YingLongOptions<T>`](/docs/reference/wiki/models/yinglongoptions/) | Configuration options for YingLong (Alibaba's Enterprise Time Series Foundation Model). |
| [`ZeroInflatedRegressionOptions`](/docs/reference/wiki/models/zeroinflatedregressionoptions/) | Configuration options for Zero-Inflated regression models. |


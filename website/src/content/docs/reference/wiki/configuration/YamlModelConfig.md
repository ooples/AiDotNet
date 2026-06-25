---
title: "YamlModelConfig"
description: "Root POCO that YAML/JSON config files deserialize into."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Root POCO that YAML/JSON config files deserialize into.
Each section maps to either an enum-based factory selection or a direct config POCO.

## For Beginners

This class represents the structure of a YAML configuration file
that can be used to set up an AI model builder. Instead of writing C# code for every setting,
you can define your configuration in a YAML file and load it automatically.

## How It Works

Supported sections:

## Properties

| Property | Summary |
|:-----|:--------|
| `AbTesting` | A/B testing configuration for comparing model versions. |
| `ActivationFunction` | YAML configuration for ConfigureActivationFunction(). |
| `ActiveLearner` | YAML configuration for the ActiveLearner section. |
| `ActiveLearning` | YAML configuration for ConfigureActiveLearning(). |
| `ActiveLearningStrategy` | YAML configuration for the ActiveLearningStrategy section. |
| `AdaptedMetaModel` | YAML configuration for the AdaptedMetaModel section. |
| `AdaptiveDistillationStrategy` | YAML configuration for the AdaptiveDistillationStrategy section. |
| `AdversarialAttack` | YAML configuration for ConfigureAdversarialAttack(). |
| `AdversarialDefense` | YAML configuration for ConfigureAdversarialDefense(). |
| `AdversarialRobustness` | YAML configuration for ConfigureAdversarialRobustness(). |
| `AlignmentMethod` | YAML configuration for the AlignmentMethod section. |
| `AnomalyDetector` | YAML configuration for ConfigureAnomalyDetector(). |
| `AssociativeMemory` | YAML configuration for the AssociativeMemory section. |
| `AsyncTreeBasedModel` | YAML configuration for the AsyncTreeBasedModel section. |
| `AudioCodec` | YAML configuration for the AudioCodec section. |
| `AudioDiffusionModel` | YAML configuration for the AudioDiffusionModel section. |
| `AudioEffect` | YAML configuration for ConfigureAudioEffect(). |
| `AudioEnhancer` | YAML configuration for ConfigureAudioEnhancer(). |
| `AudioEventDetector` | YAML configuration for the AudioEventDetector section. |
| `AudioFeatureExtractor` | YAML configuration for the AudioFeatureExtractor section. |
| `AudioFingerprinter` | YAML configuration for the AudioFingerprinter section. |
| `AudioFoundationModel` | YAML configuration for the AudioFoundationModel section. |
| `AudioGenerator` | YAML configuration for ConfigureAudioGenerator(). |
| `AudioLanguageModel` | YAML configuration for the AudioLanguageModel section. |
| `AudioVisualCorrespondenceModel` | YAML configuration for the AudioVisualCorrespondenceModel section. |
| `AudioVisualEventLocalizationModel` | YAML configuration for the AudioVisualEventLocalizationModel section. |
| `Augmentation` | YAML configuration for ConfigureAugmentation(). |
| `AutoML` | YAML configuration for ConfigureAutoML(). |
| `AutoMLModel` | YAML configuration for the AutoMLModel section. |
| `AuxiliaryLossLayer` | YAML configuration for the AuxiliaryLossLayer section. |
| `BatchStrategy` | YAML configuration for the BatchStrategy section. |
| `BayesianLayer` | YAML configuration for the BayesianLayer section. |
| `BayesianStrategy` | YAML configuration for the BayesianStrategy section. |
| `BeatTracker` | YAML configuration for the BeatTracker section. |
| `Benchmark` | YAML configuration for ConfigureBenchmark(). |
| `Benchmarking` | Benchmarking configuration for running standardized benchmark suites. |
| `BiasDetector` | YAML configuration for ConfigureBiasDetector(). |
| `Blip2Model` | YAML configuration for the Blip2Model section. |
| `BlipModel` | YAML configuration for the BlipModel section. |
| `Caching` | Model caching configuration for storing loaded models in memory. |
| `CausalDiscovery` | YAML configuration for ConfigureCausalDiscovery(). |
| `CausalInference` | YAML configuration for ConfigureCausalInference(). |
| `CausalModel` | YAML configuration for the CausalModel section. |
| `CertifiedDefense` | YAML configuration for ConfigureCertifiedDefense(). |
| `CheckpointManager` | YAML configuration for ConfigureCheckpointManager(). |
| `ClassificationMetric` | YAML configuration for ConfigureClassificationMetric(). |
| `Classifier` | YAML configuration for ConfigureClassifier(). |
| `ClassifierComparisonTest` | YAML configuration for the ClassifierComparisonTest section. |
| `ClusterMetric` | YAML configuration for ConfigureClusterMetric(). |
| `Clustering` | YAML configuration for ConfigureClustering(). |
| `ClusteringBatchStrategy` | YAML configuration for the ClusteringBatchStrategy section. |
| `CodeModel` | YAML configuration for the CodeModel section. |
| `CommitteeStrategy` | YAML configuration for the CommitteeStrategy section. |
| `CommunicationBackend` | YAML configuration for the CommunicationBackend section. |
| `CompetenceBasedScheduler` | YAML configuration for the CompetenceBasedScheduler section. |
| `CompositeCriterion` | YAML configuration for the CompositeCriterion section. |
| `Compression` | Model compression configuration for reducing model size. |
| `CompressionMetadata` | YAML configuration for the CompressionMetadata section. |
| `ConditioningModule` | YAML configuration for the ConditioningModule section. |
| `ConfidentDifficultyEstimator` | YAML configuration for the ConfidentDifficultyEstimator section. |
| `ContentClassifier` | YAML configuration for the ContentClassifier section. |
| `ContextCompressor` | YAML configuration for the ContextCompressor section. |
| `ContextFlow` | YAML configuration for the ContextFlow section. |
| `ContinualDistillationStrategy` | YAML configuration for the ContinualDistillationStrategy section. |
| `ContinualLearner` | YAML configuration for the ContinualLearner section. |
| `ContinualLearnerConfig` | YAML configuration for the ContinualLearnerConfig section. |
| `ContinualLearning` | YAML configuration for ConfigureContinualLearning(). |
| `ContinualLearningStrategy` | YAML configuration for the ContinualLearningStrategy section. |
| `CrossValidation` | YAML configuration for ConfigureCrossValidation(). |
| `CrossValidationStrategy` | YAML configuration for the CrossValidationStrategy section. |
| `CurriculumDistillationStrategy` | YAML configuration for the CurriculumDistillationStrategy section. |
| `CurriculumLearner` | YAML configuration for the CurriculumLearner section. |
| `CurriculumLearnerConfig` | YAML configuration for the CurriculumLearnerConfig section. |
| `CurriculumLearnerConfigBuilder` | YAML configuration for the CurriculumLearnerConfigBuilder section. |
| `CurriculumLearning` | YAML configuration for ConfigureCurriculumLearning(). |
| `CurriculumScheduler` | YAML configuration for ConfigureCurriculumScheduler(). |
| `DallE3Model` | YAML configuration for the DallE3Model section. |
| `DataLoader` | YAML configuration for ConfigureDataLoader(). |
| `DataPreparation` | YAML configuration for ConfigureDataPreparation(). |
| `DataSplitter` | YAML configuration for ConfigureDataSplitter(). |
| `DataTransformer` | YAML configuration for ConfigureDataTransformer(). |
| `DataVersionControl` | YAML configuration for ConfigureDataVersionControl(). |
| `DecisionFunctionClassifier` | YAML configuration for the DecisionFunctionClassifier section. |
| `DetachedTensor` | YAML configuration for the DetachedTensor section. |
| `DifficultyEstimator` | YAML configuration for the DifficultyEstimator section. |
| `DiffusionModel` | YAML configuration for ConfigureDiffusionModel(). |
| `DistanceMetric` | YAML configuration for ConfigureDistanceMetric(). |
| `DistillationStrategy` | YAML configuration for ConfigureDistillationStrategy(). |
| `DistributedTraining` | YAML configuration for ConfigureDistributedTraining(). |
| `DiversityStrategy` | YAML configuration for the DiversityStrategy section. |
| `DocumentClassifier` | YAML configuration for the DocumentClassifier section. |
| `DocumentModel` | YAML configuration for ConfigureDocumentModel(). |
| `DocumentQA` | YAML configuration for the DocumentQA section. |
| `DocumentStore` | YAML configuration for ConfigureDocumentStore(). |
| `DomainAdapter` | YAML configuration for the DomainAdapter section. |
| `DomainDecompositionTrainingHistory` | YAML configuration for the DomainDecompositionTrainingHistory section. |
| `DriftDetection` | YAML configuration for ConfigureDriftDetection(). |
| `DriftDetector` | YAML configuration for the DriftDetector section. |
| `DriftDetectorStandalone` | YAML configuration for the DriftDetectorStandalone section. |
| `EmbeddingModel` | YAML configuration for ConfigureEmbeddingModel(). |
| `EmotionRecognizer` | YAML configuration for the EmotionRecognizer section. |
| `EnsembleDifficultyEstimator` | YAML configuration for the EnsembleDifficultyEstimator section. |
| `Environment` | YAML configuration for ConfigureEnvironment(). |
| `EpisodicDataLoader` | YAML configuration for the EpisodicDataLoader section. |
| `ExperimentRun` | YAML configuration for the ExperimentRun section. |
| `ExperimentTracker` | YAML configuration for ConfigureExperimentTracker(). |
| `ExplorationStrategy` | YAML configuration for ConfigureExplorationStrategy(). |
| `Export` | Export configuration for exporting models to different formats. |
| `ExternalClusterMetric` | YAML configuration for ConfigureExternalClusterMetric(). |
| `FactorModel` | YAML configuration for the FactorModel section. |
| `FairnessEvaluator` | YAML configuration for ConfigureFairnessEvaluator(). |
| `FeatureImportance` | YAML configuration for the FeatureImportance section. |
| `FeatureMapper` | YAML configuration for the FeatureMapper section. |
| `FederatedClientDataLoader` | YAML configuration for the FederatedClientDataLoader section. |
| `FederatedHeterogeneityCorrection` | YAML configuration for the FederatedHeterogeneityCorrection section. |
| `FederatedLearning` | YAML configuration for ConfigureFederatedLearning(). |
| `FederatedServerOptimizer` | YAML configuration for the FederatedServerOptimizer section. |
| `FewShotExampleSelector` | YAML configuration for the FewShotExampleSelector section. |
| `FinancialModel` | YAML configuration for ConfigureFinancialModel(). |
| `FinancialNLPModel` | YAML configuration for the FinancialNLPModel section. |
| `FineTuning` | YAML configuration for ConfigureFineTuning(). |
| `FitDetector` | YAML configuration for ConfigureFitDetector(). |
| `FitnessCalculator` | YAML configuration for ConfigureFitnessCalculator(). |
| `FlamingoModel` | YAML configuration for the FlamingoModel section. |
| `ForecastingModel` | YAML configuration for the ForecastingModel section. |
| `FormUnderstanding` | YAML configuration for the FormUnderstanding section. |
| `GPUAcceleratedExplainer` | YAML configuration for the GPUAcceleratedExplainer section. |
| `GaussianProcess` | YAML configuration for ConfigureGaussianProcess(). |
| `GaussianProcessClassifier` | YAML configuration for the GaussianProcessClassifier section. |
| `Generator` | YAML configuration for the Generator section. |
| `GenreClassifier` | YAML configuration for the GenreClassifier section. |
| `Gpt4VisionModel` | YAML configuration for the Gpt4VisionModel section. |
| `GpuAcceleration` | GPU acceleration configuration for hardware-accelerated training and inference. |
| `GpuDiagnostics` | YAML configuration for ConfigureGpuDiagnostics(). |
| `GradientBasedOptimizer` | YAML configuration for the GradientBasedOptimizer section. |
| `GradientBatchStrategy` | YAML configuration for the GradientBatchStrategy section. |
| `GradientCache` | YAML configuration for the GradientCache section. |
| `GradientComputable` | YAML configuration for the GradientComputable section. |
| `GradientConstraintStrategy` | YAML configuration for the GradientConstraintStrategy section. |
| `GradientModel` | YAML configuration for the GradientModel section. |
| `GraphConvolutionLayer` | YAML configuration for the GraphConvolutionLayer section. |
| `GraphDataLoader` | YAML configuration for the GraphDataLoader section. |
| `GraphStore` | YAML configuration for the GraphStore section. |
| `HyperparameterOptimizer` | YAML configuration for ConfigureHyperparameterOptimizer(). |
| `ImageBindModel` | YAML configuration for the ImageBindModel section. |
| `InferenceOptimizations` | Inference optimization configuration for KV caching, batching, and speculative decoding. |
| `InitializationStrategy` | YAML configuration for the InitializationStrategy section. |
| `InputGradientComputable` | YAML configuration for the InputGradientComputable section. |
| `InputOutputDataLoader` | YAML configuration for the InputOutputDataLoader section. |
| `InstanceSegmentation` | YAML configuration for the InstanceSegmentation section. |
| `InstanceSegmenter` | YAML configuration for the InstanceSegmenter section. |
| `IntermediateActivationStrategy` | YAML configuration for the IntermediateActivationStrategy section. |
| `Interpolation` | YAML configuration for ConfigureInterpolation(). |
| `Interpolation2D` | YAML configuration for ConfigureInterpolation2D(). |
| `Interpretability` | Interpretability configuration for model explainability (SHAP, LIME, etc.). |
| `InterpretableModel` | YAML configuration for the InterpretableModel section. |
| `JitCompilation` | JIT compilation configuration — toggles `TensorCodecOptions.EnableCompilation` plus fusion/CSE phase flags on the returned model. |
| `KernelFunction` | YAML configuration for ConfigureKernelFunction(). |
| `KnowledgeDistillation` | YAML configuration for ConfigureKnowledgeDistillation(). |
| `KnowledgeDistillationTrainer` | YAML configuration for the KnowledgeDistillationTrainer section. |
| `KnowledgeGraph` | YAML configuration for ConfigureKnowledgeGraph(). |
| `LLaVAModel` | YAML configuration for the LLaVAModel section. |
| `LanguageIdentifier` | YAML configuration for the LanguageIdentifier section. |
| `LatentDiffusionModel` | YAML configuration for the LatentDiffusionModel section. |
| `Layer` | YAML configuration for ConfigureLayer(). |
| `LayoutDetector` | YAML configuration for the LayoutDetector section. |
| `LearningRateScheduler` | YAML configuration for ConfigureLearningRateScheduler(). |
| `License` | License key configuration for encrypted model loading/saving and online validation. |
| `LicenseKey` | YAML configuration for ConfigureLicenseKey(). |
| `Likelihood` | YAML configuration for the Likelihood section. |
| `LinkFunction` | YAML configuration for ConfigureLinkFunction(). |
| `LoRA` | YAML configuration for ConfigureLoRA(). |
| `LossFunction` | YAML configuration for ConfigureLossFunction(). |
| `MatrixDecomposition` | YAML configuration for ConfigureMatrixDecomposition(). |
| `MeanFunction` | YAML configuration for the MeanFunction section. |
| `MedicalSegmentation` | YAML configuration for the MedicalSegmentation section. |
| `MemoryBank` | YAML configuration for the MemoryBank section. |
| `MemoryBasedStrategy` | YAML configuration for the MemoryBasedStrategy section. |
| `MemoryManagement` | Training memory management configuration (gradient checkpointing, activation pooling, model sharding). |
| `MetaLearnerOptions` | YAML configuration for the MetaLearnerOptions section. |
| `MetaLearning` | YAML configuration for ConfigureMetaLearning(). |
| `MetaLearningTask` | YAML configuration for the MetaLearningTask section. |
| `Metric` | YAML configuration for the Metric section. |
| `MixedPrecision` | Mixed precision training configuration. |
| `ModelCache` | YAML configuration for the ModelCache section. |
| `ModelCompressionStrategy` | YAML configuration for ConfigureModelCompressionStrategy(). |
| `ModelExplainer` | YAML configuration for ConfigureModelExplainer(). |
| `ModelExporter` | YAML configuration for the ModelExporter section. |
| `ModelOptions` | YAML configuration for ConfigureModelOptions(). |
| `ModelRegistry` | YAML configuration for ConfigureModelRegistry(). |
| `MomentumEncoder` | YAML configuration for the MomentumEncoder section. |
| `MultiFidelityTrainingHistory` | YAML configuration for the MultiFidelityTrainingHistory section. |
| `MultiLabelClassifier` | YAML configuration for the MultiLabelClassifier section. |
| `MultiObjectiveIndividual` | YAML configuration for the MultiObjectiveIndividual section. |
| `MultimodalEmbedding` | YAML configuration for the MultimodalEmbedding section. |
| `MultipleComparisonTest` | YAML configuration for the MultipleComparisonTest section. |
| `MultipleSampleTest` | YAML configuration for the MultipleSampleTest section. |
| `MusicSourceSeparator` | YAML configuration for the MusicSourceSeparator section. |
| `MusicTranscriber` | YAML configuration for the MusicTranscriber section. |
| `NERModel` | YAML configuration for the NERModel section. |
| `NTMController` | YAML configuration for the NTMController section. |
| `NeuralNetwork` | YAML configuration for the NeuralNetwork section. |
| `NeuralNetworkModel` | YAML configuration for the NeuralNetworkModel section. |
| `NoisePredictor` | YAML configuration for the NoisePredictor section. |
| `NoiseScheduler` | YAML configuration for ConfigureNoiseScheduler(). |
| `NonLinearRegression` | YAML configuration for the NonLinearRegression section. |
| `OCRModel` | YAML configuration for the OCRModel section. |
| `ObjectDetector` | YAML configuration for the ObjectDetector section. |
| `ObjectTracker` | YAML configuration for the ObjectTracker section. |
| `OnlineClassifier` | YAML configuration for the OnlineClassifier section. |
| `OnlineLearning` | YAML configuration for ConfigureOnlineLearning(). |
| `OnlineLearningModel` | YAML configuration for the OnlineLearningModel section. |
| `OnnxModel` | YAML configuration for the OnnxModel section. |
| `OpenVocabSegmentation` | YAML configuration for the OpenVocabSegmentation section. |
| `Optimizer` | Optimizer selection section. |
| `OrdinalClassifier` | YAML configuration for the OrdinalClassifier section. |
| `OutlierRemoval` | YAML configuration for the OutlierRemoval section. |
| `PDEResidualGradient` | YAML configuration for the PDEResidualGradient section. |
| `PDESpecification` | YAML configuration for ConfigurePDESpecification(). |
| `PageSegmenter` | YAML configuration for the PageSegmenter section. |
| `PairedTest` | YAML configuration for the PairedTest section. |
| `PanopticSegmentation` | YAML configuration for the PanopticSegmentation section. |
| `Parameterizable` | YAML configuration for the Parameterizable section. |
| `PipelineParallelism` | YAML configuration for ConfigurePipelineParallelism(). |
| `PitchDetector` | YAML configuration for the PitchDetector section. |
| `PlanCaching` | YAML configuration for ConfigurePlanCaching(). |
| `PointCloudClassification` | YAML configuration for the PointCloudClassification section. |
| `PointCloudModel` | YAML configuration for ConfigurePointCloudModel(). |
| `PointCloudSegmentation` | YAML configuration for the PointCloudSegmentation section. |
| `PortfolioOptimizer` | YAML configuration for the PortfolioOptimizer section. |
| `Postprocessing` | YAML configuration for ConfigurePostprocessing(). |
| `Postprocessor` | YAML configuration for the Postprocessor section. |
| `Preprocessing` | YAML configuration for ConfigurePreprocessing(). |
| `ProbabilisticClassificationMetric` | YAML configuration for the ProbabilisticClassificationMetric section. |
| `ProbabilisticClassifier` | YAML configuration for the ProbabilisticClassifier section. |
| `Profiling` | Performance profiling configuration. |
| `ProgramExecutionEngine` | YAML configuration for the ProgramExecutionEngine section. |
| `ProgramSynthesis` | YAML configuration for ConfigureProgramSynthesis(). |
| `ProgramSynthesisServing` | YAML configuration for ConfigureProgramSynthesisServing(). |
| `ProgramSynthesizer` | YAML configuration for the ProgramSynthesizer section. |
| `ProjectorHead` | YAML configuration for the ProjectorHead section. |
| `PromptAnalyzer` | YAML configuration for the PromptAnalyzer section. |
| `PromptChain` | YAML configuration for the PromptChain section. |
| `PromptCompressor` | YAML configuration for the PromptCompressor section. |
| `PromptOptimizer` | YAML configuration for the PromptOptimizer section. |
| `PromptTemplate` | YAML configuration for the PromptTemplate section. |
| `PromptableSegmentation` | YAML configuration for the PromptableSegmentation section. |
| `PruningMask` | YAML configuration for the PruningMask section. |
| `PruningStrategy` | YAML configuration for the PruningStrategy section. |
| `Quantization` | Model quantization configuration for compressing models with lower precision. |
| `Quantizer` | YAML configuration for the Quantizer section. |
| `QueryStrategy` | YAML configuration for ConfigureQueryStrategy(). |
| `RAGMetric` | YAML configuration for the RAGMetric section. |
| `RLAgent` | YAML configuration for ConfigureRLAgent(). |
| `RLDataLoader` | YAML configuration for the RLDataLoader section. |
| `RLPolicy` | YAML configuration for the RLPolicy section. |
| `RadialBasisFunction` | YAML configuration for ConfigureRadialBasisFunction(). |
| `RadianceField` | YAML configuration for the RadianceField section. |
| `ReadingOrderDetector` | YAML configuration for the ReadingOrderDetector section. |
| `Reasoning` | Reasoning strategy configuration. |
| `ReasoningStrategy` | YAML configuration for the ReasoningStrategy section. |
| `ReferringSegmentation` | YAML configuration for the ReferringSegmentation section. |
| `Regression` | YAML configuration for ConfigureRegression(). |
| `RegressionMetric` | YAML configuration for ConfigureRegressionMetric(). |
| `Regularization` | YAML configuration for ConfigureRegularization(). |
| `ReinforcementLearning` | YAML configuration for ConfigureReinforcementLearning(). |
| `Reranker` | YAML configuration for the Reranker section. |
| `ResamplingStrategy` | YAML configuration for the ResamplingStrategy section. |
| `RetrievalAugmentedGeneration` | YAML configuration for ConfigureRetrievalAugmentedGeneration(). |
| `Retriever` | YAML configuration for the Retriever section. |
| `RiskModel` | YAML configuration for the RiskModel section. |
| `RowOperation` | YAML configuration for the RowOperation section. |
| `SSLMethod` | YAML configuration for ConfigureSSLMethod(). |
| `Safety` | YAML configuration for ConfigureSafety(). |
| `SafetyFilter` | YAML configuration for the SafetyFilter section. |
| `SceneClassifier` | YAML configuration for the SceneClassifier section. |
| `ScoringRule` | YAML configuration for ConfigureScoringRule(). |
| `SegmentationModel` | YAML configuration for the SegmentationModel section. |
| `SegmentationVisualization` | YAML configuration for ConfigureSegmentationVisualization(). |
| `SelfPacedScheduler` | YAML configuration for the SelfPacedScheduler section. |
| `SelfSupervisedLearning` | YAML configuration for ConfigureSelfSupervisedLearning(). |
| `SelfSupervisedLoss` | YAML configuration for the SelfSupervisedLoss section. |
| `SemanticSegmentation` | YAML configuration for the SemanticSegmentation section. |
| `SemiSupervisedClassifier` | YAML configuration for the SemiSupervisedClassifier section. |
| `SequenceLossFunction` | YAML configuration for the SequenceLossFunction section. |
| `ShardedModel` | YAML configuration for the ShardedModel section. |
| `ShardedOptimizer` | YAML configuration for the ShardedOptimizer section. |
| `ShardingConfiguration` | YAML configuration for the ShardingConfiguration section. |
| `SimilarityMetric` | YAML configuration for ConfigureSimilarityMetric(). |
| `SpeakerDiarizer` | YAML configuration for the SpeakerDiarizer section. |
| `SpeakerEmbeddingExtractor` | YAML configuration for the SpeakerEmbeddingExtractor section. |
| `SpeakerVerifier` | YAML configuration for the SpeakerVerifier section. |
| `SpeechRecognizer` | YAML configuration for ConfigureSpeechRecognizer(). |
| `StatefulDataLoader` | YAML configuration for the StatefulDataLoader section. |
| `StatisticalTest` | YAML configuration for the StatisticalTest section. |
| `StoppingCriterion` | YAML configuration for ConfigureStoppingCriterion(). |
| `StreamingDataLoader` | YAML configuration for the StreamingDataLoader section. |
| `SubmodularBatchStrategy` | YAML configuration for the SubmodularBatchStrategy section. |
| `SurvivalAnalysis` | YAML configuration for ConfigureSurvivalAnalysis(). |
| `SurvivalModel` | YAML configuration for the SurvivalModel section. |
| `SyntheticTabularGenerator` | YAML configuration for the SyntheticTabularGenerator section. |
| `TableExtractor` | YAML configuration for the TableExtractor section. |
| `TargetScaling` | YAML configuration for ConfigureTargetScaling(). |
| `TeacherModel` | YAML configuration for the TeacherModel section. |
| `Telemetry` | Telemetry configuration for tracking inference metrics. |
| `TextDetector` | YAML configuration for the TextDetector section. |
| `TextRecognizer` | YAML configuration for the TextRecognizer section. |
| `TextToSpeech` | YAML configuration for ConfigureTextToSpeech(). |
| `TextVectorizer` | YAML configuration for ConfigureTextVectorizer(). |
| `ThreeDDiffusionModel` | YAML configuration for the ThreeDDiffusionModel section. |
| `TimeSeriesClassifier` | YAML configuration for the TimeSeriesClassifier section. |
| `TimeSeriesDecomposition` | YAML configuration for ConfigureTimeSeriesDecomposition(). |
| `TimeSeriesFeatureExtractor` | YAML configuration for the TimeSeriesFeatureExtractor section. |
| `TimeSeriesFoundationModel` | YAML configuration for the TimeSeriesFoundationModel section. |
| `TimeSeriesMetric` | YAML configuration for the TimeSeriesMetric section. |
| `TimeSeriesModel` | Time series model selection section. |
| `TokenEmbedding` | YAML configuration for the TokenEmbedding section. |
| `Tokenizer` | YAML configuration for the Tokenizer section. |
| `Tool` | YAML configuration for ConfigureTool(). |
| `TradingAgent` | YAML configuration for the TradingAgent section. |
| `Trainer` | YAML configuration for the Trainer section. |
| `TrainingGroups` | YAML configuration for ConfigureTrainingGroups(). |
| `TrainingMonitor` | YAML configuration for ConfigureTrainingMonitor(). |
| `TrainingPipeline` | YAML configuration for ConfigureTrainingPipeline(). |
| `Transform` | YAML configuration for the Transform section. |
| `TreeBasedClassifier` | YAML configuration for the TreeBasedClassifier section. |
| `TreeBasedRegression` | YAML configuration for the TreeBasedRegression section. |
| `TwoDInterpolation` | YAML configuration for the TwoDInterpolation section. |
| `TwoSampleTest` | YAML configuration for the TwoSampleTest section. |
| `UncertaintyBasedCriterion` | YAML configuration for the UncertaintyBasedCriterion section. |
| `UncertaintyEstimator` | YAML configuration for the UncertaintyEstimator section. |
| `UncertaintyQuantification` | YAML configuration for ConfigureUncertaintyQuantification(). |
| `UncertaintyStrategy` | YAML configuration for the UncertaintyStrategy section. |
| `UnderSampler` | YAML configuration for the UnderSampler section. |
| `UnifiedMultimodalModel` | YAML configuration for the UnifiedMultimodalModel section. |
| `VAEModel` | YAML configuration for the VAEModel section. |
| `VectorActivationFunction` | YAML configuration for the VectorActivationFunction section. |
| `VectorIndex` | YAML configuration for the VectorIndex section. |
| `Versioning` | Model versioning configuration for managing multiple model versions. |
| `VideoCLIPModel` | YAML configuration for the VideoCLIPModel section. |
| `VideoDiffusionModel` | YAML configuration for the VideoDiffusionModel section. |
| `VideoModel` | YAML configuration for ConfigureVideoModel(). |
| `VideoSegmentation` | YAML configuration for the VideoSegmentation section. |
| `VoiceActivityDetector` | YAML configuration for the VoiceActivityDetector section. |
| `VolatilityModel` | YAML configuration for the VolatilityModel section. |
| `WaveletFunction` | YAML configuration for ConfigureWaveletFunction(). |
| `WeightLoadable` | YAML configuration for the WeightLoadable section. |
| `WeightStreaming` | YAML configuration for ConfigureWeightStreaming(). |
| `WeightedSampler` | YAML configuration for the WeightedSampler section. |
| `WindowFunction` | YAML configuration for ConfigureWindowFunction(). |


---
title: "Interfaces"
description: "All 377 public types in the AiDotNet.interfaces namespace, organized by kind."
section: "API Reference"
---

**377** public types in this namespace, organized by kind.

## Models & Types (87)

| Type | Summary |
|:-----|:--------|
| [`AcousticCharacteristics<T>`](/docs/reference/wiki/interfaces/acousticcharacteristics/) | Acoustic characteristics of a scene. |
| [`AudioEffectParameter<T>`](/docs/reference/wiki/interfaces/audioeffectparameter/) | Represents an adjustable parameter for an audio effect. |
| [`AudioEventResult<T>`](/docs/reference/wiki/interfaces/audioeventresult/) | Result of audio event detection. |
| [`AudioEvent<T>`](/docs/reference/wiki/interfaces/audioevent/) | A detected audio event. |
| [`AudioFingerprint<T>`](/docs/reference/wiki/interfaces/audiofingerprint/) | Represents an audio fingerprint. |
| [`AudioVisualEvent`](/docs/reference/wiki/interfaces/audiovisualevent/) | Represents an audio-visual event with temporal boundaries. |
| [`BeatTrackingResult<T>`](/docs/reference/wiki/interfaces/beattrackingresult/) | Result of beat tracking. |
| [`ChordRecognitionResult<T>`](/docs/reference/wiki/interfaces/chordrecognitionresult/) | Result of chord recognition. |
| [`ChordSegment<T>`](/docs/reference/wiki/interfaces/chordsegment/) | A segment of audio with a detected chord. |
| [`ChordStatistics<T>`](/docs/reference/wiki/interfaces/chordstatistics/) | Statistics for a chord in the recognition result. |
| [`Contradiction`](/docs/reference/wiki/interfaces/contradiction/) | Represents a detected contradiction between reasoning steps. |
| [`CritiqueResult<T>`](/docs/reference/wiki/interfaces/critiqueresult/) | Result of critiquing a reasoning step or chain. |
| [`DataLoaderCheckpoint`](/docs/reference/wiki/interfaces/dataloadercheckpoint/) | Serializable checkpoint for a stateful data loader. |
| [`DatasetMetadata`](/docs/reference/wiki/interfaces/datasetmetadata/) | Metadata about a dataset. |
| [`DiarizationResult<T>`](/docs/reference/wiki/interfaces/diarizationresult/) | Result of speaker diarization. |
| [`DirectionEstimate<T>`](/docs/reference/wiki/interfaces/directionestimate/) | A direction estimate without full source information. |
| [`DownbeatResult<T>`](/docs/reference/wiki/interfaces/downbeatresult/) | Result of downbeat detection. |
| [`EmotionResult<T>`](/docs/reference/wiki/interfaces/emotionresult/) | Represents the result of emotion recognition. |
| [`EpisodeResult<T>`](/docs/reference/wiki/interfaces/episoderesult/) | Result of running a single RL episode. |
| [`EventStatistics<T>`](/docs/reference/wiki/interfaces/eventstatistics/) | Statistics for an event type. |
| [`FewShotExample`](/docs/reference/wiki/interfaces/fewshotexample/) | Represents a single few-shot example with input and output. |
| [`FingerprintMatch`](/docs/reference/wiki/interfaces/fingerprintmatch/) | Represents a match between two fingerprints. |
| [`FrequencyRange<T>`](/docs/reference/wiki/interfaces/frequencyrange/) | A frequency range. |
| [`GeneticParameters`](/docs/reference/wiki/interfaces/geneticparameters/) | Parameters for configuring a genetic algorithm. |
| [`GenreClassificationResult<T>`](/docs/reference/wiki/interfaces/genreclassificationresult/) | Result of genre classification. |
| [`GenreFeatures<T>`](/docs/reference/wiki/interfaces/genrefeatures/) | Features extracted for genre classification (generic version). |
| [`GenrePrediction<T>`](/docs/reference/wiki/interfaces/genreprediction/) | A single genre prediction with confidence. |
| [`GenreSegment<T>`](/docs/reference/wiki/interfaces/genresegment/) | A segment with genre information. |
| [`GenreTrackingResult<T>`](/docs/reference/wiki/interfaces/genretrackingresult/) | Result of tracking genre over time. |
| [`GpuOptimizerState`](/docs/reference/wiki/interfaces/gpuoptimizerstate/) | Holds the optimizer state buffers for GPU-resident training. |
| [`IncrementalState<T>`](/docs/reference/wiki/interfaces/incrementalstate/) | Represents the internal state of a time series transformer for incremental processing. |
| [`KeyDetectionResult<T>`](/docs/reference/wiki/interfaces/keydetectionresult/) | Result of key detection. |
| [`KeyHypothesis<T>`](/docs/reference/wiki/interfaces/keyhypothesis/) | A key hypothesis with confidence score. |
| [`KeySegment<T>`](/docs/reference/wiki/interfaces/keysegment/) | A segment with a detected key. |
| [`KeyTrackingResult<T>`](/docs/reference/wiki/interfaces/keytrackingresult/) | Result of key tracking over time. |
| [`LanguageResult<T>`](/docs/reference/wiki/interfaces/languageresult/) | Represents the result of language identification. |
| [`LanguageSegment<T>`](/docs/reference/wiki/interfaces/languagesegment/) | Represents a time-segmented language detection result. |
| [`LayerInfo<T>`](/docs/reference/wiki/interfaces/layerinfo/) | Metadata about a single layer within a layered model, including its position in the flat parameter vector and computational characteristics. |
| [`LocalizationResult<T>`](/docs/reference/wiki/interfaces/localizationresult/) | Result of sound localization. |
| [`MedicalSegmentationResult<T>`](/docs/reference/wiki/interfaces/medicalsegmentationresult/) | Result of medical image segmentation. |
| [`Mesh3D<T>`](/docs/reference/wiki/interfaces/mesh3d/) | Represents a 3D mesh with vertices, faces, and optional textures. |
| [`ModelDownloadProgress`](/docs/reference/wiki/interfaces/modeldownloadprogress/) | Progress information for model downloads. |
| [`ModulationPoint<T>`](/docs/reference/wiki/interfaces/modulationpoint/) | A point where the key changes (modulation). |
| [`MultimodalInput<T>`](/docs/reference/wiki/interfaces/multimodalinput/) | Represents an input item for unified multimodal models. |
| [`MultimodalOutput<T>`](/docs/reference/wiki/interfaces/multimodaloutput/) | Represents an output from unified multimodal models. |
| [`OpenVocabSegmentationResult<T>`](/docs/reference/wiki/interfaces/openvocabsegmentationresult/) | Result of open-vocabulary segmentation. |
| [`OptimizationHistoryEntry<T>`](/docs/reference/wiki/interfaces/optimizationhistoryentry/) | Represents a single entry in the optimization history. |
| [`OverlapRegion<T>`](/docs/reference/wiki/interfaces/overlapregion/) | Represents a region where speakers overlap. |
| [`PanopticSegment<T>`](/docs/reference/wiki/interfaces/panopticsegment/) | A single segment in a panoptic segmentation result. |
| [`PanopticSegmentationResult<T>`](/docs/reference/wiki/interfaces/panopticsegmentationresult/) | Result of panoptic segmentation containing both semantic and instance information. |
| [`PipelineOperation`](/docs/reference/wiki/interfaces/pipelineoperation/) | Represents a single operation in the pipeline schedule. |
| [`PitchFrame<T>`](/docs/reference/wiki/interfaces/pitchframe/) | Represents a single pitch detection frame. |
| [`PromptedSegmentationResult<T>`](/docs/reference/wiki/interfaces/promptedsegmentationresult/) | Result from a prompted segmentation operation containing one or more mask proposals. |
| [`ReasoningContext`](/docs/reference/wiki/interfaces/reasoningcontext/) | Context information for critiquing reasoning steps. |
| [`ReferringSegmentationResult<T>`](/docs/reference/wiki/interfaces/referringsegmentationresult/) | Result of referring segmentation. |
| [`ResamplingStatistics<T>`](/docs/reference/wiki/interfaces/resamplingstatistics/) | Contains statistics about a resampling operation. |
| [`SceneClassificationResult<T>`](/docs/reference/wiki/interfaces/sceneclassificationresult/) | Result of scene classification. |
| [`SceneFeatures<T>`](/docs/reference/wiki/interfaces/scenefeatures/) | Features extracted for scene classification (generic version). |
| [`ScenePrediction<T>`](/docs/reference/wiki/interfaces/sceneprediction/) | A single scene prediction with confidence. |
| [`SceneSegment<T>`](/docs/reference/wiki/interfaces/scenesegment/) | A segment with scene information. |
| [`SceneTrackingResult<T>`](/docs/reference/wiki/interfaces/scenetrackingresult/) | Result of tracking scene changes over time. |
| [`SceneTransition<T>`](/docs/reference/wiki/interfaces/scenetransition/) | A detected scene transition. |
| [`SegmentedStructure`](/docs/reference/wiki/interfaces/segmentedstructure/) | Metadata about a single segmented medical structure. |
| [`SeparationQuality<T>`](/docs/reference/wiki/interfaces/separationquality/) | Quality metrics for source separation. |
| [`SoundSource<T>`](/docs/reference/wiki/interfaces/soundsource/) | A detected sound source with position. |
| [`SoundTrackingResult<T>`](/docs/reference/wiki/interfaces/soundtrackingresult/) | Result of sound source tracking over time. |
| [`SourceSeparationResult<T>`](/docs/reference/wiki/interfaces/sourceseparationresult/) | Result of source separation. |
| [`SourceTrajectory<T>`](/docs/reference/wiki/interfaces/sourcetrajectory/) | A tracked trajectory of a sound source. |
| [`SpeakerProfile<T>`](/docs/reference/wiki/interfaces/speakerprofile/) | Represents an enrolled speaker profile. |
| [`SpeakerSegment<T>`](/docs/reference/wiki/interfaces/speakersegment/) | Represents a speaker segment in diarization output. |
| [`SpeakerStatistics<T>`](/docs/reference/wiki/interfaces/speakerstatistics/) | Statistics for a speaker in diarization output. |
| [`SpeakerTurn<T>`](/docs/reference/wiki/interfaces/speakerturn/) | Represents a speaker turn in diarization output (legacy API compatibility). |
| [`SpeakerVerificationResult<T>`](/docs/reference/wiki/interfaces/speakerverificationresult/) | Result of a speaker verification attempt. |
| [`SubModel<T>`](/docs/reference/wiki/interfaces/submodel/) | Represents a contiguous sub-model extracted from a larger `ILayeredModel`. |
| [`TempoHypothesis<T>`](/docs/reference/wiki/interfaces/tempohypothesis/) | A tempo hypothesis with confidence score. |
| [`TimeSignature`](/docs/reference/wiki/interfaces/timesignature/) | Represents a musical time signature. |
| [`TimedEmotionResult<T>`](/docs/reference/wiki/interfaces/timedemotionresult/) | Represents a timed emotion prediction. |
| [`TrajectoryPoint<T>`](/docs/reference/wiki/interfaces/trajectorypoint/) | A single point in a source trajectory. |
| [`TranscribedNote<T>`](/docs/reference/wiki/interfaces/transcribednote/) | Represents a single transcribed musical note. |
| [`TranscriptionResult<T>`](/docs/reference/wiki/interfaces/transcriptionresult/) | Represents the result of a transcription operation. |
| [`TranscriptionSegment<T>`](/docs/reference/wiki/interfaces/transcriptionsegment/) | Represents a segment of transcribed text with timing information. |
| [`TransformerState<T>`](/docs/reference/wiki/interfaces/transformerstate/) | Represents the serializable state of a fitted time series transformer. |
| [`VerificationResult<T>`](/docs/reference/wiki/interfaces/verificationresult/) | Represents the result of external tool verification. |
| [`VideoSegmentationResult<T>`](/docs/reference/wiki/interfaces/videosegmentationresult/) | Result of video segmentation for a single frame. |
| [`VoiceInfo<T>`](/docs/reference/wiki/interfaces/voiceinfo/) | Information about an available TTS voice. |
| [`WeightLoadResult`](/docs/reference/wiki/interfaces/weightloadresult/) | Result of weight loading operation. |
| [`WeightLoadValidation`](/docs/reference/wiki/interfaces/weightloadvalidation/) | Result of weight validation. |

## Interfaces (256)

| Type | Summary |
|:-----|:--------|
| [`I2DInterpolation<T>`](/docs/reference/wiki/interfaces/i2dinterpolation/) | Defines an interface for two-dimensional interpolation algorithms. |
| [`I3DDiffusionModel<T>`](/docs/reference/wiki/interfaces/i3ddiffusionmodel/) | Interface for 3D diffusion models that generate 3D content like point clouds, meshes, and scenes. |
| [`IActivationFunction<T>`](/docs/reference/wiki/interfaces/iactivationfunction/) | Defines an interface for activation functions used in neural networks and other machine learning algorithms. |
| [`IActiveLearningStrategy<T>`](/docs/reference/wiki/interfaces/iactivelearningstrategy/) | Defines a strategy for active learning that selects the most informative samples for labeling from a pool of unlabeled data. |
| [`IAdaptedMetaModel<T>`](/docs/reference/wiki/interfaces/iadaptedmetamodel/) | Extended interface for meta-learning adapted models that carry task-specific adaptation state beyond backbone parameters. |
| [`IAdversarialAttack<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/iadversarialattack/) | Defines the contract for adversarial attack algorithms that generate adversarial examples. |
| [`IAdversarialDefense<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/iadversarialdefense/) | Defines the contract for adversarial defense mechanisms that protect models against attacks. |
| [`IAggregationStrategy<TModel>`](/docs/reference/wiki/interfaces/iaggregationstrategy/) | Defines strategies for aggregating model updates from multiple clients in federated learning. |
| [`IAiModelBuilder<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/iaimodelbuilder/) | Defines a builder pattern interface for creating and configuring predictive models. |
| [`IAlignmentMethod<T>`](/docs/reference/wiki/interfaces/ialignmentmethod/) | Defines the contract for AI alignment methods that ensure models behave according to human values and intentions. |
| [`IAnomalyDetector<T>`](/docs/reference/wiki/interfaces/ianomalydetector/) | Defines methods for algorithmic anomaly/outlier detection using a fit-predict pattern. |
| [`IAssociativeMemory<T>`](/docs/reference/wiki/interfaces/iassociativememory/) | Interface for Associative Memory modules used in nested learning. |
| [`IAsyncTreeBasedModel<T>`](/docs/reference/wiki/interfaces/iasynctreebasedmodel/) | Defines an interface for asynchronous tree-based machine learning models. |
| [`IAudioCodec<T>`](/docs/reference/wiki/interfaces/iaudiocodec/) | Defines the contract for neural audio codecs that compress and decompress audio. |
| [`IAudioDiffusionModel<T>`](/docs/reference/wiki/interfaces/iaudiodiffusionmodel/) | Interface for audio diffusion models that generate sound and music. |
| [`IAudioEffect<T>`](/docs/reference/wiki/interfaces/iaudioeffect/) | Defines the contract for audio effects processors. |
| [`IAudioEnhancer<T>`](/docs/reference/wiki/interfaces/iaudioenhancer/) | Defines the contract for audio enhancement models that improve audio quality. |
| [`IAudioEventDetector<T>`](/docs/reference/wiki/interfaces/iaudioeventdetector/) | Interface for audio event detection models that identify specific sounds/events in audio. |
| [`IAudioFeatureExtractor<T>`](/docs/reference/wiki/interfaces/iaudiofeatureextractor/) | Defines the contract for audio feature extraction algorithms. |
| [`IAudioFingerprinter<T>`](/docs/reference/wiki/interfaces/iaudiofingerprinter/) | Interface for audio fingerprinting algorithms. |
| [`IAudioFoundationModel<T>`](/docs/reference/wiki/interfaces/iaudiofoundationmodel/) | Defines the contract for self-supervised audio foundation models. |
| [`IAudioGenerator<T>`](/docs/reference/wiki/interfaces/iaudiogenerator/) | Interface for audio generation models that create audio from text descriptions or other conditions. |
| [`IAudioLanguageModel<T>`](/docs/reference/wiki/interfaces/iaudiolanguagemodel/) | Interface for multimodal audio-language models that understand and reason about audio. |
| [`IAudioSafetyModule<T>`](/docs/reference/wiki/interfaces/iaudiosafetymodule/) | Interface for safety modules that operate on audio content. |
| [`IAudioVisualCorrespondenceModel<T>`](/docs/reference/wiki/interfaces/iaudiovisualcorrespondencemodel/) | Defines the contract for audio-visual correspondence learning models. |
| [`IAudioVisualEventLocalizationModel<T>`](/docs/reference/wiki/interfaces/iaudiovisualeventlocalizationmodel/) | Defines the contract for audio-visual event localization models. |
| [`IAutoMLModel<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/iautomlmodel/) | Defines the contract for AutoML models that automatically search for optimal model configurations. |
| [`IAutoregressiveMultimodalModel<T>`](/docs/reference/wiki/interfaces/iautoregressivemultimodalmodel/) | Defines the contract for autoregressive multimodal generation models that can generate tokens from any modality in an interleaved fashion. |
| [`IAuxiliaryLossLayer<T>`](/docs/reference/wiki/interfaces/iauxiliarylosslayer/) | Interface for neural network layers that report auxiliary losses in addition to the primary task loss. |
| [`IBatchIterable<TBatch>`](/docs/reference/wiki/interfaces/ibatchiterable/) | Defines capability to iterate through data in batches. |
| [`IBatchSampler`](/docs/reference/wiki/interfaces/ibatchsampler/) | Extended interface for samplers that support batch-level sampling. |
| [`IBeatTracker<T>`](/docs/reference/wiki/interfaces/ibeattracker/) | Interface for beat tracking models that detect tempo and beat positions in audio. |
| [`IBenchmark<T>`](/docs/reference/wiki/interfaces/ibenchmark/) | Defines the contract for reasoning benchmarks that evaluate model performance. |
| [`IBiasDetector<T>`](/docs/reference/wiki/interfaces/ibiasdetector/) | Defines an interface for detecting bias in machine learning model predictions. |
| [`IBlip2Model<T>`](/docs/reference/wiki/interfaces/iblip2model/) | Defines the contract for BLIP-2 (Bootstrapped Language-Image Pre-training 2) models. |
| [`IBlipModel<T>`](/docs/reference/wiki/interfaces/iblipmodel/) | Defines the contract for BLIP (Bootstrapped Language-Image Pre-training) models. |
| [`ICausalModel<T>`](/docs/reference/wiki/interfaces/icausalmodel/) | Interface for causal inference models (meta-learners). |
| [`ICertifiedDefense<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/icertifieddefense/) | Defines the contract for certified defense mechanisms that provide provable robustness guarantees. |
| [`ICheckpointManager<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/icheckpointmanager/) | Defines the contract for checkpoint management systems that save and restore training state. |
| [`ICheckpointableModel`](/docs/reference/wiki/interfaces/icheckpointablemodel/) | Defines the contract for models that support saving and loading their internal state (checkpointing). |
| [`IChordRecognizer<T>`](/docs/reference/wiki/interfaces/ichordrecognizer/) | Interface for chord recognition models that identify musical chords in audio. |
| [`IChunkingStrategy`](/docs/reference/wiki/interfaces/ichunkingstrategy/) | Defines the contract for text chunking strategies that split documents into smaller segments. |
| [`IClassifier<T>`](/docs/reference/wiki/interfaces/iclassifier/) | Defines the common interface for all classification algorithms in the AiDotNet library. |
| [`IClientModel<TData, TUpdate>`](/docs/reference/wiki/interfaces/iclientmodel/) | Defines the functionality for a client-side model in federated learning. |
| [`IClientSelectionStrategy`](/docs/reference/wiki/interfaces/iclientselectionstrategy/) | Selects which clients participate in a federated learning round. |
| [`ICloneable<T>`](/docs/reference/wiki/interfaces/icloneable/) | Interface for objects that can be cloned or copied. |
| [`ICompressionMetadata<T>`](/docs/reference/wiki/interfaces/icompressionmetadata/) | Defines the contract for compression metadata that stores information needed to decompress model weights. |
| [`IConditioningModule<T>`](/docs/reference/wiki/interfaces/iconditioningmodule/) | Interface for conditioning modules that encode various inputs into embeddings for diffusion models. |
| [`IContextCompressor<T>`](/docs/reference/wiki/interfaces/icontextcompressor/) | Defines the contract for compressing context documents to reduce token usage while preserving relevance. |
| [`IContextFlow<T>`](/docs/reference/wiki/interfaces/icontextflow/) | Interface for Context Flow mechanism - maintains distinct information pathways and update rates for each nested optimization level. |
| [`IContinualLearningStrategy<T>`](/docs/reference/wiki/interfaces/icontinuallearningstrategy/) | Defines a strategy for continual learning that helps neural networks learn multiple tasks sequentially without forgetting previously learned knowledge. |
| [`IContrastiveLoss<T>`](/docs/reference/wiki/interfaces/icontrastiveloss/) | Interface for contrastive and self-supervised loss functions that operate on pairs of embeddings/representations rather than predictions vs ground truth labels. |
| [`ICountable`](/docs/reference/wiki/interfaces/icountable/) | Defines capability to report dataset size and iteration progress. |
| [`ICrossValidator<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/icrossvalidator/) | Defines the contract for cross-validation implementations in machine learning models. |
| [`IDallE3Model<T>`](/docs/reference/wiki/interfaces/idalle3model/) | Defines the contract for DALL-E 3-style text-to-image generation models. |
| [`IDataLoader<T>`](/docs/reference/wiki/interfaces/idataloader/) | Base interface for all data loaders providing common data loading capabilities. |
| [`IDataSampler`](/docs/reference/wiki/interfaces/idatasampler/) | Defines the contract for sampling indices from a dataset during batch iteration. |
| [`IDataTransformer<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/idatatransformer/) | Defines a data transformer that can fit to data and transform it. |
| [`IDataVersionControl<T>`](/docs/reference/wiki/interfaces/idataversioncontrol/) | Defines the contract for data version control systems that track dataset changes over time. |
| [`IDatasetFactory<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/idatasetfactory/) | Factory for creating datasets. |
| [`IDataset<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/idataset/) | Interface for datasets used in active learning scenarios. |
| [`IDecisionFunctionClassifier<T>`](/docs/reference/wiki/interfaces/idecisionfunctionclassifier/) | Interface for classifiers that compute a decision function for predictions. |
| [`IDetectionBackbone<T>`](/docs/reference/wiki/interfaces/idetectionbackbone/) | Marker interface for detection backbones. |
| [`IDiagnosticsProvider`](/docs/reference/wiki/interfaces/idiagnosticsprovider/) | Interface for components that provide diagnostic information for monitoring and debugging. |
| [`IDiffusionModel<T>`](/docs/reference/wiki/interfaces/idiffusionmodel/) | Interface for diffusion-based generative models. |
| [`IDisposableGaussianProcess<T>`](/docs/reference/wiki/interfaces/idisposablegaussianprocess/) | A Gaussian Process whose implementation owns native or unmanaged resources (e.g., cached Cholesky factors, GPU memory) and must be released deterministically. |
| [`IDistillationStrategy<T>`](/docs/reference/wiki/interfaces/idistillationstrategy/) | Defines a strategy for computing knowledge distillation loss between student and teacher models. |
| [`IDocumentStore<T>`](/docs/reference/wiki/interfaces/idocumentstore/) | Defines the contract for document stores that index and retrieve vectorized documents. |
| [`IDomainReasoner<T>`](/docs/reference/wiki/interfaces/idomainreasoner/) | Interface for domain-specific reasoning models that solve problems using LLM-based reasoning strategies (chain-of-thought, tree search, consensus). |
| [`IDownloadable`](/docs/reference/wiki/interfaces/idownloadable/) | Defines capability to automatically download and cache datasets. |
| [`IDriftDetector<T>`](/docs/reference/wiki/interfaces/idriftdetector/) | Defines the interface for concept drift detection. |
| [`IEmbeddingModel<T>`](/docs/reference/wiki/interfaces/iembeddingmodel/) | Defines the contract for embedding models that convert text into vector representations. |
| [`IEmotionRecognizer<T>`](/docs/reference/wiki/interfaces/iemotionrecognizer/) | Defines the contract for speech emotion recognition models. |
| [`IEnvironment<T>`](/docs/reference/wiki/interfaces/ienvironment/) | Represents a reinforcement learning environment that an agent interacts with. |
| [`IEpisode<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/iepisode/) | Represents a single episode in meta-learning, wrapping an `IMetaLearningTask` with additional metadata such as domain, difficulty, and timing information. |
| [`IEpisodicDataLoader<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/iepisodicdataloader/) | Interface for data loaders that provide episodic tasks for meta-learning. |
| [`IEpisodicDataset<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/iepisodicdataset/) | Interface for episodic datasets used in meta-learning. |
| [`IEvolvable<TGene, T>`](/docs/reference/wiki/interfaces/ievolvable/) | Represents an individual that can evolve through genetic operations. |
| [`IExperiment`](/docs/reference/wiki/interfaces/iexperiment/) | Represents a machine learning experiment that groups related training runs. |
| [`IExperimentRun<T>`](/docs/reference/wiki/interfaces/iexperimentrun/) | Represents a single training run within an experiment. |
| [`IExperimentTracker<T>`](/docs/reference/wiki/interfaces/iexperimenttracker/) | Defines the contract for experiment tracking systems that log machine learning experiments. |
| [`IExtendedDataset<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/iextendeddataset/) | Extended dataset interface with additional metadata and features. |
| [`IFairnessEvaluator<T>`](/docs/reference/wiki/interfaces/ifairnessevaluator/) | Defines an interface for evaluating fairness in machine learning models. |
| [`IFeatureAware`](/docs/reference/wiki/interfaces/ifeatureaware/) | Interface for models that can provide information about their feature usage. |
| [`IFeatureImportance<T>`](/docs/reference/wiki/interfaces/ifeatureimportance/) | Interface for models that can provide feature importance scores. |
| [`IFeatureMapProvider<T>`](/docs/reference/wiki/interfaces/ifeaturemapprovider/) | Mixin interface for neural networks that produce multi-scale feature pyramids — typically detection / segmentation backbones (ResNet, CSPDarknet, EfficientNet, SwinTransformer) whose outputs feed FPN, PAN, anchor generators, or DETR-style t… |
| [`IFederatedClientDataLoader<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/ifederatedclientdataloader/) | Represents a data loader that can provide per-client datasets for federated learning. |
| [`IFederatedHeterogeneityCorrection<T>`](/docs/reference/wiki/interfaces/ifederatedheterogeneitycorrection/) | Applies a heterogeneity correction transform to client updates in federated learning. |
| [`IFederatedServerOptimizer<T>`](/docs/reference/wiki/interfaces/ifederatedserveroptimizer/) | Applies a server-side optimization step in federated learning (FedOpt family). |
| [`IFederatedTrainer<TModel, TData, TMetadata>`](/docs/reference/wiki/interfaces/ifederatedtrainer/) | Defines the core functionality for federated learning trainers that coordinate distributed training across multiple clients. |
| [`IFewShotExampleSelector<T>`](/docs/reference/wiki/interfaces/ifewshotexampleselector/) | Defines the contract for selecting few-shot examples to include in prompts. |
| [`IFineTuning<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/ifinetuning/) | Defines the contract for fine-tuning methods that adapt pre-trained models to specific tasks or preferences. |
| [`IFitDetector<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/ifitdetector/) | Defines an interface for detecting how well a machine learning model fits the data. |
| [`IFitnessCalculator<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/ifitnesscalculator/) | Defines an interface for calculating how well a machine learning model performs. |
| [`IFlamingoModel<T>`](/docs/reference/wiki/interfaces/iflamingomodel/) | Defines the contract for Flamingo-style models with in-context visual learning capabilities. |
| [`IFullModel<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/ifullmodel/) | Represents a complete machine learning model that combines prediction capabilities with serialization and checkpointing support. |
| [`IFunctionOptimizer<T>`](/docs/reference/wiki/interfaces/ifunctionoptimizer/) | Interface for optimizing a scalar-valued function over a vector of parameters. |
| [`IGaussianProcessClassifier<T>`](/docs/reference/wiki/interfaces/igaussianprocessclassifier/) | Defines an interface for Gaussian Process classification, a probabilistic approach to classification that provides uncertainty estimates along with class predictions. |
| [`IGaussianProcess<T>`](/docs/reference/wiki/interfaces/igaussianprocess/) | Defines an interface for Gaussian Process regression, a powerful probabilistic machine learning technique. |
| [`IGenerator<T>`](/docs/reference/wiki/interfaces/igenerator/) | Defines the contract for text generation models used in retrieval-augmented generation. |
| [`IGeneticAlgorithm<T, TInput, TOutput, TIndividual, TGene>`](/docs/reference/wiki/interfaces/igeneticalgorithm/) | Represents a machine learning model that uses genetic algorithms or evolutionary computation while maintaining the core capabilities of a full model. |
| [`IGenreClassifier<T>`](/docs/reference/wiki/interfaces/igenreclassifier/) |  |
| [`IGlobalExplainer<T, TExplanation>`](/docs/reference/wiki/interfaces/iglobalexplainer/) | Interface for explainers that provide global (model-wide) explanations. |
| [`IGpt4VisionModel<T>`](/docs/reference/wiki/interfaces/igpt4visionmodel/) | Defines the contract for GPT-4V-style models that combine vision understanding with large language model capabilities. |
| [`IGpuOptimizerConfig`](/docs/reference/wiki/interfaces/igpuoptimizerconfig/) | Configuration for GPU-resident optimizer updates. |
| [`IGradientBasedOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/igradientbasedoptimizer/) |  |
| [`IGradientCache<T>`](/docs/reference/wiki/interfaces/igradientcache/) | Defines an interface for storing and retrieving pre-computed gradients to improve performance in machine learning models. |
| [`IGradientComputable<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/igradientcomputable/) | Base interface for models that can compute gradients explicitly without updating parameters. |
| [`IGradientModel<T>`](/docs/reference/wiki/interfaces/igradientmodel/) | Represents a gradient for optimization algorithms. |
| [`IGraphConvolutionLayer<T>`](/docs/reference/wiki/interfaces/igraphconvolutionlayer/) | Defines the contract for graph convolutional layers that process graph-structured data. |
| [`IGraphDataLoader<T>`](/docs/reference/wiki/interfaces/igraphdataloader/) | Interface for data loaders that provide graph-structured data for graph neural networks. |
| [`IGraphStore<T>`](/docs/reference/wiki/interfaces/igraphstore/) | Defines the contract for graph storage backends that manage nodes and edges. |
| [`IHomomorphicEncryptionProvider<T>`](/docs/reference/wiki/interfaces/ihomomorphicencryptionprovider/) | Provides homomorphic encryption operations for federated learning aggregation. |
| [`IHyperparameterOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/ihyperparameteroptimizer/) | Defines the contract for hyperparameter optimization algorithms. |
| [`IImageBindModel<T>`](/docs/reference/wiki/interfaces/iimagebindmodel/) | Defines the contract for ImageBind models that bind multiple modalities (6+) into a shared embedding space. |
| [`IImageSafetyModule<T>`](/docs/reference/wiki/interfaces/iimagesafetymodule/) | Interface for safety modules that operate on image content. |
| [`IInputGradientComputable<T>`](/docs/reference/wiki/interfaces/iinputgradientcomputable/) | Interface for models that support computing gradients with respect to input data. |
| [`IInputOutputDataLoader<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/iinputoutputdataloader/) | Interface for data loaders that provide standard input-output (X, Y) data for supervised learning. |
| [`IInstanceSegmentation<T>`](/docs/reference/wiki/interfaces/iinstancesegmentation/) | Interface for instance segmentation models that detect and mask individual object instances. |
| [`IIntermediateActivationStrategy<T>`](/docs/reference/wiki/interfaces/iintermediateactivationstrategy/) | Defines methods for distillation strategies that utilize intermediate layer activations. |
| [`IInterpolation<T>`](/docs/reference/wiki/interfaces/iinterpolation/) | Defines an interface for interpolation algorithms that estimate values between known data points. |
| [`IInterpretableModel<T>`](/docs/reference/wiki/interfaces/iinterpretablemodel/) | Interface for models that support interpretability features. |
| [`IKernelFunction<T>`](/docs/reference/wiki/interfaces/ikernelfunction/) | Defines an interface for kernel functions that measure similarity between data points in machine learning algorithms. |
| [`IKeyDetector<T>`](/docs/reference/wiki/interfaces/ikeydetector/) | Interface for musical key detection models that identify the key and mode of music. |
| [`IKnowledgeDistillationTrainer<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/iknowledgedistillationtrainer/) | Defines the contract for knowledge distillation trainers that train student models using knowledge transferred from teacher models. |
| [`ILLaVAModel<T>`](/docs/reference/wiki/interfaces/illavamodel/) | Defines the contract for LLaVA (Large Language and Vision Assistant) models. |
| [`ILanguageIdentifier<T>`](/docs/reference/wiki/interfaces/ilanguageidentifier/) | Defines the contract for spoken language identification from audio. |
| [`ILatentDiffusionModel<T>`](/docs/reference/wiki/interfaces/ilatentdiffusionmodel/) | Interface for latent diffusion models that operate in a compressed latent space. |
| [`ILayer<T>`](/docs/reference/wiki/interfaces/ilayer/) | Defines the contract for neural network layers in the AiDotNet framework. |
| [`ILayeredModel<T>`](/docs/reference/wiki/interfaces/ilayeredmodel/) | Provides layer-level access to a neural network's architecture and parameters. |
| [`ILinearRegression<T>`](/docs/reference/wiki/interfaces/ilinearregression/) | Defines an interface for linear regression in machine learning, which predict outputs as a weighted sum of inputs plus an optional constant. |
| [`ILinkFunction<T>`](/docs/reference/wiki/interfaces/ilinkfunction/) | Interface for link functions used in Generalized Linear Models (GLMs). |
| [`ILoRAAdapter<T>`](/docs/reference/wiki/interfaces/iloraadapter/) | Interface for LoRA (Low-Rank Adaptation) adapters that wrap existing layers with parameter-efficient adaptations. |
| [`ILoRAConfiguration<T>`](/docs/reference/wiki/interfaces/iloraconfiguration/) | Interface for configuring how LoRA (Low-Rank Adaptation) should be applied to neural network layers. |
| [`ILocalExplainer<T, TExplanation>`](/docs/reference/wiki/interfaces/ilocalexplainer/) | Interface for explainers that provide local (per-instance) explanations. |
| [`ILossFunction<T>`](/docs/reference/wiki/interfaces/ilossfunction/) | Interface for loss functions used in neural networks. |
| [`IMatrixDecomposition<T>`](/docs/reference/wiki/interfaces/imatrixdecomposition/) | Represents a matrix decomposition that can be used to solve linear systems and invert matrices. |
| [`IMedicalSegmentation<T>`](/docs/reference/wiki/interfaces/imedicalsegmentation/) | Interface for medical image segmentation models that handle 2D slices and 3D volumetric data. |
| [`IMetaDataset<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/imetadataset/) | Represents a high-level meta-dataset that can generate episodes for meta-learning. |
| [`IMetaLearnerOptions<T>`](/docs/reference/wiki/interfaces/imetalearneroptions/) | Configuration options interface for meta-learning algorithms. |
| [`IMetaLearner<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/imetalearner/) | Unified interface for meta-learning algorithms that train models to quickly adapt to new tasks. |
| [`IMetaLearningTask<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/imetalearningtask/) | Represents a single meta-learning task for few-shot learning. |
| [`IModelCache<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/imodelcache/) | Defines a caching mechanism for storing and retrieving optimization step data during model training. |
| [`IModelCompressionStrategy<T>`](/docs/reference/wiki/interfaces/imodelcompressionstrategy/) | Defines an interface for model compression strategies used to reduce model size while preserving accuracy. |
| [`IModelCompression<T, TMetadata>`](/docs/reference/wiki/interfaces/imodelcompression/) | Defines a type-safe interface for model compression used to reduce model size while preserving accuracy. |
| [`IModelExplainer<T>`](/docs/reference/wiki/interfaces/imodelexplainer/) | Interface for model-agnostic explainers that can explain any predictive model's decisions. |
| [`IModelRegistry<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/imodelregistry/) | Defines the contract for model registry systems that manage trained model storage and versioning. |
| [`IModelSerializer`](/docs/reference/wiki/interfaces/imodelserializer/) | Defines methods for converting machine learning models to and from binary data for storage or transmission. |
| [`IModelShape`](/docs/reference/wiki/interfaces/imodelshape/) | Provides shape metadata for a machine learning model, describing its expected input and output dimensions. |
| [`IModel<TInput, TOutput, TMetadata>`](/docs/reference/wiki/interfaces/imodel/) | Defines the core functionality for machine learning models that can be trained on data and make predictions. |
| [`IMultiLabelClassifier<T>`](/docs/reference/wiki/interfaces/imultilabelclassifier/) | Interface for multi-label classification models. |
| [`IMultiObjectiveIndividual<T>`](/docs/reference/wiki/interfaces/imultiobjectiveindividual/) | Interface for individuals supporting multi-objective optimization. |
| [`IMultimodalEmbedding<T>`](/docs/reference/wiki/interfaces/imultimodalembedding/) | Interface for multimodal embedding models that can encode multiple modalities (text, images, audio) into a shared embedding space. |
| [`IMusicSourceSeparator<T>`](/docs/reference/wiki/interfaces/imusicsourceseparator/) | Interface for music source separation models that isolate individual instruments/vocals from a mix. |
| [`IMusicTranscriber<T>`](/docs/reference/wiki/interfaces/imusictranscriber/) | Defines the contract for automatic music transcription (audio to notes). |
| [`INeuralNetworkModel<T>`](/docs/reference/wiki/interfaces/ineuralnetworkmodel/) | Defines the contract for neural network models with advanced architectural introspection capabilities. |
| [`INeuralNetwork<T>`](/docs/reference/wiki/interfaces/ineuralnetwork/) | Defines the core functionality for neural network models in the AiDotNet library. |
| [`INoisePredictor<T>`](/docs/reference/wiki/interfaces/inoisepredictor/) | Interface for noise prediction networks used in diffusion models. |
| [`INoiseScheduler<T>`](/docs/reference/wiki/interfaces/inoisescheduler/) | Interface for diffusion model noise schedulers that control the noise schedule during inference. |
| [`INonLinearRegression<T>`](/docs/reference/wiki/interfaces/inonlinearregression/) | Defines the functionality for non-linear regression models in the AiDotNet library. |
| [`IOnlineClassifier<T>`](/docs/reference/wiki/interfaces/ionlineclassifier/) | Interface for online (incremental) classification models. |
| [`IOnlineLearningModel<T>`](/docs/reference/wiki/interfaces/ionlinelearningmodel/) | Defines the interface for online (incremental) learning models. |
| [`IOnnxModelDownloader`](/docs/reference/wiki/interfaces/ionnxmodeldownloader/) | Defines the contract for downloading ONNX models from remote sources. |
| [`IOnnxModelMetadata`](/docs/reference/wiki/interfaces/ionnxmodelmetadata/) | Metadata about an ONNX model including inputs, outputs, and opset version. |
| [`IOnnxModel<T>`](/docs/reference/wiki/interfaces/ionnxmodel/) | Defines the contract for ONNX model wrappers that provide cross-platform model inference. |
| [`IOnnxTensorInfo`](/docs/reference/wiki/interfaces/ionnxtensorinfo/) | Information about an ONNX tensor (input or output). |
| [`IOpenVocabSegmentation<T>`](/docs/reference/wiki/interfaces/iopenvocabsegmentation/) | Interface for open-vocabulary segmentation models that segment objects from text descriptions without being limited to a fixed set of classes. |
| [`IOptimizer<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/ioptimizer/) | Defines the contract for optimization algorithms used in machine learning models. |
| [`IOrdinalClassifier<T>`](/docs/reference/wiki/interfaces/iordinalclassifier/) | Interface for ordinal classification (ordinal regression) models. |
| [`IOutlierDetector<T>`](/docs/reference/wiki/interfaces/ioutlierdetector/) | Defines methods for algorithmic outlier/anomaly detection using a fit-predict pattern. |
| [`IOutlierRemoval<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/ioutlierremoval/) | Defines methods for detecting and removing outliers from datasets. |
| [`IOutputDerivative<T>`](/docs/reference/wiki/interfaces/ioutputderivative/) | Interface for activation functions that can compute their derivative given the post-activation output value rather than the pre-activation input. |
| [`IPanopticSegmentation<T>`](/docs/reference/wiki/interfaces/ipanopticsegmentation/) | Interface for panoptic segmentation models that unify semantic and instance segmentation. |
| [`IParameterizable<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/iparameterizable/) | Interface for models that have optimizable parameters. |
| [`IPipelineDecomposableModel<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/ipipelinedecomposablemodel/) | Interface for models that support decomposing the backward pass into separate activation gradient and weight gradient computations. |
| [`IPipelinePartitionStrategy<T>`](/docs/reference/wiki/interfaces/ipipelinepartitionstrategy/) | Defines a strategy for partitioning model parameters across pipeline stages. |
| [`IPipelineSchedule<T>`](/docs/reference/wiki/interfaces/ipipelineschedule/) | Defines a scheduling strategy for pipeline parallel training. |
| [`IPipelineStep<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/ipipelinestep/) | Represents a step in a data processing pipeline |
| [`IPitchDetector<T>`](/docs/reference/wiki/interfaces/ipitchdetector/) | Defines the contract for pitch (fundamental frequency) detection. |
| [`IPostprocessor<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/ipostprocessor/) | Defines a postprocessor that transforms model outputs into final results. |
| [`IPredictiveModel<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/ipredictivemodel/) | Defines the core functionality of a trained predictive model that can make predictions on new data. |
| [`IPreferredDataSetFitnessCalculator<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/ipreferreddatasetfitnesscalculator/) | Optional extension interface for fitness calculators that have a preferred dataset type. |
| [`IPrivacyAccountant`](/docs/reference/wiki/interfaces/iprivacyaccountant/) | Tracks cumulative privacy loss across federated learning rounds. |
| [`IPrivacyMechanism<TModel>`](/docs/reference/wiki/interfaces/iprivacymechanism/) | Defines privacy-preserving mechanisms for federated learning to protect client data. |
| [`IProbabilisticClassifier<T>`](/docs/reference/wiki/interfaces/iprobabilisticclassifier/) | Defines the interface for classifiers that can output probability estimates for each class. |
| [`IPromptAnalyzer`](/docs/reference/wiki/interfaces/ipromptanalyzer/) | Defines the contract for analyzing prompts before sending them to language models. |
| [`IPromptCompressor`](/docs/reference/wiki/interfaces/ipromptcompressor/) | Defines the contract for compressing prompts to reduce token counts and API costs. |
| [`IPromptOptimizer<T>`](/docs/reference/wiki/interfaces/ipromptoptimizer/) | Defines the contract for optimizing prompts to improve language model performance. |
| [`IPromptTemplate`](/docs/reference/wiki/interfaces/iprompttemplate/) | Defines the contract for prompt templates used in language model interactions. |
| [`IPromptableSegmentation<T>`](/docs/reference/wiki/interfaces/ipromptablesegmentation/) | Interface for interactive, promptable segmentation models like SAM that accept user prompts (points, boxes, masks, text) to segment specific objects. |
| [`IPruningMask<T>`](/docs/reference/wiki/interfaces/ipruningmask/) | Represents a binary mask for pruning weights in a neural network layer. |
| [`IPruningStrategy<T>`](/docs/reference/wiki/interfaces/ipruningstrategy/) | Interface for pruning strategies that remove unimportant weights to create sparsity. |
| [`IQueryProcessor`](/docs/reference/wiki/interfaces/iqueryprocessor/) | Defines the contract for processing and transforming user queries before retrieval. |
| [`IRAGMetric<T>`](/docs/reference/wiki/interfaces/iragmetric/) | Defines the contract for RAG evaluation metrics. |
| [`IRLAgent<T>`](/docs/reference/wiki/interfaces/irlagent/) | Marker interface for reinforcement learning agents that integrate with AiModelBuilder. |
| [`IRLDataLoader<T>`](/docs/reference/wiki/interfaces/irldataloader/) | Interface for data loaders that provide experience data for reinforcement learning. |
| [`IRadialBasisFunction<T>`](/docs/reference/wiki/interfaces/iradialbasisfunction/) | Defines a radial basis function (RBF) that measures similarity based on distance. |
| [`IReasoningStrategy<T>`](/docs/reference/wiki/interfaces/ireasoningstrategy/) | Defines the contract for reasoning strategies that solve problems through structured thinking. |
| [`IReferringSegmentation<T>`](/docs/reference/wiki/interfaces/ireferringsegmentation/) | Interface for referring segmentation models that segment objects based on natural language descriptions, including complex reasoning about spatial relationships and attributes. |
| [`IRegression<T>`](/docs/reference/wiki/interfaces/iregression/) | Defines the common interface for all regression algorithms in the AiDotNet library. |
| [`IRegularization<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/iregularization/) |  |
| [`IReranker<T>`](/docs/reference/wiki/interfaces/ireranker/) | Defines the contract for reranking retrieved documents to improve relevance ordering. |
| [`IResamplingStrategy<T>`](/docs/reference/wiki/interfaces/iresamplingstrategy/) | Defines the interface for resampling strategies used to handle imbalanced datasets. |
| [`IResettable`](/docs/reference/wiki/interfaces/iresettable/) | Defines capability to reset iteration state back to the beginning. |
| [`IRetriever<T>`](/docs/reference/wiki/interfaces/iretriever/) | Defines the contract for retrieving relevant documents based on a query. |
| [`ISafetyFilter<T>`](/docs/reference/wiki/interfaces/isafetyfilter/) | Defines the contract for safety filters that detect and prevent harmful or inappropriate model inputs and outputs. |
| [`ISafetyModule<T>`](/docs/reference/wiki/interfaces/isafetymodule/) | Base interface for all safety modules in the composable safety pipeline. |
| [`ISceneClassifier<T>`](/docs/reference/wiki/interfaces/isceneclassifier/) | Interface for acoustic scene classification models that identify the environment/context of audio. |
| [`ISecondOrderGradientComputable<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/isecondordergradientcomputable/) | Extended gradient computation interface for MAML meta-learning algorithms. |
| [`ISegmentationModel<T>`](/docs/reference/wiki/interfaces/isegmentationmodel/) | Base interface for all image segmentation models that classify pixels into categories. |
| [`ISelfSupervisedLoss<T>`](/docs/reference/wiki/interfaces/iselfsupervisedloss/) | Interface for self-supervised loss functions used in meta-learning. |
| [`ISemanticSegmentation<T>`](/docs/reference/wiki/interfaces/isemanticsegmentation/) | Interface for semantic segmentation models that assign a class label to every pixel. |
| [`ISemiSupervisedClassifier<T>`](/docs/reference/wiki/interfaces/isemisupervisedclassifier/) | Defines the interface for semi-supervised classification algorithms that can learn from both labeled and unlabeled data. |
| [`ISequenceLossFunction<T>`](/docs/reference/wiki/interfaces/isequencelossfunction/) | Interface for sequence loss functions that operate on variable-length sequences. |
| [`IShuffleable`](/docs/reference/wiki/interfaces/ishuffleable/) | Defines capability to shuffle data for randomized iteration. |
| [`ISoundLocalizer<T>`](/docs/reference/wiki/interfaces/isoundlocalizer/) | Interface for sound localization models that estimate the spatial position of sound sources. |
| [`ISpeakerDiarizer<T>`](/docs/reference/wiki/interfaces/ispeakerdiarizer/) | Interface for speaker diarization models that segment audio by speaker ("who spoke when"). |
| [`ISpeakerEmbeddingExtractor<T>`](/docs/reference/wiki/interfaces/ispeakerembeddingextractor/) | Interface for speaker embedding extraction models (d-vector/x-vector extraction). |
| [`ISpeakerVerifier<T>`](/docs/reference/wiki/interfaces/ispeakerverifier/) | Interface for speaker verification models that determine if audio matches a claimed identity. |
| [`ISpeechRecognizer<T>`](/docs/reference/wiki/interfaces/ispeechrecognizer/) | Interface for speech recognition models that transcribe audio to text (ASR - Automatic Speech Recognition). |
| [`IStatefulDataLoader<T>`](/docs/reference/wiki/interfaces/istatefuldataloader/) | Extends `IDataLoader` with checkpoint/resume support for fault-tolerant training. |
| [`IStratifiedSampler`](/docs/reference/wiki/interfaces/istratifiedsampler/) | Interface for samplers that use class labels for stratification. |
| [`IStreamingDataLoader<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/istreamingdataloader/) | Interface for streaming data loaders that process data on-demand without loading all data into memory. |
| [`IStreamingEventDetectionSession<T>`](/docs/reference/wiki/interfaces/istreamingeventdetectionsession/) | Interface for streaming event detection sessions. |
| [`IStreamingSynthesisSession<T>`](/docs/reference/wiki/interfaces/istreamingsynthesissession/) | Interface for streaming TTS synthesis sessions. |
| [`IStreamingTranscriptionSession<T>`](/docs/reference/wiki/interfaces/istreamingtranscriptionsession/) | Interface for streaming transcription sessions. |
| [`ISurvivalModel<T>`](/docs/reference/wiki/interfaces/isurvivalmodel/) | Interface for survival analysis models. |
| [`ISyntheticTabularGenerator<T>`](/docs/reference/wiki/interfaces/isynthetictabulargenerator/) | Defines the contract for synthetic tabular data generators that learn the distribution of real tabular data and can produce new, realistic synthetic rows. |
| [`ITaskSampler<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/itasksampler/) | Controls the strategy for sampling meta-learning tasks (episodes) from a dataset. |
| [`ITeacherModel<TInput, TOutput>`](/docs/reference/wiki/interfaces/iteachermodel/) | Represents a trained teacher model for knowledge distillation. |
| [`ITextSafetyModule<T>`](/docs/reference/wiki/interfaces/itextsafetymodule/) | Interface for safety modules that operate on text content. |
| [`ITextToSpeech<T>`](/docs/reference/wiki/interfaces/itexttospeech/) | Interface for text-to-speech (TTS) models that synthesize spoken audio from text. |
| [`ITextVectorizer<T>`](/docs/reference/wiki/interfaces/itextvectorizer/) | Defines a text vectorizer that converts text documents to numeric feature matrices. |
| [`ITimeSeriesClassifier<T>`](/docs/reference/wiki/interfaces/itimeseriesclassifier/) | Interface for time series classification models. |
| [`ITimeSeriesDecomposition<T>`](/docs/reference/wiki/interfaces/itimeseriesdecomposition/) | Defines methods and properties for decomposing time series data into its component parts. |
| [`ITimeSeriesFeatureExtractor<T>`](/docs/reference/wiki/interfaces/itimeseriesfeatureextractor/) | Defines a specialized data transformer for extracting features from time series data. |
| [`ITimeSeriesModel<T>`](/docs/reference/wiki/interfaces/itimeseriesmodel/) | Defines the core functionality for time series prediction models. |
| [`ITokenEmbedding<T>`](/docs/reference/wiki/interfaces/itokenembedding/) | Exposes token embedding lookup for models that maintain a token embedding table. |
| [`ITrainableLayer<T>`](/docs/reference/wiki/interfaces/itrainablelayer/) | Defines a neural network layer with trainable parameters that can be used with tape-based automatic differentiation (autodiff). |
| [`ITrainer<T>`](/docs/reference/wiki/interfaces/itrainer/) | Interface for training machine learning models from configuration-driven recipes. |
| [`ITrainingMonitor<T>`](/docs/reference/wiki/interfaces/itrainingmonitor/) | Defines the contract for training monitoring systems that track and visualize model training progress. |
| [`ITreeBasedClassifier<T>`](/docs/reference/wiki/interfaces/itreebasedclassifier/) | Interface for tree-based classification algorithms. |
| [`ITreeBasedRegression<T>`](/docs/reference/wiki/interfaces/itreebasedregression/) | Defines the core functionality for tree-based machine learning models. |
| [`IUnifiedMultimodalModel<T>`](/docs/reference/wiki/interfaces/iunifiedmultimodalmodel/) | Defines the contract for unified multimodal models that handle multiple modalities in a single architecture, similar to GPT-4o, Gemini, or Meta's CM3Leon. |
| [`IVAEModel<T>`](/docs/reference/wiki/interfaces/ivaemodel/) | Interface for Variational Autoencoder (VAE) models used in latent diffusion. |
| [`IVectorActivationFunction<T>`](/docs/reference/wiki/interfaces/ivectoractivationfunction/) | Defines activation functions that operate on vectors and tensors in neural networks. |
| [`IVideoCLIPModel<T>`](/docs/reference/wiki/interfaces/ivideoclipmodel/) | Defines the contract for VideoCLIP-style models that align video and text in a shared embedding space. |
| [`IVideoDiffusionModel<T>`](/docs/reference/wiki/interfaces/ivideodiffusionmodel/) | Interface for video diffusion models that generate temporal sequences. |
| [`IVideoSafetyModule<T>`](/docs/reference/wiki/interfaces/ivideosafetymodule/) | Interface for safety modules that operate on video content. |
| [`IVideoSegmentation<T>`](/docs/reference/wiki/interfaces/ivideosegmentation/) | Interface for video segmentation models that track and segment objects across video frames. |
| [`IVoiceActivityDetector<T>`](/docs/reference/wiki/interfaces/ivoiceactivitydetector/) | Defines the contract for Voice Activity Detection (VAD) models. |
| [`IWaveletFunction<T>`](/docs/reference/wiki/interfaces/iwaveletfunction/) | Defines the functionality for wavelet transforms used in signal processing and data analysis. |
| [`IWeightLoadable<T>`](/docs/reference/wiki/interfaces/iweightloadable/) | Defines the contract for models that support loading weights by name. |
| [`IWeightStreamingCapableBuilder<T, TInput, TOutput>`](/docs/reference/wiki/interfaces/iweightstreamingcapablebuilder/) | Optional companion interface for builders that support PaLM-E-scale weight streaming. |
| [`IWeightedSampler<T>`](/docs/reference/wiki/interfaces/iweightedsampler/) | Interface for samplers that use sample weights. |
| [`IWindowFunction<T>`](/docs/reference/wiki/interfaces/iwindowfunction/) | Defines functionality for creating window functions used in signal processing and data analysis. |

## Enums (21)

| Type | Summary |
|:-----|:--------|
| [`ArrayType`](/docs/reference/wiki/interfaces/arraytype/) | Types of microphone arrays. |
| [`ConditioningType`](/docs/reference/wiki/interfaces/conditioningtype/) | Types of conditioning supported by diffusion models. |
| [`DallE3ImageSize`](/docs/reference/wiki/interfaces/dalle3imagesize/) | Represents the available image sizes for DALL-E 3 generation. |
| [`DallE3Quality`](/docs/reference/wiki/interfaces/dalle3quality/) | Represents the quality settings for DALL-E 3 generation. |
| [`DallE3Style`](/docs/reference/wiki/interfaces/dalle3style/) | Represents the style settings for DALL-E 3 generation. |
| [`DriftStatus`](/docs/reference/wiki/interfaces/driftstatus/) | Represents the status of drift detection. |
| [`FineTuningCategory`](/docs/reference/wiki/interfaces/finetuningcategory/) | Categories of fine-tuning methods. |
| [`FineTuningMethodType`](/docs/reference/wiki/interfaces/finetuningmethodtype/) | Specific fine-tuning method types. |
| [`FrameInterpolationMethod`](/docs/reference/wiki/interfaces/frameinterpolationmethod/) | Methods for interpolating between video frames. |
| [`GpuOptimizerType`](/docs/reference/wiki/interfaces/gpuoptimizertype/) | Enumerates the types of GPU-optimized optimizers available. |
| [`LayerCategory`](/docs/reference/wiki/interfaces/layercategory/) | Classification of neural network layer types for automated per-layer decisions. |
| [`LogLevel`](/docs/reference/wiki/interfaces/loglevel/) | Log levels for training messages. |
| [`ModalityType`](/docs/reference/wiki/interfaces/modalitytype/) | Represents the different modality types supported by ImageBind. |
| [`ModelStage`](/docs/reference/wiki/interfaces/modelstage/) | Represents the stages a model can be in during its lifecycle. |
| [`MusicalMode`](/docs/reference/wiki/interfaces/musicalmode/) | Musical mode (major or minor). |
| [`PipelineOperationType`](/docs/reference/wiki/interfaces/pipelineoperationtype/) | Types of pipeline operations. |
| [`RewardModelType`](/docs/reference/wiki/interfaces/rewardmodeltype/) | Types of reward models. |
| [`SparsityPattern`](/docs/reference/wiki/interfaces/sparsitypattern/) | Types of sparsity patterns. |
| [`SurfaceReconstructionMethod`](/docs/reference/wiki/interfaces/surfacereconstructionmethod/) | Methods for reconstructing surfaces from point clouds. |
| [`TempoRelation`](/docs/reference/wiki/interfaces/temporelation/) | Relationship of a tempo hypothesis to the primary tempo. |
| [`VoiceGender`](/docs/reference/wiki/interfaces/voicegender/) | Gender classification for TTS voices. |

## Options & Configuration (12)

| Type | Summary |
|:-----|:--------|
| [`AdagradGpuConfig`](/docs/reference/wiki/interfaces/adagradgpuconfig/) | Configuration for Adagrad optimizer on GPU. |
| [`AdamGpuConfig`](/docs/reference/wiki/interfaces/adamgpuconfig/) | Configuration for Adam optimizer on GPU. |
| [`AdamWGpuConfig`](/docs/reference/wiki/interfaces/adamwgpuconfig/) | Configuration for AdamW optimizer on GPU. |
| [`AudioFeatureOptions`](/docs/reference/wiki/interfaces/audiofeatureoptions/) | Options for audio feature extraction. |
| [`AudioGenerationOptions<T>`](/docs/reference/wiki/interfaces/audiogenerationoptions/) | Advanced options for audio generation. |
| [`LambGpuConfig`](/docs/reference/wiki/interfaces/lambgpuconfig/) | Configuration for LAMB (Layer-wise Adaptive Moments) optimizer on GPU. |
| [`LarsGpuConfig`](/docs/reference/wiki/interfaces/larsgpuconfig/) | Configuration for LARS (Layer-wise Adaptive Rate Scaling) optimizer on GPU. |
| [`MicrophoneArrayConfig<T>`](/docs/reference/wiki/interfaces/microphonearrayconfig/) | Configuration for microphone array geometry. |
| [`NagGpuConfig`](/docs/reference/wiki/interfaces/naggpuconfig/) | Configuration for Nesterov Accelerated Gradient (NAG) optimizer on GPU. |
| [`PruningConfig`](/docs/reference/wiki/interfaces/pruningconfig/) | Configuration for pruning operations. |
| [`RmsPropGpuConfig`](/docs/reference/wiki/interfaces/rmspropgpuconfig/) | Configuration for RMSprop optimizer on GPU. |
| [`SgdGpuConfig`](/docs/reference/wiki/interfaces/sgdgpuconfig/) | Configuration for SGD (Stochastic Gradient Descent) optimizer on GPU. |

## Helpers & Utilities (1)

| Type | Summary |
|:-----|:--------|
| [`LossFunctionExtensions`](/docs/reference/wiki/interfaces/lossfunctionextensions/) |  |


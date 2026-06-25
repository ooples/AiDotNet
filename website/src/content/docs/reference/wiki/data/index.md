---
title: "Data"
description: "All 348 public types in the AiDotNet.data namespace, organized by kind."
section: "API Reference"
---

**348** public types in this namespace, organized by kind.

## Models & Types (190)

| Type | Summary |
|:-----|:--------|
| [`ActiveLearningQueryStrategy`](/docs/reference/wiki/data/activelearningquerystrategy/) | Selects the most informative unlabeled samples for annotation using active learning. |
| [`ActiveLearningSampler<T>`](/docs/reference/wiki/data/activelearningsampler/) | A sampler for active learning that selects the most informative samples for labeling. |
| [`Ade20kDataLoader<T>`](/docs/reference/wiki/data/ade20kdataloader/) | Loads the ADE20K semantic segmentation dataset. |
| [`AgNewsDataLoader<T>`](/docs/reference/wiki/data/agnewsdataloader/) | Loads the AG News topic classification dataset (4 classes: World, Sports, Business, Sci/Tech). |
| [`ArcDataLoader<T>`](/docs/reference/wiki/data/arcdataloader/) | Loads the AI2 Reasoning Challenge (ARC) multiple-choice science QA benchmark (Clark et al. |
| [`ArrowDataset<T>`](/docs/reference/wiki/data/arrowdataset/) | Provides read/write access to datasets in a custom binary columnar format inspired by Apache Arrow IPC. |
| [`AsyncDataPipeline<T>`](/docs/reference/wiki/data/asyncdatapipeline/) | Provides async data pipeline operations with prefetching support. |
| [`AudioFileDataset<T>`](/docs/reference/wiki/data/audiofiledataset/) | Loads audio files from directories for audio classification and processing tasks. |
| [`AudioSetDataLoader<T>`](/docs/reference/wiki/data/audiosetdataloader/) | Loads the AudioSet large-scale audio event dataset (2M+ 10-second clips, 527 categories). |
| [`BalancedEpisodicDataLoader<T, TInput, TOutput>`](/docs/reference/wiki/data/balancedepisodicdataloader/) | Provides balanced episodic task sampling that ensures equal class representation across multiple tasks. |
| [`BigEarthNetDataLoader<T>`](/docs/reference/wiki/data/bigearthnetdataloader/) | Loads the BigEarthNet multi-label remote sensing dataset (590K Sentinel-2 patches, 19 or 43 classes). |
| [`BucketBatchSampler`](/docs/reference/wiki/data/bucketbatchsampler/) | Groups sequences by length into buckets, then batches within each bucket to minimize padding. |
| [`CacheInfo`](/docs/reference/wiki/data/cacheinfo/) | Information about the current state of a pipeline cache. |
| [`CachingDataLoader<TKey, TValue>`](/docs/reference/wiki/data/cachingdataloader/) | Wraps data loading with an in-memory cache to avoid redundant I/O. |
| [`Caltech101DataLoader<T>`](/docs/reference/wiki/data/caltech101dataloader/) | Loads the Caltech-101 image classification dataset (Fei-Fei et al. |
| [`CelebADataLoader<T>`](/docs/reference/wiki/data/celebadataloader/) | Loads the CelebA face attributes dataset. |
| [`CheXpertDataLoader<T>`](/docs/reference/wiki/data/chexpertdataloader/) | Loads the CheXpert chest radiograph dataset (224K images, 14 observations with uncertainty). |
| [`CheckpointData`](/docs/reference/wiki/data/checkpointdata/) | Data stored in a mid-epoch checkpoint. |
| [`ChestXray14DataLoader<T>`](/docs/reference/wiki/data/chestxray14dataloader/) | Loads the NIH Chest X-ray 14 multi-label classification dataset (112K images, 14 disease labels). |
| [`Cifar100DataLoader<T>`](/docs/reference/wiki/data/cifar100dataloader/) | Loads the CIFAR-100 image classification dataset (50k train / 10k test, 32x32 RGB, 100 fine / 20 coarse classes). |
| [`Cifar10DataLoader<T>`](/docs/reference/wiki/data/cifar10dataloader/) | Loads the CIFAR-10 image classification dataset (50k train / 10k test, 32x32 RGB, 10 classes). |
| [`CitationNetworkLoader<T>`](/docs/reference/wiki/data/citationnetworkloader/) | Loads citation network datasets (Cora, CiteSeer, PubMed) for node classification. |
| [`CityscapesDataLoader<T>`](/docs/reference/wiki/data/cityscapesdataloader/) | Loads the Cityscapes semantic-segmentation dataset (Cordts et al. |
| [`CnnDailyMailDataLoader<T>`](/docs/reference/wiki/data/cnndailymaildataloader/) | Loads the CNN/DailyMail abstractive-summarization dataset v3.0.0 (Hermann et al. |
| [`CocoDetectionDataLoader<T>`](/docs/reference/wiki/data/cocodetectiondataloader/) | Loads the COCO 2017 object detection dataset (118K train / 5K val, 80 categories). |
| [`CommonVoiceDataLoader<T>`](/docs/reference/wiki/data/commonvoicedataloader/) | Loads the Mozilla Common Voice multilingual speech dataset (19K+ hours, 100+ languages). |
| [`Compose<T>`](/docs/reference/wiki/data/compose/) | Chains multiple transforms of the same type into a single transform. |
| [`CoresetSelector`](/docs/reference/wiki/data/coresetselector/) | Selects a representative coreset from a dataset using distance-based strategies. |
| [`CsvDataLoader<T>`](/docs/reference/wiki/data/csvdataloader/) | Loads supervised learning data from CSV files into Matrix/Vector format. |
| [`CsvStreamingDataLoader<T, TInput, TOutput>`](/docs/reference/wiki/data/csvstreamingdataloader/) | A streaming data loader that reads from a CSV file line by line. |
| [`CurriculumDataScheduler`](/docs/reference/wiki/data/curriculumdatascheduler/) | Schedules training data presentation order and pacing for curriculum learning. |
| [`CurriculumEpisodicDataLoader<T, TInput, TOutput>`](/docs/reference/wiki/data/curriculumepisodicdataloader/) | Provides curriculum-based episodic task sampling that progressively increases task difficulty during training. |
| [`CurriculumSampler<T>`](/docs/reference/wiki/data/curriculumsampler/) | A sampler that implements curriculum learning by progressively introducing harder samples. |
| [`DataPipeline<T>`](/docs/reference/wiki/data/datapipeline/) | Provides TensorFlow-style data pipeline operations for transforming and processing data. |
| [`DataPruner`](/docs/reference/wiki/data/datapruner/) | Prunes (removes) training samples based on difficulty/importance scores. |
| [`DatasetDistiller`](/docs/reference/wiki/data/datasetdistiller/) | Performs dataset distillation to synthesize a compact representative dataset. |
| [`DatasetMixer<T>`](/docs/reference/wiki/data/datasetmixer/) | Blends multiple datasets by weight ratios, producing mixed batches for curriculum or domain mixing. |
| [`DefaultCollateFunction<T>`](/docs/reference/wiki/data/defaultcollatefunction/) | Stacks equal-size tensors into a batch tensor along dimension 0. |
| [`DenseSampler`](/docs/reference/wiki/data/densesampler/) | Dense sampling: uniformly samples frames at regular intervals from the video. |
| [`DistributedBucketSampler`](/docs/reference/wiki/data/distributedbucketsampler/) | Combines distributed partitioning with bucket batching for efficient distributed NLP training. |
| [`DistributedSampler`](/docs/reference/wiki/data/distributedsampler/) | Partitions dataset indices across N ranks for distributed (multi-GPU/multi-node) training. |
| [`DocVqaDataLoader<T>`](/docs/reference/wiki/data/docvqadataloader/) | Loads the DocVQA document visual question answering dataset. |
| [`DomainMixingSampler`](/docs/reference/wiki/data/domainmixingsampler/) | Samples from multiple data domains with configurable mixing ratios for multi-domain LLM training. |
| [`DtdDataLoader<T>`](/docs/reference/wiki/data/dtddataloader/) | Loads the Describable Textures Dataset (DTD; Cimpoi et al. |
| [`DynamicBatchSampler`](/docs/reference/wiki/data/dynamicbatchsampler/) | Creates batches that fit a maximum number of tokens/elements rather than a fixed number of samples. |
| [`ElasticDistributedSampler`](/docs/reference/wiki/data/elasticdistributedsampler/) | Distributed sampler that evenly divides data across multiple workers with elastic scaling support. |
| [`EnvironmentDataLoader<T>`](/docs/reference/wiki/data/environmentdataloader/) | Data loader for reinforcement learning that wraps an environment for experience collection. |
| [`Enwik8DataLoader<T>`](/docs/reference/wiki/data/enwik8dataloader/) | Loads the enwik8 character-level Wikipedia language modeling benchmark (first 100M bytes of an English Wikipedia XML dump). |
| [`Esc50DataLoader<T>`](/docs/reference/wiki/data/esc50dataloader/) | Loads the ESC-50 environmental sound classification dataset (2000 clips, 50 classes). |
| [`EuroSatDataLoader<T>`](/docs/reference/wiki/data/eurosatdataloader/) | Loads the EuroSAT land use/land cover classification dataset (27K patches, 64x64 RGB, 10 classes). |
| [`ExactHashDeduplicator`](/docs/reference/wiki/data/exacthashdeduplicator/) | Detects exact duplicate documents using cryptographic hashing (SHA-256). |
| [`FMoWDataLoader<T>`](/docs/reference/wiki/data/fmowdataloader/) | Loads the Functional Map of the World (fMoW) satellite imagery dataset (1M+ images, 62 categories). |
| [`FashionMnistDataLoader<T>`](/docs/reference/wiki/data/fashionmnistdataloader/) | Loads the Fashion-MNIST clothing classification dataset (60k train / 10k test, 28x28 grayscale). |
| [`FileStreamingDataLoader<T, TInput, TOutput>`](/docs/reference/wiki/data/filestreamingdataloader/) | A streaming data loader that reads from files in a directory. |
| [`FleursDataLoader<T>`](/docs/reference/wiki/data/fleursdataloader/) | Loads the FLEURS multilingual speech benchmark (102 languages, ~12 hours per language). |
| [`Flowers102DataLoader<T>`](/docs/reference/wiki/data/flowers102dataloader/) | Loads the Oxford Flowers-102 dataset (Nilsback & Zisserman 2008) — 102 fine-grained flower species. |
| [`Food101DataLoader<T>`](/docs/reference/wiki/data/food101dataloader/) | Loads the Food-101 fine-grained classification dataset (Bossard et al. |
| [`Fsd50kDataLoader<T>`](/docs/reference/wiki/data/fsd50kdataloader/) | Loads the FSD50K audio event dataset (51,197 clips, 200 sound event classes). |
| [`GigaSpeechDataLoader<T>`](/docs/reference/wiki/data/gigaspeechdataloader/) | Loads the GigaSpeech multi-domain English ASR dataset (10K hours from audiobooks, podcasts, YouTube). |
| [`GlueDataLoader<T>`](/docs/reference/wiki/data/gluedataloader/) | Loads GLUE benchmark sub-tasks (CoLA, SST-2, MRPC, QQP, STS-B, MNLI, QNLI, RTE, WNLI). |
| [`GraphClassificationTask<T>`](/docs/reference/wiki/data/graphclassificationtask/) | Represents a graph classification task where the goal is to classify entire graphs. |
| [`GraphData<T>`](/docs/reference/wiki/data/graphdata/) | Represents a single graph with nodes, edges, features, and optional labels. |
| [`GraphGenerationTask<T>`](/docs/reference/wiki/data/graphgenerationtask/) | Represents a graph generation task where the goal is to generate new valid graphs. |
| [`Gsm8kDataLoader<T>`](/docs/reference/wiki/data/gsm8kdataloader/) | Loads the GSM8K grade-school math word-problem dataset (Cobbe et al. |
| [`GtzanDataLoader<T>`](/docs/reference/wiki/data/gtzandataloader/) | Loads the GTZAN music genre classification dataset (Tzanetakis & Cook 2002). |
| [`Hdf5Dataset<T>`](/docs/reference/wiki/data/hdf5dataset/) | Provides read/write access to datasets in a custom binary format for named multidimensional arrays, inspired by the HDF5 data model. |
| [`HellaswagDataLoader<T>`](/docs/reference/wiki/data/hellaswagdataloader/) | Loads the HellaSwag 4-way commonsense NLI benchmark (Zellers et al. |
| [`HeuristicTextFilter`](/docs/reference/wiki/data/heuristictextfilter/) | Filters text documents using simple heuristic rules for quality assessment. |
| [`Hmdb51DataLoader<T>`](/docs/reference/wiki/data/hmdb51dataloader/) | Loads the HMDB51 action recognition dataset (6,766 clips, 51 classes). |
| [`HumanEvalDataLoader<T>`](/docs/reference/wiki/data/humanevaldataloader/) | Loads the HumanEval Python code-generation benchmark (Chen et al. |
| [`INaturalistDataLoader<T>`](/docs/reference/wiki/data/inaturalistdataloader/) | Loads the iNaturalist species classification dataset (~2.7M images, 10,000 species in 2021 version). |
| [`IdentityTransform<T>`](/docs/reference/wiki/data/identitytransform/) | A no-op transform that passes input through unchanged. |
| [`ImageClassificationDataset<T>`](/docs/reference/wiki/data/imageclassificationdataset/) | An in-memory image classification dataset with an optional composable transform pipeline. |
| [`ImageFolderDataset<T>`](/docs/reference/wiki/data/imagefolderdataset/) | Loads images from a directory structure where each subdirectory is a class label. |
| [`ImageNet1kDataLoader<T>`](/docs/reference/wiki/data/imagenet1kdataloader/) | Loads the ImageNet-1K (ILSVRC 2012) image classification dataset (~1.28M train / 50K val, 1000 classes). |
| [`ImageNet21kDataLoader<T>`](/docs/reference/wiki/data/imagenet21kdataloader/) | Loads the ImageNet-21K dataset (~14.2M images, 21,841 categories from the full WordNet hierarchy). |
| [`ImageQualityFilter`](/docs/reference/wiki/data/imagequalityfilter/) | Filters images based on resolution, aspect ratio, and pixel statistics. |
| [`Imdb50kDataLoader<T>`](/docs/reference/wiki/data/imdb50kdataloader/) | Loads the IMDB 50k movie review sentiment analysis dataset (25k train / 25k test, binary classification). |
| [`ImportanceSampler<T>`](/docs/reference/wiki/data/importancesampler/) | A sampler that implements importance sampling for variance reduction. |
| [`InMemoryDataLoader<T, TInput, TOutput>`](/docs/reference/wiki/data/inmemorydataloader/) | A simple in-memory data loader for supervised learning data. |
| [`InterleavedDataset<T>`](/docs/reference/wiki/data/interleaveddataset/) | A dataset where each sample is an interleaved sequence of modality segments, such as alternating image-text-image-text in vision-language models. |
| [`InterleavedSegment<T>`](/docs/reference/wiki/data/interleavedsegment/) | A single segment within an interleaved sequence. |
| [`InterleavedSequence<T>`](/docs/reference/wiki/data/interleavedsequence/) | A single interleaved sequence containing ordered segments of different modalities. |
| [`JsonlDataLoader<T>`](/docs/reference/wiki/data/jsonldataloader/) | A typed data loader facade that wraps `JsonlStreamingLoader` and implements `StreamingDataLoaderBase` for IDataLoader compliance. |
| [`Kinetics400DataLoader<T>`](/docs/reference/wiki/data/kinetics400dataloader/) | Loads the Kinetics-400 human action recognition dataset (~300K clips, 400 classes). |
| [`KittiDataLoader<T>`](/docs/reference/wiki/data/kittidataloader/) | Loads the KITTI 3D object detection dataset (LiDAR point clouds with 3D bounding boxes). |
| [`LambdaTransform<TInput, TOutput>`](/docs/reference/wiki/data/lambdatransform/) | Wraps a `Func` delegate as an `ITransform`. |
| [`LanguageIdFilter`](/docs/reference/wiki/data/languageidfilter/) | Filters documents based on detected language using character n-gram profiles. |
| [`LeafFederatedDataLoader<T>`](/docs/reference/wiki/data/leaffederateddataloader/) | Data loader that reads LEAF benchmark JSON splits and exposes both aggregated (X, Y) data and per-client partitions. |
| [`LibriSpeechDataLoader<T>`](/docs/reference/wiki/data/librispeechdataloader/) | Loads the LibriSpeech automatic speech recognition dataset (~1000 hours of 16kHz English speech). |
| [`LinkPredictionTask<T>`](/docs/reference/wiki/data/linkpredictiontask/) | Represents a link prediction task where the goal is to predict missing or future edges. |
| [`LjSpeechDataLoader<T>`](/docs/reference/wiki/data/ljspeechdataloader/) | Loads the LJSpeech 1.1 single-speaker TTS corpus (Ito & Johnson 2017). |
| [`LmdbDataset<T>`](/docs/reference/wiki/data/lmdbdataset/) | Provides read-only access to datasets stored in a custom binary key-value format inspired by LMDB (Lightning Memory-Mapped Database). |
| [`M4DatasetLoader<T>`](/docs/reference/wiki/data/m4datasetloader/) | Loads time series datasets from the M4 Competition for benchmarking forecasting models. |
| [`M4TimeSeries<T>`](/docs/reference/wiki/data/m4timeseries/) | Represents a single time series from the M4 Competition. |
| [`MaestroDataLoader<T>`](/docs/reference/wiki/data/maestrodataloader/) | Loads the MAESTRO piano performance dataset (~200 hours, aligned MIDI and audio). |
| [`MathDataLoader<T>`](/docs/reference/wiki/data/mathdataloader/) | Loads the Hendrycks MATH benchmark — competition math problems (Hendrycks et al. |
| [`MbppDataLoader<T>`](/docs/reference/wiki/data/mbppdataloader/) | Loads the MBPP (Mostly Basic Python Problems) benchmark — 1,000 entry-level Python coding problems with natural-language descriptions and unit tests (Austin et al. |
| [`MemoryMappedDataset<T>`](/docs/reference/wiki/data/memorymappeddataset/) | Memory-mapped dataset access for efficient I/O on large binary datasets. |
| [`MemoryMappedStreamingDataLoader<T, TInput, TOutput>`](/docs/reference/wiki/data/memorymappedstreamingdataloader/) | A streaming data loader that uses memory-mapped files for efficient random access to large binary datasets. |
| [`MetaLearningTask<T, TInput, TOutput>`](/docs/reference/wiki/data/metalearningtask/) | Represents a single meta-learning task for few-shot learning, containing support and query sets. |
| [`MidEpochCheckpointer`](/docs/reference/wiki/data/midepochcheckpointer/) | Saves and restores training state mid-epoch for fault tolerance. |
| [`MinHashDeduplicator`](/docs/reference/wiki/data/minhashdeduplicator/) | Detects near-duplicate documents using MinHash with Locality-Sensitive Hashing (LSH). |
| [`MinMaxScaleTransform<T>`](/docs/reference/wiki/data/minmaxscaletransform/) | Scales values to a target range using min-max normalization. |
| [`MmluDataLoader<T>`](/docs/reference/wiki/data/mmludataloader/) | Loads MMLU — Massive Multitask Language Understanding (Hendrycks et al. |
| [`MnistDataLoader<T>`](/docs/reference/wiki/data/mnistdataloader/) | Loads the MNIST handwritten digit classification dataset (60k train / 10k test, 28x28 grayscale). |
| [`ModalitySample<T>`](/docs/reference/wiki/data/modalitysample/) | Represents a single modality's data within a multimodal sample. |
| [`ModelNet40ClassificationDataLoader<T>`](/docs/reference/wiki/data/modelnet40classificationdataloader/) | Loads the ModelNet40 point cloud classification dataset. |
| [`MolecularDatasetLoader<T>`](/docs/reference/wiki/data/moleculardatasetloader/) | Loads molecular graph datasets (ZINC, QM9) for graph-level property prediction and generation. |
| [`MultiSourceMixer<TItem>`](/docs/reference/wiki/data/multisourcemixer/) | Mixes multiple data sources with configurable weights for multi-domain training. |
| [`MultimodalDataset<T>`](/docs/reference/wiki/data/multimodaldataset/) | A dataset of multimodal samples for training models that process multiple data types simultaneously. |
| [`MultimodalSample<T>`](/docs/reference/wiki/data/multimodalsample/) | A collection of modality samples representing one multimodal data point. |
| [`Musdb18DataLoader<T>`](/docs/reference/wiki/data/musdb18dataloader/) | Loads the MUSDB18 music source separation dataset (150 tracks, 4 stems: vocals, drums, bass, other). |
| [`NodeClassificationTask<T>`](/docs/reference/wiki/data/nodeclassificationtask/) | Represents a node classification task where the goal is to predict labels for individual nodes. |
| [`NormalizeTransform<T>`](/docs/reference/wiki/data/normalizetransform/) | Normalizes an array of values using mean and standard deviation: (x - mean) / std. |
| [`NsynthDataLoader<T>`](/docs/reference/wiki/data/nsynthdataloader/) | Loads the NSynth Neural Synth dataset (Engel et al. |
| [`NuScenesDataLoader<T>`](/docs/reference/wiki/data/nuscenesdataloader/) | Loads the nuScenes dataset (LiDAR point clouds with 3D bounding box annotations). |
| [`OGBDatasetLoader<T>`](/docs/reference/wiki/data/ogbdatasetloader/) | Loads datasets from the Open Graph Benchmark (OGB) for standardized evaluation. |
| [`OneHotEncodeTransform<T>`](/docs/reference/wiki/data/onehotencodetransform/) | Converts a class index (integer label) to a one-hot encoded vector. |
| [`OpenImagesDataLoader<T>`](/docs/reference/wiki/data/openimagesdataloader/) | Loads the Open Images V7 object detection dataset (~9M images, 600 categories). |
| [`OxfordPetsDataLoader<T>`](/docs/reference/wiki/data/oxfordpetsdataloader/) | Loads the Oxford-IIIT Pet dataset (Parkhi et al. |
| [`PackedSequenceBatch<T>`](/docs/reference/wiki/data/packedsequencebatch/) | Represents a batch of packed (non-padded) variable-length sequences. |
| [`PackedSequenceCollateFunction<T>`](/docs/reference/wiki/data/packedsequencecollatefunction/) | Packs variable-length sequences into a contiguous tensor without padding, along with sequence lengths for reconstruction. |
| [`PaddingCollateFunction<T>`](/docs/reference/wiki/data/paddingcollatefunction/) | Pads variable-length sequences to the maximum length in the batch, then stacks them. |
| [`ParallelBatchLoader<TBatch>`](/docs/reference/wiki/data/parallelbatchloader/) | Provides parallel batch loading with multiple workers for improved throughput. |
| [`ParquetDataLoader<T>`](/docs/reference/wiki/data/parquetdataloader/) | Reads tabular data from Apache Parquet columnar files using the Parquet.Net library. |
| [`PascalVocDataLoader<T>`](/docs/reference/wiki/data/pascalvocdataloader/) | Loads the Pascal VOC object detection dataset (20 categories, XML annotations). |
| [`PennTreebankDataLoader<T>`](/docs/reference/wiki/data/penntreebankdataloader/) | Loads the Penn Treebank language modeling dataset (Mikolov-preprocessed split). |
| [`PerplexityFilter`](/docs/reference/wiki/data/perplexityfilter/) | Filters documents based on perplexity scores from a simple n-gram language model. |
| [`Places365DataLoader<T>`](/docs/reference/wiki/data/places365dataloader/) | Loads the Places365 scene recognition dataset (1.8M train / 36.5K val, 256x256 RGB, 365 scene categories). |
| [`PrefetchDataLoader<TBatch>`](/docs/reference/wiki/data/prefetchdataloader/) | Wraps a batch-producing function with asynchronous prefetching for pipelined data loading. |
| [`ProteinDataLoader<T>`](/docs/reference/wiki/data/proteindataloader/) | Loads protein structure datasets as flattened feature/label tensors for graph-level classification. |
| [`PubLayNetDataLoader<T>`](/docs/reference/wiki/data/publaynetdataloader/) | Loads the PubLayNet document layout analysis dataset. |
| [`Qm9DataLoader<T>`](/docs/reference/wiki/data/qm9dataloader/) | Thin wrapper around `MolecularDatasetLoader` for the QM9 dataset. |
| [`RandomSampler`](/docs/reference/wiki/data/randomsampler/) | A sampler that randomly shuffles the dataset indices each epoch. |
| [`RetinalFundusDataLoader<T>`](/docs/reference/wiki/data/retinalfundusdataloader/) | Loads retinal fundus photography datasets for diabetic retinopathy detection (5-class grading). |
| [`ScanNetSemanticSegmentationDataLoader<T>`](/docs/reference/wiki/data/scannetsemanticsegmentationdataloader/) | Loads the ScanNet semantic segmentation dataset. |
| [`SelfPacedSampler<T>`](/docs/reference/wiki/data/selfpacedsampler/) | A sampler that implements self-paced learning with automatic difficulty adjustment. |
| [`SemanticDeduplicator`](/docs/reference/wiki/data/semanticdeduplicator/) | Detects semantic duplicates using embedding cosine similarity. |
| [`SemanticKittiDataLoader<T>`](/docs/reference/wiki/data/semantickittidataloader/) | Loads the SemanticKITTI dataset (per-point semantic labels for LiDAR point clouds). |
| [`SequencePackingCollateFunction<T>`](/docs/reference/wiki/data/sequencepackingcollatefunction/) | Packs multiple variable-length sequences into fixed-length blocks for efficient LLM training. |
| [`SequentialSampler`](/docs/reference/wiki/data/sequentialsampler/) | A sampler that returns indices in sequential order without shuffling. |
| [`ShapeNetCorePartSegmentationDataLoader<T>`](/docs/reference/wiki/data/shapenetcorepartsegmentationdataloader/) | Loads the ShapeNetCore part segmentation dataset. |
| [`ShardedStreamingDataLoader<T>`](/docs/reference/wiki/data/shardedstreamingdataloader/) | A typed data loader facade that wraps `ShardedStreamingDataset` and implements `StreamingDataLoaderBase` for IDataLoader compliance. |
| [`SkinLesionDataLoader<T>`](/docs/reference/wiki/data/skinlesiondataloader/) | Loads the ISIC Skin Lesion classification dataset (~25K images, 8 diagnostic categories). |
| [`SlowFastSampler`](/docs/reference/wiki/data/slowfastsampler/) | SlowFast sampling: produces two sets of frame indices at different temporal resolutions for SlowFast networks (Feichtenhofer et al., 2019). |
| [`SnapshotPipeline<T>`](/docs/reference/wiki/data/snapshotpipeline/) | Persists an entire processed pipeline to disk for fast reload across epochs, with automatic invalidation when source data or pipeline configuration changes. |
| [`SomethingSomethingV2DataLoader<T>`](/docs/reference/wiki/data/somethingsomethingv2dataloader/) | Loads the Something-Something V2 action recognition dataset (220K clips, 174 classes). |
| [`SpecAugmentTransform<T>`](/docs/reference/wiki/data/specaugmenttransform/) | Applies SpecAugment data augmentation (Park et al., 2019) to spectrogram tensors. |
| [`SpectrogramTransform<T>`](/docs/reference/wiki/data/spectrogramtransform/) | Transforms raw audio waveform tensors into Mel spectrogram representations. |
| [`SpeechCommandsDataLoader<T>`](/docs/reference/wiki/data/speechcommandsdataloader/) | Loads the Google Speech Commands v2 dataset (~65K clips, 35 words, 16kHz). |
| [`SquadDataLoader<T>`](/docs/reference/wiki/data/squaddataloader/) | Loads the SQuAD question answering dataset (100K+ Q&A pairs on Wikipedia articles). |
| [`StandardScaleTransform<T>`](/docs/reference/wiki/data/standardscaletransform/) | Applies Z-score normalization: (x - mean) / std, computing mean and std from a reference dataset. |
| [`StanfordCarsDataLoader<T>`](/docs/reference/wiki/data/stanfordcarsdataloader/) | Loads the Stanford Cars dataset (Krause et al. |
| [`StatefulDataLoader<T, TInput, TOutput>`](/docs/reference/wiki/data/statefuldataloader/) | Wraps any `InMemoryDataLoader` with checkpoint/resume support. |
| [`Stl10DataLoader<T>`](/docs/reference/wiki/data/stl10dataloader/) | Loads the STL-10 image classification dataset (Coates et al. |
| [`StratifiedBatchSampler`](/docs/reference/wiki/data/stratifiedbatchsampler/) | A batch sampler that ensures each batch contains samples from all classes. |
| [`StratifiedEpisodicDataLoader<T, TInput, TOutput>`](/docs/reference/wiki/data/stratifiedepisodicdataloader/) | Provides stratified episodic task sampling that maintains dataset class proportions across tasks. |
| [`StratifiedSampler`](/docs/reference/wiki/data/stratifiedsampler/) | A sampler that ensures each class is represented proportionally in each epoch. |
| [`StreamingDataLoader<T, TInput, TOutput>`](/docs/reference/wiki/data/streamingdataloader/) | A data loader that streams data from disk or other sources without loading all data into memory. |
| [`StreamingTextDataset<T>`](/docs/reference/wiki/data/streamingtextdataset/) | A streaming text dataset that lazily reads and tokenizes text files for language model training. |
| [`SubsetSampler`](/docs/reference/wiki/data/subsetsampler/) | A sampler that returns a subset of indices. |
| [`SuperGlueDataLoader<T>`](/docs/reference/wiki/data/supergluedataloader/) | Loads SuperGLUE benchmark sub-tasks (BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WiC, WSC). |
| [`SvhnDataLoader<T>`](/docs/reference/wiki/data/svhndataloader/) | Loads the SVHN (Street View House Numbers) Format-2 dataset (Netzer et al. |
| [`TaskBatch<T, TInput, TOutput>`](/docs/reference/wiki/data/taskbatch/) | Represents a batch of tasks for meta-learning with advanced batching strategies. |
| [`TemporalGraphDataLoader<T>`](/docs/reference/wiki/data/temporalgraphdataloader/) | Loads temporal graph datasets (timestamped interactions for dynamic link prediction). |
| [`TemporalJitterAugmentation<T>`](/docs/reference/wiki/data/temporaljitteraugmentation/) | Applies temporal jitter to video frame sequences by randomly shifting frame indices. |
| [`TemporalSegmentSampler`](/docs/reference/wiki/data/temporalsegmentsampler/) | Temporal Segment Networks (TSN) sampling: divides video into equal segments and samples one frame per segment. |
| [`TextLineDataset<T>`](/docs/reference/wiki/data/textlinedataset/) | Streams text line-by-line from a file for language modeling tasks. |
| [`TimitDataLoader<T>`](/docs/reference/wiki/data/timitdataloader/) | Loads the TIMIT acoustic-phonetic continuous-speech corpus (Garofolo et al. |
| [`TinyImageNetDataLoader<T>`](/docs/reference/wiki/data/tinyimagenetdataloader/) | Loads the Tiny ImageNet 200-class image-classification dataset (500 train + 50 val + 50 test per class at 64×64). |
| [`TinyStoriesDataLoader<T>`](/docs/reference/wiki/data/tinystoriesdataloader/) | Loads the TinyStories synthetic LM corpus (Eldan & Li, 2023): ≈ 2.1M GPT-generated short stories with deliberately small vocabulary for small-scale language-model research. |
| [`ToTensorTransform<T>`](/docs/reference/wiki/data/totensortransform/) | Converts a flat array to a Tensor with the specified shape. |
| [`TokenizedTextDataset<T>`](/docs/reference/wiki/data/tokenizedtextdataset/) | In-memory dataset of pre-tokenized text sequences for language model training. |
| [`TransformedDataLoader<T>`](/docs/reference/wiki/data/transformeddataloader/) | Wraps a data loader and applies a composable transform pipeline to feature data during batch extraction. |
| [`TruthfulQaDataLoader<T>`](/docs/reference/wiki/data/truthfulqadataloader/) | Loads the TruthfulQA benchmark (Lin et al. |
| [`Ucf101DataLoader<T>`](/docs/reference/wiki/data/ucf101dataloader/) | Loads the UCF101 action recognition dataset (13,320 clips, 101 classes). |
| [`UniformEpisodicDataLoader<T, TInput, TOutput>`](/docs/reference/wiki/data/uniformepisodicdataloader/) | Provides uniform random episodic task sampling for N-way K-shot meta-learning scenarios. |
| [`UrbanSound8kDataLoader<T>`](/docs/reference/wiki/data/urbansound8kdataloader/) | Loads the UrbanSound8K environmental sound dataset (8732 clips, 10 classes, 10 folds). |
| [`VctkDataLoader<T>`](/docs/reference/wiki/data/vctkdataloader/) | Loads the VCTK Corpus 0.92 multi-speaker TTS dataset. |
| [`VideoFrameDataset<T>`](/docs/reference/wiki/data/videoframedataset/) | Loads videos represented as directories of sequentially numbered image frames. |
| [`VoxPopuliDataLoader<T>`](/docs/reference/wiki/data/voxpopulidataloader/) | Loads the VoxPopuli multilingual speech dataset from European Parliament recordings. |
| [`WaymoDataLoader<T>`](/docs/reference/wiki/data/waymodataloader/) | Loads the Waymo Open Dataset (LiDAR point clouds with 3D bounding boxes). |
| [`WebDatasetDataLoader<T>`](/docs/reference/wiki/data/webdatasetdataloader/) | A typed data loader facade that wraps `WebDataset` and implements `StreamingDataLoaderBase` for IDataLoader compliance. |
| [`WeightedSampler<T>`](/docs/reference/wiki/data/weightedsampler/) | A sampler that samples indices based on their weights. |
| [`WikiText103DataLoader<T>`](/docs/reference/wiki/data/wikitext103dataloader/) | Loads the WikiText-103 language modeling dataset (100M+ tokens from Wikipedia articles). |
| [`WikiText2DataLoader<T>`](/docs/reference/wiki/data/wikitext2dataloader/) | Loads the WikiText-2 language modeling dataset (≈ 2M tokens train, ≈ 245k val, ≈ 281k test from Wikipedia articles). |
| [`Wikidata5mDataLoader<T>`](/docs/reference/wiki/data/wikidata5mdataloader/) | Loads Wikidata5M knowledge graph triplets as tensor features and labels. |
| [`XSumDataLoader<T>`](/docs/reference/wiki/data/xsumdataloader/) | Loads the XSum extreme abstractive-summarization dataset (Narayan et al. |
| [`ZincDataLoader<T>`](/docs/reference/wiki/data/zincdataloader/) | Thin wrapper around `MolecularDatasetLoader` for the ZINC dataset. |

## Base Classes (11)

| Type | Summary |
|:-----|:--------|
| [`DataLoaderBase<T>`](/docs/reference/wiki/data/dataloaderbase/) | Abstract base class providing common functionality for all data loaders. |
| [`DataSamplerBase`](/docs/reference/wiki/data/datasamplerbase/) | Base class for all data samplers providing common functionality. |
| [`EpisodicDataLoaderBase<T, TInput, TOutput>`](/docs/reference/wiki/data/episodicdataloaderbase/) | Provides a base implementation for episodic data loaders with common functionality for N-way K-shot meta-learning. |
| [`EpochAdaptiveSamplerBase<T>`](/docs/reference/wiki/data/epochadaptivesamplerbase/) | Base class for epoch-adaptive samplers that change behavior over training epochs. |
| [`GraphDataLoaderBase<T>`](/docs/reference/wiki/data/graphdataloaderbase/) | Abstract base class for graph data loaders providing common graph-related functionality. |
| [`InputOutputDataLoaderBase<T, TInput, TOutput>`](/docs/reference/wiki/data/inputoutputdataloaderbase/) | Abstract base class for input-output data loaders providing common supervised learning functionality. |
| [`MetaLearningTaskBase<T, TInput, TOutput>`](/docs/reference/wiki/data/metalearningtaskbase/) | Abstract base class for meta-learning tasks, providing common functionality and validation. |
| [`PointCloudDatasetLoaderBase<T>`](/docs/reference/wiki/data/pointclouddatasetloaderbase/) | Base class for point cloud dataset loaders that expose tensor inputs and outputs. |
| [`RLDataLoaderBase<T>`](/docs/reference/wiki/data/rldataloaderbase/) | Abstract base class for RL data loaders providing common reinforcement learning functionality. |
| [`StreamingDataLoaderBase<T, TInput, TOutput>`](/docs/reference/wiki/data/streamingdataloaderbase/) | Abstract base class for streaming data loaders that process data on-demand. |
| [`WeightedSamplerBase<T>`](/docs/reference/wiki/data/weightedsamplerbase/) | Base class for weighted samplers providing common weight-based functionality. |

## Interfaces (3)

| Type | Summary |
|:-----|:--------|
| [`ICollateFunction<TSample, TBatch>`](/docs/reference/wiki/data/icollatefunction/) | Defines how individual samples are assembled into a batch. |
| [`ITransform<TInput, TOutput>`](/docs/reference/wiki/data/itransform/) | Core interface for composable data transforms in the data pipeline. |
| [`IVideoFrameSampler`](/docs/reference/wiki/data/ivideoframesampler/) | Interface for video frame sampling strategies. |

## Enums (26)

| Type | Summary |
|:-----|:--------|
| [`ActiveLearningStrategy`](/docs/reference/wiki/data/activelearningstrategy/) | Active learning selection strategies. |
| [`ArcVariant`](/docs/reference/wiki/data/arcvariant/) | Variant of the ARC benchmark to load. |
| [`BatchingStrategy`](/docs/reference/wiki/data/batchingstrategy/) | Defines strategies for batching tasks in meta-learning. |
| [`CacheEvictionPolicy`](/docs/reference/wiki/data/cacheevictionpolicy/) | Policy for evicting cache entries when the cache is full. |
| [`CitationDataset<T>`](/docs/reference/wiki/data/citationdataset/) | Available citation network datasets. |
| [`CoresetStrategy`](/docs/reference/wiki/data/coresetstrategy/) | Strategy for selecting coreset samples. |
| [`CurriculumOrder`](/docs/reference/wiki/data/curriculumorder/) | Order in which curriculum samples are presented. |
| [`CurriculumPacing`](/docs/reference/wiki/data/curriculumpacing/) | Pacing function controlling how fast the data pool grows. |
| [`CurriculumStage`](/docs/reference/wiki/data/curriculumstage/) | Represents stages in a meta-learning curriculum. |
| [`CurriculumStrategy`](/docs/reference/wiki/data/curriculumstrategy/) | Defines how the curriculum progresses over epochs. |
| [`DatasetSplit`](/docs/reference/wiki/data/datasetsplit/) | Standard dataset split options. |
| [`GlueTask`](/docs/reference/wiki/data/gluetask/) | GLUE benchmark sub-tasks. |
| [`ImageTensorLayout`](/docs/reference/wiki/data/imagetensorlayout/) | Specifies the axis ordering for image tensors returned by vision data loaders. |
| [`M4Frequency`](/docs/reference/wiki/data/m4frequency/) | Frequency categories in the M4 Competition. |
| [`MemoryCacheEvictionPolicy`](/docs/reference/wiki/data/memorycacheevictionpolicy/) | Policy for evicting items from the in-memory cache when it's full. |
| [`ModalityType`](/docs/reference/wiki/data/modalitytype/) | Identifies the type of data modality in a multimodal sample. |
| [`MolecularDataset<T>`](/docs/reference/wiki/data/moleculardataset/) | Available molecular datasets. |
| [`OGBTask<T>`](/docs/reference/wiki/data/ogbtask/) | OGB task types. |
| [`PointPaddingStrategy`](/docs/reference/wiki/data/pointpaddingstrategy/) | Strategies for padding point clouds when fewer points than requested exist. |
| [`PointSamplingStrategy`](/docs/reference/wiki/data/pointsamplingstrategy/) | Strategies for sampling points from a point cloud. |
| [`PruneStrategy`](/docs/reference/wiki/data/prunestrategy/) | Strategy for selecting samples to prune. |
| [`QueryStrategy`](/docs/reference/wiki/data/querystrategy/) | Strategy for selecting samples in active learning. |
| [`ScanNetInputFormat`](/docs/reference/wiki/data/scannetinputformat/) | Supported ScanNet input data formats. |
| [`ScanNetLabelMode`](/docs/reference/wiki/data/scannetlabelmode/) | ScanNet label mapping modes. |
| [`SuperGlueTask`](/docs/reference/wiki/data/supergluetask/) | SuperGLUE benchmark sub-tasks. |
| [`UncertaintyPolicy`](/docs/reference/wiki/data/uncertaintypolicy/) | Policy for handling uncertain labels in CheXpert. |

## Options & Configuration (114)

| Type | Summary |
|:-----|:--------|
| [`ActiveLearningQueryStrategyOptions`](/docs/reference/wiki/data/activelearningquerystrategyoptions/) | Configuration options for active learning query strategies. |
| [`Ade20kDataLoaderOptions`](/docs/reference/wiki/data/ade20kdataloaderoptions/) | Configuration options for the ADE20K semantic segmentation data loader. |
| [`AgNewsDataLoaderOptions`](/docs/reference/wiki/data/agnewsdataloaderoptions/) | Configuration options for the AG News topic classification data loader. |
| [`ArcDataLoaderOptions`](/docs/reference/wiki/data/arcdataloaderoptions/) | Configuration options for the AI2 Reasoning Challenge (ARC) data loader. |
| [`ArrowDatasetOptions`](/docs/reference/wiki/data/arrowdatasetoptions/) | Configuration options for Apache Arrow-based dataset access. |
| [`AudioFileDatasetOptions`](/docs/reference/wiki/data/audiofiledatasetoptions/) | Configuration options for the `AudioFileDataset`. |
| [`AudioSetDataLoaderOptions`](/docs/reference/wiki/data/audiosetdataloaderoptions/) | Configuration options for the AudioSet data loader. |
| [`BigEarthNetDataLoaderOptions`](/docs/reference/wiki/data/bigearthnetdataloaderoptions/) | Configuration options for the BigEarthNet data loader. |
| [`CachingDataLoaderOptions`](/docs/reference/wiki/data/cachingdataloaderoptions/) | Configuration options for caching data loader. |
| [`Caltech101DataLoaderOptions`](/docs/reference/wiki/data/caltech101dataloaderoptions/) | Configuration options for the Caltech-101 image classification data loader (Fei-Fei et al. |
| [`CelebADataLoaderOptions`](/docs/reference/wiki/data/celebadataloaderoptions/) | Configuration options for the CelebA face attributes data loader. |
| [`CheXpertDataLoaderOptions`](/docs/reference/wiki/data/chexpertdataloaderoptions/) | Configuration options for the CheXpert data loader. |
| [`ChestXray14DataLoaderOptions`](/docs/reference/wiki/data/chestxray14dataloaderoptions/) | Configuration options for the NIH Chest X-ray 14 data loader. |
| [`Cifar100DataLoaderOptions`](/docs/reference/wiki/data/cifar100dataloaderoptions/) | Configuration options for the CIFAR-100 data loader. |
| [`Cifar10DataLoaderOptions`](/docs/reference/wiki/data/cifar10dataloaderoptions/) | Configuration options for the CIFAR-10 data loader. |
| [`CityscapesDataLoaderOptions`](/docs/reference/wiki/data/cityscapesdataloaderoptions/) | Configuration options for the Cityscapes semantic segmentation loader. |
| [`CnnDailyMailDataLoaderOptions`](/docs/reference/wiki/data/cnndailymaildataloaderoptions/) | Configuration options for the CNN/DailyMail summarization loader (Hermann et al. |
| [`CocoDetectionDataLoaderOptions`](/docs/reference/wiki/data/cocodetectiondataloaderoptions/) | Configuration options for the COCO Detection data loader. |
| [`CommonVoiceDataLoaderOptions`](/docs/reference/wiki/data/commonvoicedataloaderoptions/) | Configuration options for the Mozilla Common Voice data loader. |
| [`CoresetSelectorOptions`](/docs/reference/wiki/data/coresetselectoroptions/) | Configuration options for coreset selection. |
| [`CurriculumDataSchedulerOptions`](/docs/reference/wiki/data/curriculumdatascheduleroptions/) | Configuration options for curriculum data scheduling. |
| [`DataPrunerOptions`](/docs/reference/wiki/data/datapruneroptions/) | Configuration options for data pruning. |
| [`DatasetDistillerOptions`](/docs/reference/wiki/data/datasetdistilleroptions/) | Configuration options for dataset distillation. |
| [`DiskCacheOptions`](/docs/reference/wiki/data/diskcacheoptions/) | Configuration options for disk-based pipeline caching. |
| [`DocVqaDataLoaderOptions`](/docs/reference/wiki/data/docvqadataloaderoptions/) | Configuration options for the DocVQA (Document Visual Question Answering) data loader. |
| [`DtdDataLoaderOptions`](/docs/reference/wiki/data/dtddataloaderoptions/) | Configuration options for the Describable Textures Dataset (DTD) loader (Cimpoi et al. |
| [`ElasticDistributedSamplerOptions`](/docs/reference/wiki/data/elasticdistributedsampleroptions/) | Configuration options for elastic distributed sampling. |
| [`Enwik8DataLoaderOptions`](/docs/reference/wiki/data/enwik8dataloaderoptions/) | Configuration options for the enwik8 character-level LM data loader. |
| [`Esc50DataLoaderOptions`](/docs/reference/wiki/data/esc50dataloaderoptions/) | Configuration options for the ESC-50 data loader. |
| [`EuroSatDataLoaderOptions`](/docs/reference/wiki/data/eurosatdataloaderoptions/) | Configuration options for the EuroSAT data loader. |
| [`ExactHashDeduplicatorOptions`](/docs/reference/wiki/data/exacthashdeduplicatoroptions/) | Configuration options for exact hash-based deduplication. |
| [`FMoWDataLoaderOptions`](/docs/reference/wiki/data/fmowdataloaderoptions/) | Configuration options for the Functional Map of the World (fMoW) data loader. |
| [`FashionMnistDataLoaderOptions`](/docs/reference/wiki/data/fashionmnistdataloaderoptions/) | Configuration options for the Fashion-MNIST data loader. |
| [`FleursDataLoaderOptions`](/docs/reference/wiki/data/fleursdataloaderoptions/) | Configuration options for the FLEURS data loader. |
| [`Flowers102DataLoaderOptions`](/docs/reference/wiki/data/flowers102dataloaderoptions/) | Configuration options for the Oxford Flowers-102 dataset (Nilsback & Zisserman 2008). |
| [`Food101DataLoaderOptions`](/docs/reference/wiki/data/food101dataloaderoptions/) | Configuration options for the Food-101 image classification data loader. |
| [`Fsd50kDataLoaderOptions`](/docs/reference/wiki/data/fsd50kdataloaderoptions/) | Configuration options for the FSD50K data loader. |
| [`GigaSpeechDataLoaderOptions`](/docs/reference/wiki/data/gigaspeechdataloaderoptions/) | Configuration options for the GigaSpeech data loader. |
| [`GlueDataLoaderOptions`](/docs/reference/wiki/data/gluedataloaderoptions/) | Configuration options for the GLUE benchmark data loader. |
| [`Gsm8kDataLoaderOptions`](/docs/reference/wiki/data/gsm8kdataloaderoptions/) | Configuration options for the GSM8K math word-problem benchmark. |
| [`GtzanDataLoaderOptions`](/docs/reference/wiki/data/gtzandataloaderoptions/) | Configuration options for the GTZAN music genre classification loader. |
| [`Hdf5DatasetOptions`](/docs/reference/wiki/data/hdf5datasetoptions/) | Configuration options for HDF5 dataset access. |
| [`HellaswagDataLoaderOptions`](/docs/reference/wiki/data/hellaswagdataloaderoptions/) | Configuration options for the HellaSwag commonsense NLI benchmark (Zellers et al. |
| [`HeuristicTextFilterOptions`](/docs/reference/wiki/data/heuristictextfilteroptions/) | Configuration options for heuristic text quality filtering. |
| [`Hmdb51DataLoaderOptions`](/docs/reference/wiki/data/hmdb51dataloaderoptions/) | Configuration options for the HMDB51 data loader. |
| [`HumanEvalDataLoaderOptions`](/docs/reference/wiki/data/humanevaldataloaderoptions/) | Configuration options for the HumanEval Python code-generation benchmark (Chen et al. |
| [`INaturalistDataLoaderOptions`](/docs/reference/wiki/data/inaturalistdataloaderoptions/) | Configuration options for the iNaturalist data loader. |
| [`ImageFolderDatasetOptions`](/docs/reference/wiki/data/imagefolderdatasetoptions/) | Configuration options for the `ImageFolderDataset`. |
| [`ImageNet1kDataLoaderOptions`](/docs/reference/wiki/data/imagenet1kdataloaderoptions/) | Configuration options for the ImageNet-1K (ILSVRC 2012) data loader. |
| [`ImageNet21kDataLoaderOptions`](/docs/reference/wiki/data/imagenet21kdataloaderoptions/) | Configuration options for the ImageNet-21K data loader. |
| [`ImageQualityFilterOptions`](/docs/reference/wiki/data/imagequalityfilteroptions/) | Configuration options for image quality filtering. |
| [`Imdb50kDataLoaderOptions`](/docs/reference/wiki/data/imdb50kdataloaderoptions/) | Configuration options for the IMDB 50k sentiment analysis data loader. |
| [`Kinetics400DataLoaderOptions`](/docs/reference/wiki/data/kinetics400dataloaderoptions/) | Configuration options for the Kinetics-400 data loader. |
| [`KittiDataLoaderOptions`](/docs/reference/wiki/data/kittidataloaderoptions/) | Configuration options for the KITTI 3D object detection data loader. |
| [`LanguageIdFilterOptions`](/docs/reference/wiki/data/languageidfilteroptions/) | Configuration options for language identification filtering. |
| [`LibriSpeechDataLoaderOptions`](/docs/reference/wiki/data/librispeechdataloaderoptions/) | Configuration options for the LibriSpeech data loader. |
| [`LjSpeechDataLoaderOptions`](/docs/reference/wiki/data/ljspeechdataloaderoptions/) | Configuration options for the LJSpeech 1.1 data loader (Ito & Johnson 2017). |
| [`LmdbDatasetOptions`](/docs/reference/wiki/data/lmdbdatasetoptions/) | Configuration options for LMDB-based dataset access. |
| [`MaestroDataLoaderOptions`](/docs/reference/wiki/data/maestrodataloaderoptions/) | Configuration options for the MAESTRO data loader. |
| [`MathDataLoaderOptions`](/docs/reference/wiki/data/mathdataloaderoptions/) | Configuration options for the Hendrycks MATH benchmark loader. |
| [`MbppDataLoaderOptions`](/docs/reference/wiki/data/mbppdataloaderoptions/) | Configuration options for the Mostly Basic Python Problems (MBPP) loader (Austin et al. |
| [`MidEpochCheckpointerOptions`](/docs/reference/wiki/data/midepochcheckpointeroptions/) | Configuration options for mid-epoch checkpointing. |
| [`MinHashDeduplicatorOptions`](/docs/reference/wiki/data/minhashdeduplicatoroptions/) | Configuration options for MinHash-based near-duplicate detection. |
| [`MmluDataLoaderOptions`](/docs/reference/wiki/data/mmludataloaderoptions/) | Configuration options for the MMLU (Massive Multitask Language Understanding) loader (Hendrycks et al. |
| [`MnistDataLoaderOptions`](/docs/reference/wiki/data/mnistdataloaderoptions/) | Configuration options for the MNIST data loader. |
| [`ModelNet40ClassificationDataLoaderOptions`](/docs/reference/wiki/data/modelnet40classificationdataloaderoptions/) | Configuration options for the ModelNet40 classification data loader. |
| [`MultiSourceMixerOptions`](/docs/reference/wiki/data/multisourcemixeroptions/) | Configuration options for multi-source data mixing. |
| [`Musdb18DataLoaderOptions`](/docs/reference/wiki/data/musdb18dataloaderoptions/) | Configuration options for the MUSDB18 data loader. |
| [`NsynthDataLoaderOptions`](/docs/reference/wiki/data/nsynthdataloaderoptions/) | Configuration for the NSynth (Neural Synth) audio dataset loader (Engel et al. |
| [`NuScenesDataLoaderOptions`](/docs/reference/wiki/data/nuscenesdataloaderoptions/) | Configuration options for the nuScenes data loader. |
| [`OpenImagesDataLoaderOptions`](/docs/reference/wiki/data/openimagesdataloaderoptions/) | Configuration options for the Open Images V7 data loader. |
| [`OxfordPetsDataLoaderOptions`](/docs/reference/wiki/data/oxfordpetsdataloaderoptions/) | Configuration options for the Oxford-IIIT Pet dataset loader (Parkhi et al. |
| [`ParallelBatchLoaderConfig`](/docs/reference/wiki/data/parallelbatchloaderconfig/) | Configuration for parallel batch loading. |
| [`ParquetDataLoaderOptions`](/docs/reference/wiki/data/parquetdataloaderoptions/) | Configuration options for the `ParquetDataLoader`. |
| [`PascalVocDataLoaderOptions`](/docs/reference/wiki/data/pascalvocdataloaderoptions/) | Configuration options for the Pascal VOC data loader. |
| [`PennTreebankDataLoaderOptions`](/docs/reference/wiki/data/penntreebankdataloaderoptions/) | Configuration options for the Penn Treebank (PTB) data loader. |
| [`PerplexityFilterOptions`](/docs/reference/wiki/data/perplexityfilteroptions/) | Configuration options for perplexity-based text quality filtering. |
| [`Places365DataLoaderOptions`](/docs/reference/wiki/data/places365dataloaderoptions/) | Configuration options for the Places365 data loader. |
| [`PrefetchDataLoaderOptions`](/docs/reference/wiki/data/prefetchdataloaderoptions/) | Configuration options for prefetch-enabled data loading. |
| [`ProteinDataLoaderOptions`](/docs/reference/wiki/data/proteindataloaderoptions/) | Configuration options for the protein structure graph data loader. |
| [`PubLayNetDataLoaderOptions`](/docs/reference/wiki/data/publaynetdataloaderoptions/) | Configuration options for the PubLayNet document layout analysis data loader. |
| [`Qm9DataLoaderOptions`](/docs/reference/wiki/data/qm9dataloaderoptions/) | Configuration options for the QM9 molecular property prediction data loader. |
| [`RetinalFundusDataLoaderOptions`](/docs/reference/wiki/data/retinalfundusdataloaderoptions/) | Configuration options for the Retinal Fundus data loader. |
| [`ScanNetSemanticSegmentationDataLoaderOptions`](/docs/reference/wiki/data/scannetsemanticsegmentationdataloaderoptions/) | Configuration options for the ScanNet semantic segmentation data loader. |
| [`SemanticDeduplicatorOptions`](/docs/reference/wiki/data/semanticdeduplicatoroptions/) | Configuration options for semantic-level deduplication using embeddings. |
| [`SemanticKittiDataLoaderOptions`](/docs/reference/wiki/data/semantickittidataloaderoptions/) | Configuration options for the SemanticKITTI data loader. |
| [`ShapeNetCorePartSegmentationDataLoaderOptions`](/docs/reference/wiki/data/shapenetcorepartsegmentationdataloaderoptions/) | Configuration options for the ShapeNetCore part segmentation data loader. |
| [`ShardedStreamingDatasetOptions`](/docs/reference/wiki/data/shardedstreamingdatasetoptions/) | Configuration options for the sharded streaming dataset. |
| [`SkinLesionDataLoaderOptions`](/docs/reference/wiki/data/skinlesiondataloaderoptions/) | Configuration options for the ISIC Skin Lesion data loader. |
| [`SomethingSomethingV2DataLoaderOptions`](/docs/reference/wiki/data/somethingsomethingv2dataloaderoptions/) | Configuration options for the Something-Something V2 data loader. |
| [`SpeechCommandsDataLoaderOptions`](/docs/reference/wiki/data/speechcommandsdataloaderoptions/) | Configuration options for the Google Speech Commands v2 data loader. |
| [`SquadDataLoaderOptions`](/docs/reference/wiki/data/squaddataloaderoptions/) | Configuration options for the SQuAD data loader. |
| [`StanfordCarsDataLoaderOptions`](/docs/reference/wiki/data/stanfordcarsdataloaderoptions/) | Configuration for the Stanford Cars dataset loader (Krause et al. |
| [`Stl10DataLoaderOptions`](/docs/reference/wiki/data/stl10dataloaderoptions/) | Configuration options for the STL-10 image classification dataset (Coates et al. |
| [`StreamingTextDatasetOptions`](/docs/reference/wiki/data/streamingtextdatasetoptions/) | Configuration options for the streaming text dataset. |
| [`SuperGlueDataLoaderOptions`](/docs/reference/wiki/data/supergluedataloaderoptions/) | Configuration options for the SuperGLUE benchmark data loader. |
| [`SvhnDataLoaderOptions`](/docs/reference/wiki/data/svhndataloaderoptions/) | Configuration options for the SVHN (Street View House Numbers) loader (Netzer et al. |
| [`TemporalGraphDataLoaderOptions`](/docs/reference/wiki/data/temporalgraphdataloaderoptions/) | Configuration options for the temporal graph data loader. |
| [`TimitDataLoaderOptions`](/docs/reference/wiki/data/timitdataloaderoptions/) | Configuration for the TIMIT acoustic-phonetic continuous-speech corpus loader (Garofolo et al. |
| [`TinyImageNetDataLoaderOptions`](/docs/reference/wiki/data/tinyimagenetdataloaderoptions/) | Configuration options for the Tiny ImageNet (200-class, 64×64) data loader. |
| [`TinyStoriesDataLoaderOptions`](/docs/reference/wiki/data/tinystoriesdataloaderoptions/) | Configuration options for the TinyStories data loader (Eldan & Li, 2023). |
| [`TruthfulQaDataLoaderOptions`](/docs/reference/wiki/data/truthfulqadataloaderoptions/) | Configuration for the TruthfulQA benchmark loader (Lin et al. |
| [`Ucf101DataLoaderOptions`](/docs/reference/wiki/data/ucf101dataloaderoptions/) | Configuration options for the UCF101 data loader. |
| [`UrbanSound8kDataLoaderOptions`](/docs/reference/wiki/data/urbansound8kdataloaderoptions/) |  |
| [`VctkDataLoaderOptions`](/docs/reference/wiki/data/vctkdataloaderoptions/) | Configuration for the VCTK Corpus 0.92 multi-speaker TTS loader (Yamagishi et al. |
| [`VideoFrameDatasetOptions`](/docs/reference/wiki/data/videoframedatasetoptions/) | Configuration options for the `VideoFrameDataset`. |
| [`VoxPopuliDataLoaderOptions`](/docs/reference/wiki/data/voxpopulidataloaderoptions/) | Configuration options for the VoxPopuli data loader. |
| [`WaymoDataLoaderOptions`](/docs/reference/wiki/data/waymodataloaderoptions/) | Configuration options for the Waymo Open Dataset data loader. |
| [`WebDatasetOptions`](/docs/reference/wiki/data/webdatasetoptions/) | Configuration options for the WebDataset loader. |
| [`WikiText103DataLoaderOptions`](/docs/reference/wiki/data/wikitext103dataloaderoptions/) | Configuration options for the WikiText-103 data loader. |
| [`WikiText2DataLoaderOptions`](/docs/reference/wiki/data/wikitext2dataloaderoptions/) | Configuration options for the WikiText-2 data loader. |
| [`Wikidata5mDataLoaderOptions`](/docs/reference/wiki/data/wikidata5mdataloaderoptions/) | Configuration options for the Wikidata5M knowledge graph data loader. |
| [`XSumDataLoaderOptions`](/docs/reference/wiki/data/xsumdataloaderoptions/) | Configuration options for the XSum extreme summarization loader (Narayan et al. |
| [`ZincDataLoaderOptions`](/docs/reference/wiki/data/zincdataloaderoptions/) | Configuration options for the ZINC molecular dataset data loader. |

## Helpers & Utilities (4)

| Type | Summary |
|:-----|:--------|
| [`DataLoaders`](/docs/reference/wiki/data/dataloaders/) | Static factory class for creating data loaders with beginner-friendly methods. |
| [`DataPipelineExtensions`](/docs/reference/wiki/data/datapipelineextensions/) | Extension methods for creating data pipelines from various sources. |
| [`ParallelBatchLoaderExtensions`](/docs/reference/wiki/data/parallelbatchloaderextensions/) | Provides extension methods for parallel batch loading. |
| [`Samplers`](/docs/reference/wiki/data/samplers/) | Static factory class for creating data samplers with beginner-friendly methods. |


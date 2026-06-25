---
title: "DataLoaders"
description: "Static factory class for creating data loaders with beginner-friendly methods."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Data.Loaders`

Static factory class for creating data loaders with beginner-friendly methods.

## For Beginners

This is your starting point for loading data into AiDotNet!
Choose the method that matches your data format:

**Common Patterns:**
```cs
// From arrays (simplest for small datasets)
var loader = DataLoaders.FromArrays(features, labels);

// From Matrix and Vector (most common for ML)
var loader = DataLoaders.FromMatrixVector(featureMatrix, labelVector);

// From Tensors (for deep learning)
var loader = DataLoaders.FromTensors(inputTensor, outputTensor);
```

All loaders support:

- Batching: `loader.BatchSize = 32;`
- Shuffling: `loader.Shuffle();`
- Splitting: `var (train, val, test) = loader.Split();`

## How It Works

DataLoaders provides the easiest way to create data loaders for common scenarios.
It follows a factory pattern with static methods that handle type inference and
common configurations automatically.

## Methods

| Method | Summary |
|:-----|:--------|
| `Ade20k(Ade20kDataLoaderOptions)` | Creates an ADE20K semantic segmentation data loader (~25K images, 150 classes). |
| `AudioFiles(AudioFileDatasetOptions)` | Creates an audio file dataset loader that reads WAV/PCM audio files from directories. |
| `AudioSet(AudioSetDataLoaderOptions)` | Creates an AudioSet large-scale audio event dataset loader (2M+ clips, 527 categories). |
| `BigEarthNet(BigEarthNetDataLoaderOptions)` | Creates a BigEarthNet multi-label remote sensing dataset loader (590K Sentinel-2 patches, 19 or 43 classes). |
| `CelebA(CelebADataLoaderOptions)` | Creates a CelebA face attributes data loader (~200K images, 40 binary attributes). |
| `CheXpert(CheXpertDataLoaderOptions)` | Creates a CheXpert chest radiograph dataset loader (224K images, 14 observations with uncertainty). |
| `ChestXray14(ChestXray14DataLoaderOptions)` | Creates a NIH Chest X-ray 14 multi-label classification dataset loader (112K images, 14 diseases). |
| `Cifar10(Cifar10DataLoaderOptions)` | Creates a CIFAR-10 dataset loader (50k train / 10k test, 32x32 RGB, 10 classes). |
| `Cifar100(Cifar100DataLoaderOptions)` | Creates a CIFAR-100 dataset loader (50k train / 10k test, 32x32 RGB, 100 classes). |
| `CocoDetection(CocoDetectionDataLoaderOptions)` | Creates a COCO 2017 object detection dataset loader (118K train / 5K val, 80 categories). |
| `CommonVoice(CommonVoiceDataLoaderOptions)` | Creates a Mozilla Common Voice multilingual speech dataset loader (19K+ hours, 100+ languages). |
| `DocVqa(DocVqaDataLoaderOptions)` | Creates a DocVQA document visual question answering data loader. |
| `Empty` | Creates an empty data loader placeholder (useful for meta-learning or custom scenarios). |
| `Esc50(Esc50DataLoaderOptions)` | Creates an ESC-50 environmental sound classification dataset loader (2000 clips, 50 classes). |
| `EuroSat(EuroSatDataLoaderOptions)` | Creates an EuroSAT land use/land cover classification dataset loader (27K patches, 64x64, 10 classes). |
| `FMoW(FMoWDataLoaderOptions)` | Creates an fMoW (Functional Map of the World) satellite imagery dataset loader (1M+ images, 62 categories). |
| `FashionMnist(FashionMnistDataLoaderOptions)` | Creates a Fashion-MNIST dataset loader (60k train / 10k test, 28x28 grayscale clothing images). |
| `Fleurs(FleursDataLoaderOptions)` | Creates a FLEURS multilingual speech benchmark loader (102 languages, ~12 hours per language). |
| `FromArrays([0:,0:],[])` | Creates a data loader from 2D feature array and 1D label array. |
| `FromArrays([],[])` | Creates a data loader from 1D feature array (single feature) and 1D label array. |
| `FromArrays([][],[])` | Creates a data loader from jagged feature array and 1D label array. |
| `FromCsv(String,Func<String,Int32,ValueTuple<,>>,Int32,Boolean,Int32)` | Creates a streaming data loader from a CSV file. |
| `FromDirectory(String,String,Func<String,CancellationToken,Task<ValueTuple<,>>>,Int32,SearchOption,Int32,Int32)` | Creates a streaming data loader from a directory of files. |
| `FromJsonl(String,Func<JObject,ValueTuple<Tensor<>,Tensor<>>>,Int32,String,String,Int32)` | Creates a typed JSONL data loader for reading JSON Lines files, compatible with `AiModelBuilder.ConfigureDataLoader()`. |
| `FromJsonl(String[],Func<JObject,ValueTuple<Tensor<>,Tensor<>>>,Int32,String,String,Int32)` | Creates a typed JSONL data loader for reading multiple JSONL files. |
| `FromLeafFederatedJsonFiles(String,String,LeafFederatedDatasetLoadOptions)` | Creates a LEAF federated data loader from LEAF benchmark JSON files. |
| `FromMatrices(Matrix<>,Matrix<>)` | Creates a data loader from a feature Matrix and label Matrix (for multi-output regression). |
| `FromMatrix(Matrix<>)` | Creates a data loader from a feature Matrix only (for unsupervised learning like clustering). |
| `FromMatrixVector(Matrix<>,Vector<>)` | Creates a data loader from a feature Matrix and label Vector. |
| `FromParquet(ParquetDataLoaderOptions)` | Creates a Parquet file data loader for reading columnar data from Apache Parquet files. |
| `FromShards(String[],Func<Byte[],ValueTuple<Tensor<>,Tensor<>>>,Int32,ShardedStreamingDatasetOptions)` | Creates a typed sharded streaming data loader for deterministic, resumable reading, compatible with `AiModelBuilder.ConfigureDataLoader()`. |
| `FromTensorVector(Tensor<>,Vector<>)` | Creates a data loader from a Tensor of features and a Vector of labels. |
| `FromTensors(Tensor<>,Tensor<>)` | Creates a data loader from input and output Tensors. |
| `FromTextDocuments(String[],Vector<>,CountVectorizer<>)` | Creates a data loader from text documents using a Count vectorizer. |
| `FromTextDocuments(String[],Vector<>,HashingVectorizer<>)` | Creates a data loader from text documents using a Hashing vectorizer. |
| `FromTextDocuments(String[],Vector<>,ITextVectorizer<>)` | Creates a data loader from text documents using any text vectorizer. |
| `FromTextDocuments(String[],Vector<>,TfidfVectorizer<>)` | Creates a data loader from text documents using a TF-IDF vectorizer. |
| `FromTextDocuments(String[],[],CountVectorizer<>)` | Creates a data loader from text documents using a Count vectorizer with array labels. |
| `FromTextDocuments(String[],[],HashingVectorizer<>)` | Creates a data loader from text documents using a Hashing vectorizer with array labels. |
| `FromTextDocuments(String[],[],ITextVectorizer<>)` | Creates a data loader from text documents using any text vectorizer with array labels. |
| `FromTextDocuments(String[],[],TfidfVectorizer<>)` | Creates a data loader from text documents using a TF-IDF vectorizer with array labels. |
| `FromWebDataset(String,Func<Dictionary<String,Byte[]>,ValueTuple<Tensor<>,Tensor<>>>,Int32,WebDatasetOptions)` | Creates a typed WebDataset data loader for a single TAR archive. |
| `FromWebDataset(String[],Func<Dictionary<String,Byte[]>,ValueTuple<Tensor<>,Tensor<>>>,Int32,WebDatasetOptions)` | Creates a typed WebDataset data loader for reading samples from TAR archives, compatible with `AiModelBuilder.ConfigureDataLoader()`. |
| `Fsd50k(Fsd50kDataLoaderOptions)` | Creates an FSD50K audio event dataset loader (51,197 clips, 200 sound event classes). |
| `GigaSpeech(GigaSpeechDataLoaderOptions)` | Creates a GigaSpeech multi-domain English ASR dataset loader (up to 10K hours). |
| `Glue(GlueDataLoaderOptions)` | Creates a GLUE benchmark data loader (9 NLU tasks for evaluating language understanding). |
| `Hmdb51(Hmdb51DataLoaderOptions)` | Creates an HMDB-51 video action recognition data loader (51 classes, 6.8K clips). |
| `INaturalist(INaturalistDataLoaderOptions)` | Creates an iNaturalist species classification dataset loader (~2.7M images, 10,000 species). |
| `ImageFolder(ImageFolderDatasetOptions)` | Creates an image folder dataset loader that reads images from a directory structure where each subdirectory name is a class label. |
| `ImageNet1k(ImageNet1kDataLoaderOptions)` | Creates an ImageNet-1K (ILSVRC 2012) dataset loader (~1.28M train / 50K val, 1000 classes). |
| `ImageNet21k(ImageNet21kDataLoaderOptions)` | Creates an ImageNet-21K dataset loader (~14.2M images, 21,841 categories). |
| `Imdb50k(Imdb50kDataLoaderOptions)` | Creates an IMDB 50k movie review sentiment dataset loader (25k train / 25k test). |
| `Kinetics400(Kinetics400DataLoaderOptions)` | Creates a Kinetics-400 video action recognition data loader (400 classes, ~300K clips). |
| `Kitti(KittiDataLoaderOptions)` | Creates a KITTI 3D object detection data loader (LiDAR point clouds with bounding boxes). |
| `LibriSpeech(LibriSpeechDataLoaderOptions)` | Creates a LibriSpeech ASR dataset loader (~1000 hours of 16kHz English speech). |
| `Maestro(MaestroDataLoaderOptions)` | Creates a MAESTRO piano performance dataset loader (~200 hours, aligned MIDI and audio). |
| `Mnist(MnistDataLoaderOptions)` | Creates a MNIST handwritten digit dataset loader (60k train / 10k test, 28x28 grayscale). |
| `ModelNet40Classification(ModelNet40ClassificationDataLoaderOptions)` | Creates a ModelNet40 classification data loader. |
| `Multimodal` | Creates a new empty multimodal dataset for building vision-language or multi-modal training samples. |
| `Multimodal(IEnumerable<MultimodalSample<>>)` | Creates a multimodal dataset pre-populated with samples. |
| `Musdb18(Musdb18DataLoaderOptions)` | Creates a MUSDB18 music source separation dataset loader (150 tracks, 4 stems). |
| `NuScenes(NuScenesDataLoaderOptions)` | Creates a nuScenes 3D object detection data loader (LiDAR point clouds, 23 classes). |
| `OpenImages(OpenImagesDataLoaderOptions)` | Creates an Open Images V7 object detection dataset loader (~9M images, 600 categories). |
| `PascalVoc(PascalVocDataLoaderOptions)` | Creates a Pascal VOC object detection dataset loader (20 categories, XML annotations). |
| `Places365(Places365DataLoaderOptions)` | Creates a Places365 scene recognition dataset loader (1.8M train / 36.5K val, 365 scene categories). |
| `Protein(ProteinDataLoaderOptions)` | Creates a protein structure graph classification data loader. |
| `PubLayNet(PubLayNetDataLoaderOptions)` | Creates a PubLayNet document layout analysis data loader (~360K images, 5 categories). |
| `Qm9(Qm9DataLoaderOptions)` | Creates a QM9 molecular property prediction data loader (~134K molecules, 19 properties). |
| `RetinalFundus(RetinalFundusDataLoaderOptions)` | Creates a Retinal Fundus diabetic retinopathy grading dataset loader (5 severity levels). |
| `ScanNetSemanticSegmentation(ScanNetSemanticSegmentationDataLoaderOptions)` | Creates a ScanNet semantic segmentation data loader. |
| `SemanticKitti(SemanticKittiDataLoaderOptions)` | Creates a SemanticKITTI per-point semantic segmentation data loader (28 classes). |
| `ShapeNetCorePartSegmentation(ShapeNetCorePartSegmentationDataLoaderOptions)` | Creates a ShapeNetCore part segmentation data loader. |
| `SkinLesion(SkinLesionDataLoaderOptions)` | Creates an ISIC Skin Lesion classification dataset loader (~25K images, 8 diagnostic categories). |
| `SomethingSomethingV2(SomethingSomethingV2DataLoaderOptions)` | Creates a Something-Something V2 video understanding data loader (174 classes, ~220K clips). |
| `Squad(SquadDataLoaderOptions)` | Creates a SQuAD reading comprehension data loader (100K+ question-answer pairs). |
| `Streaming(Int32,Func<Int32,CancellationToken,Task<ValueTuple<,>>>,Int32,Int32,Int32)` | Creates a streaming data loader that reads samples on-demand. |
| `StreamingText(StreamingTextDatasetOptions)` | Creates a streaming text dataset for LLM pre-training on large text corpora. |
| `SuperGlue(SuperGlueDataLoaderOptions)` | Creates a SuperGLUE benchmark data loader (8 advanced NLU tasks). |
| `TemporalGraph(TemporalGraphDataLoaderOptions)` | Creates a temporal graph data loader for dynamic link prediction. |
| `Ucf101(Ucf101DataLoaderOptions)` | Creates a UCF-101 video action recognition data loader (101 classes, 13.3K clips). |
| `UrbanSound8k(UrbanSound8kDataLoaderOptions)` | Creates an UrbanSound8K urban sound classification dataset loader (8732 clips, 10 classes). |
| `VideoFrames(VideoFrameDatasetOptions)` | Creates a video frame dataset loader that extracts frames from video directories. |
| `VoxPopuli(VoxPopuliDataLoaderOptions)` | Creates a VoxPopuli multilingual speech dataset loader (400K+ hours, 23 languages). |
| `Waymo(WaymoDataLoaderOptions)` | Creates a Waymo Open Dataset 3D detection data loader (LiDAR point clouds). |
| `WikiText103(WikiText103DataLoaderOptions)` | Creates a WikiText-103 language modeling data loader (100M+ tokens from Wikipedia). |
| `Wikidata5m(Wikidata5mDataLoaderOptions)` | Creates a Wikidata5M knowledge graph link prediction data loader (~5M entities, 21M triplets). |
| `WithBatchSize(Matrix<>,Vector<>,Int32)` | Creates a data loader with pre-configured batch size. |
| `WithCheckpointing(InputOutputDataLoaderBase<,,>)` | Wraps any data loader with stateful checkpointing support for mid-epoch resume. |
| `WithTransforms(InputOutputDataLoaderBase<,Tensor<>,Tensor<>>,ITransform<[],[]>)` | Wraps a data loader with a composable transform pipeline applied to features. |
| `Zinc(ZincDataLoaderOptions)` | Creates a ZINC molecular dataset data loader (~250K drug-like molecules). |


---
title: "ModelTask"
description: "Defines the specific task or capability that a machine learning model performs."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the specific task or capability that a machine learning model performs.

## For Beginners

This tells you what a model actually does — its job.
A model can perform multiple tasks. For example, a Vision Transformer might do
both Classification and FeatureExtraction. Knowing the task helps you pick
the right model for your specific problem.

## Fields

| Field | Summary |
|:-----|:--------|
| `ActionRecognition` | Recognizes and classifies human actions in video. |
| `AnomalyDetection` | Identifies unusual patterns or outliers that don't conform to expected behavior. |
| `AudioGeneration` | Generates audio content (music, sound effects, speech). |
| `BinaryClassification` | Assigns inputs to one of exactly two categories. |
| `CausalInference` | Estimates cause-and-effect relationships from observational data. |
| `Classification` | Assigns inputs to one of several predefined categories. |
| `Clustering` | Groups similar data points together without predefined labels. |
| `CodeGeneration` | Generates source code from natural language or other code. |
| `Compression` | Reduces the size or dimensionality of data while preserving essential information. |
| `Denoising` | Removes noise or unwanted artifacts from data. |
| `DepthEstimation` | Estimates the distance of objects from the camera or sensor. |
| `Detection` | Locates and identifies objects within an input (typically an image). |
| `DimensionalityReduction` | Reduces the number of features or dimensions in data while preserving structure. |
| `DrugDiscovery` | Discovers or designs new drug molecules. |
| `Editing` | Modifies or manipulates existing data based on instructions, conditions, or references. |
| `Embedding` | Produces dense vector representations that capture semantic meaning. |
| `Enhancement` | Improves the overall quality of audio, images, or other signals. |
| `FeatureExtraction` | Extracts meaningful features or representations from raw data. |
| `Forecasting` | Predicts future values based on historical time-dependent data. |
| `FrameInterpolation` | Generates intermediate frames between existing video frames. |
| `Generation` | Creates new data (images, text, audio, etc.) that resembles the training data. |
| `GraphClassification` | Classifies entire graphs into categories. |
| `ImageEditing` | Modifies or manipulates images based on instructions or conditions. |
| `ImageToVideo` | Generates video by animating a single input image. |
| `Inpainting` | Fills in missing or masked regions of data. |
| `LinkPrediction` | Predicts missing edges between nodes in a graph. |
| `MolecularGeneration` | Predicts properties of molecular structures. |
| `MotionGeneration` | Generates realistic human or object motion sequences. |
| `MultiClassClassification` | Assigns inputs to one of three or more categories. |
| `MusicGeneration` | Generates music compositions. |
| `NamedEntityRecognition` | Identifies named entities (people, places, organizations) in text. |
| `NodeClassification` | Predicts labels or properties of nodes in a graph. |
| `ObjectDetection` | Detects and classifies objects in images. |
| `OpticalFlow` | Estimates the motion of pixels between consecutive frames. |
| `PoseEstimation` | Estimates body joint positions from images or video. |
| `ProteinFolding` | Predicts protein 3D structures from amino acid sequences. |
| `QuestionAnswering` | Answers questions based on given context or knowledge. |
| `Ranking` | Orders items by relevance or importance for a given query. |
| `Recommendation` | Suggests relevant items to users based on preferences and behavior. |
| `RecommendationFiltering` | Filters and recommends items based on user preferences. |
| `Regression` | Predicts a continuous numeric value from input features. |
| `Restoration` | Repairs, enhances, or recovers degraded data. |
| `Segmentation` | Assigns a label to every pixel or element in the input. |
| `SemanticSearch` | Finds relevant documents or passages for a query. |
| `SentimentAnalysis` | Determines the emotional tone of text. |
| `SignalProcessing` | Processes, analyzes, or transforms raw signals (audio, radio, sensor data). |
| `SourceSeparation` | Separates individual sources from a mixed signal. |
| `SpeechRecognition` | Converts spoken audio into text transcription. |
| `StyleTransfer` | Transfers the visual style of one image to the content of another. |
| `Summarization` | Condenses long text into shorter summaries. |
| `SuperResolution` | Increases the resolution or quality of data. |
| `SurvivalAnalysis` | Predicts time-to-event outcomes accounting for censored data. |
| `Synthesis` | Creates synthetic data that preserves statistical properties of real data. |
| `TabularPrediction` | Predicts outcomes from tabular/structured data. |
| `TextGeneration` | Generates text from prompts or context. |
| `TextToImage` | Generates images from text descriptions. |
| `TextToSpeech` | Converts text into spoken audio output. |
| `TextToVideo` | Generates video content from text descriptions. |
| `ThreeDGeneration` | Creates 3D models, meshes, or point clouds. |
| `Tracking` | Follows specific objects across multiple frames in a video. |
| `Translation` | Converts input from one form to another (e.g., between languages). |
| `VideoGeneration` | Generates video content from various inputs (text, image, or video). |
| `VideoToVideo` | Transforms an existing video into a modified version. |


namespace AiDotNet.Enums;

/// <summary>
/// Defines the specific task or capability that a machine learning model performs.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This tells you what a model actually does — its job.
/// A model can perform multiple tasks. For example, a Vision Transformer might do
/// both Classification and FeatureExtraction. Knowing the task helps you pick
/// the right model for your specific problem.
/// </para>
/// </remarks>
public enum ModelTask
{
    /// <summary>
    /// Assigns inputs to one of several predefined categories.
    /// Example: determining whether an email is spam or not spam.
    /// </summary>
    Classification,

    /// <summary>
    /// Predicts a continuous numeric value from input features.
    /// Example: predicting house prices based on size, location, and features.
    /// </summary>
    Regression,

    /// <summary>
    /// Creates new data (images, text, audio, etc.) that resembles the training data.
    /// Example: generating realistic face images or writing text continuations.
    /// </summary>
    Generation,

    /// <summary>
    /// Assigns a label to every pixel or element in the input.
    /// Example: identifying which pixels in a photo belong to a car, road, or sky.
    /// </summary>
    Segmentation,

    /// <summary>
    /// Locates and identifies objects within an input (typically an image).
    /// Example: drawing bounding boxes around people and cars in a photo.
    /// </summary>
    Detection,

    /// <summary>
    /// Produces dense vector representations that capture semantic meaning.
    /// Example: converting sentences to vectors where similar meanings are close together.
    /// </summary>
    Embedding,

    /// <summary>
    /// Converts input from one form to another (e.g., between languages).
    /// Example: translating English text to French.
    /// </summary>
    Translation,

    /// <summary>
    /// Predicts future values based on historical time-dependent data.
    /// Example: forecasting next month's sales based on past sales patterns.
    /// </summary>
    Forecasting,

    /// <summary>
    /// Groups similar data points together without predefined labels.
    /// Example: segmenting customers into groups based on purchasing behavior.
    /// </summary>
    Clustering,

    /// <summary>
    /// Removes noise or unwanted artifacts from data.
    /// Example: cleaning up a grainy photo or removing background noise from audio.
    /// </summary>
    Denoising,

    /// <summary>
    /// Increases the resolution or quality of data.
    /// Example: upscaling a low-resolution image to high resolution.
    /// </summary>
    SuperResolution,

    /// <summary>
    /// Transfers the visual style of one image to the content of another.
    /// Example: making a photo look like a Van Gogh painting.
    /// </summary>
    StyleTransfer,

    /// <summary>
    /// Fills in missing or masked regions of data.
    /// Example: removing an unwanted object from a photo and filling the gap naturally.
    /// </summary>
    Inpainting,

    /// <summary>
    /// Converts spoken audio into text transcription.
    /// Example: transcribing a podcast episode into written text.
    /// </summary>
    SpeechRecognition,

    /// <summary>
    /// Converts text into spoken audio output.
    /// Example: reading a news article aloud with a natural-sounding voice.
    /// </summary>
    TextToSpeech,

    /// <summary>
    /// Separates individual sources from a mixed signal.
    /// Example: isolating vocals from background music in a song.
    /// </summary>
    SourceSeparation,

    /// <summary>
    /// Identifies unusual patterns or outliers that don't conform to expected behavior.
    /// Example: detecting fraudulent credit card transactions.
    /// </summary>
    AnomalyDetection,

    /// <summary>
    /// Suggests relevant items to users based on preferences and behavior.
    /// Example: recommending movies a user might enjoy based on watch history.
    /// </summary>
    Recommendation,

    /// <summary>
    /// Orders items by relevance or importance for a given query.
    /// Example: ranking search results by relevance to a search query.
    /// </summary>
    Ranking,

    /// <summary>
    /// Estimates the distance of objects from the camera or sensor.
    /// Example: creating a depth map from a single 2D photograph.
    /// </summary>
    DepthEstimation,

    /// <summary>
    /// Estimates the motion of pixels between consecutive frames.
    /// Example: tracking how objects move between video frames for motion analysis.
    /// </summary>
    OpticalFlow,

    /// <summary>
    /// Follows specific objects across multiple frames in a video.
    /// Example: tracking a person as they walk through a surveillance video.
    /// </summary>
    Tracking,

    /// <summary>
    /// Recognizes and classifies human actions in video.
    /// Example: identifying whether a person is running, jumping, or sitting.
    /// </summary>
    ActionRecognition,

    /// <summary>
    /// Modifies or manipulates images based on instructions or conditions.
    /// Example: changing the color of a car in a photo or adding/removing objects.
    /// </summary>
    ImageEditing,

    /// <summary>
    /// Generates images from text descriptions.
    /// Example: creating a photorealistic image from "a cat sitting on a rainbow."
    /// </summary>
    TextToImage,

    /// <summary>
    /// Generates video content from text descriptions.
    /// Example: creating a short video clip from "a dog running on a beach."
    /// </summary>
    TextToVideo,

    /// <summary>
    /// Creates 3D models, meshes, or point clouds.
    /// Example: generating a 3D model of a chair from a text description or image.
    /// </summary>
    ThreeDGeneration,

    /// <summary>
    /// Generates realistic human or object motion sequences.
    /// Example: creating natural walking animations for a virtual character.
    /// </summary>
    MotionGeneration,

    /// <summary>
    /// Predicts time-to-event outcomes accounting for censored data.
    /// Example: estimating patient survival time after treatment.
    /// </summary>
    SurvivalAnalysis,

    /// <summary>
    /// Estimates cause-and-effect relationships from observational data.
    /// Example: determining the effect of a new drug on patient outcomes.
    /// </summary>
    CausalInference,

    /// <summary>
    /// Creates synthetic data that preserves statistical properties of real data.
    /// Example: generating realistic patient records for model training without privacy concerns.
    /// </summary>
    Synthesis,

    /// <summary>
    /// Extracts meaningful features or representations from raw data.
    /// Example: extracting visual features from images for downstream classification.
    /// </summary>
    FeatureExtraction,

    /// <summary>
    /// Repairs, enhances, or recovers degraded data.
    /// Example: restoring old or damaged photographs to their original quality.
    /// </summary>
    Restoration,

    /// <summary>
    /// Reduces the size or dimensionality of data while preserving essential information.
    /// Example: compressing images or pruning neural network weights.
    /// </summary>
    Compression,

    /// <summary>
    /// Generates intermediate frames between existing video frames.
    /// Example: converting 30fps video to smooth 60fps by generating in-between frames.
    /// </summary>
    FrameInterpolation,

    /// <summary>
    /// Improves the overall quality of audio, images, or other signals.
    /// Example: enhancing speech clarity by removing noise and improving frequency response.
    /// </summary>
    Enhancement,

    /// <summary>
    /// Processes, analyzes, or transforms raw signals (audio, radio, sensor data).
    /// Example: applying parametric EQ or spectral analysis to audio signals.
    /// </summary>
    SignalProcessing,

    /// <summary>
    /// Reduces the number of features or dimensions in data while preserving structure.
    /// Example: projecting high-dimensional data to 2D for visualization using PCA or t-SNE.
    /// </summary>
    DimensionalityReduction
}

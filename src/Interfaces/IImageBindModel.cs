using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Represents the different modality types supported by ImageBind.
/// </summary>
public enum ModalityType
{
    /// <summary>RGB image modality.</summary>
    Image,
    /// <summary>Text/language modality.</summary>
    Text,
    /// <summary>Audio waveform modality.</summary>
    Audio,
    /// <summary>Video (image sequence) modality.</summary>
    Video,
    /// <summary>Thermal/infrared image modality.</summary>
    Thermal,
    /// <summary>Depth map modality.</summary>
    Depth,
    /// <summary>IMU (Inertial Measurement Unit) sensor data.</summary>
    IMU
}

/// <summary>
/// Defines the contract for ImageBind models that bind multiple modalities (6+) into a shared embedding space.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// ImageBind learns a joint embedding space across six modalities: images, text, audio, depth,
/// thermal, and IMU data. It uses images as a binding modality - since web data contains
/// many (image, text) pairs, (image, audio) pairs from videos, etc., the model can learn
/// cross-modal relationships even without direct pairs between all modalities.
/// </para>
/// <para><b>For Beginners:</b> ImageBind connects ALL types of data together!
///
/// The breakthrough insight:
/// - Images are paired with many things: text captions, video audio, depth sensors, etc.
/// - By learning all these (image, X) pairs, images become a "bridge"
/// - Now audio and text can be compared, even without audio-text training data!
///
/// Six modalities in one model:
/// 1. Images: Regular RGB photos
/// 2. Text: Natural language descriptions
/// 3. Audio: Sound waveforms (speech, music, effects)
/// 4. Video: Moving images (sequences of frames)
/// 5. Thermal: Heat maps from infrared cameras
/// 6. Depth: 3D distance information
/// 7. IMU: Motion sensor data (accelerometer, gyroscope)
///
/// Why this matters:
/// - Search audio by describing sounds in text
/// - Find images that match a piece of music
/// - Match thermal images to regular photos
/// - Universal multimodal understanding!
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("ImageBindModel")]
public interface IImageBindModel<T>
{
    /// <summary>
    /// Gets the dimensionality of the shared embedding space.
    /// </summary>
    int EmbeddingDimension { get; }

    /// <summary>
    /// Gets the list of supported modalities.
    /// </summary>
    IReadOnlyList<ModalityType> SupportedModalities { get; }

    /// <summary>
    /// Converts an image into a shared embedding vector.
    /// </summary>
    /// <param name="image">Preprocessed image tensor [channels, height, width].</param>
    /// <returns>Normalized embedding vector.</returns>
    Vector<T> GetImageEmbedding(Tensor<T> image);

    /// <summary>
    /// Converts text into a shared embedding vector.
    /// </summary>
    /// <param name="text">Text string to embed.</param>
    /// <returns>Normalized embedding vector.</returns>
    Vector<T> GetTextEmbedding(string text);

    /// <summary>
    /// Converts audio into a shared embedding vector.
    /// </summary>
    /// <param name="audioWaveform">Audio waveform tensor [samples] or [channels, samples].</param>
    /// <param name="sampleRate">Audio sample rate in Hz.</param>
    /// <returns>Normalized embedding vector.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Convert sound into the same vector space as images and text!
    ///
    /// This allows:
    /// - Find images that match a sound (bird chirping → bird photos)
    /// - Search audio with text ("dog barking" → actual barking sounds)
    /// - Compare different sounds for similarity
    /// </para>
    /// </remarks>
    Vector<T> GetAudioEmbedding(Tensor<T> audioWaveform, int sampleRate = 16000);

    /// <summary>
    /// Converts video into a shared embedding vector.
    /// </summary>
    /// <param name="frames">Video frames as sequence of image tensors.</param>
    /// <returns>Normalized embedding vector.</returns>
    Vector<T> GetVideoEmbedding(IEnumerable<Tensor<T>> frames);

    /// <summary>
    /// Converts thermal image into a shared embedding vector.
    /// </summary>
    /// <param name="thermalImage">Thermal/infrared image tensor.</param>
    /// <returns>Normalized embedding vector.</returns>
    /// <remarks>
    /// <para>
    /// Thermal images capture heat signatures. ImageBind can match thermal images
    /// to their RGB counterparts or find related audio/text.
    /// </para>
    /// </remarks>
    Vector<T> GetThermalEmbedding(Tensor<T> thermalImage);

    /// <summary>
    /// Converts depth map into a shared embedding vector.
    /// </summary>
    /// <param name="depthMap">Depth map tensor [height, width] with distance values.</param>
    /// <returns>Normalized embedding vector.</returns>
    /// <remarks>
    /// <para>
    /// Depth maps represent 3D structure. ImageBind can find RGB images
    /// with similar spatial structure or match to text descriptions.
    /// </para>
    /// </remarks>
    Vector<T> GetDepthEmbedding(Tensor<T> depthMap);

    /// <summary>
    /// Converts IMU sensor data into a shared embedding vector.
    /// </summary>
    /// <param name="imuData">IMU readings [timesteps, 6] for accelerometer and gyroscope (x,y,z each).</param>
    /// <returns>Normalized embedding vector.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> IMU is the motion sensor in your phone!
    ///
    /// IMU captures movement patterns:
    /// - Walking, running, jumping
    /// - Phone gestures
    /// - Device orientation
    ///
    /// ImageBind can match these motions to videos or text descriptions!
    /// </para>
    /// </remarks>
    Vector<T> GetIMUEmbedding(Tensor<T> imuData);

    /// <summary>
    /// Gets embedding for any supported modality using a generic interface.
    /// </summary>
    /// <param name="modality">The type of modality.</param>
    /// <param name="data">The data to embed (type depends on modality).</param>
    /// <returns>Normalized embedding vector.</returns>
    Vector<T> GetEmbedding(ModalityType modality, object data);

    /// <summary>
    /// Computes similarity between embeddings from any two modalities.
    /// </summary>
    /// <param name="embedding1">First embedding vector.</param>
    /// <param name="embedding2">Second embedding vector.</param>
    /// <returns>Cosine similarity score in range [-1, 1].</returns>
    T ComputeCrossModalSimilarity(Vector<T> embedding1, Vector<T> embedding2);

    /// <summary>
    /// Performs cross-modal retrieval from one modality to another.
    /// </summary>
    /// <param name="queryEmbedding">Query embedding from source modality.</param>
    /// <param name="targetEmbeddings">Database of embeddings from target modality.</param>
    /// <param name="topK">Number of results to return.</param>
    /// <returns>Indices and scores of most similar items.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Search across different types of data!
    ///
    /// Examples:
    /// - Audio → Images: "Find images that match this sound"
    /// - Text → Audio: "Find sounds matching 'thunderstorm'"
    /// - Thermal → RGB: "Find color photos of this heat signature"
    /// - IMU → Video: "Find videos of people doing this motion"
    /// </para>
    /// </remarks>
    IEnumerable<(int Index, T Score)> CrossModalRetrieval(
        Vector<T> queryEmbedding,
        IEnumerable<Vector<T>> targetEmbeddings,
        int topK = 10);

    /// <summary>
    /// Performs zero-shot classification across modalities.
    /// </summary>
    /// <param name="modality">The modality of the input data.</param>
    /// <param name="data">The data to classify.</param>
    /// <param name="classLabels">Text labels for classification.</param>
    /// <returns>Dictionary mapping labels to probability scores.</returns>
    /// <remarks>
    /// <para>
    /// Works for any supported modality - classify audio by text labels,
    /// classify thermal images, classify motion patterns, etc.
    /// </para>
    /// </remarks>
    Dictionary<string, T> ZeroShotClassify(
        ModalityType modality,
        object data,
        IEnumerable<string> classLabels);

    /// <summary>
    /// Finds the best matching modality representation for a query.
    /// </summary>
    /// <param name="queryModality">Type of the query data.</param>
    /// <param name="queryData">The query data.</param>
    /// <param name="candidates">Dictionary of (modality, data) candidates.</param>
    /// <returns>Best matching candidate with its similarity score.</returns>
    (ModalityType Modality, object Data, T Score) FindBestMatch(
        ModalityType queryModality,
        object queryData,
        IEnumerable<(ModalityType Modality, object Data)> candidates);

    /// <summary>
    /// Computes emergent cross-modal relationships without explicit pairing.
    /// </summary>
    /// <param name="audio">Audio waveform.</param>
    /// <param name="text">Text description.</param>
    /// <returns>Similarity score between audio and text.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The magic of ImageBind!
    ///
    /// Even though ImageBind was never trained on (audio, text) pairs directly,
    /// it can still compare them through the shared embedding space!
    ///
    /// This works because:
    /// - Audio is aligned to images (from video)
    /// - Text is aligned to images (from captions)
    /// - Therefore, audio and text become implicitly aligned!
    ///
    /// "emergent" means this capability appeared without explicit training.
    /// </para>
    /// </remarks>
    T ComputeEmergentAudioTextSimilarity(Tensor<T> audio, string text);

    /// <summary>
    /// Generates text description for non-text modalities.
    /// </summary>
    /// <param name="modality">The input modality type.</param>
    /// <param name="data">The data to describe.</param>
    /// <param name="candidateDescriptions">Pool of possible descriptions.</param>
    /// <param name="topK">Number of best descriptions to return.</param>
    /// <returns>Best matching descriptions with scores.</returns>
    IEnumerable<(string Description, T Score)> GenerateDescriptions(
        ModalityType modality,
        object data,
        IEnumerable<string> candidateDescriptions,
        int topK = 5);

    /// <summary>
    /// Computes the alignment between two modalities given paired data.
    /// </summary>
    /// <param name="modality1">First modality type.</param>
    /// <param name="data1">Data from first modality.</param>
    /// <param name="modality2">Second modality type.</param>
    /// <param name="data2">Data from second modality.</param>
    /// <returns>Alignment score and optional alignment details.</returns>
    (T AlignmentScore, Dictionary<string, object> Details) ComputeAlignment(
        ModalityType modality1,
        object data1,
        ModalityType modality2,
        object data2);

    /// <summary>
    /// Performs multimodal fusion by combining embeddings from multiple modalities.
    /// </summary>
    /// <param name="modalityEmbeddings">Dictionary of (modality, embedding) pairs.</param>
    /// <param name="fusionMethod">Method for combining: "mean", "concat", "attention".</param>
    /// <returns>Fused embedding vector.</returns>
    Vector<T> FuseModalities(
        Dictionary<ModalityType, Vector<T>> modalityEmbeddings,
        string fusionMethod = "mean");
}

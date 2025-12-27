using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for VideoCLIP-style models that align video and text in a shared embedding space.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// VideoCLIP extends CLIP's contrastive learning paradigm to the video domain, enabling
/// text-to-video and video-to-text retrieval, action recognition, and temporal understanding.
/// </para>
/// <para><b>For Beginners:</b> VideoCLIP is like CLIP but for videos!
///
/// While CLIP matches images with text, VideoCLIP matches VIDEOS with text:
/// - Understands actions and events that unfold over time
/// - Can find videos matching text descriptions
/// - Can generate descriptions for video clips
///
/// Key capabilities:
/// - Temporal understanding: "A person picks up a ball then throws it"
/// - Action recognition: "Playing basketball", "Cooking", "Dancing"
/// - Video retrieval: Find videos matching any text query
/// - Video-text alignment: Match video segments to text descriptions
///
/// Architecture differences from CLIP:
/// - Processes multiple frames, not just one image
/// - Uses temporal attention/pooling across frames
/// - Learns motion and action patterns
/// </para>
/// </remarks>
public interface IVideoCLIPModel<T> : IMultimodalEmbedding<T>
{
    /// <summary>
    /// Gets the number of frames the model processes per video clip.
    /// </summary>
    int NumFrames { get; }

    /// <summary>
    /// Gets the frame rate (frames per second) for video sampling.
    /// </summary>
    double FrameRate { get; }

    /// <summary>
    /// Gets the temporal aggregation method used.
    /// </summary>
    /// <remarks>
    /// Common methods: "mean_pooling", "temporal_transformer", "late_fusion"
    /// </remarks>
    string TemporalAggregation { get; }

    /// <summary>
    /// Converts a video (sequence of frames) into an embedding vector.
    /// </summary>
    /// <param name="frames">Sequence of preprocessed frame tensors with shape [channels, height, width].</param>
    /// <returns>A normalized embedding vector representing the entire video.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This converts a video into a single vector!
    ///
    /// Process:
    /// 1. Each frame is encoded independently (like CLIP)
    /// 2. Frame features are aggregated over time
    /// 3. Result is a single vector capturing the video's content and actions
    ///
    /// Now you can compare videos to text or other videos!
    /// </para>
    /// </remarks>
    Vector<T> GetVideoEmbedding(IEnumerable<Tensor<T>> frames);

    /// <summary>
    /// Converts multiple videos into embedding vectors in a batch.
    /// </summary>
    /// <param name="videos">Collection of videos, each as a sequence of frames.</param>
    /// <returns>Collection of normalized embedding vectors.</returns>
    IEnumerable<Vector<T>> GetVideoEmbeddings(IEnumerable<IEnumerable<Tensor<T>>> videos);

    /// <summary>
    /// Computes similarity between a text description and a video.
    /// </summary>
    /// <param name="text">Text description of an action or event.</param>
    /// <param name="frames">Video frames to compare against.</param>
    /// <returns>Similarity score, typically in range [-1, 1].</returns>
    T ComputeVideoTextSimilarity(string text, IEnumerable<Tensor<T>> frames);

    /// <summary>
    /// Performs zero-shot action classification on a video.
    /// </summary>
    /// <param name="frames">Video frames to classify.</param>
    /// <param name="actionLabels">Candidate action labels.</param>
    /// <returns>Dictionary mapping actions to probability scores.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Recognize actions without training!
    ///
    /// Example:
    /// - Video: Someone shooting a basketball
    /// - Labels: ["playing basketball", "playing soccer", "swimming", "running"]
    /// - Result: {"playing basketball": 0.85, "running": 0.08, ...}
    ///
    /// Works with any action you can describe in text!
    /// </para>
    /// </remarks>
    Dictionary<string, T> ZeroShotActionRecognition(
        IEnumerable<Tensor<T>> frames,
        IEnumerable<string> actionLabels);

    /// <summary>
    /// Retrieves the most relevant videos for a text query.
    /// </summary>
    /// <param name="query">Text description of desired video content.</param>
    /// <param name="videoEmbeddings">Pre-computed embeddings of video database.</param>
    /// <param name="topK">Number of results to return.</param>
    /// <returns>Indices of top matching videos with their scores.</returns>
    IEnumerable<(int Index, T Score)> RetrieveVideos(
        string query,
        IEnumerable<Vector<T>> videoEmbeddings,
        int topK = 10);

    /// <summary>
    /// Retrieves the most relevant text descriptions for a video.
    /// </summary>
    /// <param name="frames">Video frames to find descriptions for.</param>
    /// <param name="candidateTexts">Pool of text descriptions to search.</param>
    /// <param name="topK">Number of results to return.</param>
    /// <returns>Indices of best matching texts with scores.</returns>
    IEnumerable<(int Index, T Score)> RetrieveTextsForVideo(
        IEnumerable<Tensor<T>> frames,
        IEnumerable<string> candidateTexts,
        int topK = 10);

    /// <summary>
    /// Localizes moments in a video that match a text description.
    /// </summary>
    /// <param name="frames">Full video as sequence of frames.</param>
    /// <param name="query">Text describing the moment to find.</param>
    /// <param name="windowSize">Number of frames per moment window.</param>
    /// <returns>List of (startFrame, endFrame, score) for matching moments.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Find specific moments in a video!
    ///
    /// Example:
    /// - Video: 5 minutes of a cooking show
    /// - Query: "chopping vegetables"
    /// - Result: [(300, 450, 0.92), (1200, 1350, 0.87)] - two segments where chopping happens
    /// </para>
    /// </remarks>
    IEnumerable<(int StartFrame, int EndFrame, T Score)> LocalizeMoments(
        IEnumerable<Tensor<T>> frames,
        string query,
        int windowSize = 16);

    /// <summary>
    /// Generates a caption describing the video content.
    /// </summary>
    /// <param name="frames">Video frames to caption.</param>
    /// <param name="maxLength">Maximum caption length.</param>
    /// <returns>Generated caption describing the video.</returns>
    string GenerateVideoCaption(IEnumerable<Tensor<T>> frames, int maxLength = 77);

    /// <summary>
    /// Answers a question about video content.
    /// </summary>
    /// <param name="frames">Video frames.</param>
    /// <param name="question">Question about the video.</param>
    /// <param name="maxLength">Maximum answer length.</param>
    /// <returns>Generated answer.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Ask questions about videos!
    ///
    /// Examples:
    /// - "What is the person doing?" → "Playing guitar"
    /// - "How many people are in the video?" → "Three"
    /// - "What happens at the end?" → "The dog catches the frisbee"
    /// </para>
    /// </remarks>
    string AnswerVideoQuestion(
        IEnumerable<Tensor<T>> frames,
        string question,
        int maxLength = 64);

    /// <summary>
    /// Extracts frame-level features before temporal aggregation.
    /// </summary>
    /// <param name="frames">Video frames.</param>
    /// <returns>Feature tensor with shape [numFrames, featureDim].</returns>
    Tensor<T> ExtractFrameFeatures(IEnumerable<Tensor<T>> frames);

    /// <summary>
    /// Computes temporal similarity matrix between video segments.
    /// </summary>
    /// <param name="video1Frames">First video frames.</param>
    /// <param name="video2Frames">Second video frames.</param>
    /// <returns>Similarity matrix with shape [numFrames1, numFrames2].</returns>
    /// <remarks>
    /// Useful for video alignment, finding corresponding moments, or detecting repetitions.
    /// </remarks>
    Tensor<T> ComputeTemporalSimilarityMatrix(
        IEnumerable<Tensor<T>> video1Frames,
        IEnumerable<Tensor<T>> video2Frames);

    /// <summary>
    /// Predicts the next action or event in a video.
    /// </summary>
    /// <param name="frames">Observed video frames.</param>
    /// <param name="possibleNextActions">Candidate actions that might happen next.</param>
    /// <returns>Probability distribution over possible next actions.</returns>
    Dictionary<string, T> PredictNextAction(
        IEnumerable<Tensor<T>> frames,
        IEnumerable<string> possibleNextActions);
}

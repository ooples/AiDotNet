using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

// Note: ModalityType enum is defined in IImageBindModel.cs

/// <summary>
/// Represents an input item for unified multimodal models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class MultimodalInput<T>
{
    /// <summary>The modality type of this input.</summary>
    public ModalityType Modality { get; set; }

    /// <summary>Processed tensor data for this input.</summary>
    internal Tensor<T>? InternalData { get; set; }

    /// <summary>Optional text content (for text modality).</summary>
    public string? TextContent { get; set; }

    /// <summary>Optional metadata about the input.</summary>
    public Dictionary<string, object>? Metadata { get; set; }

    /// <summary>Temporal ordering for sequential inputs.</summary>
    public int SequenceIndex { get; set; }

    /// <summary>
    /// Creates a text input for the multimodal model.
    /// </summary>
    /// <param name="text">The text content.</param>
    /// <param name="sequenceIndex">Optional sequence ordering.</param>
    public static MultimodalInput<T> FromText(string text, int sequenceIndex = 0)
    {
        return new MultimodalInput<T>
        {
            Modality = ModalityType.Text,
            TextContent = text,
            SequenceIndex = sequenceIndex
        };
    }

    /// <summary>
    /// Creates an image input from pixel data.
    /// </summary>
    /// <param name="pixels">Pixel values.</param>
    /// <param name="channels">Number of color channels.</param>
    /// <param name="height">Image height in pixels.</param>
    /// <param name="width">Image width in pixels.</param>
    /// <param name="sequenceIndex">Optional sequence ordering.</param>
    public static MultimodalInput<T> FromImage(Vector<T> pixels, int channels, int height, int width, int sequenceIndex = 0)
    {
        var input = new MultimodalInput<T>
        {
            Modality = ModalityType.Image,
            SequenceIndex = sequenceIndex,
            InternalData = new Tensor<T>([channels, height, width])
        };
        var srcData = pixels.ToArray();
        Array.Copy(srcData, input.InternalData.Data.ToArray(), Math.Min(srcData.Length, input.InternalData.Data.Length));
        return input;
    }

    /// <summary>
    /// Creates an audio input from waveform samples.
    /// </summary>
    /// <param name="samples">Audio samples.</param>
    /// <param name="sampleRate">Sample rate in Hz.</param>
    /// <param name="sequenceIndex">Optional sequence ordering.</param>
    public static MultimodalInput<T> FromAudio(Vector<T> samples, int sampleRate, int sequenceIndex = 0)
    {
        var input = new MultimodalInput<T>
        {
            Modality = ModalityType.Audio,
            SequenceIndex = sequenceIndex,
            Metadata = new Dictionary<string, object> { ["SampleRate"] = sampleRate },
            InternalData = new Tensor<T>([1, samples.Length])
        };
        var srcData = samples.ToArray();
        Array.Copy(srcData, input.InternalData.Data.ToArray(), srcData.Length);
        return input;
    }

    /// <summary>
    /// Creates a video input from frame data.
    /// </summary>
    /// <param name="frames">Frame pixel data.</param>
    /// <param name="numFrames">Number of frames.</param>
    /// <param name="channels">Number of color channels.</param>
    /// <param name="height">Frame height in pixels.</param>
    /// <param name="width">Frame width in pixels.</param>
    /// <param name="frameRate">Frame rate in fps.</param>
    /// <param name="sequenceIndex">Optional sequence ordering.</param>
    public static MultimodalInput<T> FromVideo(Vector<T> frames, int numFrames, int channels, int height, int width, double frameRate, int sequenceIndex = 0)
    {
        var input = new MultimodalInput<T>
        {
            Modality = ModalityType.Video,
            SequenceIndex = sequenceIndex,
            Metadata = new Dictionary<string, object> { ["FrameRate"] = frameRate },
            InternalData = new Tensor<T>([numFrames, channels, height, width])
        };
        var srcData = frames.ToArray();
        Array.Copy(srcData, input.InternalData.Data.ToArray(), Math.Min(srcData.Length, input.InternalData.Data.Length));
        return input;
    }
}

/// <summary>
/// Represents an output from unified multimodal models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class MultimodalOutput<T>
{
    /// <summary>The modality type of this output.</summary>
    public ModalityType Modality { get; set; }

    /// <summary>Processed tensor data for this output.</summary>
    internal Tensor<T>? InternalData { get; set; }

    /// <summary>Text content (for text outputs).</summary>
    public string? TextContent { get; set; }

    /// <summary>Confidence score for the output.</summary>
    public T? Confidence { get; set; }

    /// <summary>Optional metadata about the output.</summary>
    public Dictionary<string, object>? Metadata { get; set; }

    /// <summary>
    /// Gets image data as pixel values for image outputs.
    /// </summary>
    public Vector<T>? GetImagePixels()
    {
        if (Modality != ModalityType.Image || InternalData is null)
            return null;
        return new Vector<T>(InternalData.Data.ToArray());
    }

    /// <summary>
    /// Gets the dimensions of image output data.
    /// </summary>
    public (int Channels, int Height, int Width)? GetImageDimensions()
    {
        if (Modality != ModalityType.Image || InternalData is null || InternalData.Shape.Length != 3)
            return null;
        return (InternalData.Shape[0], InternalData.Shape[1], InternalData.Shape[2]);
    }

    /// <summary>
    /// Gets audio data as waveform samples for audio outputs.
    /// </summary>
    public Vector<T>? GetAudioSamples()
    {
        if (Modality != ModalityType.Audio || InternalData is null)
            return null;
        return new Vector<T>(InternalData.Data.ToArray());
    }

    /// <summary>
    /// Gets video frame data for video outputs.
    /// </summary>
    public Vector<T>? GetVideoFrames()
    {
        if (Modality != ModalityType.Video || InternalData is null)
            return null;
        return new Vector<T>(InternalData.Data.ToArray());
    }

    /// <summary>
    /// Gets the dimensions of video output data.
    /// </summary>
    public (int Frames, int Channels, int Height, int Width)? GetVideoDimensions()
    {
        if (Modality != ModalityType.Video || InternalData is null || InternalData.Shape.Length != 4)
            return null;
        return (InternalData.Shape[0], InternalData.Shape[1], InternalData.Shape[2], InternalData.Shape[3]);
    }
}

/// <summary>
/// Defines the contract for unified multimodal models that handle multiple modalities
/// in a single architecture, similar to GPT-4o, Gemini, or Meta's CM3Leon.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Unified multimodal models represent the next generation of AI systems that can
/// seamlessly process and generate content across multiple modalities (text, image,
/// audio, video) within a single unified architecture.
/// </para>
/// <para><b>For Beginners:</b> One model that can see, hear, read, and create!
///
/// Key capabilities:
/// - Any-to-any generation: Text → Image, Image → Text, Audio → Text, etc.
/// - Interleaved understanding: Process mixed sequences of text, images, audio
/// - Cross-modal reasoning: Answer questions using information from multiple sources
/// - Unified embeddings: All modalities share a common representation space
///
/// Architecture concepts:
/// 1. Modality Encoders: Specialized encoders for each input type
/// 2. Unified Transformer: Core model that processes all modalities
/// 3. Modality Decoders: Generate outputs in any modality
/// 4. Cross-Attention: Allow modalities to attend to each other
/// </para>
/// </remarks>
public interface IUnifiedMultimodalModel<T>
{
    /// <summary>
    /// Gets the supported input modalities.
    /// </summary>
    IReadOnlyList<ModalityType> SupportedInputModalities { get; }

    /// <summary>
    /// Gets the supported output modalities.
    /// </summary>
    IReadOnlyList<ModalityType> SupportedOutputModalities { get; }

    /// <summary>
    /// Gets the unified embedding dimension.
    /// </summary>
    int EmbeddingDimension { get; }

    /// <summary>
    /// Gets the maximum sequence length for interleaved inputs.
    /// </summary>
    int MaxSequenceLength { get; }

    /// <summary>
    /// Gets whether the model supports streaming generation.
    /// </summary>
    bool SupportsStreaming { get; }

    /// <summary>
    /// Encodes any supported modality into the unified embedding space.
    /// </summary>
    /// <param name="input">The multimodal input to encode.</param>
    /// <returns>Unified embedding vector.</returns>
    Vector<T> Encode(MultimodalInput<T> input);

    /// <summary>
    /// Encodes multiple interleaved inputs into a sequence of embeddings.
    /// </summary>
    /// <param name="inputs">Sequence of multimodal inputs in order.</param>
    /// <returns>Matrix of embeddings [numInputs, embeddingDim].</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Process a conversation with mixed content!
    ///
    /// Example input sequence:
    /// 1. Text: "Look at this image and describe what you see"
    /// 2. Image: [photo of a cat]
    /// 3. Text: "Now listen to this sound"
    /// 4. Audio: [meowing sound]
    /// 5. Text: "Are they related?"
    /// </para>
    /// </remarks>
    Matrix<T> EncodeSequence(IEnumerable<MultimodalInput<T>> inputs);

    /// <summary>
    /// Generates output in the specified modality given multimodal inputs.
    /// </summary>
    /// <param name="inputs">Input sequence (can be multiple modalities).</param>
    /// <param name="outputModality">Desired output modality.</param>
    /// <param name="maxLength">Maximum output length (tokens for text, frames for video, etc.).</param>
    /// <returns>Generated output in the specified modality.</returns>
    MultimodalOutput<T> Generate(
        IEnumerable<MultimodalInput<T>> inputs,
        ModalityType outputModality,
        int maxLength = 1024);

    /// <summary>
    /// Generates text response from multimodal inputs.
    /// </summary>
    /// <param name="inputs">Multimodal input sequence.</param>
    /// <param name="prompt">Text prompt/instruction.</param>
    /// <param name="maxTokens">Maximum tokens to generate.</param>
    /// <param name="temperature">Sampling temperature.</param>
    /// <returns>Generated text response.</returns>
    string GenerateText(
        IEnumerable<MultimodalInput<T>> inputs,
        string prompt,
        int maxTokens = 1024,
        double temperature = 0.7);

    /// <summary>
    /// Generates an image from multimodal inputs.
    /// </summary>
    /// <param name="inputs">Multimodal input sequence (text prompts, reference images, etc.).</param>
    /// <param name="width">Output image width.</param>
    /// <param name="height">Output image height.</param>
    /// <returns>Generated image tensor [channels, height, width].</returns>
    Tensor<T> GenerateImage(
        IEnumerable<MultimodalInput<T>> inputs,
        int width = 512,
        int height = 512);

    /// <summary>
    /// Generates audio from multimodal inputs.
    /// </summary>
    /// <param name="inputs">Multimodal input sequence.</param>
    /// <param name="durationSeconds">Target audio duration.</param>
    /// <param name="sampleRate">Output sample rate.</param>
    /// <returns>Generated audio waveform.</returns>
    Tensor<T> GenerateAudio(
        IEnumerable<MultimodalInput<T>> inputs,
        double durationSeconds = 5.0,
        int sampleRate = 44100);

    /// <summary>
    /// Conducts a multi-turn conversation with multimodal inputs.
    /// </summary>
    /// <param name="conversationHistory">Previous turns with multimodal content.</param>
    /// <param name="newInputs">New multimodal inputs for this turn.</param>
    /// <param name="maxTokens">Maximum tokens to generate.</param>
    /// <returns>Generated response.</returns>
    string Chat(
        IEnumerable<(string Role, IEnumerable<MultimodalInput<T>> Content)> conversationHistory,
        IEnumerable<MultimodalInput<T>> newInputs,
        int maxTokens = 1024);

    /// <summary>
    /// Computes cross-modal similarity between inputs.
    /// </summary>
    /// <param name="input1">First multimodal input.</param>
    /// <param name="input2">Second multimodal input.</param>
    /// <returns>Similarity score (0-1).</returns>
    T ComputeSimilarity(MultimodalInput<T> input1, MultimodalInput<T> input2);

    /// <summary>
    /// Retrieves the most similar items from a database given a query.
    /// </summary>
    /// <param name="query">Query input (any modality).</param>
    /// <param name="database">Database of items (any modalities).</param>
    /// <param name="topK">Number of results to return.</param>
    /// <returns>Indices, scores, and modalities of matching items.</returns>
    IEnumerable<(int Index, T Score, ModalityType Modality)> Retrieve(
        MultimodalInput<T> query,
        IEnumerable<MultimodalInput<T>> database,
        int topK = 10);

    /// <summary>
    /// Answers a question using multimodal context.
    /// </summary>
    /// <param name="context">Multimodal context (images, documents, audio, etc.).</param>
    /// <param name="question">The question to answer.</param>
    /// <returns>Answer and confidence score.</returns>
    (string Answer, T Confidence) AnswerQuestion(
        IEnumerable<MultimodalInput<T>> context,
        string question);

    /// <summary>
    /// Performs reasoning across multiple modalities.
    /// </summary>
    /// <param name="inputs">Multimodal inputs to reason about.</param>
    /// <param name="task">Reasoning task description.</param>
    /// <returns>Reasoning result with step-by-step explanation.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Multi-step thinking across different inputs!
    ///
    /// Example: Given an image of a recipe and audio of someone cooking,
    /// reason about whether they're following the recipe correctly.
    /// </para>
    /// </remarks>
    (string Result, IEnumerable<string> ReasoningSteps) Reason(
        IEnumerable<MultimodalInput<T>> inputs,
        string task);

    /// <summary>
    /// Translates content between modalities.
    /// </summary>
    /// <param name="input">Source input.</param>
    /// <param name="targetModality">Target modality.</param>
    /// <returns>Translated content.</returns>
    MultimodalOutput<T> Translate(
        MultimodalInput<T> input,
        ModalityType targetModality);

    /// <summary>
    /// Summarizes multimodal content.
    /// </summary>
    /// <param name="inputs">Multimodal content to summarize.</param>
    /// <param name="outputModality">Modality for the summary.</param>
    /// <param name="maxLength">Maximum summary length.</param>
    /// <returns>Summary in the specified modality.</returns>
    MultimodalOutput<T> Summarize(
        IEnumerable<MultimodalInput<T>> inputs,
        ModalityType outputModality = ModalityType.Text,
        int maxLength = 256);

    /// <summary>
    /// Detects and localizes objects/events across modalities.
    /// </summary>
    /// <param name="inputs">Multimodal inputs to analyze.</param>
    /// <param name="targetDescription">What to look for.</param>
    /// <returns>Detections with locations and modalities.</returns>
    IEnumerable<(string Label, T Confidence, ModalityType Modality, object Location)> Detect(
        IEnumerable<MultimodalInput<T>> inputs,
        string targetDescription);

    /// <summary>
    /// Generates an interleaved sequence of multiple modalities.
    /// </summary>
    /// <param name="inputs">Input sequence.</param>
    /// <param name="outputSpec">Specification of desired outputs (modality, length pairs).</param>
    /// <returns>Interleaved output sequence.</returns>
    /// <remarks>
    /// <para>
    /// This enables generation of content like illustrated stories,
    /// narrated videos, or multimedia presentations.
    /// </para>
    /// </remarks>
    IEnumerable<MultimodalOutput<T>> GenerateInterleaved(
        IEnumerable<MultimodalInput<T>> inputs,
        IEnumerable<(ModalityType Modality, int MaxLength)> outputSpec);

    /// <summary>
    /// Aligns content across modalities temporally.
    /// </summary>
    /// <param name="inputs">Multimodal inputs with temporal content.</param>
    /// <returns>Alignment matrix showing correspondences.</returns>
    Matrix<T> AlignTemporally(IEnumerable<MultimodalInput<T>> inputs);

    /// <summary>
    /// Fuses multiple modality inputs into a unified representation.
    /// </summary>
    /// <param name="inputs">Inputs to fuse.</param>
    /// <param name="fusionStrategy">Strategy: "early", "late", "attention", "hybrid".</param>
    /// <returns>Fused embedding.</returns>
    Vector<T> Fuse(
        IEnumerable<MultimodalInput<T>> inputs,
        string fusionStrategy = "attention");

    /// <summary>
    /// Checks content for safety across all modalities.
    /// </summary>
    /// <param name="inputs">Content to check.</param>
    /// <returns>Safety assessment per modality.</returns>
    Dictionary<ModalityType, (bool IsSafe, IEnumerable<string> Flags)> SafetyCheck(
        IEnumerable<MultimodalInput<T>> inputs);

    /// <summary>
    /// Gets attention weights showing cross-modal relationships.
    /// </summary>
    /// <param name="inputs">Multimodal inputs.</param>
    /// <returns>Attention weights between all input pairs.</returns>
    Tensor<T> GetCrossModalAttention(IEnumerable<MultimodalInput<T>> inputs);

    /// <summary>
    /// Performs in-context learning from multimodal examples.
    /// </summary>
    /// <param name="examples">Few-shot examples with inputs and outputs.</param>
    /// <param name="query">Query to process.</param>
    /// <returns>Predicted output based on examples.</returns>
    MultimodalOutput<T> FewShotLearn(
        IEnumerable<(IEnumerable<MultimodalInput<T>> Inputs, MultimodalOutput<T> Output)> examples,
        IEnumerable<MultimodalInput<T>> query);

    /// <summary>
    /// Edits multimodal content based on instructions.
    /// </summary>
    /// <param name="original">Original content.</param>
    /// <param name="editInstructions">Instructions for editing.</param>
    /// <returns>Edited content.</returns>
    MultimodalOutput<T> Edit(
        MultimodalInput<T> original,
        string editInstructions);

    /// <summary>
    /// Compares multiple multimodal inputs and provides analysis.
    /// </summary>
    /// <param name="inputs">Items to compare.</param>
    /// <param name="comparisonCriteria">What aspects to compare.</param>
    /// <returns>Comparison analysis.</returns>
    (string Analysis, Dictionary<string, IEnumerable<T>> Scores) Compare(
        IEnumerable<MultimodalInput<T>> inputs,
        IEnumerable<string> comparisonCriteria);
}

/// <summary>
/// Defines the contract for autoregressive multimodal generation models
/// that can generate tokens from any modality in an interleaved fashion.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This interface represents models like CM3Leon, Chameleon, or similar
/// that use a unified vocabulary across all modalities and generate
/// content token-by-token regardless of modality.
/// </para>
/// </remarks>
public interface IAutoregressiveMultimodalModel<T> : IUnifiedMultimodalModel<T>
{
    /// <summary>
    /// Gets the vocabulary size (includes all modality tokens).
    /// </summary>
    int VocabularySize { get; }

    /// <summary>
    /// Gets the number of tokens reserved for each modality.
    /// </summary>
    IReadOnlyDictionary<ModalityType, int> ModalityTokenCounts { get; }

    /// <summary>
    /// Generates the next token given the context.
    /// </summary>
    /// <param name="context">Previous tokens/inputs.</param>
    /// <param name="temperature">Sampling temperature.</param>
    /// <returns>Next token ID and its modality.</returns>
    (int TokenId, ModalityType Modality) GenerateNextToken(
        IEnumerable<MultimodalInput<T>> context,
        double temperature = 1.0);

    /// <summary>
    /// Gets token probabilities for next position.
    /// </summary>
    /// <param name="context">Context sequence.</param>
    /// <returns>Log probabilities for all tokens.</returns>
    Vector<T> GetNextTokenLogits(IEnumerable<MultimodalInput<T>> context);

    /// <summary>
    /// Tokenizes input into unified vocabulary tokens.
    /// </summary>
    /// <param name="input">Input to tokenize.</param>
    /// <returns>Token IDs in the unified vocabulary.</returns>
    IEnumerable<int> Tokenize(MultimodalInput<T> input);

    /// <summary>
    /// Detokenizes token IDs back to multimodal outputs.
    /// </summary>
    /// <param name="tokenIds">Token IDs to decode.</param>
    /// <returns>Decoded multimodal outputs.</returns>
    IEnumerable<MultimodalOutput<T>> Detokenize(IEnumerable<int> tokenIds);

    /// <summary>
    /// Computes the loss for next-token prediction.
    /// </summary>
    /// <param name="inputs">Input sequence.</param>
    /// <param name="targets">Target outputs.</param>
    /// <returns>Cross-entropy loss.</returns>
    T ComputeLoss(
        IEnumerable<MultimodalInput<T>> inputs,
        IEnumerable<MultimodalOutput<T>> targets);
}

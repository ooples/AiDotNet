namespace AiDotNet.NeuralNetworks;

/// <summary>
/// A neural network architecture for two-stream models (e.g., CLIP-family vision-language encoders)
/// that hosts a separate vision encoder and text encoder.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CLIP-style models (BASIC, DFNCLIP, EVACLIP, LiT, RegionCLIP, etc.) use two parallel encoders that
/// produce separate embeddings. The flat <see cref="NeuralNetworkArchitecture{T}.Layers"/> property
/// can only describe a single graph, so callers who want to provide custom layers for both streams
/// must use this type instead. <c>VisionLayers</c> populates the model's primary <c>Layers</c> list
/// (the image stream) and <c>TextLayers</c> populates the model's text-encoder list.
/// </para>
/// <para><b>For Beginners:</b> Some vision-language models have two separate "brains" — one that
/// looks at images and one that reads text — and they only meet at the very end when comparing
/// embeddings. If you want to swap in a custom layer stack for both halves, this architecture lets
/// you describe both halves at once, instead of cramming them into a single flat list (which silently
/// drops the text half).</para>
/// </remarks>
public class DualStreamArchitecture<T> : NeuralNetworkArchitecture<T>
{
    /// <summary>
    /// Gets the layer stack for the vision encoder. Populates the model's primary <c>Layers</c> list.
    /// </summary>
    public List<ILayer<T>> VisionLayers { get; } = new List<ILayer<T>>();

    /// <summary>
    /// Gets the layer stack for the text encoder. Populates the model's text-encoder list.
    /// </summary>
    public List<ILayer<T>> TextLayers { get; } = new List<ILayer<T>>();

    /// <summary>
    /// Initializes a new dual-stream architecture with explicit vision and text encoder layers.
    /// </summary>
    public DualStreamArchitecture(
        IEnumerable<ILayer<T>> visionLayers,
        IEnumerable<ILayer<T>> textLayers,
        InputType inputType = InputType.ThreeDimensional,
        NeuralNetworkTaskType taskType = NeuralNetworkTaskType.NaturalLanguageProcessing,
        NetworkComplexity complexity = NetworkComplexity.Medium,
        int inputSize = 0,
        int inputHeight = 0,
        int inputWidth = 0,
        int inputDepth = 1,
        int outputSize = 0,
        bool shouldReturnFullSequence = false,
        int imageEmbeddingDim = 0,
        int textEmbeddingDim = 0,
        int inputFrames = 0)
        : base(
            inputType: inputType,
            taskType: taskType,
            complexity: complexity,
            inputSize: inputSize,
            inputHeight: inputHeight,
            inputWidth: inputWidth,
            inputDepth: inputDepth,
            outputSize: outputSize,
            layers: null,
            shouldReturnFullSequence: shouldReturnFullSequence,
            imageEmbeddingDim: imageEmbeddingDim,
            textEmbeddingDim: textEmbeddingDim,
            inputFrames: inputFrames)
    {
        if (visionLayers is null) throw new ArgumentNullException(nameof(visionLayers));
        if (textLayers is null) throw new ArgumentNullException(nameof(textLayers));
        VisionLayers.AddRange(visionLayers);
        TextLayers.AddRange(textLayers);
    }
}

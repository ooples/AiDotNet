namespace AiDotNet.NeuralNetworks;

/// <summary>
/// A neural network architecture for vision + text two-stream models
/// (CLIP-family encoders: BASIC, DFNCLIP, EVACLIP, LiT, RegionCLIP, etc.) that
/// hosts a separate vision encoder and text encoder.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Concrete vision-language specialisation of <see cref="DualEncoderArchitecture{T}"/>:
/// the abstract base holds the two encoder stacks under modality-neutral names
/// (<c>EncoderALayers</c> / <c>EncoderBLayers</c>); this subclass exposes them
/// under their semantic vision-language aliases <c>VisionLayers</c> /
/// <c>TextLayers</c> so call sites read naturally.
/// </para>
/// <para><b>For Beginners:</b> Some vision-language models have two separate
/// "brains" — one that looks at images and one that reads text — and they only
/// meet at the very end when comparing embeddings. If you want to swap in a
/// custom layer stack for both halves, this architecture lets you describe
/// both halves at once, instead of cramming them into a single flat list
/// (which silently drops the text half).</para>
/// </remarks>
public class DualStreamArchitecture<T> : DualEncoderArchitecture<T>
{
    /// <summary>
    /// Gets the layer stack for the vision encoder. Alias for
    /// <see cref="DualEncoderArchitecture{T}.EncoderALayers"/>.
    /// </summary>
    public List<ILayer<T>> VisionLayers => EncoderALayers;

    /// <summary>
    /// Gets the layer stack for the text encoder. Alias for
    /// <see cref="DualEncoderArchitecture{T}.EncoderBLayers"/>.
    /// </summary>
    public List<ILayer<T>> TextLayers => EncoderBLayers;

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
            encoderALayers: visionLayers,
            encoderBLayers: textLayers,
            inputType: inputType,
            taskType: taskType,
            complexity: complexity,
            inputSize: inputSize,
            inputHeight: inputHeight,
            inputWidth: inputWidth,
            inputDepth: inputDepth,
            outputSize: outputSize,
            shouldReturnFullSequence: shouldReturnFullSequence,
            imageEmbeddingDim: imageEmbeddingDim,
            textEmbeddingDim: textEmbeddingDim,
            inputFrames: inputFrames)
    {
    }
}

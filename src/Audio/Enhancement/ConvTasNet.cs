using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Enhancement;

/// <summary>
/// Conv-TasNet: A fully-convolutional time-domain audio separation network.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Conv-TasNet (Convolutional Time-domain Audio Separation Network) is a pioneering
/// neural network architecture that operates directly in the time domain, avoiding
/// the phase reconstruction problems of frequency-domain methods.
/// </para>
/// <para>
/// The architecture consists of three main components:
/// <list type="bullet">
/// <item><description>Encoder: Converts waveform to a learned representation using 1D convolutions</description></item>
/// <item><description>Separator: Temporal Convolutional Network (TCN) that estimates source masks</description></item>
/// <item><description>Decoder: Reconstructs separated waveforms from masked representations</description></item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Conv-TasNet is like having multiple microphones that each focus
/// on one speaker in a noisy room. Give it a recording with multiple people talking,
/// and it separates them into individual clean tracks!
///
/// Traditional methods convert audio to frequency domain, process it, then convert back.
/// Conv-TasNet works directly on the waveform, which avoids problems with phase reconstruction
/// and often produces cleaner results.
///
/// Common use cases:
/// - Separating speakers in meeting recordings
/// - Isolating vocals from music
/// - Removing background noise
/// - Speech enhancement for hearing aids
/// - Denoising phone calls
/// </para>
/// <para>
/// Reference: Luo, Y., &amp; Mesgarani, N. (2019). Conv-TasNet: Surpassing Ideal Time-Frequency
/// Magnitude Masking for Speech Separation.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Use AiModelBuilder facade for audio source separation
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputSize: 16000,
///     outputSize: 16000);
///
/// var builder = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
///     .ConfigureModel(new ConvTasNet&lt;float&gt;(architecture, "conv_tasnet.onnx", 8000, 2));
///
/// // Build and use the model through the facade
/// var result = builder.Build(trainingData, trainingLabels);
/// var prediction = result.Predict(mixedAudioTensor);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.SourceSeparation)]
[ModelTask(ModelTask.Enhancement)]
[ModelTask(ModelTask.Denoising)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation", "https://arxiv.org/abs/1809.07454", Year = 2019, Authors = "Yi Luo, Nima Mesgarani")]
public partial class ConvTasNet<T> : AudioNeuralNetworkBase<T>, IAudioEnhancer<T>
{
    private readonly ConvTasNetOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private readonly INumericOperations<T> _numOps;

    // Encoder parameters
    private readonly int _encoderDim;
    private readonly int _kernelSize;
    private readonly int _stride;
    private Tensor<T> _encoderWeight;
    private Tensor<T> _encoderBias;

    // Separator (TCN) parameters
    private readonly int _numSources;
    private readonly int _bottleneckDim;
    private readonly int _hiddenDim;
    private readonly int _numBlocks;
    private readonly int _numRepeats;
    private readonly int _tcnKernelSize;

    // TCN layer weights (simplified representation)
    private readonly List<TcnBlock> _tcnBlocks;

    // Decoder parameters
    private Tensor<T> _decoderWeight;

    // Mask estimation
    private Tensor<T> _maskWeight;
    private Tensor<T> _maskBias;

    // Normalization layers
    private Tensor<T> _normGamma;
    private Tensor<T> _normBeta;

    // State for streaming
    private T[]? _encoderBuffer;
#pragma warning disable CS0414 // Reserved for future streaming implementation
    private T[][]? _tcnStates;
#pragma warning restore CS0414
    private int _bufferPosition;

    // Optimizer for training (used in native training mode)
    internal IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? Optimizer { get; set; }

    // IAudioEnhancer properties
    /// <inheritdoc/>
    public int NumChannels { get; } = 1;

    /// <inheritdoc/>
    public double EnhancementStrength { get; set; } = 1.0;

    /// <inheritdoc/>
    public int LatencySamples { get; private set; }

    /// <summary>
    /// Gets the number of sources the network separates.
    /// </summary>
    public int NumSources => _numSources;

    /// <summary>
    /// Gets the encoder dimension (number of basis functions).
    /// </summary>
    public int EncoderDimension => _encoderDim;

    /// <summary>
    /// Gets the encoder kernel size (window length in samples).
    /// </summary>
    public int EncoderKernelSize => _kernelSize;

    /// <summary>
    /// Initializes a new instance of the <see cref="ConvTasNet{T}"/> class for ONNX inference mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions.</param>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="sampleRate">Sample rate of input audio (default: 8000 Hz).</param>
    /// <param name="encoderDim">Encoder dimension (default: 512).</param>
    /// <param name="kernelSize">Encoder kernel size in samples (default: 16).</param>
    /// <param name="numSources">Number of sources to separate (default: 2).</param>
    /// <param name="onnxOptions">Optional ONNX model options.</param>
    /// <exception cref="FileNotFoundException">Thrown when the ONNX model file is not found.</exception>
    public ConvTasNet(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        int sampleRate = 8000,
        int encoderDim = 512,
        int kernelSize = 16,
        int numSources = 2,
        OnnxModelOptions? onnxOptions = null,
        ConvTasNetOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new ConvTasNetOptions();
        Options = _options;
        _numOps = MathHelper.GetNumericOperations<T>();

        if (string.IsNullOrWhiteSpace(modelPath))
        {
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        }

        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        }

        SampleRate = sampleRate;
        _encoderDim = encoderDim;
        _kernelSize = kernelSize;
        _stride = kernelSize / 2;
        _numSources = numSources;

        // Load ONNX model
        OnnxModel = new OnnxModel<T>(modelPath, onnxOptions);

        // Calculate latency (encoder kernel + some TCN lookahead)
        LatencySamples = kernelSize;

        // Initialize empty arrays (not used in ONNX mode)
        _encoderWeight = new Tensor<T>([0]);
        _encoderBias = new Tensor<T>([0]);
        _decoderWeight = new Tensor<T>([0]);
        _maskWeight = new Tensor<T>([0]);
        _maskBias = new Tensor<T>([0]);
        _normGamma = new Tensor<T>([0]);
        _normBeta = new Tensor<T>([0]);
        _tcnBlocks = new List<TcnBlock>();

        // These are set for consistency
        _bottleneckDim = 128;
        _hiddenDim = 512;
        _numBlocks = 8;
        _numRepeats = 3;
        _tcnKernelSize = 3;

    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ConvTasNet{T}"/> class for native training mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions.</param>
    /// <param name="sampleRate">Sample rate of input audio (default: 8000 Hz for speech).</param>
    /// <param name="encoderDim">Number of encoder basis functions (default: 512).</param>
    /// <param name="kernelSize">Encoder kernel size in samples (default: 16, about 2ms at 8kHz).</param>
    /// <param name="bottleneckDim">Bottleneck dimension in TCN (default: 128).</param>
    /// <param name="hiddenDim">Hidden dimension in TCN blocks (default: 512).</param>
    /// <param name="numBlocks">Number of TCN blocks per repeat (default: 8).</param>
    /// <param name="numRepeats">Number of TCN repeats (default: 3).</param>
    /// <param name="tcnKernelSize">Kernel size for TCN convolutions (default: 3).</param>
    /// <param name="numSources">Number of sources to separate (default: 2).</param>
    /// <param name="optimizer">Optimizer for training. If null, a default Adam optimizer is used.</param>
    /// <param name="lossFunction">Loss function. If null, SI-SNR loss is used.</param>
    public ConvTasNet(
        NeuralNetworkArchitecture<T> architecture,
        int sampleRate = 8000,
        int encoderDim = 512,
        int kernelSize = 16,
        int bottleneckDim = 128,
        int hiddenDim = 512,
        int numBlocks = 8,
        int numRepeats = 3,
        int tcnKernelSize = 3,
        int numSources = 2,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        ConvTasNetOptions? options = null)
        : base(architecture, lossFunction)
    {
        _options = options ?? new ConvTasNetOptions();
        Options = _options;
        _numOps = MathHelper.GetNumericOperations<T>();

        SampleRate = sampleRate;
        _encoderDim = encoderDim;
        _kernelSize = kernelSize;
        _stride = kernelSize / 2;
        _bottleneckDim = bottleneckDim;
        _hiddenDim = hiddenDim;
        _numBlocks = numBlocks;
        _numRepeats = numRepeats;
        _tcnKernelSize = tcnKernelSize;
        _numSources = numSources;

        // Calculate latency
        LatencySamples = kernelSize;

        // Initialize encoder weights
        _encoderWeight = InitializeWeights(_encoderDim * _kernelSize);
        _encoderBias = InitializeWeights(_encoderDim, 0.0);

        // Initialize normalization
        _normGamma = InitializeWeights(_encoderDim, 1.0);
        _normBeta = InitializeWeights(_encoderDim, 0.0);

        // Initialize TCN blocks
        _tcnBlocks = new List<TcnBlock>();
        for (int r = 0; r < numRepeats; r++)
        {
            for (int b = 0; b < numBlocks; b++)
            {
                int dilation = (int)Math.Pow(2, b);
                _tcnBlocks.Add(new TcnBlock(
                    _numOps,
                    bottleneckDim,
                    hiddenDim,
                    tcnKernelSize,
                    dilation));
            }
        }

        // Initialize mask estimation layer
        int maskInputDim = bottleneckDim;
        _maskWeight = InitializeWeights(numSources * encoderDim * maskInputDim);
        _maskBias = InitializeWeights(numSources * encoderDim, 0.0);

        // Initialize decoder weights (transposed convolution)
        _decoderWeight = InitializeWeights(_encoderDim * _kernelSize);

        // Initialize optimizer (Adam by default)
        Optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Initializes the neural network layers.
    /// </summary>
    protected override void InitializeLayers()
    {
        // Layers are handled manually for Conv-TasNet's specific architecture
        // The encoder, TCN, and decoder don't map directly to standard layer types
    }

    /// <summary>
    /// Preprocesses raw audio waveform for model input.
    /// </summary>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        // Conv-TasNet operates directly on waveform
        // Just ensure proper shape [batch, samples]
        if (rawAudio.Shape.Length == 1)
        {
            return rawAudio.Reshape(new[] { 1, rawAudio.Shape[0] });
        }
        return rawAudio;
    }

    /// <summary>
    /// Postprocesses model output.
    /// </summary>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        // Apply enhancement strength
        if (Math.Abs(EnhancementStrength - 1.0) > 1e-6)
        {
            var strengthT = _numOps.FromDouble(EnhancementStrength);
            var invStrength = _numOps.FromDouble(1.0 - EnhancementStrength);

            // Blend enhanced with original would require original signal
            // For now, just scale the output
            var result = new T[modelOutput.Length];
            for (int i = 0; i < modelOutput.Length; i++)
            {
                result[i] = _numOps.Multiply(modelOutput.Data.Span[i], strengthT);
            }
            return new Tensor<T>(result, modelOutput._shape);
        }
        return modelOutput;
    }

    /// <summary>
    /// Predicts separated sources from input audio.
    /// </summary>
    /// <param name="input">Input audio tensor [batch, samples] or [samples].</param>
    /// <returns>Separated sources tensor [batch, sources, samples] or [sources, samples].</returns>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        var preprocessed = PreprocessAudio(input);

        if (IsOnnxMode)
        {
            var output = RunOnnxInference(preprocessed);
            return PostprocessOutput(output);
        }

        return SeparateSources(preprocessed);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Conv-TasNet owns a manual encoder/TCN/mask/decoder graph rather than
    /// populating the base <c>Layers</c> collection, so the base inspection
    /// implementation has nothing to walk. Capture the actual semantic stages
    /// of the same graph used by <see cref="PredictCore"/> instead.
    /// </remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        SetTrainingMode(false);
        var activations = new Dictionary<string, Tensor<T>>();
        var preprocessed = PreprocessAudio(input);
        activations["PreprocessedWaveform"] = preprocessed.Clone();

        if (IsOnnxMode)
        {
            var output = PostprocessOutput(RunOnnxInference(preprocessed));
            activations["OnnxOutput"] = output.Clone();
            return activations;
        }

        int originalLength = preprocessed.Shape[1];
        var encoded = Encode(preprocessed);
        activations["Encoder"] = encoded.Clone();

        var normalized = LayerNorm(encoded);
        activations["EncoderNormalization"] = normalized.Clone();

        var bottleneck = BottleneckProject(normalized);
        activations["BottleneckProjection"] = bottleneck.Clone();

        var separatedFeatures = RunTcn(bottleneck);
        activations["TemporalConvolutionalSeparator"] = separatedFeatures.Clone();

        var masks = EstimateMasks(separatedFeatures);
        activations["SourceMasks"] = masks.Clone();

        var maskedSources = ApplyMasks(encoded, masks);
        activations["MaskedEncoderSources"] = maskedSources.Clone();

        var decoded = Decode(maskedSources, originalLength);
        activations["WaveformDecoder"] = decoded.Clone();
        activations["Output"] = PostprocessOutput(decoded).Clone();
        return activations;
    }

    /// <summary>
    /// Separates audio into individual source signals.
    /// </summary>
    /// <param name="mixture">Input mixture tensor [batch, samples].</param>
    /// <returns>Separated sources [batch, numSources, samples].</returns>
    private Tensor<T> SeparateSources(Tensor<T> mixture)
    {
        int batchSize = mixture.Shape[0];
        int numSamples = mixture.Shape[1];

        // Step 1: Encoder - convert waveform to latent representation
        var encoded = Encode(mixture);

        // Step 2: Layer normalization on encoder output
        var normalized = LayerNorm(encoded);

        // Step 3: Bottleneck projection
        var bottleneck = BottleneckProject(normalized);

        // Step 4: TCN separator
        var tcnOutput = RunTcn(bottleneck);

        // Step 5: Mask estimation
        var masks = EstimateMasks(tcnOutput);

        // Step 6: Apply masks to encoder output
        var maskedSources = ApplyMasks(encoded, masks);

        // Step 7: Decoder - convert back to waveform
        var separated = Decode(maskedSources, numSamples);

        return separated;
    }

    /// <summary>
    /// Encodes waveform using learned basis functions.
    /// </summary>
    private Tensor<T> Encode(Tensor<T> waveform)
    {
        int batchSize = waveform.Shape[0];
        int numSamples = waveform.Shape[1];
        int numFrames = (numSamples - _kernelSize) / _stride + 1;

        var encoded = new T[batchSize * numFrames * _encoderDim];

        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < numFrames; f++)
            {
                int sampleOffset = f * _stride;
                for (int d = 0; d < _encoderDim; d++)
                {
                    T sum = _encoderBias[d];
                    for (int k = 0; k < _kernelSize; k++)
                    {
                        int sampleIdx = sampleOffset + k;
                        if (sampleIdx < numSamples)
                        {
                            int waveIdx = b * numSamples + sampleIdx;
                            int weightIdx = d * _kernelSize + k;
                            sum = _numOps.Add(sum, _numOps.Multiply(waveform.Data.Span[waveIdx], _encoderWeight[weightIdx]));
                        }
                    }
                    // ReLU activation
                    int outIdx = b * numFrames * _encoderDim + f * _encoderDim + d;
                    encoded[outIdx] = _numOps.ToDouble(sum) > 0 ? sum : _numOps.Zero;
                }
            }
        }

        return new Tensor<T>(encoded, new[] { batchSize, numFrames, _encoderDim });
    }

    /// <summary>
    /// Applies layer normalization.
    /// </summary>
    private Tensor<T> LayerNorm(Tensor<T> input)
    {
        // _normGamma / _normBeta are already tensors -- no wrapping needed.
        var gammaTensor = _normGamma;
        var betaTensor = _normBeta;
        return Engine.LayerNorm(input, gammaTensor, betaTensor, 1e-5, out _, out _);
    }

    /// <summary>
    /// Projects to bottleneck dimension.
    /// </summary>
    private Tensor<T> BottleneckProject(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int numFrames = input.Shape[1];
        int inputDim = input.Shape[2];

        var projected = new T[batchSize * numFrames * _bottleneckDim];

        // Simple linear projection
        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < numFrames; f++)
            {
                for (int d = 0; d < _bottleneckDim; d++)
                {
                    T sum = _numOps.Zero;
                    for (int i = 0; i < inputDim && i < _bottleneckDim; i++)
                    {
                        int inIdx = b * numFrames * inputDim + f * inputDim + i;
                        if (i == d && inIdx < input.Length)
                        {
                            sum = input.Data.Span[inIdx]; // Identity-like projection for simplicity
                        }
                    }
                    int outIdx = b * numFrames * _bottleneckDim + f * _bottleneckDim + d;
                    projected[outIdx] = sum;
                }
            }
        }

        return new Tensor<T>(projected, new[] { batchSize, numFrames, _bottleneckDim });
    }

    /// <summary>
    /// Runs the Temporal Convolutional Network.
    /// </summary>
    private Tensor<T> RunTcn(Tensor<T> input)
    {
        var current = input;
        foreach (var block in _tcnBlocks)
        {
            current = block.Forward(current);
        }
        return current;
    }

    /// <summary>
    /// Estimates separation masks for each source.
    /// </summary>
    private Tensor<T> EstimateMasks(Tensor<T> tcnOutput)
    {
        int batchSize = tcnOutput.Shape[0];
        int numFrames = tcnOutput.Shape[1];
        int dim = tcnOutput.Shape[2];

        var masks = new T[batchSize * _numSources * numFrames * _encoderDim];

        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < _numSources; s++)
            {
                for (int f = 0; f < numFrames; f++)
                {
                    for (int d = 0; d < _encoderDim; d++)
                    {
                        // Linear projection followed by sigmoid
                        T sum = _maskBias[(s * _encoderDim + d) % _maskBias.Length];
                        for (int i = 0; i < dim; i++)
                        {
                            int inIdx = b * numFrames * dim + f * dim + i;
                            int wIdx = (s * _encoderDim * dim + d * dim + i) % _maskWeight.Length;
                            sum = _numOps.Add(sum, _numOps.Multiply(tcnOutput.Data.Span[inIdx], _maskWeight[wIdx]));
                        }

                        // Sigmoid activation for mask
                        double maskVal = 1.0 / (1.0 + Math.Exp(-_numOps.ToDouble(sum)));

                        int outIdx = b * _numSources * numFrames * _encoderDim +
                                     s * numFrames * _encoderDim +
                                     f * _encoderDim + d;
                        masks[outIdx] = _numOps.FromDouble(maskVal);
                    }
                }
            }
        }

        return new Tensor<T>(masks, new[] { batchSize, _numSources, numFrames, _encoderDim });
    }

    /// <summary>
    /// Applies masks to encoder output to separate sources.
    /// </summary>
    private Tensor<T> ApplyMasks(Tensor<T> encoded, Tensor<T> masks)
    {
        int batchSize = encoded.Shape[0];
        int numFrames = encoded.Shape[1];
        int encoderDim = encoded.Shape[2];

        var masked = new T[batchSize * _numSources * numFrames * encoderDim];

        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < _numSources; s++)
            {
                for (int f = 0; f < numFrames; f++)
                {
                    for (int d = 0; d < encoderDim; d++)
                    {
                        int encIdx = b * numFrames * encoderDim + f * encoderDim + d;
                        int maskIdx = b * _numSources * numFrames * encoderDim +
                                      s * numFrames * encoderDim +
                                      f * encoderDim + d;
                        int outIdx = maskIdx;

                        masked[outIdx] = _numOps.Multiply(encoded.Data.Span[encIdx], masks.Data.Span[maskIdx]);
                    }
                }
            }
        }

        return new Tensor<T>(masked, new[] { batchSize, _numSources, numFrames, encoderDim });
    }

    /// <summary>
    /// Decodes masked representations back to waveform.
    /// </summary>
    private Tensor<T> Decode(Tensor<T> maskedSources, int originalLength)
    {
        int batchSize = maskedSources.Shape[0];
        int numSources = maskedSources.Shape[1];
        int numFrames = maskedSources.Shape[2];
        int encoderDim = maskedSources.Shape[3];

        int outputLength = (numFrames - 1) * _stride + _kernelSize;
        if (outputLength > originalLength)
        {
            outputLength = originalLength;
        }

        var decoded = new T[batchSize * numSources * outputLength];

        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < numSources; s++)
            {
                // Transposed convolution (overlap-add)
                for (int f = 0; f < numFrames; f++)
                {
                    int sampleOffset = f * _stride;
                    for (int k = 0; k < _kernelSize && sampleOffset + k < outputLength; k++)
                    {
                        for (int d = 0; d < encoderDim; d++)
                        {
                            int inIdx = b * numSources * numFrames * encoderDim +
                                        s * numFrames * encoderDim +
                                        f * encoderDim + d;
                            int weightIdx = d * _kernelSize + k;
                            int outIdx = b * numSources * outputLength + s * outputLength + sampleOffset + k;

                            decoded[outIdx] = _numOps.Add(
                                decoded[outIdx],
                                _numOps.Multiply(maskedSources.Data.Span[inIdx], _decoderWeight[weightIdx % _decoderWeight.Length]));
                        }
                    }
                }
            }
        }

        return new Tensor<T>(decoded, new[] { batchSize, numSources, outputLength });
    }

    #region IAudioEnhancer Implementation

    /// <inheritdoc/>
    public Tensor<T> Enhance(Tensor<T> audio)
    {
        // For enhancement (denoising), use 2-source separation
        // Return the first source (assumed to be speech/target)
        var separated = Predict(audio);

        // Extract first source
        int batchDim = separated.Shape.Length > 2 ? separated.Shape[0] : 1;
        int numSamples = separated.Shape[^1];

        if (separated.Shape.Length == 2)
        {
            // [sources, samples] - take first source
            var enhanced = new T[numSamples];
            Array.Copy(separated.Data.ToArray(), 0, enhanced, 0, numSamples);
            return new Tensor<T>(enhanced, new[] { numSamples });
        }
        else
        {
            // [batch, sources, samples] - take first source for each batch
            var enhanced = new T[batchDim * numSamples];
            for (int b = 0; b < batchDim; b++)
            {
                int srcOffset = b * _numSources * numSamples;
                int dstOffset = b * numSamples;
                Array.Copy(separated.Data.ToArray(), srcOffset, enhanced, dstOffset, numSamples);
            }
            return new Tensor<T>(enhanced, new[] { batchDim, numSamples });
        }
    }

    /// <inheritdoc/>
    public Tensor<T> EnhanceWithReference(Tensor<T> audio, Tensor<T> reference)
    {
        // Conv-TasNet doesn't use reference signal
        // For echo cancellation, a different model would be more appropriate
        return Enhance(audio);
    }

    /// <inheritdoc/>
    public Tensor<T> ProcessChunk(Tensor<T> audioChunk)
    {
        // Initialize streaming buffer if needed
        if (_encoderBuffer is null)
        {
            _encoderBuffer = new T[_kernelSize];
            _bufferPosition = 0;
        }

        int chunkLen = audioChunk.Shape[^1];
        var outputChunks = new List<T[]>();

        for (int i = 0; i < chunkLen; i++)
        {
            // Add sample to buffer
            _encoderBuffer[_bufferPosition] = audioChunk.Data.Span[i];
            _bufferPosition++;

            // When buffer is full, process
            if (_bufferPosition >= _kernelSize)
            {
                var bufferTensor = new Tensor<T>(_encoderBuffer, new[] { 1, _kernelSize });
                var enhanced = Enhance(bufferTensor);
                outputChunks.Add(enhanced.Data.ToArray());

                // Shift buffer by stride
                Array.Copy(_encoderBuffer, _stride, _encoderBuffer, 0, _kernelSize - _stride);
                _bufferPosition = _kernelSize - _stride;
            }
        }

        // Concatenate output chunks
        int totalLen = outputChunks.Sum(c => c.Length);
        if (totalLen == 0)
        {
            return new Tensor<T>(new T[0], new[] { 0 });
        }

        var output = new T[totalLen];
        int offset = 0;
        foreach (var chunk in outputChunks)
        {
            Array.Copy(chunk, 0, output, offset, chunk.Length);
            offset += chunk.Length;
        }

        return new Tensor<T>(output, new[] { totalLen });
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        base.ResetState();
        _encoderBuffer = null;
        _tcnStates = null;
        _bufferPosition = 0;
    }

    /// <inheritdoc/>
    public void EstimateNoiseProfile(Tensor<T> noiseOnlyAudio)
    {
        // Conv-TasNet is trained end-to-end and doesn't use explicit noise profiles
        // This could be extended to adapt the model for specific noise types
    }

    #endregion

    #region Training

    /// <summary>
    /// Trains the model on a batch of mixture-source pairs.
    /// </summary>
    /// <param name="input">Mixture tensor [batch, samples].</param>
    /// <param name="expected">Target sources [batch, sources, samples].</param>
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
        {
            throw new InvalidOperationException("Cannot train in ONNX inference mode.");
        }

        SetTrainingMode(true);
        try
        {
            var predicted = PredictCore(input);
            _ = ComputeSiSnrLoss(predicted, expected);
            var gradients = ComputeGradients(predicted, expected);
            PublishComputedGradients(gradients);
            UpdateWeights(gradients);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <summary>
    /// Computes the SI-SNR loss for speech separation.
    /// </summary>
    private T ComputeSiSnrLoss(Tensor<T> predicted, Tensor<T> target)
    {
        // SI-SNR: Scale-Invariant Signal-to-Noise Ratio
        // Higher is better, so we negate for loss
        double totalLoss = 0;
        int batchSize = predicted.Shape[0];
        int numSources = predicted.Shape[1];
        int numSamples = predicted.Shape[2];
        double epsilon = 1e-8;

        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < numSources && s < target.Shape[1]; s++)
            {
                double dotProduct = 0;
                double targetNormSq = 0;

                for (int t = 0; t < numSamples && t < target.Shape[2]; t++)
                {
                    int predIdx = b * numSources * numSamples + s * numSamples + t;
                    int targIdx = b * target.Shape[1] * target.Shape[2] + s * target.Shape[2] + t;

                    if (predIdx < predicted.Length && targIdx < target.Length)
                    {
                        double pred = _numOps.ToDouble(predicted.Data.Span[predIdx]);
                        double targ = _numOps.ToDouble(target.Data.Span[targIdx]);
                        dotProduct += pred * targ;
                        targetNormSq += targ * targ;
                    }
                }

                // Scale factor
                double scale = dotProduct / (targetNormSq + epsilon);

                // Compute SI-SNR
                double signalPower = 0;
                double noisePower = 0;

                for (int t = 0; t < numSamples && t < target.Shape[2]; t++)
                {
                    int predIdx = b * numSources * numSamples + s * numSamples + t;
                    int targIdx = b * target.Shape[1] * target.Shape[2] + s * target.Shape[2] + t;

                    if (predIdx < predicted.Length && targIdx < target.Length)
                    {
                        double targ = _numOps.ToDouble(target.Data.Span[targIdx]);
                        double scaledTarget = scale * targ;
                        double pred = _numOps.ToDouble(predicted.Data.Span[predIdx]);
                        double noise = pred - scaledTarget;

                        signalPower += scaledTarget * scaledTarget;
                        noisePower += noise * noise;
                    }
                }

                double siSnr = 10 * Math.Log10((signalPower + epsilon) / (noisePower + epsilon));
                totalLoss -= siSnr; // Negate because higher SI-SNR is better
            }
        }

        return _numOps.FromDouble(totalLoss / (batchSize * numSources));
    }

    /// <summary>
    /// Computes gradients for backpropagation.
    /// </summary>
    private Dictionary<string, T[]> ComputeGradients(Tensor<T> predicted, Tensor<T> target)
    {
        // Simplified gradient computation
        var gradients = new Dictionary<string, T[]>
        {
            ["encoder"] = new T[_encoderWeight.Length],
            ["decoder"] = new T[_decoderWeight.Length],
            ["mask"] = new T[_maskWeight.Length]
        };

        // Compute output gradients
        int len = Math.Min(predicted.Length, target.Length);
        for (int i = 0; i < len; i++)
        {
            double pred = _numOps.ToDouble(predicted.Data.Span[i]);
            double targ = i < target.Length ? _numOps.ToDouble(target.Data.Span[i]) : 0;
            double grad = pred - targ;

            // Accumulate to decoder gradients
            int decoderIdx = i % _decoderWeight.Length;
            gradients["decoder"][decoderIdx] = _numOps.Add(
                gradients["decoder"][decoderIdx],
                _numOps.FromDouble(grad * 0.01));
        }

        return gradients;
    }

    /// <summary>Publishes the hand-derived Conv-TasNet gradients to the shared model surface.</summary>
    private void PublishComputedGradients(Dictionary<string, T[]> gradients)
    {
        var published = new Dictionary<Tensor<T>, Tensor<T>>(
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);

        void Add(string name, Tensor<T> parameter)
        {
            if (gradients.TryGetValue(name, out var values) && values.Length == parameter.Length)
                published[parameter] = new Tensor<T>(values, parameter._shape);
        }

        Add("encoder", _encoderWeight);
        Add("decoder", _decoderWeight);
        Add("mask", _maskWeight);
        PublishParameterGradients(published);
    }

    /// <summary>
    /// Updates model weights using computed gradients.
    /// </summary>
    private void UpdateWeights(Dictionary<string, T[]> gradients)
    {
        double learningRate = 1e-4;

        // Update encoder weights
        if (gradients.TryGetValue("encoder", out var encoderGrad))
        {
            for (int i = 0; i < _encoderWeight.Length; i++)
            {
                double grad = i < encoderGrad.Length ? _numOps.ToDouble(encoderGrad[i]) : 0;
                double weight = _numOps.ToDouble(_encoderWeight[i]);
                _encoderWeight[i] = _numOps.FromDouble(weight - learningRate * grad);
            }
        }

        // Update decoder weights
        if (gradients.TryGetValue("decoder", out var decoderGrad))
        {
            for (int i = 0; i < _decoderWeight.Length; i++)
            {
                double grad = i < decoderGrad.Length ? _numOps.ToDouble(decoderGrad[i]) : 0;
                double weight = _numOps.ToDouble(_decoderWeight[i]);
                _decoderWeight[i] = _numOps.FromDouble(weight - learningRate * grad);
            }
        }

        // Update mask weights
        if (gradients.TryGetValue("mask", out var maskGrad))
        {
            for (int i = 0; i < _maskWeight.Length; i++)
            {
                double grad = i < maskGrad.Length ? _numOps.ToDouble(maskGrad[i]) : 0;
                double weight = _numOps.ToDouble(_maskWeight[i]);
                _maskWeight[i] = _numOps.FromDouble(weight - learningRate * grad);
            }
        }

        // Update TCN blocks
        foreach (var block in _tcnBlocks)
        {
            block.UpdateWeights(learningRate);
        }
    }

    #endregion

    #region Serialization

    #endregion

    #region Helper Methods

    private Tensor<T> InitializeWeights(int size, double initValue = double.NaN)
    {
        var weights = new Tensor<T>([size]);
        if (double.IsNaN(initValue))
        {
            // Xavier/Glorot initialization
            double scale = Math.Sqrt(2.0 / size);
            var rand = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
            for (int i = 0; i < size; i++)
            {
                weights[i] = _numOps.FromDouble(rand.NextGaussian() * scale);
            }
        }
        else
        {
            for (int i = 0; i < size; i++)
            {
                weights[i] = _numOps.FromDouble(initValue);
            }
        }
        return weights;
    }

    #endregion

    #region Abstract Method Implementations

    /// <summary>
    /// Declares every weight Conv-TasNet owns: the encoder, the separation mask, the decoder, the
    /// layer-norm affine pair, and each temporal-convolution block's seven tensors.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Conv-TasNet implements its signal path with model-owned tensors rather than
    /// <see cref="NeuralNetworkBase{T}.Layers"/>, so the base walk finds nothing unless they are
    /// declared. Declared in the order the deleted GetParameters concatenated them -- encoder
    /// weight and bias, decoder weight, mask weight and bias, norm gamma and beta, then each TCN
    /// block -- so existing checkpoints still restore.
    /// </para>
    /// <para>
    /// This replaces ParameterCount, GetParameters, GetParameterChunks and SetParameters here, four
    /// more on TcnBlock, and the four Copy/Read helpers that moved values one element at a time.
    /// </para>
    /// <para>
    /// The weights are <c>Tensor&lt;T&gt;</c> now rather than raw <c>T[]</c>, which is what the rest
    /// of the library uses and what the trainable-parameter walk can see. A bare array cannot be
    /// declared: a <c>Vector&lt;T&gt;</c> built over one COPIES it, so a restore driven through such
    /// a view would have written into a temporary and been discarded.
    /// </para>
    /// </remarks>
    protected override IEnumerable<Tensor<T>> GetExtraTrainableTensors()
    {
        yield return _encoderWeight;
        yield return _encoderBias;
        yield return _decoderWeight;
        yield return _maskWeight;
        yield return _maskBias;
        yield return _normGamma;
        yield return _normBeta;

        foreach (var block in _tcnBlocks)
        {
            foreach (var tensor in block.EnumerateTensors())
            {
                yield return tensor;
            }
        }
    }

    // UpdateParameters is NOT overridden. It used to throw NotSupportedException; the base
    // implementation is virtual now and distributes a flat vector over the same enumeration
    // GetParameters folds, which this model already exposes correctly. The throw existed
    // because the member was ABSTRACT and demanded an answer -- 572 models answered it the
    // same way.
    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "Conv-TasNet",
            Description = $"Time-domain audio separation network ({_numSources} sources)",
            FeatureCount = SampleRate,
            Complexity = _numBlocks * _numRepeats
        };
        metadata.AdditionalInfo["EncoderDim"] = _encoderDim.ToString();
        metadata.AdditionalInfo["KernelSize"] = _kernelSize.ToString();
        metadata.AdditionalInfo["NumSources"] = _numSources.ToString();
        metadata.AdditionalInfo["Mode"] = IsOnnxMode ? "ONNX" : "Native";
        return metadata;
    }

    /// <inheritdoc/>


    /// <inheritdoc/>


    #endregion

    #region Nested Types

    /// <summary>
    /// A single block in the Temporal Convolutional Network.
    /// </summary>
    private class TcnBlock
    {
        private readonly INumericOperations<T> _ops;
        private readonly int _inputDim;
        private readonly int _hiddenDim;
        private readonly int _kernelSize;
        private readonly int _dilation;

        private Tensor<T> _conv1Weight;
        private Tensor<T> _conv1Bias;
        private Tensor<T> _conv2Weight;
        private Tensor<T> _conv2Bias;
        private Tensor<T> _depthwiseWeight;
        private Tensor<T> _normGamma;
        private Tensor<T> _normBeta;

        private T[] _gradConv1;
        private T[] _gradConv2;
        private T[] _gradDepthwise;

        public TcnBlock(INumericOperations<T> ops, int inputDim, int hiddenDim, int kernelSize, int dilation)
        {
            _ops = ops;
            _inputDim = inputDim;
            _hiddenDim = hiddenDim;
            _kernelSize = kernelSize;
            _dilation = dilation;

            // Initialize weights
            var rand = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
            double scale = Math.Sqrt(2.0 / inputDim);

            _conv1Weight = new Tensor<T>([hiddenDim * inputDim]);
            _conv1Bias = new Tensor<T>([hiddenDim]);
            _conv2Weight = new Tensor<T>([inputDim * hiddenDim]);
            _conv2Bias = new Tensor<T>([inputDim]);
            _depthwiseWeight = new Tensor<T>([hiddenDim * kernelSize]);
            _normGamma = new Tensor<T>([hiddenDim]);
            _normBeta = new Tensor<T>([hiddenDim]);

            for (int i = 0; i < _conv1Weight.Length; i++)
            {
                _conv1Weight[i] = _ops.FromDouble(rand.NextGaussian() * scale);
            }
            for (int i = 0; i < _conv2Weight.Length; i++)
            {
                _conv2Weight[i] = _ops.FromDouble(rand.NextGaussian() * scale);
            }
            for (int i = 0; i < _depthwiseWeight.Length; i++)
            {
                _depthwiseWeight[i] = _ops.FromDouble(rand.NextGaussian() * scale);
            }
            for (int i = 0; i < hiddenDim; i++)
            {
                _normGamma[i] = _ops.FromDouble(1.0);
                _normBeta[i] = _ops.Zero;
            }

            _gradConv1 = new T[_conv1Weight.Length];
            _gradConv2 = new T[_conv2Weight.Length];
            _gradDepthwise = new T[_depthwiseWeight.Length];
        }

        public Tensor<T> Forward(Tensor<T> input)
        {
            int batchSize = input.Shape[0];
            int numFrames = input.Shape[1];
            int inputDim = input.Shape[2];

            // 1x1 conv to hidden dim
            var hidden = new T[batchSize * numFrames * _hiddenDim];
            for (int b = 0; b < batchSize; b++)
            {
                for (int f = 0; f < numFrames; f++)
                {
                    for (int h = 0; h < _hiddenDim; h++)
                    {
                        T sum = _conv1Bias[h];
                        for (int i = 0; i < inputDim; i++)
                        {
                            int inIdx = b * numFrames * inputDim + f * inputDim + i;
                            int wIdx = h * inputDim + i;
                            if (inIdx < input.Length && wIdx < _conv1Weight.Length)
                            {
                                sum = _ops.Add(sum, _ops.Multiply(input.Data.Span[inIdx], _conv1Weight[wIdx]));
                            }
                        }
                        // PReLU activation
                        int outIdx = b * numFrames * _hiddenDim + f * _hiddenDim + h;
                        double val = _ops.ToDouble(sum);
                        hidden[outIdx] = val > 0 ? sum : _ops.FromDouble(val * 0.25);
                    }
                }
            }

            // Depthwise convolution with dilation
            var depthOut = new T[batchSize * numFrames * _hiddenDim];
            for (int b = 0; b < batchSize; b++)
            {
                for (int f = 0; f < numFrames; f++)
                {
                    for (int h = 0; h < _hiddenDim; h++)
                    {
                        T sum = _ops.Zero;
                        for (int k = 0; k < _kernelSize; k++)
                        {
                            int inputFrame = f - ((_kernelSize - 1) / 2 - k) * _dilation;
                            if (inputFrame >= 0 && inputFrame < numFrames)
                            {
                                int inIdx = b * numFrames * _hiddenDim + inputFrame * _hiddenDim + h;
                                int wIdx = h * _kernelSize + k;
                                if (wIdx < _depthwiseWeight.Length)
                                {
                                    sum = _ops.Add(sum, _ops.Multiply(hidden[inIdx], _depthwiseWeight[wIdx]));
                                }
                            }
                        }
                        int outIdx = b * numFrames * _hiddenDim + f * _hiddenDim + h;
                        double val = _ops.ToDouble(sum);
                        depthOut[outIdx] = val > 0 ? sum : _ops.FromDouble(val * 0.25);
                    }
                }
            }

            // 1x1 conv back to input dim
            var output = new T[batchSize * numFrames * _inputDim];
            for (int b = 0; b < batchSize; b++)
            {
                for (int f = 0; f < numFrames; f++)
                {
                    for (int i = 0; i < _inputDim; i++)
                    {
                        T sum = _conv2Bias[i];
                        for (int h = 0; h < _hiddenDim; h++)
                        {
                            int inIdx = b * numFrames * _hiddenDim + f * _hiddenDim + h;
                            int wIdx = i * _hiddenDim + h;
                            if (wIdx < _conv2Weight.Length)
                            {
                                sum = _ops.Add(sum, _ops.Multiply(depthOut[inIdx], _conv2Weight[wIdx]));
                            }
                        }
                        int outIdx = b * numFrames * _inputDim + f * _inputDim + i;
                        int inOrigIdx = b * numFrames * inputDim + f * inputDim + i;

                        // Residual connection
                        if (inOrigIdx < input.Length)
                        {
                            output[outIdx] = _ops.Add(sum, input.Data.Span[inOrigIdx]);
                        }
                        else
                        {
                            output[outIdx] = sum;
                        }
                    }
                }
            }

            return new Tensor<T>(output, new[] { batchSize, numFrames, _inputDim });
        }

        public void UpdateWeights(double learningRate)
        {
            // Apply gradients to weights
            for (int i = 0; i < _conv1Weight.Length; i++)
            {
                double grad = _ops.ToDouble(_gradConv1[i]);
                double weight = _ops.ToDouble(_conv1Weight[i]);
                _conv1Weight[i] = _ops.FromDouble(weight - learningRate * grad);
                _gradConv1[i] = _ops.Zero;
            }

            for (int i = 0; i < _conv2Weight.Length; i++)
            {
                double grad = _ops.ToDouble(_gradConv2[i]);
                double weight = _ops.ToDouble(_conv2Weight[i]);
                _conv2Weight[i] = _ops.FromDouble(weight - learningRate * grad);
                _gradConv2[i] = _ops.Zero;
            }

            for (int i = 0; i < _depthwiseWeight.Length; i++)
            {
                double grad = _ops.ToDouble(_gradDepthwise[i]);
                double weight = _ops.ToDouble(_depthwiseWeight[i]);
                _depthwiseWeight[i] = _ops.FromDouble(weight - learningRate * grad);
                _gradDepthwise[i] = _ops.Zero;
            }
        }

        /// <summary>The trainable tensors this block owns, in forward order.</summary>
        /// <remarks>
        /// Replaces this block's ParameterCount, GetParameterChunks, CopyParametersTo and
        /// ReadParametersFrom -- four members that each listed the same seven weights in the same
        /// order, plus a Copy/Read pair to move them element by element. ConvTasNet folds this one
        /// enumeration for all four purposes.
        /// </remarks>
        internal IEnumerable<Tensor<T>> EnumerateTensors()
        {
            yield return _conv1Weight;
            yield return _conv1Bias;
            yield return _conv2Weight;
            yield return _conv2Bias;
            yield return _depthwiseWeight;
            yield return _normGamma;
            yield return _normBeta;
        }
    }

    #endregion
}

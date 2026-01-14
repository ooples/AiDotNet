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
public class ConvTasNet<T> : AudioNeuralNetworkBase<T>, IAudioEnhancer<T>
{
    private readonly INumericOperations<T> _numOps;

    // Encoder parameters
    private readonly int _encoderDim;
    private readonly int _kernelSize;
    private readonly int _stride;
    private T[] _encoderWeight;
    private T[] _encoderBias;

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
    private T[] _decoderWeight;

    // Mask estimation
    private T[] _maskWeight;
    private T[] _maskBias;

    // Normalization layers
    private T[] _normGamma;
    private T[] _normBeta;

    // State for streaming
    private T[]? _encoderBuffer;
#pragma warning disable CS0414 // Reserved for future streaming implementation
    private T[][]? _tcnStates;
#pragma warning restore CS0414
    private int _bufferPosition;

    // Optimizer for training
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

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
        OnnxModelOptions? onnxOptions = null)
        : base(architecture)
    {
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
        _encoderWeight = Array.Empty<T>();
        _encoderBias = Array.Empty<T>();
        _decoderWeight = Array.Empty<T>();
        _maskWeight = Array.Empty<T>();
        _maskBias = Array.Empty<T>();
        _normGamma = Array.Empty<T>();
        _normBeta = Array.Empty<T>();
        _tcnBlocks = new List<TcnBlock>();

        // These are set for consistency
        _bottleneckDim = 128;
        _hiddenDim = 512;
        _numBlocks = 8;
        _numRepeats = 3;
        _tcnKernelSize = 3;

        // Initialize optimizer (not used in ONNX mode but required for readonly field)
        _optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
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
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction)
    {
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
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

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
            return new Tensor<T>(result, modelOutput.Shape);
        }
        return modelOutput;
    }

    /// <summary>
    /// Predicts separated sources from input audio.
    /// </summary>
    /// <param name="input">Input audio tensor [batch, samples] or [samples].</param>
    /// <returns>Separated sources tensor [batch, sources, samples] or [sources, samples].</returns>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        var preprocessed = PreprocessAudio(input);

        if (IsOnnxMode)
        {
            var output = RunOnnxInference(preprocessed);
            return PostprocessOutput(output);
        }

        return SeparateSources(preprocessed);
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
        int batchSize = input.Shape[0];
        int numFrames = input.Shape[1];
        int dim = input.Shape[2];
        double epsilon = 1e-5;

        var normalized = new T[input.Length];

        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < numFrames; f++)
            {
                // Compute mean
                double sum = 0;
                for (int d = 0; d < dim; d++)
                {
                    int idx = b * numFrames * dim + f * dim + d;
                    sum += _numOps.ToDouble(input.Data.Span[idx]);
                }
                double mean = sum / dim;

                // Compute variance
                double varSum = 0;
                for (int d = 0; d < dim; d++)
                {
                    int idx = b * numFrames * dim + f * dim + d;
                    double diff = _numOps.ToDouble(input.Data.Span[idx]) - mean;
                    varSum += diff * diff;
                }
                double variance = varSum / dim;
                double std = Math.Sqrt(variance + epsilon);

                // Normalize and apply gamma/beta
                for (int d = 0; d < dim; d++)
                {
                    int idx = b * numFrames * dim + f * dim + d;
                    double x = _numOps.ToDouble(input.Data.Span[idx]);
                    double normed = (x - mean) / std;
                    double gamma = _numOps.ToDouble(_normGamma[d % _normGamma.Length]);
                    double beta = _numOps.ToDouble(_normBeta[d % _normBeta.Length]);
                    normalized[idx] = _numOps.FromDouble(gamma * normed + beta);
                }
            }
        }

        return new Tensor<T>(normalized, input.Shape);
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

        // Forward pass
        var predicted = Predict(input);

        // Compute SI-SNR loss (Scale-Invariant Signal-to-Noise Ratio)
        var loss = ComputeSiSnrLoss(predicted, expected);

        // Backward pass (simplified gradient computation)
        var gradients = ComputeGradients(predicted, expected);

        // Update weights
        UpdateWeights(gradients);

        SetTrainingMode(false);
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

    /// <summary>
    /// Serializes the model state to a byte array.
    /// </summary>
    public override byte[] Serialize()
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);

        // Write model configuration
        writer.Write(SampleRate);
        writer.Write(_encoderDim);
        writer.Write(_kernelSize);
        writer.Write(_bottleneckDim);
        writer.Write(_hiddenDim);
        writer.Write(_numBlocks);
        writer.Write(_numRepeats);
        writer.Write(_tcnKernelSize);
        writer.Write(_numSources);

        // Write encoder weights
        writer.Write(_encoderWeight.Length);
        foreach (var w in _encoderWeight)
        {
            writer.Write(_numOps.ToDouble(w));
        }

        // Write decoder weights
        writer.Write(_decoderWeight.Length);
        foreach (var w in _decoderWeight)
        {
            writer.Write(_numOps.ToDouble(w));
        }

        // Write mask weights
        writer.Write(_maskWeight.Length);
        foreach (var w in _maskWeight)
        {
            writer.Write(_numOps.ToDouble(w));
        }

        // Write normalization parameters
        writer.Write(_normGamma.Length);
        foreach (var g in _normGamma)
        {
            writer.Write(_numOps.ToDouble(g));
        }
        foreach (var b in _normBeta)
        {
            writer.Write(_numOps.ToDouble(b));
        }

        return stream.ToArray();
    }

    /// <summary>
    /// Deserializes the model state from a byte array.
    /// </summary>
    public override void Deserialize(byte[] data)
    {
        using var stream = new MemoryStream(data);
        using var reader = new BinaryReader(stream);

        // Read and verify configuration
        int sampleRate = reader.ReadInt32();
        int encoderDim = reader.ReadInt32();
        int kernelSize = reader.ReadInt32();
        int bottleneckDim = reader.ReadInt32();
        int hiddenDim = reader.ReadInt32();
        int numBlocks = reader.ReadInt32();
        int numRepeats = reader.ReadInt32();
        int tcnKernelSize = reader.ReadInt32();
        int numSources = reader.ReadInt32();

        // Validate configuration matches
        if (encoderDim != _encoderDim || kernelSize != _kernelSize || numSources != _numSources)
        {
            throw new InvalidOperationException("Serialized model configuration does not match current model.");
        }

        // Read encoder weights
        int encoderLen = reader.ReadInt32();
        for (int i = 0; i < encoderLen && i < _encoderWeight.Length; i++)
        {
            _encoderWeight[i] = _numOps.FromDouble(reader.ReadDouble());
        }

        // Read decoder weights
        int decoderLen = reader.ReadInt32();
        for (int i = 0; i < decoderLen && i < _decoderWeight.Length; i++)
        {
            _decoderWeight[i] = _numOps.FromDouble(reader.ReadDouble());
        }

        // Read mask weights
        int maskLen = reader.ReadInt32();
        for (int i = 0; i < maskLen && i < _maskWeight.Length; i++)
        {
            _maskWeight[i] = _numOps.FromDouble(reader.ReadDouble());
        }

        // Read normalization parameters
        int normLen = reader.ReadInt32();
        for (int i = 0; i < normLen && i < _normGamma.Length; i++)
        {
            _normGamma[i] = _numOps.FromDouble(reader.ReadDouble());
        }
        for (int i = 0; i < normLen && i < _normBeta.Length; i++)
        {
            _normBeta[i] = _numOps.FromDouble(reader.ReadDouble());
        }
    }

    #endregion

    #region Helper Methods

    private T[] InitializeWeights(int size, double initValue = double.NaN)
    {
        var weights = new T[size];
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

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> gradients)
    {
        if (IsOnnxMode)
        {
            throw new NotSupportedException("Cannot update parameters in ONNX inference mode.");
        }

        // Get current parameters and apply gradient descent
        var currentParams = GetParameters();
        T learningRate = _numOps.FromDouble(0.001);
        for (int i = 0; i < Math.Min(currentParams.Length, gradients.Length); i++)
        {
            currentParams[i] = _numOps.Subtract(currentParams[i], _numOps.Multiply(learningRate, gradients[i]));
        }
        SetParameters(currentParams);
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "Conv-TasNet",
            Description = $"Time-domain audio separation network ({_numSources} sources)",
            ModelType = ModelType.NeuralNetwork,
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
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(IsOnnxMode);
        writer.Write(SampleRate);
        writer.Write(_encoderDim);
        writer.Write(_kernelSize);
        writer.Write(_stride);
        writer.Write(_numSources);
        writer.Write(_bottleneckDim);
        writer.Write(_hiddenDim);
        writer.Write(_numBlocks);
        writer.Write(_numRepeats);
        writer.Write(_tcnKernelSize);
        writer.Write(EnhancementStrength);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read configuration values for validation
        _ = reader.ReadBoolean(); // IsOnnxMode
        _ = reader.ReadInt32();   // SampleRate
        _ = reader.ReadInt32();   // _encoderDim
        _ = reader.ReadInt32();   // _kernelSize
        _ = reader.ReadInt32();   // _stride
        _ = reader.ReadInt32();   // _numSources
        _ = reader.ReadInt32();   // _bottleneckDim
        _ = reader.ReadInt32();   // _hiddenDim
        _ = reader.ReadInt32();   // _numBlocks
        _ = reader.ReadInt32();   // _numRepeats
        _ = reader.ReadInt32();   // _tcnKernelSize
        EnhancementStrength = reader.ReadDouble();
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new ConvTasNet<T>(
            Architecture,
            sampleRate: SampleRate,
            encoderDim: _encoderDim,
            kernelSize: _kernelSize,
            bottleneckDim: _bottleneckDim,
            hiddenDim: _hiddenDim,
            numBlocks: _numBlocks,
            numRepeats: _numRepeats,
            tcnKernelSize: _tcnKernelSize,
            numSources: _numSources);
    }

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

        private T[] _conv1Weight;
        private T[] _conv1Bias;
        private T[] _conv2Weight;
        private T[] _conv2Bias;
        private T[] _depthwiseWeight;
        private T[] _normGamma;
        private T[] _normBeta;

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

            _conv1Weight = new T[hiddenDim * inputDim];
            _conv1Bias = new T[hiddenDim];
            _conv2Weight = new T[inputDim * hiddenDim];
            _conv2Bias = new T[inputDim];
            _depthwiseWeight = new T[hiddenDim * kernelSize];
            _normGamma = new T[hiddenDim];
            _normBeta = new T[hiddenDim];

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
    }

    #endregion
}

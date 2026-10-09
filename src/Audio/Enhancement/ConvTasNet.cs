using AiDotNet.LearningRateSchedulers;
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
///     .ConfigureModel(new ConvTasNet&lt;float&gt;(architecture, "conv_tasnet.onnx",
///         new ConvTasNetOptions { SampleRate = 8000, NumSources = 2 }));
///
/// var trainingData = Tensor&lt;float&gt;.CreateRandom(4, 16000);
/// var trainingLabels = Tensor&lt;float&gt;.CreateRandom(4, 16000);
/// var mixedAudioTensor = Tensor&lt;float&gt;.CreateRandom(1, 16000);
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
[PreprocessesInput("ConvTasNetNetwork.Forward reshapes the caller's [batch, samples] mixture to a single-channel [batch, 1, samples] waveform before the encoder's 1-D convolution.")]
[StackInputLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Time)]
[ResearchPaper("Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation", "https://arxiv.org/abs/1809.07454", Year = 2019, Authors = "Yi Luo, Nima Mesgarani")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-3, MaxGradientNorm = 5.0,
                Schedule = LearningRateSchedulerType.ReduceOnPlateau, DecayRate = 0.5,
                ScheduleStepMode = SchedulerStepMode.StepPerEpoch,
                StepSize = 3,
                Source = "Luo and Mesgarani 2019, Sec. IV: Adam with an initial learning rate of 1e-3, halved if validation accuracy does not improve for 3 consecutive epochs, and gradient clipping at maximum L2-norm 5, over 100 epochs.")]
public partial class ConvTasNet<T> : AudioNeuralNetworkBase<T>, IAudioEnhancer<T>
{
    private readonly ConvTasNetOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private readonly INumericOperations<T> _numOps;

    // Architecture sizes, kept for the public surface, streaming and metadata.
    private readonly int _encoderDim;
    private readonly int _kernelSize;
    private readonly int _stride;
    private readonly int _numSources;
    private readonly int _numBlocks;
    private readonly int _numRepeats;

    // The paper network (encoder, TCN separator, mask head, decoder), built from library layers and published
    // through Layers so the base tape trains every weight; null in ONNX mode or for a custom layer stack.
    private ConvTasNetNetwork<T>? _network;

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
        ConvTasNetOptions? options = null,
        OnnxModelOptions? onnxOptions = null)
        : base(architecture: architecture)
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

        // Validated after the path checks so a missing model file reports itself
        // as FileNotFoundException rather than being pre-empted by the options.
        _options.Validate();

        SampleRate = _options.SampleRate;
        _encoderDim = _options.EncoderDim;
        _kernelSize = _options.KernelSize;
        _stride = _options.KernelSize / 2;
        _numSources = _options.NumSources;

        // Load ONNX model
        OnnxModel = new OnnxModel<T>(modelPath, onnxOptions);

        // Calculate latency (encoder kernel + some TCN lookahead)
        LatencySamples = _options.KernelSize;

        _numBlocks = _options.NumBlocks;
        _numRepeats = _options.NumRepeats;
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
    /// <param name="lossFunction">Loss function. If null, negative SI-SNR with permutation-invariant training (the paper's objective) is used.</param>
    public ConvTasNet(
        NeuralNetworkArchitecture<T> architecture,
        ConvTasNetOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture: architecture, lossFunction: lossFunction ?? new NegativeSiSnrLoss<T>())
    {
        _options = options ?? new ConvTasNetOptions();
        _options.Validate();
        Options = _options;
        _numOps = MathHelper.GetNumericOperations<T>();

        SampleRate = _options.SampleRate;
        _encoderDim = _options.EncoderDim;
        _kernelSize = _options.KernelSize;
        _stride = _options.KernelSize / 2;
        _numSources = _options.NumSources;
        _numBlocks = _options.NumBlocks;
        _numRepeats = _options.NumRepeats;

        // Calculate latency
        LatencySamples = _options.KernelSize;

        // The paper's Adam (lr 1e-3, clipping at L2 norm 5), from the [PaperOptimizer] recipe.
        Optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Builds the paper network and publishes its layers; a caller-supplied stack in the same layout is
    /// bound to the network, and any other stack runs as a plain sequential chain.
    /// </summary>
    protected override void InitializeLayers()
    {
        if (IsOnnxMode) return;
        var network = new ConvTasNetNetwork<T>(_options);
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            if (network.Layers.Count == Layers.Count && TryBind(network, Layers)) _network = network;
            return;
        }

        _network = network;
        Layers.AddRange(network.Layers);
    }

    private static bool TryBind(ConvTasNetNetwork<T> network, IReadOnlyList<ILayer<T>> layers)
    {
        // BindTo restores the network before it throws, so a refused list leaves it as built.
        try
        {
            network.BindTo(layers);
            return true;
        }
        catch (InvalidOperationException)
        {
            return false;
        }
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
    /// <returns>Separated sources tensor [batch, sources, samples].</returns>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        var preprocessed = PreprocessAudio(input);

        if (IsOnnxMode)
        {
            var output = RunOnnxInference(preprocessed);
            return PostprocessOutput(output);
        }

        // EnhancementStrength applies to what a caller receives, in both modes; the training forward
        // below stays raw because the loss must see the network's own output.
        return PostprocessOutput(ForwardNative(preprocessed));
    }

    /// <inheritdoc />
    /// <remarks>The same network as inference, without the EnhancementStrength scaling applied to
    /// what Predict returns.</remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        if (IsOnnxMode) throw new InvalidOperationException("Cannot train in ONNX inference mode.");
        return ForwardNative(PreprocessAudio(input));
    }

    /// <summary>
    /// Captures every named stage of the network for a single input.
    /// </summary>
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

        if (_network is null)
        {
            activations["Output"] = PostprocessOutput(ForwardNative(preprocessed)).Clone();
            return activations;
        }

        _network.BindTo(Layers);
        var separated = _network.Forward(preprocessed, activations);
        activations["Output"] = PostprocessOutput(separated).Clone();
        return activations;
    }

    private Tensor<T> ForwardNative(Tensor<T> mixture)
    {
        if (_network is null)
        {
            // A caller-supplied stack in another layout is an ordinary layer chain.
            var output = mixture;
            foreach (var layer in Layers) output = layer.Forward(output);
            return output;
        }

        _network.BindTo(Layers);
        return _network.Forward(mixture);
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
    /// One training step on the base tape: the whole network (encoder, every TCN block, the mask head and the
    /// decoder) is differentiated against the loss, negative SI-SNR with permutation-invariant training by default.
    /// </summary>
    /// <param name="input">The mixture, [batch, samples] or [samples].</param>
    /// <param name="expected">The reference sources, [batch, sources, samples].</param>
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
        {
            throw new InvalidOperationException("Cannot train in ONNX inference mode.");
        }

        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expected, Optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    #endregion

    #region Abstract Method Implementations

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

    #endregion
}

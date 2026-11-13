using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Enhancement;

/// <summary>
/// Neural network-based noise reducer for high-quality audio enhancement.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This model uses an encoder-bottleneck-decoder architecture inspired by U-Net
/// to learn the mapping from noisy audio to clean audio. It operates in the
/// time-frequency domain using STFT for analysis and synthesis.
/// Note: The current implementation uses a simplified dense decoder; a full U-Net
/// with transposed convolutions and skip connections is planned for future versions.
/// </para>
/// <para><b>For Beginners:</b> This is like a "magic eraser" for audio noise!
///
/// How it works:
/// 1. Converts audio to a spectrogram (picture of sound)
/// 2. Neural network learns to identify and remove noise patterns
/// 3. Converts cleaned spectrogram back to audio
///
/// Key features:
/// - Works on any type of noise (AC hum, fan noise, traffic, etc.)
/// - Preserves speech/music quality while removing noise
/// - Can be trained on your specific noise conditions
/// - Supports real-time streaming for live applications
///
/// Use cases:
/// - Podcast/video production (remove background noise)
/// - Voice calls (improve speech clarity)
/// - Music restoration (remove hiss/crackle from old recordings)
/// - Hearing aids (enhance speech in noisy environments)
///
/// Two modes of operation:
/// 1. ONNX Mode: Load a pre-trained model for instant use
/// 2. Native Mode: Train your own model on custom data
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var noisyAudioTensor = Tensor&lt;float&gt;.CreateRandom(1, 16000);
/// // Create a neural noise reducer with default settings
/// var reducer = new NeuralNoiseReducer&lt;float&gt;();
///
/// // Or with custom architecture
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputSize: 257,
///     outputSize: 257);
/// var customReducer = new NeuralNoiseReducer&lt;float&gt;(architecture);
///
/// // Enhance noisy audio
/// Tensor&lt;float&gt; cleanAudio = reducer.Enhance(noisyAudioTensor);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Denoising)]
[ModelTask(ModelTask.Enhancement)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
// Citation URL corrected. arXiv 1406.2279 is "$B_s \to K \ell v$ form factors from lattice QCD" — an
// unrelated particle-physics paper. This work was published in IEEE/ACM TASLP vol. 23 no. 1 (Jan 2015),
// pp. 7-19, and is not on arXiv, so the canonical DOI replaces the invented preprint id.
[ResearchPaper("A Regression Approach to Speech Enhancement Based on Deep Neural Networks",
    "https://doi.org/10.1109/TASLP.2014.2364452",
    Year = 2015,
    Authors = "Yong Xu, Jun Du, Li-Rong Dai, Chin-Hui Lee")]
public partial class NeuralNoiseReducer<T> : AudioNeuralNetworkBase<T>, IAudioEnhancer<T>
{
    /// <inheritdoc />
    /// <remarks>
    /// DERIVED, not stored: <c>PredictCore</c> runs the inherited <c>Forward</c> (a fold over
    /// <c>Layers</c>) and <c>PostprocessOutput</c> returns the mask unchanged. The last layer of
    /// <c>CreateNeuralNoiseReducerLayers</c> is <c>DenseLayer&lt;T&gt;(freqBins, sigmoid)</c> with
    /// <c>freqBins = fftSize / 2 + 1</c> computed inside the helper from the <c>_fftSize</c> this
    /// model passes - a per-bin soft mask. 257 for the default 512-point FFT, and that number appears
    /// nowhere in the options.
    /// </remarks>
    protected override int OutputFeatureWidth => _fftSize / 2 + 1;

    private readonly NeuralNoiseReducerOptions _options;

    #region Execution Mode

    /// <summary>
    /// Indicates whether to use native training mode.
    /// </summary>
    private bool _useNativeMode;

    /// <summary>
    /// Path to ONNX model (ONNX mode only).
    /// </summary>
    private string? _modelPath;

    #endregion

    #region ONNX Mode Fields

    // ONNX models are stored in base class: OnnxEncoder, OnnxDecoder, OnnxModel

    #endregion

    #region Native Mode Fields

    /// <summary>
    /// Encoder layers (downsampling path).
    /// </summary>
    private List<ILayer<T>> _encoderLayers = new();

    /// <summary>
    /// Decoder layers (upsampling path).
    /// </summary>
    private List<ILayer<T>> _decoderLayers = new();

    /// <summary>
    /// Bottleneck layer.
    /// </summary>
    private ILayer<T>? _bottleneckLayer;

    /// <summary>
    /// Output projection layer.
    /// </summary>
    private ILayer<T>? _outputLayer;

    /// <summary>
    /// Loss function for training.
    /// </summary>
    private ILossFunction<T> _lossFunction;

    #endregion

    #region Configuration

    /// <summary>
    /// FFT size for STFT analysis (non-readonly for deserialization support).
    /// </summary>
    private int _fftSize;

    /// <summary>
    /// Hop size between STFT frames (non-readonly for deserialization support).
    /// </summary>
    private int _hopSize;

    /// <summary>
    /// Number of encoder/decoder stages (non-readonly for deserialization support).
    /// </summary>
    private int _numStages;

    /// <summary>
    /// Base number of filters (doubled at each stage) (non-readonly for deserialization support).
    /// </summary>
    private int _baseFilters;

    /// <summary>
    /// Hidden dimension in bottleneck (non-readonly for deserialization support).
    /// </summary>
    private int _bottleneckDim;

    #endregion

    #region IAudioEnhancer Properties

    /// <inheritdoc/>
    public int NumChannels { get; protected set; }

    /// <inheritdoc/>
    public double EnhancementStrength { get; set; }

    /// <inheritdoc/>
    public int LatencySamples => _fftSize;

    #endregion

    #region Streaming State

    /// <summary>
    /// Input buffer for streaming mode.
    /// </summary>
    [Scratch]
    private T[] _inputBuffer = new T[0];

    /// <summary>
    /// Output buffer for overlap-add.
    /// </summary>
    [Scratch]
    private T[] _outputBuffer = new T[0];

    /// <summary>
    /// Current position in input buffer.
    /// </summary>
    private int _bufferPosition;

    /// <summary>
    /// Window function for STFT.
    /// </summary>
    [Buffer]
    private Tensor<T> _window = new Tensor<T>([0]);

    /// <summary>
    /// Noise profile estimate (optional).
    /// </summary>
    private T[]? _noiseProfile;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a NeuralNoiseReducer with default configuration for native training mode.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a noise reducer with sensible defaults
    /// (16kHz sample rate, 512 FFT size). You can train it on your own noisy/clean audio pairs.</para>
    /// </remarks>
    public NeuralNoiseReducer()
        : this(CreateDefaultArchitecture())
    {
    }

    private static NeuralNetworkArchitecture<T> CreateDefaultArchitecture()
    {
        return new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 257,   // fftSize/2 + 1 = 512/2 + 1
            outputSize: 257);
    }

    /// <summary>
    /// Creates a NeuralNoiseReducer in ONNX inference mode using a pre-trained model.
    /// </summary>
    /// <param name="architecture">The neural network architecture (user-defined for full customization).</param>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="sampleRate">Audio sample rate (default: 16000).</param>
    /// <param name="fftSize">FFT window size (default: 512).</param>
    /// <param name="hopSize">Hop size between frames (default: 256).</param>
    /// <param name="numChannels">Number of audio channels (default: 1).</param>
    /// <param name="enhancementStrength">Enhancement strength 0-1 (default: 0.8).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor when you have a pre-trained model.
    /// Just point to the ONNX file and start enhancing audio immediately.
    /// </para>
    /// </remarks>
    public NeuralNoiseReducer(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        NeuralNoiseReducerOptions? options = null)
        : base(architecture: architecture, new MeanAbsoluteErrorLoss<T>())
    {
        _options = options ?? new NeuralNoiseReducerOptions();
        _options.Validate();
        Options = _options;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (_options.EnhancementStrength < 0.0 || _options.EnhancementStrength > 1.0)
            throw new ArgumentOutOfRangeException(nameof(_options.EnhancementStrength), "Enhancement strength must be between 0.0 and 1.0.");
        if (_options.HopSize <= 0 || _options.HopSize > _options.FftSize)
            throw new ArgumentOutOfRangeException(nameof(_options.HopSize), "Hop size must be positive and not exceed FFT size.");

        _useNativeMode = false;
        _modelPath = modelPath;
        _lossFunction = new MeanAbsoluteErrorLoss<T>();

        SampleRate = _options.SampleRate;
        NumChannels = _options.NumChannels;
        _fftSize = _options.FftSize;
        _hopSize = _options.HopSize;
        _numStages = _options.NumStages;
        _baseFilters = _options.BaseFilters;
        _bottleneckDim = _options.BottleneckDim;
        EnhancementStrength = _options.EnhancementStrength;

        // Load ONNX model
        OnnxModel = new OnnxModel<T>(modelPath);

        InitializeStreamingBuffers();
    }

    /// <summary>
    /// Creates a NeuralNoiseReducer in native training mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture (user-defined for full customization).</param>
    /// <param name="sampleRate">Audio sample rate (default: 16000).</param>
    /// <param name="fftSize">FFT window size (default: 512).</param>
    /// <param name="hopSize">Hop size between frames (default: 256).</param>
    /// <param name="numChannels">Number of audio channels (default: 1).</param>
    /// <param name="numStages">Number of encoder/decoder stages (default: 4).</param>
    /// <param name="baseFilters">Base filter count (default: 32).</param>
    /// <param name="bottleneckDim">Bottleneck hidden dimension (default: 256).</param>
    /// <param name="enhancementStrength">Enhancement strength 0-1 (default: 0.8).</param>
    /// <param name="lossFunction">Loss function for training (default: L1 loss).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor when you want to train your own model.
    /// You'll need noisy/clean audio pairs for training.
    /// </para>
    /// </remarks>
    public NeuralNoiseReducer(
        NeuralNetworkArchitecture<T> architecture,
        NeuralNoiseReducerOptions? options = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture: architecture, lossFunction ?? new MeanAbsoluteErrorLoss<T>())
    {
        _options = options ?? new NeuralNoiseReducerOptions();
        _options.Validate();
        Options = _options;
        // Validate parameters
        if (_options.EnhancementStrength < 0.0 || _options.EnhancementStrength > 1.0)
            throw new ArgumentOutOfRangeException(nameof(_options.EnhancementStrength), "Enhancement strength must be between 0.0 and 1.0.");
        if (_options.HopSize <= 0 || _options.HopSize > _options.FftSize)
            throw new ArgumentOutOfRangeException(nameof(_options.HopSize), "Hop size must be positive and not exceed FFT size.");

        _useNativeMode = true;
        _modelPath = null;
        _lossFunction = lossFunction ?? new MeanAbsoluteErrorLoss<T>();

        SampleRate = _options.SampleRate;
        NumChannels = _options.NumChannels;
        _fftSize = _options.FftSize;
        _hopSize = _options.HopSize;
        _numStages = _options.NumStages;
        _baseFilters = _options.BaseFilters;
        _bottleneckDim = _options.BottleneckDim;
        EnhancementStrength = _options.EnhancementStrength;

        InitializeLayers();
        InitializeStreamingBuffers();
    }

    private void InitializeStreamingBuffers()
    {
        _inputBuffer = new T[_fftSize];
        _outputBuffer = new T[_fftSize];
        _bufferPosition = 0;
        _window = CreateHannWindow(_fftSize);
    }

    private Tensor<T> CreateHannWindow(int size)
    {
        return AudioEngineOps.SymmetricHannWindow<T>(Engine, size);
    }

    /// <summary>
    /// The <see cref="_fftSize"/>-sample frame at <paramref name="start"/> multiplied by the analysis
    /// window; samples past the end of the signal are zero.
    /// </summary>
    private T[] WindowedFrame(Tensor<T> signal, int start)
    {
        int available = Math.Max(0, Math.Min(_fftSize, signal.Length - start));
        var samples = available > 0
            ? AudioEngineOps.FitLength(Engine, Engine.TensorSlice(signal, [start], [available]), _fftSize)
            : new Tensor<T>([_fftSize]);
        return Engine.TensorMultiply(samples, _window).ToArray();
    }

    #endregion

    #region Layer Initialization

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;

        var layers = (Architecture.Layers != null && Architecture.Layers.Count > 0)
            ? Architecture.Layers.ToList()
            : LayerHelper<T>.CreateNeuralNoiseReducerLayers(
                fftSize: _fftSize, baseFilters: _baseFilters, numStages: _numStages,
                bottleneckDim: _bottleneckDim).ToList();

        Layers.Clear();
        _encoderLayers.Clear();
        Layers.AddRange(layers);

        // Assign internal references for forward pass. The default stack
        // (CreateNeuralNoiseReducerLayers) leads with a FlattenLayer so the dense
        // spectral-mapping net is rank-agnostic; skip it when indexing the
        // encoder/bottleneck/output references (a custom Architecture without a
        // leading Flatten uses offset 0).
        int offset = layers.Count > 0 && layers[0] is NeuralNetworks.Layers.FlattenLayer<T> ? 1 : 0;
        int expectedCount = offset + _numStages + 2; // [flatten] + encoder stages + bottleneck + output
        if (layers.Count < expectedCount)
        {
            throw new ArgumentException(
                $"Custom architecture must have at least {expectedCount} layers " +
                $"({_numStages} encoder stages + bottleneck + output).",
                nameof(Architecture));
        }

        for (int i = 0; i < _numStages; i++)
            _encoderLayers.Add(layers[offset + i]);
        _bottleneckLayer = layers[offset + _numStages];
        _outputLayer = layers[offset + _numStages + 1];
    }

    #endregion

    #region IAudioEnhancer Implementation

    /// <inheritdoc/>
    public Tensor<T> Enhance(Tensor<T> audio)
    {
        var samples = audio.ToVector().ToArray();
        var enhanced = ProcessOverlapAdd(samples);

        return new Tensor<T>(enhanced, [enhanced.Length]);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Reference-based enhancement is acoustic echo cancellation: given the
    /// near-end microphone signal <paramref name="audio"/> (which contains
    /// the local talker plus an echoed copy of the far-end <paramref name="reference"/>),
    /// estimate the echo path and subtract its contribution. Implemented as
    /// a Normalised Least Mean Squares (NLMS) adaptive filter — the standard
    /// classical-DSP echo canceller from Haykin 2002 "Adaptive Filter Theory"
    /// Ch. 6 and ITU-T G.168. NLMS does not need a pretrained model; the
    /// filter coefficients converge online from the streaming input.
    /// </para>
    /// <para>
    /// Filter length M defaults to 512 taps — at the 16 kHz <see cref="SampleRate"/>
    /// the noise reducer is configured for, that's 32 ms of impulse response,
    /// which covers typical headset / loudspeaker echo delays for in-room
    /// scenarios. Normalised step size μ = 0.5 is the canonical value for
    /// fast convergence without divergence on speech-like inputs.
    /// </para>
    /// </remarks>
    public Tensor<T> EnhanceWithReference(Tensor<T> audio, Tensor<T> reference)
    {
        Validation.Guard.NotNull(audio);
        Validation.Guard.NotNull(reference);

        // Reject multi-channel / unexpected shapes BEFORE flattening — the
        // documented contract is mono signals shaped [N], [1, N] or [N, 1].
        // A tensor like [2, N] would otherwise be flattened into a single
        // NLMS stream and corrupt both channels by interleaving samples.
        if (!IsMonoVectorLike(audio))
        {
            throw new ArgumentException(
                "EnhanceWithReference currently supports only mono tensors shaped [N], [1, N], or [N, 1].",
                nameof(audio));
        }
        if (!IsMonoVectorLike(reference))
        {
            throw new ArgumentException(
                "EnhanceWithReference currently supports only mono tensors shaped [N], [1, N], or [N, 1].",
                nameof(reference));
        }

        // Flatten both signals to a contiguous sample buffer. Shapes accepted:
        // [N] or [1, N] or [N, 1] — anything whose Length equals total samples.
        int n = audio.Length;
        int refLen = reference.Length;
        if (refLen == 0 || n == 0)
        {
            return audio.Clone();
        }

        var d = new double[n];
        for (int i = 0; i < n; i++) d[i] = NumOps.ToDouble(audio[i]);

        var x = new double[refLen];
        for (int i = 0; i < refLen; i++) x[i] = NumOps.ToDouble(reference[i]);

        // Sample-rate-aware filter length: NLMS filter taps cover a fixed
        // impulse-response DURATION (~32 ms), not a fixed sample count.
        // Hardcoding 512 only happens to be 32 ms at 16 kHz; at 48 kHz it
        // covers ~10.7 ms which is too short for typical in-room echo.
        int filterLength = Math.Max(64, (SampleRate * 32) / 1000);
        const double mu = 0.5;             // canonical NLMS step size (Haykin §6.4)
        const double regularisation = 1e-6;

        var enhanced = NLMSEchoCancel(d, x, filterLength, mu, regularisation);

        // Honor the public EnhancementStrength knob — the previous
        // implementation always applied full cancellation here, making the
        // user-facing strength parameter silently ineffective on this
        // path. Standard wet/dry mix: out = d + strength * (enhanced - d).
        // Clamp defensively because the property has a public setter:
        // a later assignment like -0.2 or 1.5 would otherwise invert or
        // amplify the mix instead of just scaling cancellation.
        // Math.Clamp is net5+ — fall back to Math.Max/Min for net471 compatibility.
        double mix = Math.Max(0.0, Math.Min(1.0, EnhancementStrength));
        var result = new Tensor<T>(audio._shape);
        for (int i = 0; i < n; i++)
        {
            double value = d[i] + mix * (enhanced[i] - d[i]);
            result[i] = NumOps.FromDouble(value);
        }
        return result;
    }

    /// <summary>
    /// Returns <c>true</c> when <paramref name="tensor"/> is a mono vector
    /// — shape [N], [1, N], or [N, 1] — i.e. genuinely a single audio
    /// channel rather than an interleaved or stacked multichannel tensor.
    /// </summary>
    private static bool IsMonoVectorLike(Tensor<T> tensor) =>
        tensor._shape.Length == 1
        || (tensor._shape.Length == 2 && (tensor._shape[0] == 1 || tensor._shape[1] == 1));

    /// <summary>
    /// Normalised LMS (NLMS) acoustic echo canceller. Per Haykin 2002 §6.4:
    /// <code>
    ///   y[n]   = Σ_{k=0..M-1} w_k · x[n-k]
    ///   e[n]   = d[n] - y[n]
    ///   w_k   += μ · e[n] · x[n-k] / (Σ_j x[n-j]² + ε)
    /// </code>
    /// The output is the error signal e[n], which is the near-end audio with
    /// the estimated echo subtracted out.
    /// </summary>
    /// <param name="d">Microphone signal (near-end + echo).</param>
    /// <param name="x">Far-end reference signal that produced the echo.</param>
    /// <param name="filterLength">Number of filter taps (impulse response length in samples).</param>
    /// <param name="mu">Normalised step size; must be in (0, 2) for stability — 0.5 is the canonical value.</param>
    /// <param name="regularisation">Small constant added to the input-power denominator to avoid division-by-zero on silent reference.</param>
    private static double[] NLMSEchoCancel(double[] d, double[] x, int filterLength, double mu, double regularisation)
    {
        int n = d.Length;
        int m = filterLength;
        var w = new double[m];
        var buf = new double[m];      // circular tap-delay line
        var output = new double[n];
        double powerWindow = 0.0;     // running ||x[n-M..n]||²

        for (int t = 0; t < n; t++)
        {
            // Slide tap-delay line: oldest sample drops out, newest sample comes in.
            int idx = t % m;
            double oldest = buf[idx];
            double sample = t < x.Length ? x[t] : 0.0;
            buf[idx] = sample;
            powerWindow += sample * sample - oldest * oldest;
            if (powerWindow < 0.0) powerWindow = 0.0; // guard against float drift

            // y[t] = Σ_k w[k] * x[t-k] using the circular buffer.
            double y = 0.0;
            for (int k = 0; k < m; k++)
            {
                int bufIdx = idx - k;
                if (bufIdx < 0) bufIdx += m;
                y += w[k] * buf[bufIdx];
            }

            // e[t] = d[t] - y[t]
            double e = d[t] - y;
            output[t] = e;

            // w[k] += μ · e · x[t-k] / (||x[t-M..t]||² + ε)
            double normalised = mu * e / (powerWindow + regularisation);
            for (int k = 0; k < m; k++)
            {
                int bufIdx = idx - k;
                if (bufIdx < 0) bufIdx += m;
                w[k] += normalised * buf[bufIdx];
            }
        }

        return output;
    }

    /// <inheritdoc/>
    public Tensor<T> ProcessChunk(Tensor<T> audioChunk)
    {
        var samples = audioChunk.ToVector().ToArray();
        var enhanced = ProcessStreamingChunk(samples);

        return new Tensor<T>(enhanced, [enhanced.Length]);
    }

    /// <summary>
    /// Resets streaming state (explicit interface implementation to avoid conflict with base).
    /// </summary>
    void IAudioEnhancer<T>.ResetState()
    {
        ResetEnhancerState();
    }

    /// <summary>
    /// Resets the enhancer streaming state.
    /// </summary>
    public void ResetEnhancerState()
    {
        _inputBuffer = new T[_fftSize];
        _outputBuffer = new T[_fftSize];
        _bufferPosition = 0;
    }

    /// <inheritdoc/>
    public void EstimateNoiseProfile(Tensor<T> noiseOnlyAudio)
    {
        var samples = noiseOnlyAudio.ToVector().ToArray();
        _noiseProfile = EstimateNoiseSpectrum(samples);
    }

    #endregion

    #region Audio Processing

    /// <summary>
    /// Processes audio using overlap-add STFT method.
    /// </summary>
    private T[] ProcessOverlapAdd(T[] input)
    {
        if (_window.Length == 0)
            InitializeStreamingBuffers();

        int numFrames = (input.Length - _fftSize) / _hopSize + 1;
        var signal = new Tensor<T>(input, [input.Length]);
        var synthesized = new List<Tensor<T>>();
        var starts = new List<int>();

        for (int frame = 0; frame < numFrames; frame++)
        {
            int start = frame * _hopSize;

            // Extract and window frame (zero beyond the end of the input)
            var frameData = WindowedFrame(signal, start);

            // Compute STFT
            var (magnitudes, phases) = ComputeSTFT(frameData);

            // Apply neural network enhancement
            var enhancedMagnitudes = EnhanceSpectrum(magnitudes);

            // Compute inverse STFT
            var enhanced = ComputeISTFT(enhancedMagnitudes, phases);

            // Window for overlap-add
            synthesized.Add(Engine.TensorMultiply(new Tensor<T>(enhanced, [enhanced.Length]), _window));
            starts.Add(start);
        }

        return AudioEngineOps.OverlapAdd(Engine, synthesized, starts, input.Length).ToArray();
    }

    /// <summary>
    /// Processes audio in streaming mode.
    /// </summary>
    private T[] ProcessStreamingChunk(T[] chunk)
    {
        if (_inputBuffer.Length == 0 || _outputBuffer.Length == 0)
            InitializeStreamingBuffers();

        var output = new T[chunk.Length];
        int outputPos = 0;

        for (int i = 0; i < chunk.Length; i++)
        {
            _inputBuffer[_bufferPosition] = chunk[i];
            _bufferPosition++;

            if (_bufferPosition >= _hopSize)
            {
                // Process frame. _bufferPosition == _hopSize here, so the frame reads the input buffer
                // in order: frame[j] = inputBuffer[j] * window[j].
                var frameData = Engine.TensorMultiply(new Tensor<T>(_inputBuffer, [_inputBuffer.Length]), _window).ToArray();

                var (magnitudes, phases) = ComputeSTFT(frameData);
                var enhancedMagnitudes = EnhanceSpectrum(magnitudes);
                var enhanced = ComputeISTFT(enhancedMagnitudes, phases);

                // Overlap-add to output buffer
                Engine.TensorAdd(
                    new Tensor<T>(_outputBuffer, [_outputBuffer.Length]),
                    Engine.TensorMultiply(new Tensor<T>(enhanced, [enhanced.Length]), _window))
                    .CopyTo(_outputBuffer.AsSpan());

                // Output hop samples
                for (int j = 0; j < _hopSize && outputPos < output.Length; j++)
                {
                    output[outputPos++] = _outputBuffer[j];
                }

                // Shift output buffer
                Array.Copy(_outputBuffer, _hopSize, _outputBuffer, 0, _fftSize - _hopSize);
                Array.Clear(_outputBuffer, _fftSize - _hopSize, _hopSize);

                _bufferPosition = 0;
            }
        }

        return output;
    }

    /// <summary>
    /// Applies neural network enhancement to spectrum.
    /// </summary>
    private T[] EnhanceSpectrum(T[] magnitudes)
    {
        int numBins = magnitudes.Length;
        var magnitude = new Tensor<T>(magnitudes, [numBins]);

        // Apply spectral subtraction if noise profile is available:
        // max(0, mag - strength * noise), NaN propagating as Math.Max does.
        var processed = magnitude;
        if (_noiseProfile is not null && _noiseProfile.Length == numBins)
        {
            var value = Engine.TensorSubtract(
                magnitude,
                Engine.TensorMultiplyScalar(new Tensor<T>(_noiseProfile, [numBins]), NumOps.FromDouble(EnhancementStrength)));
            processed = Engine.TensorWhere(Engine.TensorLessThan(value, NumOps.Zero), new Tensor<T>([numBins]), value);
        }

        // Network input [1, bins, 1]
        var input = Engine.Reshape(processed, [1, numBins, 1]);

        Tensor<T> mask;

        if (IsOnnxMode)
        {
            mask = RunOnnxInference(input);
        }
        else
        {
            mask = Forward(input);
        }

        // Apply mask to magnitudes with enhancement strength (a detached copy as before).
        // mask is in range 0-1, where 1 = keep, 0 = remove; bins past the mask keep 1.
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var flatMask = AudioEngineOps.Flatten(Engine, mask);
        var m = flatMask.Length >= numBins
            ? AudioEngineOps.FitLength(Engine, flatMask, numBins)
            : Engine.Concat(new[] { flatMask, Tensor<T>.CreateDefault([numBins - flatMask.Length], NumOps.One) }, 0);

        // m = 1 - strength * (1 - m)
        T one = NumOps.One;
        var softened = Engine.TensorAddScalar(
            Engine.TensorNegate(Engine.TensorMultiplyScalar(Engine.TensorAddScalar(Engine.TensorNegate(m), one), NumOps.FromDouble(EnhancementStrength))),
            one);

        return Engine.TensorMultiply(processed, softened).ToArray();
    }

    #endregion

    #region STFT Implementation

    /// <summary>
    /// Computes Short-Time Fourier Transform using FftSharp library (O(N log N) algorithm).
    /// </summary>
    private (T[] Magnitudes, T[] Phases) ComputeSTFT(T[] frame)
    {
        // O(N log N) complex FFT of the (zero-padded) frame; magnitude and phase of the positive
        // frequencies (N/2+1 bins).
        var (magnitudes, phases) = AudioEngineOps.HalfSpectrum(Engine, new Tensor<T>(frame, [frame.Length]), _fftSize);
        return (magnitudes.ToArray(), phases.ToArray());
    }

    /// <summary>
    /// Computes Inverse Short-Time Fourier Transform using FftSharp library.
    /// </summary>
    private T[] ComputeISTFT(T[] magnitudes, T[] phases)
    {
        // Reconstruct full complex spectrum from magnitude/phase (N/2+1 positive frequency bins,
        // negative frequencies by conjugate symmetry X[N-k] = conj(X[k])), then the real part of the
        // 1/N-normalised inverse FFT.
        var phaseTensor = new Tensor<T>(phases, [phases.Length]);
        if (phases.Length != magnitudes.Length)
            phaseTensor = AudioEngineOps.FitLength(Engine, phaseTensor, magnitudes.Length);
        return AudioEngineOps.InverseHalfSpectrum(Engine, new Tensor<T>(magnitudes, [magnitudes.Length]), phaseTensor, _fftSize).ToArray();
    }

    /// <summary>
    /// Estimates noise spectrum from noise-only audio.
    /// </summary>
    private T[] EstimateNoiseSpectrum(T[] noiseAudio)
    {
        int numFrames = (noiseAudio.Length - _fftSize) / _hopSize + 1;
        int numBins = _fftSize / 2 + 1;
        var signal = new Tensor<T>(noiseAudio, [noiseAudio.Length]);
        var sumMagnitudes = new Tensor<T>([numBins]);

        for (int frame = 0; frame < numFrames; frame++)
        {
            var (magnitudes, _) = ComputeSTFT(WindowedFrame(signal, frame * _hopSize));
            sumMagnitudes = Engine.TensorAdd(sumMagnitudes, new Tensor<T>(magnitudes, [numBins]));
        }

        // Average
        return Engine.TensorDivideScalar(sumMagnitudes, NumOps.FromDouble(numFrames)).ToArray();
    }

    #endregion

    #region AudioNeuralNetworkBase Implementation

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        // Convert to spectrogram for neural network input
        var samples = AudioEngineOps.Flatten(Engine, rawAudio);

        // Apply window to the first min(length, fftSize) samples
        int count = Math.Min(samples.Length, _fftSize);
        var windowed = count > 0
            ? Engine.TensorMultiply(
                Engine.TensorSlice(samples, [0], [count]),
                AudioEngineOps.FitLength(Engine, _window, count)).ToArray()
            : new T[0];

        // Compute magnitude spectrum
        var (magnitudes, _) = ComputeSTFT(windowed);

        // Reshape to [1, freq_bins, 1]
        return new Tensor<T>(magnitudes, [1, magnitudes.Length, 1]);
    }

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        // Model outputs a mask in range 0-1
        return modelOutput;
    }

    #endregion

    #region Training

    // UpdateParameters restated the base verbatim; ModelBase routes it to SetParameters.
    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        var preprocessed = PreprocessAudio(input);
        Tensor<T> output;

        if (IsOnnxMode)
        {
            output = RunOnnxInference(preprocessed);
        }
        else
        {
            output = Forward(preprocessed);
        }

        return PostprocessOutput(output);
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "NeuralNoiseReducer",
            Version = "1.0",
            Description = "Neural network-based audio noise reducer with U-Net architecture"
        };

        metadata.SetProperty("SampleRate", SampleRate);
        metadata.SetProperty("FFTSize", _fftSize);
        metadata.SetProperty("HopSize", _hopSize);
        metadata.SetProperty("NumStages", _numStages);
        metadata.SetProperty("BaseFilters", _baseFilters);
        metadata.SetProperty("BottleneckDim", _bottleneckDim);
        metadata.SetProperty("EnhancementStrength", EnhancementStrength);
        metadata.SetProperty("UseNativeMode", _useNativeMode);

        // Also surface the model-specific hyperparameters via AdditionalInfo —
        // the canonical metadata channel consumers (and the model-family
        // invariant tests) read; SetProperty above writes the separate
        // Properties dictionary, leaving AdditionalInfo empty otherwise.
        metadata.AdditionalInfo["SampleRate"] = SampleRate;
        metadata.AdditionalInfo["FFTSize"] = _fftSize;
        metadata.AdditionalInfo["HopSize"] = _hopSize;
        metadata.AdditionalInfo["NumStages"] = _numStages;
        metadata.AdditionalInfo["BaseFilters"] = _baseFilters;
        metadata.AdditionalInfo["BottleneckDim"] = _bottleneckDim;
        metadata.AdditionalInfo["EnhancementStrength"] = EnhancementStrength;
        metadata.AdditionalInfo["UseNativeMode"] = _useNativeMode;

        return metadata;
    }

    /// <inheritdoc/>


    /// <inheritdoc/>


    #endregion
}

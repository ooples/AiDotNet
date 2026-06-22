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
[ResearchPaper("A Regression Approach to Speech Enhancement Based on Deep Neural Networks", "https://arxiv.org/abs/1406.2279", Year = 2015, Authors = "Yong Xu, Jun Du, Li-Rong Dai, Chin-Hui Lee")]
public class NeuralNoiseReducer<T> : AudioNeuralNetworkBase<T>, IAudioEnhancer<T>
{
    private readonly NeuralNoiseReducerOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

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
    private Vector<T> _inputBuffer = new Vector<T>(0);

    /// <summary>
    /// Output buffer for overlap-add.
    /// </summary>
    private Vector<T> _outputBuffer = new Vector<T>(0);

    /// <summary>
    /// Current position in input buffer.
    /// </summary>
    private int _bufferPosition;

    /// <summary>
    /// Window function for STFT.
    /// </summary>
    private Vector<T> _window = new Vector<T>(0);

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
        int sampleRate = 16000,
        int fftSize = 512,
        int hopSize = 256,
        int numChannels = 1,
        double enhancementStrength = 0.8,
        NeuralNoiseReducerOptions? options = null)
        : base(architecture, new MeanAbsoluteErrorLoss<T>())
    {
        _options = options ?? new NeuralNoiseReducerOptions();
        Options = _options;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (enhancementStrength < 0.0 || enhancementStrength > 1.0)
            throw new ArgumentOutOfRangeException(nameof(enhancementStrength), "Enhancement strength must be between 0.0 and 1.0.");
        if (sampleRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(sampleRate), "Sample rate must be positive.");
        if (fftSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(fftSize), "FFT size must be positive.");
        if (hopSize <= 0 || hopSize > fftSize)
            throw new ArgumentOutOfRangeException(nameof(hopSize), "Hop size must be positive and not exceed FFT size.");
        if (numChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(numChannels), "Number of channels must be positive.");

        _useNativeMode = false;
        _modelPath = modelPath;
        _lossFunction = new MeanAbsoluteErrorLoss<T>();

        SampleRate = sampleRate;
        NumChannels = numChannels;
        _fftSize = fftSize;
        _hopSize = hopSize;
        _numStages = 4;
        _baseFilters = 32;
        _bottleneckDim = 256;
        EnhancementStrength = enhancementStrength;

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
        int sampleRate = 16000,
        int fftSize = 512,
        int hopSize = 256,
        int numChannels = 1,
        int numStages = 4,
        int baseFilters = 32,
        int bottleneckDim = 256,
        double enhancementStrength = 0.8,
        ILossFunction<T>? lossFunction = null,
        NeuralNoiseReducerOptions? options = null)
        : base(architecture, lossFunction ?? new MeanAbsoluteErrorLoss<T>())
    {
        _options = options ?? new NeuralNoiseReducerOptions();
        Options = _options;
        // Validate parameters
        if (enhancementStrength < 0.0 || enhancementStrength > 1.0)
            throw new ArgumentOutOfRangeException(nameof(enhancementStrength), "Enhancement strength must be between 0.0 and 1.0.");
        if (sampleRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(sampleRate), "Sample rate must be positive.");
        if (fftSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(fftSize), "FFT size must be positive.");
        if (hopSize <= 0 || hopSize > fftSize)
            throw new ArgumentOutOfRangeException(nameof(hopSize), "Hop size must be positive and not exceed FFT size.");
        if (numChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(numChannels), "Number of channels must be positive.");
        if (numStages <= 0)
            throw new ArgumentOutOfRangeException(nameof(numStages), "Number of stages must be positive.");
        if (baseFilters <= 0)
            throw new ArgumentOutOfRangeException(nameof(baseFilters), "Base filters must be positive.");
        if (bottleneckDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(bottleneckDim), "Bottleneck dimension must be positive.");

        _useNativeMode = true;
        _modelPath = null;
        _lossFunction = lossFunction ?? new MeanAbsoluteErrorLoss<T>();

        SampleRate = sampleRate;
        NumChannels = numChannels;
        _fftSize = fftSize;
        _hopSize = hopSize;
        _numStages = numStages;
        _baseFilters = baseFilters;
        _bottleneckDim = bottleneckDim;
        EnhancementStrength = enhancementStrength;

        InitializeLayers();
        InitializeStreamingBuffers();
    }

    private void InitializeStreamingBuffers()
    {
        _inputBuffer = new Vector<T>(_fftSize);
        _outputBuffer = new Vector<T>(_fftSize);
        _bufferPosition = 0;
        _window = CreateHannWindow(_fftSize);
    }

    private Vector<T> CreateHannWindow(int size)
    {
        var window = new Vector<T>(size);
        for (int i = 0; i < size; i++)
        {
            var value = 0.5 * (1 - Math.Cos(2 * Math.PI * i / (size - 1)));
            window[i] = NumOps.FromDouble(value);
        }
        return window;
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

        // Write straight into the tensor's flat storage. Tensor<T>.ToVector()
        // returns a COPY (it allocates a fresh Vector<T> and flattens into it),
        // so writing to that copy and returning `result` would discard every
        // value and yield an all-zero tensor.
        var result = new Tensor<T>([enhanced.Length]);
        for (int i = 0; i < enhanced.Length; i++)
        {
            result[i] = enhanced[i];
        }
        return result;
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

        // Write straight into the tensor's flat storage. Tensor<T>.ToVector()
        // returns a COPY (it allocates a fresh Vector<T> and flattens into it),
        // so writing to that copy and returning `result` would discard every
        // value and yield an all-zero tensor.
        var result = new Tensor<T>([enhanced.Length]);
        for (int i = 0; i < enhanced.Length; i++)
        {
            result[i] = enhanced[i];
        }
        return result;
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
        _inputBuffer = new Vector<T>(_fftSize);
        _outputBuffer = new Vector<T>(_fftSize);
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
        var output = new T[input.Length];

        for (int frame = 0; frame < numFrames; frame++)
        {
            int start = frame * _hopSize;

            // Extract and window frame
            var frameData = new T[_fftSize];
            for (int i = 0; i < _fftSize && start + i < input.Length; i++)
            {
                frameData[i] = NumOps.Multiply(input[start + i], _window[i]);
            }

            // Compute STFT
            var (magnitudes, phases) = ComputeSTFT(frameData);

            // Apply neural network enhancement
            var enhancedMagnitudes = EnhanceSpectrum(magnitudes);

            // Compute inverse STFT
            var enhanced = ComputeISTFT(enhancedMagnitudes, phases);

            // Window and overlap-add
            for (int i = 0; i < _fftSize && start + i < output.Length; i++)
            {
                var windowed = NumOps.Multiply(enhanced[i], _window[i]);
                output[start + i] = NumOps.Add(output[start + i], windowed);
            }
        }

        return output;
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
                // Process frame
                var frameData = new T[_fftSize];
                for (int j = 0; j < _fftSize; j++)
                {
                    int idx = (j + _bufferPosition - _hopSize) % _fftSize;
                    frameData[j] = NumOps.Multiply(_inputBuffer[idx], _window[j]);
                }

                var (magnitudes, phases) = ComputeSTFT(frameData);
                var enhancedMagnitudes = EnhanceSpectrum(magnitudes);
                var enhanced = ComputeISTFT(enhancedMagnitudes, phases);

                // Overlap-add to output buffer
                for (int j = 0; j < _fftSize; j++)
                {
                    var windowed = NumOps.Multiply(enhanced[j], _window[j]);
                    _outputBuffer[j] = NumOps.Add(_outputBuffer[j], windowed);
                }

                // Output hop samples
                for (int j = 0; j < _hopSize && outputPos < output.Length; j++)
                {
                    output[outputPos++] = _outputBuffer[j];
                }

                // Shift output buffer
                for (int j = 0; j < _fftSize - _hopSize; j++)
                {
                    _outputBuffer[j] = _outputBuffer[j + _hopSize];
                }
                for (int j = _fftSize - _hopSize; j < _fftSize; j++)
                {
                    _outputBuffer[j] = NumOps.Zero;
                }

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
        // Apply spectral subtraction if noise profile is available
        var processedMagnitudes = magnitudes;
        if (_noiseProfile is not null && _noiseProfile.Length == magnitudes.Length)
        {
            processedMagnitudes = new T[magnitudes.Length];
            for (int i = 0; i < magnitudes.Length; i++)
            {
                // Spectral subtraction: subtract estimated noise magnitude
                double mag = NumOps.ToDouble(magnitudes[i]);
                double noise = NumOps.ToDouble(_noiseProfile[i]);
                double subtracted = Math.Max(0.0, mag - EnhancementStrength * noise);
                processedMagnitudes[i] = NumOps.FromDouble(subtracted);
            }
        }

        // Create input tensor from magnitudes. Write directly to the tensor's
        // flat indexer — ToVector() returns a copy, so writing to it would be
        // discarded and the model would see an all-zero spectrum.
        var input = new Tensor<T>([1, processedMagnitudes.Length, 1]);
        for (int i = 0; i < processedMagnitudes.Length; i++)
        {
            input[i] = processedMagnitudes[i];
        }

        Tensor<T> mask;

        if (IsOnnxMode)
        {
            mask = RunOnnxInference(input);
        }
        else
        {
            mask = Forward(input);
        }

        // Apply mask to magnitudes with enhancement strength
        var enhanced = new T[magnitudes.Length];
        var maskVector = mask.ToVector().ToArray();

        for (int i = 0; i < magnitudes.Length; i++)
        {
            // mask is in range 0-1, where 1 = keep, 0 = remove
            double mag = NumOps.ToDouble(processedMagnitudes[i]);
            double m = i < maskVector.Length ? NumOps.ToDouble(maskVector[i]) : 1.0;

            // Apply enhancement strength to control how aggressively we use the mask
            m = 1.0 - EnhancementStrength * (1.0 - m);

            enhanced[i] = NumOps.FromDouble(mag * m);
        }

        return enhanced;
    }

    #endregion

    #region STFT Implementation

    /// <summary>
    /// Computes Short-Time Fourier Transform using FftSharp library (O(N log N) algorithm).
    /// </summary>
    private (T[] Magnitudes, T[] Phases) ComputeSTFT(T[] frame)
    {
        // Convert frame to double array for FFT
        var frameData = new double[_fftSize];
        for (int i = 0; i < Math.Min(frame.Length, _fftSize); i++)
        {
            frameData[i] = NumOps.ToDouble(frame[i]);
        }

        // Compute FFT using FftSharp (proper O(N log N) algorithm)
        System.Numerics.Complex[] spectrum = FftSharp.FFT.Forward(frameData);

        // Extract magnitude and phase for positive frequencies (N/2+1 bins)
        int numBins = _fftSize / 2 + 1;
        var magnitudes = new T[numBins];
        var phases = new T[numBins];

        for (int k = 0; k < numBins; k++)
        {
            double real = spectrum[k].Real;
            double imag = spectrum[k].Imaginary;
            magnitudes[k] = NumOps.FromDouble(Math.Sqrt(real * real + imag * imag));
            phases[k] = NumOps.FromDouble(Math.Atan2(imag, real));
        }

        return (magnitudes, phases);
    }

    /// <summary>
    /// Computes Inverse Short-Time Fourier Transform using FftSharp library.
    /// </summary>
    private T[] ComputeISTFT(T[] magnitudes, T[] phases)
    {
        // Reconstruct full complex spectrum from magnitude/phase
        // magnitudes/phases contain N/2+1 positive frequency bins
        var spectrum = new System.Numerics.Complex[_fftSize];

        // Fill positive frequencies
        for (int k = 0; k < magnitudes.Length; k++)
        {
            double mag = NumOps.ToDouble(magnitudes[k]);
            double phase = NumOps.ToDouble(phases[k]);
            spectrum[k] = System.Numerics.Complex.FromPolarCoordinates(mag, phase);
        }

        // Fill negative frequencies with conjugate symmetry
        // For real signals: X[N-k] = conj(X[k])
        for (int k = 1; k < magnitudes.Length - 1; k++)
        {
            spectrum[_fftSize - k] = System.Numerics.Complex.Conjugate(spectrum[k]);
        }

        // Compute inverse FFT using FftSharp (modifies spectrum in-place)
        FftSharp.FFT.Inverse(spectrum);

        // Extract real part
        var output = new T[_fftSize];
        for (int n = 0; n < _fftSize; n++)
        {
            output[n] = NumOps.FromDouble(spectrum[n].Real);
        }

        return output;
    }

    /// <summary>
    /// Estimates noise spectrum from noise-only audio.
    /// </summary>
    private T[] EstimateNoiseSpectrum(T[] noiseAudio)
    {
        int numFrames = (noiseAudio.Length - _fftSize) / _hopSize + 1;
        int numBins = _fftSize / 2 + 1;
        var avgMagnitudes = new T[numBins];

        for (int frame = 0; frame < numFrames; frame++)
        {
            int start = frame * _hopSize;
            var frameData = new T[_fftSize];

            for (int i = 0; i < _fftSize && start + i < noiseAudio.Length; i++)
            {
                frameData[i] = NumOps.Multiply(noiseAudio[start + i], _window[i]);
            }

            var (magnitudes, _) = ComputeSTFT(frameData);

            for (int i = 0; i < numBins; i++)
            {
                avgMagnitudes[i] = NumOps.Add(avgMagnitudes[i], magnitudes[i]);
            }
        }

        // Average
        var divisor = NumOps.FromDouble(numFrames);
        for (int i = 0; i < numBins; i++)
        {
            avgMagnitudes[i] = NumOps.Divide(avgMagnitudes[i], divisor);
        }

        return avgMagnitudes;
    }

    #endregion

    #region AudioNeuralNetworkBase Implementation

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        // Convert to spectrogram for neural network input
        var samples = rawAudio.ToVector().ToArray();

        // Apply window
        var windowed = new T[Math.Min(samples.Length, _fftSize)];
        for (int i = 0; i < windowed.Length; i++)
        {
            windowed[i] = NumOps.Multiply(samples[i], _window[i]);
        }

        // Compute magnitude spectrum
        var (magnitudes, _) = ComputeSTFT(windowed);

        // Reshape to [1, freq_bins, 1]. Write directly to the tensor's flat
        // indexer — ToVector() returns a COPY, so writing to it would be
        // discarded and the network would receive an all-zero spectrogram
        // (the cause of the constant sigmoid(0)=0.5 degenerate output).
        var result = new Tensor<T>([1, magnitudes.Length, 1]);
        for (int i = 0; i < magnitudes.Length; i++)
        {
            result[i] = magnitudes[i];
        }

        return result;
    }

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        // Model outputs a mask in range 0-1
        return modelOutput;
    }

    #endregion

    #region Training

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
            throw new InvalidOperationException("Training not supported in ONNX mode.");

        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expected);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (IsOnnxMode)
        {
            throw new InvalidOperationException("Cannot update parameters in ONNX mode.");
        }

        int offset = 0;

        // Update encoder layers
        foreach (var layer in _encoderLayers)
        {
            var layerParams = layer.GetParameters();
            int layerParamCount = layerParams.Length;

            if (offset + layerParamCount <= parameters.Length)
            {
                var newParams = new Vector<T>(layerParamCount);
                for (int i = 0; i < layerParamCount; i++)
                {
                    newParams[i] = parameters[offset + i];
                }
                layer.SetParameters(newParams);
                offset += layerParamCount;
            }
        }

        // Update bottleneck layer
        if (_bottleneckLayer is not null)
        {
            var layerParams = _bottleneckLayer.GetParameters();
            int layerParamCount = layerParams.Length;

            if (offset + layerParamCount <= parameters.Length)
            {
                var newParams = new Vector<T>(layerParamCount);
                for (int i = 0; i < layerParamCount; i++)
                {
                    newParams[i] = parameters[offset + i];
                }
                _bottleneckLayer.SetParameters(newParams);
                offset += layerParamCount;
            }
        }

        // Update decoder layers
        foreach (var layer in _decoderLayers)
        {
            var layerParams = layer.GetParameters();
            int layerParamCount = layerParams.Length;

            if (offset + layerParamCount <= parameters.Length)
            {
                var newParams = new Vector<T>(layerParamCount);
                for (int i = 0; i < layerParamCount; i++)
                {
                    newParams[i] = parameters[offset + i];
                }
                layer.SetParameters(newParams);
                offset += layerParamCount;
            }
        }

        // Update output layer
        if (_outputLayer is not null)
        {
            var layerParams = _outputLayer.GetParameters();
            int layerParamCount = layerParams.Length;

            if (offset + layerParamCount <= parameters.Length)
            {
                var newParams = new Vector<T>(layerParamCount);
                for (int i = 0; i < layerParamCount; i++)
                {
                    newParams[i] = parameters[offset + i];
                }
                _outputLayer.SetParameters(newParams);
            }
        }
    }

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
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
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new NeuralNoiseReducer<T>(
            Architecture,
            SampleRate,
            _fftSize,
            _hopSize,
            NumChannels,
            _numStages,
            _baseFilters,
            _bottleneckDim,
            EnhancementStrength,
            _lossFunction);
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
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(SampleRate);
        writer.Write(_fftSize);
        writer.Write(_hopSize);
        writer.Write(NumChannels);
        writer.Write(_numStages);
        writer.Write(_baseFilters);
        writer.Write(_bottleneckDim);
        writer.Write(EnhancementStrength);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        SampleRate = reader.ReadInt32();
        _fftSize = reader.ReadInt32();
        _hopSize = reader.ReadInt32();
        NumChannels = reader.ReadInt32();
        _numStages = reader.ReadInt32();
        _baseFilters = reader.ReadInt32();
        _bottleneckDim = reader.ReadInt32();
        EnhancementStrength = reader.ReadDouble();

        // Rebuild streaming buffers after FFT/hop size may have changed
        InitializeStreamingBuffers();

        // Reinitialize layers if needed for native mode
        if (_useNativeMode && (_encoderLayers is null || _encoderLayers.Count == 0))
        {
            InitializeLayers();
        }
    }

    #endregion
}

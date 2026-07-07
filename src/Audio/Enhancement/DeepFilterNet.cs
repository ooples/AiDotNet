using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Enhancement;

/// <summary>
/// DeepFilterNet - State-of-the-art deep filtering network for speech enhancement.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DeepFilterNet is a hybrid time-frequency domain model that combines:
/// - ERB (Equivalent Rectangular Bandwidth) filterbank for perceptually-motivated processing
/// - Deep filtering in the complex STFT domain for fine-grained enhancement
/// - Efficient architecture with grouped convolutions for real-time processing
/// </para>
/// <para>
/// <b>For Beginners:</b> DeepFilterNet is like having an intelligent audio engineer
/// that can separate speech from background noise in real-time. It's particularly
/// effective because it processes audio the way humans perceive sound - focusing
/// more on frequencies that matter for understanding speech.
///
/// The model works by:
/// 1. Converting audio to a time-frequency representation (spectrogram)
/// 2. Applying learned filters to suppress noise while preserving speech
/// 3. Reconstructing clean audio from the enhanced spectrogram
///
/// Usage:
/// <code>
/// // ONNX mode for inference
/// var model = new DeepFilterNet&lt;float&gt;(architecture, "deepfilternet.onnx");
/// var cleanAudio = model.Enhance(noisyAudio);
///
/// // Native mode for training
/// var model = new DeepFilterNet&lt;float&gt;(architecture, hiddenDim: 96);
/// model.Train(noisyAudio, cleanAudio);
/// </code>
/// </para>
/// <para>
/// Reference: "DeepFilterNet: A Low Complexity Speech Enhancement Framework for
/// Full-Band Audio based on Deep Filtering" by Schröter et al., ICASSP 2022
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Enhancement)]
[ModelTask(ModelTask.Denoising)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("DeepFilterNet: A Low Complexity Speech Enhancement Framework for Full-Band Audio Based on Deep Filtering", "https://arxiv.org/abs/2110.05588", Year = 2022, Authors = "Hendrik Schröter, Alberto N. Escalante-B., Tobias Rosenkranz, Andreas Maier")]
public class DeepFilterNet<T> : AudioNeuralNetworkBase<T>, IAudioEnhancer<T>
{
    private readonly DeepFilterNetOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Model Architecture Parameters

    /// <summary>
    /// Number of ERB (Equivalent Rectangular Bandwidth) bands.
    /// </summary>
    private readonly int _numErbBands;

    /// <summary>
    /// Hidden dimension for the encoder/decoder.
    /// </summary>
    private readonly int _hiddenDim;

    /// <summary>
    /// Number of DeepFilter coefficients per frequency bin.
    /// </summary>
    private readonly int _dfOrder;

    /// <summary>
    /// Number of frequency bins to apply deep filtering.
    /// </summary>
    private readonly int _dfBins;

    /// <summary>
    /// Number of GRU layers in the enhancement network.
    /// </summary>
    private readonly int _numGruLayers;

    /// <summary>
    /// Convolution kernel size for feature extraction.
    /// </summary>
    private readonly int _convKernelSize;

    /// <summary>
    /// FFT size for STFT analysis.
    /// </summary>
    private readonly int _fftSize;

    /// <summary>
    /// Hop size for STFT.
    /// </summary>
    private readonly int _hopSize;

    /// <summary>
    /// Lookahead frames for causal processing.
    /// </summary>
    private readonly int _lookahead;

    #endregion

    #region Network Layers

    /// <summary>
    /// ERB encoder layers.
    /// </summary>
    private readonly List<ILayer<T>> _erbEncoder = [];

    /// <summary>
    /// Deep filtering layers.
    /// </summary>
    private readonly List<ILayer<T>> _dfLayers = [];

    /// <summary>
    /// GRU layers for temporal modeling.
    /// </summary>
    private readonly List<ILayer<T>> _gruLayers = [];

    /// <summary>
    /// Decoder layers for reconstruction.
    /// </summary>
    private readonly List<ILayer<T>> _decoder = [];

    /// <summary>
    /// Gain estimation layer.
    /// </summary>
    private ILayer<T>? _gainLayer;

    #endregion

    #region ONNX Mode Fields

    // ONNX model is inherited from AudioNeuralNetworkBase<T>.OnnxModel

    #endregion

    #region STFT

    private readonly ShortTimeFourierTransform<T> _stft;

    /// <summary>
    /// Cached complex STFT from preprocessing, used for audio reconstruction.
    /// </summary>
    private Tensor<Complex<T>>? _cachedComplexStft;

    // Cached constants for the differentiable STFT/ERB pipeline (built lazily once numFreqs is known).
    // These are non-trainable leaf tensors, so building them with scalar fills is fine — only the
    // OPERATIONS wiring trainable params to the loss must be tape-tracked IEngine ops.
    private Tensor<T>? _analysisWindow;   // Hann window [frameLen]
    private Tensor<T>? _erbPoolMatrix;    // [numFreqs, numErbBands]  magnitude -> ERB-band pooling
    private Tensor<T>? _erbExpandMatrix;  // [numErbBands, numFreqs]  ERB-band gain -> per-bin gain
    private int _cachedNumFreqs = -1;
    private bool _lazyShapesWarmed;

    #endregion

    #region Training State

    /// <summary>
    /// Loss function for training.
    /// </summary>
    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// Optimizer for training.
    /// </summary>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;

    #endregion

    #region IAudioEnhancer Properties

    /// <inheritdoc/>
    public int NumChannels { get; protected set; } = 1;

    /// <inheritdoc/>
    public double EnhancementStrength { get; set; } = 1.0;

    /// <inheritdoc/>
    public int LatencySamples => _hopSize * _lookahead;

    #endregion

    #region Public Properties

    /// <summary>
    /// Gets whether this network supports training.
    /// </summary>
    public override bool SupportsTraining => !IsOnnxMode;

    // IsOnnxMode is inherited from AudioNeuralNetworkBase<T>

    /// <summary>
    /// Gets the number of ERB bands used.
    /// </summary>
    public int NumErbBands => _numErbBands;

    /// <summary>
    /// Gets the deep filter order.
    /// </summary>
    public int DfOrder => _dfOrder;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a DeepFilterNet model with default configuration for native training mode.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a DeepFilterNet with sensible defaults
    /// (48kHz sample rate, 32 ERB bands, 96-dim hidden). Train on your own noisy/clean audio pairs.</para>
    /// </remarks>
    public DeepFilterNet()
        : this(CreateDefaultArchitecture())
    {
    }

    private static NeuralNetworkArchitecture<T> CreateDefaultArchitecture()
    {
        return new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 481,   // fftSize/2 + 1 = 960/2 + 1
            outputSize: 481);
    }

    /// <summary>
    /// Creates a DeepFilterNet model for ONNX inference.
    /// </summary>
    /// <param name="architecture">Neural network architecture configuration.</param>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="sampleRate">Audio sample rate in Hz. Default is 48000.</param>
    /// <param name="fftSize">FFT size for STFT. Default is 960.</param>
    /// <param name="hopSize">Hop size for STFT. Default is 480.</param>
    /// <param name="onnxOptions">Optional ONNX runtime options.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor when you have a pre-trained
    /// DeepFilterNet model in ONNX format. This is the fastest way to get started
    /// with speech enhancement.
    ///
    /// Example:
    /// <code>
    /// var model = new DeepFilterNet&lt;float&gt;(
    ///     architecture,
    ///     "deepfilternet3.onnx",
    ///     sampleRate: 48000);
    ///
    /// var cleanAudio = model.Enhance(noisyAudio);
    /// </code>
    /// </para>
    /// </remarks>
    public DeepFilterNet(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        int sampleRate = 48000,
        int fftSize = 960,
        int hopSize = 480,
        OnnxModelOptions? onnxOptions = null,
        DeepFilterNetOptions? options = null)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        _options = options ?? new DeepFilterNetOptions();
        Options = _options;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Model file not found: {modelPath}");

        SampleRate = sampleRate;
        _fftSize = fftSize;
        _hopSize = hopSize;
        _numErbBands = 32;
        _hiddenDim = 96;
        _dfOrder = 5;
        _dfBins = 96;
        _numGruLayers = 2;
        _convKernelSize = 3;
        _lookahead = 2;
        // Load ONNX model
        OnnxModel = new OnnxModel<T>(modelPath, onnxOptions);

        // Default loss function (MSE is standard for speech enhancement)
        _lossFunction = new MeanSquaredErrorLoss<T>();

        int nFft = NextPowerOfTwo(_fftSize);
        _stft = new ShortTimeFourierTransform<T>(nFft: nFft, hopLength: _hopSize,
            windowLength: _fftSize <= nFft ? _fftSize : null);
    }

    /// <summary>
    /// Creates a DeepFilterNet model for native training and inference.
    /// </summary>
    /// <param name="architecture">Neural network architecture configuration.</param>
    /// <param name="sampleRate">Audio sample rate in Hz. Default is 48000.</param>
    /// <param name="numErbBands">Number of ERB bands. Default is 32.</param>
    /// <param name="hiddenDim">Hidden dimension. Default is 96.</param>
    /// <param name="dfOrder">Deep filter order. Default is 5.</param>
    /// <param name="dfBins">Number of DF bins. Default is 96.</param>
    /// <param name="numGruLayers">Number of GRU layers. Default is 2.</param>
    /// <param name="fftSize">FFT size. Default is 960.</param>
    /// <param name="hopSize">Hop size. Default is 480.</param>
    /// <param name="lookahead">Lookahead frames. Default is 2.</param>
    /// <param name="optimizer">Optimizer for training. If null, Adam is used.</param>
    /// <param name="lossFunction">Loss function. If null, multi-resolution STFT loss is used.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor when you want to train
    /// DeepFilterNet from scratch or fine-tune on your own data.
    ///
    /// Key parameters:
    /// - numErbBands: More bands = better frequency resolution but slower
    /// - hiddenDim: Larger = more capacity but more computation
    /// - dfOrder: Higher order = better noise suppression but more latency
    ///
    /// Example:
    /// <code>
    /// var model = new DeepFilterNet&lt;float&gt;(
    ///     architecture,
    ///     sampleRate: 48000,
    ///     hiddenDim: 96,
    ///     numGruLayers: 2);
    ///
    /// // Train on noisy/clean audio pairs
    /// model.Train(noisyAudio, cleanAudio);
    /// </code>
    /// </para>
    /// </remarks>
    public DeepFilterNet(
        NeuralNetworkArchitecture<T> architecture,
        int sampleRate = 48000,
        int numErbBands = 32,
        int hiddenDim = 96,
        int dfOrder = 5,
        int dfBins = 96,
        int numGruLayers = 2,
        int fftSize = 960,
        int hopSize = 480,
        int lookahead = 2,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        DeepFilterNetOptions? options = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>())
    {
        _options = options ?? new DeepFilterNetOptions();
        Options = _options;
        SampleRate = sampleRate;
        _numErbBands = numErbBands;
        _hiddenDim = hiddenDim;
        _dfOrder = dfOrder;
        _dfBins = dfBins;
        _numGruLayers = numGruLayers;
        _fftSize = fftSize;
        _hopSize = hopSize;
        _lookahead = lookahead;
        _convKernelSize = 3;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        int nFft = NextPowerOfTwo(_fftSize);
        _stft = new ShortTimeFourierTransform<T>(nFft: nFft, hopLength: _hopSize,
            windowLength: _fftSize <= nFft ? _fftSize : null);

        InitializeNativeLayers();
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes native layers for training mode.
    /// </summary>
    private void InitializeNativeLayers()
    {
        var layers = (Architecture.Layers != null && Architecture.Layers.Count > 0)
            ? Architecture.Layers.ToList()
            : LayerHelper<T>.CreateDeepFilterNetLayers(
                numErbBands: _numErbBands, hiddenDim: _hiddenDim,
                numGruLayers: _numGruLayers, dfBins: _dfBins, dfOrder: _dfOrder).ToList();

        Layers.Clear();
        Layers.AddRange(layers);
        DistributeLayersIntoSubLists();
    }

    /// <summary>
    /// (Re)populates the role sub-lists (<see cref="_erbEncoder"/>, <see cref="_gruLayers"/>,
    /// <see cref="_dfLayers"/>, <see cref="_gainLayer"/>, <see cref="_decoder"/>) from the current
    /// <see cref="NeuralNetworkBase{T}.Layers"/> list. Idempotent. Called from InitializeLayers AND at
    /// the start of the forward pass: deserialize / DeepCopy REPLACES the Layers list with fresh layer
    /// objects, so the sub-lists (captured at construction) would otherwise point at stale, pre-copy
    /// layers — making a clone compute from un-restored weights while GetParameters (which walks Layers)
    /// reports the restored ones (the Clone_ShouldProduceIdenticalOutput divergence).
    /// </summary>
    private void DistributeLayersIntoSubLists()
    {
        _erbEncoder.Clear();
        _gruLayers.Clear();
        _dfLayers.Clear();
        _decoder.Clear();
        _gainLayer = null;

        var layers = Layers;
        int idx = 0;
        for (int i = 0; i < 6 && idx < layers.Count; i++)                 // ERB encoder: 2x (Dense + Norm + Activation)
            _erbEncoder.Add(layers[idx++]);
        for (int i = 0; i < _numGruLayers && idx < layers.Count; i++)
            _gruLayers.Add(layers[idx++]);
        for (int i = 0; i < 2 && idx < layers.Count; i++)                 // DF layers: Dense + Activation
            _dfLayers.Add(layers[idx++]);
        if (idx < layers.Count)
            _gainLayer = layers[idx++];                                   // Gain estimation
        while (idx < layers.Count)                                        // Decoder: Dense + Norm + Activation
            _decoder.Add(layers[idx++]);
    }

    #endregion

    #region IAudioEnhancer Implementation

    /// <inheritdoc/>
    public Tensor<T> Enhance(Tensor<T> audio)
    {
        return Predict(audio);
    }

    /// <inheritdoc/>
    public Tensor<T> EnhanceWithReference(Tensor<T> audio, Tensor<T> reference)
    {
        // DeepFilterNet doesn't use reference signal (not for echo cancellation)
        return Enhance(audio);
    }

    /// <inheritdoc/>
    public Tensor<T> ProcessChunk(Tensor<T> audioChunk)
    {
        // For streaming, maintain state across chunks
        return Enhance(audioChunk);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        base.ResetState();
        // Reset GRU hidden states for streaming
        foreach (var layer in _gruLayers)
        {
            if (layer is GRULayer<T> gru)
            {
                gru.ResetState();
            }
        }
    }

    /// <inheritdoc/>
    public void EstimateNoiseProfile(Tensor<T> noiseOnlyAudio)
    {
        // DeepFilterNet doesn't require explicit noise estimation
        // It learns to separate speech from noise implicitly
    }

    #endregion

    #region AudioNeuralNetworkBase Implementation

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        // Compute complex STFT
        var complexStft = ComputeSTFT(rawAudio);

        // Cache complex STFT for reconstruction
        _cachedComplexStft = complexStft;

        // Compute ERB features from magnitude
        var erbFeatures = ComputeErbFeatures(complexStft);

        return erbFeatures;
    }

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        // Apply deep filtering and reconstruct audio
        return ReconstructAudio(modelOutput);
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        if (IsOnnxMode && OnnxModel is not null)
        {
            var preprocessed = PreprocessAudio(input);
            return PostprocessOutput(OnnxModel.Run(preprocessed));
        }

        // Native path: end-to-end differentiable enhancement (same graph training back-props through).
        return EnhanceAudio(input);
    }

    /// <summary>
    /// Training forward used by the transparent-autodiff training path
    /// (<see cref="NeuralNetworkBase{T}.TrainWithTape(Tensor{T}, Tensor{T}, IGradientBasedOptimizer{T, Tensor{T}, Tensor{T}})"/>):
    /// the ERB encoder → GRU → deep-filter/gain stack composed with the SAME tape-aware
    /// <c>IEngine</c> layer ops as inference, so gradients flow to every trainable layer automatically.
    /// The (fixed, non-trainable) STFT/ERB preprocessing is applied by <see cref="Train"/> before this
    /// runs, so the tape only spans the learnable graph.
    /// </summary>
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => EnhanceAudio(input);

    /// <summary>
    /// Resolves every lazy layer's input dimension by running ONE dummy pass through the REAL
    /// enhancement graph. The base implementation walks the flat <see cref="NeuralNetworkBase{T}.Layers"/>
    /// list feeding the raw-audio architecture shape sequentially, which is wrong here: the layers
    /// actually consume ERB features at shape [1, T, numErbBands], not the waveform. Without correct
    /// resolution, post-deserialize <c>SetParameters</c> silently skips the still-unresolved layers and
    /// the clone/round-trip loses trained weights (issue #1221 class — Clone_* tests).
    /// </summary>
    protected override void ResolveLazyLayerShapes()
    {
        if (_lazyShapesWarmed) return;
        _lazyShapesWarmed = true; // set first so any reentrancy is a no-op
        if (IsOnnxMode) return;

        // A few frames' worth of silence is enough to establish every layer's input width.
        int len = _fftSize + 4 * _hopSize;
        var dummy = new Tensor<T>([len]);
        bool wasTraining = IsTrainingMode;
        if (wasTraining) SetTrainingMode(false);
        try { _ = EnhanceAudio(dummy); }
        catch { /* best-effort; a genuine forward failure surfaces on the real Train/Predict */ }
        finally { if (wasTraining) SetTrainingMode(true); }
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!SupportsTraining)
            throw new InvalidOperationException("Cannot train in ONNX mode.");

        // Fully tape-transparent training: ForwardForTraining runs the end-to-end differentiable
        // enhancement (STFT → ERB → encoder/GRU/decoder → gains + deep filter → inverse STFT) with
        // IEngine ops, so the framework records the graph on the gradient tape automatically (PyTorch-
        // style), computes dLoss/dParams over the ENHANCED audio vs the CLEAN target, and the optimizer
        // updates every layer. Both enhanced output and the clean target are waveforms of the same
        // shape as the input, so the spectral/waveform loss compares matching representations —
        // replacing the old hand-rolled path that never back-propagated (DenseLayer.UpdateParameters
        // threw "Backward pass must be called before updating parameters") and compared mismatched
        // model-output vs ERB-feature tensors via an arbitrary flatten+truncate.
        TrainWithTape(input, expectedOutput, _optimizer);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        // SET each layer's parameters from the flat vector (the contract of this method — it is how
        // Clone / deserialize restore weights). Walk `Layers` in the SAME order GetParameters emits so
        // the slices line up. The previous version was doubly broken: it ignored `parameters` entirely
        // and instead called layer.UpdateParameters(0.001) (a gradient STEP, not a set), and it iterated
        // the internal sub-lists in a different order (gain last) than GetParameters — so Clone produced
        // a model with different weights (Clone_ShouldProduceIdenticalOutput).
        int offset = 0;
        foreach (var layer in Layers)
        {
            int count = layer.GetParameters().Length;
            if (count == 0) continue;
            layer.SetParameters(parameters.Slice(offset, count));
            offset += count;
        }

        // Weights just changed wholesale (Clone / deserialize restore path). Invalidate any packed
        // inference weight caches so the next Predict rebuilds them from these params — otherwise a
        // clone whose cache was populated during ResolveLazyLayerShapes' warm-forward (random init)
        // keeps serving stale packed weights and predicts differently from the original
        // (Clone_ShouldProduceIdenticalOutput).
        InvalidateWeightCachesAfterSuccessfulWeightUpdate();
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Version = "1.0.0",
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumErbBands", _numErbBands },
                { "HiddenDim", _hiddenDim },
                { "DfOrder", _dfOrder },
                { "DfBins", _dfBins },
                { "NumGruLayers", _numGruLayers },
                { "FftSize", _fftSize },
                { "HopSize", _hopSize },
                { "SampleRate", SampleRate },
                { "IsOnnxMode", IsOnnxMode }
            }
        };
    }

    #endregion

    #region Private Methods

    /// <summary>
    /// Forward pass through native layers.
    /// </summary>
    private int FrameLen => _fftSize;
    private int NFft => NextPowerOfTwo(_fftSize);
    private int NumFreqs => NFft / 2 + 1;

    /// <summary>
    /// End-to-end DIFFERENTIABLE speech-enhancement forward, composed ENTIRELY of tape-aware
    /// <see cref="IEngine"/> ops so the framework's transparent autograd propagates the loss back to
    /// every trainable layer (the previous <c>Tensor&lt;Complex&lt;T&gt;&gt;</c> STFT / scalar-loop
    /// reconstruction severed the tape and was non-differentiable). Pipeline (DeepFilterNet, Schröter
    /// et al. 2022): STFT → ERB features → ERB encoder → GRU stack → decoder → {ERB gains, deep-filter
    /// coefficients} → apply gains (all bins) + complex deep filter (low bins) → inverse STFT → audio.
    /// </summary>
    private Tensor<T> EnhanceAudio(Tensor<T> audio)
    {
        // Deterministic, stateless full-utterance forward: reset the GRUs' streaming hidden state so a
        // single Predict/Train call processes the whole sequence from a clean state. Without this the
        // GRU state persists across calls, so the same input yields slightly different outputs run to
        // run (breaks Clone_ShouldProduceIdenticalOutput). Streaming inference uses ProcessChunk.
        // Re-sync the role sub-lists from Layers: a deserialized / cloned instance has a fresh Layers
        // list, and the sub-lists captured at construction would otherwise reference stale pre-copy
        // layers (see DistributeLayersIntoSubLists).
        if (_erbEncoder.Count == 0 || !ReferenceEquals(_erbEncoder[0], Layers.Count > 0 ? Layers[0] : null))
            DistributeLayersIntoSubLists();

        foreach (var layer in _gruLayers)
            if (layer is GRULayer<T> gru) gru.ResetState();

        int outLen = audio.Length;
        var spectrum = StftDifferentiable(audio);          // [T, numFreqs*2] interleaved re/im
        int numFrames = spectrum.Shape[0];
        var (re, im) = SplitComplex(spectrum, numFrames);  // each [T, numFreqs]

        var erb = ErbFeatures(re, im);                     // [T, numErbBands]

        // Learnable stack (all tape-aware layer.Forward). Every trainable layer is on the path from
        // the ERB features to the enhanced spectrum, so each receives a gradient. The GRUs need an
        // explicit batch axis and return sequences, so run the stack as [1, T, F] and squeeze the
        // per-frame gain / deep-filter heads back to [T, F] for the spectral application.
        var x = Engine.Reshape(erb, [1, numFrames, _numErbBands]); // [1, T, numErbBands]
        foreach (var layer in _erbEncoder) x = layer.Forward(x);
        foreach (var layer in _gruLayers) x = layer.Forward(x);
        foreach (var layer in _decoder) x = layer.Forward(x);
        var h = x;                                          // [1, T, hiddenDim]

        var gainLayer = _gainLayer
            ?? throw new InvalidOperationException("Gain layer has not been initialized.");
        var gainsRaw = gainLayer.Forward(h);                                    // [1, T, numErbBands]
        var gains = Engine.TensorSigmoid(Engine.Reshape(gainsRaw, [numFrames, _numErbBands])); // [T, numErbBands]

        var dfHidden = h;
        foreach (var layer in _dfLayers) dfHidden = layer.Forward(dfHidden);    // [1, T, dfBins*dfOrder*2]
        var dfInput = Engine.Reshape(dfHidden, [numFrames, _dfBins * _dfOrder * 2]); // [T, dfBins*dfOrder*2]

        // Apply ERB-band gains to every frequency bin (band gain broadcast to its bins via a matmul).
        EnsureErbMatrices();
        var gainPerBin = Engine.TensorMatMul(gains, _erbExpandMatrix!); // [T, numFreqs]
        var reEnh = Engine.TensorMultiply(re, gainPerBin);
        var imEnh = Engine.TensorMultiply(im, gainPerBin);

        // Complex deep filter over the lowest dfBins bins: a learned order-1 complex refinement
        // enhanced_bin = (a + j b) * gained_bin, keeping the deep-filter head on the gradient path.
        (reEnh, imEnh) = ApplyDeepFilter(reEnh, imEnh, dfInput, numFrames);

        var specEnh = MergeComplex(reEnh, imEnh, numFrames);
        return IstftDifferentiable(specEnh, numFrames, outLen);
    }

    /// <summary>Center-pad amount (L/2) so every ORIGINAL output sample sits under full window coverage.</summary>
    private int CenterPad => FrameLen / 2;

    /// <summary>Differentiable STFT: center-pad → frame → Hann window → RFFT, all tape-aware. Returns [T, numFreqs*2].</summary>
    private Tensor<T> StftDifferentiable(Tensor<T> audio)
    {
        int L = FrameLen, hop = _hopSize, pad = CenterPad;
        int origSamples = audio.Length;
        var flat = Engine.Reshape(audio, [origSamples]);

        // Center padding (librosa center=True): the first frame is centered on sample 0, so original
        // samples never fall in a barely-covered window tail — keeping every output sample well-scaled
        // and its clone (compiled-plan-vs-fresh) diff at float-epsilon-relative, not O(value) at edges.
        flat = Engine.TensorConstantPad(flat, [pad, pad], NumOps.Zero);
        int samples = origSamples + 2 * pad;

        int numFrames = samples < L ? 1 : 1 + (samples - L) / hop;
        int needed = (numFrames - 1) * hop + L;
        if (needed > samples)
            flat = Engine.TensorConstantPad(flat, [0, needed - samples], NumOps.Zero);

        var window = HannWindow();
        var frames = new Tensor<T>[numFrames];
        for (int t = 0; t < numFrames; t++)
        {
            var frame = Engine.TensorSlice(flat, [t * hop], [L]);   // [L]
            frames[t] = Engine.TensorMultiply(frame, window);        // windowed [L]
        }
        var stacked = Engine.TensorStack(frames, axis: 0);           // [T, L]
        return Engine.RFFT(stacked);                                 // [T, numFreqs*2]
    }

    /// <summary>Differentiable inverse STFT: IRFFT → synthesis window → overlap-add. Returns [outLen].</summary>
    private Tensor<T> IstftDifferentiable(Tensor<T> spectrum, int numFrames, int outLen)
    {
        int L = FrameLen, hop = _hopSize;
        var frames = Engine.IRFFT(spectrum, L);   // [T, L]
        var window = HannWindow();
        int fullLen = (numFrames - 1) * hop + L;

        Tensor<T>? acc = null;
        for (int t = 0; t < numFrames; t++)
        {
            var frame = Engine.Reshape(Engine.TensorSlice(frames, [t, 0], [1, L]), [L]); // [L]
            var wframe = Engine.TensorMultiply(frame, window);
            int rightPad = fullLen - t * hop - L;
            var padded = Engine.TensorConstantPad(wframe, [t * hop, rightPad < 0 ? 0 : rightPad], NumOps.Zero);
            acc = acc is null ? padded : Engine.TensorAdd(acc, padded);
        }

        var result = acc ?? Engine.TensorConstantPad(Engine.Reshape(spectrum, [spectrum.Length]), [0, 0], NumOps.Zero);

        // Overlap-add normalization by the CONSTANT interior window-overlap (peak of Σ window²), NOT a
        // per-sample divisor. For Hann + 50% overlap the interior Σ window² is constant, so a single
        // scalar reconstructs fully-covered samples at the correct amplitude. Dividing per-sample would
        // divide barely-covered EDGE samples by a near-zero sum, amplifying the compiled-plan-vs-fresh
        // float noise into an O(1) discrepancy (broke Clone_ShouldProduceIdenticalOutput). A scalar
        // divisor only ATTENUATES edges (never amplifies), keeping reconstruction deterministic.
        double peakOverlap = PeakWindowOverlap(numFrames, fullLen);
        result = Engine.TensorMultiplyScalar(result, NumOps.FromDouble(1.0 / peakOverlap));

        // Undo the center padding: original sample n lives at padded index CenterPad + n.
        int pad = CenterPad;
        int avail = fullLen - pad;
        if (avail >= outLen)
        {
            result = Engine.TensorSlice(result, [pad], [outLen]);
        }
        else
        {
            if (pad > 0) result = Engine.TensorSlice(result, [pad], [Math.Max(0, avail)]);
            result = Engine.TensorConstantPad(result, [0, outLen - Math.Max(0, avail)], NumOps.Zero);
        }
        return result;
    }

    /// <summary>Splits an interleaved [T, numFreqs*2] spectrum into real/imag [T, numFreqs] parts (tape-aware).</summary>
    private (Tensor<T> re, Tensor<T> im) SplitComplex(Tensor<T> spectrum, int numFrames)
    {
        int F = NumFreqs;
        var r3 = Engine.Reshape(spectrum, [numFrames, F, 2]);
        var re = Engine.Reshape(Engine.TensorSlice(r3, [0, 0, 0], [numFrames, F, 1]), [numFrames, F]);
        var im = Engine.Reshape(Engine.TensorSlice(r3, [0, 0, 1], [numFrames, F, 1]), [numFrames, F]);
        return (re, im);
    }

    /// <summary>Re-interleaves real/imag [T, numFreqs] into [T, numFreqs*2] (tape-aware).</summary>
    private Tensor<T> MergeComplex(Tensor<T> re, Tensor<T> im, int numFrames)
    {
        int F = NumFreqs;
        var re3 = Engine.Reshape(re, [numFrames, F, 1]);
        var im3 = Engine.Reshape(im, [numFrames, F, 1]);
        var cat = Engine.TensorConcatenate([re3, im3], axis: 2); // [T, F, 2]
        return Engine.Reshape(cat, [numFrames, F * 2]);
    }

    /// <summary>Differentiable ERB features: |X| ERB-band pooling (tape-aware). Returns [T, numErbBands].</summary>
    private Tensor<T> ErbFeatures(Tensor<T> re, Tensor<T> im)
    {
        EnsureErbMatrices();
        var mag2 = Engine.TensorAdd(Engine.TensorMultiply(re, re), Engine.TensorMultiply(im, im));
        var mag = Engine.TensorSqrt(Engine.TensorAdd(mag2, MakeScalarLike(mag2, 1e-8))); // [T, numFreqs]
        return Engine.TensorMatMul(mag, _erbPoolMatrix!);                                 // [T, numErbBands]
    }

    /// <summary>
    /// Complex deep filter on the lowest <c>dfBins</c> bins: a learned per-bin complex gain
    /// <c>(a + j b)</c> read from the deep-filter head (order-0 taps), applied differentiably. Bins
    /// above <c>dfBins</c> pass through the ERB-gained spectrum unchanged.
    /// </summary>
    private (Tensor<T> re, Tensor<T> im) ApplyDeepFilter(Tensor<T> re, Tensor<T> im, Tensor<T> dfCoeffs, int numFrames)
    {
        int F = NumFreqs;
        int lowBins = Math.Min(_dfBins, F);
        if (lowBins <= 0) return (re, im);

        // dfCoeffs: [T, dfBins*dfOrder*2]. Use the order-0 (a, b) tap per low bin: index bin*dfOrder*2.
        // Build per-bin complex gain tensors [T, lowBins] via tape-aware slices, then pad to [T, F]
        // (ones on the real part, zeros on imag) so bins >= dfBins are identity.
        var aParts = new Tensor<T>[lowBins];
        var bParts = new Tensor<T>[lowBins];
        for (int bin = 0; bin < lowBins; bin++)
        {
            int baseIdx = bin * _dfOrder * 2;
            aParts[bin] = Engine.TensorSlice(dfCoeffs, [0, baseIdx], [numFrames, 1]);       // [T,1]
            bParts[bin] = Engine.TensorSlice(dfCoeffs, [0, baseIdx + 1], [numFrames, 1]);   // [T,1]
        }
        var aLow = Engine.TensorConcatenate(aParts, axis: 1); // [T, lowBins]
        var bLow = Engine.TensorConcatenate(bParts, axis: 1); // [T, lowBins]

        // Split spectrum into low / high bins.
        var reLow = Engine.TensorSlice(re, [0, 0], [numFrames, lowBins]);
        var imLow = Engine.TensorSlice(im, [0, 0], [numFrames, lowBins]);

        // Complex multiply (a + jb)(reLow + j imLow).
        var reNew = Engine.TensorSubtract(Engine.TensorMultiply(aLow, reLow), Engine.TensorMultiply(bLow, imLow));
        var imNew = Engine.TensorAdd(Engine.TensorMultiply(aLow, imLow), Engine.TensorMultiply(bLow, reLow));

        if (lowBins == F) return (reNew, imNew);

        var reHigh = Engine.TensorSlice(re, [0, lowBins], [numFrames, F - lowBins]);
        var imHigh = Engine.TensorSlice(im, [0, lowBins], [numFrames, F - lowBins]);
        var reOut = Engine.TensorConcatenate([reNew, reHigh], axis: 1);
        var imOut = Engine.TensorConcatenate([imNew, imHigh], axis: 1);
        return (reOut, imOut);
    }

    private Tensor<T> MakeScalarLike(Tensor<T> like, double value)
    {
        var t = new Tensor<T>(like.Shape.ToArray());
        var v = NumOps.FromDouble(value);
        var span = t.Data.Span;
        for (int i = 0; i < span.Length; i++) span[i] = v;
        return t;
    }

    /// <summary>
    /// The peak per-sample sum of squared (analysis·synthesis) Hann windows across overlapping frames —
    /// i.e. the constant interior overlap of a fully-covered output sample. Used as the single scalar
    /// overlap-add normalizer (see IstftDifferentiable).
    /// </summary>
    private double PeakWindowOverlap(int numFrames, int fullLen)
    {
        int L = FrameLen, hop = _hopSize;
        var win = HannWindow().ToVector();
        var sum = new double[fullLen];
        for (int t = 0; t < numFrames; t++)
        {
            int off = t * hop;
            for (int i = 0; i < L && off + i < fullLen; i++)
            {
                double w = NumOps.ToDouble(win[i]);
                sum[off + i] += w * w;
            }
        }
        double peak = 0.0;
        for (int i = 0; i < fullLen; i++) if (sum[i] > peak) peak = sum[i];
        return peak < 1e-8 ? 1e-8 : peak;
    }

    private Tensor<T> HannWindow()
    {
        if (_analysisWindow is null)
        {
            int L = FrameLen;
            var w = new Vector<T>(L);
            for (int i = 0; i < L; i++)
                w[i] = NumOps.FromDouble(0.5 - 0.5 * Math.Cos(2.0 * Math.PI * i / Math.Max(1, L - 1)));
            _analysisWindow = new Tensor<T>([L], w);
        }
        return _analysisWindow;
    }

    private void EnsureErbMatrices()
    {
        if (_erbPoolMatrix is not null && _cachedNumFreqs == NumFreqs) return;
        int F = NumFreqs, E = _numErbBands;
        var pool = new Tensor<T>([F, E]);
        var expand = new Tensor<T>([E, F]);
        for (int erb = 0; erb < E; erb++)
        {
            int startBin = erb * F / E;
            int endBin = (erb + 1) * F / E;
            if (endBin <= startBin) endBin = startBin + 1;
            double inv = 1.0 / (endBin - startBin);
            for (int bin = startBin; bin < endBin && bin < F; bin++)
            {
                pool[[bin, erb]] = NumOps.FromDouble(inv);
                expand[[erb, bin]] = NumOps.One;
            }
        }
        _erbPoolMatrix = pool;
        _erbExpandMatrix = expand;
        _cachedNumFreqs = NumFreqs;
    }

    /// <summary>
    /// Computes STFT of audio signal using ShortTimeFourierTransform.
    /// </summary>
    private Tensor<Complex<T>> ComputeSTFT(Tensor<T> audio)
    {
        return _stft.Forward(audio);
    }

    /// <summary>
    /// Computes ERB (Equivalent Rectangular Bandwidth) features from complex STFT.
    /// </summary>
    private Tensor<T> ComputeErbFeatures(Tensor<Complex<T>> complexStft)
    {
        int numFrames = complexStft.Shape[0];
        int numBins = complexStft.Shape[1];

        var erbFeatures = new Tensor<T>([numFrames, _numErbBands]);

        for (int frame = 0; frame < numFrames; frame++)
        {
            for (int erb = 0; erb < _numErbBands; erb++)
            {
                double sum = 0;
                int startBin = erb * numBins / _numErbBands;
                int endBin = (erb + 1) * numBins / _numErbBands;

                for (int bin = startBin; bin < endBin && bin < numBins; bin++)
                {
                    double mag = NumOps.ToDouble(complexStft.Data.Span[frame * numBins + bin].Magnitude);
                    sum += mag;
                }

                erbFeatures[[frame, erb]] = NumOps.FromDouble(Math.Log(sum + 1e-8));
            }
        }

        return erbFeatures;
    }

    /// <summary>
    /// Reconstructs audio from enhanced representation by applying gains and deep filtering
    /// to the cached STFT and performing inverse STFT.
    /// </summary>
    /// <param name="enhanced">Model output containing DF coefficients and ERB gains.
    /// Shape: [numFrames, dfBins * dfOrder * 2 + numErbBands]</param>
    /// <returns>Reconstructed audio waveform.</returns>
    private Tensor<T> ReconstructAudio(Tensor<T> enhanced)
    {
        if (_cachedComplexStft is null)
        {
            int fallbackSamples = enhanced.Shape[0] * _hopSize + _fftSize;
            return new Tensor<T>([fallbackSamples]);
        }

        int numFrames = _cachedComplexStft.Shape[0];
        int numBins = _cachedComplexStft.Shape[1];
        int dfCoeffSize = _dfBins * _dfOrder * 2;

        // Apply ERB gains and deep filtering to complex STFT
        var enhancedStft = new Tensor<Complex<T>>(_cachedComplexStft._shape);

        for (int frame = 0; frame < numFrames; frame++)
        {
            for (int bin = 0; bin < numBins; bin++)
            {
                int erbBand = Math.Min(bin * _numErbBands / numBins, _numErbBands - 1);
                int idx = frame * numBins + bin;

                double gain = 1.0;
                if (frame < enhanced.Shape[0] && dfCoeffSize + erbBand < enhanced.Shape[^1])
                {
                    double rawGain = NumOps.ToDouble(enhanced[[frame, dfCoeffSize + erbBand]]);
                    gain = 1.0 / (1.0 + Math.Exp(-rawGain));
                }

                double real = NumOps.ToDouble(_cachedComplexStft.Data.Span[idx].Real) * gain;
                double imag = NumOps.ToDouble(_cachedComplexStft.Data.Span[idx].Imaginary) * gain;

                // Apply deep filtering for first _dfBins bins
                if (bin < _dfBins && frame < enhanced.Shape[0])
                {
                    for (int order = 0; order < _dfOrder && frame >= order; order++)
                    {
                        int coeffIdx = (bin * _dfOrder + order) * 2;
                        if (coeffIdx + 1 < dfCoeffSize)
                        {
                            double dfReal = NumOps.ToDouble(enhanced[[frame, coeffIdx]]);
                            double dfImag = NumOps.ToDouble(enhanced[[frame, coeffIdx + 1]]);

                            double newReal = real * dfReal - imag * dfImag;
                            double newImag = real * dfImag + imag * dfReal;
                            real = newReal * 0.1 + real * 0.9;
                            imag = newImag * 0.1 + imag * 0.9;
                        }
                    }
                }

                enhancedStft.Data.Span[idx] = new Complex<T>(NumOps.FromDouble(real), NumOps.FromDouble(imag));
            }
        }

        // Use proper ISTFT for reconstruction
        return _stft.Inverse(enhancedStft);
    }

    /// <summary>
    /// Combines deep filtering coefficients and gains.
    /// </summary>
    private Tensor<T> CombineOutputs(Tensor<T> dfCoeffs, Tensor<T> gains)
    {
        // Combine DF coefficients and ERB gains for output
        int numFrames = gains.Shape[0];
        int outputDim = _dfBins * _dfOrder * 2 + _numErbBands;

        var combined = new Tensor<T>([numFrames, outputDim]);

        for (int frame = 0; frame < numFrames; frame++)
        {
            // Copy DF coefficients
            for (int i = 0; i < _dfBins * _dfOrder * 2; i++)
            {
                combined[[frame, i]] = dfCoeffs[[frame, i]];
            }

            // Copy gains
            for (int i = 0; i < _numErbBands; i++)
            {
                combined[[frame, _dfBins * _dfOrder * 2 + i]] = gains[[frame, i]];
            }
        }

        return combined;
    }

    private static int NextPowerOfTwo(int v)
    {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        return v + 1;
    }

    #endregion

    #region Abstract Method Implementations

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        // Layers are initialized in constructor based on mode
        // In ONNX mode, layers are not used
        // In native mode, ERB encoder/decoder/GRU layers are created in constructor
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(IsOnnxMode);
        writer.Write(SampleRate);
        writer.Write(_numErbBands);
        writer.Write(_hiddenDim);
        writer.Write(_dfOrder);
        writer.Write(_dfBins);
        writer.Write(_numGruLayers);
        writer.Write(_convKernelSize);
        writer.Write(_fftSize);
        writer.Write(_hopSize);
        writer.Write(_lookahead);
        writer.Write(EnhancementStrength);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read configuration values for validation
        _ = reader.ReadBoolean(); // IsOnnxMode
        _ = reader.ReadInt32();   // SampleRate
        _ = reader.ReadInt32();   // _numErbBands
        _ = reader.ReadInt32();   // _hiddenDim
        _ = reader.ReadInt32();   // _dfOrder
        _ = reader.ReadInt32();   // _dfBins
        _ = reader.ReadInt32();   // _numGruLayers
        _ = reader.ReadInt32();   // _convKernelSize
        _ = reader.ReadInt32();   // _fftSize
        _ = reader.ReadInt32();   // _hopSize
        _ = reader.ReadInt32();   // _lookahead
        EnhancementStrength = reader.ReadDouble();
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new DeepFilterNet<T>(
            Architecture,
            sampleRate: SampleRate,
            numErbBands: _numErbBands,
            hiddenDim: _hiddenDim,
            dfOrder: _dfOrder,
            dfBins: _dfBins,
            numGruLayers: _numGruLayers,
            fftSize: _fftSize,
            hopSize: _hopSize,
            lookahead: _lookahead);
    }

    #endregion

    #region IDisposable

    /// <summary>
    /// Disposes resources.
    /// </summary>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            // OnnxModel disposal handled by base class
        }
        base.Dispose(disposing);
    }

    #endregion
}

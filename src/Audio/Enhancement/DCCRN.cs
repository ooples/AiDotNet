using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
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
/// DCCRN - Deep Complex Convolution Recurrent Network for speech enhancement.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DCCRN operates directly on complex-valued spectrograms, preserving phase information
/// for high-quality speech enhancement. Key features:
/// - Complex-valued convolutions for better spectral modeling
/// - LSTM layers for temporal dependencies
/// - Skip connections for gradient flow
/// - Mask-based enhancement for clean speech estimation
/// </para>
/// <para>
/// <b>For Beginners:</b> DCCRN is a neural network designed specifically for
/// cleaning up noisy audio. Unlike simpler methods that only work with the
/// "loudness" of frequencies, DCCRN also considers the "timing" (phase),
/// which results in more natural-sounding enhanced audio.
///
/// Think of it like this: regular enhancement is like adjusting volume
/// of different frequencies, while DCCRN can also adjust the timing of
/// sound waves to better reconstruct the original clean speech.
///
/// Usage:
/// <code>
/// var model = new DCCRN&lt;float&gt;(architecture, "dccrn.onnx");
/// var cleanAudio = model.Enhance(noisyAudio);
/// </code>
/// </para>
/// <para>
/// Reference: "DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware
/// Speech Enhancement" by Hu et al., Interspeech 2020
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelTask(ModelTask.Enhancement)]
[ModelTask(ModelTask.Denoising)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement", "https://arxiv.org/abs/2008.00264", Year = 2020, Authors = "Yanxin Hu, Yun Liu, Shubo Lv, Mengtao Xing, Shimin Zhang, Yihui Fu, Jian Wu, Bihong Zhang, Lei Xie")]
public class DCCRN<T> : AudioNeuralNetworkBase<T>, IAudioEnhancer<T>
{
    private readonly DCCRNOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Model Architecture Parameters

    /// <summary>
    /// Number of encoder/decoder stages.
    /// </summary>
    private int _numStages;

    /// <summary>
    /// Base number of channels.
    /// </summary>
    private int _baseChannels;

    /// <summary>
    /// LSTM hidden dimension.
    /// </summary>
    private int _lstmHiddenDim;

    /// <summary>
    /// Number of LSTM layers.
    /// </summary>
    private int _numLstmLayers;

    /// <summary>
    /// FFT size for STFT.
    /// </summary>
    private int _fftSize;

    /// <summary>
    /// Hop size for STFT.
    /// </summary>
    private int _hopSize;

    /// <summary>
    /// Whether to use complex mask estimation.
    /// </summary>
    private bool _useComplexMask;

    /// <summary>
    /// Kernel size for convolutions.
    /// </summary>
    private int _kernelSize;

    /// <summary>
    /// Stride for convolutions.
    /// </summary>
    private int _stride;

    #endregion

    #region Network Layers

    /// <summary>
    /// Encoder layers (complex convolutions).
    /// </summary>
    private readonly List<ILayer<T>> _encoder = [];

    /// <summary>
    /// LSTM layers.
    /// </summary>
    private readonly List<ILayer<T>> _lstmLayers = [];

    /// <summary>
    /// Projection layer to map LSTM output back to encoder spatial dimensions.
    /// </summary>
    private ILayer<T>? _lstmProjection;

    /// <summary>
    /// Encoder output dimension (channels * freqBins) for LSTM projection.
    /// </summary>
    private int _encoderOutputDim;

    /// <summary>
    /// Decoder layers (complex transposed convolutions).
    /// </summary>
    private readonly List<ILayer<T>> _decoder = [];

    /// <summary>
    /// Skip connection layers.
    /// </summary>
    private readonly List<ILayer<T>> _skipLayers = [];

    /// <summary>
    /// Mask estimation layer.
    /// </summary>
    private ILayer<T>? _maskLayer;

    /// <summary>
    /// Guards <see cref="ResolveLazyLayerShapes"/> so the one-shot warm-up forward runs at most once.
    /// </summary>
    private bool _lazyShapesWarmed;

    // TRUE complex convolutions (Hu et al. 2020, the "Deep Complex" innovation). Each complex conv holds
    // a real (Wr) and imaginary (Wi) kernel; a complex feature map is carried as [B, 2*C, F, T] with the
    // real channels first and the imaginary channels second. Encoder kernels are [Cout, Cin, kH, kW]
    // (Conv2D layout); decoder kernels are [Cin, Cout, kH, kW] (ConvTranspose2D layout). No bias — each
    // conv is immediately followed by BatchNorm, whose shift makes a conv bias redundant. These raw
    // tensors are the model's trainable parameters for the conv stages; they are surfaced via
    // GetExtraTrainableTensors so the base tape watches/updates them and GetParameters / serialization
    // round-trip them, exactly like ViT cls/pos tokens.
    private readonly System.Collections.Generic.List<Tensor<T>> _encWr = [];
    private readonly System.Collections.Generic.List<Tensor<T>> _encWi = [];
    private readonly System.Collections.Generic.List<Tensor<T>> _decWr = [];
    private readonly System.Collections.Generic.List<Tensor<T>> _decWi = [];

    // Per-stage BatchNorm applied after each complex conv (encoder: numStages; decoder: numStages-1, the
    // final output stage has none). These stay as real BatchNorm layers over the 2*C real channels.
    private readonly System.Collections.Generic.List<ILayer<T>> _encBN = [];
    private readonly System.Collections.Generic.List<ILayer<T>> _decBN = [];

    // Paper conv geometry (Hu 2020): kernel (5,2) freq×time, stride (2,1) — downsample frequency only,
    // preserve time; freq padding 2 (kH/2), no time padding.
    private const int ComplexKernelH = 5;
    private const int ComplexKernelW = 2;
    private const int ComplexStrideH = 2;
    private const int ComplexStrideW = 1;
    private const int ComplexPadH = 2;
    // Time padding 1 (not 0): with kernel width 2 and stride 1, padW=0 shrinks time by 1 each stage, so a
    // short clip (< numStages+1 frames) collapses to width 0 and Conv2D throws "Invalid output dimensions
    // (Fx0)". padW=1 keeps time from collapsing (it grows by 1/stage instead); the decoder transpose conv
    // shrinks it back and MatchSpatial aligns the residual to the STFT.
    private const int ComplexPadW = 1;

    /// <summary>Complex channel count at encoder stage i (Cout). The real tensor carries 2× this.</summary>
    private int EncoderComplexChannels(int stage) => _baseChannels * (int)Math.Pow(2, Math.Min(stage, 4));

    #endregion

    #region ONNX Mode Fields

    // ONNX model is inherited from AudioNeuralNetworkBase<T>.OnnxModel

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

    /// <summary>
    /// Cached encoder outputs for skip connections.
    /// </summary>
    private readonly List<Tensor<T>> _encoderOutputs = [];

    #endregion

    #region IAudioEnhancer Properties

    /// <inheritdoc/>
    public int NumChannels { get; protected set; } = 1;

    /// <inheritdoc/>
    public double EnhancementStrength { get; set; } = 1.0;

    /// <inheritdoc/>
    public int LatencySamples => _fftSize;

    #endregion

    #region Public Properties

    /// <summary>
    /// Gets whether this network supports training.
    /// </summary>
    public override bool SupportsTraining => !IsOnnxMode;

    // IsOnnxMode is inherited from AudioNeuralNetworkBase<T>

    /// <summary>
    /// Gets the number of encoder/decoder stages.
    /// </summary>
    public int NumStages => _numStages;

    /// <summary>
    /// Gets whether complex mask is used.
    /// </summary>
    public bool UseComplexMask => _useComplexMask;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a DCCRN model for ONNX inference.
    /// </summary>
    /// <param name="architecture">Neural network architecture configuration.</param>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="sampleRate">Audio sample rate in Hz. Default is 16000.</param>
    /// <param name="fftSize">FFT size for STFT. Default is 512.</param>
    /// <param name="hopSize">Hop size for STFT. Default is 256.</param>
    /// <param name="onnxOptions">Optional ONNX runtime options.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to load a pre-trained DCCRN
    /// model for speech enhancement.
    ///
    /// Example:
    /// <code>
    /// var model = new DCCRN&lt;float&gt;(architecture, "dccrn_16k.onnx");
    /// var clean = model.Enhance(noisy);
    /// </code>
    /// </para>
    /// </remarks>
    public DCCRN(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        int sampleRate = 16000,
        int fftSize = 512,
        int hopSize = 256,
        OnnxModelOptions? onnxOptions = null,
        DCCRNOptions? options = null)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        _options = options ?? new DCCRNOptions();
        Options = _options;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Model file not found: {modelPath}");

        SampleRate = sampleRate;
        _fftSize = fftSize;
        _hopSize = hopSize;
        _numStages = 6;
        _baseChannels = 32;
        _lstmHiddenDim = 256;
        _numLstmLayers = 2;
        _useComplexMask = true;
        _kernelSize = 5;
        _stride = 2;
        OnnxModel = new OnnxModel<T>(modelPath, onnxOptions);

        // Default loss function (MSE is standard for speech enhancement)
        _lossFunction = new MeanSquaredErrorLoss<T>();
    }

    /// <summary>
    /// Creates a DCCRN model for native training and inference.
    /// </summary>
    /// <param name="architecture">Neural network architecture configuration.</param>
    /// <param name="sampleRate">Audio sample rate in Hz. Default is 16000.</param>
    /// <param name="numStages">Number of encoder/decoder stages. Default is 6.</param>
    /// <param name="baseChannels">Base number of channels. Default is 32.</param>
    /// <param name="lstmHiddenDim">LSTM hidden dimension. Default is 256.</param>
    /// <param name="numLstmLayers">Number of LSTM layers. Default is 2.</param>
    /// <param name="fftSize">FFT size. Default is 512.</param>
    /// <param name="hopSize">Hop size. Default is 256.</param>
    /// <param name="useComplexMask">Use complex mask estimation. Default is true.</param>
    /// <param name="kernelSize">Convolution kernel size. Default is 5.</param>
    /// <param name="stride">Convolution stride. Default is 2.</param>
    /// <param name="optimizer">Optimizer for training.</param>
    /// <param name="lossFunction">Loss function for training.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to train DCCRN from scratch.
    ///
    /// Key parameters:
    /// - numStages: More stages = deeper network, better results but slower
    /// - baseChannels: More channels = more capacity
    /// - useComplexMask: true preserves phase for better quality
    ///
    /// Example:
    /// <code>
    /// var model = new DCCRN&lt;float&gt;(
    ///     architecture,
    ///     numStages: 6,
    ///     baseChannels: 32,
    ///     useComplexMask: true);
    ///
    /// model.Train(noisyBatch, cleanBatch);
    /// </code>
    /// </para>
    /// </remarks>
    public DCCRN(
        NeuralNetworkArchitecture<T> architecture,
        int sampleRate = 16000,
        int numStages = 6,
        int baseChannels = 32,
        int lstmHiddenDim = 256,
        int numLstmLayers = 2,
        int fftSize = 512,
        int hopSize = 256,
        bool useComplexMask = true,
        int kernelSize = 5,
        int stride = 2,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        DCCRNOptions? options = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>())
    {
        _options = options ?? new DCCRNOptions();
        Options = _options;
        // Validate parameters
        if (sampleRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(sampleRate), "Sample rate must be positive.");
        if (numStages <= 0)
            throw new ArgumentOutOfRangeException(nameof(numStages), "Number of stages must be positive.");
        if (baseChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(baseChannels), "Base channels must be positive.");
        if (lstmHiddenDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(lstmHiddenDim), "LSTM hidden dimension must be positive.");
        if (numLstmLayers <= 0)
            throw new ArgumentOutOfRangeException(nameof(numLstmLayers), "Number of LSTM layers must be positive.");
        if (fftSize <= 0 || (fftSize & (fftSize - 1)) != 0)
            throw new ArgumentOutOfRangeException(nameof(fftSize), "FFT size must be a positive power of 2.");
        if (hopSize <= 0 || hopSize > fftSize)
            throw new ArgumentOutOfRangeException(nameof(hopSize), "Hop size must be positive and not exceed FFT size.");

        SampleRate = sampleRate;
        _numStages = numStages;
        _baseChannels = baseChannels;
        _lstmHiddenDim = lstmHiddenDim;
        _numLstmLayers = numLstmLayers;
        _fftSize = fftSize;
        _hopSize = hopSize;
        _useComplexMask = useComplexMask;
        _kernelSize = kernelSize;
        _stride = stride;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeNativeLayers();
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes native layers for training mode.
    /// </summary>
    private void InitializeNativeLayers()
    {
        int freqBins = _fftSize / 2 + 1;

        // Frequency size after numStages strided complex convs (kH=5, padH=2, strideH=2). The complex
        // conv weights themselves live outside Layers (raw tensors surfaced via GetExtraTrainableTensors);
        // the layer list holds only the per-stage BatchNorms, the LSTM stack, the projection Dense, and
        // the mask activation.
        int downFreqBins = freqBins;
        for (int s = 0; s < _numStages; s++)
            downFreqBins = (downFreqBins + 2 * ComplexPadH - ComplexKernelH) / ComplexStrideH + 1;
        downFreqBins = Math.Max(1, downFreqBins);

        // Bottleneck feature width = REAL channel count (2 * complex Cout at the deepest stage) * F'. The
        // factor 2 is the complex real/imag split carried in the channel dim.
        int deepestComplex = EncoderComplexChannels(_numStages - 1);
        _encoderOutputDim = 2 * deepestComplex * downFreqBins;

        var layers = (Architecture.Layers != null && Architecture.Layers.Count > 0)
            ? Architecture.Layers.ToList()
            : LayerHelper<T>.CreateDCCRNLayers(
                numStages: _numStages, numLstmLayers: _numLstmLayers,
                lstmHiddenDim: _lstmHiddenDim, projectionDim: _encoderOutputDim,
                useComplexMask: _useComplexMask).ToList();

        Layers.Clear();
        Layers.AddRange(layers);
        DistributeLayers();
    }

    /// <summary>
    /// Populates the internal sub-lists (_encBN / _lstmLayers / _lstmProjection / _decBN / _maskLayer)
    /// from the current <see cref="NeuralNetworkBase{T}.Layers"/>. Split out from
    /// <see cref="InitializeNativeLayers"/> so deserialization can re-derive the sub-lists from the layers
    /// the BASE already reconstructed and restored trained weights into — WITHOUT rebuilding fresh layers
    /// (which would clear Layers and discard those restored weights, dropping trained state on a clone;
    /// the Clone_AfterTraining #1221 class). Layout: [encoder BN × numStages] [LSTM × numLstm]
    /// [projection Dense] [decoder BN × (numStages-1)] [mask activation].
    /// </summary>
    private void DistributeLayers()
    {
        _encBN.Clear();
        _decBN.Clear();
        _lstmLayers.Clear();
        _encoder.Clear();
        _skipLayers.Clear();
        _decoder.Clear();

        int idx = 0;
        for (int i = 0; i < _numStages && idx < Layers.Count; i++)
            _encBN.Add(Layers[idx++]);
        for (int i = 0; i < _numLstmLayers && idx < Layers.Count; i++)
            _lstmLayers.Add(Layers[idx++]);
        if (idx < Layers.Count)
            _lstmProjection = Layers[idx++];
        for (int i = 0; i < _numStages - 1 && idx < Layers.Count; i++)
            _decBN.Add(Layers[idx++]);
        if (idx < Layers.Count)
            _maskLayer = Layers[idx];
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
        return Enhance(audio);
    }

    /// <inheritdoc/>
    public Tensor<T> ProcessChunk(Tensor<T> audioChunk)
    {
        return Enhance(audioChunk);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        base.ResetState();
        foreach (var layer in _lstmLayers)
        {
            if (layer is LSTMLayer<T> lstm)
            {
                lstm.ResetState();
            }
        }
    }

    /// <inheritdoc/>
    public void EstimateNoiseProfile(Tensor<T> noiseOnlyAudio)
    {
        // DCCRN learns noise suppression implicitly
    }

    #endregion

    #region AudioNeuralNetworkBase Implementation

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        return ComputeComplexSTFT(rawAudio);
    }

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        return ComputeInverseSTFT(modelOutput);
    }

    /// <summary>
    /// Resolves every lazy layer's input dimension by running ONE dummy pass through the REAL enhancement
    /// graph at the actual STFT shape [1, 2, numBins, frames]. The base implementation walks the flat
    /// <see cref="NeuralNetworkBase{T}.Layers"/> list feeding the raw-audio architecture shape
    /// sequentially, which is wrong here: the encoder convolutions consume the 2-channel complex STFT, not
    /// the waveform. Without correct resolution the first conv locked its input depth to 1 and then threw
    /// "Expected input depth 1, but got 2" on the real forward, and post-deserialize SetParameters silently
    /// skipped still-unresolved layers so Clone/round-trip lost weights (issue #1221 class).
    /// </summary>
    protected override void ResolveLazyLayerShapes()
    {
        if (_lazyShapesWarmed) return;
        _lazyShapesWarmed = true; // set first so any reentrancy is a no-op
        if (IsOnnxMode) return;

        int numBins = _fftSize / 2 + 1;
        // Enough frames that every strided stage keeps a positive time extent.
        int frames = Math.Max(4, (int)Math.Pow(2, _numStages));
        var dummy = new Tensor<T>([1, 2, numBins, frames]);

        bool wasTraining = IsTrainingMode;
        if (wasTraining) SetTrainingMode(false);
        try { _ = ForwardNative(dummy); }
        catch { /* best-effort; a genuine forward failure surfaces on the real Train/Predict */ }
        finally { if (wasTraining) SetTrainingMode(true); }
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        var stft = PreprocessAudio(input);

        Tensor<T> enhancedStft;
        if (IsOnnxMode && OnnxModel is not null)
        {
            enhancedStft = OnnxModel.Run(stft);
        }
        else
        {
            enhancedStft = ForwardNative(stft);
        }

        return PostprocessOutput(enhancedStft);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!SupportsTraining)
            throw new InvalidOperationException("Cannot train in ONNX mode.");

        // Tape-transparent training. The loss lives in the STFT domain (enhanced STFT vs clean STFT), so
        // the STFT front-end only produces the constant input/target tensors — it needs no gradient — and
        // the whole ENHANCEMENT graph (ForwardNative = encoder/LSTM/decoder/mask/apply, all Engine ops)
        // is what the tape records. TrainWithTape runs ForwardForTraining(noisyStft), compares to
        // cleanStft, backpropagates, and steps the optimizer over every layer. This replaces the old body
        // that computed a gradient it never applied (it only called _optimizer.UpdateParameters(Layers)
        // with no backward pass), so DCCRN previously never actually trained.
        var noisyStft = PreprocessAudio(input);
        var cleanStft = PreprocessAudio(expectedOutput);
        // The enhanced STFT (prediction) has the noisy input's frame count; align the clean target to the
        // same [F, T] so the STFT-domain loss compares matching representations. input and expectedOutput
        // can differ slightly in length (the generated harness does not guarantee equal-length pairs), and
        // MeanSquaredErrorLoss.ComputeTapeLoss requires an exact shape match.
        cleanStft = MatchSpatial(cleanStft, noisyStft.Shape[2], noisyStft.Shape[3]);
        TrainWithTape(noisyStft, cleanStft, _optimizer);
    }

    /// <summary>
    /// Training forward pass: consumes a complex STFT [B, 2, F, T] and returns the enhanced STFT. Used by
    /// <see cref="Train"/> via TrainWithTape (which supplies STFT tensors), so it runs ONLY the enhancement
    /// graph — the STFT/inverse-STFT bracketing lives in Train/PredictCore, keeping the tape on the
    /// learnable layers.
    /// </summary>
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => ForwardNative(input);

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        // Ensure lazy layer shapes are resolved BEFORE assigning parameters. On a freshly-created clone
        // the encoder/decoder conv layers are still deferred (InputDepth=-1, ParameterCount=0), so a bare
        // Layers walk would see 0 parameters, skip every layer, and leave the clone on its own random
        // initialization — diverging from the original (Clone_ShouldProduceIdenticalOutput). The one-shot
        // STFT-shaped warm forward resolves every layer so the slices below actually land.
        ResolveLazyLayerShapes();

        // SET each layer's parameters from the flat vector, walking Layers in the SAME order the base
        // GetParameters emits them so the slices line up. This is the Clone / deserialize restore contract
        // (a value SET, not a gradient step). The previous version iterated internal sub-lists
        // (_encoder/_lstm/_decoder) — which omits the mask layer and can differ from GetParameters' order —
        // producing clones with the wrong weights (Clone_ShouldProduceIdenticalOutput).
        int offset = 0;
        foreach (var layer in Layers)
        {
            int count = layer.GetParameters().Length;
            if (count == 0) continue;
            layer.SetParameters(parameters.Slice(offset, count));
            offset += count;
        }

        // Weights changed wholesale — invalidate any packed inference-weight caches so the next Predict
        // rebuilds them from these params (otherwise a clone keeps serving stale packed weights and
        // predicts differently from the original — Clone_ShouldProduceIdenticalOutput).
        InvalidateWeightCachesAfterSuccessfulWeightUpdate();
    }

    /// <summary>
    /// Returns per-layer activations. DCCRN's forward is grouped (STFT front-end -> encoder / LSTM /
    /// decoder / mask), not a flat <see cref="NeuralNetworkBase{T}.Layers"/> walk, so the base
    /// implementation feeds the raw waveform straight into the encoder convolutions and mis-shapes them
    /// (rank-3 / wrong-depth crash). Run the real STFT + encoder stack here and record each stage.
    /// </summary>
    public override System.Collections.Generic.Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));

        var activations = new System.Collections.Generic.Dictionary<string, Tensor<T>>();
        if (IsOnnxMode) return activations;

        var x = ComputeComplexSTFT(input);
        activations["stft_input"] = x.Clone();
        var leaky = NumOps.FromDouble(0.2);
        for (int i = 0; i < _numStages; i++)
        {
            x = ComplexConv2D(x, _encWr, _encWi, i, EncoderComplexChannels(i), transpose: false);
            activations[$"encoder_{i}_complexconv"] = x.Clone();
            x = Engine.LeakyReLU(x, leaky);
            if (i < _encBN.Count) x = _encBN[i].Forward(x);
            activations[$"encoder_{i}_bn"] = x.Clone();
        }
        return activations;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Version = "1.0.0",
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumStages", _numStages },
                { "BaseChannels", _baseChannels },
                { "LstmHiddenDim", _lstmHiddenDim },
                { "NumLstmLayers", _numLstmLayers },
                { "FftSize", _fftSize },
                { "HopSize", _hopSize },
                { "UseComplexMask", _useComplexMask },
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
    private Tensor<T> ForwardNative(Tensor<T> stft)
    {
        _encoderOutputs.Clear();
        var x = stft; // [B, 2, F, T] — one complex channel (real=ch0, imag=ch1)
        var leaky = NumOps.FromDouble(0.2);

        // Encoder: TRUE complex convolution (Hu 2020) at each stage, stride (2,1) so only frequency is
        // downsampled and time is preserved (the paper's asymmetric conv). Complex feature maps flow as
        // [B, 2*C, F', T]. Each stage is complex-conv -> LeakyReLU -> BatchNorm; the stage output is cached
        // for the mirrored decoder skip. All ops are tape-aware (Engine.Conv2D + Engine primitives), so
        // autodiff flows to the complex kernels with no manual backward and no scalar loops.
        for (int i = 0; i < _numStages; i++)
        {
            x = ComplexConv2D(x, _encWr, _encWi, i, EncoderComplexChannels(i), transpose: false);
            x = Engine.LeakyReLU(x, leaky);
            if (i < _encBN.Count) x = _encBN[i].Forward(x);
            _encoderOutputs.Add(x);
        }

        // Bottleneck LSTM over time. x is [B, 2*C, F', T]; the LSTM models the TIME axis with (2*C*F')
        // features per step, so permute time to the front — [B, T, 2*C, F'] — before flattening. A bare
        // Reshape without the permute would scramble channel/freq/time ordering.
        int batchSize = x.Shape[0];
        int channels = x.Shape[1];   // = 2*C (interleaved real then imag halves)
        int freqBins = x.Shape[2];
        int timeFrames = x.Shape[3];

        var reshaped = Engine.Reshape(
            Engine.TensorPermute(x, [0, 3, 1, 2]),
            [batchSize, timeFrames, channels * freqBins]);

        foreach (var lstm in _lstmLayers)
        {
            reshaped = lstm.Forward(reshaped);
        }

        // Project LSTM output [B, T, hidden] back to the encoder feature width [B, T, 2*C*F'] via the
        // projection Dense applied to a flattened [B*T, hidden] batch, then reshape.
        if (_lstmProjection is not null)
        {
            int hidden = reshaped.Shape[2];
            var flat = Engine.Reshape(reshaped, [batchSize * timeFrames, hidden]);
            var projected = _lstmProjection.Forward(flat);
            reshaped = Engine.Reshape(projected, [batchSize, timeFrames, channels * freqBins]);
        }

        // Reshape back to spatial [B, 2*C, F', T], inverting the permute ([B,T,2C,F'] -> [B,2C,F',T]).
        x = Engine.TensorPermute(
            Engine.Reshape(reshaped, [batchSize, timeFrames, channels, freqBins]),
            [0, 2, 3, 1]);

        // Decoder: mirror the encoder. At decoder stage d (encoder stage e = numStages-1-d) concat the
        // cached skip along the channel axis, then a complex TRANSPOSE conv upsamples frequency back. All
        // but the final stage get LeakyReLU + BatchNorm; the final stage emits the single complex channel
        // that is the complex ratio mask. MatchSpatial aligns x to the skip (and later to the STFT) since
        // transpose conv without output-padding can land a bin/frame short of the strided encoder.
        for (int d = 0; d < _numStages; d++)
        {
            int e = _numStages - 1 - d;
            if (e < _encoderOutputs.Count)
            {
                var skip = _encoderOutputs[e];
                x = MatchSpatial(x, skip.Shape[2], skip.Shape[3]);
                x = Engine.TensorConcatenate([x, skip], axis: 1);
            }

            int cout = (d < _numStages - 1) ? EncoderComplexChannels(_numStages - 2 - d) : 1;
            x = ComplexConv2D(x, _decWr, _decWi, d, cout, transpose: true);
            if (d < _numStages - 1)
            {
                x = Engine.LeakyReLU(x, leaky);
                if (d < _decBN.Count) x = _decBN[d].Forward(x);
            }
        }

        // x is now [B, 2, F', T'] — the complex ratio mask. Align to the input STFT, apply the bounded mask
        // activation (Tanh CRM / Sigmoid magnitude), and apply it to the noisy STFT.
        x = MatchSpatial(x, stft.Shape[2], stft.Shape[3]);
        if (_maskLayer is not null) x = _maskLayer.Forward(x);
        return ApplyComplexMask(stft, x);
    }

    /// <summary>
    /// Crops or zero-pads <paramref name="x"/> ([B, C, F, T]) so its freq (axis 2) and time (axis 3)
    /// extents equal <paramref name="targetF"/> / <paramref name="targetT"/>. Cropping uses a tape-aware
    /// slice; padding concatenates a zero block (the pad contributes no gradient, x's slice keeps its
    /// graph). Lets the U-Net skip concatenations and the final mask align even though the transposed
    /// convolutions don't bit-exactly invert the strided encoder downsampling.
    /// </summary>
    private Tensor<T> MatchSpatial(Tensor<T> x, int targetF, int targetT)
    {
        if (x.Shape[2] > targetF)
            x = Engine.TensorSlice(x, [0, 0, 0, 0], [x.Shape[0], x.Shape[1], targetF, x.Shape[3]]);
        else if (x.Shape[2] < targetF)
            x = Engine.TensorConcatenate(
                [x, new Tensor<T>([x.Shape[0], x.Shape[1], targetF - x.Shape[2], x.Shape[3]])], axis: 2);

        if (x.Shape[3] > targetT)
            x = Engine.TensorSlice(x, [0, 0, 0, 0], [x.Shape[0], x.Shape[1], x.Shape[2], targetT]);
        else if (x.Shape[3] < targetT)
            x = Engine.TensorConcatenate(
                [x, new Tensor<T>([x.Shape[0], x.Shape[1], x.Shape[2], targetT - x.Shape[3]])], axis: 3);

        return x;
    }

    /// <summary>
    /// Computes complex STFT.
    /// </summary>
    private Tensor<T> ComputeComplexSTFT(Tensor<T> audio)
    {
        var samples = audio.ToVector().ToArray();
        int numFrames = (samples.Length - _fftSize) / _hopSize + 1;
        int numBins = _fftSize / 2 + 1;

        // Shape: [1, 2, numBins, numFrames] for batch, real/imag, freq, time
        var stft = new Tensor<T>([1, 2, numBins, numFrames]);
        var window = CreateHannWindow(_fftSize);

        for (int frame = 0; frame < numFrames; frame++)
        {
            int start = frame * _hopSize;
            var frameData = new double[_fftSize];

            for (int i = 0; i < _fftSize && start + i < samples.Length; i++)
            {
                frameData[i] = NumOps.ToDouble(samples[start + i]) * window[i];
            }

            for (int k = 0; k < numBins; k++)
            {
                double real = 0, imag = 0;
                for (int n = 0; n < _fftSize; n++)
                {
                    double angle = -2 * Math.PI * k * n / _fftSize;
                    real += frameData[n] * Math.Cos(angle);
                    imag += frameData[n] * Math.Sin(angle);
                }
                stft[[0, 0, k, frame]] = NumOps.FromDouble(real);
                stft[[0, 1, k, frame]] = NumOps.FromDouble(imag);
            }
        }

        return stft;
    }

    /// <summary>
    /// Computes inverse STFT.
    /// </summary>
    private Tensor<T> ComputeInverseSTFT(Tensor<T> stft)
    {
        int numBins = stft.Shape[2];
        int numFrames = stft.Shape[3];
        int numSamples = numFrames * _hopSize + _fftSize;

        var audio = new Tensor<T>([numSamples]);
        var window = CreateHannWindow(_fftSize);

        for (int frame = 0; frame < numFrames; frame++)
        {
            int start = frame * _hopSize;

            for (int n = 0; n < _fftSize && start + n < numSamples; n++)
            {
                double sum = 0;
                for (int k = 0; k < numBins; k++)
                {
                    double real = NumOps.ToDouble(stft[[0, 0, k, frame]]);
                    double imag = NumOps.ToDouble(stft[[0, 1, k, frame]]);
                    double angle = 2 * Math.PI * k * n / _fftSize;
                    sum += real * Math.Cos(angle) - imag * Math.Sin(angle);
                }
                double currentVal = NumOps.ToDouble(audio[[start + n]]);
                audio[[start + n]] = NumOps.FromDouble(currentVal + sum * window[n] / _fftSize);
            }
        }

        return audio;
    }

    /// <summary>
    /// Applies complex mask to STFT.
    /// </summary>
    /// <summary>
    /// Surfaces the complex-conv real/imaginary kernels as the model's raw trainable tensors so the base
    /// gradient tape watches and updates them and GetParameters / serialization round-trip them (the same
    /// mechanism ViT cls/pos tokens use). They are created lazily on first forward once input channel
    /// counts are known, so this yields nothing until then; the base runs a warm-up forward before
    /// collecting parameters, so they exist by collection time.
    /// </summary>
    protected override System.Collections.Generic.IEnumerable<Tensor<T>> GetExtraTrainableTensors()
    {
        foreach (var w in _encWr) if (w is not null) yield return w;
        foreach (var w in _encWi) if (w is not null) yield return w;
        foreach (var w in _decWr) if (w is not null) yield return w;
        foreach (var w in _decWi) if (w is not null) yield return w;
    }

    /// <summary>
    /// True complex convolution (Hu 2020): for complex input carried as [B, 2*Cin, F, T] (real channels
    /// first, imaginary second) and complex kernels Wr, Wi, computes
    ///   Re(out) = conv(Xr, Wr) - conv(Xi, Wi),   Im(out) = conv(Xr, Wi) + conv(Xi, Wr)
    /// and returns [B, 2*Cout, F', T']. Uses tape-aware Engine.Conv2D / Engine.ConvTranspose2D with the
    /// paper's asymmetric stride (2,1) / kernel (5,2), so autodiff flows to Wr/Wi automatically. No bias
    /// (a BatchNorm follows every conv). Kernels are materialized lazily (Cin inferred from the input) and
    /// deterministically Glorot-initialized so a fresh instance is reproducible; Clone/deserialize then
    /// overwrite them with the source weights.
    /// </summary>
    private Tensor<T> ComplexConv2D(Tensor<T> input,
        System.Collections.Generic.List<Tensor<T>> wrList, System.Collections.Generic.List<Tensor<T>> wiList,
        int stageIdx, int coutComplex, bool transpose)
    {
        int cinComplex = input.Shape[1] / 2;
        while (wrList.Count <= stageIdx) { wrList.Add(null!); wiList.Add(null!); }
        if (wrList[stageIdx] is null)
        {
            // Conv2D kernel layout [Cout, Cin, kH, kW]; ConvTranspose2D layout [Cin, Cout, kH, kW].
            int[] kshape = transpose
                ? [cinComplex, coutComplex, ComplexKernelH, ComplexKernelW]
                : [coutComplex, cinComplex, ComplexKernelH, ComplexKernelW];
            wrList[stageIdx] = InitComplexKernel(kshape, cinComplex, coutComplex, (transpose ? 5000 : 0) + stageIdx);
            wiList[stageIdx] = InitComplexKernel(kshape, cinComplex, coutComplex, (transpose ? 6000 : 1000) + stageIdx);
        }
        var wr = wrList[stageIdx];
        var wi = wiList[stageIdx];

        int b = input.Shape[0], f = input.Shape[2], t = input.Shape[3];
        var xr = Engine.TensorSlice(input, [0, 0, 0, 0], [b, cinComplex, f, t]);
        var xi = Engine.TensorSlice(input, [0, cinComplex, 0, 0], [b, cinComplex, f, t]);

        int[] stride = [ComplexStrideH, ComplexStrideW];
        int[] pad = [ComplexPadH, ComplexPadW];
        Tensor<T> convXrWr, convXiWi, convXrWi, convXiWr;
        if (transpose)
        {
            int[] outPad = [0, 0];
            convXrWr = Engine.ConvTranspose2D(xr, wr, stride, pad, outPad);
            convXiWi = Engine.ConvTranspose2D(xi, wi, stride, pad, outPad);
            convXrWi = Engine.ConvTranspose2D(xr, wi, stride, pad, outPad);
            convXiWr = Engine.ConvTranspose2D(xi, wr, stride, pad, outPad);
        }
        else
        {
            int[] dil = [1, 1];
            convXrWr = Engine.Conv2D(xr, wr, stride, pad, dil);
            convXiWi = Engine.Conv2D(xi, wi, stride, pad, dil);
            convXrWi = Engine.Conv2D(xr, wi, stride, pad, dil);
            convXiWr = Engine.Conv2D(xi, wr, stride, pad, dil);
        }
        var outR = Engine.TensorSubtract(convXrWr, convXiWi);
        var outI = Engine.TensorAdd(convXrWi, convXiWr);
        return Engine.TensorConcatenate([outR, outI], axis: 1);
    }

    /// <summary>Deterministic Glorot-uniform kernel initializer (one-time, off the tape).</summary>
    private Tensor<T> InitComplexKernel(int[] shape, int cinComplex, int coutComplex, int seed)
    {
        int fanIn = cinComplex * ComplexKernelH * ComplexKernelW;
        int fanOut = coutComplex * ComplexKernelH * ComplexKernelW;
        double limit = Math.Sqrt(6.0 / (fanIn + fanOut));
        var kernel = new Tensor<T>(shape);
        var rng = new Random(seed);
        var span = kernel.Data.Span;
        for (int i = 0; i < span.Length; i++)
            span[i] = NumOps.FromDouble((rng.NextDouble() * 2.0 - 1.0) * limit);
        return kernel;
    }

    private Tensor<T> ApplyComplexMask(Tensor<T> stft, Tensor<T> mask)
    {
        // stft and mask are [B, 2, F, T] with channel 0 = real, channel 1 = imaginary. Implemented with
        // tape-aware Engine ops (slice / multiply / add / subtract / concat) so gradients flow back into
        // the mask-estimation graph — the previous scalar quadruple-loop with NumOps.ToDouble severed the
        // tape, so nothing upstream of the mask ever trained.
        int b = stft.Shape[0];
        int f = stft.Shape[2];
        int t = stft.Shape[3];

        var sr = Engine.TensorSlice(stft, [0, 0, 0, 0], [b, 1, f, t]); // Re(noisy)
        var si = Engine.TensorSlice(stft, [0, 1, 0, 0], [b, 1, f, t]); // Im(noisy)

        if (_useComplexMask)
        {
            // Complex ratio mask (Hu 2020, CRM): (sr + i·si) · (mr + i·mi).
            var mr = Engine.TensorSlice(mask, [0, 0, 0, 0], [b, 1, f, t]);
            var mi = Engine.TensorSlice(mask, [0, 1, 0, 0], [b, 1, f, t]);
            var outR = Engine.TensorSubtract(Engine.TensorMultiply(sr, mr), Engine.TensorMultiply(si, mi));
            var outI = Engine.TensorAdd(Engine.TensorMultiply(sr, mi), Engine.TensorMultiply(si, mr));
            return Engine.TensorConcatenate([outR, outI], axis: 1);
        }
        else
        {
            // Magnitude mask: scale both components by the single-channel gain.
            var m = Engine.TensorSlice(mask, [0, 0, 0, 0], [b, 1, f, t]);
            var outR = Engine.TensorMultiply(sr, m);
            var outI = Engine.TensorMultiply(si, m);
            return Engine.TensorConcatenate([outR, outI], axis: 1);
        }
    }

    /// <summary>
    /// Concatenates tensors along channel dimension with dimension validation.
    /// </summary>
    private Tensor<T> ConcatenateChannels(Tensor<T> a, Tensor<T> b)
    {
        // Validate spatial dimensions match for skip connections
        if (a.Shape[0] != b.Shape[0])
        {
            throw new ArgumentException(
                $"Batch dimension mismatch in skip connection: encoder={a.Shape[0]}, decoder={b.Shape[0]}");
        }

        // Warn if spatial dimensions don't match (use minimum to avoid index errors)
        if (a.Shape[2] != b.Shape[2] || a.Shape[3] != b.Shape[3])
        {
            System.Diagnostics.Debug.WriteLine(
                $"DCCRN: Spatial dimension mismatch in skip connection. " +
                $"Encoder: [{a.Shape[2]},{a.Shape[3]}], Decoder: [{b.Shape[2]},{b.Shape[3]}]. " +
                "Using minimum dimensions.");
        }

        int height = Math.Min(a.Shape[2], b.Shape[2]);
        int width = Math.Min(a.Shape[3], b.Shape[3]);
        int newChannels = a.Shape[1] + b.Shape[1];
        var result = new Tensor<T>([a.Shape[0], newChannels, height, width]);

        // Copy a (use minimum dimensions for safety)
        for (int i = 0; i < a.Shape[0]; i++)
            for (int c = 0; c < a.Shape[1]; c++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                        result[[i, c, h, w]] = a[[i, c, h, w]];

        // Copy b
        for (int i = 0; i < b.Shape[0]; i++)
            for (int c = 0; c < b.Shape[1]; c++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                        result[[i, a.Shape[1] + c, h, w]] = b[[i, c, h, w]];

        return result;
    }

    /// <summary>
    /// Creates a Hann window.
    /// </summary>
    private double[] CreateHannWindow(int size)
    {
        var window = new double[size];
        for (int i = 0; i < size; i++)
        {
            window[i] = 0.5 * (1 - Math.Cos(2 * Math.PI * i / (size - 1)));
        }
        return window;
    }

    #endregion

    #region Abstract Method Implementations

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        // Layers are initialized in constructor based on mode
        // In ONNX mode, layers are not used
        // In native mode, encoder/decoder/LSTM layers are created in constructor
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(IsOnnxMode);
        writer.Write(SampleRate);
        writer.Write(_numStages);
        writer.Write(_baseChannels);
        writer.Write(_lstmHiddenDim);
        writer.Write(_numLstmLayers);
        writer.Write(_fftSize);
        writer.Write(_hopSize);
        writer.Write(_useComplexMask);
        writer.Write(_kernelSize);
        writer.Write(_stride);
        writer.Write(EnhancementStrength);

        // Persist the complex-conv kernels (Wr/Wi). These are the model's trainable weights for the conv
        // stages but live OUTSIDE Layers (raw tensors via GetExtraTrainableTensors), so the base layer
        // serialization above does NOT capture them — we must write them here. They are created lazily on
        // first forward, so at serialize time they exist iff a forward/train has run; an untrained,
        // never-forwarded model writes empty lists (count 0) and re-lazy-initializes on load. This is the
        // write half of the PyTorch LazyModule contract: lazy on the compute path, materialized from the
        // checkpoint on load (see DeserializeNetworkSpecificData) — no eager materialization at construction.
        WriteComplexKernels(writer, _encWr);
        WriteComplexKernels(writer, _encWi);
        WriteComplexKernels(writer, _decWr);
        WriteComplexKernels(writer, _decWi);
    }

    /// <summary>Writes a list of complex-conv kernels as [count]([rank][dims...][values...] | -1 for null).</summary>
    private static void WriteComplexKernels(BinaryWriter writer, System.Collections.Generic.List<Tensor<T>> kernels)
    {
        writer.Write(kernels.Count);
        foreach (var k in kernels)
        {
            if (k is null) { writer.Write(-1); continue; }
            writer.Write(k.Shape.Length);
            for (int r = 0; r < k.Shape.Length; r++) writer.Write(k.Shape[r]);
            var span = k.Data.Span;
            for (int i = 0; i < span.Length; i++) writer.Write(Convert.ToDouble(span[i]));
        }
    }

    /// <summary>
    /// Materializes a list of complex-conv kernels from the checkpoint (shape + values). This is the load
    /// half of the PyTorch LazyModule contract: rather than requiring the tensors to pre-exist (which would
    /// force eager construction and a memory regression on foundation-scale models), we recreate them here
    /// from the serialized shapes so a freshly-deserialized clone carries the trained weights instead of
    /// dropping them (the Clone_AfterTraining #1221 class).
    /// </summary>
    private void ReadComplexKernels(BinaryReader reader, System.Collections.Generic.List<Tensor<T>> kernels)
    {
        kernels.Clear();
        int count = reader.ReadInt32();
        for (int i = 0; i < count; i++)
        {
            int rank = reader.ReadInt32();
            if (rank < 0) { kernels.Add(null!); continue; }
            var shape = new int[rank];
            for (int r = 0; r < rank; r++) shape[r] = reader.ReadInt32();
            var t = new Tensor<T>(shape);
            var span = t.Data.Span;
            for (int j = 0; j < span.Length; j++) span[j] = NumOps.FromDouble(reader.ReadDouble());
            kernels.Add(t);
        }
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Restore configuration values
        _ = reader.ReadBoolean(); // IsOnnxMode (determined by constructor, cannot change)
        SampleRate = reader.ReadInt32();
        _numStages = reader.ReadInt32();
        _baseChannels = reader.ReadInt32();
        _lstmHiddenDim = reader.ReadInt32();
        _numLstmLayers = reader.ReadInt32();
        _fftSize = reader.ReadInt32();
        _hopSize = reader.ReadInt32();
        _useComplexMask = reader.ReadBoolean();
        _kernelSize = reader.ReadInt32();
        _stride = reader.ReadInt32();
        EnhancementStrength = reader.ReadDouble();

        // Re-derive the internal sub-lists from the layers the BASE already reconstructed and restored
        // trained weights into. Do NOT call InitializeNativeLayers here — it would Layers.Clear() and
        // rebuild FRESH layers, discarding the base-restored trained weights and dropping the trained
        // state on a clone (Clone_AfterTraining, #1221 class). Only rebuild if the base produced no layers
        // (e.g. a bare native model with nothing serialized).
        if (!IsOnnxMode)
        {
            if (Layers.Count > 0)
                DistributeLayers();
            else
                InitializeNativeLayers();
        }

        // Materialize the complex-conv kernels from the checkpoint (same order they were written). These
        // live outside Layers, so the base did not restore them; recreating them here (from the serialized
        // shapes) carries the trained conv weights into a deserialized clone.
        ReadComplexKernels(reader, _encWr);
        ReadComplexKernels(reader, _encWi);
        ReadComplexKernels(reader, _decWr);
        ReadComplexKernels(reader, _decWi);
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new DCCRN<T>(
            Architecture,
            sampleRate: SampleRate,
            numStages: _numStages,
            baseChannels: _baseChannels,
            lstmHiddenDim: _lstmHiddenDim,
            numLstmLayers: _numLstmLayers,
            fftSize: _fftSize,
            hopSize: _hopSize,
            useComplexMask: _useComplexMask);
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

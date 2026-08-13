using AiDotNet.Attributes;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.SourceSeparation;

/// <summary>
/// Music source separation model for separating audio into stems (vocals, drums, bass, other).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This implements a U-Net based source separation approach similar to Spleeter/Demucs.
/// The model separates mixed audio into individual instrument stems using spectral masking.
/// </para>
/// <para><b>For Beginners:</b> Source separation is like unmixing a smoothie:
/// <list type="bullet">
/// <item>Input: Mixed audio with multiple instruments and vocals</item>
/// <item>Output: Separate tracks for vocals, drums, bass, and other instruments</item>
/// <item>Uses neural networks to predict which parts of the spectrum belong to each source</item>
/// </list>
///
/// Usage with ONNX model:
/// <code>
/// var separator = await MusicSourceSeparator&lt;float&gt;.CreateAsync();
/// var stems = separator.Separate(mixedAudio);
/// var vocals = stems.GetSource("vocals");
/// </code>
///
/// Usage for training:
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 1025, outputSize: 4*1025);
/// var separator = new MusicSourceSeparator&lt;float&gt;(architecture);
/// separator.Train(mixed, stems);
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.SourceSeparation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ModelInputShapeConstraint(MinimumElementCountMember = "MinimumWaveformLength")]
[ResearchPaper("Demucs: Deep Extractor for Music Sources with extra unlabeled data remixed", "https://doi.org/10.48550/arXiv.1909.01174", Year = 2019, Authors = "Alexandre Défossez, Nicolas Usunier, Léon Bottou, Francis Bach")]
public partial class MusicSourceSeparator<T> : AudioNeuralNetworkBase<T>, IMusicSourceSeparator<T>
{
    #region Fields

    private readonly SourceSeparationOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private readonly ShortTimeFourierTransform<T> _stft;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    // Native vs ONNX inference mode. Not readonly: DeserializeNetworkSpecificData
    // sets this from the persisted model bytes so a loaded native-mode checkpoint
    // doesn't silently fall back to ONNX (or vice-versa).
    private bool _useNativeMode;
    private bool _disposed;

    // Faithful waveform-Demucs (Défossez et al. 2019) native architecture. The encoder/decoder blocks
    // and bottleneck are held as typed sub-lists so the custom PredictCore can wire the U-Net skip
    // connections (encoder output added to the matching decoder input) — a flat Layers walk cannot
    // express skips. All sub-layers are ALSO registered in Layers (in forward order) so the base
    // parameter-management / serialization walk continues to work unchanged.
    private readonly System.Collections.Generic.List<Conv1DLayer<T>> _demucsEncConv = new();
    private readonly System.Collections.Generic.List<Conv1DLayer<T>> _demucsEncGate = new();
    private readonly System.Collections.Generic.List<Conv1DLayer<T>> _demucsDecGate = new();
    private readonly System.Collections.Generic.List<Conv1DTransposeLayer<T>> _demucsDecDeconv = new();
    private LSTMLayer<T>? _demucsBottleneck;
    private int _demucsDepth;

    private int MinimumWaveformLength()
    {
        int minimumLength = 1;
        for (int i = 0; i < _options.DemucsDepth; i++)
            minimumLength = checked(minimumLength * _options.DemucsStride);
        return minimumLength;
    }

    /// <summary>Standard source names for 4-stem separation.</summary>
    public static readonly string[] StandardSources = ["vocals", "drums", "bass", "other"];

    /// <summary>Source names for 2-stem separation.</summary>
    public static readonly string[] TwoStemSources = ["vocals", "accompaniment"];

    /// <summary>Source names for 5-stem separation.</summary>
    public static readonly string[] FiveStemSources = ["vocals", "drums", "bass", "piano", "other"];

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a MusicSourceSeparator for ONNX inference mode.
    /// </summary>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="options">Optional configuration options.</param>
    public MusicSourceSeparator(string modelPath, SourceSeparationOptions? options = null)
        : base(CreateMinimalArchitecture(options))
    {
        _options = options ?? new SourceSeparationOptions();
        _useNativeMode = false;
        _options.ModelPath = modelPath;

        // Set base class properties
        base.SampleRate = _options.SampleRate;

        // Initialize ONNX model
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);

        // Initialize STFT
        _stft = CreateStft();
    }

    /// <summary>
    /// Creates a MusicSourceSeparator for native training mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture.</param>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="optimizer">Optional custom optimizer.</param>
    public MusicSourceSeparator(
        NeuralNetworkArchitecture<T> architecture,
        SourceSeparationOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new SourceSeparationOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);

        // Set base class properties
        base.SampleRate = _options.SampleRate;

        // Initialize STFT
        _stft = CreateStft();

        // Initialize layers
        InitializeLayers();
    }

    /// <summary>
    /// Creates a MusicSourceSeparator for CPU-based spectral processing.
    /// </summary>
    /// <param name="options">Optional configuration options.</param>
    public MusicSourceSeparator(SourceSeparationOptions? options = null)
        : base(CreateMinimalArchitecture(options))
    {
        _options = options ?? new SourceSeparationOptions();
        _useNativeMode = false;

        // Set base class properties
        base.SampleRate = _options.SampleRate;

        // Initialize ONNX if path provided
        if (_options.ModelPath is string modelPath && !string.IsNullOrEmpty(modelPath))
        {
            OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        }

        // Initialize STFT
        _stft = CreateStft();
    }

    private static NeuralNetworkArchitecture<T> CreateMinimalArchitecture(SourceSeparationOptions? options)
    {
        var opts = options ?? new SourceSeparationOptions();
        // Frequency bins = FFT size / 2 + 1
        int freqBins = opts.FftSize / 2 + 1;
        // Output: masks for each stem
        int outputSize = opts.StemCount * freqBins;
        return new NeuralNetworkArchitecture<T>(inputFeatures: freqBins, outputSize: outputSize);
    }

    private ShortTimeFourierTransform<T> CreateStft()
    {
        return new ShortTimeFourierTransform<T>(
            nFft: _options.FftSize,
            hopLength: _options.HopLength);
    }

    #endregion

    #region Static Factory Methods

    /// <summary>
    /// Creates a MusicSourceSeparator asynchronously, downloading models if needed.
    /// </summary>
    internal static async Task<MusicSourceSeparator<T>> CreateAsync(
        SourceSeparationOptions? options = null,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new SourceSeparationOptions();
        string modelPath = options.ModelPath ?? string.Empty;

        if (string.IsNullOrEmpty(modelPath))
        {
            var downloader = new OnnxModelDownloader();
            var modelRepo = GetModelRepository(options.StemCount);
            modelPath = await downloader.DownloadAsync(
                modelRepo,
                "model.onnx",
                progress: progress,
                cancellationToken);
            options.ModelPath = modelPath;
        }

        return new MusicSourceSeparator<T>(modelPath, options);
    }

    /// <summary>
    /// Creates a MusicSourceSeparator for CPU-based spectral processing without neural network.
    /// </summary>
    public static MusicSourceSeparator<T> CreateCpuOnly(SourceSeparationOptions? options = null)
    {
        return new MusicSourceSeparator<T>(options);
    }

    #endregion

    #region IMusicSourceSeparator Properties

    /// <summary>
    /// Gets the sources this model can separate.
    /// </summary>
    public IReadOnlyList<string> SupportedSources => _options.StemCount switch
    {
        2 => TwoStemSources,
        5 => FiveStemSources,
        _ => StandardSources
    };

    /// <summary>
    /// Gets the number of stems/sources this model produces.
    /// </summary>
    public int NumStems => _options.StemCount;

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the neural network layers.
    /// </summary>
    protected override void InitializeLayers()
    {
        // Use architecture layers if provided
        if (Architecture.Layers is not null && Architecture.Layers.Any())
        {
            foreach (var layer in Architecture.Layers)
            {
                Layers.Add(layer);
            }
            // A caller-supplied layer chain is an ordinary sequential network, not the
            // private Demucs encoder/decoder topology assembled below. Keep the typed
            // Demucs views empty so Predict/Train route through the base Layers walk.
            _demucsDepth = 0;
            return;
        }

        // Faithful waveform-Demucs (Défossez et al. 2019): L conv encoder blocks — Conv1d(kernel,
        // stride)+ReLU then a 1x1 Conv1d that doubles channels feeding a channel-split GLU — an LSTM
        // bottleneck, and L mirrored decoder blocks — a 1x1 Conv1d(2*C)+channel-GLU then a transposed
        // Conv1d(kernel, stride) upsample (ReLU except the final block) — with U-Net skip connections
        // wired in PredictCore. Channels double each encoder level. The final decoder block emits
        // StemCount source channels. Padding keeps encoder and decoder lengths aligned for the
        // skip-add: L -> L/stride -> L/stride^2 and back.
        //
        // FROM THE OPTIONS, WITH THE PAPER'S DEFAULTS. These were `const int depth = 2, baseChannels =
        // 8, ...` with a comment saying the small values "keep the invariant suite fast": a test
        // constraint fixing a production model's entire capacity, at a size (two levels, eight
        // channels) with no ability to separate real music. A caller could configure nothing but
        // StemCount. The paper's six levels at base 64 are the defaults now, and a test that wants a
        // small network asks for one.
        int depth = _options.DemucsDepth;
        int baseChannels = _options.DemucsBaseChannels;
        int kernel = _options.DemucsKernelSize;
        int stride = _options.DemucsStride;
        int padding = _options.DemucsPadding;

        if (depth < 1)
            throw new InvalidOperationException($"DemucsDepth must be at least 1; got {depth}.");
        if (baseChannels < 1)
            throw new InvalidOperationException($"DemucsBaseChannels must be at least 1; got {baseChannels}.");
        if (kernel < 1)
            throw new InvalidOperationException($"DemucsKernelSize must be at least 1; got {kernel}.");
        if (stride < 1)
            throw new InvalidOperationException($"DemucsStride must be at least 1; got {stride}.");
        if (padding < 0)
            throw new InvalidOperationException($"DemucsPadding cannot be negative; got {padding}.");

        _demucsDepth = depth;

        for (int i = 0; i < depth; i++)
        {
            int outCh = baseChannels << i; // base, 2*base, 4*base, ...
            var conv = new Conv1DLayer<T>(outputChannels: outCh, kernelSize: kernel, stride: stride,
                padding: padding, activation: new AiDotNet.ActivationFunctions.ReLUActivation<T>());
            var gate = new Conv1DLayer<T>(outputChannels: outCh * 2, kernelSize: 1, stride: 1,
                padding: 0, activation: new AiDotNet.ActivationFunctions.IdentityActivation<T>());
            _demucsEncConv.Add(conv);
            _demucsEncGate.Add(gate);
            Layers.Add(conv);
            Layers.Add(gate);
        }

        int topCh = baseChannels << (depth - 1); // the deepest encoder level's width
        _demucsBottleneck = new LSTMLayer<T>(hiddenSize: topCh);
        Layers.Add(_demucsBottleneck);

        for (int i = depth - 1; i >= 0; i--)
        {
            int inCh = baseChannels << i;                                         // mirrors the encoder, deepest first
            int outCh = i == 0 ? _options.StemCount : (baseChannels << (i - 1));  // last block emits the stems
            var gate = new Conv1DLayer<T>(outputChannels: inCh * 2, kernelSize: 1, stride: 1,
                padding: 0, activation: new AiDotNet.ActivationFunctions.IdentityActivation<T>());
            var deconv = new Conv1DTransposeLayer<T>(outputChannels: outCh, kernelSize: kernel,
                stride: stride, padding: padding,
                activation: i == 0
                    ? (IActivationFunction<T>)new AiDotNet.ActivationFunctions.IdentityActivation<T>()
                    : new AiDotNet.ActivationFunctions.ReLUActivation<T>());
            _demucsDecGate.Add(gate);
            _demucsDecDeconv.Add(deconv);
            Layers.Add(gate);
            Layers.Add(deconv);
        }
    }

    /// <summary>
    /// Rebuilds the typed views used by the explicit Demucs U-Net forward from the
    /// framework-owned <see cref="NeuralNetworkBase{T}.Layers"/> collection.
    /// </summary>
    /// <remarks>
    /// Deserialization replaces every entry in <c>Layers</c> with a restored layer
    /// instance. Without rebinding these views, the explicit forward continued to use
    /// the fresh random layers created by the constructor and silently ignored all
    /// deserialized/trained weights. Keeping <c>Layers</c> as the single ownership graph
    /// also preserves the standard custom-layers contract: a non-Demucs custom chain is
    /// executed sequentially by the base class.
    /// </remarks>
    private bool TryBindDemucsTopologyFromLayers()
    {
        _demucsEncConv.Clear();
        _demucsEncGate.Clear();
        _demucsDecGate.Clear();
        _demucsDecDeconv.Clear();
        _demucsBottleneck = null;
        _demucsDepth = 0;

        int bottleneckIndex = -1;
        for (int i = 0; i < Layers.Count; i++)
        {
            if (Layers[i] is LSTMLayer<T>)
            {
                if (bottleneckIndex >= 0)
                    return false;
                bottleneckIndex = i;
            }
        }

        if (bottleneckIndex <= 0 || bottleneckIndex % 2 != 0)
            return false;

        int depth = bottleneckIndex / 2;
        if (Layers.Count != depth * 4 + 1)
            return false;

        // Validate the complete shape before publishing any typed references.
        for (int i = 0; i < depth; i++)
        {
            if (Layers[i * 2] is not Conv1DLayer<T>
                || Layers[i * 2 + 1] is not Conv1DLayer<T>
                || Layers[bottleneckIndex + 1 + i * 2] is not Conv1DLayer<T>
                || Layers[bottleneckIndex + 2 + i * 2] is not Conv1DTransposeLayer<T>)
            {
                return false;
            }
        }

        for (int i = 0; i < depth; i++)
        {
            _demucsEncConv.Add((Conv1DLayer<T>)Layers[i * 2]);
            _demucsEncGate.Add((Conv1DLayer<T>)Layers[i * 2 + 1]);
            _demucsDecGate.Add((Conv1DLayer<T>)Layers[bottleneckIndex + 1 + i * 2]);
            _demucsDecDeconv.Add((Conv1DTransposeLayer<T>)Layers[bottleneckIndex + 2 + i * 2]);
        }

        _demucsBottleneck = (LSTMLayer<T>)Layers[bottleneckIndex];
        _demucsDepth = depth;
        return true;
    }

    private bool HasBoundDemucsTopology =>
        _demucsDepth > 0
        && _demucsBottleneck is not null
        && _demucsEncConv.Count == _demucsDepth
        && _demucsEncGate.Count == _demucsDepth
        && _demucsDecGate.Count == _demucsDepth
        && _demucsDecDeconv.Count == _demucsDepth;

    #endregion

    #region IMusicSourceSeparator Methods

    /// <summary>
    /// Separates all sources from the audio mix.
    /// </summary>
    public SourceSeparationResult<T> Separate(Tensor<T> audio)
    {
        ThrowIfDisposed();

        if (IsOnnxMode && OnnxEncoder is not null)
        {
            return SeparateWithModel(audio);
        }
        else if (_useNativeMode && Layers.Count > 0)
        {
            return SeparateWithNativeNetwork(audio);
        }
        else
        {
            return SeparateSpectral(audio);
        }
    }

    /// <summary>
    /// Separates all sources asynchronously.
    /// </summary>
    public Task<SourceSeparationResult<T>> SeparateAsync(
        Tensor<T> audio,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Separate(audio), cancellationToken);
    }

    /// <summary>
    /// Extracts a specific source from the mix.
    /// </summary>
    public Tensor<T> ExtractSource(Tensor<T> audio, string source)
    {
        var result = Separate(audio);
        return result.GetSource(source);
    }

    /// <summary>
    /// Removes a specific source from the mix.
    /// </summary>
    public Tensor<T> RemoveSource(Tensor<T> audio, string source)
    {
        var result = Separate(audio);
        var sources = result.Sources;

        // Sum all sources except the one to remove
        Tensor<T>? output = null;
        foreach (var kvp in sources)
        {
            if (kvp.Key == source) continue;

            if (output is null)
            {
                output = new Tensor<T>(kvp.Value._shape);
                for (int i = 0; i < output.Length; i++)
                {
                    output[i] = kvp.Value[i];
                }
            }
            else
            {
                for (int i = 0; i < output.Length; i++)
                {
                    output[i] = NumOps.Add(output[i], kvp.Value[i]);
                }
            }
        }

        return output ?? new Tensor<T>(audio._shape);
    }

    /// <summary>
    /// Gets the soft mask for a specific source.
    /// </summary>
    public Tensor<T> GetSourceMask(Tensor<T> audio, string source)
    {
        ThrowIfDisposed();

        // Compute STFT
        var stft = _stft.Forward(audio);
        var magnitude = ComputeMagnitude(stft);

        // Prepare input and get all masks
        var modelInput = new Tensor<T>([1, magnitude.Shape[0], magnitude.Shape[1]]);
        for (int t = 0; t < magnitude.Shape[0]; t++)
        {
            for (int f = 0; f < magnitude.Shape[1]; f++)
            {
                modelInput[0, t, f] = magnitude[t, f];
            }
        }

        // Get masks from prediction
        var masks = Predict(modelInput);

        // Find the index of the requested source
        int sourceIndex = SupportedSources.ToList().IndexOf(source);
        if (sourceIndex < 0)
        {
            throw new ArgumentException($"Unknown source: {source}. Supported: {string.Join(", ", SupportedSources)}");
        }

        // Extract mask for the specific source
        var sourceMask = new Tensor<T>([magnitude.Shape[0], magnitude.Shape[1]]);
        int numBins = magnitude.Shape[1];

        for (int t = 0; t < magnitude.Shape[0]; t++)
        {
            for (int f = 0; f < numBins; f++)
            {
                if (masks.Shape.Length >= 4 && sourceIndex < masks.Shape[1])
                {
                    sourceMask[t, f] = masks[0, sourceIndex, t, f];
                }
                else
                {
                    // Fallback: equal mask
                    sourceMask[t, f] = NumOps.FromDouble(1.0 / NumStems);
                }
            }
        }

        return sourceMask;
    }

    /// <summary>
    /// Remixes the separated sources with custom volumes.
    /// </summary>
    public Tensor<T> Remix(SourceSeparationResult<T> separationResult, IReadOnlyDictionary<string, double> sourceVolumes)
    {
        var sources = separationResult.Sources;
        Tensor<T>? output = null;

        foreach (var kvp in sources)
        {
            string sourceName = kvp.Key;
            double volume = sourceVolumes.TryGetValue(sourceName, out var v) ? v : 1.0;

            if (Math.Abs(volume) < 1e-10) continue;

            if (output is null)
            {
                output = new Tensor<T>(kvp.Value._shape);
            }

            for (int i = 0; i < output.Length && i < kvp.Value.Length; i++)
            {
                double val = NumOps.ToDouble(kvp.Value[i]) * volume;
                output[i] = NumOps.Add(output[i], NumOps.FromDouble(val));
            }
        }

        return output ?? new Tensor<T>(separationResult.OriginalMix._shape);
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Predicts source masks from spectrogram magnitude.
    /// </summary>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        if (IsOnnxMode && OnnxEncoder is not null)
        {
            return OnnxEncoder.Run(input);
        }

        // Native mode - use layers
        if (!_useNativeMode || Layers.Count == 0)
        {
            // Return uniform masks as fallback
            return CreateUniformMasks(input);
        }

        // The paper topology needs the explicit skip-connected forward. A user-provided
        // custom layer chain retains the normal framework contract and runs sequentially.
        return HasBoundDemucsTopology ? DemucsForward(input) : base.PredictCore(input);
    }

    // Faithful waveform-Demucs forward with U-Net skip connections. A flat layer walk cannot express
    // the skips (encoder output added to the matching decoder input), so the forward is explicit here.
    private Tensor<T> DemucsForward(Tensor<T> input) => RunDemucs(input, activations: null);

    /// <summary>
    /// The one waveform-Demucs forward pass, optionally recording each stage into
    /// <paramref name="activations"/>.
    /// </summary>
    /// <param name="input">Any-rank tensor holding a mono waveform; it is flattened to its total length.</param>
    /// <param name="activations">When non-null, receives a clone of each stage's output.</param>
    /// <returns>The separated stems as <c>[StemCount, length]</c>.</returns>
    /// <remarks>
    /// ONE COPY, because there were two. <c>DemucsForward</c> and
    /// <see cref="GetNamedLayerActivations"/> restated this body almost token for token -- same
    /// reshape, same encoder loop, same permute-LSTM-permute bottleneck, same null check with the same
    /// comment, same decoder loop -- differing only in the <c>activations[...] = x.Clone()</c> lines.
    /// Two copies of one forward pass diverge: any fix to the shapes or padding had to be made twice
    /// or the captured activations would describe a network that no longer matched the trained one.
    /// </remarks>
    private Tensor<T> RunDemucs(
        Tensor<T> input,
        System.Collections.Generic.Dictionary<string, Tensor<T>>? activations)
    {
        var eng = Engine;

        // Treat the input as a mono waveform of the given total length: [batch=1, channels=1, length].
        int total = 1;
        for (int d = 0; d < input.Rank; d++) total *= input.Shape[d];

        // Each encoder level divides the time axis by the stride, so a waveform shorter than
        // stride^depth runs out of samples partway down and the bottleneck sees a zero-length
        // sequence. Checked here, once, against the depth actually built -- the alternative is a
        // shape error thrown from inside whichever conv happens to be the one that runs dry.
        long minimumLength = 1;
        for (int i = 0; i < _demucsDepth; i++) minimumLength *= _options.DemucsStride;
        if (total < minimumLength)
        {
            throw new ArgumentException(
                $"The waveform has {total} samples, but a {_demucsDepth}-level Demucs stack at stride " +
                $"{_options.DemucsStride} needs at least {minimumLength}. Supply longer audio, or build " +
                "the model with a smaller SourceSeparationOptions.DemucsDepth.",
                nameof(input));
        }

        var x = eng.Reshape(input, new[] { 1, 1, total });

        var skips = new System.Collections.Generic.List<Tensor<T>>(_demucsDepth);
        for (int i = 0; i < _demucsDepth; i++)
        {
            x = _demucsEncConv[i].Forward(x);        // Conv1d(k8,s4) + ReLU -> [1, C, L/4]
            var gated = _demucsEncGate[i].Forward(x); // 1x1 Conv1d -> [1, 2C, L/4]
            x = ChannelGlu(gated);                    // channel-split GLU -> [1, C, L/4]
            if (activations is not null) activations[$"Encoder_{i}"] = x.Clone();
            skips.Add(x);
        }

        // LSTM bottleneck over the time axis: [1, C, T] -> [1, T, C] -> LSTM -> [1, T, C] -> [1, C, T].
        var overTime = eng.TensorPermute(x, new[] { 0, 2, 1 });
        // Non-null whenever the Demucs stack is bound, which is the only path that reaches here;
        // assert the invariant rather than suppress it, so a binding regression names itself.
        if (_demucsBottleneck is null)
            throw new InvalidOperationException("The Demucs LSTM bottleneck has not been bound.");
        overTime = _demucsBottleneck.Forward(overTime);
        x = eng.TensorPermute(overTime, new[] { 0, 2, 1 });
        if (activations is not null) activations["Bottleneck"] = x.Clone();

        for (int i = 0; i < _demucsDepth; i++)
        {
            x = eng.TensorAdd(x, skips[_demucsDepth - 1 - i]); // U-Net skip add
            var gated = _demucsDecGate[i].Forward(x);          // 1x1 Conv1d -> [1, 2C, T]
            x = ChannelGlu(gated);                             // channel GLU -> [1, C, T]
            x = _demucsDecDeconv[i].Forward(x);                // transposed Conv1d(k8,s4) upsample
            if (activations is not null) activations[$"Decoder_{i}"] = x.Clone();
        }

        // Drop the synthetic batch axis: [1, StemCount, length] -> [StemCount, length].
        return eng.Reshape(x, new[] { x.Shape[1], x.Shape[2] });
    }

    // Demucs channel-split Gated Linear Unit: split a [1, 2C, T] feature map along the channel axis
    // into (a, b) and return a * sigmoid(b) -> [1, C, T].
    private Tensor<T> ChannelGlu(Tensor<T> gated)
    {
        var eng = Engine;
        int halfC = gated.Shape[1] / 2;
        var a = eng.TensorNarrow(gated, 1, 0, halfC);
        var b = eng.TensorNarrow(gated, 1, halfC, halfC);
        return eng.TensorMultiply(a, eng.Sigmoid(b));
    }

    /// <summary>
    /// Captures per-stage activations of the waveform-Demucs forward pass. A flat <c>Layers</c> walk
    /// cannot express the encoder/bottleneck/decoder structure with its skip connections, so the
    /// activations are captured explicitly here, mirroring <see cref="DemucsForward"/>.
    /// </summary>
    public override System.Collections.Generic.Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));

        var activations = new System.Collections.Generic.Dictionary<string, Tensor<T>>();
        if (!_useNativeMode || Layers.Count == 0)
        {
            return activations;
        }

        if (!HasBoundDemucsTopology)
        {
            return base.GetNamedLayerActivations(input);
        }

        // The SAME forward Predict and Train run, asked to record as it goes -- see RunDemucs. The
        // activations cannot describe a different network than the one being trained if there is only
        // one network to describe.
        RunDemucs(input, activations);
        return activations;
    }

    /// <summary>
    /// Training forward pass. Routes through the SAME waveform-Demucs forward (with the U-Net skip
    /// connections) as <see cref="PredictCore"/>, NOT the base flat <c>Layers</c> walk — which would
    /// feed the raw input straight into the first Conv1d (wrong rank) and cannot express the skips.
    /// </summary>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        EnsureLayerRandomSeedsWired();
        return HasBoundDemucsTopology ? DemucsForward(input) : base.ForwardForTraining(input);
    }

    /// <summary>
    /// Trains the model on mixed audio and ground truth stems.
    /// </summary>
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (!_useNativeMode)
        {
            throw new InvalidOperationException(
                "Training is not supported in ONNX inference mode. " +
                "Create the model with NeuralNetworkArchitecture for training.");
        }

        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expected, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    /// <summary>
    /// Preprocesses raw audio into spectrogram format.
    /// </summary>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        var stft = _stft.Forward(rawAudio);
        return ComputeMagnitude(stft);
    }

    /// <summary>
    /// Postprocesses model output (applies sigmoid to mask values).
    /// </summary>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        var result = new Tensor<T>(modelOutput._shape);
        for (int i = 0; i < modelOutput.Length; i++)
        {
            double val = NumOps.ToDouble(modelOutput[i]);
            // Sigmoid to ensure mask values are in [0, 1]
            result[i] = NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-val)));
        }
        return result;
    }

    /// <summary>
    /// Gets model metadata for serialization.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "MusicSourceSeparator-Native" : "MusicSourceSeparator-ONNX",
            Description = "Music source separation model (Spleeter/Demucs-style)",
            FeatureCount = _options.FftSize / 2 + 1,
            Complexity = 1
        };
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString();
        metadata.AdditionalInfo["FftSize"] = _options.FftSize.ToString();
        metadata.AdditionalInfo["StemCount"] = _options.StemCount.ToString();
        return metadata;
    }

    /// <summary>
    /// Serializes network-specific data.
    /// </summary>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_options.SampleRate);
        writer.Write(_options.FftSize);
        writer.Write(_options.HopLength);
        writer.Write(_options.StemCount);
        writer.Write(_options.HpssKernelSize);
        writer.Write(_useNativeMode);
    }

    /// <summary>
    /// Deserializes network-specific data — the inverse of
    /// <see cref="SerializeNetworkSpecificData"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <see cref="MusicSourceSeparationOptions"/> properties are init-only and
    /// only accept values at construction; we cannot rebind <c>_options</c>
    /// here. Instead, read every field the serializer wrote and verify it
    /// matches the in-memory configuration. A mismatch indicates the model
    /// being loaded was trained with a different STFT / stem layout, which
    /// would silently corrupt every separation if accepted unchanged. Throw
    /// with the offending field so the caller can reconstruct the separator
    /// with the right options.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        int sampleRate = reader.ReadInt32();
        int fftSize = reader.ReadInt32();
        int hopLength = reader.ReadInt32();
        int stemCount = reader.ReadInt32();
        int hpssKernelSize = reader.ReadInt32();
        bool useNativeMode = reader.ReadBoolean();

        if (sampleRate != _options.SampleRate)
            throw new InvalidOperationException(
                $"Deserialized SampleRate ({sampleRate}) does not match constructor option ({_options.SampleRate}). " +
                "Reconstruct MusicSourceSeparator with matching options before loading this model.");
        if (fftSize != _options.FftSize)
            throw new InvalidOperationException(
                $"Deserialized FftSize ({fftSize}) does not match constructor option ({_options.FftSize}).");
        if (hopLength != _options.HopLength)
            throw new InvalidOperationException(
                $"Deserialized HopLength ({hopLength}) does not match constructor option ({_options.HopLength}).");
        if (stemCount != _options.StemCount)
            throw new InvalidOperationException(
                $"Deserialized StemCount ({stemCount}) does not match constructor option ({_options.StemCount}).");
        if (hpssKernelSize != _options.HpssKernelSize)
            throw new InvalidOperationException(
                $"Deserialized HpssKernelSize ({hpssKernelSize}) does not match constructor option ({_options.HpssKernelSize}).");

        _useNativeMode = useNativeMode;

        // The base deserializer has just replaced Layers with the restored instances.
        // Rebind the explicit Demucs forward to those instances so inference and
        // training consume the restored weights rather than constructor-fresh layers.
        // THE RETURN VALUE DECIDES WHETHER THE MODEL IS USABLE, so discarding it defeated the point
        // of returning it. On failure the method has already cleared every typed list and set
        // _demucsDepth to 0, so HasBoundDemucsTopology goes false, PredictCore silently routes to
        // base.PredictCore, and the caller gets separations from a generic forward pass rather than
        // Demucs -- from a model they just deserialized and have every reason to believe is intact.
        if (_useNativeMode && !TryBindDemucsTopologyFromLayers())
        {
            throw new InvalidOperationException(
                "Deserialization restored the layer list, but it does not match the Demucs topology, so " +
                "the explicit Demucs forward could not be rebound. Continuing would silently fall back " +
                "to a generic forward pass and return separations that are not Demucs's.");
        }
    }

    #endregion

    #region Separation Methods

    private SourceSeparationResult<T> SeparateWithModel(Tensor<T> audio)
    {
        if (OnnxEncoder is null)
            throw new InvalidOperationException("Model not loaded.");

        // Compute STFT of input
        var stft = _stft.Forward(audio);
        var magnitude = ComputeMagnitude(stft);
        var phase = ComputePhase(stft);

        // Prepare input for model
        var modelInput = new Tensor<T>([1, magnitude.Shape[0], magnitude.Shape[1]]);
        for (int t = 0; t < magnitude.Shape[0]; t++)
        {
            for (int f = 0; f < magnitude.Shape[1]; f++)
            {
                modelInput[0, t, f] = magnitude[t, f];
            }
        }

        // Run model to get masks
        var masks = OnnxEncoder.Run(modelInput);

        // Apply masks and reconstruct
        return ApplyMasksAndReconstruct(audio, stft, magnitude, phase, masks);
    }

    private SourceSeparationResult<T> SeparateWithNativeNetwork(Tensor<T> audio)
    {
        // DEMUCS IS A WAVEFORM MODEL, so it does not belong in the spectrogram-masking pipeline below.
        // Its forward returns [StemCount, length] -- the separated audio itself, not masks over an STFT
        // magnitude. Routing it through the mask path fed a rank-2 tensor to ApplyMasksAndReconstruct,
        // whose branches test for rank >= 4 and rank >= 3; both fail, `mask` stays at its 0 initializer,
        // and EVERY stem reconstructs as silence. Nothing throws, and the result object is well-formed.
        if (HasBoundDemucsTopology)
        {
            return SeparateWithDemucs(audio);
        }

        var stft = _stft.Forward(audio);
        var magnitude = ComputeMagnitude(stft);
        var phase = ComputePhase(stft);

        var modelInput = new Tensor<T>([1, magnitude.Shape[0], magnitude.Shape[1]]);
        for (int t = 0; t < magnitude.Shape[0]; t++)
        {
            for (int f = 0; f < magnitude.Shape[1]; f++)
            {
                modelInput[0, t, f] = magnitude[t, f];
            }
        }

        var masks = Predict(modelInput);
        masks = PostprocessOutput(masks);

        return ApplyMasksAndReconstruct(audio, stft, magnitude, phase, masks);
    }

    /// <summary>
    /// Separates in the waveform domain, the way Demucs is defined (Defossez et al. 2019): the network
    /// maps the mixture waveform straight to one waveform per stem, with no STFT anywhere in the path.
    /// </summary>
    private SourceSeparationResult<T> SeparateWithDemucs(Tensor<T> audio)
    {
        // [StemCount, length]. The row order matches SupportedSources, which is the order the final
        // decoder block's output channels were trained against.
        var stems = DemucsForward(audio);

        int stemRows = stems.Shape[0];
        int stemLength = stems.Shape[1];
        var sourceNames = SupportedSources;

        var sources = new Dictionary<string, Tensor<T>>();
        for (int stem = 0; stem < NumStems && stem < sourceNames.Count; stem++)
        {
            var waveform = new Tensor<T>([stemLength]);
            if (stem < stemRows)
            {
                for (int i = 0; i < stemLength; i++)
                {
                    waveform[i] = stems[stem, i];
                }
            }
            else
            {
                // Fewer output channels than named stems means the decoder was built for a different
                // StemCount than the model is being asked for. Silence would look like a separation
                // that simply found nothing in this stem, which is the failure this method exists to
                // stop being silent about.
                throw new InvalidOperationException(
                    $"The Demucs decoder emits {stemRows} stem channels but {NumStems} stems were " +
                    $"requested. Rebuild the model with StemCount = {NumStems}.");
            }

            sources[sourceNames[stem]] = waveform;
        }

        return new SourceSeparationResult<T>
        {
            Sources = sources,
            OriginalMix = audio,
            SampleRate = _options.SampleRate,
            Duration = (double)audio.Length / _options.SampleRate
        };
    }

    private SourceSeparationResult<T> SeparateSpectral(Tensor<T> audio)
    {
        var stft = _stft.Forward(audio);
        var magnitude = ComputeMagnitude(stft);
        var phase = ComputePhase(stft);

        // Perform HPSS
        var (harmonicMag, percussiveMag) = HarmonicPercussiveSeparation(magnitude);

        // Reconstruct signals
        var harmonic = ReconstructFromMagnitudePhase(harmonicMag, phase);
        var percussive = ReconstructFromMagnitudePhase(percussiveMag, phase);

        Tensor<T> vocals, other;
        if (_options.StemCount >= 4)
        {
            (vocals, other) = SeparateVocals(harmonicMag, phase);
        }
        else
        {
            vocals = harmonic;
            other = new Tensor<T>(audio._shape);
        }

        var bass = ExtractBassline(harmonicMag, phase);

        var sources = new Dictionary<string, Tensor<T>>
        {
            ["vocals"] = vocals,
            ["drums"] = percussive,
            ["bass"] = bass,
            ["other"] = other
        };

        return new SourceSeparationResult<T>
        {
            Sources = sources,
            OriginalMix = audio,
            SampleRate = _options.SampleRate,
            Duration = (double)audio.Length / _options.SampleRate
        };
    }

    private SourceSeparationResult<T> ApplyMasksAndReconstruct(
        Tensor<T> audio,
        Tensor<Complex<T>> stft,
        Tensor<T> magnitude,
        Tensor<T> phase,
        Tensor<T> masks)
    {
        int numFrames = stft.Shape[0];
        int numBins = stft.Shape[1];

        var sources = new Dictionary<string, Tensor<T>>();
        var sourceNames = SupportedSources;

        for (int stem = 0; stem < NumStems && stem < sourceNames.Count; stem++)
        {
            var stemMag = new Tensor<T>([numFrames, numBins]);

            for (int t = 0; t < numFrames; t++)
            {
                for (int f = 0; f < numBins; f++)
                {
                    double mag = NumOps.ToDouble(magnitude[t, f]);
                    double mask = 0;

                    if (masks.Shape.Length >= 4 && stem < masks.Shape[1] && t < masks.Shape[2] && f < masks.Shape[3])
                    {
                        mask = NumOps.ToDouble(masks[0, stem, t, f]);
                    }
                    else if (masks.Shape.Length >= 3 && stem < masks.Shape[0])
                    {
                        mask = NumOps.ToDouble(masks[stem, t, f]);
                    }

                    stemMag[t, f] = NumOps.FromDouble(mag * Math.Max(0.0, Math.Min(1.0, mask)));
                }
            }

            sources[sourceNames[stem]] = ReconstructFromMagnitudePhase(stemMag, phase);
        }

        return new SourceSeparationResult<T>
        {
            Sources = sources,
            OriginalMix = audio,
            SampleRate = _options.SampleRate,
            Duration = (double)audio.Length / _options.SampleRate
        };
    }

    #endregion

    #region Signal Processing Helpers

    private Tensor<T> ComputeMagnitude(Tensor<Complex<T>> stft)
    {
        int numFrames = stft.Shape[0];
        int numBins = stft.Shape[1];
        var magnitude = new Tensor<T>([numFrames, numBins]);

        for (int t = 0; t < numFrames; t++)
        {
            for (int f = 0; f < numBins; f++)
            {
                var complex = stft[t, f];
                double real = NumOps.ToDouble(complex.Real);
                double imag = NumOps.ToDouble(complex.Imaginary);
                magnitude[t, f] = NumOps.FromDouble(Math.Sqrt(real * real + imag * imag));
            }
        }

        return magnitude;
    }

    private Tensor<T> ComputePhase(Tensor<Complex<T>> stft)
    {
        int numFrames = stft.Shape[0];
        int numBins = stft.Shape[1];
        var phase = new Tensor<T>([numFrames, numBins]);

        for (int t = 0; t < numFrames; t++)
        {
            for (int f = 0; f < numBins; f++)
            {
                var complex = stft[t, f];
                double real = NumOps.ToDouble(complex.Real);
                double imag = NumOps.ToDouble(complex.Imaginary);
                phase[t, f] = NumOps.FromDouble(Math.Atan2(imag, real));
            }
        }

        return phase;
    }

    private (Tensor<T> harmonic, Tensor<T> percussive) HarmonicPercussiveSeparation(Tensor<T> magnitude)
    {
        int numFrames = magnitude.Shape[0];
        int numBins = magnitude.Shape[1];
        int kernelSize = _options.HpssKernelSize;

        var harmonicEnhanced = MedianFilterTime(magnitude, kernelSize);
        var percussiveEnhanced = MedianFilterFrequency(magnitude, kernelSize);

        var harmonicMag = new Tensor<T>([numFrames, numBins]);
        var percussiveMag = new Tensor<T>([numFrames, numBins]);

        for (int t = 0; t < numFrames; t++)
        {
            for (int f = 0; f < numBins; f++)
            {
                double h = NumOps.ToDouble(harmonicEnhanced[t, f]);
                double p = NumOps.ToDouble(percussiveEnhanced[t, f]);
                double m = NumOps.ToDouble(magnitude[t, f]);

                double sum = h + p + 1e-10;
                harmonicMag[t, f] = NumOps.FromDouble(m * h / sum);
                percussiveMag[t, f] = NumOps.FromDouble(m * p / sum);
            }
        }

        return (harmonicMag, percussiveMag);
    }

    private Tensor<T> MedianFilterTime(Tensor<T> input, int kernelSize)
    {
        int numFrames = input.Shape[0];
        int numBins = input.Shape[1];
        var output = new Tensor<T>([numFrames, numBins]);
        int halfKernel = kernelSize / 2;
        int windowSize = (2 * halfKernel) + 1;

        for (int f = 0; f < numBins; f++)
        {
            var window = new double[windowSize];

            for (int t = 0; t < numFrames; t++)
            {
                int count = 0;
                for (int k = -halfKernel; k <= halfKernel; k++)
                {
                    int ti = Math.Max(0, Math.Min(numFrames - 1, t + k));
                    window[count++] = NumOps.ToDouble(input[ti, f]);
                }

                Array.Sort(window, 0, count);
                output[t, f] = NumOps.FromDouble(window[count / 2]);
            }
        }

        return output;
    }

    private Tensor<T> MedianFilterFrequency(Tensor<T> input, int kernelSize)
    {
        int numFrames = input.Shape[0];
        int numBins = input.Shape[1];
        var output = new Tensor<T>([numFrames, numBins]);
        int halfKernel = kernelSize / 2;
        int windowSize = (2 * halfKernel) + 1;

        for (int t = 0; t < numFrames; t++)
        {
            var window = new double[windowSize];

            for (int f = 0; f < numBins; f++)
            {
                int count = 0;
                for (int k = -halfKernel; k <= halfKernel; k++)
                {
                    int fi = Math.Max(0, Math.Min(numBins - 1, f + k));
                    window[count++] = NumOps.ToDouble(input[t, fi]);
                }

                Array.Sort(window, 0, count);
                output[t, f] = NumOps.FromDouble(window[count / 2]);
            }
        }

        return output;
    }

    private Tensor<T> ReconstructFromMagnitudePhase(Tensor<T> magnitude, Tensor<T> phase)
    {
        int numFrames = magnitude.Shape[0];
        int numBins = magnitude.Shape[1];

        var stft = new Tensor<Complex<T>>([numFrames, numBins]);
        for (int t = 0; t < numFrames; t++)
        {
            for (int f = 0; f < numBins; f++)
            {
                double mag = NumOps.ToDouble(magnitude[t, f]);
                double ph = NumOps.ToDouble(phase[t, f]);
                T real = NumOps.FromDouble(mag * Math.Cos(ph));
                T imag = NumOps.FromDouble(mag * Math.Sin(ph));
                stft[t, f] = new Complex<T>(real, imag);
            }
        }

        return _stft.Inverse(stft);
    }

    private (Tensor<T> vocals, Tensor<T> other) SeparateVocals(Tensor<T> harmonicMag, Tensor<T> phase)
    {
        int numFrames = harmonicMag.Shape[0];
        int numBins = harmonicMag.Shape[1];

        double vocalLowBin = 300.0 * _options.FftSize / _options.SampleRate;
        double vocalHighBin = 4000.0 * _options.FftSize / _options.SampleRate;

        var vocalMag = new Tensor<T>([numFrames, numBins]);
        var otherMag = new Tensor<T>([numFrames, numBins]);

        for (int t = 0; t < numFrames; t++)
        {
            for (int f = 0; f < numBins; f++)
            {
                double mag = NumOps.ToDouble(harmonicMag[t, f]);
                double vocalWeight = 0;

                if (f >= vocalLowBin && f <= vocalHighBin)
                    vocalWeight = 0.7;
                else if (f < vocalLowBin && f >= vocalLowBin - 50)
                    vocalWeight = 0.3 * (f - (vocalLowBin - 50)) / 50;
                else if (f > vocalHighBin && f <= vocalHighBin + 100)
                    vocalWeight = 0.3 * (1 - (f - vocalHighBin) / 100);

                vocalMag[t, f] = NumOps.FromDouble(mag * vocalWeight);
                otherMag[t, f] = NumOps.FromDouble(mag * (1 - vocalWeight));
            }
        }

        return (ReconstructFromMagnitudePhase(vocalMag, phase), ReconstructFromMagnitudePhase(otherMag, phase));
    }

    private Tensor<T> ExtractBassline(Tensor<T> harmonicMag, Tensor<T> phase)
    {
        int numFrames = harmonicMag.Shape[0];
        int numBins = harmonicMag.Shape[1];
        double bassMaxBin = 250.0 * _options.FftSize / _options.SampleRate;

        var bassMag = new Tensor<T>([numFrames, numBins]);

        for (int t = 0; t < numFrames; t++)
        {
            for (int f = 0; f < numBins; f++)
            {
                double mag = NumOps.ToDouble(harmonicMag[t, f]);
                double bassWeight = f <= bassMaxBin ? 1.0 : 0.0;
                if (f > bassMaxBin && f <= bassMaxBin + 20)
                    bassWeight = 1.0 - (f - bassMaxBin) / 20;

                bassMag[t, f] = NumOps.FromDouble(mag * bassWeight);
            }
        }

        return ReconstructFromMagnitudePhase(bassMag, phase);
    }

    private Tensor<T> CreateUniformMasks(Tensor<T> input)
    {
        // Create uniform masks for fallback
        int numFrames = input.Shape.Length > 2 ? input.Shape[1] : input.Shape[0];
        int numBins = input.Shape.Length > 2 ? input.Shape[2] : input.Shape[1];

        var masks = new Tensor<T>([1, NumStems, numFrames, numBins]);
        double uniformValue = 1.0 / NumStems;

        for (int stem = 0; stem < NumStems; stem++)
        {
            for (int t = 0; t < numFrames; t++)
            {
                for (int f = 0; f < numBins; f++)
                {
                    masks[0, stem, t, f] = NumOps.FromDouble(uniformValue);
                }
            }
        }

        return masks;
    }

    private static string GetModelRepository(int stemCount)
    {
        return stemCount switch
        {
            2 => "deezer/spleeter-2stems-onnx",
            4 => "deezer/spleeter-4stems-onnx",
            5 => "deezer/spleeter-5stems-onnx",
            _ => "deezer/spleeter-4stems-onnx"
        };
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(GetType().FullName ?? nameof(MusicSourceSeparator<T>));
        }
    }

    /// <summary>
    /// Disposes of managed resources.
    /// </summary>
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;

        if (disposing)
        {
            OnnxEncoder?.Dispose();
        }

        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}

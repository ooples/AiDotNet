using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.Effects;

/// <summary>
/// Audio Super-Resolution model for upsampling low-resolution audio to high-resolution
/// (Kuleshov et al., 2017; Li et al., 2021).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Audio Super-Resolution uses deep neural networks to predict missing high-frequency
/// content in low-resolution audio. Given input at a low sample rate (e.g., 8 kHz telephone
/// quality), it reconstructs audio at a higher sample rate (e.g., 44.1 kHz studio quality)
/// by predicting the missing frequency bands. The architecture uses residual blocks with
/// attention modules to capture both local and global spectral patterns.
/// </para>
/// <para>
/// <b>For Beginners:</b> Audio Super-Resolution is like AI-powered upscaling for sound.
/// Just as image super-resolution makes blurry photos sharper, this model makes low-quality
/// audio sound clearer and more detailed.
///
/// Common uses:
/// - Upscaling old telephone recordings (8 kHz to 44.1 kHz)
/// - Recovering quality from heavily compressed audio (MP3 at 64 kbps)
/// - Enhancing voice recordings from cheap microphones
/// - Restoring bandwidth-limited historical recordings
///
/// How it works:
/// 1. Takes a low-resolution audio waveform as input
/// 2. Passes through residual blocks that learn to predict missing high-frequency content
/// 3. Outputs a high-resolution waveform with restored detail
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 1, outputSize: 1);
/// var model = new AudioSuperResolution&lt;float&gt;(arch, "audio_sr.onnx");
/// var highRes = model.Enhance(lowResAudio);
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.SuperResolution)]
[ModelTask(ModelTask.Restoration)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Audio Super Resolution using Neural Networks", "https://arxiv.org/abs/1708.00853", Year = 2017, Authors = "Volodymyr Kuleshov, S. Zayd Enam, Stefano Ermon")]
public class AudioSuperResolution<T> : AudioNeuralNetworkBase<T>, IAudioEnhancer<T>
{
    #region Fields

    private readonly AudioSuperResolutionOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    // Conv-U-Net block references (Kuleshov et al. 2017), re-derived from Layers each forward so
    // they survive clone/deserialize. Null when Layers came from a custom Architecture.Layers list,
    // in which case the forward falls back to a plain sequential pass.
    private int _numBlocks;
    private ILayer<T>[]? _downBlocks;
    private ILayer<T>? _bottleneck;
    private ILayer<T>[]? _upBlocks;
    private ILayer<T>? _finalConv;

    #endregion

    #region IAudioEnhancer Properties

    /// <inheritdoc />
    public int NumChannels { get; } = 1;

    /// <inheritdoc />
    public double EnhancementStrength { get; set; } = 1.0;

    /// <inheritdoc />
    public int LatencySamples => _options.InputSampleRate; // one second of input latency

    #endregion

    #region Constructors

    /// <summary>Creates an Audio Super-Resolution model in ONNX inference mode.</summary>
    public AudioSuperResolution(NeuralNetworkArchitecture<T> architecture, string modelPath, AudioSuperResolutionOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options = options ?? new AudioSuperResolutionOptions();
        _useNativeMode = false;
        base.SampleRate = _options.OutputSampleRate;
        _options.ModelPath = modelPath;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates an Audio Super-Resolution model in native training mode.</summary>
    public AudioSuperResolution(NeuralNetworkArchitecture<T> architecture, AudioSuperResolutionOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new AudioSuperResolutionOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.OutputSampleRate;
        InitializeLayers();
    }

    internal static async Task<AudioSuperResolution<T>> CreateAsync(AudioSuperResolutionOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new AudioSuperResolutionOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("audio_super_resolution", $"audio_sr_{options.Variant}.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: 1, outputSize: 1);
        return new AudioSuperResolution<T>(arch, mp, options);
    }

    #endregion

    #region IAudioEnhancer

    /// <inheritdoc />
    public Tensor<T> Enhance(Tensor<T> audio)
    {
        ThrowIfDisposed();
        var features = PreprocessAudio(audio);
        var output = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);
        return ApplyStrength(audio, output);
    }

    /// <inheritdoc />
    public Tensor<T> EnhanceWithReference(Tensor<T> audio, Tensor<T> reference)
    {
        // For super-resolution, the reference can serve as a quality target but
        // the core upsampling doesn't change. Apply the same enhancement.
        return Enhance(audio);
    }

    /// <inheritdoc />
    public Tensor<T> ProcessChunk(Tensor<T> audioChunk) => Enhance(audioChunk);

    /// <inheritdoc />
    public void EstimateNoiseProfile(Tensor<T> noiseOnlyAudio)
    {
        // Super-resolution focuses on bandwidth extension, not noise reduction.
        // This method is intentionally minimal as noise profiling is not applicable.
    }

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        _numBlocks = _options.NumResBlocks;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultAudioSuperResolutionLayers(
            numBlocks: _options.NumResBlocks, channels: _options.HiddenDim, kernelSize: 9));
        ExtractLayerReferences();
    }

    /// <summary>
    /// Groups the flat <see cref="NeuralNetworkBase{T}.Layers"/> list into the conv-U-Net's
    /// downsampling / bottleneck / upsampling / final blocks so the forward pass can wire the
    /// symmetric skip connections and additive residual. Re-derived from Layers (not cached at
    /// construction) so it stays valid after clone/deserialize repopulate the layer list.
    /// </summary>
    private void ExtractLayerReferences()
    {
        // Helper layout: [down × numBlocks, bottleneck, up × numBlocks, final] = 2*numBlocks + 2.
        if (_numBlocks <= 0 || Layers.Count != 2 * _numBlocks + 2)
        {
            _downBlocks = null; _upBlocks = null; _bottleneck = null; _finalConv = null;
            return;
        }
        _downBlocks = new ILayer<T>[_numBlocks];
        _upBlocks = new ILayer<T>[_numBlocks];
        for (int b = 0; b < _numBlocks; b++) _downBlocks[b] = Layers[b];
        _bottleneck = Layers[_numBlocks];
        for (int b = 0; b < _numBlocks; b++) _upBlocks[b] = Layers[_numBlocks + 1 + b];
        _finalConv = Layers[2 * _numBlocks + 1];
    }

    /// <summary>
    /// 1-D sub-pixel (dimension-shuffle) upsampling: <c>[b, 2C, L] → [b, C, 2L]</c>, doubling the
    /// temporal resolution while halving the channels (Kuleshov et al. 2017 §2). Built from
    /// tape-aware <see cref="NeuralNetworkBase{T}.Engine"/> reshape/permute so gradients flow.
    /// </summary>
    private Tensor<T> SubPixelShuffle1D(Tensor<T> x)
    {
        int batch = x.Shape[0], twoC = x.Shape[1], len = x.Shape[2];
        int c = twoC / 2;
        var r = Engine.Reshape(x, new[] { batch, c, 2, len });   // split channels into (C, 2)
        r = Engine.TensorPermute(r, new[] { 0, 1, 3, 2 });        // -> [b, C, L, 2]
        return Engine.Reshape(r, new[] { batch, c, len * 2 });    // interleave -> [b, C, 2L]
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        return RunUNet(input);
    }

    /// <summary>
    /// Runs the Kuleshov et al. 2017 conv U-Net: downsampling conv blocks (storing features for the
    /// skip connections), a bottleneck, upsampling blocks (each: conv → sub-pixel ×2 → concatenate
    /// the symmetric downsampling features), a final conv sub-pixelled to one channel, and an
    /// ADDITIVE residual with the input so the network learns <c>y − x</c>. All non-conv steps use
    /// tape-aware Engine ops so the same path serves inference and training.
    /// </summary>
    private Tensor<T> RunUNet(Tensor<T> input)
    {
        ExtractLayerReferences();
        // Custom / non-default layer list (Architecture.Layers supplied): no U-Net structure to
        // wire — run the provided layers sequentially, as before.
        if (_downBlocks is null || _upBlocks is null || _bottleneck is null || _finalConv is null)
        {
            var seq = input;
            foreach (var l in Layers) seq = l.Forward(seq);
            return seq;
        }

        int len = input.Shape[^1];
        var x = Engine.Reshape(input, new[] { 1, 1, len });    // [batch=1, channels=1, L]
        var skips = new Tensor<T>[_numBlocks];

        var h = x;
        for (int b = 0; b < _numBlocks; b++) { h = _downBlocks[b].Forward(h); skips[b] = h; }
        h = _bottleneck.Forward(h);
        for (int b = 0; b < _numBlocks; b++)
        {
            h = _upBlocks[b].Forward(h);                 // conv -> 2C channels
            h = SubPixelShuffle1D(h);                    // -> C channels, 2× length
            // Stacking (concatenation) skip with the symmetric downsampling block, on the channel axis.
            h = Engine.TensorConcatenate(new[] { h, skips[_numBlocks - 1 - b] }, axis: 1);
        }
        h = _finalConv.Forward(h);                       // conv -> 2 channels
        h = SubPixelShuffle1D(h);                        // -> 1 channel, full length L
        h = Engine.TensorAdd(h, x);                      // additive input residual (learn y - x)
        return Engine.Reshape(h, input._shape);
    }

    /// <summary>Training forward — the same tape-aware U-Net path as inference.</summary>
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => RunUNet(input);

    /// <summary>
    /// Collects per-block activations along the actual U-Net dataflow. The base implementation runs
    /// <see cref="NeuralNetworkBase{T}.Layers"/> as a flat sequence, which is invalid here (the
    /// conv blocks need a [1, 1, L] tensor and the up/skip/sub-pixel steps are non-sequential), so it
    /// is overridden to walk the same path as <see cref="RunUNet"/>.
    /// </summary>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        ThrowIfDisposed();
        ExtractLayerReferences();
        if (_downBlocks is null || _upBlocks is null || _bottleneck is null || _finalConv is null)
            return base.GetNamedLayerActivations(input);

        var acts = new Dictionary<string, Tensor<T>>();
        int len = input.Shape[^1];
        var x = Engine.Reshape(input, new[] { 1, 1, len });
        var skips = new Tensor<T>[_numBlocks];
        var h = x;
        for (int b = 0; b < _numBlocks; b++)
        {
            h = _downBlocks[b].Forward(h);
            skips[b] = h;
            acts[$"Down_{b}_{_downBlocks[b].GetType().Name}"] = h.Clone();
        }
        h = _bottleneck.Forward(h);
        acts[$"Bottleneck_{_bottleneck.GetType().Name}"] = h.Clone();
        for (int b = 0; b < _numBlocks; b++)
        {
            h = _upBlocks[b].Forward(h);
            h = SubPixelShuffle1D(h);
            h = Engine.TensorConcatenate(new[] { h, skips[_numBlocks - 1 - b] }, axis: 1);
            acts[$"Up_{b}_{_upBlocks[b].GetType().Name}"] = h.Clone();
        }
        h = _finalConv.Forward(h);
        h = SubPixelShuffle1D(h);
        acts[$"Final_{_finalConv.GetType().Name}"] = Engine.TensorAdd(h, x).Clone();
        return acts;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode.");
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

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("ONNX mode.");
        int idx = 0; foreach (var l in Layers) { int c = (int)l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; }
    }

    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => rawAudio;
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "AudioSuperResolution-Native" : "AudioSuperResolution-ONNX",
            Description = $"Audio Super-Resolution {_options.Variant} ({_options.InputSampleRate / 1000}kHz -> {_options.OutputSampleRate / 1000}kHz)",
        };
        m.AdditionalInfo["Variant"] = _options.Variant;
        m.AdditionalInfo["UpsampleFactor"] = _options.UpsampleFactor.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.InputSampleRate); w.Write(_options.OutputSampleRate);
        w.Write(_options.UpsampleFactor); w.Write(_options.Variant);
        w.Write(_options.HiddenDim); w.Write(_options.NumResBlocks);
        w.Write(_options.NumHeads); w.Write(_options.NumAttentionLayers);
        w.Write(_options.DropoutRate);
        w.Write(_options.LearningRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.InputSampleRate = r.ReadInt32(); _options.OutputSampleRate = r.ReadInt32();
        _options.UpsampleFactor = r.ReadInt32(); _options.Variant = r.ReadString();
        _options.HiddenDim = r.ReadInt32(); _options.NumResBlocks = r.ReadInt32(); _numBlocks = _options.NumResBlocks;
        _options.NumHeads = r.ReadInt32(); _options.NumAttentionLayers = r.ReadInt32();
        _options.DropoutRate = r.ReadDouble();
        _options.LearningRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new AudioSuperResolution<T>(Architecture, mp, _options);
        return new AudioSuperResolution<T>(Architecture, _options);
    }

    #endregion

    #region Private Helpers

    private Tensor<T> ApplyStrength(Tensor<T> original, Tensor<T> enhanced)
    {
        double strength = EnhancementStrength;
        if (Math.Abs(strength - 1.0) < 1e-9) return enhanced;

        // Blend original and enhanced based on strength.
        // For super-resolution, enhanced may be longer than original (upsampled).
        // Blend only the overlapping region; keep enhanced-only samples intact.
        var result = new Tensor<T>(enhanced._shape);
        int blendLen = Math.Min(original.Length, enhanced.Length);
        for (int i = 0; i < blendLen; i++)
        {
            double orig = NumOps.ToDouble(original[i]);
            double enh = NumOps.ToDouble(enhanced[i]);
            result[i] = NumOps.FromDouble(orig + (enh - orig) * strength);
        }
        for (int i = blendLen; i < enhanced.Length; i++)
            result[i] = enhanced[i];
        return result;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(AudioSuperResolution<T>)); }

    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }

    #endregion
}

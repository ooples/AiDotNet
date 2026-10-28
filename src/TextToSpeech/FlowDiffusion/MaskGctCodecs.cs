using AiDotNet.Audio.Codecs;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>Loading helpers shared by MaskGCT's codecs.</summary>
internal static class MaskGctLoad
{
    /// <summary>A weight-normalized convolution under <c>weight_g</c>/<c>weight_v</c> or the parametrized names.</summary>
    public static void WeightNormConv<T>(NormedConv1DLayer<T> conv, string name, Func<string, int, double[]?> read)
    {
        var shape = conv.TorchWeightShape;
        int count = shape.Aggregate(1, (a, b) => a * b);
        var direction = read($"{name}.parametrizations.weight.original1", count);
        var gain = direction is null ? read($"{name}.weight_g", shape[0]) : read($"{name}.parametrizations.weight.original0", shape[0]);
        direction ??= read($"{name}.weight_v", count);
        if (direction is null || gain is null) throw new InvalidDataException($"The checkpoint has no weight-normalized convolution '{name}'.");
        conv.LoadTorchWeights(direction, gain, read($"{name}.bias", conv.OutputChannels) ?? throw new InvalidDataException($"The checkpoint has no '{name}.bias'."));
    }

    /// <summary>Amphion's <c>ResidualVQ</c> of factorized quantizers (<c>quantizer.quantizers.{q}.in_project</c>,
    /// <c>out_project</c>, <c>codebook.weight</c>).</summary>
    public static void Quantizer<T>(IEngine engine, FactorizedVectorQuantizerLayer<T> vq, int quantizers, int codes, int codeDim, string prefix,
        Func<string, int, double[]?> read)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        for (int q = 0; q < quantizers; q++)
        {
            string p = $"{prefix}quantizers.{q}";
            WeightNormConv(vq.InProjection(q), $"{p}.in_project", read);
            WeightNormConv(vq.OutProjection(q), $"{p}.out_project", read);
            var codebook = read($"{p}.codebook.weight", codes * codeDim) ?? throw new InvalidDataException($"The checkpoint has no '{p}.codebook.weight'.");
            var target = vq.Codebook(q);
            for (int i = 0; i < codebook.Length; i++) target[i] = ops.FromDouble(codebook[i]);
            engine.InvalidatePersistentTensor(target);
        }
    }
}

/// <summary>
/// MaskGCT's semantic codec as inference uses it (App. A.4; Amphion <c>RepCodec.quantize</c>): w2v-BERT 2.0's 17th hidden
/// state, normalized by the training set's mean and standard deviation, through a Vocos backbone (12 ConvNeXt blocks of
/// width 384, kernel 7) and a linear map back to 1024, quantized by one factorized, L2-normalized codebook of 8,192 codes of
/// dimension 8.
/// </summary>
/// <remarks>The codec's decoder only trains it (feature reconstruction); MaskGCT encodes with it and never decodes.</remarks>
internal sealed class MaskGctSemanticCodec<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly VocosGenerator<T> _encoder;
    private readonly DenseLayer<T> _project;
    private readonly FactorizedVectorQuantizerLayer<T> _quantizer;
    private readonly int _hidden, _dim, _intermediate, _codes, _codeDim;
    private readonly double[] _mean, _std;

    public MaskGctSemanticCodec(IEngine engine, List<LayerBase<T>> owner, int hidden, int dim, int intermediate, int layers, int codes, int codeDim, Random initialization)
    {
        _engine = engine;
        _hidden = hidden;
        _dim = dim;
        _intermediate = intermediate;
        _codes = codes;
        _codeDim = codeDim;
        _encoder = new VocosGenerator<T>(engine, initialization, hidden, dim, intermediate, layers, 2, 1, withHead: false);
        owner.AddRange(_encoder.Layers);
        owner.Add(_project = new DenseLayer<T>(hidden, (IActivationFunction<T>)new IdentityActivation<T>()));
        TorchParameters.Linear(engine, _project, dim, hidden, TorchParameters.Gaussian<T>(initialization, dim * hidden, 0.02).Select(v => NumOps.ToDouble(v)).ToArray(), null);
        owner.Add(_quantizer = new FactorizedVectorQuantizerLayer<T>(hidden, 1, codes, codeDim, lossesOnRawVectors: true));
        _mean = new double[hidden];
        _std = Enumerable.Repeat(1.0, hidden).ToArray();
    }

    /// <summary>Codes per second at 16 kHz (50: one per w2v-BERT frame).</summary>
    public int Hidden => _hidden;

    /// <summary>w2v-BERT hidden states <c>[frames, hidden]</c> normalized by the feature statistics (IndexTTS2's
    /// <c>get_emb</c>, which conditions on them directly).</summary>
    public Tensor<T> Normalize(Tensor<T> hidden)
    {
        int frames = hidden.Shape[0];
        var normalized = new Tensor<T>(new[] { frames, _hidden });
        for (int t = 0; t < frames; t++)
            for (int c = 0; c < _hidden; c++) normalized[t, c] = NumOps.FromDouble((NumOps.ToDouble(hidden[t, c]) - _mean[c]) / _std[c]);
        return normalized;
    }

    /// <summary>The quantized features <c>[frames, hidden]</c> of codes (<c>quantizer.vq2emb</c>: the code vectors
    /// through the output projection).</summary>
    public Tensor<T> Embed(IReadOnlyList<int> codes)
    {
        var grid = new int[1, codes.Count];
        for (int i = 0; i < codes.Count; i++) grid[0, i] = codes[i];
        return _engine.TensorTranspose(_engine.Reshape(_quantizer.Decode(grid), new[] { _hidden, codes.Count }));
    }

    /// <summary>The semantic codes of w2v-BERT hidden states <c>[frames, hidden]</c> (unnormalized).</summary>
    public int[] Encode(Tensor<T> hidden) => EncodeNormalized(Normalize(hidden));

    /// <summary>The semantic codes of normalized hidden states <c>[frames, hidden]</c>.</summary>
    public int[] EncodeNormalized(Tensor<T> normalizedRows)
    {
        using var _ = new NoGradScope<T>();
        int frames = normalizedRows.Shape[0];
        var normalized = _engine.Reshape(_engine.TensorTranspose(normalizedRows), new[] { 1, _hidden, frames });
        var encoded = _project.Forward(_engine.TensorTranspose(_engine.Reshape(_encoder.Backbone(normalized), new[] { _dim, frames })));   // [T, hidden]
        var codes = _quantizer.Encode(_engine.Reshape(_engine.TensorTranspose(encoded), new[] { 1, _hidden, frames }), 1);
        return Enumerable.Range(0, frames).Select(t => codes[0, t]).ToArray();
    }

    /// <summary>Sets the feature statistics (<c>wav2vec2bert_stats.pt</c>: the mean and the variance per channel).</summary>
    public void LoadStatistics(double[] mean, double[] variance)
    {
        if (mean.Length != _hidden || variance.Length != _hidden) throw new ArgumentException($"Expected {_hidden} values each.");
        for (int c = 0; c < _hidden; c++)
        {
            _mean[c] = mean[c];
            _std[c] = Math.Sqrt(variance[c]);
        }
    }

    /// <summary>Loads Amphion's <c>RepCodec</c> encoder and quantizer (<c>encoder.0.*</c>, <c>encoder.1.*</c>,
    /// <c>quantizer.*</c>); the decoder is not read.</summary>
    public void Load(Func<string, int, double[]?> read)
    {
        double[] Need(string name, int count) => read(name, count) ?? throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
        _encoder.LoadTorchWeights((name, shape) => Need(name, shape.Aggregate(1, (a, b) => a * b)), _hidden, _intermediate, "encoder.0.", "");
        TorchParameters.Linear(_engine, _project, _dim, _hidden, Need("encoder.1.weight", _dim * _hidden), Need("encoder.1.bias", _hidden));
        MaskGctLoad.Quantizer(_engine, _quantizer, 1, _codes, _codeDim, "quantizer.", read);
    }
}

/// <summary>
/// MaskGCT's acoustic codec (App. A.4; Amphion <c>CodecEncoder</c> and <c>CodecDecoder</c>): DAC's encoder (width 96,
/// strides 3, 4, 5, 8: 480 samples a frame at 24 kHz) to 256 channels, twelve residual factorized, L2-normalized codebooks
/// of 1,024 codes of dimension 8, and a Vocos decoder (30 ConvNeXt blocks of width 512, an ISTFT head with n_fft 1920 and
/// "same" padding).
/// </summary>
internal sealed class MaskGctAcousticCodec<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly List<LayerBase<T>> _layers;
    private readonly DacEncoder<T> _encoder;
    private readonly FactorizedVectorQuantizerLayer<T> _quantizer;
    private readonly VocosGenerator<T> _decoder;
    private readonly int _latent, _quantizers, _codes, _codeDim, _intermediate, _fft;

    public MaskGctAcousticCodec(IEngine engine, List<LayerBase<T>> owner, int encoderDim, int[] strides, int latent, int quantizers, int codes, int codeDim,
        int vocosDim, int vocosIntermediate, int vocosLayers, int fft, int hop, Random initialization)
    {
        _engine = engine;
        _layers = owner;
        _latent = latent;
        _quantizers = quantizers;
        _codes = codes;
        _codeDim = codeDim;
        _intermediate = vocosIntermediate;
        _fft = fft;
        _encoder = new DacEncoder<T>(engine, encoderDim, strides, latent, owner);
        owner.Add(_quantizer = new FactorizedVectorQuantizerLayer<T>(latent, quantizers, codes, codeDim, lossesOnRawVectors: true));
        _decoder = new VocosGenerator<T>(engine, initialization, latent, vocosDim, vocosIntermediate, vocosLayers, fft, hop, samePadding: true);
        owner.AddRange(_decoder.Layers);
        HopLength = strides.Aggregate(1, (a, b) => a * b);
        if (HopLength != hop) throw new ArgumentException($"The encoder's hop {HopLength} differs from the decoder's {hop}.");
    }

    /// <summary>Samples per frame (480).</summary>
    public int HopLength { get; }

    /// <summary>The codebooks.</summary>
    public int Quantizers => _quantizers;

    /// <summary>The codes <c>[frames, quantizers]</c> of a 24 kHz waveform (<c>extract_acoustic_code</c>).</summary>
    public int[,] Encode(IReadOnlyList<double> audio)
    {
        using var _ = new NoGradScope<T>();
        foreach (var layer in _layers) layer.SetTrainingMode(false);
        var wave = new Tensor<T>(new[] { 1, 1, audio.Count });
        for (int i = 0; i < audio.Count; i++) wave[0, 0, i] = NumOps.FromDouble(audio[i]);
        var codes = _quantizer.Encode(_encoder.Forward(wave));
        int frames = codes.GetLength(1);
        var result = new int[frames, _quantizers];
        for (int t = 0; t < frames; t++)
            for (int q = 0; q < _quantizers; q++) result[t, q] = codes[q, t];
        return result;
    }

    /// <summary>The 24 kHz waveform of codes <c>[frames, quantizers]</c> (<c>vq2emb</c> and the decoder).</summary>
    public double[] Decode(int[,] codes)
    {
        using var _ = new NoGradScope<T>();
        foreach (var layer in _layers) layer.SetTrainingMode(false);
        int frames = codes.GetLength(0);
        var transposed = new int[_quantizers, frames];
        for (int t = 0; t < frames; t++)
            for (int q = 0; q < _quantizers; q++)
            {
                int c = codes[t, q];
                if (c < 0 || c >= _codes) throw new ArgumentOutOfRangeException(nameof(codes), c, $"Codes lie in [0, {_codes}).");
                transposed[q, t] = c;
            }
        var audio = _decoder.Forward(_quantizer.Decode(transposed));
        return audio.ToArray().Select(v => NumOps.ToDouble(v)).ToArray();
    }

    /// <summary>Loads the release's encoder (<c>acoustic_codec/model.safetensors</c>: <c>block.*</c>) through
    /// <paramref name="readEncoder"/>, and its quantizer and Vocos decoder (<c>model_1.safetensors</c>: <c>quantizer.*</c>,
    /// <c>model.backbone.*</c>, <c>model.head.*</c>) through <paramref name="readDecoder"/>.</summary>
    public void Load(Func<string, int, double[]?> readEncoder, Func<string, int, double[]?> readDecoder)
    {
        foreach (var (name, layer) in _encoder.SequentialNamed(""))
        {
            string key = name.TrimStart('.');
            switch (layer)
            {
                case NormedConv1DLayer<T> conv:
                    MaskGctLoad.WeightNormConv(conv, key, readEncoder);
                    break;
                case SnakeLayer<T> snake:
                    snake.LoadAlpha(readEncoder($"{key}.alpha", snake.Channels) ?? throw new InvalidDataException($"The checkpoint has no '{key}.alpha'."));
                    break;
            }
        }
        MaskGctLoad.Quantizer(_engine, _quantizer, _quantizers, _codes, _codeDim, "quantizer.", readDecoder);
        double[] Need(string name, int count) => readDecoder(name, count) ?? throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
        _decoder.LoadTorchWeights((name, shape) => Need(name, shape.Aggregate(1, (a, b) => a * b)), _latent, _intermediate, "model.backbone.", "model.head.");
        var window = readDecoder("model.head.istft.window", _fft);
        if (window is not null) _decoder.Istft!.LoadWindow(window);
    }
}

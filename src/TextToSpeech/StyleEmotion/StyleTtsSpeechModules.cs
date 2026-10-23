using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.StyleEmotion;

/// <summary>
/// The reference's 1-D <c>ResBlk1d</c>: <c>(shortcut(x) + residual(x)) / √2</c>, the residual (instance norm) →
/// LeakyReLU(0.2) → dropout → weight-normalized 3-tap conv → (learned depthwise stride-2 downsampling) → (instance norm) →
/// LeakyReLU → dropout → 3-tap conv to the output width; the shortcut a bias-free 1×1 conv when the width changes, then
/// average pooling by 2 when downsampling.
/// </summary>
internal sealed class StyleTtsResBlock1d<T>
{
    private readonly IEngine _engine;
    private readonly double _dropout;

    public StyleTtsResBlock1d(IEngine engine, List<LayerBase<T>> layers, int dimIn, int dimOut, bool normalize, bool downsample,
        double dropout = 0.2)
    {
        _engine = engine;
        _dropout = dropout;
        Conv1 = StyleTtsOps.Own(layers, StyleTtsOps.WeightNormConv<T>(dimIn, dimIn, 3));
        Conv2 = StyleTtsOps.Own(layers, StyleTtsOps.WeightNormConv<T>(dimIn, dimOut, 3));
        if (normalize)
        {
            Norm1 = StyleTtsOps.Own(layers, new InstanceNormalizationLayer<T>(dimIn, 1e-5, affine: true));
            Norm2 = StyleTtsOps.Own(layers, new InstanceNormalizationLayer<T>(dimIn, 1e-5, affine: true));
        }
        if (dimIn != dimOut) Shortcut = StyleTtsOps.Own(layers, StyleTtsOps.WeightNormConv<T>(dimIn, dimOut, 1, bias: false));
        if (downsample) Pool = StyleTtsOps.Own(layers, StyleTtsOps.WeightNormConv<T>(dimIn, dimIn, 3, stride: 2, padding: 1, groups: dimIn));
        DimIn = dimIn;
        DimOut = dimOut;
    }

    public NormedConv1DLayer<T> Conv1 { get; }
    public NormedConv1DLayer<T> Conv2 { get; }
    public InstanceNormalizationLayer<T>? Norm1 { get; }
    public InstanceNormalizationLayer<T>? Norm2 { get; }
    public NormedConv1DLayer<T>? Shortcut { get; }
    public NormedConv1DLayer<T>? Pool { get; }
    public int DimIn { get; }
    public int DimOut { get; }

    public Tensor<T> Forward(Tensor<T> x, bool training, Random random)
    {
        var shortcut = Shortcut is null ? x : Shortcut.Forward(x);
        if (Pool is not null) shortcut = StyleTtsOps.HalveTime(_engine, shortcut);
        var r = Norm1 is null ? x : Norm1.Forward(x);
        r = StyleTtsOps.Dropout(_engine, StyleTtsOps.LeakyRelu(_engine, r, 0.2), _dropout, training, random);
        r = Conv1.Forward(r);
        if (Pool is not null) r = Pool.Forward(r);
        if (Norm2 is not null) r = Norm2.Forward(r);
        r = StyleTtsOps.Dropout(_engine, StyleTtsOps.LeakyRelu(_engine, r, 0.2), _dropout, training, random);
        r = Conv2.Forward(r);
        return StyleTtsOps.Scale(_engine, _engine.TensorAdd(shortcut, r), StyleTtsOps.InverseSqrt2);
    }

    public void Load(TorchReader read, string name)
    {
        StyleTtsOps.LoadWeightNorm(Conv1, read, name + ".conv1");
        StyleTtsOps.LoadWeightNorm(Conv2, read, name + ".conv2");
        if (Norm1 is not null) StyleTtsOps.LoadInstanceNorm(_engine, Norm1, read, name + ".norm1", DimIn);
        if (Norm2 is not null) StyleTtsOps.LoadInstanceNorm(_engine, Norm2, read, name + ".norm2", DimIn);
        if (Shortcut is not null) StyleTtsOps.LoadWeightNorm(Shortcut, read, name + ".conv1x1", bias: false);
        if (Pool is not null) StyleTtsOps.LoadWeightNorm(Pool, read, name + ".pool");
    }
}

/// <summary>The reference's <c>AdaIN1d</c> (StyleTTS Eq. 1): <c>(1 + γ(s)) · IN(x) + β(s)</c> with γ, β a linear map of
/// the style.</summary>
internal sealed class StyleTtsAdaIn<T>
{
    private readonly IEngine _engine;

    public StyleTtsAdaIn(IEngine engine, List<LayerBase<T>> layers, int styleDim, int channels)
    {
        _engine = engine;
        Norm = StyleTtsOps.Own(layers, new InstanceNormalizationLayer<T>(channels, 1e-5, affine: false));
        Style = StyleTtsOps.Own(layers, StyleTtsOps.Linear<T>(2 * channels));
        Channels = channels;
        StyleDim = styleDim;
    }

    public InstanceNormalizationLayer<T> Norm { get; }
    public DenseLayer<T> Style { get; }
    public int Channels { get; }
    public int StyleDim { get; }

    public Tensor<T> Forward(Tensor<T> x, Tensor<T> style)
    {
        int t = x.Shape[2];
        var h = Style.Forward(style);                                                                       // [1, 2C]
        var gamma = _engine.TensorTile(_engine.Reshape(_engine.TensorSlice(h, new[] { 0, 0 }, new[] { 1, Channels }), new[] { 1, Channels, 1 }), new[] { 1, 1, t });
        var beta = _engine.TensorTile(_engine.Reshape(_engine.TensorSlice(h, new[] { 0, Channels }, new[] { 1, Channels }), new[] { 1, Channels, 1 }), new[] { 1, 1, t });
        var ops = MathHelper.GetNumericOperations<T>();
        return _engine.TensorAdd(_engine.TensorMultiply(_engine.TensorAddScalar(gamma, ops.One), Norm.Forward(x)), beta);
    }

    public void Load(TorchReader read, string name) =>
        StyleTtsOps.LoadLinear(_engine, Style, read, name + ".fc", StyleDim, 2 * Channels);
}

/// <summary>
/// The reference's <c>AdainResBlk1d</c>: <c>(residual(x, s) + shortcut(x)) / √2</c>, the residual AdaIN → LeakyReLU(0.2)
/// → (learned depthwise transposed stride-2 upsampling) → dropout → 3-tap conv → AdaIN → LeakyReLU → dropout → 3-tap conv;
/// the shortcut nearest-neighbour upsampling then a bias-free 1×1 conv when the width changes.
/// </summary>
internal sealed class StyleTtsAdaInResBlock<T>
{
    private readonly IEngine _engine;
    private readonly double _dropout;

    public StyleTtsAdaInResBlock(IEngine engine, List<LayerBase<T>> layers, int dimIn, int dimOut, int styleDim, bool upsample,
        double dropout = 0.0)
    {
        _engine = engine;
        _dropout = dropout;
        Upsample = upsample;
        Conv1 = StyleTtsOps.Own(layers, StyleTtsOps.WeightNormConv<T>(dimIn, dimOut, 3));
        Conv2 = StyleTtsOps.Own(layers, StyleTtsOps.WeightNormConv<T>(dimOut, dimOut, 3));
        Norm1 = new StyleTtsAdaIn<T>(engine, layers, styleDim, dimIn);
        Norm2 = new StyleTtsAdaIn<T>(engine, layers, styleDim, dimOut);
        if (dimIn != dimOut) Shortcut = StyleTtsOps.Own(layers, StyleTtsOps.WeightNormConv<T>(dimIn, dimOut, 1, bias: false));
        if (upsample)
            Pool = StyleTtsOps.Own(layers, new NormedConv1DLayer<T>(dimIn, dimIn, 3, 2, 1, dimIn, 1, true, ConvolutionNormalization.Weight, outputPadding: 1));
    }

    public bool Upsample { get; }
    public NormedConv1DLayer<T> Conv1 { get; }
    public NormedConv1DLayer<T> Conv2 { get; }
    public StyleTtsAdaIn<T> Norm1 { get; }
    public StyleTtsAdaIn<T> Norm2 { get; }
    public NormedConv1DLayer<T>? Shortcut { get; }
    public NormedConv1DLayer<T>? Pool { get; }

    public Tensor<T> Forward(Tensor<T> x, Tensor<T> style, bool training, Random random)
    {
        var r = StyleTtsOps.LeakyRelu(_engine, Norm1.Forward(x, style), 0.2);
        if (Pool is not null) r = Pool.Forward(r);
        r = Conv1.Forward(StyleTtsOps.Dropout(_engine, r, _dropout, training, random));
        r = StyleTtsOps.LeakyRelu(_engine, Norm2.Forward(r, style), 0.2);
        r = Conv2.Forward(StyleTtsOps.Dropout(_engine, r, _dropout, training, random));
        var shortcut = Upsample ? StyleTtsOps.RepeatFrames(_engine, x) : x;
        if (Shortcut is not null) shortcut = Shortcut.Forward(shortcut);
        return StyleTtsOps.Scale(_engine, _engine.TensorAdd(r, shortcut), StyleTtsOps.InverseSqrt2);
    }

    public void Load(TorchReader read, string name)
    {
        StyleTtsOps.LoadWeightNorm(Conv1, read, name + ".conv1");
        StyleTtsOps.LoadWeightNorm(Conv2, read, name + ".conv2");
        Norm1.Load(read, name + ".norm1");
        Norm2.Load(read, name + ".norm2");
        if (Shortcut is not null) StyleTtsOps.LoadWeightNorm(Shortcut, read, name + ".conv1x1", bias: false);
        if (Pool is not null) StyleTtsOps.LoadWeightNorm(Pool, read, name + ".pool");
    }
}

/// <summary>
/// The reference's mel <c>Decoder</c> (StyleTTS §II-A): pitch and energy each downsampled by a residual block, a 1×1 conv
/// and an instance norm; two residual blocks over the aligned phoneme features with them; five AdaIN residual blocks, the
/// first three reading the pitch, energy and a residual projection of the phoneme features again, the third upsampling
/// the frame rate by 2; a 1×1 conv to the mel bins.
/// </summary>
internal sealed class StyleTtsDecoder<T>
{
    private readonly IEngine _engine;

    public StyleTtsDecoder(IEngine engine, List<LayerBase<T>> layers, int dimIn, int styleDim, int residualDim, int dimOut)
    {
        _engine = engine;
        int bottleneck = 2 * dimIn;
        Encode.Add(new StyleTtsResBlock1d<T>(engine, layers, dimIn + 2, bottleneck, normalize: true, downsample: false));
        Encode.Add(new StyleTtsResBlock1d<T>(engine, layers, bottleneck, bottleneck, normalize: true, downsample: false));
        Decode.Add(new StyleTtsAdaInResBlock<T>(engine, layers, bottleneck + residualDim + 2, bottleneck, styleDim, false));
        Decode.Add(new StyleTtsAdaInResBlock<T>(engine, layers, bottleneck + residualDim + 2, bottleneck, styleDim, false));
        Decode.Add(new StyleTtsAdaInResBlock<T>(engine, layers, bottleneck + residualDim + 2, dimIn, styleDim, true));
        Decode.Add(new StyleTtsAdaInResBlock<T>(engine, layers, dimIn, dimIn, styleDim, false));
        Decode.Add(new StyleTtsAdaInResBlock<T>(engine, layers, dimIn, dimIn, styleDim, false));
        F0Block = new StyleTtsResBlock1d<T>(engine, layers, 1, residualDim, normalize: true, downsample: true);
        F0Projection = StyleTtsOps.Own(layers, StyleTtsOps.WeightNormConv<T>(residualDim, 1, 1));
        F0Norm = StyleTtsOps.Own(layers, new InstanceNormalizationLayer<T>(1, 1e-5, affine: true));
        NBlock = new StyleTtsResBlock1d<T>(engine, layers, 1, residualDim, normalize: true, downsample: true);
        NProjection = StyleTtsOps.Own(layers, StyleTtsOps.WeightNormConv<T>(residualDim, 1, 1));
        NNorm = StyleTtsOps.Own(layers, new InstanceNormalizationLayer<T>(1, 1e-5, affine: true));
        AsrResidual = StyleTtsOps.Own(layers, StyleTtsOps.WeightNormConv<T>(dimIn, residualDim, 1));
        AsrNorm = StyleTtsOps.Own(layers, new InstanceNormalizationLayer<T>(residualDim, 1e-5, affine: true));
        ToOut = StyleTtsOps.Own(layers, StyleTtsOps.WeightNormConv<T>(dimIn, dimOut, 1));
        ResidualDim = residualDim;
    }

    public List<StyleTtsResBlock1d<T>> Encode { get; } = new();
    public List<StyleTtsAdaInResBlock<T>> Decode { get; } = new();
    public StyleTtsResBlock1d<T> F0Block { get; }
    public NormedConv1DLayer<T> F0Projection { get; }
    public InstanceNormalizationLayer<T> F0Norm { get; }
    public StyleTtsResBlock1d<T> NBlock { get; }
    public NormedConv1DLayer<T> NProjection { get; }
    public InstanceNormalizationLayer<T> NNorm { get; }
    public NormedConv1DLayer<T> AsrResidual { get; }
    public InstanceNormalizationLayer<T> AsrNorm { get; }
    public NormedConv1DLayer<T> ToOut { get; }
    public int ResidualDim { get; }

    /// <summary>The mel spectrogram <c>[1, mels, 2T]</c> of aligned phoneme features <c>[1, dimIn, T]</c>, pitch and
    /// energy <c>[1, 2T]</c> and a style <c>[1, style]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> asr, Tensor<T> f0, Tensor<T> energy, Tensor<T> style, bool training, Random random)
    {
        Tensor<T> Contour(StyleTtsResBlock1d<T> block, NormedConv1DLayer<T> projection, InstanceNormalizationLayer<T> norm, Tensor<T> c)
            => norm.Forward(projection.Forward(block.Forward(_engine.Reshape(c, new[] { 1, 1, c.Length }), training, random)));
        var pitch = Contour(F0Block, F0Projection, F0Norm, f0);
        var norm = Contour(NBlock, NProjection, NNorm, energy);
        var x = _engine.TensorConcatenate(new[] { asr, pitch, norm }, 1);
        foreach (var block in Encode) x = block.Forward(x, training, random);
        var asrResidual = AsrNorm.Forward(AsrResidual.Forward(asr));
        bool residual = true;
        foreach (var block in Decode)
        {
            if (residual) x = _engine.TensorConcatenate(new[] { x, asrResidual, pitch, norm }, 1);
            x = block.Forward(x, style, training, random);
            if (block.Upsample) residual = false;
        }
        return ToOut.Forward(x);
    }

    public void Load(TorchReader read, string name)
    {
        for (int i = 0; i < Encode.Count; i++) Encode[i].Load(read, $"{name}.encode.{i}");
        for (int i = 0; i < Decode.Count; i++) Decode[i].Load(read, $"{name}.decode.{i}");
        F0Block.Load(read, name + ".F0_conv.0");
        StyleTtsOps.LoadWeightNorm(F0Projection, read, name + ".F0_conv.1");
        StyleTtsOps.LoadInstanceNorm(_engine, F0Norm, read, name + ".F0_conv.2", 1);
        NBlock.Load(read, name + ".N_conv.0");
        StyleTtsOps.LoadWeightNorm(NProjection, read, name + ".N_conv.1");
        StyleTtsOps.LoadInstanceNorm(_engine, NNorm, read, name + ".N_conv.2", 1);
        StyleTtsOps.LoadWeightNorm(AsrResidual, read, name + ".asr_res.0");
        StyleTtsOps.LoadInstanceNorm(_engine, AsrNorm, read, name + ".asr_res.1", ResidualDim);
        StyleTtsOps.LoadWeightNorm(ToOut, read, name + ".to_out.0");
    }
}

/// <summary>
/// The reference's <c>TextEncoder</c> (StyleTTS §II-A, Tacotron 2's): phoneme embeddings, three blocks of a
/// weight-normalized 5-tap conv, a layer norm over channels, LeakyReLU(0.2) and dropout 0.2, then a bidirectional LSTM.
/// </summary>
internal sealed class StyleTtsTextEncoder<T>
{
    private readonly IEngine _engine;

    public StyleTtsTextEncoder(IEngine engine, List<LayerBase<T>> layers, int channels, int kernel, int depth, int symbols)
    {
        _engine = engine;
        Embedding = StyleTtsOps.Own(layers, new TiedEmbeddingLayer<T>(symbols, channels));
        for (int i = 0; i < depth; i++)
            Blocks.Add((StyleTtsOps.Own(layers, StyleTtsOps.WeightNormConv<T>(channels, channels, kernel)),
                StyleTtsOps.Own(layers, new LayerNormalizationLayer<T>(channels, 1e-5))));
        Lstm = new StyleTtsBiLstm<T>(engine, layers, channels, channels / 2);
        Channels = channels;
        Symbols = symbols;
    }

    public TiedEmbeddingLayer<T> Embedding { get; }
    public List<(NormedConv1DLayer<T> Conv, LayerNormalizationLayer<T> Norm)> Blocks { get; } = new();
    public StyleTtsBiLstm<T> Lstm { get; }
    public int Channels { get; }
    public int Symbols { get; }

    /// <summary>The phoneme features <c>[1, channels, L]</c> of token ids <c>[L]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> ids, bool training, Random random)
    {
        int length = ids.Length;
        var x = _engine.Reshape(_engine.TensorTranspose(Embedding.Forward(ids)), new[] { 1, Channels, length });
        foreach (var (conv, norm) in Blocks)
        {
            var rows = norm.Forward(_engine.TensorTranspose(_engine.Reshape(conv.Forward(x), new[] { Channels, length })));
            rows = StyleTtsOps.Dropout(_engine, StyleTtsOps.LeakyRelu(_engine, rows, 0.2), 0.2, training, random);
            x = _engine.Reshape(_engine.TensorTranspose(rows), new[] { 1, Channels, length });
        }
        var output = Lstm.Apply(_engine.TensorTranspose(_engine.Reshape(x, new[] { Channels, length })));    // [L, C]
        return _engine.Reshape(_engine.TensorTranspose(output), new[] { 1, Channels, length });
    }

    public void Load(TorchReader read, string name)
    {
        Embedding.LoadTable(read(name + ".embedding.weight", new[] { Symbols, Channels }));
        for (int i = 0; i < Blocks.Count; i++)
        {
            StyleTtsOps.LoadWeightNorm(Blocks[i].Conv, read, $"{name}.cnn.{i}.0");
            TorchParameters.LayerNorm(_engine, Blocks[i].Norm, read($"{name}.cnn.{i}.1.gamma", new[] { Channels }),
                read($"{name}.cnn.{i}.1.beta", new[] { Channels }));
        }
        Lstm.Load(read, name + ".lstm");
    }
}

/// <summary>The reference's <c>AdaLayerNorm</c>: <c>(1 + γ(s)) · LN(x) + β(s)</c> over the channels of each row
/// <c>[L, C]</c> (StyleTTS's duration predictor, §II-A).</summary>
internal sealed class StyleTtsAdaLayerNorm<T>
{
    private readonly IEngine _engine;

    public StyleTtsAdaLayerNorm(IEngine engine, List<LayerBase<T>> layers, int styleDim, int channels)
    {
        _engine = engine;
        Style = StyleTtsOps.Own(layers, StyleTtsOps.Linear<T>(2 * channels));
        Channels = channels;
        StyleDim = styleDim;
    }

    public DenseLayer<T> Style { get; }
    public int Channels { get; }
    public int StyleDim { get; }

    public Tensor<T> Forward(Tensor<T> rows, Tensor<T> style)
    {
        int length = rows.Shape[0];
        var ops = MathHelper.GetNumericOperations<T>();
        var mean = _engine.TensorTile(_engine.ReduceMean(rows, new[] { 1 }, keepDims: true), new[] { 1, Channels });
        var centered = _engine.TensorSubtract(rows, mean);
        var variance = _engine.ReduceMean(_engine.TensorMultiply(centered, centered), new[] { 1 }, keepDims: true);
        var inverse = _engine.TensorTile(_engine.TensorPow(_engine.TensorAddScalar(variance, ops.FromDouble(1e-5)), ops.FromDouble(-0.5)), new[] { 1, Channels });
        var normed = _engine.TensorMultiply(centered, inverse);
        var h = Style.Forward(style);                                                                       // [1, 2C]
        var gamma = _engine.TensorTile(_engine.TensorSlice(h, new[] { 0, 0 }, new[] { 1, Channels }), new[] { length, 1 });
        var beta = _engine.TensorTile(_engine.TensorSlice(h, new[] { 0, Channels }, new[] { 1, Channels }), new[] { length, 1 });
        return _engine.TensorAdd(_engine.TensorMultiply(_engine.TensorAddScalar(gamma, ops.One), normed), beta);
    }

    public void Load(TorchReader read, string name) =>
        StyleTtsOps.LoadLinear(_engine, Style, read, name + ".fc", StyleDim, 2 * Channels);
}

/// <summary>
/// The reference's <c>ProsodyPredictor</c> (StyleTTS §II-A): the duration encoder — layers of a bidirectional LSTM over the
/// phoneme features with the style concatenated, dropout, and AdaLN — then a bidirectional LSTM and a linear duration
/// head; for pitch and energy, the duration encoder's output aligned to the frames, a shared bidirectional LSTM and two
/// stacks of three AdaIN residual blocks (the second upsampling by 2) with 1×1 projections.
/// </summary>
internal sealed class StyleTtsProsodyPredictor<T>
{
    private readonly IEngine _engine;
    private readonly double _dropout;

    public StyleTtsProsodyPredictor(IEngine engine, List<LayerBase<T>> layers, int styleDim, int hidden, int layerCount, double dropout,
        int durationBins = 1)
    {
        _engine = engine;
        _dropout = dropout;
        for (int i = 0; i < layerCount; i++)
            EncoderLayers.Add((new StyleTtsBiLstm<T>(engine, layers, hidden + styleDim, hidden / 2), new StyleTtsAdaLayerNorm<T>(engine, layers, styleDim, hidden)));
        Lstm = new StyleTtsBiLstm<T>(engine, layers, hidden + styleDim, hidden / 2);
        DurationProjection = StyleTtsOps.Own(layers, StyleTtsOps.Linear<T>(durationBins));
        DurationBins = durationBins;
        Shared = new StyleTtsBiLstm<T>(engine, layers, hidden + styleDim, hidden / 2);
        foreach (var stack in new[] { F0, N })
        {
            stack.Add(new StyleTtsAdaInResBlock<T>(engine, layers, hidden, hidden, styleDim, false, dropout));
            stack.Add(new StyleTtsAdaInResBlock<T>(engine, layers, hidden, hidden / 2, styleDim, true, dropout));
            stack.Add(new StyleTtsAdaInResBlock<T>(engine, layers, hidden / 2, hidden / 2, styleDim, false, dropout));
        }
        F0Projection = StyleTtsOps.Own(layers, new NormedConv1DLayer<T>(hidden / 2, 1, 1, 1, 1, 1, 0, false, ConvolutionNormalization.None));
        NProjection = StyleTtsOps.Own(layers, new NormedConv1DLayer<T>(hidden / 2, 1, 1, 1, 1, 1, 0, false, ConvolutionNormalization.None));
        Hidden = hidden;
        StyleDim = styleDim;
    }

    public List<(StyleTtsBiLstm<T> Lstm, StyleTtsAdaLayerNorm<T> Norm)> EncoderLayers { get; } = new();
    public StyleTtsBiLstm<T> Lstm { get; }
    public DenseLayer<T> DurationProjection { get; }
    public StyleTtsBiLstm<T> Shared { get; }
    public List<StyleTtsAdaInResBlock<T>> F0 { get; } = new();
    public List<StyleTtsAdaInResBlock<T>> N { get; } = new();
    public NormedConv1DLayer<T> F0Projection { get; }
    public NormedConv1DLayer<T> NProjection { get; }
    public int Hidden { get; }
    public int StyleDim { get; }

    /// <summary>1 for StyleTTS's scalar duration; StyleTTS 2's <c>max_dur</c> (50), whose sigmoids sum to the duration.</summary>
    public int DurationBins { get; }

    private Tensor<T> WithStyle(Tensor<T> rows, Tensor<T> style) =>
        _engine.TensorConcatenate(new[] { rows, _engine.TensorTile(style, new[] { rows.Shape[0], 1 }) }, 1);

    /// <summary>The duration encoder's output <c>[L, hidden + style]</c> for phoneme features <c>[1, hidden, L]</c>.</summary>
    public Tensor<T> EncodeText(Tensor<T> text, Tensor<T> style, bool training, Random random)
    {
        int length = text.Shape[2];
        var x = WithStyle(_engine.TensorTranspose(_engine.Reshape(text, new[] { Hidden, length })), style);
        foreach (var (lstm, norm) in EncoderLayers)
        {
            x = StyleTtsOps.Dropout(_engine, lstm.Apply(x), _dropout, training, random);
            x = WithStyle(norm.Forward(x, style), style);
        }
        return x;
    }

    /// <summary>The predicted duration of each phoneme <c>[L]</c> (in frames before rounding) from the duration encoder's
    /// output; with duration bins, the per-bin logits <c>[L, bins]</c> instead.</summary>
    public Tensor<T> Durations(Tensor<T> encoded, bool training, Random random)
    {
        var x = Lstm.Apply(encoded);
        var d = DurationProjection.Forward(StyleTtsOps.Dropout(_engine, x, 0.5, training, random));
        return DurationBins == 1 ? _engine.Reshape(d, new[] { d.Shape[0] }) : d;
    }

    /// <summary>Pitch and energy <c>[1, 2T]</c> from the duration encoder's output aligned to the frames,
    /// <c>[1, hidden + style, T]</c> (the reference's <c>F0Ntrain</c>).</summary>
    public (Tensor<T> F0, Tensor<T> Energy) Prosody(Tensor<T> aligned, Tensor<T> style, bool training, Random random)
    {
        int frames = aligned.Shape[2];
        var shared = Shared.Apply(_engine.TensorTranspose(_engine.Reshape(aligned, new[] { Hidden + StyleDim, frames })));
        var x = _engine.Reshape(_engine.TensorTranspose(shared), new[] { 1, Hidden, frames });
        Tensor<T> Run(List<StyleTtsAdaInResBlock<T>> stack, NormedConv1DLayer<T> projection)
        {
            var h = x;
            foreach (var block in stack) h = block.Forward(h, style, training, random);
            var y = projection.Forward(h);
            return _engine.Reshape(y, new[] { 1, y.Shape[2] });
        }
        return (Run(F0, F0Projection), Run(N, NProjection));
    }

    public void Load(TorchReader read, string name)
    {
        for (int i = 0; i < EncoderLayers.Count; i++)
        {
            EncoderLayers[i].Lstm.Load(read, $"{name}.text_encoder.lstms.{2 * i}");
            EncoderLayers[i].Norm.Load(read, $"{name}.text_encoder.lstms.{2 * i + 1}");
        }
        Lstm.Load(read, name + ".lstm");
        StyleTtsOps.LoadLinear(_engine, DurationProjection, read, name + ".duration_proj.linear_layer", Hidden, DurationBins);
        Shared.Load(read, name + ".shared");
        for (int i = 0; i < F0.Count; i++) F0[i].Load(read, $"{name}.F0.{i}");
        for (int i = 0; i < N.Count; i++) N[i].Load(read, $"{name}.N.{i}");
        StyleTtsOps.LoadPlain(F0Projection, read, name + ".F0_proj");
        StyleTtsOps.LoadPlain(NProjection, read, name + ".N_proj");
    }
}

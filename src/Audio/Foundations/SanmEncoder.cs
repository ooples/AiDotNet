using AiDotNet.ActivationFunctions;
using AiDotNet.Audio.Features;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Audio.Foundations;

/// <summary>The sizes of SenseVoice-Small's SAN-M encoder (<c>SenseVoiceEncoderSmall</c>) and its front end.</summary>
/// <param name="Mels">Kaldi filterbank bands (80).</param>
/// <param name="LfrM">Frames stacked by low frame rate (7) and the stride between stacks (6).</param>
/// <param name="Width">The encoder's width (512), heads (4) and feed-forward width (2048).</param>
/// <param name="Blocks">SAN-M blocks before the norm (50, the first reading the stacked features) and after it
/// (<c>tp_blocks</c>, 20).</param>
/// <param name="Kernel">The FSMN memory's kernel (11).</param>
/// <param name="NormEpsilon">The layer norms' epsilon: SenseVoice's own LayerNorm (1e-5), or FunASR's (1e-12) for
/// Paraformer.</param>
internal sealed record SanmEncoderConfiguration(int Mels = 80, int LfrM = 7, int LfrN = 6, int Width = 512, int Heads = 4,
    int FeedForward = 2048, int Blocks = 50, int TpBlocks = 20, int Kernel = 11, double NormEpsilon = 1e-5)
{
    /// <summary>The stacked feature width, <c>Mels · LfrM</c> (560).</summary>
    public int InputSize => Mels * LfrM;
}

/// <summary>
/// SenseVoice-Small's encoder (An et al. 2024, FunAudioLLM §2.2; the release's <c>SenseVoiceEncoderSmall</c> and
/// FunASR's <c>WavFrontend</c>): 80-band Kaldi filterbanks of the 16 kHz waveform scaled to 16-bit range, seven frames
/// stacked every six (low frame rate, 60 ms), mean-variance normalized; then scaled by √width, sinusoidal positions from 1,
/// and SAN-M blocks — pre-norm self-attention whose values also pass through an FSMN memory (a depthwise convolution plus
/// the values) added to the attention output, and a ReLU feed-forward — in two groups with a layer norm after each.
/// </summary>
/// <remarks>The first block reads the stacked features and keeps no residual, as the release's does when the widths
/// differ. Dither is off (inference). SenseVoice's own prefix of task queries is not prepended: MinMo reads the encoder
/// over speech.</remarks>
internal sealed class SanmEncoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly KaldiFbank<T> _fbank;
    private double[] _shift, _scale;

    internal sealed record Block(LayerNormalizationLayer<T> Norm1, DenseLayer<T> QueryKeyValue, NormedConv1DLayer<T> Fsmn,
        DenseLayer<T> Output, LayerNormalizationLayer<T> Norm2, DenseLayer<T> FeedForward1, DenseLayer<T> FeedForward2, int InputSize);

    public SanmEncoder(IEngine engine, List<LayerBase<T>> layers, SanmEncoderConfiguration configuration, Random initialization)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.Width % c.Heads != 0) throw new ArgumentException("The width must divide by the heads.", nameof(configuration));
        if (c.Blocks < 1) throw new ArgumentException("SAN-M needs at least one block.", nameof(configuration));
        _fbank = new KaldiFbank<T>(engine, 16000, c.Mels, WindowType.Hamming);
        _shift = new double[c.InputSize];
        _scale = Enumerable.Repeat(1.0, c.InputSize).ToArray();
        TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T> { layers.Add(layer); return layer; }
        double Uniform(int fanIn) => (2 * initialization.NextDouble() - 1) / Math.Sqrt(fanIn);
        DenseLayer<T> Linear(int inputs, int outputs)
        {
            var layer = Add(new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()));
            TorchParameters.Linear(engine, layer, inputs, outputs, Enumerable.Range(0, inputs * outputs).Select(_ => Uniform(inputs)).ToArray(),
                Enumerable.Range(0, outputs).Select(_ => Uniform(inputs)).ToArray());
            return layer;
        }
        Block NewBlock(int inputSize)
        {
            var norm1 = Add(new LayerNormalizationLayer<T>(inputSize, c.NormEpsilon));
            var qkv = Linear(inputSize, 3 * c.Width);
            var fsmn = Add(new NormedConv1DLayer<T>(c.Width, c.Width, c.Kernel, 1, 1, c.Width, 0, false, ConvolutionNormalization.None, useBias: false));
            fsmn.Reinitialize(() => Uniform(c.Kernel));
            var output = Linear(c.Width, c.Width);
            var norm2 = Add(new LayerNormalizationLayer<T>(c.Width, c.NormEpsilon));
            return new Block(norm1, qkv, fsmn, output, norm2, Linear(c.Width, c.FeedForward), Linear(c.FeedForward, c.Width), inputSize);
        }
        Blocks.Add(NewBlock(c.InputSize));
        for (int i = 1; i < c.Blocks; i++) Blocks.Add(NewBlock(c.Width));
        AfterNorm = Add(new LayerNormalizationLayer<T>(c.Width, c.NormEpsilon));
        for (int i = 0; i < c.TpBlocks; i++) TpBlocks.Add(NewBlock(c.Width));
        TpNorm = c.TpBlocks > 0 ? Add(new LayerNormalizationLayer<T>(c.Width, c.NormEpsilon)) : null;
    }

    public SanmEncoderConfiguration Configuration { get; }
    public List<Block> Blocks { get; } = new();
    public LayerNormalizationLayer<T> AfterNorm { get; }
    public List<Block> TpBlocks { get; } = new();
    /// <summary>The norm after the tp blocks; null without them (Paraformer).</summary>
    public LayerNormalizationLayer<T>? TpNorm { get; }

    /// <summary>The CMVN's shift and scale (<c>am.mvn</c>'s AddShift and Rescale): features become (x + shift) · scale.</summary>
    public (double[] Shift, double[] Scale) Normalization
    {
        get => ((double[])_shift.Clone(), (double[])_scale.Clone());
        set
        {
            if (value.Shift.Length < Configuration.InputSize || value.Scale.Length < Configuration.InputSize)
                throw new ArgumentException($"The normalization needs {Configuration.InputSize} values.");
            _shift = value.Shift.Take(Configuration.InputSize).ToArray();
            _scale = value.Scale.Take(Configuration.InputSize).ToArray();
        }
    }

    /// <summary>Reads FunASR's <c>am.mvn</c> (Kaldi nnet text: the vectors after <c>&lt;AddShift&gt;</c> and
    /// <c>&lt;Rescale&gt;</c>).</summary>
    public static (double[] Shift, double[] Scale) ReadNormalization(string path)
    {
        var lines = File.ReadAllLines(path);
        double[]? shift = null, scale = null;
        double[] Vector(string line)
        {
            var items = line.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            return items.Skip(3).Take(items.Length - 4).Select(v => double.Parse(v, System.Globalization.CultureInfo.InvariantCulture)).ToArray();
        }
        for (int i = 0; i + 1 < lines.Length; i++)
        {
            string head = lines[i].Trim().Split(' ')[0];
            if (head == "<AddShift>" && lines[i + 1].TrimStart().StartsWith("<LearnRateCoef>", StringComparison.Ordinal)) shift = Vector(lines[i + 1]);
            if (head == "<Rescale>" && lines[i + 1].TrimStart().StartsWith("<LearnRateCoef>", StringComparison.Ordinal)) scale = Vector(lines[i + 1]);
        }
        return (shift ?? throw new InvalidDataException("am.mvn has no <AddShift>."), scale ?? throw new InvalidDataException("am.mvn has no <Rescale>."));
    }

    /// <summary>The front end's features <c>[frames, Mels · LfrM]</c> of a 16 kHz waveform in [−1, 1].</summary>
    public Tensor<T> Features(Tensor<T> audio)
    {
        var c = Configuration;
        var scaled = _engine.TensorMultiplyScalar(_engine.Reshape(audio, new[] { audio.Length }), NumOps.FromDouble(1 << 15));
        var fbank = _fbank.Forward(scaled);                                                              // [T, mels]
        int t = fbank.Shape[0];
        if (t == 0) throw new ArgumentException("The recording is shorter than one filterbank frame.", nameof(audio));
        // apply_lfr: (m − 1)/2 copies of the first frame before, stacks of m every n, the last frame repeated past the end.
        int left = (c.LfrM - 1) / 2, padded = t + left, frames = (t + c.LfrN - 1) / c.LfrN;
        var features = new Tensor<T>(new[] { frames, c.InputSize });
        for (int f = 0; f < frames; f++)
            for (int j = 0; j < c.LfrM; j++)
            {
                int row = Math.Min(f * c.LfrN + j, padded - 1) - left;
                row = Math.Max(0, row);
                for (int m = 0; m < c.Mels; m++)
                {
                    int k = j * c.Mels + m;
                    features[f, k] = NumOps.FromDouble((NumOps.ToDouble(fbank[row, m]) + _shift[k]) * _scale[k]);
                }
            }
        return features;
    }

    /// <summary>The encoder's output <c>[frames, width]</c> of features <c>[frames, Mels · LfrM]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> features)
    {
        var c = Configuration;
        int length = features.Shape[0], input = features.Shape[1];
        var x = AddPositions(_engine.TensorMultiplyScalar(features, NumOps.FromDouble(Math.Sqrt(c.Width))), 0);
        foreach (var block in Blocks) x = Run(block, x);
        x = AfterNorm.Forward(x);
        foreach (var block in TpBlocks) x = Run(block, x);
        return TpNorm is null ? x : TpNorm.Forward(x);
    }

    /// <summary>SinusoidalPositionEncoder (and its streaming form from <paramref name="start"/>): positions start + 1 …,
    /// timescales 10000^(−k/(depth/2 − 1)), sines then cosines, added to <c>[L, depth]</c>.</summary>
    internal Tensor<T> AddPositions(Tensor<T> x, int start)
    {
        int length = x.Shape[0], input = x.Shape[1], half = input / 2;
        var positions = new Tensor<T>(new[] { length, input });
        double increment = Math.Log(10000.0) / (input / 2.0 - 1);
        for (int t = 0; t < length; t++)
            for (int k = 0; k < half; k++)
            {
                double scaled = (start + t + 1) * Math.Exp(-increment * k);
                positions[t, k] = NumOps.FromDouble(Math.Sin(scaled));
                positions[t, half + k] = NumOps.FromDouble(Math.Cos(scaled));
            }
        return _engine.TensorAdd(x, positions);
    }

    /// <summary>The encoder's output for a 16 kHz waveform.</summary>
    public Tensor<T> Encode(Tensor<T> audio) => Forward(Features(audio));

    private Tensor<T> Run(Block block, Tensor<T> x) => RunChunk(block, x, null, null).Output;

    /// <summary>
    /// A SAN-M block on <c>[L, width]</c>, attending over <paramref name="pastKeys"/> and <paramref name="pastValues"/>
    /// (<c>[P, width]</c>, the streaming encoder's look-back) before this input's own keys and values, which are returned
    /// with the output. The FSMN memory reads this input only.
    /// </summary>
    internal (Tensor<T> Output, Tensor<T> Keys, Tensor<T> Values) RunChunk(Block block, Tensor<T> x, Tensor<T>? pastKeys, Tensor<T>? pastValues)
    {
        var c = Configuration;
        int length = x.Shape[0], d = c.Width, heads = c.Heads, dk = d / heads;
        var normed = block.Norm1.Forward(x);
        var qkv = block.QueryKeyValue.Forward(normed);                                                   // [L, 3d]
        var q = _engine.TensorSlice(qkv, new[] { 0, 0 }, new[] { length, d });
        var ownKeys = _engine.TensorSlice(qkv, new[] { 0, d }, new[] { length, d });
        var v = _engine.TensorSlice(qkv, new[] { 0, 2 * d }, new[] { length, d });
        var k = pastKeys is null ? ownKeys : _engine.TensorConcatenate(new[] { pastKeys, ownKeys }, 0);
        var attendedValues = pastValues is null ? v : _engine.TensorConcatenate(new[] { pastValues, v }, 0);
        int keys = k.Shape[0];
        // forward_fsmn: the values padded (k − 1)/2 each side, the depthwise convolution, plus the values.
        int pad = (c.Kernel - 1) / 2, right = c.Kernel - 1 - pad;
        var channels = _engine.Reshape(_engine.TensorTranspose(v), new[] { 1, d, length });
        var paddedValues = _engine.TensorConcatenate(new[] { new Tensor<T>(new[] { 1, d, pad }), channels, new Tensor<T>(new[] { 1, d, right }) }, 2);
        var memory = _engine.TensorAdd(_engine.TensorTranspose(_engine.Reshape(block.Fsmn.Forward(paddedValues), new[] { d, length })), v);
        Tensor<T> Heads(Tensor<T> t, int rows) => _engine.TensorPermute(_engine.Reshape(t, new[] { rows, heads, dk }), new[] { 1, 0, 2 });
        var scores = _engine.BatchMatMul(_engine.TensorMultiplyScalar(Heads(q, length), NumOps.FromDouble(Math.Pow(dk, -0.5))),
            _engine.TensorPermute(Heads(k, keys), new[] { 0, 2, 1 }));
        var weights = _engine.Softmax(scores, axis: 2);
        var context = _engine.Reshape(_engine.TensorPermute(_engine.BatchMatMul(weights, Heads(attendedValues, keys)), new[] { 1, 0, 2 }), new[] { length, d });
        var attended = _engine.TensorAdd(block.Output.Forward(context), memory);
        x = block.InputSize == d ? _engine.TensorAdd(x, attended) : attended;
        var h = block.FeedForward2.Forward(_engine.ReLU(block.FeedForward1.Forward(block.Norm2.Forward(x))));
        return (_engine.TensorAdd(x, h), ownKeys, v);
    }

    /// <summary>Loads the release's encoder state (<c>{prefix}encoders0.0.*</c>, <c>encoders.{i}.*</c>,
    /// <c>tp_encoders.{i}.*</c>, <c>after_norm</c>, <c>tp_norm</c>; SenseVoiceSmall's <c>model.pt</c> under
    /// <c>encoder.</c>).</summary>
    public void LoadTorchWeights(Func<string, double[]?> read, string prefix = "encoder.")
    {
        var c = Configuration;
        double[] Read(string name) => read(prefix + name) ?? throw new InvalidDataException($"The SenseVoice checkpoint has no tensor '{prefix}{name}'.");
        void Norm(LayerNormalizationLayer<T> norm, string name) => TorchParameters.LayerNorm(_engine, norm, Read(name + ".weight"), Read(name + ".bias"));
        void Dense(DenseLayer<T> layer, string name, int inputs, int outputs) =>
            TorchParameters.Linear(_engine, layer, inputs, outputs, Read(name + ".weight"), Read(name + ".bias"));
        void Load(Block b, string p)
        {
            Norm(b.Norm1, p + "norm1");
            Dense(b.QueryKeyValue, p + "self_attn.linear_q_k_v", b.InputSize, 3 * c.Width);
            var fsmn = Read(p + "self_attn.fsmn_block.weight");
            b.Fsmn.LoadTorchWeights(fsmn, null, new double[c.Width]);
            Dense(b.Output, p + "self_attn.linear_out", c.Width, c.Width);
            Norm(b.Norm2, p + "norm2");
            Dense(b.FeedForward1, p + "feed_forward.w_1", c.Width, c.FeedForward);
            Dense(b.FeedForward2, p + "feed_forward.w_2", c.FeedForward, c.Width);
        }
        Load(Blocks[0], "encoders0.0.");
        for (int i = 1; i < Blocks.Count; i++) Load(Blocks[i], $"encoders.{i - 1}.");
        Norm(AfterNorm, "after_norm");
        for (int i = 0; i < TpBlocks.Count; i++) Load(TpBlocks[i], $"tp_encoders.{i}.");
        if (TpNorm is not null) Norm(TpNorm, "tp_norm");
    }
}

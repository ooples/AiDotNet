using AiDotNet.Audio.MusicGen;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The sizes of FireRedTTS's streamable Mel codec (§3.3.2, laid out as SoCodec's <c>StagedVQVAE</c>).</summary>
/// <param name="Groups">The streams (§3.3.2: four), each a product of two codebooks of 128 (16,384 codewords).</param>
/// <param name="ConvGroups">The residual units' convolution groups (SoCodec: 32).</param>
internal sealed record FireRedTtsMelCodecConfiguration(int MelChannels, int Dim, int[] Strides, int CodebookSize, int CodebookDim, int Groups,
    int ConvGroups, double EmaDecay, double CommitmentWeight);

/// <summary>
/// FireRedTTS's streamable Mel codec (Guo et al. 2024, §3.3.2): the 10 ms, 100-band mel compressed into four streams of
/// 16,384 codewords at 20 ms and decoded back, built as SoCodec ([33], Guo et al. 2024; <c>staged_vqvae.py</c>) builds
/// its codec: causal convolutions (left-padded, so the codec streams), residual units of two grouped 3-wide causal
/// convolutions with Swish, strided causal convolutions down and causal transposed convolutions up, a product quantizer
/// of 4 groups × 2 codebooks with ordered group dropout, and a decoder that adds the speaker embedding to its input.
/// </summary>
/// <remarks>SoCodec's encoder reads HuBERT features; the paper's codec reads the mel, so the encoder's input is the mel and
/// its single decoder reconstructs it. GAN training (the paper: "GAN-based") uses MegaTTS's mel discriminator with LSGAN
/// losses, the paper and SoCodec giving no losses (<see cref="FireRedTTS{T}"/>).</remarks>
internal sealed class FireRedTtsMelCodec<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    internal sealed record Unit(NormedConv1DLayer<T> A, NormedConv1DLayer<T> B);

    public FireRedTtsMelCodec(IEngine engine, List<LayerBase<T>> layers, FireRedTtsMelCodecConfiguration configuration, Random random)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.Dim % c.ConvGroups != 0) throw new ArgumentException("The codec's width must divide by its convolution groups.", nameof(configuration));
        NormedConv1DLayer<T> Conv(int inputs, int outputs, int kernel, int stride = 1, int groups = 1, bool transposed = false)
        {
            var conv = new NormedConv1DLayer<T>(inputs, outputs, kernel, stride, 1, groups, 0, transposed, ConvolutionNormalization.None);
            layers.Add(conv);
            // nn.Conv1d's and nn.ConvTranspose1d's default: U(±1/√fan_in), fan_in = in/groups · k (out · k transposed).
            int fan = transposed ? outputs * kernel : inputs / groups * kernel;
            conv.LoadTorchWeights(IndexTtsOps.Uniform(random, conv.TorchWeightShape.Aggregate(1, (a, b) => a * b), fan), null,
                IndexTtsOps.Uniform(random, outputs, fan));
            return conv;
        }
        Unit MakeUnit() => new(Conv(c.Dim, c.Dim, 3, groups: c.ConvGroups), Conv(c.Dim, c.Dim, 3, groups: c.ConvGroups));
        int d = c.Dim;
        EncoderIn = Conv(c.MelChannels, d, 3);
        foreach (int stride in c.Strides) EncoderBlocks.Add((MakeUnit(), MakeUnit(), Conv(d, d, 2 * stride, stride)));
        EncoderOut = Conv(d, d, 3);
        Quantizer = new FireRedTtsProductQuantizer<T>(engine, layers,
            new FireRedTtsQuantizerConfiguration(d, c.CodebookSize, c.CodebookDim, 2 * c.Groups, c.Groups, c.EmaDecay, 0.0, c.CommitmentWeight, Ordered: true),
            random);
        DecoderIn = Conv(d, d, 3);
        foreach (int stride in c.Strides) DecoderBlocks.Add((Conv(d, d, 2 * stride, stride, transposed: true), MakeUnit(), MakeUnit()));
        DecoderOut = Conv(d, c.MelChannels, 3);
    }

    public FireRedTtsMelCodecConfiguration Configuration { get; }
    public NormedConv1DLayer<T> EncoderIn { get; }
    public List<(Unit First, Unit Second, NormedConv1DLayer<T> Down)> EncoderBlocks { get; } = new();
    public NormedConv1DLayer<T> EncoderOut { get; }
    public FireRedTtsProductQuantizer<T> Quantizer { get; }
    public NormedConv1DLayer<T> DecoderIn { get; }
    public List<(NormedConv1DLayer<T> Up, Unit First, Unit Second)> DecoderBlocks { get; } = new();
    public NormedConv1DLayer<T> DecoderOut { get; }

    /// <summary>The mel frames per codec frame.</summary>
    public int Downsampling => Configuration.Strides.Aggregate(1, (a, b) => a * b);

    // CausalConv1d over [1, C, L]: left-padded by k − 1.
    private Tensor<T> Causal(NormedConv1DLayer<T> conv, Tensor<T> x, int kernel)
    {
        var padded = _engine.TensorConcatenate(new[] { new Tensor<T>(new[] { 1, x.Shape[1], kernel - 1 }), x }, 2);
        return conv.Forward(padded);
    }

    // CausalConvTranspose1d: the transposed convolution without padding, the last k − stride samples cut.
    private Tensor<T> CausalUp(NormedConv1DLayer<T> conv, Tensor<T> x, int kernel, int stride)
    {
        var y = conv.Forward(x);
        int keep = y.Shape[2] - (kernel - stride);
        return _engine.TensorSlice(y, new[] { 0, 0, 0 }, new[] { 1, y.Shape[1], keep });
    }

    private Tensor<T> ResidualUnit(Unit unit, Tensor<T> x) =>
        _engine.TensorAdd(x, Causal(unit.B, _engine.Swish(Causal(unit.A, x, 3)), 3));

    private Tensor<T> Channels(Tensor<T> rows) => _engine.Reshape(_engine.TensorTranspose(rows), new[] { 1, rows.Shape[1], rows.Shape[0] });

    private Tensor<T> Rows(Tensor<T> channels) => _engine.TensorTranspose(_engine.Reshape(channels, new[] { channels.Shape[1], channels.Shape[2] }));

    /// <summary>The encodings <c>[⌈F / downsampling⌉, Dim]</c> of a mel <c>[F, mel]</c>.</summary>
    public Tensor<T> Encode(Tensor<T> mel)
    {
        var x = _engine.Swish(Causal(EncoderIn, Channels(mel), 3));
        for (int i = 0; i < EncoderBlocks.Count; i++)
        {
            var (first, second, down) = EncoderBlocks[i];
            int stride = Configuration.Strides[i];
            x = _engine.Swish(ResidualUnit(first, x));
            x = _engine.Swish(ResidualUnit(second, x));
            x = _engine.Swish(Causal(down, x, 2 * stride));
        }
        return Rows(Causal(EncoderOut, x, 3));
    }

    /// <summary>The mel <c>[n · downsampling, mel]</c> of a quantized sequence <c>[n, Dim]</c>, the speaker embedding
    /// <c>[Dim]</c> added to the decoder's input.</summary>
    public Tensor<T> Decode(Tensor<T> quantized, Tensor<T> speaker)
    {
        int n = quantized.Shape[0], d = Configuration.Dim;
        var x = Channels(_engine.TensorAdd(quantized, _engine.TensorTile(_engine.Reshape(speaker, new[] { 1, d }), new[] { n, 1 })));
        x = _engine.Swish(Causal(DecoderIn, x, 3));
        for (int i = 0; i < DecoderBlocks.Count; i++)
        {
            var (up, first, second) = DecoderBlocks[i];
            int stride = Configuration.Strides[i];
            x = _engine.Swish(CausalUp(up, x, 2 * stride, stride));
            x = _engine.Swish(ResidualUnit(first, x));
            x = _engine.Swish(ResidualUnit(second, x));
        }
        return Rows(Causal(DecoderOut, x, 3));
    }

    /// <summary>The four streams' tokens <c>[⌈F / downsampling⌉, Groups]</c> of a mel.</summary>
    public int[,] Tokenize(Tensor<T> mel) => Quantizer.GroupTokens(Quantizer.Project(Encode(mel)));

    /// <summary>The mel of tokens <c>[n, Groups]</c>.</summary>
    public Tensor<T> Detokenize(int[,] tokens, Tensor<T> speaker) => Decode(Quantizer.Decode(tokens), speaker);

    /// <summary>Loads SoCodec's names (<c>{encoder}layers.*</c>, <c>{decoder}layers.*</c>, <c>{quantizer}quantizers.0.*</c>):
    /// in an encoder, conv, Swish, then per stride a block (unit, norm, Swish, unit, norm, Swish, strided conv), norm and
    /// Swish, then the last conv; a decoder mirrors it with transposed convolutions; a unit is conv, norm, Swish, conv.</summary>
    public void Load(Func<string, int[], double[]> read, Func<string, bool> has, string encoder, string decoder, string quantizer)
    {
        void Conv(NormedConv1DLayer<T> conv, string name) =>
            conv.LoadTorchWeights(read(name + ".weight", conv.TorchWeightShape), null, read(name + ".bias", new[] { conv.OutputChannels }));
        void LoadUnit(Unit unit, string name)
        {
            Conv(unit.A, name + ".layers.0");
            Conv(unit.B, name + ".layers.3");
        }
        Conv(EncoderIn, encoder + "layers.0");
        for (int i = 0; i < EncoderBlocks.Count; i++)
        {
            string p = $"{encoder}layers.{2 + 3 * i}.layers.";
            LoadUnit(EncoderBlocks[i].First, p + "0");
            LoadUnit(EncoderBlocks[i].Second, p + "3");
            Conv(EncoderBlocks[i].Down, p + "6");
        }
        Conv(EncoderOut, $"{encoder}layers.{2 + 3 * EncoderBlocks.Count}");
        Conv(DecoderIn, decoder + "layers.0");
        for (int i = 0; i < DecoderBlocks.Count; i++)
        {
            string p = $"{decoder}layers.{2 + 3 * i}.layers.";
            Conv(DecoderBlocks[i].Up, p + "0");
            LoadUnit(DecoderBlocks[i].First, p + "3");
            LoadUnit(DecoderBlocks[i].Second, p + "6");
        }
        Conv(DecoderOut, $"{decoder}layers.{2 + 3 * DecoderBlocks.Count}");
        Quantizer.Load(read, has, quantizer + "quantizers.0.");
    }
}

/// <summary>The sizes of FireRedTTS's multi-stream acoustic language model (§3.3.2; MusicGen-small's decoder).</summary>
/// <param name="Lookahead">d in sᵢ₋d + aᵢ: the frames the semantic embedding lags the acoustic one.</param>
internal sealed record FireRedTtsAcousticLmConfiguration(int Streams, int StreamTokens, int SemanticTokens, int ModelDim, int Heads, int Layers,
    int FeedForwardDim, double Dropout, int[] Delays, int Lookahead, int SpeakerDim, int ReferenceHidden);

/// <summary>
/// FireRedTTS's multi-stream acoustic language model (§3.3.2): MusicGen's decoder ([34], Copet et al. 2023) over the
/// Mel codec's four streams with MusicGen's delay pattern, conditioned by the semantic tokens embedded, upsampled to the
/// acoustic rate by repetition and added d frames late (sᵢ₋d + aᵢ, the "lookahead pattern"), and by the speaker
/// embedding as a prefix latent (the text-to-speech GPT's <c>reference_embedding</c> form). The acoustic sequence's
/// length follows the semantic one (two frames per token), so generation needs no stop token.
/// </summary>
internal sealed class FireRedTtsAcousticLm<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public FireRedTtsAcousticLm(IEngine engine, List<LayerBase<T>> layers, FireRedTtsAcousticLmConfiguration configuration, Random random)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        Core = new MusicGenCore<T>(engine, layers, new MusicGenConfiguration(c.Streams, c.StreamTokens, c.ModelDim, c.Heads, c.Layers,
            c.FeedForwardDim, c.Dropout, c.Delays, CrossAttention: false));
        Core.InitializeLikeReference(random);
        Semantic = IndexTtsOps.Embedding(layers, random, c.SemanticTokens, c.ModelDim, 1.0);
        Reference1 = IndexTtsOps.Linear(engine, layers, random, c.SpeakerDim, c.ReferenceHidden);
        Reference2 = IndexTtsOps.Linear(engine, layers, random, c.ReferenceHidden, c.ModelDim);
    }

    public FireRedTtsAcousticLmConfiguration Configuration { get; }
    public MusicGenCore<T> Core { get; }
    public TiedEmbeddingLayer<T> Semantic { get; }
    public DenseLayer<T> Reference1 { get; }
    public DenseLayer<T> Reference2 { get; }

    /// <summary>The speaker prefix <c>[1, ModelDim]</c>.</summary>
    public Tensor<T> Prefix(Tensor<T> speaker) =>
        Reference2.Forward(_engine.Tanh(Reference1.Forward(_engine.Reshape(speaker, new[] { 1, Configuration.SpeakerDim }))));

    /// <summary>
    /// The rows <c>[steps, ModelDim]</c> added to the delay pattern's steps: step s carries acoustic frame i = s − 1 (the
    /// first stream's frame at that step) and gets the semantic embedding of token ⌊(i − d) / 2⌋ (zero before the first
    /// frame and past the last).
    /// </summary>
    public Tensor<T> Conditioning(IReadOnlyList<int> semantic, int steps)
    {
        var c = Configuration;
        int frames = 2 * semantic.Count;
        var embedded = Semantic.Forward(IndexTtsOps.Ids<T>(semantic));                                     // [n, D]
        var index = new List<int>();
        var keep = new Tensor<T>(new[] { steps, 1 });
        for (int s = 0; s < steps; s++)
        {
            int frame = s - 1 - c.Lookahead;
            bool valid = frame >= 0 && frame < frames;
            index.Add(valid ? frame / 2 : 0);
            keep[s, 0] = valid ? NumOps.One : NumOps.Zero;
        }
        var rows = _engine.TensorIndexSelect(embedded, new Tensor<int>(new[] { steps }, new Vector<int>(index.ToArray())), 0);
        return _engine.TensorMultiply(rows, _engine.TensorTile(keep, new[] { 1, c.ModelDim }));
    }

    /// <summary>MusicGen's cross-entropy of the streams <c>[frames, Streams]</c> (frames = 2 × the semantic tokens).</summary>
    public Tensor<T> Loss(int[,] streams, IReadOnlyList<int> semantic, Tensor<T> speaker, bool training, Random random)
    {
        int frames = streams.GetLength(0), k = Configuration.Streams;
        if (frames != 2 * semantic.Count) throw new ArgumentException("The acoustic streams hold two frames per semantic token.", nameof(streams));
        var codes = new int[k, frames];
        for (int t = 0; t < frames; t++)
            for (int q = 0; q < k; q++) codes[q, t] = streams[t, q];
        return Core.Loss(codes, Prefix(speaker), training, random, Conditioning(semantic, frames + 1));
    }

    /// <summary>The streams <c>[2n, Streams]</c> for n semantic tokens, each code drawn by <paramref name="choose"/>.</summary>
    public int[,] Generate(IReadOnlyList<int> semantic, Tensor<T> speaker, Func<double[], int> choose)
    {
        var c = Configuration;
        int frames = 2 * semantic.Count, steps = frames + c.Delays.Max() + 1;
        var codes = Core.Generate(Prefix(speaker), null, 1.0, frames, choose, null, Conditioning(semantic, steps));
        var streams = new int[frames, c.Streams];
        for (int t = 0; t < frames; t++)
            for (int q = 0; q < c.Streams; q++) streams[t, q] = codes[q, t];
        return streams;
    }
}

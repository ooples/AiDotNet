using AiDotNet.Audio.Features;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>
/// NATSpeech's log-mel spectrogram (<c>librosa_wav2spec</c>): a centred STFT padded with zeros (periodic Hann window),
/// its magnitude through librosa's Slaney mel filterbank, and <c>log10(max(1e-6, x))</c>: <c>[frames, mels]</c>.
/// </summary>
internal sealed class MegaTtsMel
{
    private readonly double[] _window;
    private readonly double[,] _basis;

    public MegaTtsMel(int sampleRate, int fftSize, int hopSize, int windowSize, int mels, double fMin, double fMax)
    {
        FftSize = fftSize;
        HopSize = hopSize;
        Mels = mels;
        _window = new double[fftSize];
        int offset = (fftSize - windowSize) / 2;
        for (int i = 0; i < windowSize; i++) _window[offset + i] = 0.5 - 0.5 * Math.Cos(2 * Math.PI * i / windowSize);
        _basis = MelFilterbanks.Slaney(sampleRate, fftSize, mels, fMin, fMax);
    }

    public int FftSize { get; }
    public int HopSize { get; }
    public int Mels { get; }

    public double[,] LogMel(double[] audio)
    {
        int pad = FftSize / 2, frames = 1 + audio.Length / HopSize, bins = FftSize / 2 + 1;
        var padded = new double[audio.Length + 2 * pad];
        Array.Copy(audio, 0, padded, pad, audio.Length);
        var re = new double[frames * FftSize];
        for (int f = 0; f < frames; f++)
            for (int k = 0; k < FftSize; k++) re[f * FftSize + k] = padded[f * HopSize + k] * _window[k];
        AiDotNet.Tensors.Engines.AiDotNetEngine.Current.FFT(new Tensor<double>(new[] { frames, FftSize }, new Vector<double>(re)),
            new Tensor<double>(new[] { frames, FftSize }), out var outRe, out var outIm);
        var sRe = outRe.AsSpan();
        var sIm = outIm.AsSpan();
        var mel = new double[frames, Mels];
        var magnitude = new double[bins];
        for (int f = 0; f < frames; f++)
        {
            for (int k = 0; k < bins; k++)
            {
                double a = sRe[f * FftSize + k], b = sIm[f * FftSize + k];
                magnitude[k] = Math.Sqrt(a * a + b * b);
            }
            for (int m = 0; m < Mels; m++)
            {
                double acc = 0;
                for (int k = 0; k < bins; k++) acc += _basis[m, k] * magnitude[k];
                mel[f, m] = Math.Log10(Math.Max(1e-6, acc));
            }
        }
        return mel;
    }
}

/// <summary>
/// The VQ-VAE bottleneck (van den Oord et al. 2017), as Mega-TTS's Eq. 1 writes it: each vector replaced by its nearest
/// codebook entry (straight-through gradients), with the codebook term ‖sg[E(y)] − z_q‖² and the commitment term
/// ‖sg[z_q] − E(y)‖², both unweighted.
/// </summary>
internal sealed class MegaTtsVectorQuantizer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public MegaTtsVectorQuantizer(IEngine engine, List<LayerBase<T>> layers, int codes, int dimension)
    {
        _engine = engine;
        Codes = codes;
        Dimension = dimension;
        Codebook = new TiedEmbeddingLayer<T>(codes, dimension);
        layers.Add(Codebook);
    }

    public int Codes { get; }
    public int Dimension { get; }
    public TiedEmbeddingLayer<T> Codebook { get; }

    private Tensor<T> Table => _engine.Reshape(Codebook.Forward(new Tensor<T>(new[] { Codes }, new Vector<T>(Enumerable.Range(0, Codes).Select(i => NumOps.FromDouble(i)).ToArray()))),
        new[] { Codes, Dimension });

    /// <summary>The nearest codes <c>[B, L]</c> of vectors <c>[B, L, d]</c>.</summary>
    public int[,] Encode(Tensor<T> x)
    {
        int b = x.Shape[0], l = x.Shape[1];
        Tensor<T> table;
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>()) table = Table;
        var codes = new int[b, l];
        for (int i = 0; i < b; i++)
            for (int j = 0; j < l; j++)
            {
                double best = double.PositiveInfinity;
                for (int c = 0; c < Codes; c++)
                {
                    double distance = 0;
                    for (int d = 0; d < Dimension; d++)
                    {
                        double diff = NumOps.ToDouble(x[i, j, d]) - NumOps.ToDouble(table[c, d]);
                        distance += diff * diff;
                    }
                    if (distance < best) { best = distance; codes[i, j] = c; }
                }
            }
        return codes;
    }

    /// <summary>The codebook vectors <c>[B, L, d]</c> of codes <c>[B, L]</c>.</summary>
    public Tensor<T> Lookup(int[,] codes)
    {
        int b = codes.GetLength(0), l = codes.GetLength(1);
        var ids = new Tensor<T>(new[] { b * l });
        for (int i = 0; i < b; i++)
            for (int j = 0; j < l; j++) ids[i * l + j] = NumOps.FromDouble(codes[i, j]);
        return _engine.Reshape(Codebook.Forward(ids), new[] { b, l, Dimension });
    }

    /// <summary>Quantizes <c>[B, L, d]</c>: the straight-through output, the codes, and the codebook plus commitment loss
    /// over the positions <paramref name="mask"/> <c>[B, L, 1]</c> keeps.</summary>
    public (Tensor<T> Quantized, int[,] Codes, Tensor<T> Loss) Forward(Tensor<T> x, Tensor<T> mask)
    {
        var codes = Encode(x);
        var zq = Lookup(codes);
        var shape = x.Shape.ToArray();
        var m = _engine.TensorBroadcastTo(mask, shape);
        var stopX = _engine.StopGradient(x);
        var stopQ = _engine.StopGradient(zq);
        Tensor<T> Sse(Tensor<T> a, Tensor<T> c)
        {
            var d = _engine.TensorMultiply(_engine.TensorSubtract(a, c), m);
            return _engine.ReduceSum(_engine.TensorMultiply(d, d), Enumerable.Range(0, shape.Length).ToArray(), false);
        }
        double count = Math.Max(1, m.ToArray().Sum(v => NumOps.ToDouble(v)));
        var loss = _engine.TensorMultiplyScalar(_engine.TensorAdd(Sse(stopX, zq), Sse(stopQ, x)), NumOps.FromDouble(1 / count));
        // Straight-through: x + sg[z_q − x].
        var quantized = _engine.TensorAdd(x, _engine.StopGradient(_engine.TensorSubtract(zq, x)));
        return (quantized, codes, loss);
    }
}

/// <summary>Mega-TTS's sizes (Jiang et al. 2023, Table 7).</summary>
internal sealed record MegaTtsConfiguration
{
    public int PhoneVocabulary { get; init; } = 200;
    public int Hidden { get; init; } = 320;
    public int ContentLayers { get; init; } = 4;
    public int ContentHeads { get; init; } = 2;
    public int ContentKernel { get; init; } = 5;
    public int ProsodyBlocks { get; init; } = 5;
    public int ProsodyKernel { get; init; } = 5;
    public int ProsodyBins { get; init; } = 20;
    public int Codes { get; init; } = 2048;
    public int CodeDimension { get; init; } = 256;
    public int TimbreBlocks { get; init; } = 5;
    public int TimbreKernel { get; init; } = 31;
    public int DecoderBlocks { get; init; } = 5;
    public int DecoderKernel { get; init; } = 5;
    public int Mels { get; init; } = 80;
    public int DurationLayers { get; init; } = 3;
    public int DurationKernel { get; init; } = 3;
    public int LmLayers { get; init; } = 8;
    public int LmHidden { get; init; } = 512;
    public int LmHeads { get; init; } = 8;
    public int LmKernel { get; init; } = 5;
    public int[] DiscriminatorWindows { get; init; } = { 32, 64, 128 };
    public int DiscriminatorHidden { get; init; } = 192;
}

/// <summary>
/// Mega-TTS (Jiang et al. 2023): the VQGAN-based acoustic model (a content encoder, a prosody encoder with a VQ bottleneck
/// at the phoneme level, a timbre encoder averaged over time, a duration predictor, a mel decoder and a multi-window
/// discriminator) and the prosody large language model (P-LLM), a causal Transformer over the phoneme-level prosody codes
/// conditioned on the content and timbre.
/// </summary>
internal sealed class MegaTtsNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public MegaTtsNetwork(IEngine engine, List<LayerBase<T>> acoustic, List<LayerBase<T>> lm, List<LayerBase<T>> discriminator, MegaTtsConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        Content = new MegaTtsContentEncoder<T>(engine, acoustic, c.PhoneVocabulary, c.Hidden, c.ContentLayers, c.ContentHeads, c.ContentKernel);
        ProsodyPrenet = MegaTts3Ops.Linear(acoustic, c.Hidden);
        ProsodyFrames = new MegaTtsConvBlocks<T>(engine, acoustic, c.Hidden, c.Hidden, c.ProsodyKernel, c.ProsodyBlocks);
        ProsodyPhones = new MegaTtsConvBlocks<T>(engine, acoustic, c.Hidden, c.CodeDimension, c.ProsodyKernel, c.ProsodyBlocks);
        Quantizer = new MegaTtsVectorQuantizer<T>(engine, acoustic, c.Codes, c.CodeDimension);
        ProsodyOut = MegaTts3Ops.Linear(acoustic, c.Hidden);
        TimbrePrenet = MegaTts3Ops.Linear(acoustic, c.Hidden);
        Timbre = new MegaTtsConvBlocks<T>(engine, acoustic, c.Hidden, c.Hidden, c.TimbreKernel, c.TimbreBlocks);
        Duration = new MegaTtsDurationPredictor<T>(engine, acoustic, c.Hidden, c.Hidden, c.DurationLayers, c.DurationKernel);
        Decoder = new MegaTtsConvBlocks<T>(engine, acoustic, c.Hidden, c.Mels, c.DecoderKernel, c.DecoderBlocks);
        CodeEmbedding = new TiedEmbeddingLayer<T>(c.Codes + 2, c.LmHidden);
        lm.Add(CodeEmbedding);
        LmContent = MegaTts3Ops.Linear(lm, c.LmHidden);
        LmTimbre = MegaTts3Ops.Linear(lm, c.LmHidden);
        for (int i = 0; i < c.LmLayers; i++) LmLayers.Add(new MegaTtsTransformerLayer<T>(engine, lm, c.LmHidden, c.LmHeads, c.LmKernel, decoder: true));
        LmNorm = new LayerNormalizationLayer<T>(c.LmHidden, 1e-5);
        lm.Add(LmNorm);
        LmHead = MegaTts3Ops.Linear(lm, c.Codes + 2);
        Discriminator = new MegaTtsMelDiscriminator<T>(engine, discriminator, c.DiscriminatorWindows, c.Mels, c.DiscriminatorHidden);
    }

    public MegaTtsConfiguration Configuration { get; }
    public MegaTtsContentEncoder<T> Content { get; }
    public DenseLayer<T> ProsodyPrenet { get; }
    public MegaTtsConvBlocks<T> ProsodyFrames { get; }
    public MegaTtsConvBlocks<T> ProsodyPhones { get; }
    public MegaTtsVectorQuantizer<T> Quantizer { get; }
    public DenseLayer<T> ProsodyOut { get; }
    public DenseLayer<T> TimbrePrenet { get; }
    public MegaTtsConvBlocks<T> Timbre { get; }
    public MegaTtsDurationPredictor<T> Duration { get; }
    public MegaTtsConvBlocks<T> Decoder { get; }
    public TiedEmbeddingLayer<T> CodeEmbedding { get; }
    public DenseLayer<T> LmContent { get; }
    public DenseLayer<T> LmTimbre { get; }
    public List<MegaTtsTransformerLayer<T>> LmLayers { get; } = new();
    public LayerNormalizationLayer<T> LmNorm { get; }
    public DenseLayer<T> LmHead { get; }
    public MegaTtsMelDiscriminator<T> Discriminator { get; }

    /// <summary>The start token of a prosody sequence.</summary>
    public int StartCode => Configuration.Codes;

    /// <summary>The phone-level hidden states <c>[1, L, ProsodyDim]</c> before quantization: the low band of the mel (its
    /// first bins) through the first stack, averaged over each phone's frames (<paramref name="mel2ph"/>, 1-based), and the
    /// second stack.</summary>
    public Tensor<T> ProsodyHidden(Tensor<T> mel, int[] mel2ph, int phones)
    {
        var c = Configuration;
        int frames = mel.Shape[0];
        var low = _engine.Reshape(_engine.TensorSlice(mel, new[] { 0, 0 }, new[] { frames, c.ProsodyBins }), new[] { 1, frames, c.ProsodyBins });
        var h = ProsodyFrames.Forward(MegaTts3Ops.Rows(_engine, ProsodyPrenet, low), Ones(frames));
        // Phoneme-level pooling: the mean of each phone's frames.
        var pool = new Tensor<T>(new[] { 1, phones, frames });
        var counts = new int[phones];
        foreach (int p in mel2ph)
            if (p > 0) counts[p - 1]++;
        for (int f = 0; f < frames; f++)
            if (mel2ph[f] > 0) pool[0, mel2ph[f] - 1, f] = NumOps.FromDouble(1.0 / counts[mel2ph[f] - 1]);
        var pooled = _engine.BatchMatMul(pool, h);
        var mask = new Tensor<T>(new[] { 1, phones, 1 });
        for (int i = 0; i < phones; i++)
            if (counts[i] > 0) mask[0, i, 0] = NumOps.One;
        return ProsodyPhones.Forward(pooled, mask);
    }

    /// <summary>The prosody states <c>[1, L, h]</c> of quantized vectors.</summary>
    public Tensor<T> ProsodyStates(Tensor<T> quantized) => MegaTts3Ops.Rows(_engine, ProsodyOut, quantized);

    /// <summary>The global timbre vector <c>[1, h]</c> of a reference mel <c>[frames, mels]</c>: the stack over its
    /// frames, averaged over time.</summary>
    public Tensor<T> TimbreVector(Tensor<T> referenceMel)
    {
        int frames = referenceMel.Shape[0];
        var x = MegaTts3Ops.Rows(_engine, TimbrePrenet, _engine.Reshape(referenceMel, new[] { 1, frames, referenceMel.Shape[1] }));
        var h = Timbre.Forward(x, Ones(frames));
        return _engine.ReduceMean(h, new[] { 1 }, false);
    }

    private Tensor<T> Ones(int frames)
    {
        var ones = new Tensor<T>(new[] { 1, frames, 1 });
        for (int f = 0; f < frames; f++) ones[0, f, 0] = NumOps.One;
        return ones;
    }

    /// <summary>The mel <c>[frames, mels]</c> the decoder makes from the phone-level content and prosody, expanded by
    /// <paramref name="mel2ph"/>, plus the timbre vector.</summary>
    public Tensor<T> DecodeMel(Tensor<T> content, Tensor<T> prosody, Tensor<T> timbre, int[] mel2ph)
    {
        var c = Configuration;
        int l = content.Shape[1], frames = mel2ph.Length;
        var phoneLevel = _engine.TensorAdd(content, prosody);
        var expand = new Tensor<T>(new[] { 1, frames, l });
        for (int f = 0; f < frames; f++)
            if (mel2ph[f] > 0) expand[0, f, mel2ph[f] - 1] = NumOps.One;
        var x = _engine.BatchMatMul(expand, phoneLevel);
        x = _engine.TensorAdd(x, _engine.TensorBroadcastTo(_engine.Reshape(timbre, new[] { 1, 1, c.Hidden }), new[] { 1, frames, c.Hidden }));
        return _engine.Reshape(Decoder.Forward(x, Ones(frames)), new[] { frames, c.Mels });
    }

    /// <summary>The predicted durations <c>[L]</c> from the content and prosody (the input's gradient scaled by 0.1, as
    /// NATSpeech's <c>predictor_grad</c>), zero on the padding phones (id 0).</summary>
    public Tensor<T> PredictDurations(Tensor<T> content, Tensor<T> prosody, int[] phones)
    {
        int l = content.Shape[1];
        var input = _engine.TensorAdd(content, prosody);
        var stop = _engine.StopGradient(input);
        input = _engine.TensorAdd(stop, _engine.TensorMultiplyScalar(_engine.TensorSubtract(input, stop), NumOps.FromDouble(0.1)));
        var padding = new bool[1, l];
        for (int i = 0; i < l; i++) padding[0, i] = phones[i] == 0;
        return _engine.Reshape(Duration.Forward(input, padding), new[] { l });
    }

    /// <summary>Loads a state dict in the module names of <c>tools/reference-data/megatts_model_reference.py</c>'s
    /// composition of the NATSpeech and SyntaSpeech modules: <c>content</c>, <c>prosody_prenet</c>,
    /// <c>prosody_frames</c>, <c>prosody_phones</c>, <c>codebook</c>, <c>prosody_out</c>, <c>timbre_prenet</c>,
    /// <c>timbre</c>, <c>duration</c>, <c>decoder</c>, <c>code_embed</c>, <c>lm_content</c>, <c>lm_timbre</c>,
    /// <c>lm_layers</c>, <c>lm_norm</c>, <c>lm_head</c> and <c>disc</c>.</summary>
    public void LoadTorch(Func<string, double[]> read)
    {
        var c = Configuration;
        Content.LoadTorch(_engine, read, "content");
        MegaTts3Ops.LoadDense(_engine, read, ProsodyPrenet, "prosody_prenet", c.ProsodyBins, c.Hidden);
        ProsodyFrames.LoadTorch(_engine, read, "prosody_frames");
        ProsodyPhones.LoadTorch(_engine, read, "prosody_phones");
        Quantizer.Codebook.LoadTable(read("codebook.weight"));
        MegaTts3Ops.LoadDense(_engine, read, ProsodyOut, "prosody_out", c.CodeDimension, c.Hidden);
        MegaTts3Ops.LoadDense(_engine, read, TimbrePrenet, "timbre_prenet", c.Mels, c.Hidden);
        Timbre.LoadTorch(_engine, read, "timbre");
        Duration.LoadTorch(_engine, read, "duration");
        Decoder.LoadTorch(_engine, read, "decoder");
        CodeEmbedding.LoadTable(read("code_embed.weight"));
        MegaTts3Ops.LoadDense(_engine, read, LmContent, "lm_content", c.Hidden, c.LmHidden);
        MegaTts3Ops.LoadDense(_engine, read, LmTimbre, "lm_timbre", c.Hidden, c.LmHidden);
        for (int i = 0; i < LmLayers.Count; i++) LmLayers[i].LoadTorch(_engine, read, $"lm_layers.{i}.op");
        CodecBased.TorchParameters.LayerNorm(_engine, LmNorm, read("lm_norm.weight"), read("lm_norm.bias"));
        MegaTts3Ops.LoadDense(_engine, read, LmHead, "lm_head", c.LmHidden, c.Codes + 2);
        Discriminator.LoadTorch(read, "disc");
    }

    /// <summary>The P-LLM's logits <c>[L, codes + 2]</c> for previous codes (the start token first) with each position's
    /// phone content <c>[1, L, h]</c> and the timbre vector <c>[1, h]</c>.</summary>
    public Tensor<T> ProsodyLogits(int[] previous, Tensor<T> content, Tensor<T> timbre)
    {
        var c = Configuration;
        int l = previous.Length;
        var ids = new Tensor<T>(new[] { l }, new Vector<T>(previous.Select(p => NumOps.FromDouble(p)).ToArray()));
        var x = _engine.Reshape(CodeEmbedding.Forward(ids), new[] { 1, l, c.LmHidden });
        x = _engine.TensorAdd(x, MegaTts3Ops.Rows(_engine, LmContent, content));
        x = _engine.TensorAdd(x, _engine.TensorBroadcastTo(_engine.Reshape(MegaTts3Ops.Rows(_engine, LmTimbre, timbre), new[] { 1, 1, c.LmHidden }), new[] { 1, l, c.LmHidden }));
        x = _engine.TensorAdd(x, MegaTtsPositions.Of<T>(new bool[1, l], c.LmHidden));
        foreach (var layer in LmLayers) x = layer.Forward(x, null, null);
        return _engine.Reshape(MegaTts3Ops.Rows(_engine, LmHead, LmNorm.Forward(x)), new[] { l, c.Codes + 2 });
    }
}

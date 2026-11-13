using AiDotNet.ModelLoading;
using AiDotNet.ActivationFunctions;
using AiDotNet.Audio.Features;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Which S3 tokenizer: CosyVoice 1's or CosyVoice 2's.</summary>
internal enum S3TokenizerVersion
{
    /// <summary><c>speech_tokenizer_v1</c>: sinusoidal positions and a 4,096-entry codebook.</summary>
    V1,

    /// <summary><c>speech_tokenizer_v2_25hz</c>: rotary positions, an FSMN memory in every attention block and a finite
    /// scalar quantizer of 3⁸ = 6,561 codes.</summary>
    V2,
}

/// <summary>The sizes of a CosyVoice supervised semantic speech tokenizer (S3Tokenizer, <c>ModelConfig</c>).</summary>
/// <param name="FirstStride">The first convolution's stride: 1 for <c>speech_tokenizer_v1</c> (50 tokens a second),
/// 2 for <c>speech_tokenizer_v1_25hz</c> and v2 (25).</param>
/// <param name="Version">v1 or v2 (CosyVoice 2's); v2 needs 64-wide heads, as its rotary table is fixed at 64.</param>
internal sealed record S3TokenizerConfiguration(int Mels = 128, int Context = 1500, int State = 1280, int Heads = 20, int Layers = 6,
    int CodebookSize = 4096, int FirstStride = 1, S3TokenizerVersion Version = S3TokenizerVersion.V1)
{
    /// <summary>CosyVoice 2's <c>speech_tokenizer_v2_25hz</c>.</summary>
    public static S3TokenizerConfiguration V2Default => new(CodebookSize: 6561, FirstStride: 2, Version: S3TokenizerVersion.V2);

    /// <summary>Tokens per second of 16 kHz audio (the 100 Hz mel frames, halved by the second convolution and divided
    /// by the first's stride).</summary>
    public int TokenRate => 50 / FirstStride;
}

/// <summary>
/// CosyVoice's supervised semantic speech tokenizer S3 (Du et al. 2024, §3.1), v1: a Whisper-style audio encoder (two
/// GELU convolutions over a 128-band log-mel, sinusoidal positions, pre-norm self-attention blocks) whose states are
/// quantized to the nearest entry of a codebook after L2 normalization.
/// </summary>
/// <remarks>
/// <para>Ported from xingchensong/S3Tokenizer (<c>s3tokenizer/model.py</c>, Apache-2.0), the PyTorch form of the
/// released <c>speech_tokenizer_v1.onnx</c>; state-dict names follow it (<c>encoder.*</c>,
/// <c>quantizer._codebook.embed</c>).</para>
/// <para>Utterances longer than 30 s (3000 frames) are cut into 30 s windows overlapping by 4 s, tokenized separately
/// and merged by dropping half the overlap from each side of every seam, as <c>_quantize_mixed_batch</c> does.</para>
/// <para>The tokenizer is released frozen: its supervised ASR training is not part of the release, so this module is
/// inference-only.</para>
/// </remarks>
internal sealed class S3Tokenizer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private const int SampleRate = 16000, FftSize = 400, Hop = 160, WindowFrames = 3000, OverlapSeconds = 4;
    private readonly IEngine _engine;
    private Tensor<T> _positions;
    private Tensor<T> _codebook;

    internal sealed record Block(LayerNormalizationLayer<T> AttentionNorm, DenseLayer<T> Query, BiasFreeLinearLayer<T> Key,
        DenseLayer<T> Value, DenseLayer<T> Output, LayerNormalizationLayer<T> MlpNorm, DenseLayer<T> Mlp1, DenseLayer<T> Mlp2)
    {
        /// <summary>v2's FSMN memory: a depthwise 31-tap convolution over the values (no bias); null for v1.</summary>
        public NormedConv1DLayer<T>? Fsmn { get; init; }
    }

    // v2's FSMN kernel and rotary table width (precompute_freqs_cis(64, …)).
    private const int FsmnKernel = 31, RotaryDim = 64;

    public S3Tokenizer(IEngine engine, List<LayerBase<T>> layers, S3TokenizerConfiguration configuration, Random initialization)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.State <= 0 || c.Heads <= 0)
            throw new ArgumentOutOfRangeException(nameof(configuration), "State and heads must be positive.");
        if (c.State % c.Heads != 0) throw new ArgumentException("The width must divide by the heads.", nameof(configuration));
        if (c.Version == S3TokenizerVersion.V2 && c.State / c.Heads != RotaryDim)
            throw new ArgumentException($"S3Tokenizer v2's rotary table is {RotaryDim} wide; its heads must be too.", nameof(configuration));
        if (c.Version == S3TokenizerVersion.V2 && c.CodebookSize != 6561)
            throw new ArgumentException("S3Tokenizer v2 quantizes to 3^8 = 6,561 codes.", nameof(configuration));
        TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T> { layers.Add(layer); return layer; }
        double Uniform(int fanIn) => (2 * initialization.NextDouble() - 1) / Math.Sqrt(fanIn);
        DenseLayer<T> Linear(int inputs, int outputs)
        {
            var layer = Add(new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()));
            TorchParameters.Linear(engine, layer, inputs, outputs, Enumerable.Range(0, inputs * outputs).Select(_ => Uniform(inputs)).ToArray(),
                Enumerable.Range(0, outputs).Select(_ => Uniform(inputs)).ToArray());
            return layer;
        }
        NormedConv1DLayer<T> Conv(int inputs, int stride)
        {
            var conv = Add(new NormedConv1DLayer<T>(inputs, c.State, 3, stride, 1, 1, 1, false, ConvolutionNormalization.None));
            conv.Reinitialize(() => Uniform(inputs * 3));
            return conv;
        }

        Conv1 = Conv(c.Mels, c.FirstStride);
        Conv2 = Conv(c.State, 2);
        for (int i = 0; i < c.Layers; i++)
        {
            var attentionNorm = Add(new LayerNormalizationLayer<T>(c.State, 1e-5));
            var query = Linear(c.State, c.State);
            var key = Add(new BiasFreeLinearLayer<T>(c.State, c.State));
            TorchParameters.Linear(key, c.State, c.State, Enumerable.Range(0, c.State * c.State).Select(_ => Uniform(c.State)).ToArray());
            var value = Linear(c.State, c.State);
            var output = Linear(c.State, c.State);
            var mlpNorm = Add(new LayerNormalizationLayer<T>(c.State, 1e-5));
            NormedConv1DLayer<T>? fsmn = null;
            if (c.Version == S3TokenizerVersion.V2)
            {
                fsmn = Add(new NormedConv1DLayer<T>(c.State, c.State, FsmnKernel, 1, 1, c.State, 0, false, ConvolutionNormalization.None, useBias: false));
                fsmn.Reinitialize(() => Uniform(FsmnKernel));
            }
            Blocks.Add(new Block(attentionNorm, query, key, value, output, mlpNorm, Linear(c.State, 4 * c.State), Linear(4 * c.State, c.State))
            {
                Fsmn = fsmn,
            });
        }
        if (c.Version == S3TokenizerVersion.V2)
        {
            _positions = new Tensor<T>(new[] { 0, c.State });
            _codebook = new Tensor<T>(new[] { 0, c.State });
            ProjectDown = Linear(c.State, 8);
        }
        else
        {
            _positions = Sinusoids(c.Context, c.State);
            // The released codebook is learned; a fresh tokenizer draws one from N(0, 1) until weights are loaded.
            _codebook = new Tensor<T>(new[] { c.CodebookSize, c.State }, TorchParameters.Gaussian<T>(initialization, c.CodebookSize * c.State, 1.0));
        }
    }

    /// <summary>v2's FSQ projection to the 8 quantized dimensions; null for v1.</summary>
    public DenseLayer<T>? ProjectDown { get; }

    public S3TokenizerConfiguration Configuration { get; }
    public NormedConv1DLayer<T> Conv1 { get; }
    public NormedConv1DLayer<T> Conv2 { get; }
    public List<Block> Blocks { get; } = new();

    /// <summary>Whisper's positional table <c>[length, channels]</c>: sin of the first half, cos of the second, at
    /// timescales 10000^(k / (channels/2 − 1)).</summary>
    internal static Tensor<T> Sinusoids(int length, int channels)
    {
        int half = channels / 2;
        double increment = Math.Log(10000.0) / (half - 1);
        // A fixed table of constants, built in double and cast once.
        var table = new double[length, channels];
        for (int t = 0; t < length; t++)
            for (int k = 0; k < half; k++)
            {
                double scaled = t * Math.Exp(-increment * k);
                table[t, k] = Math.Sin(scaled);
                table[t, half + k] = Math.Cos(scaled);
            }
        return EngineStft<T>.Table(table);
    }

    /// <summary>Whisper's 128-band log-mel <c>[128, frames]</c> of 16 kHz audio (<c>log_mel_spectrogram</c>): a centred,
    /// reflect-padded 400-point STFT with hop 160 and a periodic Hann window, the last frame dropped, power through
    /// librosa's Slaney filters, <c>log10(max(x, 1e-10))</c> floored at 8 below its maximum, then <c>(x + 4) / 4</c>.</summary>
    /// <remarks>The frames are scaled by 2<sup>−9</sup> before the DFT and the power's 2<sup>−18</sup> is added back as
    /// 18 log₁₀ 2 after the log, so a half-precision power (up to ~400² per bin) cannot overflow.</remarks>
    public static Tensor<T> LogMel(Tensor<T> audio, int mels = 128)
    {
        if (audio is null) throw new ArgumentNullException(nameof(audio));
        int count = audio.Length;
        if (count <= FftSize / 2) throw new ArgumentException("Reflect padding needs more than 200 samples.", nameof(audio));
        var engine = AiDotNetEngine.Current;
        int bins = FftSize / 2 + 1, frames = count / Hop, pad = FftSize / 2;
        const int Shift = 9;
        // Fixed tables: the scaled periodic Hann window and the DFT kernels cos/sin(2π (k n mod N) / N), [N, bins].
        var window = new double[FftSize];
        for (int n = 0; n < FftSize; n++) window[n] = (0.5 - 0.5 * Math.Cos(2 * Math.PI * n / FftSize)) / (1 << Shift);
        var cos = new double[FftSize, bins];
        var sin = new double[FftSize, bins];
        for (int n = 0; n < FftSize; n++)
            for (int k = 0; k < bins; k++)
            {
                int index = k * n % FftSize;
                cos[n, k] = Math.Cos(2 * Math.PI * index / FftSize);
                sin[n, k] = Math.Sin(2 * Math.PI * index / FftSize);
            }
        var positions = new int[frames * FftSize];
        for (int f = 0; f < frames; f++)
            for (int n = 0; n < FftSize; n++) positions[f * FftSize + n] = f * Hop + n;

        var padded = engine.PadNd(engine.Reshape(audio, new[] { count }), new[] { pad, pad }, AiDotNet.Tensors.Engines.PadMode.Reflect, NumOps.Zero);
        var framed = engine.Reshape(engine.TensorIndexSelect(padded, new Tensor<int>(positions, new[] { frames * FftSize }), 0), new[] { frames, FftSize });
        var windowed = engine.TensorMultiply(framed, engine.TensorBroadcastTo(
            new Tensor<double>(window, new[] { 1, FftSize }).Cast<T>(), new[] { frames, FftSize }));
        var re = engine.TensorMatMul(windowed, EngineStft<T>.Table(cos));
        var im = engine.TensorMatMul(windowed, EngineStft<T>.Table(sin));
        var power = engine.TensorAdd(engine.TensorMultiply(re, re), engine.TensorMultiply(im, im));                 // [frames, bins]
        var filters = EngineStft<T>.Table(MelFilterbanks.Slaney(SampleRate, FftSize, mels, 0, SampleRate / 2.0));     // [mels, bins]
        var mel = engine.TensorMatMul(filters, engine.TensorTranspose(power));                                        // [mels, frames]
        // log10(max(mel·4^Shift, 1e-10)) = max(log10(mel) + 2·Shift·log10 2, −10): the floor is applied in the log
        // domain, where it is representable in any T (1e-10 / 2^(2·Shift) underflows to zero in half precision, and
        // silence then gave log10(0) = −∞ everywhere, which the max-relative floor cannot recover).
        var log = engine.TensorMax(
            engine.TensorAddScalar(engine.TensorLog10(mel), NumOps.FromDouble(2 * Shift * Math.Log10(2.0))),
            NumOps.FromDouble(-10.0));
        T lowest = NumOps.Subtract(engine.TensorMaxValue(log), NumOps.FromDouble(8.0));
        return engine.TensorMultiplyScalar(engine.TensorAddScalar(engine.TensorMax(log, lowest), NumOps.FromDouble(4.0)), NumOps.FromDouble(0.25));
    }

    /// <summary>The encoder states <c>[frames', State]</c> of a log-mel <c>[Mels, frames]</c> (<c>AudioEncoder</c>).</summary>
    public Tensor<T> Encode(Tensor<T> mel)
    {
        var c = Configuration;
        int frames = mel.Shape[1], d = c.State, heads = c.Heads, dk = d / heads;
        bool v2 = c.Version == S3TokenizerVersion.V2;
        var x = ActivationHelper.ExactGelu(Conv1.Forward(_engine.Reshape(mel, new[] { 1, c.Mels, frames })), _engine);
        x = ActivationHelper.ExactGelu(Conv2.Forward(x), _engine);
        int length = x.Shape[2];
        x = _engine.Reshape(_engine.TensorPermute(x, new[] { 0, 2, 1 }), new[] { length, d });
        Tensor<T>? cos = null, sin = null;
        if (v2)
        {
            if (length > 2048) throw new ArgumentException("At most 2,048 positions fit v2's rotary table.", nameof(mel));
            // precompute_freqs_cis(64, 2048): angle t · 10000^(−2k/64), the 32 frequencies repeated for both halves.
            var cosTable = new double[length, dk];
            var sinTable = new double[length, dk];
            for (int t = 0; t < length; t++)
                for (int k = 0; k < dk / 2; k++)
                {
                    double angle = t * (1.0 / Math.Pow(10000.0, 2.0 * k / dk));
                    cosTable[t, k] = cosTable[t, k + dk / 2] = Math.Cos(angle);
                    sinTable[t, k] = sinTable[t, k + dk / 2] = Math.Sin(angle);
                }
            cos = _engine.Reshape(EngineStft<T>.Table(cosTable), new[] { 1, length, dk });
            sin = _engine.Reshape(EngineStft<T>.Table(sinTable), new[] { 1, length, dk });
        }
        else
        {
            if (length > c.Context) throw new ArgumentException($"At most {2 * c.Context * c.FirstStride} frames fit the positional table.", nameof(mel));
            x = _engine.TensorAdd(x, _engine.TensorSlice(_positions, new[] { 0, 0 }, new[] { length, d }));
        }
        Tensor<T> Rotate(Tensor<T> h)                                                                    // [H, L, dk]
        {
            if (cos is null || sin is null) return h;
            var shape = h.Shape.ToArray();
            var first = _engine.TensorSlice(h, new[] { 0, 0, 0 }, new[] { shape[0], shape[1], dk / 2 });
            var second = _engine.TensorSlice(h, new[] { 0, 0, dk / 2 }, new[] { shape[0], shape[1], dk / 2 });
            var rotated = _engine.TensorConcatenate(new[] { _engine.TensorNegate(second), first }, 2);
            return _engine.TensorAdd(_engine.TensorMultiply(h, _engine.TensorBroadcastTo(cos, shape)),
                _engine.TensorMultiply(rotated, _engine.TensorBroadcastTo(sin, shape)));
        }
        // Queries and keys are each scaled by d_k^−¼, as the reference does.
        var scale = NumOps.FromDouble(Math.Pow(dk, -0.25));
        Tensor<T> Heads(Tensor<T> t) => _engine.TensorPermute(_engine.Reshape(t, new[] { length, heads, dk }), new[] { 1, 0, 2 });
        foreach (var block in Blocks)
        {
            var normed = block.AttentionNorm.Forward(x);
            var values = block.Value.Forward(normed);                                                      // [L, d]
            var q = _engine.TensorMultiplyScalar(Rotate(Heads(block.Query.Forward(normed))), scale);
            var k = _engine.TensorMultiplyScalar(Rotate(Heads(block.Key.Forward(normed))), scale);
            var v = Heads(values);
            var weights = _engine.Softmax(_engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 })), axis: 2);
            var context = _engine.Reshape(_engine.TensorPermute(_engine.BatchMatMul(weights, v), new[] { 1, 0, 2 }), new[] { length, d });
            var attended = block.Output.Forward(context);
            if (block.Fsmn is not null)
            {
                // forward_fsmn: the values padded by 15 frames each side, the depthwise convolution, plus the values.
                var channels = _engine.Reshape(_engine.TensorTranspose(values), new[] { 1, d, length });
                var padding = new Tensor<T>(new[] { 1, d, (FsmnKernel - 1) / 2 });
                var memory = block.Fsmn.Forward(_engine.TensorConcatenate(new[] { padding, channels, padding }, 2));
                attended = _engine.TensorAdd(attended, _engine.TensorAdd(_engine.TensorTranspose(_engine.Reshape(memory, new[] { d, length })), values));
            }
            x = _engine.TensorAdd(x, attended);
            x = _engine.TensorAdd(x, block.Mlp2.Forward(ActivationHelper.ExactGelu(block.Mlp1.Forward(block.MlpNorm.Forward(x)), _engine)));
        }
        return x;
    }

    /// <summary>The codebook index of each state row: the nearest codebook entry to the L2-normalized state
    /// (<c>VectorQuantization.encode</c>; the codebook itself is not normalized).</summary>
    public int[] Quantize(Tensor<T> hidden)
    {
        if (ProjectDown is not null) return QuantizeFsq(hidden);
        int rows = hidden.Shape[0], d = Configuration.State, size = Configuration.CodebookSize;
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var squared = _engine.Reshape(_engine.ReduceSum(_engine.TensorMultiply(_codebook, _codebook), new[] { 1 }, false), new[] { 1, size });
        // F.normalize: x / max(‖x‖, ε), ε = 1e-12 (raised to the smallest normal value where 1e-12 underflows T).
        var x = _engine.Reshape(hidden, new[] { rows, d });
        var norm = _engine.TensorSqrt(_engine.ReduceSum(_engine.TensorMultiply(x, x), new[] { 1 }, true));
        x = _engine.TensorDivide(x, _engine.TensorBroadcastTo(_engine.TensorMax(norm, Epsilon(1e-12)), new[] { rows, d }));
        var xx = _engine.ReduceSum(_engine.TensorMultiply(x, x), new[] { 1 }, true);                                 // [rows, 1]
        var dot = _engine.TensorMatMul(x, _engine.TensorTranspose(_codebook));                                     // [rows, size]
        // −(‖x‖² − 2 x·e + ‖e‖²); torch's max returns the first index of the maximum.
        var score = _engine.TensorNegate(_engine.TensorAdd(_engine.TensorSubtract(_engine.TensorBroadcastTo(xx, new[] { rows, size }),
            _engine.TensorMultiplyScalar(dot, NumOps.FromDouble(2.0))), _engine.TensorBroadcastTo(squared, new[] { rows, size })));
        return _engine.TensorArgMax(score, 1).ToArray();
    }

    // A positive guard ε, raised to the smallest normal half-precision value where ε itself underflows T.
    private static T Epsilon(double value)
        => NumOps.ToDouble(NumOps.FromDouble(value)) == 0 ? NumOps.FromDouble(6.103515625e-05) : NumOps.FromDouble(value);

    /// <summary>v2's finite scalar quantization (<c>FSQCodebook.encode</c>): the 8 projected values through tanh, scaled by
    /// 0.999, rounded half to even into {−1, 0, 1}, shifted to {0, 1, 2} and read as base-3 digits, least significant
    /// first.</summary>
    private int[] QuantizeFsq(Tensor<T> hidden)
    {
        var projected = (ProjectDown ?? throw new InvalidOperationException("v2 has an FSQ projection.")).Forward(hidden);
        int rows = projected.Shape[0];
        // round(tanh(x) · 0.999) + 1 ∈ {0, 1, 2}, read in integers.
        var digits = _engine.TensorAddScalar(_engine.TensorRound(_engine.TensorMultiplyScalar(_engine.TensorTanh(projected),
            NumOps.FromDouble(0.9990000128746033))), NumOps.One).Cast<int>().ToArray();
        var codes = new int[rows];
        for (int r = 0; r < rows; r++)
            for (int j = 7; j >= 0; j--) codes[r] = 3 * codes[r] + digits[r * 8 + j];
        return codes;
    }

    /// <summary>The speech tokens of a log-mel <c>[Mels, frames]</c>, windowing utterances over 30 s.</summary>
    public int[] Tokenize(Tensor<T> mel)
    {
        int frames = mel.Shape[1], mels = Configuration.Mels;
        if (frames <= WindowFrames) return Quantize(Encode(mel));
        int overlapFrames = OverlapSeconds * SampleRate / Hop, stride = WindowFrames - overlapFrames;
        var segments = new List<int[]>();
        for (int start = 0; start < frames; start += stride)
        {
            int end = Math.Min(start + WindowFrames, frames);
            segments.Add(Quantize(Encode(_engine.TensorSlice(mel, new[] { 0, start }, new[] { mels, end - start }))));
        }
        // merge_tokenized_segments: drop half the overlap's tokens on each side of every seam.
        int drop = OverlapSeconds / 2 * Configuration.TokenRate;
        var merged = new List<int>();
        for (int i = 0; i < segments.Count; i++)
        {
            var tokens = segments[i];
            int left = i == 0 ? 0 : drop, right = i != segments.Count - 1 ? tokens.Length - drop : tokens.Length;
            for (int j = left; j < right; j++) merged.Add(tokens[j]);
        }
        return merged.ToArray();
    }

    /// <summary>Loads an S3Tokenizer state dict (<c>encoder.conv1/2</c>, <c>encoder.positional_embedding</c>,
    /// <c>encoder.blocks.{i}.*</c>, <c>quantizer._codebook.embed</c>).</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read) => LoadTorchWeights(TorchTensorSource.From(read));

    internal void LoadTorchWeights(TorchTensorSource source)
    {
        var c = Configuration;
        int d = c.State;
        void Dense(DenseLayer<T> layer, string name, int inputs, int outputs) =>
            TorchParameters.Linear(_engine, layer, inputs, outputs, source, name + ".weight", name + ".bias");
        void Norm(LayerNormalizationLayer<T> norm, string name) =>
            TorchParameters.LayerNorm(_engine, norm, source, name + ".weight", name + ".bias");
        Conv1.LoadTorchWeights(source, "encoder.conv1.weight", new[] { d, c.Mels, 3 }, null, "encoder.conv1.bias");
        Conv2.LoadTorchWeights(source, "encoder.conv2.weight", new[] { d, d, 3 }, null, "encoder.conv2.bias");
        if (c.Version == S3TokenizerVersion.V1)
        {
            _positions = new Tensor<T>(new[] { c.Context, d });
            source.CopyTo("encoder.positional_embedding", new[] { c.Context, d }, _positions);
        }
        for (int i = 0; i < Blocks.Count; i++)
        {
            var b = Blocks[i];
            string p = $"encoder.blocks.{i}";
            Norm(b.AttentionNorm, p + ".attn_ln");
            Dense(b.Query, p + ".attn.query", d, d);
            TorchParameters.Linear(b.Key, d, d, source, p + ".attn.key.weight");
            Dense(b.Value, p + ".attn.value", d, d);
            Dense(b.Output, p + ".attn.out", d, d);
            Norm(b.MlpNorm, p + ".mlp_ln");
            Dense(b.Mlp1, p + ".mlp.0", d, 4 * d);
            Dense(b.Mlp2, p + ".mlp.2", 4 * d, d);
            if (b.Fsmn is not null)
                b.Fsmn.LoadTorchWeights(source, p + ".attn.fsmn_block.weight", new[] { d, 1, FsmnKernel }, null, null);
        }
        if (ProjectDown is not null)
            Dense(ProjectDown, "quantizer._codebook.project_down", d, 8);
        else
        {
            _codebook = new Tensor<T>(new[] { c.CodebookSize, d });
            source.CopyTo("quantizer._codebook.embed", new[] { c.CodebookSize, d }, _codebook);
        }
    }
}

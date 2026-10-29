using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.FlowDiffusion;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The sizes of Vevo's content tokenizer (the release's <c>hubert_large_l18_c32.yaml</c>).</summary>
internal sealed record VevoContentCodecConfiguration(int InputChannels, int EncodeChannels, int CodeDim, int Codes, int[] Strides,
    int[] Dilations, int Kernel = 3, int UnitKernel = 3);

/// <summary>
/// Vevo's content tokenizer (the release's <c>VevoRepCodec</c>, after Huang et al.'s RepCodec): HuBERT features, unnormalized,
/// through a 3-wide convolution, encoder blocks (residual units of ELU, a dilated 3-wide convolution, ELU and a 1×1
/// convolution, all bias-free; then a 3-wide convolution), a bias-free 3-wide projector, and the nearest entry of an
/// EMA-trained codebook. Only its encoder side tokenizes; the decoder trained it.
/// </summary>
internal sealed class VevoContentCodec<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    internal sealed record Unit(NormedConv1DLayer<T> Conv, NormedConv1DLayer<T> Pointwise, int Dilation);

    internal sealed record Block(List<Unit> Units, NormedConv1DLayer<T> Conv);

    public VevoContentCodec(IEngine engine, List<LayerBase<T>> layers, VevoContentCodecConfiguration configuration, Random random)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.Strides.Any(s => s != 1)) throw new ArgumentException("Vevo's content tokenizer keeps the frame rate (strides 1).", nameof(configuration));
        NormedConv1DLayer<T> Conv(int inputs, int outputs, int kernel, int dilation, bool bias)
        {
            var conv = new NormedConv1DLayer<T>(inputs, outputs, kernel, 1, dilation, 1, (kernel - 1) / 2 * dilation, false, ConvolutionNormalization.None, bias);
            layers.Add(conv);
            conv.LoadTorchWeights(IndexTtsOps.Uniform(random, outputs * inputs * kernel, inputs * kernel), null,
                bias ? IndexTtsOps.Uniform(random, outputs, inputs * kernel) : new double[outputs]);
            return conv;
        }
        Input = Conv(c.InputChannels, c.EncodeChannels, c.Kernel, 1, false);
        foreach (var _ in c.Strides)
        {
            var units = c.Dilations.Select(d => new Unit(Conv(c.EncodeChannels, c.EncodeChannels, c.UnitKernel, d, false),
                Conv(c.EncodeChannels, c.EncodeChannels, 1, 1, false), d)).ToList();
            Blocks.Add(new Block(units, Conv(c.EncodeChannels, c.EncodeChannels, 3, 1, true)));
        }
        Projector = Conv(c.EncodeChannels, c.CodeDim, 3, 1, false);
        Codebook = IndexTtsOps.Values(layers, IndexTtsOps.Normal(random, c.CodeDim * c.Codes, 1.0));       // embed [dim, codes]
    }

    public VevoContentCodecConfiguration Configuration { get; }
    public NormedConv1DLayer<T> Input { get; }
    public List<Block> Blocks { get; } = new();
    public NormedConv1DLayer<T> Projector { get; }
    public BiasFreeLinearLayer<T> Codebook { get; }

    /// <summary>The codes <c>[frames]</c> of HuBERT features <c>[frames, InputChannels]</c>: per frame the codebook entry at
    /// the least squared distance (the first on a tie).</summary>
    public int[] Codes(Tensor<T> features)
    {
        var c = Configuration;
        int frames = features.Shape[0];
        var x = Input.Forward(_engine.Reshape(_engine.TensorTranspose(features), new[] { 1, c.InputChannels, frames }));
        foreach (var block in Blocks)
        {
            foreach (var unit in block.Units)
                x = _engine.TensorAdd(x, unit.Pointwise.Forward(_engine.ELU(unit.Conv.Forward(_engine.ELU(x, 1.0)), 1.0)));
            x = block.Conv.Forward(x);
        }
        var z = _engine.TensorTranspose(_engine.Reshape(Projector.Forward(x), new[] { c.CodeDim, frames }));          // [frames, dim]
        var embed = _engine.Reshape(IndexTtsOps.Value(Codebook), new[] { c.CodeDim, c.Codes });
        var cross = _engine.TensorMatMul(z, embed);
        var norms = _engine.ReduceSum(_engine.TensorMultiply(embed, embed), new[] { 0 }, keepDims: false);
        var codes = new int[frames];
        for (int i = 0; i < frames; i++)
        {
            double best = double.PositiveInfinity;
            for (int k = 0; k < c.Codes; k++)
            {
                double distance = NumOps.ToDouble(norms[k]) - 2 * NumOps.ToDouble(cross[i, k]);
                if (distance < best)
                {
                    best = distance;
                    codes[i] = k;
                }
            }
        }
        return codes;
    }

    /// <summary><c>duration_reduction_func</c> with n-gram 1: consecutive repeats removed.</summary>
    public static int[] RemoveRepeats(IReadOnlyList<int> codes)
    {
        var kept = new List<int>(codes.Count);
        for (int i = 0; i < codes.Count; i++)
            if (i == 0 || codes[i] != codes[i - 1]) kept.Add(codes[i]);
        return kept.ToArray();
    }

    /// <summary>Loads the release's <c>encoder.*</c>, <c>projector.*</c> and the codebook
    /// (<c>quantizer.codebook.layers.0.embed</c>, <c>[dim, codes]</c>).</summary>
    public void Load(Func<string, int[], double[]> read)
    {
        var c = Configuration;
        void Conv(NormedConv1DLayer<T> conv, string name, bool bias) => conv.LoadTorchWeights(read(name + ".weight", conv.TorchWeightShape), null,
            bias ? read(name + ".bias", new[] { conv.OutputChannels }) : new double[conv.OutputChannels]);
        Conv(Input, "encoder.conv.conv", false);
        for (int i = 0; i < Blocks.Count; i++)
        {
            string p = $"encoder.conv_blocks.{i}.";
            for (int j = 0; j < Blocks[i].Units.Count; j++)
            {
                Conv(Blocks[i].Units[j].Conv, $"{p}res_units.{j}.conv1.conv", false);
                Conv(Blocks[i].Units[j].Pointwise, $"{p}res_units.{j}.conv2", false);
            }
            Conv(Blocks[i].Conv, p + "conv.conv", true);
        }
        Conv(Projector, "projector.project.conv", false);
        TorchParameters.Linear(Codebook, 1, c.CodeDim * c.Codes, read("quantizer.codebook.layers.0.embed", new[] { c.CodeDim, c.Codes }));
    }
}

/// <summary>The decoding settings of Vevo's autoregressive transformer (its <c>generate</c> arguments).</summary>
internal sealed record VevoDecoding(bool Sample, double Temperature, int TopK, double TopP, double RepetitionPenalty, int MinNewTokens, int MaxLength,
    bool ConstrainToOutputTokens = false);

/// <summary>
/// Vevo's autoregressive transformer (the release's <c>AutoregressiveTransformer</c>): a Llama whose single vocabulary holds
/// the input tokens (shifted past the outputs), the outputs (content-style tokens) and 20 specials, reading
/// <c>[BOS_in, input…, EOS_in, BOS_out, output…, EOS_out]</c> and trained by cross-entropy on the output part. Phones in
/// for Vevo-TTS; content tokens (and phones) in for Vevo-Voice and Vevo-Style.
/// </summary>
internal sealed class VevoAutoregressive<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public VevoAutoregressive(IEngine engine, List<LayerBase<T>> layers, int inputVocabulary, int outputVocabulary, int hidden, int intermediate,
        int depth, int heads, Random random)
    {
        _engine = engine;
        InputVocabulary = inputVocabulary;
        OutputVocabulary = outputVocabulary;
        Network = new LlamaNetwork<T>(engine, layers, new LlamaConfiguration
        {
            VocabSize = inputVocabulary + outputVocabulary + 20, Hidden = hidden, Intermediate = intermediate, Layers = depth, Heads = heads,
            KeyValueHeads = heads, RmsEpsilon = 1e-6, RopeTheta = 10000,
        });
        Network.InitializeLikeReference(random);
    }

    public int InputVocabulary { get; }
    public int OutputVocabulary { get; }
    public LlamaNetwork<T> Network { get; }
    public int Pad => InputVocabulary + OutputVocabulary;
    public int InputBos => Pad + 1;
    public int InputEos => Pad + 2;
    public int OutputBos => Pad + 3;
    public int OutputEos => Pad + 4;

    private int[] Prefix(IReadOnlyList<int> input)
    {
        var ids = new List<int>(input.Count + 2) { InputBos };
        foreach (int t in input)
        {
            if (t < 0 || t >= InputVocabulary) throw new ArgumentOutOfRangeException(nameof(input), $"Input tokens lie in [0, {InputVocabulary}).");
            ids.Add(t + OutputVocabulary);
        }
        ids.Add(InputEos);
        return ids.ToArray();
    }

    private Tensor<T> Embed(IReadOnlyList<int> ids)
    {
        var grid = new int[1, ids.Count];
        for (int i = 0; i < ids.Count; i++) grid[0, i] = ids[i];
        return Network.Embed(grid);
    }

    /// <summary>The training loss (<c>forward</c>): the language model's cross-entropy over
    /// <c>[BOS_out, output…, EOS_out]</c>, the input part unscored.</summary>
    public Tensor<T> Loss(IReadOnlyList<int> input, IReadOnlyList<int> output, bool training = false)
    {
        var prefix = Prefix(input);
        var ids = prefix.Concat(new[] { OutputBos }).Concat(output.Select(o =>
            o >= 0 && o < OutputVocabulary ? o : throw new ArgumentOutOfRangeException(nameof(output), $"Output tokens lie in [0, {OutputVocabulary})."))).Concat(new[] { OutputEos }).ToArray();
        var labels = new int[1, ids.Length];
        for (int i = 0; i < ids.Length; i++) labels[0, i] = i < prefix.Length ? -100 : ids[i];
        Network.Training = training;
        var (_, logits) = Network.Forward(Embed(ids));
        return Network.CausalLmLoss(logits!, labels);
    }

    /// <summary>The logits <c>[1, L, vocabulary]</c> of the teacher-forced sequence.</summary>
    public Tensor<T> Logits(IReadOnlyList<int> input, IReadOnlyList<int> output)
    {
        var ids = Prefix(input).Concat(new[] { OutputBos }).Concat(output).Concat(new[] { OutputEos }).ToArray();
        return Network.Forward(Embed(ids)).Logits!;
    }

    /// <summary>
    /// Generates output tokens (<c>generate</c> through Hugging Face's): after <c>[BOS_in, input…, EOS_in, BOS_out,
    /// prompt…]</c>, the end token barred until <see cref="VevoDecoding.MinNewTokens"/> are generated, the repetition
    /// penalty, then temperature, top-k and top-p, until the end token or <see cref="VevoDecoding.MaxLength"/> positions in
    /// all. A leading start token and the end token are dropped.
    /// </summary>
    public int[] Generate(IReadOnlyList<int> input, IReadOnlyList<int> promptOutput, VevoDecoding decoding, Random random)
    {
        var ids = Prefix(input).Concat(new[] { OutputBos }).Concat(promptOutput).ToList();
        int promptLength = ids.Count;
        var cache = new LlamaCache<T>();
        Network.Training = false;
        var (_, logits) = Network.Forward(Embed(ids), null, null, cache, lastLogitsOnly: true);
        var generated = new List<int>();
        while (ids.Count < decoding.MaxLength)
        {
            int vocabulary = logits!.Shape[2];
            var scores = new double[vocabulary];
            for (int v = 0; v < vocabulary; v++)
                scores[v] = decoding.ConstrainToOutputTokens && v >= OutputVocabulary && v != OutputEos ? double.NegativeInfinity : NumOps.ToDouble(logits[0, 0, v]);
            int next = VevoSampling.Next(scores, ids, generated.Count, OutputEos, decoding, random);
            ids.Add(next);
            generated.Add(next);
            if (next == OutputEos) break;
            (_, logits) = Network.Forward(Embed(new[] { next }), null, null, cache, lastLogitsOnly: true);
        }
        var result = generated.ToList();
        if (result.Count > 0 && result[0] == OutputBos) result.RemoveAt(0);
        if (result.Count > 0 && result[result.Count - 1] == OutputEos) result.RemoveAt(result.Count - 1);
        return result.ToArray();
    }

    /// <summary>Loads the release's <c>model.model.*</c> and <c>model.lm_head.weight</c>.</summary>
    public void Load(Func<string, int[], double[]> read) => Network.LoadTorchWeights(read, "model.model.", "model.lm_head.weight");
}

/// <summary>Hugging Face's logits processors and warpers in <c>generate</c>'s order, as Vevo's decoding uses them.</summary>
internal static class VevoSampling
{
    public static int Next(double[] scores, IReadOnlyList<int> ids, int newTokens, int eos, VevoDecoding decoding, Random random)
    {
        if (decoding.RepetitionPenalty != 1.0)
            foreach (int t in new HashSet<int>(ids))
                if (t < scores.Length) scores[t] = scores[t] < 0 ? scores[t] * decoding.RepetitionPenalty : scores[t] / decoding.RepetitionPenalty;
        if (newTokens < decoding.MinNewTokens && eos < scores.Length) scores[eos] = double.NegativeInfinity;
        if (!decoding.Sample)
        {
            int best = 0;
            for (int v = 1; v < scores.Length; v++) if (scores[v] > scores[best]) best = v;
            return best;
        }
        if (decoding.Temperature != 1.0)
            for (int v = 0; v < scores.Length; v++) scores[v] /= decoding.Temperature;
        if (decoding.TopK > 0 && decoding.TopK < scores.Length)
        {
            double threshold = scores.OrderByDescending(x => x).ElementAt(decoding.TopK - 1);
            for (int v = 0; v < scores.Length; v++) if (scores[v] < threshold) scores[v] = double.NegativeInfinity;
        }
        if (decoding.TopP < 1.0)
        {
            var order = Enumerable.Range(0, scores.Length).OrderBy(v => scores[v]).ToArray();
            double max = scores.Max(), total = 0, cumulative = 0;
            var p = order.Select(v => double.IsNegativeInfinity(scores[v]) ? 0 : Math.Exp(scores[v] - max)).ToArray();
            foreach (var x in p) total += x;
            for (int i = 0; i < order.Length - 1; i++)
            {
                cumulative += p[i] / total;
                if (cumulative <= 1 - decoding.TopP) scores[order[i]] = double.NegativeInfinity;
            }
        }
        double top = scores.Max(), sum = 0;
        var weights = scores.Select(x => double.IsNegativeInfinity(x) ? 0 : Math.Exp(x - top)).ToArray();
        foreach (var w in weights) sum += w;
        double u = random.NextDouble() * sum, acc = 0;
        int last = 0;
        for (int v = 0; v < weights.Length; v++)
        {
            if (weights[v] <= 0) continue;
            acc += weights[v];
            last = v;
            if (acc > u) return v;
        }
        return last;
    }
}

/// <summary>The sizes of Vevo's flow-matching transformer (the release's <c>flow_matching_transformer</c> config).</summary>
internal sealed record VevoFlowConfiguration(int MelDim, int Hidden, int Layers, int Heads, int Codes, double PromptDropProbability, double Sigma,
    bool CosineSchedule);

/// <summary>
/// Vevo's flow-matching transformer (the release's <c>FlowMatchingTransformer</c> over its <c>DiffLlama</c>): content-style
/// tokens embedded per frame (<c>cond_emb</c>) and lifted by <c>cond_mlp</c>, added to an MLP of the noisy mel, run through
/// MaskGCT's bidirectional Llama with step-adaptive RMSNorms, and an MLP back to the mel's velocity. Sampling is Euler from
/// noise, the prompt's mel prepended, with rescaled classifier-free guidance against a prompt-free, condition-free pass.
/// </summary>
internal sealed class VevoFlowMatching<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public VevoFlowMatching(IEngine engine, List<LayerBase<T>> layers, VevoFlowConfiguration configuration, Random random)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        Condition = IndexTtsOps.Embedding(layers, random, c.Codes, c.Hidden, 0.02);
        Transformer = new MaskGctTransformer<T>(engine, layers, new MaskGctTransformerConfiguration { Hidden = c.Hidden, Layers = c.Layers, Heads = c.Heads }, random);
        DenseLayer<T> Linear(int inputs, int outputs)
        {
            var layer = new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>());
            layers.Add(layer);
            TorchParameters.Linear(engine, layer, inputs, outputs, IndexTtsOps.Normal(random, inputs * outputs, 0.02), new double[outputs]);
            return layer;
        }
        MelIn = (Linear(c.MelDim, 4 * c.Hidden), Linear(4 * c.Hidden, c.Hidden));
        MelOut = (Linear(c.Hidden, 4 * c.Hidden), Linear(4 * c.Hidden, c.MelDim));
    }

    public VevoFlowConfiguration Configuration { get; }
    public TiedEmbeddingLayer<T> Condition { get; }
    public MaskGctTransformer<T> Transformer { get; }
    public (DenseLayer<T> In, DenseLayer<T> Out) MelIn { get; }
    public (DenseLayer<T> In, DenseLayer<T> Out) MelOut { get; }

    private Tensor<T> Mlp((DenseLayer<T> In, DenseLayer<T> Out) mlp, Tensor<T> x)
    {
        var h = mlp.In.Forward(x);
        return mlp.Out.Forward(_engine.TensorMultiply(h, _engine.Sigmoid(h)));
    }

    /// <summary>The embedded condition <c>[frames, hidden]</c> of content-style tokens (<c>cond_emb</c>).</summary>
    public Tensor<T> Embed(IReadOnlyList<int> codes) => Condition.Forward(IndexTtsOps.Ids<T>(codes));

    /// <summary>The predicted velocity <c>[frames, mel]</c> of a noisy mel <c>[frames, mel]</c> at time t
    /// (<c>diff_estimator</c>).</summary>
    public Tensor<T> Velocity(Tensor<T> x, double t, Tensor<T> condition) => Mlp(MelOut, Transformer.Forward(Mlp(MelIn, x), t, condition));

    /// <summary>
    /// <c>reverse_diffusion</c>: Euler over <paramref name="steps"/> midpoints from <paramref name="noise"/>
    /// <c>[target, mel]</c>, conditioned on <paramref name="condition"/> <c>[prompt + target, hidden]</c> with the prompt's
    /// mel <c>[prompt, mel]</c> in front; guidance <c>v + cfg (v − v_u)</c> rescaled to v's standard deviation and mixed
    /// back by <paramref name="rescale"/>.
    /// </summary>
    public Tensor<T> Solve(Tensor<T> condition, Tensor<T> prompt, Tensor<T> noise, int steps, double cfg, double rescale)
    {
        int promptFrames = prompt.Shape[0], target = noise.Shape[0], mel = Configuration.MelDim;
        if (condition.Shape[0] != promptFrames + target) throw new ArgumentException("The condition covers the prompt and the target.", nameof(condition));
        double h = 1.0 / steps;
        var xt = noise;
        var silent = new Tensor<T>(new[] { target, Configuration.Hidden });
        for (int i = 0; i < steps; i++)
        {
            double t = (i + 0.5) * h;
            var full = Velocity(_engine.TensorConcatenate(new[] { prompt, xt }, 0), t, condition);
            var v = _engine.TensorNarrow(full, 0, promptFrames, target);
            if (cfg > 0)
            {
                var unconditional = Velocity(xt, t, silent);
                double std = Std(v);
                var guided = _engine.TensorAdd(v, _engine.TensorMultiplyScalar(_engine.TensorSubtract(v, unconditional), NumOps.FromDouble(cfg)));
                var rescaled = _engine.TensorMultiplyScalar(guided, NumOps.FromDouble(std / Std(guided)));
                v = _engine.TensorAdd(_engine.TensorMultiplyScalar(rescaled, NumOps.FromDouble(rescale)),
                    _engine.TensorMultiplyScalar(guided, NumOps.FromDouble(1 - rescale)));
            }
            xt = _engine.TensorAdd(xt, _engine.TensorMultiplyScalar(v, NumOps.FromDouble(h)));
        }
        return xt;
    }

    // torch.std over every element (Bessel's correction).
    private static double Std(Tensor<T> x)
    {
        int n = x.Length;
        double mean = 0;
        for (int i = 0; i < n; i++) mean += NumOps.ToDouble(x.GetFlat(i));
        mean /= n;
        double sum = 0;
        for (int i = 0; i < n; i++)
        {
            double d = NumOps.ToDouble(x.GetFlat(i)) - mean;
            sum += d * d;
        }
        return Math.Sqrt(sum / (n - 1));
    }

    /// <summary>
    /// The training loss (<c>compute_loss</c> and the trainer's L1): t drawn in [1e-5, 1] (cosine-scheduled), a prompt of
    /// <paramref name="promptFrames"/> clean frames unless the condition is dropped (<paramref name="dropCondition"/>, which
    /// also zeroes the condition), <c>x_t = (1 − (1 − σ) t) z + t x</c> elsewhere, and the mean absolute error of the
    /// velocity against <c>x − (1 − σ) z</c> over the noised frames.
    /// </summary>
    public Tensor<T> Loss(Tensor<T> mel, IReadOnlyList<int> codes, double t, bool dropCondition, int promptFrames, Tensor<T> noise, bool training = false)
    {
        var c = Configuration;
        int frames = mel.Shape[0], dim = c.MelDim;
        t = Math.Min(Math.Max(t, 1e-5), 1.0);
        if (c.CosineSchedule) t = 1 - Math.Cos(t * Math.PI * 0.5);
        int prompt = dropCondition ? 0 : promptFrames;
        var keep = new Tensor<T>(new[] { frames, dim });                     // 1 on the prompt
        for (int f = 0; f < Math.Min(prompt, frames); f++)
            for (int d = 0; d < dim; d++) keep[f, d] = NumOps.One;
        var noised = _engine.TensorAdd(_engine.TensorMultiplyScalar(noise, NumOps.FromDouble(1 - (1 - c.Sigma) * t)), _engine.TensorMultiplyScalar(mel, NumOps.FromDouble(t)));
        var ones = new Tensor<T>(new[] { frames, dim });
        ones.Fill(NumOps.One);
        var xt = _engine.TensorAdd(_engine.TensorMultiply(noised, _engine.TensorSubtract(ones, keep)), _engine.TensorMultiply(mel, keep));
        var condition = Embed(codes);
        if (prompt == 0) condition = _engine.TensorMultiplyScalar(condition, NumOps.Zero);
        var predicted = Velocity(xt, t, condition);
        var target = _engine.TensorSubtract(mel, _engine.TensorMultiplyScalar(noise, NumOps.FromDouble(1 - c.Sigma)));
        var error = _engine.TensorMultiply(_engine.TensorAbs(_engine.TensorSubtract(predicted, target)), _engine.TensorSubtract(ones, keep));
        int counted = frames - Math.Min(prompt, frames);
        return _engine.TensorMultiplyScalar(_engine.ReduceSum(error, null), NumOps.FromDouble(1.0 / (dim * Math.Max(counted, 1))));
    }

    /// <summary>Loads the release's <c>cond_emb</c> and <c>diff_estimator.*</c>.</summary>
    public void Load(Func<string, int[], double[]> read)
    {
        var c = Configuration;
        Condition.LoadTable(read("cond_emb.weight", new[] { c.Codes, c.Hidden }));
        Transformer.LoadTorchWeights((name, shape) => read(name, shape), "diff_estimator.");
        void Linear(DenseLayer<T> layer, string name, int inputs, int outputs) =>
            TorchParameters.Linear(_engine, layer, inputs, outputs, read(name + ".weight", new[] { outputs, inputs }), read(name + ".bias", new[] { outputs }));
        Linear(MelIn.In, "diff_estimator.mel_mlp.0", c.MelDim, 4 * c.Hidden);
        Linear(MelIn.Out, "diff_estimator.mel_mlp.2", 4 * c.Hidden, c.Hidden);
        Linear(MelOut.In, "diff_estimator.mel_out_mlp.0", c.Hidden, 4 * c.Hidden);
        Linear(MelOut.Out, "diff_estimator.mel_out_mlp.2", 4 * c.Hidden, c.MelDim);
    }
}

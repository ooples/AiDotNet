using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.StyleEmotion;

/// <summary>
/// One block of StyleTTS 2's style-diffusion transformer (the reference's <c>StyleTransformerBlock</c> or
/// <c>TransformerBlock</c> in <c>Modules/diffusion/modules.py</c>): pre-normalized self-attention whose query and key/value
/// inputs are normalized separately — by AdaLN from the reference style for the multispeaker model, by a plain layer norm
/// otherwise — with bias-free q and kv projections and an output projection, then an exact-GELU feed-forward, each residual.
/// </summary>
internal sealed class StyleTts2DiffusionBlock<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public StyleTts2DiffusionBlock(IEngine engine, List<LayerBase<T>> layers, int features, int heads, int headFeatures, int multiplier,
        int? styleFeatures)
    {
        _engine = engine;
        Features = features;
        Heads = heads;
        HeadFeatures = headFeatures;
        int mid = heads * headFeatures;
        if (styleFeatures is int style)
        {
            StyleNorm = new StyleTtsAdaLayerNorm<T>(engine, layers, style, features);
            StyleContextNorm = new StyleTtsAdaLayerNorm<T>(engine, layers, style, features);
        }
        else
        {
            Norm = StyleTtsOps.Own(layers, new LayerNormalizationLayer<T>(features, 1e-5));
            ContextNorm = StyleTtsOps.Own(layers, new LayerNormalizationLayer<T>(features, 1e-5));
        }
        Query = StyleTtsOps.Own(layers, new BiasFreeLinearLayer<T>(features, mid));
        KeyValue = StyleTtsOps.Own(layers, new BiasFreeLinearLayer<T>(features, 2 * mid));
        Output = StyleTtsOps.Own(layers, StyleTtsOps.Linear<T>(features));
        FeedForward = StyleTtsOps.Own(layers, StyleTtsOps.Linear<T>(features * multiplier));
        FeedForwardOutput = StyleTtsOps.Own(layers, StyleTtsOps.Linear<T>(features));
        Multiplier = multiplier;
    }

    public int Features { get; }
    public int Heads { get; }
    public int HeadFeatures { get; }
    public int Multiplier { get; }
    public StyleTtsAdaLayerNorm<T>? StyleNorm { get; }
    public StyleTtsAdaLayerNorm<T>? StyleContextNorm { get; }
    public LayerNormalizationLayer<T>? Norm { get; }
    public LayerNormalizationLayer<T>? ContextNorm { get; }
    public BiasFreeLinearLayer<T> Query { get; }
    public BiasFreeLinearLayer<T> KeyValue { get; }
    public DenseLayer<T> Output { get; }
    public DenseLayer<T> FeedForward { get; }
    public DenseLayer<T> FeedForwardOutput { get; }

    /// <summary>The block applied to rows <c>[L, features]</c> with the reference style <c>[1, style]</c> (unused by the
    /// single-speaker block).</summary>
    public Tensor<T> Forward(Tensor<T> x, Tensor<T>? style)
    {
        int length = x.Shape[0], mid = Heads * HeadFeatures;
        var queryInput = StyleNorm is not null ? StyleNorm.Forward(x, style!) : Norm!.Forward(x);
        var contextInput = StyleContextNorm is not null ? StyleContextNorm.Forward(x, style!) : ContextNorm!.Forward(x);
        Tensor<T> Split(Tensor<T> rows) => _engine.TensorPermute(_engine.Reshape(rows, new[] { length, Heads, HeadFeatures }), new[] { 1, 0, 2 });
        var q = Split(Query.Forward(queryInput));
        var kv = KeyValue.Forward(contextInput);
        var k = Split(_engine.TensorSlice(kv, new[] { 0, 0 }, new[] { length, mid }));
        var v = Split(_engine.TensorSlice(kv, new[] { 0, mid }, new[] { length, mid }));
        var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 })),
            NumOps.FromDouble(1.0 / Math.Sqrt(HeadFeatures)));
        var attended = _engine.BatchMatMul(_engine.Softmax(scores, axis: 2), v);
        var context = _engine.Reshape(_engine.TensorPermute(attended, new[] { 1, 0, 2 }), new[] { length, mid });
        x = _engine.TensorAdd(Output.Forward(context), x);
        var hidden = FeedForwardOutput.Forward(ActivationHelper.ExactGelu(FeedForward.Forward(x), _engine));
        return _engine.TensorAdd(hidden, x);
    }

    public void Load(TorchReader read, string name)
    {
        int f = Features, mid = Heads * HeadFeatures;
        if (StyleNorm is not null)
        {
            StyleNorm.Load(read, name + ".attention.norm");
            StyleContextNorm!.Load(read, name + ".attention.norm_context");
        }
        else
        {
            TorchParameters.LayerNorm(_engine, Norm!, read(name + ".attention.norm.weight", new[] { f }), read(name + ".attention.norm.bias", new[] { f }));
            TorchParameters.LayerNorm(_engine, ContextNorm!, read(name + ".attention.norm_context.weight", new[] { f }),
                read(name + ".attention.norm_context.bias", new[] { f }));
        }
        TorchParameters.Linear(Query, f, mid, read(name + ".attention.to_q.weight", new[] { mid, f }));
        TorchParameters.Linear(KeyValue, f, 2 * mid, read(name + ".attention.to_kv.weight", new[] { 2 * mid, f }));
        StyleTtsOps.LoadLinear(_engine, Output, read, name + ".attention.attention.to_out", mid, f);
        StyleTtsOps.LoadLinear(_engine, FeedForward, read, name + ".feed_forward.0", f, f * Multiplier);
        StyleTtsOps.LoadLinear(_engine, FeedForwardOutput, read, name + ".feed_forward.2", f * Multiplier, f);
    }
}

/// <summary>
/// StyleTTS 2's style denoiser (§3.2.1; the reference's <c>StyleTransformer1d</c> for multispeaker models and
/// <c>Transformer1d</c> otherwise): the noisy style vector repeated along the PL-BERT tokens and concatenated with their
/// embeddings; a mapping from the noise level (learned Fourier features, a linear layer and GELU) plus, for the multispeaker
/// model, the reference style (a linear layer and GELU), through two GELU linear layers, added before every block; the
/// mean over tokens projected back to the style width. The unconditional input is a learned per-position table.
/// </summary>
internal sealed class StyleTts2StyleDenoiser<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public StyleTts2StyleDenoiser(IEngine engine, List<LayerBase<T>> layers, int channels, int embeddingFeatures, int layerCount,
        int heads, int headFeatures, int multiplier, int? styleFeatures, int maxLength, Random random)
    {
        if (channels % 2 != 0) throw new ArgumentException("The style width must be even.", nameof(channels));
        _engine = engine;
        Channels = channels;
        EmbeddingFeatures = embeddingFeatures;
        StyleFeatures = styleFeatures;
        int features = channels + embeddingFeatures;
        for (int i = 0; i < layerCount; i++)
            Blocks.Add(new StyleTts2DiffusionBlock<T>(engine, layers, features, heads, headFeatures, multiplier, styleFeatures));
        Out = StyleTtsOps.Own(layers, StyleTtsOps.Linear<T>(channels));
        Mapping = StyleTtsOps.Own(layers, StyleTtsOps.Linear<T>(features));
        MappingOutput = StyleTtsOps.Own(layers, StyleTtsOps.Linear<T>(features));
        TimeFrequencies = StyleTtsOps.Own(layers, new TiedEmbeddingLayer<T>(1, channels / 2));
        // torch.randn: standard normal frequencies.
        TimeFrequencies.Reinitialize(() => StyleTts2StyleDiffusion<T>.Gaussian(random));
        Time = StyleTtsOps.Own(layers, StyleTtsOps.Linear<T>(features));
        if (styleFeatures is not null) StyleFeaturesLayer = StyleTtsOps.Own(layers, StyleTtsOps.Linear<T>(features));
        FixedEmbedding = StyleTtsOps.Own(layers, new TiedEmbeddingLayer<T>(maxLength, embeddingFeatures));
        MaxLength = maxLength;
    }

    public int Channels { get; }
    public int EmbeddingFeatures { get; }
    public int? StyleFeatures { get; }
    public int MaxLength { get; }
    public List<StyleTts2DiffusionBlock<T>> Blocks { get; } = new();
    public DenseLayer<T> Out { get; }
    public DenseLayer<T> Mapping { get; }
    public DenseLayer<T> MappingOutput { get; }
    public TiedEmbeddingLayer<T> TimeFrequencies { get; }
    public DenseLayer<T> Time { get; }
    public DenseLayer<T>? StyleFeaturesLayer { get; }
    public TiedEmbeddingLayer<T> FixedEmbedding { get; }

    private Tensor<T> Gelu(Tensor<T> x) => ActivationHelper.ExactGelu(x, _engine);

    /// <summary>The learned unconditional embedding <c>[L, embedding]</c> for <paramref name="length"/> tokens.</summary>
    public Tensor<T> Unconditional(int length)
    {
        if (length > MaxLength) throw new ArgumentException($"At most {MaxLength} tokens.", nameof(length));
        var positions = new Tensor<T>(new[] { length });
        for (int i = 0; i < length; i++) positions[i] = NumOps.FromDouble(i);
        return FixedEmbedding.Forward(positions);
    }

    private Tensor<T> MappingOf(double time, Tensor<T>? style)
    {
        var t = new Tensor<T>(new[] { 1, 1 });
        t[0, 0] = NumOps.FromDouble(time);
        var frequencies = _engine.TensorMultiplyScalar(TimeFrequencies.Forward(new Tensor<T>(new[] { 1 })), NumOps.FromDouble(time * 2 * Math.PI));
        var fourier = _engine.TensorConcatenate(new[] { t, _engine.TensorSin(frequencies), _engine.TensorCos(frequencies) }, 1);
        var mapping = Gelu(Time.Forward(fourier));
        if (StyleFeaturesLayer is not null)
        {
            if (style is null) throw new ArgumentNullException(nameof(style), "The multispeaker denoiser reads the reference style.");
            mapping = _engine.TensorAdd(mapping, Gelu(StyleFeaturesLayer.Forward(style)));
        }
        return Gelu(MappingOutput.Forward(Gelu(Mapping.Forward(mapping))));
    }

    /// <summary>The network output <c>[1, channels]</c> for the scaled noisy style <c>[1, channels]</c>, the noise level
    /// feature <paramref name="time"/>, token embeddings <c>[L, embedding]</c> and the reference style.</summary>
    public Tensor<T> Run(Tensor<T> x, double time, Tensor<T> embedding, Tensor<T>? style)
    {
        int length = embedding.Shape[0];
        var mapping = _engine.TensorTile(MappingOf(time, style), new[] { length, 1 });
        var h = _engine.TensorConcatenate(new[] { _engine.TensorTile(x, new[] { length, 1 }), embedding }, 1);
        foreach (var block in Blocks) h = block.Forward(_engine.TensorAdd(h, mapping), style);
        return Out.Forward(_engine.ReduceMean(h, new[] { 0 }, keepDims: true));
    }

    /// <summary>The output with classifier-free guidance: <c>u + (c − u) · scale</c> of the conditional and unconditional
    /// runs (just the conditional one at scale 1). With <paramref name="maskProbability"/> the conditioning is first
    /// replaced by the unconditional table with that probability (the reference's <c>embedding_mask_proba</c>).</summary>
    public Tensor<T> Forward(Tensor<T> x, double time, Tensor<T> embedding, Tensor<T>? style, double embeddingScale,
        double maskProbability = 0.0, Random? random = null)
    {
        if (maskProbability > 0 && (random ?? throw new ArgumentNullException(nameof(random))).NextDouble() < maskProbability)
            embedding = Unconditional(embedding.Shape[0]);
        var conditional = Run(x, time, embedding, style);
        if (embeddingScale == 1.0) return conditional;
        var unconditional = Run(x, time, Unconditional(embedding.Shape[0]), style);
        return _engine.TensorAdd(unconditional, _engine.TensorMultiplyScalar(_engine.TensorSubtract(conditional, unconditional),
            NumOps.FromDouble(embeddingScale)));
    }

    public void Load(TorchReader read, string name)
    {
        int c = Channels, f = Channels + EmbeddingFeatures;
        for (int i = 0; i < Blocks.Count; i++) Blocks[i].Load(read, $"{name}.blocks.{i}");
        TorchParameters.Linear(_engine, Out, f, c, read(name + ".to_out.1.weight", new[] { c, f, 1 }), read(name + ".to_out.1.bias", new[] { c }));
        StyleTtsOps.LoadLinear(_engine, Mapping, read, name + ".to_mapping.0", f, f);
        StyleTtsOps.LoadLinear(_engine, MappingOutput, read, name + ".to_mapping.2", f, f);
        TimeFrequencies.LoadTable(read(name + ".to_time.0.0.weights", new[] { c / 2 }));
        StyleTtsOps.LoadLinear(_engine, Time, read, name + ".to_time.0.1", c + 1, f);
        if (StyleFeaturesLayer is not null) StyleTtsOps.LoadLinear(_engine, StyleFeaturesLayer, read, name + ".to_features.0", StyleFeatures!.Value, f);
        FixedEmbedding.LoadTable(read(name + ".fixed_embedding.embedding.weight", new[] { MaxLength, EmbeddingFeatures }));
    }
}

/// <summary>
/// StyleTTS 2's style diffusion (§3.2.1): the EDM preconditioning of Karras et al. (2022) around the style denoiser —
/// <c>D(x; σ) = c_skip x + c_out F(c_in x, ¼ ln σ)</c> — trained with the log-normal noise-level distribution and the EDM loss
/// weight, and sampled with the ancestral DPM-2 sampler over the Karras schedule (the reference's <c>KDiffusion</c>,
/// <c>ADPM2Sampler</c> and <c>KarrasSchedule</c>).
/// </summary>
internal sealed class StyleTts2StyleDiffusion<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public StyleTts2StyleDiffusion(IEngine engine, StyleTts2StyleDenoiser<T> denoiser, double sigmaData, double logSigmaMean, double logSigmaStd)
    {
        _engine = engine;
        Denoiser = denoiser;
        SigmaData = sigmaData;
        LogSigmaMean = logSigmaMean;
        LogSigmaStd = logSigmaStd;
    }

    public StyleTts2StyleDenoiser<T> Denoiser { get; }

    /// <summary>σ_data; the reference re-estimates it from each training batch's styles.</summary>
    public double SigmaData { get; set; }

    public double LogSigmaMean { get; }
    public double LogSigmaStd { get; }

    internal static double Gaussian(Random random) =>
        Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());

    /// <summary>The denoised style <c>[1, channels]</c> of a noisy one at noise level σ.</summary>
    public Tensor<T> Denoise(Tensor<T> noisy, double sigma, Tensor<T> embedding, Tensor<T>? style, double embeddingScale,
        double maskProbability = 0.0, Random? random = null)
    {
        double d = SigmaData, s2 = sigma * sigma + d * d;
        double cSkip = d * d / s2, cOut = sigma * d / Math.Sqrt(s2), cIn = 1.0 / Math.Sqrt(s2), cNoise = Math.Log(sigma) * 0.25;
        var prediction = Denoiser.Forward(_engine.TensorMultiplyScalar(noisy, NumOps.FromDouble(cIn)), cNoise, embedding, style, embeddingScale,
            maskProbability, random);
        return _engine.TensorAdd(_engine.TensorMultiplyScalar(noisy, NumOps.FromDouble(cSkip)), _engine.TensorMultiplyScalar(prediction, NumOps.FromDouble(cOut)));
    }

    /// <summary>The Karras et al. (2022, eq. 5) noise levels for <paramref name="steps"/> steps, then 0.</summary>
    public static double[] KarrasSchedule(int steps, double sigmaMin, double sigmaMax, double rho)
    {
        var sigmas = new double[steps + 1];
        double inverse = 1.0 / rho, max = Math.Pow(sigmaMax, inverse), min = Math.Pow(sigmaMin, inverse);
        for (int i = 0; i < steps; i++) sigmas[i] = Math.Pow(max + (steps == 1 ? 0 : (double)i / (steps - 1)) * (min - max), rho);
        return sigmas;
    }

    /// <summary>
    /// Ancestral DPM-2 sampling from <paramref name="noise"/> <c>[1, channels]</c>: <c>steps − 1</c> second-order steps from
    /// σ_i to σ_{i+1} through the midpoint level, each adding fresh noise of the ancestral level (as the reference runs
    /// it; the last level of the schedule is never reached). <paramref name="stepNoise"/> supplies each step's draw; null
    /// draws standard normals from <paramref name="random"/>.
    /// </summary>
    public Tensor<T> Sample(Tensor<T> noise, Tensor<T> embedding, Tensor<T>? style, int steps, double embeddingScale, Random random,
        Func<int, Tensor<T>>? stepNoise = null, double maskProbability = 0.0, double sigmaMin = 1e-4, double sigmaMax = 3.0, double rho = 9.0)
    {
        var sigmas = KarrasSchedule(steps, sigmaMin, sigmaMax, rho);
        var x = _engine.TensorMultiplyScalar(noise, NumOps.FromDouble(sigmas[0]));
        for (int i = 0; i < steps - 1; i++)
        {
            double sigma = sigmas[i], next = sigmas[i + 1];
            double up = Math.Sqrt(next * next * (sigma * sigma - next * next) / (sigma * sigma));
            double down = Math.Sqrt(next * next - up * up), mid = (sigma + down) / 2;
            var d = _engine.TensorMultiplyScalar(_engine.TensorSubtract(x, Denoise(x, sigma, embedding, style, embeddingScale, maskProbability, random)),
                NumOps.FromDouble(1.0 / sigma));
            var xMid = _engine.TensorAdd(x, _engine.TensorMultiplyScalar(d, NumOps.FromDouble(mid - sigma)));
            var dMid = _engine.TensorMultiplyScalar(_engine.TensorSubtract(xMid, Denoise(xMid, mid, embedding, style, embeddingScale, maskProbability, random)),
                NumOps.FromDouble(1.0 / mid));
            x = _engine.TensorAdd(x, _engine.TensorMultiplyScalar(dMid, NumOps.FromDouble(down - sigma)));
            Tensor<T> fresh;
            if (stepNoise is not null) fresh = stepNoise(i);
            else
            {
                fresh = new Tensor<T>(x._shape);
                for (int j = 0; j < fresh.Length; j++) fresh[j] = NumOps.FromDouble(Gaussian(random));
            }
            x = _engine.TensorAdd(x, _engine.TensorMultiplyScalar(fresh, NumOps.FromDouble(up)));
        }
        return x;
    }

    /// <summary>
    /// The EDM training loss of a style <c>[1, channels]</c>: σ = exp(N(mean, std)), the denoising error at σ weighted by
    /// <c>(σ² + σ_data²) / (σ σ_data)²</c>; the conditioning is replaced by the unconditional table with probability
    /// <paramref name="maskProbability"/> (classifier-free guidance training).
    /// </summary>
    public Tensor<T> Loss(Tensor<T> target, Tensor<T> embedding, Tensor<T>? style, double maskProbability, Random random)
    {
        double sigma = Math.Exp(LogSigmaMean + LogSigmaStd * Gaussian(random));
        var noise = new Tensor<T>(target._shape);
        for (int j = 0; j < noise.Length; j++) noise[j] = NumOps.FromDouble(Gaussian(random));
        if (maskProbability > 0 && random.NextDouble() < maskProbability) embedding = Denoiser.Unconditional(embedding.Shape[0]);
        return Loss(target, embedding, style, sigma, noise);
    }

    /// <summary>The EDM loss at noise level <paramref name="sigma"/> with the given standard-normal
    /// <paramref name="noise"/> (the reference's <c>KDiffusion.forward</c>).</summary>
    internal Tensor<T> Loss(Tensor<T> target, Tensor<T> embedding, Tensor<T>? style, double sigma, Tensor<T> noise)
    {
        var noisy = _engine.TensorAdd(target, _engine.TensorMultiplyScalar(noise, NumOps.FromDouble(sigma)));
        var error = _engine.TensorSubtract(Denoise(noisy, sigma, embedding, style, 1.0), target);
        double d = SigmaData, weight = (sigma * sigma + d * d) / (sigma * d * sigma * d);
        return _engine.TensorMultiplyScalar(_engine.ReduceMean(_engine.TensorMultiply(error, error), new[] { 0, 1 }, keepDims: false),
            NumOps.FromDouble(weight));
    }
}

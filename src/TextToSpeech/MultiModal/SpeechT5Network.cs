using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>
/// SpeechT5's speech encoder pre-net (paper §2.1: wav2vec 2.0's feature extractor; the official
/// <c>SpeechEncoderPrenet</c>): bias-free strided convolutions with GELU (the first instance-normalized per channel), the
/// features' L2 penalty, a layer norm and a projection to the model width, the masked frames replaced by a learned
/// vector, a weight-normalized grouped convolutional position term and fairseq's sinusoidal positions.
/// </summary>
internal sealed class SpeechT5SpeechEncoderPrenet<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly SpeechT5Configuration _c;
    private readonly int _features;

    public SpeechT5SpeechEncoderPrenet(IEngine engine, List<LayerBase<T>> layers, SpeechT5Configuration c, Random random)
    {
        _engine = engine;
        _c = c;
        int inputs = 1;
        for (int i = 0; i < c.FeatureLayers.Length; i++)
        {
            var (channels, kernel, stride) = c.FeatureLayers[i];
            var conv = new NormedConv1DLayer<T>(inputs, channels, kernel, stride, 1, 1, 0, false, ConvolutionNormalization.None, false);
            layers.Add(conv);
            // kaiming_normal_ (fan_in, gain √2).
            conv.LoadTorchWeights(IndexTtsOps.Normal(random, channels * inputs * kernel, Math.Sqrt(2.0 / (inputs * kernel))), null, new double[channels]);
            Convolutions.Add(conv);
            inputs = channels;
        }
        _features = inputs;
        GroupGamma = IndexTtsOps.Values(layers, Enumerable.Repeat(1.0, c.FeatureLayers[0].Channels).ToArray());
        GroupBeta = IndexTtsOps.Values(layers, new double[c.FeatureLayers[0].Channels]);
        FeatureNorm = IndexTtsOps.LayerNorm(layers, _features);
        Projection = _features != c.Hidden ? IndexTtsOps.Linear(engine, layers, random, _features, c.Hidden) : null;
        MaskEmbedding = IndexTtsOps.Values(layers, Enumerable.Range(0, c.Hidden).Select(_ => random.NextDouble()).ToArray());
        int per = c.Hidden / c.ConvPositionGroups, k = c.ConvPositionKernel;
        double std = Math.Sqrt(4.0 / (k * c.Hidden));
        var v = IndexTtsOps.Normal(random, c.Hidden * per * k, std);
        PositionDirection = IndexTtsOps.Values(layers, v);
        // weight_norm(dim=2) starts with g = ‖v‖ over the output and input dimensions, per kernel tap.
        PositionGain = IndexTtsOps.Values(layers, Enumerable.Range(0, k).Select(t =>
            Math.Sqrt(Enumerable.Range(0, c.Hidden * per).Sum(r => v[r * k + t] * v[r * k + t]))).ToArray());
        PositionBias = IndexTtsOps.Values(layers, new double[c.Hidden]);
    }

    public List<NormedConv1DLayer<T>> Convolutions { get; } = new();
    public BiasFreeLinearLayer<T> GroupGamma { get; }
    public BiasFreeLinearLayer<T> GroupBeta { get; }
    public LayerNormalizationLayer<T> FeatureNorm { get; }
    public DenseLayer<T>? Projection { get; }
    public BiasFreeLinearLayer<T> MaskEmbedding { get; }
    public BiasFreeLinearLayer<T> PositionDirection { get; }
    public BiasFreeLinearLayer<T> PositionGain { get; }
    public BiasFreeLinearLayer<T> PositionBias { get; }

    /// <summary>Feature frames of <paramref name="samples"/> samples.</summary>
    public int Frames(int samples)
    {
        int n = samples;
        foreach (var (_, kernel, stride) in _c.FeatureLayers) n = (n - kernel) / stride + 1;
        return n;
    }

    /// <summary>The extracted features <c>[F, C]</c> (before the layer norm) of a waveform.</summary>
    public Tensor<T> Features(Tensor<T> wave)
    {
        var x = _engine.Reshape(wave, new[] { 1, 1, wave.Length });
        for (int i = 0; i < Convolutions.Count; i++)
        {
            x = Convolutions[i].Forward(x);
            if (i == 0)
            {
                // Fp32GroupNorm(C, C): each channel normalized over time (ε 1e-5), then the affine.
                int c = x.Shape[1], t = x.Shape[2];
                var shape = x.Shape.ToArray();
                var mean = _engine.TensorBroadcastTo(_engine.ReduceMean(x, new[] { 2 }, true), shape);
                var centered = _engine.TensorSubtract(x, mean);
                var variance = _engine.TensorBroadcastTo(_engine.ReduceMean(_engine.TensorMultiply(centered, centered), new[] { 2 }, true), shape);
                x = _engine.TensorDivide(centered, _engine.TensorSqrt(_engine.TensorAddScalar(variance, NumOps.FromDouble(1e-5))));
                x = _engine.TensorAdd(_engine.TensorMultiply(x, _engine.TensorBroadcastTo(_engine.Reshape(IndexTtsOps.Value(GroupGamma), new[] { 1, c, 1 }), shape)),
                    _engine.TensorBroadcastTo(_engine.Reshape(IndexTtsOps.Value(GroupBeta), new[] { 1, c, 1 }), shape));
            }
            x = SpeechT5Ops.Gelu(_engine, x);
        }
        int frames = x.Shape[2];
        return _engine.TensorTranspose(_engine.Reshape(x, new[] { _features, frames }));
    }

    /// <summary>
    /// The encoder inputs <c>[F, d]</c> of a waveform with the frames in <paramref name="mask"/> replaced by the mask
    /// embedding, and the features' penalty <c>mean(f²)</c>.
    /// </summary>
    public (Tensor<T> Inputs, Tensor<T> Penalty) Forward(Tensor<T> wave, bool[]? mask, bool training, Random? random, int? frames = null)
    {
        var features = Features(wave);
        // forward_targets crops the features to the frames the labels cover.
        if (frames is int keep && keep < features.Shape[0]) features = _engine.TensorNarrow(features, 0, 0, keep);
        int count = features.Shape[0], d = _c.Hidden;
        var penalty = _engine.ReduceMean(_engine.TensorMultiply(features, features), new[] { 0, 1 }, false);
        var x = FeatureNorm.Forward(features);
        if (Projection is not null) x = Projection.Forward(x);
        x = SpeechT5Ops.Drop(_engine, x, _c.Dropout, training, random);
        if (mask is not null)
        {
            if (mask.Length != count) throw new ArgumentException($"The mask covers {mask.Length} frames, the features {count}.", nameof(mask));
            var unmasked = new Tensor<T>(new[] { count, d });
            var put = new Tensor<T>(new[] { count, d });
            for (int t = 0; t < count; t++)
                for (int k = 0; k < d; k++) (mask[t] ? put : unmasked)[t, k] = NumOps.One;
            var emb = _engine.TensorBroadcastTo(_engine.Reshape(IndexTtsOps.Value(MaskEmbedding), new[] { 1, d }), new[] { count, d });
            x = _engine.TensorAdd(_engine.TensorMultiply(x, unmasked), _engine.TensorMultiply(emb, put));
        }
        x = _engine.TensorAdd(x, ConvolutionalPositions(x));
        x = _engine.TensorAdd(x, SpeechT5Ops.FairseqTable<T>(count, d, 2));    // padding index 1: positions from 2
        return (x, penalty);
    }

    /// <summary>The grouped convolution with weight norm over the kernel taps (<c>weight_norm(dim=2)</c>), the trailing
    /// frame of an even kernel dropped (<c>SamePad</c>), and GELU.</summary>
    private Tensor<T> ConvolutionalPositions(Tensor<T> x)
    {
        int d = _c.Hidden, k = _c.ConvPositionKernel, groups = _c.ConvPositionGroups, per = d / groups, frames = x.Shape[0];
        var v = _engine.Reshape(IndexTtsOps.Value(PositionDirection), new[] { d * per, k });
        var norm = _engine.TensorSqrt(_engine.ReduceSum(_engine.TensorMultiply(v, v), new[] { 0 }, true));          // [1, k]
        var g = _engine.Reshape(IndexTtsOps.Value(PositionGain), new[] { 1, k });
        var w = _engine.TensorMultiply(v, _engine.TensorBroadcastTo(_engine.TensorDivide(g, norm), new[] { d * per, k }));
        var kernel = _engine.Reshape(w, new[] { d, per, 1, k });
        var input = _engine.Reshape(_engine.TensorTranspose(x), new[] { 1, d, 1, frames });
        var parts = new Tensor<T>[groups];
        for (int i = 0; i < groups; i++)
            parts[i] = _engine.Conv2D(_engine.TensorSlice(input, new[] { 0, i * per, 0, 0 }, new[] { 1, per, 1, frames }),
                _engine.TensorSlice(kernel, new[] { i * per, 0, 0, 0 }, new[] { per, per, 1, k }), new[] { 1, 1 }, new[] { 0, k / 2 }, new[] { 1, 1 });
        var joined = _engine.TensorConcatenate(parts, 1);                                                   // [1, d, 1, L]
        int length = joined.Shape[3];
        var y = _engine.Reshape(joined, new[] { d, length });
        y = _engine.TensorAdd(y, _engine.TensorBroadcastTo(_engine.Reshape(IndexTtsOps.Value(PositionBias), new[] { d, 1 }), new[] { d, length }));
        if (k % 2 == 0) y = _engine.TensorNarrow(y, 1, 0, length - 1);
        return SpeechT5Ops.Gelu(_engine, _engine.TensorTranspose(y));
    }

    public void Load(Func<string, int[], double[]> read, string prefix)
    {
        int inputs = 1;
        for (int i = 0; i < Convolutions.Count; i++)
        {
            var (channels, kernel, _) = _c.FeatureLayers[i];
            Convolutions[i].LoadTorchWeights(read($"{prefix}feature_extractor.conv_layers.{i}.0.weight", new[] { channels, inputs, kernel }), null, new double[channels]);
            inputs = channels;
        }
        int c0 = _c.FeatureLayers[0].Channels;
        TorchParameters.Linear(GroupGamma, 1, c0, read(prefix + "feature_extractor.conv_layers.0.2.weight", new[] { c0 }));
        TorchParameters.Linear(GroupBeta, 1, c0, read(prefix + "feature_extractor.conv_layers.0.2.bias", new[] { c0 }));
        IndexTtsOps.LoadNorm(_engine, read, FeatureNorm, prefix + "layer_norm", _features);
        if (Projection is not null) IndexTtsOps.LoadLinear(_engine, read, Projection, prefix + "post_extract_proj", _features, _c.Hidden);
        TorchParameters.Linear(MaskEmbedding, 1, _c.Hidden, read(prefix + "mask_emb", new[] { _c.Hidden }));
        int d = _c.Hidden, per = d / _c.ConvPositionGroups, k = _c.ConvPositionKernel;
        TorchParameters.Linear(PositionDirection, 1, d * per * k, read(prefix + "pos_conv.0.weight_v", new[] { d, per, k }));
        TorchParameters.Linear(PositionGain, 1, k, read(prefix + "pos_conv.0.weight_g", new[] { 1, 1, k }));
        TorchParameters.Linear(PositionBias, 1, d, read(prefix + "pos_conv.0.bias", new[] { d }));
    }
}

/// <summary>
/// SpeechT5's HuBERT post-net (the official <c>SpeechEncoderPostnet</c>): the masked frames' encoder states projected to
/// the label space, each scored against every label embedding by cosine similarity over a temperature, the target's own
/// column as the first logit (its duplicate among the others set to −∞), so the cross-entropy target is 0.
/// </summary>
internal sealed class SpeechT5HubertPostnet<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly SpeechT5Configuration _c;

    public SpeechT5HubertPostnet(IEngine engine, List<LayerBase<T>> layers, SpeechT5Configuration c, Random random)
    {
        _engine = engine;
        _c = c;
        int classes = c.HubertClasses.Sum();
        LabelEmbeddings = IndexTtsOps.Values(layers, Enumerable.Range(0, classes * c.FinalDim).Select(_ => random.NextDouble()).ToArray());
        Projection = IndexTtsOps.Linear(engine, layers, random, c.Hidden, c.FinalDim * c.HubertClasses.Length);
    }

    public BiasFreeLinearLayer<T> LabelEmbeddings { get; }
    public DenseLayer<T> Projection { get; }

    /// <summary>The logits <c>[M, 1 + C]</c> of each label set for the masked frames of encoder states <c>[F, d]</c>.</summary>
    public List<Tensor<T>> MaskedLogits(Tensor<T> states, bool[] mask, IReadOnlyList<int[]> targets)
    {
        var rows = Enumerable.Range(0, mask.Length).Where(t => mask[t]).ToArray();
        var picked = _engine.TensorIndexSelect(states, new Tensor<int>(new[] { rows.Length }, new Vector<int>(rows)), 0);
        var projected = Projection.Forward(picked);                                                      // [M, final·n]
        var all = _engine.Reshape(IndexTtsOps.Value(LabelEmbeddings), new[] { _c.HubertClasses.Sum(), _c.FinalDim });
        var result = new List<Tensor<T>>();
        int offset = 0;
        for (int set = 0; set < _c.HubertClasses.Length; set++)
        {
            int classes = _c.HubertClasses[set];
            var x = _engine.TensorNarrow(projected, 1, set * _c.FinalDim, _c.FinalDim);
            var embeddings = _engine.TensorNarrow(all, 0, offset, classes);
            offset += classes;
            // cosine_similarity: x·e / max(‖x‖·‖e‖, 1e-8).
            var xn = _engine.TensorSqrt(_engine.ReduceSum(_engine.TensorMultiply(x, x), new[] { 1 }, true));            // [M, 1]
            var en = _engine.TensorSqrt(_engine.ReduceSum(_engine.TensorMultiply(embeddings, embeddings), new[] { 1 }, true)); // [C, 1]
            var dots = _engine.TensorMatMul(x, _engine.TensorTranspose(embeddings));                                       // [M, C]
            var denominator = _engine.TensorMax(_engine.TensorMatMul(xn, _engine.TensorTranspose(en)), NumOps.FromDouble(1e-8));
            var cosine = _engine.TensorMultiplyScalar(_engine.TensorDivide(dots, denominator), NumOps.FromDouble(1.0 / _c.LogitTemperature));
            var target = new Tensor<int>(new[] { rows.Length });
            var hide = new Tensor<T>(new[] { rows.Length, classes });
            for (int m = 0; m < rows.Length; m++)
            {
                target[m] = m * classes + targets[set][rows[m]];
                hide[m, targets[set][rows[m]]] = NumOps.FromDouble(double.NegativeInfinity);
            }
            var positive = _engine.Reshape(_engine.TensorIndexSelect(_engine.Reshape(cosine, new[] { rows.Length * classes, 1 }), target, 0), new[] { rows.Length, 1 });
            result.Add(_engine.TensorConcatenate(new[] { positive, _engine.TensorAdd(cosine, hide) }, 1));
        }
        return result;
    }

    public void Load(Func<string, int[], double[]> read, string prefix)
    {
        int classes = _c.HubertClasses.Sum();
        TorchParameters.Linear(LabelEmbeddings, 1, classes * _c.FinalDim, read(prefix + "label_embs_concat", new[] { classes, _c.FinalDim }));
        IndexTtsOps.LoadLinear(_engine, read, Projection, prefix + "final_proj", _c.Hidden, _c.FinalDim * _c.HubertClasses.Length);
    }
}

/// <summary>
/// SpeechT5's cross-modal vector quantizer (paper §2.2; fairseq's <c>GumbelVectorQuantizer</c>): G groups of V code
/// vectors chosen per frame from a linear map of the encoder state — by Gumbel-softmax (hard, straight-through) at an
/// annealed temperature in training, by argmax otherwise — and the diversity term from the batch-averaged softmax's
/// perplexity.
/// </summary>
internal sealed class SpeechT5Quantizer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _d, _groups, _vars;

    public SpeechT5Quantizer(IEngine engine, List<LayerBase<T>> layers, Random random, int d, int groups, int vars)
    {
        _engine = engine;
        _d = d;
        _groups = groups;
        _vars = vars;
        Projection = IndexTtsOps.Linear(engine, layers, random, d, groups * vars);
        TorchParameters.Linear(engine, Projection, d, groups * vars, IndexTtsOps.Normal(random, d * groups * vars, 1.0), new double[groups * vars]);
        Vectors = IndexTtsOps.Values(layers, Enumerable.Range(0, groups * vars * (d / groups)).Select(_ => random.NextDouble()).ToArray());
    }

    public DenseLayer<T> Projection { get; }
    public BiasFreeLinearLayer<T> Vectors { get; }

    /// <summary>The Gumbel temperature: (2, 0.5, 0.999995) annealed per update.</summary>
    public double Temperature { get; set; } = 2.0;

    /// <summary>The quantized states <c>[T, d]</c> and the perplexity of the average code probabilities.</summary>
    public (Tensor<T> Quantized, Tensor<T> Perplexity) Forward(Tensor<T> x, bool training, Random? random)
    {
        int t = x.Shape[0], gv = _groups * _vars, varDim = _d / _groups;
        var logits = _engine.Reshape(Projection.Forward(x), new[] { t * _groups, _vars });
        var probabilities = _engine.Softmax(logits, axis: 1);
        var average = _engine.ReduceMean(_engine.Reshape(probabilities, new[] { t, _groups, _vars }), new[] { 0 }, false);  // [G, V]
        var entropy = _engine.ReduceSum(_engine.TensorMultiply(average, _engine.TensorLog(_engine.TensorAddScalar(average, NumOps.FromDouble(1e-7)))), new[] { 1 }, false);
        var perplexity = _engine.ReduceSum(_engine.TensorExp(_engine.TensorNegate(entropy)), null, false);

        Tensor<T> onehot;
        if (training && random is not null)
        {
            // F.gumbel_softmax(hard=True): the argmax of the noised soft sample, with its gradient (y_hard − sg(y) + y).
            var noise = new Tensor<T>(new[] { t * _groups, _vars });
            for (int i = 0; i < noise.Length; i++) noise[i] = NumOps.FromDouble(-Math.Log(-Math.Log(Math.Max(random.NextDouble(), 1e-300))));
            var soft = _engine.Softmax(_engine.TensorMultiplyScalar(_engine.TensorAdd(logits, noise), NumOps.FromDouble(1.0 / Temperature)), axis: 1);
            var hard = OneHot(soft);
            onehot = _engine.TensorAdd(_engine.TensorSubtract(hard, new Tensor<T>(soft.Shape.ToArray(), soft.ToVector())), soft);
        }
        else
        {
            onehot = OneHot(logits);
        }
        // Each group's chosen vector, concatenated: per group a [V] × [V, varDim] product.
        var vectors = _engine.Reshape(IndexTtsOps.Value(Vectors), new[] { _groups, _vars, varDim });
        var perGroup = _engine.Reshape(onehot, new[] { t, _groups, _vars });
        var parts = new Tensor<T>[_groups];
        for (int g = 0; g < _groups; g++)
            parts[g] = _engine.TensorMatMul(_engine.Reshape(_engine.TensorNarrow(perGroup, 1, g, 1), new[] { t, _vars }),
                _engine.Reshape(_engine.TensorNarrow(vectors, 0, g, 1), new[] { _vars, varDim }));
        return (_engine.TensorConcatenate(parts, 1), perplexity);
    }

    private Tensor<T> OneHot(Tensor<T> scores)
    {
        int rows = scores.Shape[0], cols = scores.Shape[1];
        var onehot = new Tensor<T>(new[] { rows, cols });
        for (int r = 0; r < rows; r++)
        {
            int best = 0;
            for (int c = 1; c < cols; c++)
                if (NumOps.ToDouble(scores[r, c]) > NumOps.ToDouble(scores[r, best])) best = c;
            onehot[r, best] = NumOps.One;
        }
        return onehot;
    }

    /// <summary>The diversity term <c>(GV − perplexity) / GV</c>.</summary>
    public Tensor<T> Diversity(Tensor<T> perplexity)
    {
        double gv = _groups * _vars;
        return _engine.TensorMultiplyScalar(_engine.TensorAddScalar(_engine.TensorNegate(perplexity), NumOps.FromDouble(gv)), NumOps.FromDouble(1.0 / gv));
    }

    public void Load(Func<string, int[], double[]> read, string prefix)
    {
        IndexTtsOps.LoadLinear(_engine, read, Projection, prefix + "weight_proj", _d, _groups * _vars);
        TorchParameters.Linear(Vectors, 1, _groups * _vars * (_d / _groups), read(prefix + "vars", new[] { 1, _groups * _vars, _d / _groups }));
    }
}

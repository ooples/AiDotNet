using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Audio.Foundations;

/// <summary>wav2vec 2.0's self-supervised objective (Baevski et al. 2020, §3.2–3.3, §4.2; BASE values).</summary>
internal sealed record Wav2Vec2PretrainingConfiguration
{
    /// <summary>G codebooks of V entries (G = 2, V = 320).</summary>
    public int Groups { get; init; } = 2;
    public int EntriesPerGroup { get; init; } = 320;

    /// <summary>The quantized vector's width, d (the codebook entries are d/G = 128 wide).</summary>
    public int CodevectorDim { get; init; } = 256;

    /// <summary>The width both the context and the quantized vectors are projected to for the contrastive loss.</summary>
    public int ProjectedDim { get; init; } = 256;

    /// <summary>K distractors per masked step (100), drawn from other masked steps of the same utterance.</summary>
    public int Negatives { get; init; } = 100;

    /// <summary>The contrastive temperature κ (0.1).</summary>
    public double LogitTemperature { get; init; } = 0.1;

    /// <summary>The diversity loss weight α (0.1).</summary>
    public double DiversityWeight { get; init; } = 0.1;

    /// <summary>The weight of the L2 penalty on the feature encoder's output (fairseq's <c>features_pen</c> weight 10).</summary>
    public double FeaturePenaltyWeight { get; init; } = 10.0;

    /// <summary>The probability a frame starts a masked span (p = 0.065) of M = 10 frames.</summary>
    public double MaskProbability { get; init; } = 0.065;

    /// <summary>The factor on the feature encoder's gradients (0.1).</summary>
    public double FeatureGradientScale { get; init; } = 0.1;

    /// <summary>The Gumbel-softmax temperature, annealed from 2 to 0.5 by a factor of 0.999995 per update.</summary>
    public double MaxTemperature { get; init; } = 2.0;
    public double MinTemperature { get; init; } = 0.5;
    public double TemperatureDecay { get; init; } = 0.999995;
}

/// <summary>
/// wav2vec 2.0 pre-training (transformers <c>Wav2Vec2ForPreTraining</c> and <c>Wav2Vec2GumbelVectorQuantizer</c>,
/// with fairseq's feature-encoder gradient scaling and activation penalty): the unmasked features are quantized with a
/// Gumbel-softmax over G codebooks, the context network sees the masked features, and the loss is the contrastive
/// task of finding each masked step's quantized vector among K distractors by cosine similarity over κ, plus α times
/// the codebook diversity loss.
/// </summary>
internal sealed class Wav2Vec2Pretraining<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly int _featureDim;
    private readonly int _hiddenSize;

    public Wav2Vec2Pretraining(IEngine engine, List<LayerBase<T>> layers, Wav2Vec2PretrainingConfiguration configuration, int featureDim, int hiddenSize)
    {
        _engine = engine;
        Configuration = configuration;
        _featureDim = featureDim;
        _hiddenSize = hiddenSize;
        var c = configuration;
        if (c.CodevectorDim % c.Groups != 0) throw new ArgumentException("The codevector width must divide into the codebooks.");
        TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            layers.Add(layer);
            return layer;
        }
        DenseLayer<T> Linear(int outputs) => Own(new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()));
        WeightProjection = Linear(c.Groups * c.EntriesPerGroup);
        for (int g = 0; g < c.Groups; g++) Codebooks.Add(Own(new BiasFreeLinearLayer<T>(c.EntriesPerGroup, c.CodevectorDim / c.Groups)));
        ProjectHidden = Linear(c.ProjectedDim);
        ProjectQuantized = Linear(c.ProjectedDim);
    }

    public Wav2Vec2PretrainingConfiguration Configuration { get; }

    /// <summary>The logits of every codebook entry from the features (weight_proj).</summary>
    public DenseLayer<T> WeightProjection { get; }

    /// <summary>Each codebook as a map from its entry probabilities to the entry (rows: the codevectors).</summary>
    public List<BiasFreeLinearLayer<T>> Codebooks { get; } = new();

    public DenseLayer<T> ProjectHidden { get; }
    public DenseLayer<T> ProjectQuantized { get; }

    /// <summary>The current Gumbel-softmax temperature.</summary>
    public double Temperature { get; set; } = 2.0;

    /// <summary>Anneals the temperature after an update.</summary>
    public void AfterUpdate()
        => Temperature = Math.Max(Temperature * Configuration.TemperatureDecay, Configuration.MinTemperature);

    /// <summary>The loss terms of one batch.</summary>
    internal sealed record Losses(Tensor<T> Total, Tensor<T> Contrastive, Tensor<T> Diversity, Tensor<T> FeaturePenalty, int Masked, int[,] CodeIds);

    private Tensor<T> Dense(LayerBase<T> layer, Tensor<T> x)
    {
        int batch = x.Shape[0], length = x.Shape[1];
        var y = layer.Forward(_engine.Reshape(x, new[] { batch * length, x.Shape[2] }));
        return _engine.Reshape(y, new[] { batch, length, y.Shape[1] });
    }

    private static Tensor<T> Constant(Tensor<T> x) => new(x._shape, new Vector<T>(x.ToArray()));

    // The quantized features [B, L, d] and the codebook perplexity (Σ_g exp(entropy of the masked marginal)).
    // fixedIds, when given, replaces each step's argmax with its recorded entry (the groups' indices combined as below);
    // the Gumbel noise is still drawn, so the random stream stays aligned with an unfixed draw.
    private (Tensor<T> Quantized, Tensor<T> Perplexity, int[,] Ids) Quantize(Tensor<T> features, bool[,] mask, bool training, Random random,
        int[,]? fixedIds)
    {
        var c = Configuration;
        int batch = features.Shape[0], length = features.Shape[1], rows = batch * length, v = c.EntriesPerGroup;
        var logits = _engine.Reshape(Dense(WeightProjection, features), new[] { rows, c.Groups * v });
        int maskedCount = 0;
        var maskWeights = new Tensor<T>(new[] { rows, 1 });
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < length; t++)
                if (mask[b, t])
                {
                    maskWeights[b * length + t, 0] = NumOps.One;
                    maskedCount++;
                }
        var parts = new Tensor<T>[c.Groups];
        // Each step's assigned entry, the groups' indices combined as one id (sum over g of id_g * V^g).
        var ids = new int[batch, length];
        int place = 1;
        Tensor<T>? perplexity = null;
        for (int g = 0; g < c.Groups; g++)
        {
            var group = _engine.TensorSlice(logits, new[] { 0, g * v }, new[] { rows, v });
            var values = group.ToArray();
            Tensor<T> probabilities, marginalSource;
            var hard = new Tensor<T>(new[] { rows, v });
            if (training)
            {
                // Straight-through Gumbel-softmax (hard = true): forward one-hot, backward through the soft sample.
                var noise = new Tensor<T>(new[] { rows, v });
                for (int i = 0; i < noise.Length; i++) noise[i] = NumOps.FromDouble(-Math.Log(-Math.Log(1.0 - random.NextDouble())));
                var soft = _engine.Softmax(_engine.TensorMultiplyScalar(_engine.TensorAdd(group, noise), NumOps.FromDouble(1.0 / Temperature)), axis: 1);
                var sv = soft.ToArray();
                for (int r = 0; r < rows; r++)
                {
                    int best = 0;
                    if (fixedIds is not null) best = fixedIds[r / length, r % length] / place % v;
                    else for (int j = 1; j < v; j++) if (NumOps.ToDouble(sv[r * v + j]) > NumOps.ToDouble(sv[r * v + best])) best = j;
                    hard[r, best] = NumOps.One;
                    ids[r / length, r % length] += best * place;
                }
                probabilities = _engine.TensorAdd(soft, _engine.TensorSubtract(hard, Constant(soft)));
                marginalSource = _engine.Softmax(group, axis: 1);
            }
            else
            {
                for (int r = 0; r < rows; r++)
                {
                    int best = 0;
                    if (fixedIds is not null) best = fixedIds[r / length, r % length] / place % v;
                    else for (int j = 1; j < v; j++) if (NumOps.ToDouble(values[r * v + j]) > NumOps.ToDouble(values[r * v + best])) best = j;
                    hard[r, best] = NumOps.One;
                    ids[r / length, r % length] += best * place;
                }
                probabilities = hard;
                marginalSource = hard;
            }
            parts[g] = Codebooks[g].Forward(probabilities);
            // The marginal over the masked steps, and exp of its entropy (xlogy: 0·log 0 = 0).
            var marginal = _engine.TensorMultiplyScalar(
                _engine.ReduceSum(_engine.TensorMultiply(marginalSource, _engine.TensorTile(maskWeights, new[] { 1, v })), new[] { 0 }, keepDims: false),
                NumOps.FromDouble(1.0 / Math.Max(maskedCount, 1)));
            // xlogy: an entry of zero probability contributes 0·log 0 = 0 (its log taken of 1 instead).
            var mv = marginal.ToArray();
            var zeros = new Tensor<T>(marginal._shape);
            for (int i = 0; i < mv.Length; i++) zeros[i] = NumOps.ToDouble(mv[i]) == 0 ? NumOps.One : NumOps.Zero;
            var safe = _engine.TensorAdd(marginal, zeros);
            var entropy = _engine.TensorMultiplyScalar(_engine.ReduceSum(_engine.TensorMultiply(marginal, _engine.TensorLog(safe)), new[] { 0 }, keepDims: false), NumOps.FromDouble(-1.0));
            var term = _engine.TensorExp(entropy);
            perplexity = perplexity is null ? term : _engine.TensorAdd(perplexity, term);
            place *= v;
        }
        var quantized = _engine.Reshape(c.Groups == 1 ? parts[0] : _engine.TensorConcatenate(parts, 1), new[] { batch, length, c.CodevectorDim });
        return (quantized, perplexity ?? throw new InvalidOperationException("The quantizer has no codebooks."), ids);
    }

    /// <summary>
    /// Distractor indices <c>[B, L, K]</c> into the flattened <c>[B·L]</c> steps (transformers
    /// <c>_sample_negative_indices</c>): for each masked step, K draws with replacement among the other masked steps of
    /// its utterance.
    /// </summary>
    public int[,,] SampleNegatives(bool[,] mask, Random random)
    {
        int batch = mask.GetLength(0), length = mask.GetLength(1), k = Configuration.Negatives;
        var negatives = new int[batch, length, k];
        for (int b = 0; b < batch; b++)
        {
            var masked = Enumerable.Range(0, length).Where(t => mask[b, t]).ToArray();
            if (masked.Length < 2) continue;
            for (int m = 0; m < masked.Length; m++)
                for (int j = 0; j < k; j++)
                {
                    int pick = random.Next(masked.Length - 1);
                    if (pick >= m) pick++;
                    negatives[b, masked[m], j] = b * length + masked[pick];
                }
        }
        return negatives;
    }

    /// <summary>The pre-training loss of a forward with its applied mask and distractor indices.</summary>
    /// <param name="output">The forward, with its applied mask.</param>
    /// <param name="negatives">Each step's distractor indices.</param>
    /// <param name="training">Whether the quantizer samples (straight-through Gumbel-softmax) or takes the argmax.</param>
    /// <param name="random">The Gumbel noise's generator.</param>
    /// <param name="fixedIds">Each step's codebook entry to use in place of the quantizer's choice (the <see cref="Losses.CodeIds"/>
    /// of an earlier call), or null for the quantizer's own; gradients still flow through the soft sample.</param>
    public Losses Loss(Wav2Vec2Network<T>.Output output, int[,,] negatives, bool training, Random random, int[,]? fixedIds = null)
    {
        var c = Configuration;
        var mask = output.Mask ?? throw new ArgumentException("Pre-training needs the forward's time mask.");
        int batch = output.Hidden.Shape[0], length = output.Hidden.Shape[1], k = negatives.GetLength(2), p = c.ProjectedDim;
        var predicted = Dense(ProjectHidden, output.Hidden);                                                       // [B, L, P]
        var (quantized, perplexity, ids) = Quantize(output.Features, mask, training, random, fixedIds);
        var targets = Dense(ProjectQuantized, quantized);                                                           // [B, L, P]

        // Candidates [B, L, 1 + K, P]: the step's own target, then its distractors.
        var flat = _engine.Reshape(targets, new[] { batch * length, p });
        var indices = new int[batch * length * k];
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < length; t++)
                for (int j = 0; j < k; j++) indices[(b * length + t) * k + j] = negatives[b, t, j];
        var distractors = _engine.Reshape(_engine.TensorGather(flat, new Tensor<int>(new[] { indices.Length }, new Vector<int>(indices)), axis: 0),
            new[] { batch, length, k, p });
        var candidates = _engine.TensorConcatenate(new[] { _engine.Reshape(targets, new[] { batch, length, 1, p }), distractors }, 2);

        // Cosine similarity over κ.
        Tensor<T> Norm(Tensor<T> x, int axis) => _engine.TensorSqrt(_engine.ReduceSum(_engine.TensorMultiply(x, x), new[] { axis }, keepDims: true));
        var pred = _engine.Reshape(predicted, new[] { batch, length, 1, p });
        var dots = _engine.ReduceSum(_engine.TensorMultiply(_engine.TensorTile(pred, new[] { 1, 1, k + 1, 1 }), candidates), new[] { 3 }, keepDims: false);
        var predictedNorm = _engine.TensorTile(_engine.Reshape(Norm(pred, 3), new[] { batch, length, 1 }), new[] { 1, 1, k + 1 });
        var norms = _engine.TensorMultiply(predictedNorm, _engine.Reshape(Norm(candidates, 3), new[] { batch, length, k + 1 }));
        // torch.cosine_similarity divides by max(‖a‖·‖b‖, 1e-8).
        var nv = norms.ToArray();
        var floor = new Tensor<T>(norms._shape);
        for (int i = 0; i < nv.Length; i++) floor[i] = NumOps.FromDouble(Math.Max(0.0, 1e-8 - NumOps.ToDouble(nv[i])));
        var logits = _engine.TensorMultiplyScalar(_engine.TensorDivide(dots, _engine.TensorAdd(norms, floor)),
            NumOps.FromDouble(1.0 / c.LogitTemperature));                                                         // [B, L, 1 + K]

        // A distractor identical to the target is excluded (−∞), as in the reference.
        var tv = targets.ToArray();
        var exclude = new Tensor<T>(new[] { batch, length, k + 1 });
        var minusInfinity = NumOps.FromDouble(double.NegativeInfinity);
        int masked = 0;
        var weights = new Tensor<T>(new[] { batch, length });
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < length; t++)
            {
                if (!mask[b, t]) continue;
                masked++;
                weights[b, t] = NumOps.One;
                int self = b * length + t;
                for (int j = 0; j < k; j++)
                {
                    int other = negatives[b, t, j];
                    bool same = true;
                    for (int d = 0; d < p && same; d++) same = NumOps.ToDouble(tv[self * p + d]) == NumOps.ToDouble(tv[other * p + d]);
                    if (same) exclude[b, t, 1 + j] = minusInfinity;
                }
            }
        logits = _engine.TensorAdd(logits, exclude);
        // Cross-entropy with target 0 over the masked steps, summed: logsumexp(logits) − logits₀.
        var lv = logits.ToArray();
        var shift = new Tensor<T>(new[] { batch, length, 1 });
        for (int r = 0; r < batch * length; r++)
        {
            double max = double.NegativeInfinity;
            for (int j = 0; j <= k; j++) max = Math.Max(max, NumOps.ToDouble(lv[r * (k + 1) + j]));
            shift[r / length, r % length, 0] = NumOps.FromDouble(max);
        }
        var shifted = _engine.TensorSubtract(logits, _engine.TensorTile(shift, new[] { 1, 1, k + 1 }));
        var logSumExp = _engine.TensorAdd(_engine.TensorLog(_engine.ReduceSum(_engine.TensorExp(shifted), new[] { 2 }, keepDims: false)),
            _engine.Reshape(shift, new[] { batch, length }));
        var first = _engine.Reshape(_engine.TensorSlice(logits, new[] { 0, 0, 0 }, new[] { batch, length, 1 }), new[] { batch, length });
        var contrastive = _engine.ReduceSum(_engine.TensorMultiply(_engine.TensorSubtract(logSumExp, first), weights), new[] { 0, 1 }, keepDims: false);

        int entries = c.Groups * c.EntriesPerGroup;
        var diversity = _engine.TensorMultiplyScalar(_engine.TensorAddScalar(_engine.TensorMultiplyScalar(perplexity, NumOps.FromDouble(-1.0)),
            NumOps.FromDouble(entries)), NumOps.FromDouble((double)masked / entries));
        var raw = output.RawFeatures;
        var penalty = _engine.TensorMultiplyScalar(_engine.ReduceMean(_engine.TensorMultiply(raw, raw), Enumerable.Range(0, raw.Rank).ToArray(), keepDims: false),
            NumOps.FromDouble(masked));
        var total = _engine.TensorAdd(_engine.TensorAdd(contrastive, _engine.TensorMultiplyScalar(diversity, NumOps.FromDouble(c.DiversityWeight))),
            _engine.TensorMultiplyScalar(penalty, NumOps.FromDouble(c.FeaturePenaltyWeight)));
        return new Losses(total, contrastive, diversity, penalty, masked, ids);
    }

    /// <summary>The reference's initialization: weight_proj N(0, 1) with zero bias, codevectors U(0, 1), the two
    /// projections with PyTorch's default U(±1/√in).</summary>
    public void InitializeLikeReference(Random random)
    {
        var c = Configuration;
        int entries = c.Groups * c.EntriesPerGroup, width = c.CodevectorDim / c.Groups;
        var w = new double[entries * _featureDim];
        for (int i = 0; i < w.Length; i++) w[i] = Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());
        TorchParameters.Linear(_engine, WeightProjection, _featureDim, entries, w, new double[entries]);
        foreach (var codebook in Codebooks)
        {
            var table = new double[width * c.EntriesPerGroup];
            for (int i = 0; i < table.Length; i++) table[i] = random.NextDouble();
            TorchParameters.Linear(codebook, c.EntriesPerGroup, width, table);
        }
        void Uniform(DenseLayer<T> layer, int inputs, int outputs)
        {
            double bound = 1.0 / Math.Sqrt(inputs);
            var weight = new double[outputs * inputs];
            var bias = new double[outputs];
            for (int i = 0; i < weight.Length; i++) weight[i] = (2 * random.NextDouble() - 1) * bound;
            for (int i = 0; i < bias.Length; i++) bias[i] = (2 * random.NextDouble() - 1) * bound;
            TorchParameters.Linear(_engine, layer, inputs, outputs, weight, bias);
        }
        Uniform(ProjectHidden, _hiddenSize, c.ProjectedDim);
        Uniform(ProjectQuantized, c.CodevectorDim, c.ProjectedDim);
        Temperature = c.MaxTemperature;
    }

    /// <summary>Loads the quantizer and projections of a transformers <c>Wav2Vec2ForPreTraining</c> state dict.</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        var c = Configuration;
        int entries = c.Groups * c.EntriesPerGroup, width = c.CodevectorDim / c.Groups;
        TorchParameters.Linear(_engine, WeightProjection, _featureDim, entries,
            read("quantizer.weight_proj.weight", new[] { entries, _featureDim }), read("quantizer.weight_proj.bias", new[] { entries }));
        var codevectors = read("quantizer.codevectors", new[] { 1, entries, width });
        for (int g = 0; g < c.Groups; g++)
        {
            // Codebook g as a [V → d/G] map whose weight [out, in] has codevector rows as columns.
            var weight = new double[width * c.EntriesPerGroup];
            for (int e = 0; e < c.EntriesPerGroup; e++)
                for (int d = 0; d < width; d++) weight[d * c.EntriesPerGroup + e] = codevectors[(g * c.EntriesPerGroup + e) * width + d];
            TorchParameters.Linear(Codebooks[g], c.EntriesPerGroup, width, weight);
        }
        TorchParameters.Linear(_engine, ProjectHidden, _hiddenSize, c.ProjectedDim,
            read("project_hid.weight", new[] { c.ProjectedDim, _hiddenSize }), read("project_hid.bias", new[] { c.ProjectedDim }));
        TorchParameters.Linear(_engine, ProjectQuantized, c.CodevectorDim, c.ProjectedDim,
            read("project_q.weight", new[] { c.ProjectedDim, c.CodevectorDim }), read("project_q.bias", new[] { c.ProjectedDim }));
    }
}

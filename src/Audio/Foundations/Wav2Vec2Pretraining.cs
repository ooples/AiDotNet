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
    /// <remarks><paramref name="MaskedCount"/> and <paramref name="CodeIdValues"/> stay tensors so a compiled plan can
    /// replay the objective; <see cref="Masked"/> and <see cref="CodeIds"/> read them on the host.</remarks>
    internal sealed record Losses(Tensor<T> Total, Tensor<T> Contrastive, Tensor<T> Diversity, Tensor<T> FeaturePenalty,
        Tensor<T> MaskedCount, Tensor<T> CodeIdValues, int MaxCodeId)
    {
        /// <summary><see cref="CodeIdValues"/>, for a consumer that needs each id to name its entry exactly.</summary>
        /// <exception cref="NotSupportedException">The element type cannot hold every id up to
        /// <see cref="MaxCodeId"/> exactly (binary16 stops at 2048), so a stored id could name a neighbouring entry.</exception>
        public Tensor<T> ExactCodeIdValues => RepresentsEveryId(MaxCodeId)
            ? CodeIdValues
            : throw new NotSupportedException(
                $"{typeof(T).Name} cannot represent every codebook id up to {MaxCodeId} exactly. Use a wider element "
                + "type or a smaller codebook.");

        /// <summary>The number of masked steps.</summary>
        public int Masked => (int)Math.Round(MathHelper.GetNumericOperations<T>().ToDouble(MaskedCount.ToArray()[0]));

        /// <summary>Each step's codebook entry <c>[B, L]</c>, the groups' indices combined.</summary>
        public int[,] CodeIds
        {
            [AiDotNet.Attributes.ScalarKernel("Reads the step's entries back to the host for a caller that wants them as integers.")]
            get
            {
                var ops = MathHelper.GetNumericOperations<T>();
                int batch = CodeIdValues.Shape[0], length = CodeIdValues.Shape[1];
                var data = ExactCodeIdValues.ToArray();
                var ids = new int[batch, length];
                for (int b = 0; b < batch; b++)
                    for (int t = 0; t < length; t++) ids[b, t] = (int)Math.Round(ops.ToDouble(data[b * length + t]));
                return ids;
            }
        }
    }

    /// <summary>Whether <typeparamref name="T"/> holds every integer from 0 to <paramref name="maxId"/> exactly.</summary>
    /// <remarks>A binary floating type holds every integer up to a power of two and then only even ones, so the largest
    /// id and its predecessor (one of which is odd) both surviving a round trip is equivalent to all of them doing so.</remarks>
    internal static bool RepresentsEveryId(int maxId)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        bool Exact(int value) => value < 0 || ops.ToDouble(ops.FromDouble(value)) == value;
        return Exact(maxId) && Exact(maxId - 1);
    }

    /// <summary>The largest combined id Σ_g (V − 1)·V^g of the configured codebooks.</summary>
    private int MaxCodeId()
    {
        long max = 0, place = 1;
        for (int g = 0; g < Configuration.Groups; g++)
        {
            max = checked(max + (Configuration.EntriesPerGroup - 1) * place);
            place = checked(place * Configuration.EntriesPerGroup);
        }
        return checked((int)max);
    }

    private Tensor<T> Dense(LayerBase<T> layer, Tensor<T> x)
    {
        int batch = x.Shape[0], length = x.Shape[1];
        var y = layer.Forward(_engine.Reshape(x, new[] { batch * length, x.Shape[2] }));
        return _engine.Reshape(y, new[] { batch, length, y.Shape[1] });
    }

    // The quantized features [B, L, d], the codebook perplexity (Σ_g exp(entropy of the masked marginal)) and each step's
    // entry [B, L] (the groups' indices combined as Σ_g id_g·V^g), from the [B, L] mask of 0 and 1 and its count.
    // fixedIds, when given, replaces each step's argmax with its recorded entry; the Gumbel noise is still drawn, so the
    // random stream stays aligned with an unfixed draw. In training every per-step value (the noise, the temperature)
    // is a draw a compiled plan refills before every replay, and the hard choice is taken on the engine.
    private (Tensor<T> Quantized, Tensor<T> Perplexity, Tensor<T> Ids) Quantize(Tensor<T> features, Tensor<T> maskValues, Tensor<T> maskedCount,
        bool training, Random random, int[,]? fixedIds, bool redrawPerStep)
    {
        var c = Configuration;
        int batch = features.Shape[0], length = features.Shape[1], rows = batch * length, v = c.EntriesPerGroup;
        var logits = _engine.Reshape(Dense(WeightProjection, features), new[] { rows, c.Groups * v });
        var maskWeights = _engine.Reshape(maskValues, new[] { rows, 1 });
        // 1 / max(count, 1) for a whole count.
        var perMasked = _engine.TensorAddScalar(_engine.ReLU(_engine.TensorAddScalar(maskedCount, NumOps.FromDouble(-1.0))), NumOps.One);
        var parts = new Tensor<T>[c.Groups];
        var minusOne = NumOps.FromDouble(-1.0);
        Tensor<T>? inverseTemperature = null;
        Tensor<T>? ids = null;
        int place = 1;
        Tensor<T>? perplexity = null;
        for (int g = 0; g < c.Groups; g++)
        {
            var group = _engine.TensorSlice(logits, new[] { 0, g * v }, new[] { rows, v });
            Tensor<T> probabilities, marginalSource, hard;
            if (training)
            {
                // Straight-through Gumbel-softmax (hard = true): forward one-hot, backward through the soft sample.
                var noise = StepValues(redrawPerStep, new[] { rows, v }, t => FillGumbel(t, random));
                // The temperature anneals every update: a per-step value, refilled before every replay like a draw.
                inverseTemperature ??= AiDotNet.Training.CompiledStepRandom<T>.Draw(new[] { 1, 1 }, FillInverseTemperature);
                var noisy = _engine.TensorAdd(group, noise);
                var soft = _engine.Softmax(_engine.TensorMultiply(noisy, _engine.TensorBroadcastTo(inverseTemperature, noisy._shape)), axis: 1);
                if (fixedIds is not null)
                {
                    hard = FixedChoices(fixedIds, rows, length, v, place);
                }
                else
                {
                    // The argmax as a one-hot on the engine. Rounding can make two entries share the row's maximum even
                    // when their noise differs; marking both would add two codevectors and report the sum of their ids.
                    // Keep the first: the entry at which the running count of maxima along the row is one.
                    var rowMax = _engine.ReduceMax(soft, new[] { 1 }, keepDims: true, out _);
                    var isMax = _engine.TensorEquals(soft, _engine.TensorBroadcastTo(rowMax, soft._shape));
                    var firstMax = _engine.TensorEquals(_engine.TensorCumSum(isMax, 1), NumOps.One);
                    hard = _engine.StopGradient(_engine.TensorMultiply(isMax, firstMax));
                }
                probabilities = _engine.TensorAdd(soft, _engine.TensorSubtract(hard, _engine.StopGradient(soft)));
                marginalSource = _engine.Softmax(group, axis: 1);
            }
            else
            {
                hard = fixedIds is not null ? FixedChoices(fixedIds, rows, length, v, place) : HostArgMax(group, rows, v);
                probabilities = hard;
                marginalSource = hard;
            }
            int scale = place;
            var entry = Constants(new[] { 1, v }, Enumerable.Range(0, v).Select(j => (double)j * scale).ToArray());
            var groupIds = _engine.ReduceSum(_engine.TensorMultiply(hard, _engine.TensorBroadcastTo(entry, hard._shape)), new[] { 1 }, keepDims: false);
            ids = ids is null ? groupIds : _engine.TensorAdd(ids, groupIds);
            parts[g] = Codebooks[g].Forward(probabilities);
            // The marginal over the masked steps, and exp of its entropy (xlogy: 0·log 0 = 0).
            var summed = _engine.ReduceSum(_engine.TensorMultiply(marginalSource, _engine.TensorTile(maskWeights, new[] { 1, v })), new[] { 0 }, keepDims: false);
            var marginal = _engine.TensorDivide(summed, _engine.TensorBroadcastTo(_engine.Reshape(perMasked, new[] { 1 }), summed._shape));
            // xlogy: an entry of zero probability contributes 0·log 0 = 0 (its log taken of 1 instead).
            var safe = _engine.TensorAdd(marginal, _engine.StopGradient(_engine.TensorEquals(marginal, NumOps.Zero)));
            var entropy = _engine.TensorMultiplyScalar(_engine.ReduceSum(_engine.TensorMultiply(marginal, _engine.TensorLog(safe)), new[] { 0 }, keepDims: false), minusOne);
            var term = _engine.TensorExp(entropy);
            perplexity = perplexity is null ? term : _engine.TensorAdd(perplexity, term);
            place *= v;
        }
        var quantized = _engine.Reshape(c.Groups == 1 ? parts[0] : _engine.TensorConcatenate(parts, 1), new[] { batch, length, c.CodevectorDim });
        return (quantized, perplexity ?? throw new InvalidOperationException("The quantizer has no codebooks."),
            _engine.Reshape(ids ?? throw new InvalidOperationException("The quantizer has no codebooks."), new[] { batch, length }));
    }

    // Gumbel(0, 1) samples, in the order the eager host loop drew them.
    [AiDotNet.Attributes.ScalarKernel("Fills one step's Gumbel noise from the model's generator; no traced value is read.")]
    private static void FillGumbel(Tensor<T> noise, Random random)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var span = noise.AsWritableSpan();
        for (int i = 0; i < span.Length; i++) span[i] = ops.FromDouble(-Math.Log(-Math.Log(1.0 - random.NextDouble())));
    }

    // A per-step draw a compiled plan refills before every replay; or, for a caller that draws the same values every
    // step from a freshly seeded generator (a fixed draw), the values once, as a constant of the plan.
    private static Tensor<T> StepValues(bool redrawPerStep, int[] shape, Action<Tensor<T>> fill)
    {
        if (redrawPerStep) return AiDotNet.Training.CompiledStepRandom<T>.Draw(shape, fill);
        var values = new Tensor<T>(shape);
        fill(values);
        return values;
    }

    private void FillInverseTemperature(Tensor<T> value) => value.AsWritableSpan()[0] = NumOps.FromDouble(1.0 / Temperature);

    // One group's recorded choices as a one-hot [rows, v].
    private static Tensor<T> FixedChoices(int[,] fixedIds, int rows, int length, int v, int place)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var values = new T[rows * v];
        for (int i = 0; i < values.Length; i++) values[i] = ops.Zero;
        for (int r = 0; r < rows; r++) values[r * v + fixedIds[r / length, r % length] / place % v] = ops.One;
        return new Tensor<T>(new[] { rows, v }, new Vector<T>(values));
    }

    // The inference argmax (first of equal maxima) as a one-hot [rows, v]: an eager path, read on the host.
    [AiDotNet.Attributes.ScalarKernel("The inference quantizer's argmax with the reference's first-maximum tie rule; training takes the engine path.")]
    private static Tensor<T> HostArgMax(Tensor<T> group, int rows, int v)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var data = group.ToArray();
        var values = new T[rows * v];
        for (int i = 0; i < values.Length; i++) values[i] = ops.Zero;
        for (int r = 0; r < rows; r++)
        {
            int best = 0;
            for (int j = 1; j < v; j++) if (ops.ToDouble(data[r * v + j]) > ops.ToDouble(data[r * v + best])) best = j;
            values[r * v + best] = ops.One;
        }
        return new Tensor<T>(new[] { rows, v }, new Vector<T>(values));
    }

    private static Tensor<T> Constants(int[] shape, double[] values)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return new Tensor<T>(shape, new Vector<T>(values.Select(ops.FromDouble).ToArray()));
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

    /// <summary>
    /// The distractors of <see cref="SampleNegatives"/> as a flattened <c>[B·L·K]</c> index tensor drawn per step: a
    /// compiled plan redraws them before every replay, after the mask they are drawn among (<paramref name="mask"/>,
    /// refilled in place by the forward's own draw).
    /// </summary>
    public Tensor<int> DrawNegatives(bool[,] mask, Random random)
    {
        int batch = mask.GetLength(0), length = mask.GetLength(1), k = Configuration.Negatives;
        return AiDotNet.Training.CompiledStepRandom<T>.DrawIndices(new[] { batch * length * k }, t =>
        {
            var drawn = SampleNegatives(mask, random);
            var span = t.AsWritableSpan();
            for (int b = 0; b < batch; b++)
                for (int s = 0; s < length; s++)
                    for (int j = 0; j < k; j++) span[(b * length + s) * k + j] = drawn[b, s, j];
        });
    }

    /// <summary>The pre-training loss of a forward with its applied mask and distractor indices.</summary>
    /// <param name="output">The forward, with its applied mask.</param>
    /// <param name="negatives">Each step's distractor indices.</param>
    /// <param name="training">Whether the quantizer samples (straight-through Gumbel-softmax) or takes the argmax.</param>
    /// <param name="random">The Gumbel noise's generator.</param>
    /// <param name="fixedIds">Each step's codebook entry to use in place of the quantizer's choice (the <see cref="Losses.CodeIds"/>
    /// of an earlier call), or null for the quantizer's own; gradients still flow through the soft sample.</param>
    /// <param name="redrawPerStep">Whether the Gumbel noise is a per-step draw (the default) or, for a caller that reseeds
    /// its generator to the same draw every step, a constant.</param>
    public Losses Loss(Wav2Vec2Network<T>.Output output, int[,,] negatives, bool training, Random random, int[,]? fixedIds = null,
        bool redrawPerStep = true)
    {
        int batch = negatives.GetLength(0), length = negatives.GetLength(1), k = negatives.GetLength(2);
        // The flattened overload infers K from the total count, so a wrong batch or frame axis would be re-read as a
        // different K and its indices assigned to the wrong utterances without an error.
        if (batch != output.Hidden.Shape[0] || length != output.Hidden.Shape[1])
            throw new ArgumentException(
                $"The negatives are [{batch}, {length}, {k}] but the forward has {output.Hidden.Shape[0]} utterances of "
                + $"{output.Hidden.Shape[1]} frames.", nameof(negatives));
        var indices = new int[batch * length * k];
        for (int b = 0; b < batch; b++)
            for (int t = 0; t < length; t++)
                for (int j = 0; j < k; j++) indices[(b * length + t) * k + j] = negatives[b, t, j];
        return Loss(output, new Tensor<int>(new[] { indices.Length }, new Vector<int>(indices)), training, random, fixedIds, redrawPerStep);
    }

    /// <summary>The pre-training loss of a forward with its applied mask and flattened <c>[B·L·K]</c> distractor
    /// indices (<see cref="DrawNegatives"/>); every value is computed on the engine, so a compiled plan replays it.</summary>
    public Losses Loss(Wav2Vec2Network<T>.Output output, Tensor<int> negatives, bool training, Random random, int[,]? fixedIds = null,
        bool redrawPerStep = true)
    {
        var c = Configuration;
        var mask = output.Mask ?? throw new ArgumentException("Pre-training needs the forward's time mask.");
        var maskValues = output.MaskValues ?? Wav2Vec2Network<T>.MaskTensor(mask);
        int batch = output.Hidden.Shape[0], length = output.Hidden.Shape[1], p = c.ProjectedDim;
        int steps = Math.Max(1, batch * length);
        if (negatives.Length % steps != 0)
            throw new ArgumentException(
                $"{negatives.Length} distractor indices do not divide into {batch} utterances of {length} frames.", nameof(negatives));
        int k = negatives.Length / steps;
        var maskedCount = _engine.Reshape(_engine.ReduceSum(maskValues, new[] { 0, 1 }, keepDims: false), new[] { 1 });
        var predicted = Dense(ProjectHidden, output.Hidden);                                                       // [B, L, P]
        var (quantized, perplexity, ids) = Quantize(output.Features, maskValues, maskedCount, training, random, fixedIds, redrawPerStep);
        var targets = Dense(ProjectQuantized, quantized);                                                           // [B, L, P]

        // Candidates [B, L, 1 + K, P]: the step's own target, then its distractors.
        var flat = _engine.Reshape(targets, new[] { batch * length, p });
        var distractors = _engine.Reshape(_engine.TensorGather(flat, negatives, axis: 0), new[] { batch, length, k, p });
        var candidates = _engine.TensorConcatenate(new[] { _engine.Reshape(targets, new[] { batch, length, 1, p }), distractors }, 2);

        // Cosine similarity over κ.
        Tensor<T> Norm(Tensor<T> x, int axis) => _engine.TensorSqrt(_engine.ReduceSum(_engine.TensorMultiply(x, x), new[] { axis }, keepDims: true));
        var pred = _engine.Reshape(predicted, new[] { batch, length, 1, p });
        var dots = _engine.ReduceSum(_engine.TensorMultiply(_engine.TensorTile(pred, new[] { 1, 1, k + 1, 1 }), candidates), new[] { 3 }, keepDims: false);
        var predictedNorm = _engine.TensorTile(_engine.Reshape(Norm(pred, 3), new[] { batch, length, 1 }), new[] { 1, 1, k + 1 });
        var norms = _engine.TensorMultiply(predictedNorm, _engine.Reshape(Norm(candidates, 3), new[] { batch, length, k + 1 }));
        // torch.cosine_similarity divides by max(‖a‖·‖b‖, 1e-8).
        var floor = _engine.StopGradient(_engine.ReLU(_engine.TensorAddScalar(_engine.TensorMultiplyScalar(norms, NumOps.FromDouble(-1.0)), NumOps.FromDouble(1e-8))));
        var logits = _engine.TensorMultiplyScalar(_engine.TensorDivide(dots, _engine.TensorAdd(norms, floor)),
            NumOps.FromDouble(1.0 / c.LogitTemperature));                                                         // [B, L, 1 + K]

        // A distractor identical to the target is excluded (−∞), as in the reference.
        var weights = maskValues;
        if (k > 0)
        {
            // A masked step's distractor whose target equals its own in every component: max |difference| = 0.
            var own = _engine.TensorBroadcastTo(_engine.Reshape(targets, new[] { batch, length, 1, p }), new[] { batch, length, k, p });
            var gap = _engine.ReduceMax(_engine.TensorAbs(_engine.TensorSubtract(distractors, own)), new[] { 3 }, keepDims: false, out _);
            var same = _engine.TensorMultiply(_engine.TensorEquals(gap, NumOps.Zero),
                _engine.TensorBroadcastTo(_engine.Reshape(weights, new[] { batch, length, 1 }), new[] { batch, length, k }));
            var excluded = _engine.StopGradient(_engine.TensorConcatenate(new[] { new Tensor<T>(new[] { batch, length, 1 }), same }, 2));
            var minusInfinity = new Tensor<T>(new[] { batch, length, k + 1 });
            minusInfinity.Fill(NumOps.FromDouble(double.NegativeInfinity));
            logits = _engine.TensorWhere(excluded, minusInfinity, logits);
        }
        // Cross-entropy with target 0 over the masked steps, summed: logsumexp(logits) − logits₀.
        var shift = _engine.StopGradient(_engine.ReduceMax(logits, new[] { 2 }, keepDims: true, out _));
        var shifted = _engine.TensorSubtract(logits, _engine.TensorTile(shift, new[] { 1, 1, k + 1 }));
        var logSumExp = _engine.TensorAdd(_engine.TensorLog(_engine.ReduceSum(_engine.TensorExp(shifted), new[] { 2 }, keepDims: false)),
            _engine.Reshape(shift, new[] { batch, length }));
        var first = _engine.Reshape(_engine.TensorSlice(logits, new[] { 0, 0, 0 }, new[] { batch, length, 1 }), new[] { batch, length });
        var contrastive = _engine.ReduceSum(_engine.TensorMultiply(_engine.TensorSubtract(logSumExp, first), weights), new[] { 0, 1 }, keepDims: false);

        int entries = c.Groups * c.EntriesPerGroup;
        Tensor<T> Times(Tensor<T> x, Tensor<T> scalar) => _engine.TensorMultiply(x, _engine.Reshape(scalar, x._shape));
        var diversity = Times(_engine.TensorAddScalar(_engine.TensorMultiplyScalar(perplexity, NumOps.FromDouble(-1.0)), NumOps.FromDouble(entries)),
            _engine.TensorMultiplyScalar(maskedCount, NumOps.FromDouble(1.0 / entries)));
        var raw = output.RawFeatures;
        var penalty = Times(_engine.ReduceMean(_engine.TensorMultiply(raw, raw), Enumerable.Range(0, raw.Rank).ToArray(), keepDims: false), maskedCount);
        var total = _engine.TensorAdd(_engine.TensorAdd(contrastive, _engine.TensorMultiplyScalar(diversity, NumOps.FromDouble(c.DiversityWeight))),
            _engine.TensorMultiplyScalar(penalty, NumOps.FromDouble(c.FeaturePenaltyWeight)));
        return new Losses(total, contrastive, diversity, penalty, maskedCount, ids, MaxCodeId());
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
    public void LoadTorchWeights(Func<string, int[], double[]> read) => LoadTorchWeights(AiDotNet.ModelLoading.TorchTensorSource.From(read));

    /// <summary>Loads the quantizer and projections straight from <paramref name="source"/>.</summary>
    internal void LoadTorchWeights(AiDotNet.ModelLoading.TorchTensorSource source)
    {
        var c = Configuration;
        int entries = c.Groups * c.EntriesPerGroup, width = c.CodevectorDim / c.Groups;
        TorchParameters.Linear(_engine, WeightProjection, _featureDim, entries, source, "quantizer.weight_proj.weight", "quantizer.weight_proj.bias");
        // Split per group, so read.
        var codevectors = source.Read("quantizer.codevectors", new[] { 1, entries, width });
        for (int g = 0; g < c.Groups; g++)
        {
            // Codebook g as a [V → d/G] map whose weight [out, in] has codevector rows as columns.
            var weight = new double[width * c.EntriesPerGroup];
            for (int e = 0; e < c.EntriesPerGroup; e++)
                for (int d = 0; d < width; d++) weight[d * c.EntriesPerGroup + e] = codevectors[(g * c.EntriesPerGroup + e) * width + d];
            TorchParameters.Linear(Codebooks[g], c.EntriesPerGroup, width, weight);
        }
        TorchParameters.Linear(_engine, ProjectHidden, _hiddenSize, c.ProjectedDim, source, "project_hid.weight", "project_hid.bias");
        TorchParameters.Linear(_engine, ProjectQuantized, c.CodevectorDim, c.ProjectedDim, source, "project_q.weight", "project_q.bias");
    }
}

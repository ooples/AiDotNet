using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.Training;

/// <summary>
/// One WGAN-GP critic update (Gulrajani et al. 2017): minimize <c>E[D(fake)] - E[D(real)] + lambda * GP</c>, where GP
/// penalizes the squared deviation from 1 of the critic's input-gradient norm at random real/fake interpolates.
/// </summary>
/// <remarks>
/// <para>
/// The tabular GANs (CTGAN, CopulaGAN, CausalGAN, TableGAN) each carried a copy of this step, with its own copy of the
/// gradient penalty and its own hand-written eager fallback. It lives here once and runs, in order: the GPU-resident
/// <see cref="WganGpFusedStep{T}"/> primitive when the optimizer maps to a fused kernel; otherwise one step of the
/// shared training step (<see cref="FusedTrainingStep{T}.Step"/>): the fused compiled plan on any engine, or the
/// shared eager tape step.
/// </para>
/// <para>
/// Replay safety: the critic sees ONE stacked <c>[real; fake]</c> input, and the penalty slices its real and fake
/// halves from that input inside the traced forward, so a compiled replay recomputes the penalty from the refreshed
/// batch. The interpolation weights are a per-step draw (<see cref="CompiledStepRandom{T}"/>), redrawn on every
/// replay. The copies this replaces captured the first batch's real and fake tensors and the first epsilon in the
/// penalty, so on the CPU fused path the Lipschitz term kept penalizing the first batch for the whole run.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
internal static class WassersteinCriticStep<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>Runs one critic update and returns its loss.</summary>
    /// <param name="owner">The model; keys the compiled plan and the fused optimizer state.</param>
    /// <param name="criticLayers">The critic's layers; their registered tensors are the ones updated.</param>
    /// <param name="real">The real batch (first dimension is the batch).</param>
    /// <param name="fake">The generated batch, detached from the generator, shaped like <paramref name="real"/>.</param>
    /// <param name="criticForward">The critic's training-mode forward, built from engine ops.</param>
    /// <param name="gradientPenaltyWeight">lambda, the gradient-penalty weight (10 in the paper).</param>
    /// <param name="optimizer">The critic's optimizer.</param>
    /// <param name="random">The model's generator, for the per-sample interpolation weights.</param>
    /// <param name="fusedPrimitive">The owner's GPU-resident primitive, created on first use.</param>
    public static T Step(
        object owner,
        IReadOnlyList<ILayer<T>> criticLayers,
        Tensor<T> real,
        Tensor<T> fake,
        Func<Tensor<T>, Tensor<T>> criticForward,
        double gradientPenaltyWeight,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer,
        Random random,
        ref WganGpFusedStep<T>? fusedPrimitive)
    {
        Guard.NotNull(owner);
        Guard.NotNull(criticLayers);
        Guard.NotNull(real);
        Guard.NotNull(fake);
        Guard.NotNull(criticForward);
        Guard.NotNull(optimizer);
        Guard.NotNull(random);
        if (real.Length != fake.Length || real.Shape[0] != fake.Shape[0])
        {
            throw new ArgumentException(
                $"The fake batch [{string.Join(", ", fake._shape)}] must match the real batch [{string.Join(", ", real._shape)}].",
                nameof(fake));
        }

        var engine = AiDotNetEngine.Current;
        var parameters = TapeTrainingStep<T>.CollectParameters(criticLayers);

        // GPU-resident primitive: real, fake and epsilon are persistent slots it refreshes per step (#1845).
        if (parameters.Count > 0
            && FusedTrainingSession<T>.TryMapToFusedOptimizerConfig(optimizer, out var config))
        {
            var primitive = fusedPrimitive ??= new WganGpFusedStep<T>();
            if (primitive.TryStep(
                    discParameters: parameters,
                    realBatch: real,
                    fakeBatch: fake,
                    discForward: criticForward,
                    epsilonSampler: size => engine.TensorRandomUniformRange<T>(new[] { size, 1 }, NumOps.Zero, NumOps.One),
                    gradientPenaltyWeight: gradientPenaltyWeight,
                    optimizerType: config.Type,
                    learningRate: config.LearningRate,
                    beta1: config.Beta1,
                    beta2: config.Beta2,
                    epsilon: config.Epsilon,
                    weightDecay: config.WeightDecay,
                    lrSchedule: config.Schedule,
                    extras: config.Extras,
                    out T primitiveLoss))
            {
                return primitiveLoss;
            }
        }

        int realCount = real.Shape[0];
        var stacked = engine.TensorConcatenate(new[] { real, fake }, axis: 0);
        Tensor<T>? tracedInput = null;
        Tensor<T> Forward(Tensor<T> both)
        {
            tracedInput = both;
            return criticForward(both);
        }

        Tensor<T> Loss(Tensor<T> scores, Tensor<T> _)
        {
            var both = tracedInput ?? throw new InvalidOperationException(
                "The WGAN-GP critic loss ran before its forward; the training step's contract is forward, then loss.");
            var realScores = SliceRows(engine, scores, 0, realCount);
            var fakeScores = SliceRows(engine, scores, realCount, scores.Shape[0] - realCount);
            var scoreAxes = Enumerable.Range(0, realScores.Shape.Length).ToArray();
            var wasserstein = engine.TensorSubtract(
                engine.ReduceMean(fakeScores, scoreAxes, keepDims: false),
                engine.ReduceMean(realScores, scoreAxes, keepDims: false));
            var penalty = GradientPenalty(
                engine,
                SliceRows(engine, both, 0, realCount),
                SliceRows(engine, both, realCount, both.Shape[0] - realCount),
                criticForward,
                random);
            return engine.TensorAdd(
                wasserstein, engine.TensorMultiplyScalar(penalty, NumOps.FromDouble(gradientPenaltyWeight)));
        }

        var trainableLayers = criticLayers.OfType<ITrainableLayer<T>>().ToList();
        return FusedTrainingStep<T>.Step(
            owner,
            trainableLayers,
            stacked,
            new Tensor<T>(new[] { 1 }),
            Forward,
            Loss,
            optimizer,
            extraTensors: parameters);
    }

    /// <summary>
    /// The WGAN-GP penalty <c>E[(||grad_x D(x_hat)||_2 - 1)^2]</c> at <c>x_hat = eps * real + (1 - eps) * fake</c>, one
    /// eps drawn uniformly from [0, 1) per sample. The inner backward records on the outer tape (createGraph) so the
    /// penalty's own gradient reaches the critic's weights (#1844).
    /// </summary>
    internal static Tensor<T> GradientPenalty(
        IEngine engine,
        Tensor<T> real,
        Tensor<T> fake,
        Func<Tensor<T>, Tensor<T>> criticForward,
        Random random)
    {
        int batchSize = Math.Max(1, real.Shape[0]);
        int elementsPerSample = Math.Max(1, real.Length / batchSize);

        var epsilon = CompiledStepRandom<T>.Uniform(new[] { batchSize, 1 }, random);
        var epsilonBroadcast = engine.TensorTile(epsilon, new[] { 1, elementsPerSample });
        var realRows = engine.Reshape(real, new[] { batchSize, elementsPerSample });
        var fakeRows = engine.Reshape(fake, new[] { batchSize, elementsPerSample });
        // eps * real + (1 - eps) * fake == fake + eps * (real - fake)
        var interpolatedRows = engine.TensorAdd(
            fakeRows, engine.TensorMultiply(epsilonBroadcast, engine.TensorSubtract(realRows, fakeRows)));
        var interpolated = engine.Reshape(interpolatedRows, real._shape);

        Tensor<T> inputGradient;
        using (var gradientTape = new GradientTape<T>())
        {
            var scores = criticForward(interpolated);
            var scoreAxes = Enumerable.Range(0, scores.Shape.Length).ToArray();
            var summedScores = engine.ReduceSum(scores, scoreAxes, keepDims: false);
            var gradients = gradientTape.ComputeGradients(summedScores, new[] { interpolated }, createGraph: true);
            inputGradient = gradients.TryGetValue(interpolated, out var gradient)
                ? gradient
                : new Tensor<T>(interpolated._shape);
        }

        var gradientRows = engine.Reshape(inputGradient, new[] { batchSize, elementsPerSample });
        var normSquared = engine.ReduceSum(engine.TensorMultiply(gradientRows, gradientRows), new[] { 1 }, keepDims: false);
        var norm = engine.TensorSqrt(engine.TensorAddScalar(normSquared, NumOps.FromDouble(1e-12)));
        var deviation = engine.TensorAddScalar(norm, NumOps.Negate(NumOps.One));
        var penalty = engine.TensorMultiply(deviation, deviation);
        var penaltyAxes = Enumerable.Range(0, penalty.Shape.Length).ToArray();
        return engine.ReduceMean(penalty, penaltyAxes, keepDims: false);
    }

    private static Tensor<T> SliceRows(IEngine engine, Tensor<T> tensor, int start, int count)
    {
        var begin = new int[tensor.Rank];
        begin[0] = start;
        var shape = tensor._shape.ToArray();
        shape[0] = count;
        return engine.TensorSlice(tensor, begin, shape);
    }
}
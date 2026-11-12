using AiDotNet.AutoML.NAS;
using AiDotNet.AutoML.SearchSpace;

namespace AiDotNet.AutoML;

/// <summary>
/// Neural Architecture Search implementation with gradient-based (DARTS) support
/// </summary>
/// <typeparam name="T">The numeric type for calculations</typeparam>
public class NeuralArchitectureSearch<T>
{
    private readonly INumericOperations<T> _ops;
    private readonly NeuralArchitectureSearchStrategy _strategy;
    private readonly SearchSpaceBase<T> _searchSpace;
    private readonly int _maxEpochs;
    private readonly Random _random;
    private readonly DartsOptimizerOptions _optimizerOptions;

    public AutoMLStatus Status { get; private set; } = AutoMLStatus.NotStarted;
    public Architecture<T>? BestArchitecture { get; private set; }
    public T BestScore { get; private set; }

    /// <summary>
    /// Creates a search whose DARTS optimizers use <paramref name="optimizerOptions"/> (the defaults are those of
    /// Liu et al. 2019, Sec. 5.1).
    /// </summary>
    /// <param name="strategy">The search strategy.</param>
    /// <param name="maxEpochs">The number of search epochs.</param>
    /// <param name="optimizerOptions">The architecture (Adam) and weight (SGD) optimizer settings; null uses
    /// the paper defaults.</param>
    public NeuralArchitectureSearch(
        NeuralArchitectureSearchStrategy strategy = NeuralArchitectureSearchStrategy.GradientBased,
        int maxEpochs = 50,
        DartsOptimizerOptions? optimizerOptions = null)
    {
        optimizerOptions ??= new DartsOptimizerOptions();

        if (optimizerOptions.ArchitectureLearningRate <= 0.0 || optimizerOptions.WeightMinLearningRate < 0.0
            || optimizerOptions.WeightMaxLearningRate < optimizerOptions.WeightMinLearningRate)
        {
            throw new ArgumentOutOfRangeException(nameof(optimizerOptions),
                "Learning rates must satisfy architecture > 0 and 0 <= weight min <= weight max.");
        }

        _ops = MathHelper.GetNumericOperations<T>();
        _strategy = strategy;
        _searchSpace = new SearchSpaceBase<T>();
        _maxEpochs = maxEpochs;
        _random = RandomHelper.CreateSecureRandom();
        _optimizerOptions = optimizerOptions;
        BestScore = _ops.Zero;
    }

    /// <summary>The DARTS optimizer settings.</summary>
    public DartsOptimizerOptions OptimizerOptions => _optimizerOptions;

    /// <summary>
    /// Runs the neural architecture search
    /// </summary>
    public async Task<Architecture<T>> SearchAsync(
        Tensor<T> trainData,
        Tensor<T> trainLabels,
        Tensor<T> valData,
        Tensor<T> valLabels,
        CancellationToken cancellationToken = default)
    {
        Status = AutoMLStatus.Running;

        try
        {
            Architecture<T>? result = null;

            switch (_strategy)
            {
                case NeuralArchitectureSearchStrategy.GradientBased:
                    result = await Task.Run(() => RunGradientBasedSearch(trainData, trainLabels, valData, valLabels), cancellationToken);
                    break;

                case NeuralArchitectureSearchStrategy.RandomSearch:
                    result = await Task.Run(() => RunRandomSearch(trainData, trainLabels, valData, valLabels), cancellationToken);
                    break;

                default:
                    throw new NotSupportedException($"Strategy {_strategy} is not yet implemented.");
            }

            BestArchitecture = result;
            Status = AutoMLStatus.Completed;
            return result ?? new Architecture<T>();
        }
        catch (OperationCanceledException)
        {
            Status = AutoMLStatus.Cancelled;
            throw;
        }
        catch (Exception)
        {
            Status = AutoMLStatus.Failed;
            throw;
        }
    }

    /// <summary>
    /// Runs gradient-based search using DARTS algorithm
    /// </summary>
    private Architecture<T> RunGradientBasedSearch(
        Tensor<T> trainData,
        Tensor<T> trainLabels,
        Tensor<T> valData,
        Tensor<T> valLabels)
    {
        Console.WriteLine("Starting gradient-based neural architecture search (DARTS)...");

        // Create SuperNet with differentiable architecture
        var supernet = new SuperNet<T>(_searchSpace, numNodes: 4);

        // DARTS (Liu et al. 2019, Sec. 5.1): alpha is trained with Adam (lr 3e-4, betas (0.5, 0.999), weight decay
        // 1e-3); the network weights with SGD (momentum 0.9, weight decay 3e-4, gradient-norm clipping at 5) and a
        // learning rate cosine-annealed from 0.025 to 0.001.
        var settings = _optimizerOptions;
        T architectureLR = _ops.FromDouble(settings.ArchitectureLearningRate);
        T beta1 = _ops.FromDouble(settings.ArchitectureBeta1);
        T beta2 = _ops.FromDouble(settings.ArchitectureBeta2);
        T epsilon = _ops.FromDouble(settings.ArchitectureEpsilon);
        T architectureWeightDecay = _ops.FromDouble(settings.ArchitectureWeightDecay);
        T momentum = _ops.FromDouble(settings.WeightMomentum);
        T weightDecay = _ops.FromDouble(settings.WeightDecay);
        T gradientClipNorm = _ops.FromDouble(settings.GradientClipNorm);
        T[] weightLearningRates = CosineAnnealedLearningRates(_maxEpochs,
            _ops.FromDouble(settings.WeightMaxLearningRate), _ops.FromDouble(settings.WeightMinLearningRate));

        // Initialize Adam momentum buffers for architecture parameters
        var archMomentum = new List<Matrix<T>>();
        var archVelocity = new List<Matrix<T>>();
        foreach (var alpha in supernet.GetArchitectureParameters())
        {
            archMomentum.Add(new Matrix<T>(alpha.Rows, alpha.Columns));
            archVelocity.Add(new Matrix<T>(alpha.Rows, alpha.Columns));
        }

        // Initialize SGD momentum buffers for weights (will be populated dynamically)
        var weightMomentum = new Dictionary<string, Vector<T>>();

        // Do an initial forward pass to initialize weight parameters.
        // Only ONE sample is needed to materialise lazy weight shapes;
        // pushing the full trainData tensor through was the NAS sibling
        // of #1296 and OOM-prone on Transformer-class supernet cells.
        var warmupSample = trainData.Rank >= 1 && trainData.Shape[0] > 1
            ? trainData.Slice(axis: 0, start: 0, end: 1).Contiguous()
            : trainData;
        supernet.Predict(warmupSample);

        // Now initialize momentum buffers
        foreach (var kvp in supernet.GetWeightParameters())
        {
            weightMomentum[kvp.Key] = new Vector<T>(kvp.Value.Length);
        }

        int t = 0; // Time step for Adam

        // Alternating optimization loop
        for (int epoch = 0; epoch < _maxEpochs; epoch++)
        {
            t++;

            // Phase 1: Update architecture parameters on validation set
            supernet.BackwardArchitecture(valData, valLabels);
            var archParams = supernet.GetArchitectureParameters();
            var archGrads = supernet.GetArchitectureGradients();

            for (int i = 0; i < archParams.Count; i++)
            {
                AdamStep(archParams[i], archGrads[i], archMomentum[i], archVelocity[i],
                    architectureLR, beta1, beta2, epsilon, architectureWeightDecay, t);
            }

            // Phase 2: Update network weights on training set
            supernet.BackwardWeights(trainData, trainLabels, supernet.DefaultLossFunction);
            var weightParams = supernet.GetWeightParameters();
            var weightGrads = supernet.GetWeightGradients();
            var weightKeys = weightParams.Keys.ToList();

            T weightsLR = weightLearningRates[epoch];
            T clipScale = GradientClipScale(weightKeys.Select(key => weightGrads[key]), gradientClipNorm);

            foreach (var key in weightKeys)
            {
                // Initialize the momentum buffer if this is a new weight
                if (!weightMomentum.ContainsKey(key))
                {
                    weightMomentum[key] = new Vector<T>(weightParams[key].Length);
                }

                SgdMomentumStep(weightParams[key], weightGrads[key], weightMomentum[key],
                    weightsLR, momentum, weightDecay, clipScale);
            }

            // Compute losses for logging
            var trainLoss = supernet.ComputeTrainingLoss(trainData, trainLabels);
            var valLoss = supernet.ComputeValidationLoss(valData, valLabels);

            if (epoch % 10 == 0)
            {
                Console.WriteLine($"Epoch {epoch}/{_maxEpochs} - Train Loss: {Convert.ToDouble(trainLoss):F4}, Val Loss: {Convert.ToDouble(valLoss):F4}");
            }

            // Update best score
            T currentScore = _ops.Divide(_ops.One, _ops.Add(_ops.One, valLoss)); // Convert loss to score
            if (_ops.GreaterThan(currentScore, BestScore))
            {
                BestScore = currentScore;
            }
        }

        // Derive final discrete architecture from continuous parameters
        var finalArchitecture = supernet.DeriveArchitecture();
        Console.WriteLine("\nDerived Architecture:");
        Console.WriteLine(finalArchitecture.GetDescription());

        return finalArchitecture;
    }

    /// <summary>
    /// Cosine annealing without restarts for epochs 0..epochs (inclusive):
    /// lr_e = min + (max - min) * (1 + cos(pi * e / epochs)) / 2.
    /// </summary>
    internal static T[] CosineAnnealedLearningRates(int epochs, T maxLearningRate, T minLearningRate)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;
        var epochIndex = engine.TensorArange<T>(0, epochs + 1, 1);
        var angle = engine.TensorMultiplyScalar(epochIndex, ops.FromDouble(epochs <= 0 ? 0.0 : Math.PI / epochs));
        var halfCosine = engine.TensorMultiplyScalar(engine.TensorAddScalar(engine.TensorCos(angle), ops.One), ops.FromDouble(0.5));
        return engine.TensorAddScalar(
            engine.TensorMultiplyScalar(halfCosine, ops.Subtract(maxLearningRate, minLearningRate)), minLearningRate).ToArray();
    }

    /// <summary>
    /// The factor that rescales a set of gradients to a global L2 norm of at most <paramref name="maxNorm"/>
    /// (torch.nn.utils.clip_grad_norm_: maxNorm / (norm + 1e-6) when that is below 1, otherwise 1).
    /// </summary>
    internal static T GradientClipScale(IEnumerable<Vector<T>> gradients, T maxNorm)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;
        var squaredNorms = new Vector<T>(gradients.Select(g => engine.TensorSumOfSquares(g.AsTensor())));
        T norm = ops.Sqrt(squaredNorms.Length == 0 ? ops.Zero : engine.Sum(squaredNorms));
        T scale = ops.Divide(maxNorm, ops.Add(norm, ops.FromDouble(1e-6)));
        return ops.LessThan(scale, ops.One) ? scale : ops.One;
    }

    /// <summary>
    /// One in-place Adam step with L2 weight decay (PyTorch torch.optim.Adam): g' = g + wd * theta,
    /// m = b1 m + (1 - b1) g', v = b2 v + (1 - b2) g'^2, theta -= lr * m_hat / (sqrt(v_hat) + eps).
    /// </summary>
    internal static void AdamStep(Matrix<T> parameters, Matrix<T> gradients, Matrix<T> momentum, Matrix<T> velocity,
        T learningRate, T beta1, T beta2, T epsilon, T weightDecay, int t)
    {
        var p = Vector<T>.FromSpan(parameters.AsSpan());
        var m = Vector<T>.FromSpan(momentum.AsSpan());
        var v = Vector<T>.FromSpan(velocity.AsSpan());
        AdamStep(p, Vector<T>.FromSpan(gradients.AsSpan()), m, v, learningRate, beta1, beta2, epsilon, weightDecay, t);
        p.AsSpan().CopyTo(parameters.AsWritableSpan());
        m.AsSpan().CopyTo(momentum.AsWritableSpan());
        v.AsSpan().CopyTo(velocity.AsWritableSpan());
    }

    /// <inheritdoc cref="AdamStep(Matrix{T}, Matrix{T}, Matrix{T}, Matrix{T}, T, T, T, T, T, int)"/>
    internal static void AdamStep(Vector<T> parameters, Vector<T> gradients, Vector<T> momentum, Vector<T> velocity,
        T learningRate, T beta1, T beta2, T epsilon, T weightDecay, int t)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;
        var theta = parameters.AsTensor();
        var g = engine.TensorAdd(gradients.AsTensor(), engine.TensorMultiplyScalar(theta, weightDecay));

        var m = engine.TensorAdd(engine.TensorMultiplyScalar(momentum.AsTensor(), beta1),
            engine.TensorMultiplyScalar(g, ops.Subtract(ops.One, beta1)));
        var v = engine.TensorAdd(engine.TensorMultiplyScalar(velocity.AsTensor(), beta2),
            engine.TensorMultiplyScalar(engine.TensorMultiply(g, g), ops.Subtract(ops.One, beta2)));

        T biasCorrection1 = ops.Subtract(ops.One, ops.Power(beta1, ops.FromDouble(t)));
        T biasCorrection2 = ops.Subtract(ops.One, ops.Power(beta2, ops.FromDouble(t)));
        var mHat = engine.TensorDivideScalar(m, biasCorrection1);
        var vHat = engine.TensorDivideScalar(v, biasCorrection2);
        var update = engine.TensorDivide(engine.TensorMultiplyScalar(mHat, learningRate),
            engine.TensorAddScalar(engine.TensorSqrt(vHat), epsilon));

        m.ToVector().AsSpan().CopyTo(momentum.AsWritableSpan());
        v.ToVector().AsSpan().CopyTo(velocity.AsWritableSpan());
        engine.TensorSubtract(theta, update).ToVector().AsSpan().CopyTo(parameters.AsWritableSpan());
    }

    /// <summary>
    /// One in-place SGD step with momentum and L2 weight decay (PyTorch torch.optim.SGD, dampening 0):
    /// g' = clip * g + wd * theta, buf = momentum * buf + g', theta -= lr * buf.
    /// </summary>
    internal static void SgdMomentumStep(Vector<T> parameters, Vector<T> gradients, Vector<T> buffer,
        T learningRate, T momentum, T weightDecay, T gradientScale)
    {
        var engine = AiDotNetEngine.Current;
        var theta = parameters.AsTensor();
        var g = engine.TensorAdd(engine.TensorMultiplyScalar(gradients.AsTensor(), gradientScale),
            engine.TensorMultiplyScalar(theta, weightDecay));
        var buf = engine.TensorAdd(engine.TensorMultiplyScalar(buffer.AsTensor(), momentum), g);

        buf.ToVector().AsSpan().CopyTo(buffer.AsWritableSpan());
        engine.TensorSubtract(theta, engine.TensorMultiplyScalar(buf, learningRate)).ToVector().AsSpan()
            .CopyTo(parameters.AsWritableSpan());
    }

    /// <summary>
    /// Random search with weight sharing (Li &amp; Talwalkar 2019, "Random Search and Reproducibility for Neural
    /// Architecture Search", Sec. 3): the shared weights of one supernet are trained with a uniformly sampled
    /// architecture per step, then random architectures are each applied to the supernet and scored by their
    /// validation loss with the shared weights; the lowest-loss architecture is returned.
    /// </summary>
    private Architecture<T> RunRandomSearch(
        Tensor<T> trainData,
        Tensor<T> trainLabels,
        Tensor<T> valData,
        Tensor<T> valLabels)
    {
        Console.WriteLine("Starting random architecture search (weight sharing)...");

        var operations = _searchSpace.Operations;
        var supernet = new SuperNet<T>(_searchSpace, numNodes: RandomSearchNodes);

        // One sample materialises lazy weight shapes (see RunGradientBasedSearch).
        var warmupSample = trainData.Rank >= 1 && trainData.Shape[0] > 1
            ? trainData.Slice(axis: 0, start: 0, end: 1).Contiguous()
            : trainData;
        supernet.Predict(warmupSample);

        // Shared-weight training: the DARTS weight optimizer (SGD, momentum 0.9, weight decay 3e-4, clipping at
        // 5, cosine learning rate), with a fresh uniformly sampled architecture each step.
        var settings = _optimizerOptions;
        T momentum = _ops.FromDouble(settings.WeightMomentum);
        T weightDecay = _ops.FromDouble(settings.WeightDecay);
        T gradientClipNorm = _ops.FromDouble(settings.GradientClipNorm);
        T[] weightLearningRates = CosineAnnealedLearningRates(_maxEpochs,
            _ops.FromDouble(settings.WeightMaxLearningRate), _ops.FromDouble(settings.WeightMinLearningRate));
        var weightMomentum = new Dictionary<string, Vector<T>>();

        for (int epoch = 0; epoch < _maxEpochs; epoch++)
        {
            ApplyEdgeArchitecture(supernet.GetArchitectureParameters(), operations,
                SampleEdgeArchitecture(_random, RandomSearchNodes, operations));

            supernet.BackwardWeights(trainData, trainLabels, supernet.DefaultLossFunction);
            var weightParams = supernet.GetWeightParameters();
            var weightGrads = supernet.GetWeightGradients();
            var weightKeys = weightParams.Keys.ToList();
            T clipScale = GradientClipScale(weightKeys.Select(key => weightGrads[key]), gradientClipNorm);

            foreach (var key in weightKeys)
            {
                if (!weightMomentum.ContainsKey(key))
                {
                    weightMomentum[key] = new Vector<T>(weightParams[key].Length);
                }

                SgdMomentumStep(weightParams[key], weightGrads[key], weightMomentum[key],
                    weightLearningRates[epoch], momentum, weightDecay, clipScale);
            }
        }

        // Shared-weight evaluation of random candidates.
        var candidates = Enumerable.Range(0, RandomSearchCandidates)
            .Select(_ => SampleEdgeArchitecture(_random, RandomSearchNodes, operations))
            .ToList();
        var bestArch = SelectByValidationLoss(supernet, operations, candidates, valData, valLabels, out T bestLoss);
        if (bestArch is null)
        {
            Console.WriteLine("\nNo architecture produced a finite validation loss.");
            return new Architecture<T>();
        }

        BestScore = _ops.Divide(_ops.One, _ops.Add(_ops.One, bestLoss));
        Console.WriteLine($"\nBest architecture found with loss: {Convert.ToDouble(bestLoss):F4}");
        return bestArch;
    }

    private const int RandomSearchNodes = 4;
    private const int RandomSearchCandidates = 20;
    private const double OneHotLogit = 10.0;

    /// <summary>
    /// A uniformly random architecture over the supernet's edges: every edge (node j, input i &lt;= j), with node
    /// indices as in <see cref="SuperNet{T}.DeriveArchitecture"/>, gets one operation drawn uniformly.
    /// </summary>
    internal static Architecture<T> SampleEdgeArchitecture(Random random, int numNodes, IReadOnlyList<string> operations)
    {
        var architecture = new Architecture<T>();
        foreach (int node in Enumerable.Range(0, numNodes))
        {
            foreach (int from in Enumerable.Range(0, node + 1))
            {
                architecture.AddOperation(node, from, operations[random.Next(operations.Count)]);
            }
        }

        return architecture;
    }

    /// <summary>
    /// Writes an architecture into the supernet's alphas: each listed edge becomes a one-hot logit row (+10 at
    /// its operation, -10 elsewhere, so softmax puts all but ~e^-20 of the mass on it); unlisted edges are reset
    /// to zero logits (the uniform mixture).
    /// </summary>
    internal static void ApplyEdgeArchitecture(List<Matrix<T>> alphas, IReadOnlyList<string> operations, Architecture<T> architecture)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;
        T low = ops.FromDouble(-OneHotLogit);
        T span = ops.FromDouble(2.0 * OneHotLogit);
        foreach (var alpha in alphas)
        {
            alpha.AsWritableSpan().Fill(ops.Zero);
        }

        foreach (var (toNode, fromNode, operation) in architecture.Operations)
        {
            int opIdx = IndexOfOperation(operations, operation);
            if (toNode < 0 || toNode >= alphas.Count || opIdx < 0 || opIdx >= alphas[toNode].Columns
                || fromNode < 0 || fromNode >= alphas[toNode].Rows)
            {
                continue;
            }

            int columns = alphas[toNode].Columns;
            var hot = new Tensor<int>(new[] { 1 });
            hot.Fill(opIdx);
            var mask = engine.TensorOneHot<T>(hot, columns).Reshape(new[] { columns });
            var row = engine.TensorAddScalar(engine.TensorMultiplyScalar(mask, span), low);
            alphas[toNode].SetRow(fromNode, row.ToVector());
        }
    }

    private static int IndexOfOperation(IReadOnlyList<string> operations, string operation)
    {
        return Enumerable.Range(0, operations.Count)
            .Where(i => string.Equals(operations[i], operation, StringComparison.Ordinal))
            .DefaultIfEmpty(-1)
            .First();
    }

    /// <summary>
    /// Applies each candidate to the supernet and scores it by validation loss with the shared weights. Returns
    /// the first candidate with the lowest loss (a strict less-than scan), or null when no loss is finite; the
    /// supernet is left holding the returned architecture.
    /// </summary>
    internal static Architecture<T>? SelectByValidationLoss(SuperNet<T> supernet, IReadOnlyList<string> operations,
        IList<Architecture<T>> candidates, Tensor<T> valData, Tensor<T> valLabels, out T bestLoss)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        bestLoss = ops.FromDouble(double.MaxValue);
        if (candidates.Count == 0)
        {
            return null;
        }

        var alphas = supernet.GetArchitectureParameters();
        var losses = new Vector<T>(candidates
            .Select(candidate =>
            {
                ApplyEdgeArchitecture(alphas, operations, candidate);
                return supernet.ComputeValidationLoss(valData, valLabels);
            })
            .ToArray());

        // Lowest loss = highest negated loss. NaN and infinite losses map to -inf so a finite loss always wins.
        var engine = AiDotNetEngine.Current;
        var negated = engine.TensorNanToNum(engine.TensorNegate(losses.AsTensor()),
            double.NegativeInfinity, double.NegativeInfinity, double.NegativeInfinity);
        int best = NasSamplingHelper.ArgMaxRows(negated.Reshape(new[] { 1, candidates.Count }), out T[] bestNegated)[0];
        if (!ops.GreaterThan(bestNegated[0], ops.FromDouble(double.NegativeInfinity)))
        {
            return null;
        }

        bestLoss = losses[best];
        ApplyEdgeArchitecture(alphas, operations, candidates[best]);
        return candidates[best];
    }
}

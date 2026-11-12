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

    public AutoMLStatus Status { get; private set; } = AutoMLStatus.NotStarted;
    public Architecture<T>? BestArchitecture { get; private set; }
    public T BestScore { get; private set; }

    public NeuralArchitectureSearch(
        NeuralArchitectureSearchStrategy strategy = NeuralArchitectureSearchStrategy.GradientBased,
        int maxEpochs = 50)
    {
        _ops = MathHelper.GetNumericOperations<T>();
        _strategy = strategy;
        _searchSpace = new SearchSpaceBase<T>();
        _maxEpochs = maxEpochs;
        _random = RandomHelper.CreateSecureRandom();
        BestScore = _ops.Zero;
    }

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
        T architectureLR = _ops.FromDouble(DartsArchitectureLearningRate);
        T beta1 = _ops.FromDouble(DartsArchitectureBeta1);
        T beta2 = _ops.FromDouble(DartsArchitectureBeta2);
        T epsilon = _ops.FromDouble(1e-8);
        T architectureWeightDecay = _ops.FromDouble(DartsArchitectureWeightDecay);
        T momentum = _ops.FromDouble(DartsWeightMomentum);
        T weightDecay = _ops.FromDouble(DartsWeightDecay);

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

            T weightsLR = _ops.FromDouble(CosineAnnealedLearningRate(epoch, _maxEpochs, DartsWeightMaxLearningRate, DartsWeightMinLearningRate));
            T clipScale = GradientClipScale(weightKeys.Select(key => weightGrads[key]), _ops.FromDouble(DartsGradientClipNorm));

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

    private const double DartsArchitectureLearningRate = 3e-4;
    private const double DartsArchitectureBeta1 = 0.5;
    private const double DartsArchitectureBeta2 = 0.999;
    private const double DartsArchitectureWeightDecay = 1e-3;
    private const double DartsWeightMaxLearningRate = 0.025;
    private const double DartsWeightMinLearningRate = 0.001;
    private const double DartsWeightMomentum = 0.9;
    private const double DartsWeightDecay = 3e-4;
    private const double DartsGradientClipNorm = 5.0;

    /// <summary>
    /// Cosine annealing without restarts: lr = min + (max - min) * (1 + cos(pi * epoch / epochs)) / 2.
    /// </summary>
    internal static double CosineAnnealedLearningRate(int epoch, int epochs, double maxLearningRate, double minLearningRate)
    {
        double progress = epochs <= 0 ? 0.0 : (double)epoch / epochs;
        return minLearningRate + (maxLearningRate - minLearningRate) * 0.5 * (1.0 + Math.Cos(Math.PI * progress));
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
    /// Runs random search as a baseline
    /// </summary>
    private Architecture<T> RunRandomSearch(
        Tensor<T> trainData,
        Tensor<T> trainLabels,
        Tensor<T> valData,
        Tensor<T> valLabels)
    {
        Console.WriteLine("Starting random architecture search...");

        var bestArch = new Architecture<T>();
        T bestLoss = _ops.FromDouble(double.MaxValue);

        for (int trial = 0; trial < 20; trial++)
        {
            var arch = GenerateRandomArchitecture();
            var supernet = new SuperNet<T>(_searchSpace);

            // Quick evaluation
            var loss = supernet.ComputeValidationLoss(valData, valLabels);

            if (_ops.LessThan(loss, bestLoss))
            {
                bestLoss = loss;
                bestArch = arch;
                BestScore = _ops.Divide(_ops.One, _ops.Add(_ops.One, loss));
            }

            Console.WriteLine($"Trial {trial + 1}/20 - Loss: {Convert.ToDouble(loss):F4}");
        }

        Console.WriteLine($"\nBest architecture found with loss: {Convert.ToDouble(bestLoss):F4}");
        return bestArch;
    }

    private Architecture<T> GenerateRandomArchitecture()
    {
        var arch = new Architecture<T>();
        int numNodes = _random.Next(2, 6);

        for (int i = 1; i < numNodes; i++)
        {
            for (int j = 0; j < i; j++)
            {
                if (_random.NextDouble() > 0.5)
                {
                    var opIdx = _random.Next(_searchSpace.Operations.Count);
                    arch.AddOperation(i, j, _searchSpace.Operations[opIdx]);
                }
            }
        }

        return arch;
    }
}

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

        // Learning rates
        T architectureLR = _ops.FromDouble(0.003);
        T weightsLR = _ops.FromDouble(0.025);

        // Momentum parameters
        T momentum = _ops.FromDouble(0.9);

        // Adam optimizer parameters
        T beta1 = _ops.FromDouble(0.9);
        T beta2 = _ops.FromDouble(0.999);
        T epsilon = _ops.FromDouble(1e-8);

        // Initialize Adam momentum buffers for architecture parameters
        var archMomentum = new List<Matrix<T>>();
        var archVelocity = new List<Matrix<T>>();
        foreach (var alpha in supernet.GetArchitectureParameters())
        {
            archMomentum.Add(new Matrix<T>(alpha.Rows, alpha.Columns));
            archVelocity.Add(new Matrix<T>(alpha.Rows, alpha.Columns));
        }

        // Initialize momentum buffers for weights (will be populated dynamically)
        var weightMomentum = new Dictionary<string, Vector<T>>();
        var weightVelocity = new Dictionary<string, Vector<T>>();

        // Do an initial forward pass to initialize weight parameters
        supernet.Predict(trainData);

        // Now initialize momentum buffers
        foreach (var kvp in supernet.GetWeightParameters())
        {
            weightMomentum[kvp.Key] = new Vector<T>(kvp.Value.Length);
            weightVelocity[kvp.Key] = new Vector<T>(kvp.Value.Length);
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
                UpdateParametersAdam(archParams[i], archGrads[i], archMomentum[i], archVelocity[i],
                    architectureLR, beta1, beta2, epsilon, t);
            }

            // Phase 2: Update network weights on training set
            supernet.BackwardWeights(trainData, trainLabels, supernet.DefaultLossFunction);
            var weightParams = supernet.GetWeightParameters();
            var weightGrads = supernet.GetWeightGradients();

            foreach (var key in weightParams.Keys.ToList())
            {
                // Initialize momentum/velocity if this is a new weight
                if (!weightMomentum.ContainsKey(key))
                {
                    weightMomentum[key] = new Vector<T>(weightParams[key].Length);
                    weightVelocity[key] = new Vector<T>(weightParams[key].Length);
                }

                UpdateParametersAdam(weightParams[key], weightGrads[key], weightMomentum[key], weightVelocity[key],
                    weightsLR, beta1, beta2, epsilon, t);
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
    /// Updates parameters using Adam optimizer
    /// </summary>
    private void UpdateParametersAdam(
        object parameters,
        object gradients,
        object momentum,
        object velocity,
        T learningRate,
        T beta1,
        T beta2,
        T epsilon,
        int t)
    {
        if (parameters is Matrix<T> paramMatrix && gradients is Matrix<T> gradMatrix &&
            momentum is Matrix<T> momMatrix && velocity is Matrix<T> velMatrix)
        {
            for (int i = 0; i < paramMatrix.Rows; i++)
            {
                for (int j = 0; j < paramMatrix.Columns; j++)
                {
                    // m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
                    momMatrix[i, j] = _ops.Add(
                        _ops.Multiply(beta1, momMatrix[i, j]),
                        _ops.Multiply(_ops.Subtract(_ops.One, beta1), gradMatrix[i, j])
                    );

                    // v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
                    velMatrix[i, j] = _ops.Add(
                        _ops.Multiply(beta2, velMatrix[i, j]),
                        _ops.Multiply(_ops.Subtract(_ops.One, beta2), _ops.Multiply(gradMatrix[i, j], gradMatrix[i, j]))
                    );

                    // Bias correction
                    T mHat = _ops.Divide(momMatrix[i, j], _ops.Subtract(_ops.One, _ops.Power(beta1, _ops.FromDouble(t))));
                    T vHat = _ops.Divide(velMatrix[i, j], _ops.Subtract(_ops.One, _ops.Power(beta2, _ops.FromDouble(t))));

                    // Update: θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
                    T update = _ops.Divide(_ops.Multiply(learningRate, mHat), _ops.Add(_ops.Sqrt(vHat), epsilon));
                    paramMatrix[i, j] = _ops.Subtract(paramMatrix[i, j], update);
                }
            }
        }
        else if (parameters is Vector<T> paramVector && gradients is Vector<T> gradVector &&
                 momentum is Vector<T> momVector && velocity is Vector<T> velVector)
        {
            for (int i = 0; i < paramVector.Length; i++)
            {
                // m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
                momVector[i] = _ops.Add(
                    _ops.Multiply(beta1, momVector[i]),
                    _ops.Multiply(_ops.Subtract(_ops.One, beta1), gradVector[i])
                );

                // v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
                velVector[i] = _ops.Add(
                    _ops.Multiply(beta2, velVector[i]),
                    _ops.Multiply(_ops.Subtract(_ops.One, beta2), _ops.Multiply(gradVector[i], gradVector[i]))
                );

                // Bias correction
                T mHat = _ops.Divide(momVector[i], _ops.Subtract(_ops.One, _ops.Power(beta1, _ops.FromDouble(t))));
                T vHat = _ops.Divide(velVector[i], _ops.Subtract(_ops.One, _ops.Power(beta2, _ops.FromDouble(t))));

                // Update
                T update = _ops.Divide(_ops.Multiply(learningRate, mHat), _ops.Add(_ops.Sqrt(vHat), epsilon));
                paramVector[i] = _ops.Subtract(paramVector[i], update);
            }
        }
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

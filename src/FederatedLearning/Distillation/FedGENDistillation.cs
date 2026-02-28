namespace AiDotNet.FederatedLearning.Distillation;

/// <summary>
/// FedGEN — Data-free federated distillation using a lightweight generator on the server.
/// </summary>
/// <remarks>
/// <para>
/// FedGEN (Zhu et al., 2021) eliminates the need for a public dataset by training a
/// small generator on the server that produces synthetic samples. Clients share class-conditional
/// statistics (means and variances) rather than logits. The server's generator learns to
/// produce samples that match the consensus statistics, then distills this knowledge back.
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of needing a shared test dataset (like FedMD), the server
/// creates its own fake data that captures what all clients know. Clients only share statistical
/// summaries (like "class A has these average features"), which is more privacy-preserving
/// than sharing predictions on real data.
/// </para>
/// <para>
/// Reference: Zhu et al. (2021), "Data-Free Knowledge Distillation for Heterogeneous Federated Learning".
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FedGENDistillation<T> : Infrastructure.FederatedLearningComponentBase<T>, IFederatedDistillationStrategy<T>
{
    private readonly int _numClasses;
    private readonly int _featureDim;
    private readonly int _generatorEpochs;
    private readonly double _generatorLearningRate;
    private readonly int _seed;
    private int _roundCounter;
    private Random _rng;

    // Server-side generator parameters: simple linear generator per class
    // generatorWeights[classIdx] = feature vector for that class centroid
    private T[][]? _generatorWeights;

    /// <summary>
    /// Creates a new FedGEN distillation strategy.
    /// </summary>
    /// <param name="numClasses">Number of output classes.</param>
    /// <param name="featureDim">Feature dimensionality for generated samples.</param>
    /// <param name="generatorEpochs">Generator training epochs per round. Default: 5.</param>
    /// <param name="generatorLearningRate">Generator learning rate. Default: 0.01.</param>
    /// <param name="seed">Random seed. Default: 42.</param>
    public FedGENDistillation(int numClasses = 10, int featureDim = 64,
        int generatorEpochs = 5, double generatorLearningRate = 0.01, int seed = 42)
    {
        _numClasses = numClasses;
        _featureDim = featureDim;
        _generatorEpochs = generatorEpochs;
        _generatorLearningRate = generatorLearningRate;
        _seed = seed;
        _rng = new Random(seed);
    }

    /// <inheritdoc/>
    public Matrix<T> ExtractKnowledge(Vector<T> localModelParameters, Matrix<T>? publicData)
    {
        // FedGEN extracts class-conditional statistics (mean, variance) per class
        // Encoded as a matrix: rows = classes, cols = 2*featureDim (mean + variance)
        int d = localModelParameters.Length;
        int statDim = Math.Min(_featureDim, d / _numClasses);

        var stats = new Matrix<T>(_numClasses, statDim * 2);
        for (int c = 0; c < _numClasses; c++)
        {
            int offset = (c * statDim) % d;
            for (int f = 0; f < statDim; f++)
            {
                int idx = (offset + f) % d;
                // Mean: parameter value
                stats[c, f] = localModelParameters[idx];
                // Variance: absolute value of parameter (heuristic for spread)
                double absVal = Math.Abs(NumOps.ToDouble(localModelParameters[idx]));
                stats[c, statDim + f] = NumOps.FromDouble(Math.Max(0.01, absVal));
            }
        }

        return stats;
    }

    /// <inheritdoc/>
    public Matrix<T> AggregateKnowledge(Dictionary<int, Matrix<T>> clientKnowledge, Dictionary<int, double>? clientWeights)
    {
        if (clientKnowledge.Count == 0)
            throw new ArgumentException("No client knowledge provided.", nameof(clientKnowledge));

        var first = clientKnowledge.Values.First();
        int rows = first.Rows;
        int cols = first.Columns;

        // Average class-conditional statistics across clients
        double totalWeight = 0;
        var aggregated = new Matrix<T>(rows, cols);

        foreach (var (clientId, stats) in clientKnowledge)
        {
            double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
            totalWeight += w;

            T weight = NumOps.FromDouble(w);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    aggregated[i, j] = NumOps.Add(aggregated[i, j], NumOps.Multiply(stats[i, j], weight));
                }
            }
        }

        T invTotal = NumOps.FromDouble(1.0 / totalWeight);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                aggregated[i, j] = NumOps.Multiply(aggregated[i, j], invTotal);
            }
        }

        // Update generator weights from aggregated class statistics
        int statDim = cols / 2;
        _generatorWeights = new T[rows][];
        for (int c = 0; c < rows; c++)
        {
            _generatorWeights[c] = new T[statDim];
            for (int f = 0; f < statDim; f++)
            {
                _generatorWeights[c][f] = aggregated[c, f]; // Use mean as centroid
            }
        }

        return aggregated;
    }

    /// <inheritdoc/>
    public Vector<T> ApplyKnowledge(Vector<T> localModelParameters, Matrix<T> globalKnowledge,
        Matrix<T>? publicData, double temperature)
    {
        // Generate synthetic samples from aggregated statistics and distill
        int d = localModelParameters.Length;
        int statDim = globalKnowledge.Columns / 2;
        int samplesPerClass = 10;
        int totalSamples = _numClasses * samplesPerClass;

        // Generate synthetic data from class-conditional Gaussians
        var syntheticData = new Matrix<T>(totalSamples, statDim);
        var syntheticLabels = new int[totalSamples];

        _rng = new Random(_seed + _roundCounter);
        _roundCounter++;
        for (int c = 0; c < _numClasses; c++)
        {
            for (int s = 0; s < samplesPerClass; s++)
            {
                int idx = c * samplesPerClass + s;
                syntheticLabels[idx] = c;
                for (int f = 0; f < statDim; f++)
                {
                    double mean = NumOps.ToDouble(globalKnowledge[c, f]);
                    double variance = NumOps.ToDouble(globalKnowledge[c, statDim + f]);
                    double stddev = Math.Sqrt(variance);
                    // Box-Muller for Gaussian sampling
                    double u1 = 1.0 - _rng.NextDouble();
                    double u2 = _rng.NextDouble();
                    double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                    syntheticData[idx, f] = NumOps.FromDouble(mean + stddev * z);
                }
            }
        }

        // Distill: update local model to classify synthetic data correctly
        var updatedParams = new T[d];
        Array.Copy(localModelParameters.ToArray(), updatedParams, d);

        for (int epoch = 0; epoch < _generatorEpochs; epoch++)
        {
            for (int i = 0; i < totalSamples; i++)
            {
                int label = syntheticLabels[i];
                // Compute prediction scores for this sample
                for (int c = 0; c < Math.Min(_numClasses, d / statDim); c++)
                {
                    T score = NumOps.Zero;
                    for (int f = 0; f < statDim; f++)
                    {
                        int paramIdx = (c * statDim + f) % d;
                        score = NumOps.Add(score, NumOps.Multiply(syntheticData[i, f], updatedParams[paramIdx]));
                    }

                    // Gradient: if c == label, push score up; else push down
                    double target = (c == label) ? 1.0 : 0.0;
                    double grad = NumOps.ToDouble(score) - target;
                    T scaledGrad = NumOps.FromDouble(grad * _generatorLearningRate / totalSamples);

                    for (int f = 0; f < statDim; f++)
                    {
                        int paramIdx = (c * statDim + f) % d;
                        T update = NumOps.Multiply(syntheticData[i, f], scaledGrad);
                        updatedParams[paramIdx] = NumOps.Subtract(updatedParams[paramIdx], update);
                    }
                }
            }
        }

        return new Vector<T>(updatedParams);
    }
}

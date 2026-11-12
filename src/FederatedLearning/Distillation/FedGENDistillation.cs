using AiDotNet.Attributes;
using AiDotNet.Enums;

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
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
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
        _rng = RandomHelper.CreateSeededRandom(seed);
    }

    /// <inheritdoc/>
    public Matrix<T> ExtractKnowledge(Vector<T> localModelParameters, Matrix<T>? publicData)
    {
        Guard.NotNull(localModelParameters);

        // FedGEN extracts class-conditional statistics (mean, variance) per class
        // Encoded as a matrix: rows = classes, cols = 2*featureDim (mean + variance)
        int d = localModelParameters.Length;
        int statDim = StatDimension(d);

        // Mean of class c, feature f: parameter (c * statDim + f) mod d.
        var index = new int[_numClasses * statDim];
        for (int k = 0; k < index.Length; k++)
        {
            index[k] = k % d;
        }

        var means = Engine.TensorGather(localModelParameters.AsTensor(), new Tensor<int>(index, new[] { index.Length }), 0)
            .Reshape(new[] { _numClasses, statDim });

        // Variance: absolute value of parameter (heuristic for spread), floored at 0.01.
        var variances = Engine.TensorMax(Engine.TensorAbs(means), NumOps.FromDouble(0.01));
        return Engine.TensorConcatenate(new[] { means, variances }, 1).ToMatrix();
    }

    /// <inheritdoc/>
    public Matrix<T> AggregateKnowledge(Dictionary<int, Matrix<T>> clientKnowledge, Dictionary<int, double>? clientWeights)
    {
        Guard.NotNull(clientKnowledge);
        if (clientKnowledge.Count == 0)
            throw new ArgumentException("No client knowledge provided.", nameof(clientKnowledge));

        var first = clientKnowledge.Values.First();
        int rows = first.Rows;
        int cols = first.Columns;

        double totalWeight = clientKnowledge.Keys.Sum(id => clientWeights?.GetValueOrDefault(id, 1.0) ?? 1.0);
        if (totalWeight <= 0)
            throw new ArgumentException("Client weights must sum to a positive value.", nameof(clientWeights));

        // Average class-conditional statistics across clients (weights normalized by their total).
        var aggregated = new Tensor<T>(new[] { rows, cols });
        foreach (var (clientId, stats) in clientKnowledge)
        {
            if (stats.Rows != rows || stats.Columns != cols)
            {
                throw new ArgumentException(
                    $"Client {clientId} statistics are {stats.Rows}x{stats.Columns}; expected {rows}x{cols}.",
                    nameof(clientKnowledge));
            }

            double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
            aggregated = Engine.TensorAdd(aggregated,
                Engine.TensorMultiplyScalar(Tensor<T>.FromMatrix(stats), NumOps.FromDouble(w / totalWeight)));
        }

        // Update generator weights from aggregated class statistics (mean as centroid).
        int statDim = cols / 2;
        var centroids = Engine.TensorSlice(aggregated, new[] { 0, 0 }, new[] { rows, statDim }).ToArray();
        _generatorWeights = new T[rows][];
        for (int c = 0; c < rows; c++)
        {
            _generatorWeights[c] = new T[statDim];
            Array.Copy(centroids, c * statDim, _generatorWeights[c], 0, statDim);
        }

        return aggregated.ToMatrix();
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Draws class-conditional Gaussian samples from the global statistics, then runs full-batch gradient
    /// descent on the squared error between the class scores s = X Wᵀ and the one-hot labels, where
    /// W[c, f] = θ[c·statDim + f]: ∇W = (S − Y)ᵀ X / N.
    /// </remarks>
    public Vector<T> ApplyKnowledge(Vector<T> localModelParameters, Matrix<T> globalKnowledge,
        Matrix<T>? publicData, double temperature)
    {
        Guard.NotNull(localModelParameters);
        Guard.NotNull(globalKnowledge);

        // Generate synthetic samples from aggregated statistics and distill
        int d = localModelParameters.Length;
        int statDim = globalKnowledge.Columns / 2;
        if (statDim == 0 || globalKnowledge.Rows < _numClasses)
        {
            throw new ArgumentException(
                $"Global knowledge must have at least {_numClasses} rows and two columns.", nameof(globalKnowledge));
        }

        if (d < statDim)
        {
            throw new ArgumentException(
                $"Model has {d} parameters, fewer than the {statDim} features of one class.", nameof(localModelParameters));
        }

        int samplesPerClass = 10;
        int totalSamples = _numClasses * samplesPerClass;

        // Box-Muller draws in the original (class, sample, feature) order of the seeded generator.
        _rng = RandomHelper.CreateSeededRandom(_seed + _roundCounter);
        _roundCounter++;
        var z = new double[totalSamples * statDim];
        for (int k = 0; k < z.Length; k++)
        {
            double u1 = 1.0 - _rng.NextDouble();
            double u2 = _rng.NextDouble();
            z[k] = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }

        // x = mean + sqrt(variance) * z, class statistics broadcast over the samples of the class.
        var knowledge = Tensor<T>.FromMatrix(globalKnowledge);
        var sampleShape = new[] { _numClasses, samplesPerClass, statDim };
        var means = Engine.TensorSlice(knowledge, new[] { 0, 0 }, new[] { _numClasses, statDim }).Reshape(new[] { _numClasses, 1, statDim });
        var stddevs = Engine.TensorSqrt(Engine.TensorSlice(knowledge, new[] { 0, statDim }, new[] { _numClasses, statDim }))
            .Reshape(new[] { _numClasses, 1, statDim });
        var noise = new Tensor<double>(z, sampleShape).Cast<T>();
        var syntheticData = Engine.TensorAdd(
            Engine.TensorBroadcastTo(means, sampleShape),
            Engine.TensorMultiply(Engine.TensorBroadcastTo(stddevs, sampleShape), noise))
            .Reshape(new[] { totalSamples, statDim });

        // One-hot targets of the scored classes (sample i belongs to class i / samplesPerClass).
        int scoredClasses = Math.Min(_numClasses, d / statDim);
        var oneHot = new int[totalSamples * scoredClasses];
        for (int i = 0; i < totalSamples; i++)
        {
            int label = i / samplesPerClass;
            if (label < scoredClasses)
            {
                oneHot[i * scoredClasses + label] = 1;
            }
        }

        var targets = new Tensor<int>(oneHot, new[] { totalSamples, scoredClasses }).Cast<T>();
        var weightIndex = new int[scoredClasses * statDim];
        for (int k = 0; k < weightIndex.Length; k++)
        {
            weightIndex[k] = k;
        }

        var weightIndexTensor = new Tensor<int>(weightIndex, new[] { weightIndex.Length });
        var step = NumOps.FromDouble(_generatorLearningRate / totalSamples);

        // Distill: update local model to classify synthetic data correctly
        var parameters = localModelParameters.Clone();
        for (int epoch = 0; epoch < _generatorEpochs; epoch++)
        {
            var weights = Engine.TensorSlice(parameters.AsTensor(), new[] { 0 }, new[] { weightIndex.Length })
                .Reshape(new[] { scoredClasses, statDim });
            var scores = Engine.TensorMatMul(syntheticData, Engine.TensorTranspose(weights));

            // Gradient: if c == label, push score up; else push down
            var residual = Engine.TensorMultiplyScalar(Engine.TensorSubtract(scores, targets), step);
            var gradWeights = Engine.TensorMatMul(Engine.TensorTranspose(residual), syntheticData);
            var gradParams = Engine.TensorIndexAdd(new Tensor<T>(new[] { d }), 0, weightIndexTensor,
                gradWeights.Reshape(new[] { weightIndex.Length }));
            parameters = Engine.Subtract(parameters, gradParams.ToVector());
        }

        return parameters;
    }

    private int StatDimension(int parameterCount)
    {
        int statDim = Math.Min(_featureDim, parameterCount / _numClasses);
        if (statDim <= 0)
        {
            throw new ArgumentException(
                $"Model has {parameterCount} parameters; at least {_numClasses} are needed for {_numClasses} classes.");
        }

        return statDim;
    }
}

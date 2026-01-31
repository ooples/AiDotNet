using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.FederatedLearning;

/// <summary>
/// Dirichlet distribution-based splitter for federated learning with controlled heterogeneity.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Dirichlet distribution is used to control how "unequal" the
/// class distributions are across clients. A concentration parameter (alpha) controls the
/// heterogeneity: smaller alpha means more heterogeneous (clients have very different distributions),
/// larger alpha means more homogeneous (clients have similar distributions).
/// </para>
/// <para>
/// <b>Alpha Values:</b>
/// - alpha = 0.1: Extreme heterogeneity (clients may have only 1-2 classes)
/// - alpha = 1.0: Moderate heterogeneity
/// - alpha = 10.0: Nearly IID (all clients have similar class distributions)
/// - alpha = 100.0: Practically IID
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - Systematic federated learning experiments
/// - Studying effect of heterogeneity levels
/// - Reproducing research paper setups
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class DirichletSplitter<T> : DataSplitterBase<T>
{
    private readonly int _numClients;
    private readonly double _alpha;
    private readonly double _testRatio;
    private readonly int _minSamplesPerClient;

    /// <summary>
    /// Creates a new Dirichlet distribution splitter.
    /// </summary>
    /// <param name="numClients">Number of clients. Default is 10.</param>
    /// <param name="alpha">Dirichlet concentration parameter. Default is 0.5 (moderate heterogeneity).</param>
    /// <param name="testRatio">Ratio for global test set. Default is 0.1 (10%).</param>
    /// <param name="minSamplesPerClient">Minimum samples each client must have. Default is 10.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public DirichletSplitter(
        int numClients = 10,
        double alpha = 0.5,
        double testRatio = 0.1,
        int minSamplesPerClient = 10,
        int randomSeed = 42)
        : base(shuffle: true, randomSeed)
    {
        if (numClients < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(numClients),
                "Number of clients must be at least 2.");
        }

        if (alpha <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(alpha),
                "Alpha must be positive.");
        }

        if (testRatio < 0 || testRatio >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(testRatio),
                "Test ratio must be between 0 and 1.");
        }

        _numClients = numClients;
        _alpha = alpha;
        _testRatio = testRatio;
        _minSamplesPerClient = minSamplesPerClient;
    }

    /// <inheritdoc/>
    public override int NumSplits => _numClients;

    /// <inheritdoc/>
    public override bool RequiresLabels => true;

    /// <inheritdoc/>
    public override string Description => $"Dirichlet Federated ({_numClients} clients, alpha={_alpha:F2})";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);
        var splits = GetSplits(X, y!).ToList();
        return splits.Count > 0 ? splits[0] : throw new InvalidOperationException("No splits generated.");
    }

    /// <inheritdoc/>
    public override IEnumerable<DataSplitResult<T>> GetSplits(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        if (y == null)
        {
            throw new ArgumentNullException(nameof(y), "Dirichlet splitting requires labels.");
        }

        // Group by class
        var classGroups = GroupByLabel(y);
        int numClasses = classGroups.Count;
        var classes = classGroups.Keys.ToList();

        // Global test set (stratified)
        var testIndices = new List<int>();
        var trainClassIndices = new Dictionary<double, List<int>>();

        foreach (var kvp in classGroups)
        {
            var indices = kvp.Value.ToArray();
            ShuffleIndices(indices);

            int testCount = Math.Max(1, (int)(indices.Length * _testRatio));
            testIndices.AddRange(indices.Take(testCount));
            trainClassIndices[kvp.Key] = indices.Skip(testCount).ToList();
        }

        // Sample Dirichlet proportions for each client
        var clientProportions = SampleDirichletProportions(numClasses, _numClients);

        // Initialize client data
        var clientIndices = new List<List<int>>();
        for (int c = 0; c < _numClients; c++)
        {
            clientIndices.Add(new List<int>());
        }

        // Distribute samples from each class according to Dirichlet proportions
        for (int k = 0; k < numClasses; k++)
        {
            var classKey = classes[k];
            var available = trainClassIndices[classKey];
            int totalForClass = available.Count;

            // Calculate how many samples each client gets from this class
            var samplesPerClient = new int[_numClients];
            int allocated = 0;

            for (int c = 0; c < _numClients; c++)
            {
                samplesPerClient[c] = (int)(totalForClass * clientProportions[c, k]);
                allocated += samplesPerClient[c];
            }

            // Distribute remainder
            int remainder = totalForClass - allocated;
            for (int r = 0; r < remainder; r++)
            {
                samplesPerClient[r % _numClients]++;
            }

            // Assign samples to clients
            int offset = 0;
            for (int c = 0; c < _numClients; c++)
            {
                for (int i = 0; i < samplesPerClient[c] && offset < available.Count; i++)
                {
                    clientIndices[c].Add(available[offset++]);
                }
            }
        }

        // Ensure minimum samples per client
        EnsureMinimumSamples(clientIndices, trainClassIndices, classes);

        // Create splits
        for (int c = 0; c < _numClients; c++)
        {
            yield return BuildResult(X, y, clientIndices[c].ToArray(), testIndices.ToArray(),
                foldIndex: c, totalFolds: _numClients);
        }
    }

    private double[,] SampleDirichletProportions(int numClasses, int numClients)
    {
        var proportions = new double[numClients, numClasses];

        // For each client, sample from Dirichlet distribution
        for (int c = 0; c < numClients; c++)
        {
            // Sample gamma variates and normalize to get Dirichlet sample
            var gammas = new double[numClasses];
            double sum = 0;

            for (int k = 0; k < numClasses; k++)
            {
                gammas[k] = SampleGamma(_alpha, 1.0);
                sum += gammas[k];
            }

            for (int k = 0; k < numClasses; k++)
            {
                proportions[c, k] = sum > 0 ? gammas[k] / sum : 1.0 / numClasses;
            }
        }

        return proportions;
    }

    private double SampleGamma(double shape, double scale)
    {
        // Marsaglia and Tsang's method for gamma distribution
        if (shape < 1)
        {
            // For shape < 1, use the relationship with shape + 1
            double u = _random.NextDouble();
            return SampleGamma(shape + 1, scale) * Math.Pow(u, 1.0 / shape);
        }

        double d = shape - 1.0 / 3.0;
        double c = 1.0 / Math.Sqrt(9.0 * d);

        while (true)
        {
            double x, v;
            do
            {
                x = SampleStandardNormal();
                v = 1.0 + c * x;
            }
            while (v <= 0);

            v = v * v * v;
            double u = _random.NextDouble();

            if (u < 1.0 - 0.0331 * (x * x) * (x * x))
            {
                return scale * d * v;
            }

            if (Math.Log(u) < 0.5 * x * x + d * (1.0 - v + Math.Log(v)))
            {
                return scale * d * v;
            }
        }
    }

    private double SampleStandardNormal()
    {
        // Box-Muller transform
        double u1 = _random.NextDouble();
        double u2 = _random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    private void EnsureMinimumSamples(
        List<List<int>> clientIndices,
        Dictionary<double, List<int>> trainClassIndices,
        List<double> classes)
    {
        // Find clients with too few samples
        var deficientClients = new List<int>();
        for (int c = 0; c < _numClients; c++)
        {
            if (clientIndices[c].Count < _minSamplesPerClient)
            {
                deficientClients.Add(c);
            }
        }

        // Redistribute from surplus clients if needed
        foreach (int deficient in deficientClients)
        {
            int needed = _minSamplesPerClient - clientIndices[deficient].Count;

            // Find a surplus client to take from
            for (int c = 0; c < _numClients && needed > 0; c++)
            {
                if (c != deficient && clientIndices[c].Count > _minSamplesPerClient + needed)
                {
                    int toMove = Math.Min(needed, clientIndices[c].Count - _minSamplesPerClient);
                    for (int i = 0; i < toMove; i++)
                    {
                        int idx = clientIndices[c][clientIndices[c].Count - 1];
                        clientIndices[c].RemoveAt(clientIndices[c].Count - 1);
                        clientIndices[deficient].Add(idx);
                        needed--;
                    }
                }
            }
        }
    }
}

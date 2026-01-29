using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.FederatedLearning;

/// <summary>
/// IID (Independent and Identically Distributed) client splitter for federated learning.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> In federated learning, data is distributed across multiple clients
/// (devices or institutions). IID partitioning means each client gets a random sample
/// of the overall data, so all clients have similar data distributions.
/// </para>
/// <para>
/// <b>How It Works:</b>
/// 1. Shuffle all data randomly
/// 2. Divide equally (or according to specified ratios) among clients
/// 3. Each client's data is statistically similar to the global distribution
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - Simulating ideal federated learning scenarios
/// - Baseline experiments before testing non-IID
/// - When clients should have representative samples
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class IIDClientSplitter<T> : DataSplitterBase<T>
{
    private readonly int _numClients;
    private readonly double[]? _clientRatios;
    private readonly double _testRatio;

    /// <summary>
    /// Creates a new IID client splitter.
    /// </summary>
    /// <param name="numClients">Number of clients to partition data into. Default is 10.</param>
    /// <param name="clientRatios">Optional specific ratios for each client. If null, equal distribution.</param>
    /// <param name="testRatio">Ratio of data to hold out as global test set. Default is 0.1 (10%).</param>
    /// <param name="shuffle">Whether to shuffle before splitting. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public IIDClientSplitter(
        int numClients = 10,
        double[]? clientRatios = null,
        double testRatio = 0.1,
        bool shuffle = true,
        int randomSeed = 42)
        : base(shuffle, randomSeed)
    {
        if (numClients < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(numClients),
                "Number of clients must be at least 2.");
        }

        if (clientRatios != null)
        {
            if (clientRatios.Length != numClients)
            {
                throw new ArgumentException(
                    $"Client ratios length ({clientRatios.Length}) must match number of clients ({numClients}).");
            }

            double sum = clientRatios.Sum();
            if (Math.Abs(sum - 1.0) > 0.001)
            {
                throw new ArgumentException($"Client ratios must sum to 1.0, got {sum}.");
            }
        }

        if (testRatio < 0 || testRatio >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(testRatio),
                "Test ratio must be between 0 and 1.");
        }

        _numClients = numClients;
        _clientRatios = clientRatios;
        _testRatio = testRatio;
    }

    /// <inheritdoc/>
    public override int NumSplits => _numClients;

    /// <inheritdoc/>
    public override string Description => $"IID Federated ({_numClients} clients, {_testRatio * 100:F0}% test)";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        // Return first client's split
        ValidateInputs(X, y);
        var splits = GetSplits(X, y).ToList();
        return splits.Count > 0 ? splits[0] : throw new InvalidOperationException("No splits generated.");
    }

    /// <inheritdoc/>
    public override IEnumerable<DataSplitResult<T>> GetSplits(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        int nSamples = X.Rows;
        var indices = GetShuffledIndices(nSamples);

        // Hold out global test set
        int testSize = (int)(nSamples * _testRatio);
        var testIndices = indices.Take(testSize).ToArray();
        var trainPool = indices.Skip(testSize).ToArray();

        // Compute client sizes
        int[] clientSizes = ComputeClientSizes(trainPool.Length);

        // Distribute to clients
        int offset = 0;
        for (int c = 0; c < _numClients; c++)
        {
            var clientIndices = trainPool.Skip(offset).Take(clientSizes[c]).ToArray();
            offset += clientSizes[c];

            yield return BuildResult(X, y, clientIndices, testIndices,
                foldIndex: c, totalFolds: _numClients);
        }
    }

    private int[] ComputeClientSizes(int totalSize)
    {
        var sizes = new int[_numClients];

        if (_clientRatios != null)
        {
            for (int i = 0; i < _numClients; i++)
            {
                sizes[i] = Math.Max(1, (int)(totalSize * _clientRatios[i]));
            }
        }
        else
        {
            int baseSize = totalSize / _numClients;
            int remainder = totalSize % _numClients;

            for (int i = 0; i < _numClients; i++)
            {
                sizes[i] = baseSize + (i < remainder ? 1 : 0);
            }
        }

        return sizes;
    }
}

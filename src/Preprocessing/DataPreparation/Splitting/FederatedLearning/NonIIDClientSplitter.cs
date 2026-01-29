using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.FederatedLearning;

/// <summary>
/// Non-IID (non-Independent and Identically Distributed) client splitter for federated learning.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> In real-world federated learning, data on different clients
/// is often NOT identically distributed. For example, users' photos on different phones
/// reflect their individual preferences. This splitter simulates such heterogeneous distributions.
/// </para>
/// <para>
/// <b>Heterogeneity Types:</b>
/// - Label skew: Each client has only some classes
/// - Quantity skew: Clients have different amounts of data
/// - Feature skew: Clients have different feature distributions
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - Realistic federated learning experiments
/// - Testing robustness to heterogeneous data
/// - Simulating domain adaptation scenarios
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class NonIIDClientSplitter<T> : DataSplitterBase<T>
{
    private readonly int _numClients;
    private readonly int _classesPerClient;
    private readonly double _testRatio;
    private readonly bool _allowOverlap;

    /// <summary>
    /// Creates a new Non-IID client splitter.
    /// </summary>
    /// <param name="numClients">Number of clients. Default is 10.</param>
    /// <param name="classesPerClient">Number of classes each client has. Default is 2.</param>
    /// <param name="testRatio">Ratio for global test set. Default is 0.1 (10%).</param>
    /// <param name="allowOverlap">Whether multiple clients can share the same class. Default is true.</param>
    /// <param name="shuffle">Whether to shuffle within classes. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public NonIIDClientSplitter(
        int numClients = 10,
        int classesPerClient = 2,
        double testRatio = 0.1,
        bool allowOverlap = true,
        bool shuffle = true,
        int randomSeed = 42)
        : base(shuffle, randomSeed)
    {
        if (numClients < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(numClients),
                "Number of clients must be at least 2.");
        }

        if (classesPerClient < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(classesPerClient),
                "Classes per client must be at least 1.");
        }

        if (testRatio < 0 || testRatio >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(testRatio),
                "Test ratio must be between 0 and 1.");
        }

        _numClients = numClients;
        _classesPerClient = classesPerClient;
        _testRatio = testRatio;
        _allowOverlap = allowOverlap;
    }

    /// <inheritdoc/>
    public override int NumSplits => _numClients;

    /// <inheritdoc/>
    public override bool RequiresLabels => true;

    /// <inheritdoc/>
    public override string Description => $"Non-IID Federated ({_numClients} clients, {_classesPerClient} classes each)";

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
            throw new ArgumentNullException(nameof(y), "Non-IID splitting requires labels.");
        }

        // Group by class
        var classGroups = GroupByLabel(y);
        var classes = classGroups.Keys.ToList();
        int numClasses = classes.Count;

        if (_classesPerClient > numClasses)
        {
            throw new ArgumentException(
                $"Classes per client ({_classesPerClient}) exceeds available classes ({numClasses}).");
        }

        // Prepare class indices
        var classIndices = new Dictionary<double, List<int>>();
        foreach (var kvp in classGroups)
        {
            var indices = kvp.Value.ToArray();
            if (_shuffle)
            {
                ShuffleIndices(indices);
            }

            // Hold out test samples from each class
            int testCount = Math.Max(1, (int)(indices.Length * _testRatio));
            classIndices[kvp.Key] = indices.Skip(testCount).ToList();
        }

        // Global test set
        var testIndices = new List<int>();
        foreach (var kvp in classGroups)
        {
            var indices = kvp.Value.ToArray();
            if (_shuffle)
            {
                ShuffleIndices(indices);
            }
            int testCount = Math.Max(1, (int)(indices.Length * _testRatio));
            testIndices.AddRange(indices.Take(testCount));
        }

        // Assign classes to clients
        var clientClasses = AssignClassesToClients(classes, numClasses);

        // Create splits for each client
        for (int c = 0; c < _numClients; c++)
        {
            var clientIndices = new List<int>();

            foreach (var classKey in clientClasses[c])
            {
                if (classIndices.TryGetValue(classKey, out var available))
                {
                    // Each client takes a portion of its assigned classes
                    int takeCount = _allowOverlap
                        ? available.Count / (_numClients / Math.Max(1, numClasses / _classesPerClient))
                        : available.Count;

                    takeCount = Math.Max(1, takeCount);
                    var toAdd = available.Take(takeCount).ToList();
                    clientIndices.AddRange(toAdd);

                    if (!_allowOverlap)
                    {
                        // Remove taken samples to prevent overlap
                        foreach (var idx in toAdd)
                        {
                            available.Remove(idx);
                        }
                    }
                }
            }

            yield return BuildResult(X, y, clientIndices.ToArray(), testIndices.ToArray(),
                foldIndex: c, totalFolds: _numClients);
        }
    }

    private List<List<double>> AssignClassesToClients(List<double> classes, int numClasses)
    {
        var clientClasses = new List<List<double>>(_numClients);

        if (_allowOverlap)
        {
            // Each client randomly selects _classesPerClient classes
            for (int c = 0; c < _numClients; c++)
            {
                var shuffledClasses = classes.OrderBy(_ => _random.Next()).ToList();
                clientClasses.Add(shuffledClasses.Take(_classesPerClient).ToList());
            }
        }
        else
        {
            // Round-robin assignment without overlap
            int classIndex = 0;
            for (int c = 0; c < _numClients; c++)
            {
                var assigned = new List<double>();
                for (int i = 0; i < _classesPerClient && classIndex < numClasses; i++)
                {
                    assigned.Add(classes[classIndex % numClasses]);
                    classIndex++;
                }
                clientClasses.Add(assigned);
            }
        }

        return clientClasses;
    }
}

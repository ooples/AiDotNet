using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.Evaluation.CrossValidation;

/// <summary>
/// Monte Carlo cross-validation (repeated random sub-sampling validation).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Monte Carlo CV randomly splits data multiple times:
/// <list type="bullet">
/// <item>Each iteration creates a random train/test split</item>
/// <item>Unlike K-Fold, samples may appear in validation multiple times or never</item>
/// <item>More flexible control over train/test sizes</item>
/// <item>Good for small datasets where K-Fold has high variance</item>
/// </list>
/// </para>
/// <para><b>Comparison to K-Fold:</b>
/// <list type="bullet">
/// <item>K-Fold: Each sample in validation exactly once</item>
/// <item>Monte Carlo: Samples may repeat or be missing in validation</item>
/// <item>Monte Carlo: More iterations possible with less computation</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MonteCarloStrategy<T> : ICrossValidationStrategy<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly int _nIterations;
    private readonly double _testSize;
    private readonly int? _randomSeed;

    public string Name => $"Monte Carlo ({_nIterations} iterations)";
    public string Description => "Repeated random sub-sampling validation.";
    public int NumSplits => _nIterations;

    /// <summary>
    /// Initializes Monte Carlo cross-validation.
    /// </summary>
    /// <param name="nIterations">Number of random splits. Default: 100.</param>
    /// <param name="testSize">Fraction of data for validation (0-1). Default: 0.2.</param>
    /// <param name="randomSeed">Random seed for reproducibility.</param>
    public MonteCarloStrategy(int nIterations = 100, double testSize = 0.2, int? randomSeed = null)
    {
        if (nIterations < 1)
            throw new ArgumentException("Number of iterations must be at least 1.");
        if (testSize <= 0 || testSize >= 1)
            throw new ArgumentException("Test size must be between 0 and 1 (exclusive).");

        _nIterations = nIterations;
        _testSize = testSize;
        _randomSeed = randomSeed;
    }

    public IEnumerable<(int[] TrainIndices, int[] ValidationIndices)> Split(int dataSize, ReadOnlySpan<T> labels = default)
    {
        if (dataSize < 2)
            throw new ArgumentException("Dataset size must be at least 2.");

        var random = _randomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(_randomSeed.Value)
            : RandomHelper.CreateSecureRandom();

        int validationSize = Math.Max(1, (int)(dataSize * _testSize));
        var allIndices = Enumerable.Range(0, dataSize).ToArray();

        for (int iter = 0; iter < _nIterations; iter++)
        {
            // Shuffle and split
            var shuffled = allIndices.OrderBy(_ => random.Next()).ToArray();
            var validationIndices = shuffled.Take(validationSize).ToArray();
            var trainIndices = shuffled.Skip(validationSize).ToArray();

            yield return (trainIndices, validationIndices);
        }
    }
}

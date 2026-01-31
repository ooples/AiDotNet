using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.CrossValidation;

/// <summary>
/// Leave-P-Out cross-validation where all combinations of p samples form the test sets.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Leave-P-Out (LPO) tests all possible combinations of p samples.
/// This is an exhaustive evaluation but can be extremely expensive.
/// </para>
/// <para>
/// <b>Number of Splits:</b>
/// The number of splits is C(n, p) = n! / (p! × (n-p)!)
/// - n=10, p=2: 45 splits
/// - n=20, p=2: 190 splits
/// - n=10, p=3: 120 splits
/// </para>
/// <para>
/// <b>Warning:</b> The number of combinations grows very quickly!
/// LPO is only practical for very small datasets and small p values.
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - Tiny datasets (&lt;30 samples)
/// - When you need exhaustive evaluation
/// - Statistical research
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class LeavePOutSplitter<T> : DataSplitterBase<T>
{
    private readonly int _p;
    private int _numCombinations;

    /// <summary>
    /// Creates a new Leave-P-Out cross-validation splitter.
    /// </summary>
    /// <param name="p">Number of samples to leave out for each test set. Default is 2.</param>
    public LeavePOutSplitter(int p = 2) : base(shuffle: false, randomSeed: 42)
    {
        if (p < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(p), "P must be at least 1.");
        }

        _p = p;
    }

    /// <inheritdoc/>
    public override int NumSplits => _numCombinations;

    /// <inheritdoc/>
    public override string Description => $"Leave-{_p}-Out cross-validation";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        return GetSplits(X, y).First();
    }

    /// <inheritdoc/>
    public override IEnumerable<DataSplitResult<T>> GetSplits(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        int n = X.Rows;

        if (_p >= n)
        {
            throw new ArgumentException(
                $"P ({_p}) must be less than the number of samples ({n}).");
        }

        _numCombinations = (int)BinomialCoefficient(n, _p);

        if (_numCombinations > 10000)
        {
            Console.WriteLine(
                $"Warning: Leave-{_p}-Out with {n} samples will generate {_numCombinations} splits. " +
                "This may take a very long time.");
        }

        int splitIndex = 0;
        foreach (var testIndices in GenerateCombinations(n, _p))
        {
            // Train indices are everything not in test
            var testSet = new HashSet<int>(testIndices);
            var trainIndices = new int[n - _p];
            int trainIdx = 0;
            for (int i = 0; i < n; i++)
            {
                if (!testSet.Contains(i))
                {
                    trainIndices[trainIdx++] = i;
                }
            }

            yield return BuildResult(X, y, trainIndices, testIndices,
                foldIndex: splitIndex++, totalFolds: _numCombinations);
        }
    }

    /// <summary>
    /// Generates all combinations of k items from n items.
    /// </summary>
    private static IEnumerable<int[]> GenerateCombinations(int n, int k)
    {
        int[] combination = new int[k];

        // Initialize first combination
        for (int i = 0; i < k; i++)
        {
            combination[i] = i;
        }

        while (true)
        {
            yield return (int[])combination.Clone();

            // Find rightmost element that can be incremented
            int i = k - 1;
            while (i >= 0 && combination[i] == n - k + i)
            {
                i--;
            }

            if (i < 0)
            {
                yield break;
            }

            // Increment and reset all elements to the right
            combination[i]++;
            for (int j = i + 1; j < k; j++)
            {
                combination[j] = combination[j - 1] + 1;
            }
        }
    }

    /// <summary>
    /// Calculates the binomial coefficient C(n, k) = n! / (k! × (n-k)!)
    /// </summary>
    private static long BinomialCoefficient(int n, int k)
    {
        if (k > n - k)
        {
            k = n - k;
        }

        long result = 1;
        for (int i = 0; i < k; i++)
        {
            result = result * (n - i) / (i + 1);
        }

        return result;
    }
}

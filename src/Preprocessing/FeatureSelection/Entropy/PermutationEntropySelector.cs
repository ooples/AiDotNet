using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Entropy;

/// <summary>
/// Permutation Entropy based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on permutation entropy, which measures the complexity
/// of time series or sequential data based on ordinal patterns.
/// </para>
/// <para><b>For Beginners:</b> Permutation entropy looks at the order of values
/// in small windows. It counts how many different orderings appear and how often.
/// High permutation entropy means complex, unpredictable sequences; low entropy
/// means regular, predictable patterns like trends.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class PermutationEntropySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _embeddingDimension;
    private readonly int _delay;

    private Vector<T>? _entropyValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int EmbeddingDimension => _embeddingDimension;
    public int Delay => _delay;
    public Vector<T>? EntropyValues => _entropyValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public PermutationEntropySelector(
        int nFeaturesToSelect = 10,
        int embeddingDimension = 3,
        int delay = 1,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (embeddingDimension < 2)
            throw new ArgumentException("Embedding dimension must be at least 2.", nameof(embeddingDimension));
        if (delay < 1)
            throw new ArgumentException("Delay must be at least 1.", nameof(delay));

        _nFeaturesToSelect = nFeaturesToSelect;
        _embeddingDimension = embeddingDimension;
        _delay = delay;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        int minLength = (_embeddingDimension - 1) * _delay + 1;
        if (n < minLength)
            throw new ArgumentException($"Need at least {minLength} samples for embedding dimension {_embeddingDimension} and delay {_delay}.");

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);
        int m = _embeddingDimension;

        // Maximum entropy for m! possible permutations
        int factorial = 1;
        for (int i = 2; i <= m; i++)
            factorial *= i;
        double maxEntropy = Math.Log(factorial) / Math.Log(2);

        // Pattern of a window = the stable rank of every position (equal values rank by position), which
        // identifies its stable argsort permutation one to one
        int totalPatterns = n - (m - 1) * _delay;
        var pairShape = new[] { totalPatterns, m, m };
        var earlier = engine.TensorBroadcastTo(engine.Reshape(engine.TensorTranspose(
            engine.TensorTriu(PreprocessingTensorOps.Filled(new[] { m, m }, NumOps.One), 1)), new[] { 1, m, m }), pairShape);

        var entropyValues = new Tensor<T>[p];
        for (int j = 0; j < p; j++)
        {
            // Extract embedded vectors [patterns, m]: position d of window i is col[i + d * delay]
            var col = PreprocessingTensorOps.ColumnVector(X, j);
            var lags = new Tensor<T>[m];
            for (int d = 0; d < m; d++)
                lags[d] = engine.Reshape(engine.TensorSlice(col, new[] { d * _delay }, new[] { totalPatterns }), new[] { totalPatterns, 1 });
            var embedded = engine.TensorConcatenate(lags, 1);

            // rank_d = #{e : v_e < v_d} + #{e < d : v_e == v_d}
            var self = engine.TensorBroadcastTo(engine.Reshape(embedded, new[] { totalPatterns, m, 1 }), pairShape);
            var other = engine.TensorBroadcastTo(engine.Reshape(embedded, new[] { totalPatterns, 1, m }), pairShape);
            var ranks = engine.ReduceSum(engine.TensorAdd(engine.TensorLessThan(other, self),
                engine.TensorMultiply(engine.TensorEquals(other, self), earlier)), new[] { 2 }, false);

            // Count permutation patterns (ids in order of first appearance)
            var rankValues = ranks.Cast<int>().AsSpan().ToArray();
            var patternIds = new Dictionary<long, int>();
            var windowPattern = new int[totalPatterns];
            for (int i = 0; i < totalPatterns; i++)
            {
                long code = 0;
                for (int d = 0; d < m; d++)
                    code = code * m + rankValues[i * m + d];
                if (!patternIds.TryGetValue(code, out int id))
                {
                    id = patternIds.Count;
                    patternIds[code] = id;
                }

                windowPattern[i] = id;
            }

            // Counted in int (a Half sum of the 0/1 one-hot stops at 2048)
            var patternCounts = engine.Reshape(PreprocessingTensorOps.CategoryCounts<double>(
                new Tensor<int>(windowPattern, new[] { totalPatterns }), patternIds.Count), new[] { 1, patternIds.Count });

            // Compute entropy (bits) normalized by maximum entropy
            entropyValues[j] = maxEntropy > 0
                ? engine.TensorDivideScalar(FeatureSelectionTensorOps.EntropyBits(patternCounts, totalPatterns), maxEntropy).Cast<T>()
                : new Tensor<T>(new[] { 1 });
        }

        var entropies = p == 0 ? new Tensor<T>(new[] { 0 }) : engine.TensorConcatenate(entropyValues, 0);
        _entropyValues = entropies.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(entropies, numToSelect);

        IsFitted = true;
    }
    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        Fit(data);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PermutationEntropySelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("PermutationEntropySelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PermutationEntropySelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}

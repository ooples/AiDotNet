using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Entropy;

/// <summary>
/// Differential Entropy based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their differential (continuous) entropy,
/// estimated using nearest-neighbor methods.
/// </para>
/// <para><b>For Beginners:</b> Differential entropy measures uncertainty in
/// continuous distributions. Unlike discrete entropy, it can be negative.
/// Features with higher differential entropy contain more unpredictable
/// information, which may indicate useful variability.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class DifferentialEntropySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _kNeighbors;

    private Vector<T>? _entropyValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int KNeighbors => _kNeighbors;
    public Vector<T>? EntropyValues => _entropyValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public DifferentialEntropySelector(
        int nFeaturesToSelect = 10,
        int kNeighbors = 5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (kNeighbors < 1)
            throw new ArgumentException("Number of neighbors must be at least 1.", nameof(kNeighbors));

        _nFeaturesToSelect = nFeaturesToSelect;
        _kNeighbors = kNeighbors;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        if (n <= _kNeighbors)
            throw new ArgumentException($"Number of samples ({n}) must be greater than k ({_kNeighbors}).");

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);

        // Kozachenko-Leonenko estimator for 1D differential entropy
        // H = log(n-1) - digamma(k) + log(2) + (1/n) * sum(log(2 * epsilon_i))
        T offset = NumOps.FromDouble(Math.Log(n - 1) - Digamma(_kNeighbors));
        T floor = NumOps.FromDouble(1e-10);
        T two = NumOps.FromDouble(2);
        var shape = new[] { n, n };
        var excludeSelf = engine.TensorMultiplyScalar(engine.TensorEye<T>(n), NumOps.MaxValue);

        var entropies = new Tensor<T>[p];
        for (int j = 0; j < p; j++)
        {
            // k-th nearest neighbour distance of every point (itself excluded), at least 1e-10
            var col = PreprocessingTensorOps.ColumnVector(X, j);
            var distances = engine.TensorAbs(engine.TensorSubtract(
                engine.TensorBroadcastTo(engine.Reshape(col, new[] { n, 1 }), shape),
                engine.TensorBroadcastTo(engine.Reshape(col, new[] { 1, n }), shape)));
            var sorted = engine.TensorSort(engine.TensorAdd(distances, excludeSelf), 1, false).Item1;
            var epsilon = engine.TensorMax(engine.TensorSlice(sorted, new[] { 0, _kNeighbors - 1 }, new[] { n, 1 }), floor);

            entropies[j] = engine.TensorAddScalar(
                engine.ReduceMean(engine.TensorLog(engine.TensorMultiplyScalar(epsilon, two)), new[] { 0 }, false), offset);
        }

        var entropyValues = engine.Reshape(p == 0 ? new Tensor<T>(new[] { 0 }) : engine.TensorConcatenate(entropies, 0), new[] { p });
        _entropyValues = entropyValues.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(entropyValues, numToSelect);

        IsFitted = true;
    }
    private double Digamma(int k)
    {
        // Approximation of digamma function for positive integers
        // psi(k) = -gamma + sum(1/i for i in 1..k-1) where gamma ≈ 0.5772
        if (k == 1) return -0.5772156649;
        double result = -0.5772156649;
        for (int i = 1; i < k; i++)
            result += 1.0 / i;
        return result;
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
            throw new InvalidOperationException("DifferentialEntropySelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("DifferentialEntropySelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DifferentialEntropySelector has not been fitted.");

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

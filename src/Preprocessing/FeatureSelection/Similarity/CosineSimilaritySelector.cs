using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Similarity;

/// <summary>
/// Cosine Similarity based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their cosine similarity with the target,
/// measuring directional alignment rather than magnitude.
/// </para>
/// <para><b>For Beginners:</b> Cosine similarity measures the angle between two
/// vectors, ignoring their length. Features pointing in similar directions
/// as the target are selected, which is useful when scale doesn't matter.
/// </para>
/// </remarks>
public class CosineSimilaritySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _cosineSimilarities;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? CosineSimilarities => _cosineSimilarities;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public CosineSimilaritySelector(
        int nFeaturesToSelect = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "CosineSimilaritySelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Compute norm of y
        double yNorm = 0;
        for (int i = 0; i < n; i++)
            yNorm += y[i] * y[i];
        yNorm = Math.Sqrt(yNorm);

        _cosineSimilarities = new double[p];

        for (int j = 0; j < p; j++)
        {
            double dotProduct = 0;
            double xNorm = 0;

            for (int i = 0; i < n; i++)
            {
                dotProduct += X[i, j] * y[i];
                xNorm += X[i, j] * X[i, j];
            }

            xNorm = Math.Sqrt(xNorm);
            _cosineSimilarities[j] = (xNorm > 1e-10 && yNorm > 1e-10)
                ? Math.Abs(dotProduct / (xNorm * yNorm))
                : 0;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _cosineSimilarities[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CosineSimilaritySelector has not been fitted.");

        int numRows = data.Rows;
        int numCols = _selectedIndices.Length;
        var result = new T[numRows, numCols];

        for (int i = 0; i < numRows; i++)
            for (int j = 0; j < numCols; j++)
                result[i, j] = data[i, _selectedIndices[j]];

        return new Matrix<T>(result);
    }

    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("CosineSimilaritySelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CosineSimilaritySelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

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

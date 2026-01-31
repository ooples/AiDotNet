using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Transfer;

/// <summary>
/// Transfer Learning-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses feature importance learned from a source domain to guide feature
/// selection in a target domain. Particularly useful when the target domain
/// has limited data but the source domain has abundant labeled data.
/// </para>
/// <para><b>For Beginners:</b> Sometimes we have lots of data from one situation
/// (like medical records from one hospital) but little data from another
/// (a new hospital). Transfer learning uses what we learned about important
/// features from the first situation to help with the second.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class TransferLearningSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double[]? _sourceImportances;
    private readonly double _transferWeight;

    private double[]? _targetImportances;
    private double[]? _combinedImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double TransferWeight => _transferWeight;
    public double[]? TargetImportances => _targetImportances;
    public double[]? CombinedImportances => _combinedImportances;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public TransferLearningSelector(
        int nFeaturesToSelect = 10,
        double[]? sourceImportances = null,
        double transferWeight = 0.5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (transferWeight < 0 || transferWeight > 1)
            throw new ArgumentException("Transfer weight must be between 0 and 1.", nameof(transferWeight));

        _nFeaturesToSelect = nFeaturesToSelect;
        _sourceImportances = sourceImportances;
        _transferWeight = transferWeight;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "TransferLearningSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Convert to arrays
        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Compute target domain importances
        _targetImportances = ComputeImportances(X, y, n, p);

        // Combine with source importances if available
        _combinedImportances = new double[p];
        if (_sourceImportances is not null && _sourceImportances.Length == p)
        {
            // Normalize both
            double maxSource = _sourceImportances.Max() + 1e-10;
            double maxTarget = _targetImportances.Max() + 1e-10;

            for (int j = 0; j < p; j++)
            {
                double normSource = _sourceImportances[j] / maxSource;
                double normTarget = _targetImportances[j] / maxTarget;
                _combinedImportances[j] = _transferWeight * normSource + (1 - _transferWeight) * normTarget;
            }
        }
        else
        {
            Array.Copy(_targetImportances, _combinedImportances, p);
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _combinedImportances[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[] ComputeImportances(double[,] X, double[] y, int n, int p)
    {
        var importances = new double[p];
        double yMean = y.Average();

        for (int j = 0; j < p; j++)
        {
            double xMean = 0;
            for (int i = 0; i < n; i++)
                xMean += X[i, j];
            xMean /= n;

            double sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i < n; i++)
            {
                double xDiff = X[i, j] - xMean;
                double yDiff = y[i] - yMean;
                sxy += xDiff * yDiff;
                sxx += xDiff * xDiff;
                syy += yDiff * yDiff;
            }

            importances[j] = (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
        }

        return importances;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TransferLearningSelector has not been fitted.");

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
        throw new NotSupportedException("TransferLearningSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TransferLearningSelector has not been fitted.");

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

using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Correlation;

/// <summary>
/// Partial Correlation for feature selection controlling for other variables.
/// </summary>
/// <remarks>
/// <para>
/// Partial correlation measures the relationship between two variables while
/// controlling for (removing the effect of) other variables. Helps identify
/// direct relationships vs those mediated by confounders.
/// </para>
/// <para><b>For Beginners:</b> Regular correlation can be misleading if two
/// variables are both caused by a third (confounder). Partial correlation
/// "holds constant" other variables to find the true direct relationship.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class PartialCorrelation<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly bool _controlForAllOthers;

    private double[]? _partialCorrelations;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? PartialCorrelations => _partialCorrelations;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public PartialCorrelation(
        int nFeaturesToSelect = 10,
        bool controlForAllOthers = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _controlForAllOthers = controlForAllOthers;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "PartialCorrelation requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        _partialCorrelations = new double[p];

        if (_controlForAllOthers && p > 1)
        {
            // Compute full correlation matrix including target
            var fullMatrix = new double[n, p + 1];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < p; j++)
                    fullMatrix[i, j] = X[i, j];
                fullMatrix[i, p] = y[i];
            }

            var corrMatrix = ComputeCorrelationMatrix(fullMatrix, n, p + 1);
            var precisionMatrix = InvertMatrix(corrMatrix, p + 1);

            if (precisionMatrix is not null)
            {
                // Partial correlation from precision matrix
                for (int j = 0; j < p; j++)
                {
                    double pjj = precisionMatrix[j, j];
                    double pyy = precisionMatrix[p, p];
                    double pjy = precisionMatrix[j, p];

                    if (pjj > 1e-10 && pyy > 1e-10)
                        _partialCorrelations[j] = -pjy / Math.Sqrt(pjj * pyy);
                    else
                        _partialCorrelations[j] = 0;
                }
            }
            else
            {
                // Fallback to simple correlation if matrix inversion fails
                for (int j = 0; j < p; j++)
                    _partialCorrelations[j] = Math.Abs(corrMatrix[j, p]);
            }
        }
        else
        {
            // Simple correlation when not controlling for others
            for (int j = 0; j < p; j++)
                _partialCorrelations[j] = Math.Abs(ComputeSimpleCorrelation(X, y, j, n));
        }

        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _partialCorrelations
            .Select((pc, idx) => (PCorr: Math.Abs(pc), Index: idx))
            .OrderByDescending(x => x.PCorr)
            .Take(nToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[,] ComputeCorrelationMatrix(double[,] data, int n, int p)
    {
        var means = new double[p];
        var stds = new double[p];

        // Compute means
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                means[j] += data[i, j];
            means[j] /= n;
        }

        // Compute standard deviations
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                stds[j] += Math.Pow(data[i, j] - means[j], 2);
            stds[j] = Math.Sqrt(stds[j] / n);
            if (stds[j] < 1e-10) stds[j] = 1;
        }

        // Compute correlation matrix
        var corr = new double[p, p];
        for (int i = 0; i < p; i++)
        {
            corr[i, i] = 1;
            for (int j = i + 1; j < p; j++)
            {
                double cov = 0;
                for (int k = 0; k < n; k++)
                    cov += (data[k, i] - means[i]) * (data[k, j] - means[j]);
                cov /= n;
                corr[i, j] = corr[j, i] = cov / (stds[i] * stds[j]);
            }
        }

        return corr;
    }

    private double[,]? InvertMatrix(double[,] matrix, int size)
    {
        // Simple matrix inversion using Gauss-Jordan elimination
        var augmented = new double[size, 2 * size];

        // Create augmented matrix [A|I]
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
                augmented[i, j] = matrix[i, j];
            augmented[i, size + i] = 1;
        }

        // Forward elimination with partial pivoting
        for (int col = 0; col < size; col++)
        {
            // Find pivot
            int maxRow = col;
            for (int row = col + 1; row < size; row++)
                if (Math.Abs(augmented[row, col]) > Math.Abs(augmented[maxRow, col]))
                    maxRow = row;

            if (Math.Abs(augmented[maxRow, col]) < 1e-10)
                return null;  // Singular matrix

            // Swap rows
            if (maxRow != col)
            {
                for (int j = 0; j < 2 * size; j++)
                {
                    double temp = augmented[col, j];
                    augmented[col, j] = augmented[maxRow, j];
                    augmented[maxRow, j] = temp;
                }
            }

            // Scale pivot row
            double pivot = augmented[col, col];
            for (int j = 0; j < 2 * size; j++)
                augmented[col, j] /= pivot;

            // Eliminate column
            for (int row = 0; row < size; row++)
            {
                if (row != col)
                {
                    double factor = augmented[row, col];
                    for (int j = 0; j < 2 * size; j++)
                        augmented[row, j] -= factor * augmented[col, j];
                }
            }
        }

        // Extract inverse
        var inverse = new double[size, size];
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                inverse[i, j] = augmented[i, size + j];

        return inverse;
    }

    private double ComputeSimpleCorrelation(double[,] X, double[] y, int j, int n)
    {
        double xMean = 0, yMean = 0;
        for (int i = 0; i < n; i++)
        {
            xMean += X[i, j];
            yMean += y[i];
        }
        xMean /= n;
        yMean /= n;

        double ssXY = 0, ssXX = 0, ssYY = 0;
        for (int i = 0; i < n; i++)
        {
            double dx = X[i, j] - xMean;
            double dy = y[i] - yMean;
            ssXY += dx * dy;
            ssXX += dx * dx;
            ssYY += dy * dy;
        }

        if (ssXX < 1e-10 || ssYY < 1e-10) return 0;
        return ssXY / Math.Sqrt(ssXX * ssYY);
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PartialCorrelation has not been fitted.");

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
        throw new NotSupportedException("PartialCorrelation does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PartialCorrelation has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}

using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Constraint;

/// <summary>
/// Constraint-based Feature Selection with domain constraints.
/// </summary>
/// <remarks>
/// <para>
/// Selects features while respecting domain-specific constraints such as
/// mandatory features that must be included, forbidden features that must
/// be excluded, and group constraints where features must be selected together.
/// </para>
/// <para><b>For Beginners:</b> Sometimes domain experts know certain features
/// must be included (like patient age in medical data) or must be excluded
/// (like sensitive personal info). This method respects those rules while
/// still finding the best possible feature subset.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ConstraintBasedSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int[]? _mandatoryFeatures;
    private readonly int[]? _forbiddenFeatures;
    private readonly int[][]? _featureGroups;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ConstraintBasedSelector(
        int nFeaturesToSelect = 10,
        int[]? mandatoryFeatures = null,
        int[]? forbiddenFeatures = null,
        int[][]? featureGroups = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _mandatoryFeatures = mandatoryFeatures;
        _forbiddenFeatures = forbiddenFeatures;
        _featureGroups = featureGroups;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ConstraintBasedSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Compute feature scores (correlation with target)
        _featureScores = new double[p];
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

            _featureScores[j] = (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
        }

        // Initialize selection with mandatory features
        var selected = new HashSet<int>();
        if (_mandatoryFeatures is not null)
        {
            foreach (int idx in _mandatoryFeatures)
            {
                if (idx >= 0 && idx < p)
                    selected.Add(idx);
            }
        }

        // Get candidate features (excluding forbidden and already selected)
        var forbidden = _forbiddenFeatures?.ToHashSet() ?? new HashSet<int>();
        var candidates = Enumerable.Range(0, p)
            .Where(j => !selected.Contains(j) && !forbidden.Contains(j))
            .OrderByDescending(j => _featureScores[j])
            .ToList();

        // Add features respecting group constraints
        foreach (int candidate in candidates)
        {
            if (selected.Count >= _nFeaturesToSelect)
                break;

            // Check if candidate is part of a group
            var group = _featureGroups?.FirstOrDefault(g => g.Contains(candidate));
            if (group is not null)
            {
                // Add entire group if not already added and space permits
                var groupToAdd = group.Where(f => !selected.Contains(f) && !forbidden.Contains(f)).ToList();
                if (selected.Count + groupToAdd.Count <= _nFeaturesToSelect)
                {
                    foreach (int f in groupToAdd)
                        selected.Add(f);
                }
            }
            else
            {
                selected.Add(candidate);
            }
        }

        _selectedIndices = selected.OrderBy(x => x).ToArray();
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
            throw new InvalidOperationException("ConstraintBasedSelector has not been fitted.");

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
        throw new NotSupportedException("ConstraintBasedSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ConstraintBasedSelector has not been fitted.");

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

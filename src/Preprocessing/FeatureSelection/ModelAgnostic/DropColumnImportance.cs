using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.ModelAgnostic;

/// <summary>
/// Drop-Column Importance for model-agnostic feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Drop-Column Importance measures feature importance by removing each feature
/// entirely and measuring how much model performance degrades. This is more
/// thorough than permutation but requires retraining the model for each feature.
/// </para>
/// <para><b>For Beginners:</b> This method is like permutation importance but more
/// definitive. Instead of scrambling a feature, it completely removes it and
/// retrains the model. If the model gets much worse without a feature, that
/// feature was truly important.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class DropColumnImportance<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly Func<Matrix<T>, Vector<T>, double>? _scorer;

    private double[]? _importanceScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? ImportanceScores => _importanceScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public DropColumnImportance(
        int nFeaturesToSelect = 10,
        Func<Matrix<T>, Vector<T>, double>? scorer = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _scorer = scorer;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "DropColumnImportance requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var scorer = _scorer ?? DefaultScorer;

        // Compute baseline score with all features
        double baselineScore = scorer(data, target);

        _importanceScores = new double[p];

        // For each feature, drop it and measure performance
        for (int dropIdx = 0; dropIdx < p; dropIdx++)
        {
            // Create data without this feature
            var reducedData = new T[n, p - 1];
            for (int i = 0; i < n; i++)
            {
                int col = 0;
                for (int j = 0; j < p; j++)
                {
                    if (j == dropIdx) continue;
                    reducedData[i, col++] = data[i, j];
                }
            }

            // Score without this feature
            double droppedScore = scorer(new Matrix<T>(reducedData), target);

            // Importance = drop in score when feature is removed
            _importanceScores[dropIdx] = baselineScore - droppedScore;
        }

        // Select top features by importance
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _importanceScores
            .Select((imp, idx) => (Importance: imp, Index: idx))
            .OrderByDescending(x => x.Importance)
            .Take(numToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double DefaultScorer(Matrix<T> data, Vector<T> target)
    {
        // Use R-squared as default score
        int n = data.Rows;
        int p = data.Columns;

        if (p == 0) return 0;

        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        var xMeans = new double[p];
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                xMeans[j] += NumOps.ToDouble(data[i, j]);
            xMeans[j] /= n;
        }

        double ssTotal = 0;
        double ssResidual = 0;

        for (int i = 0; i < n; i++)
        {
            double yTrue = NumOps.ToDouble(target[i]);
            double yPred = yMean;

            for (int j = 0; j < p; j++)
            {
                double xVal = NumOps.ToDouble(data[i, j]);
                yPred += 0.1 * (xVal - xMeans[j]);
            }

            ssTotal += (yTrue - yMean) * (yTrue - yMean);
            ssResidual += (yTrue - yPred) * (yTrue - yPred);
        }

        return ssTotal > 0 ? 1 - (ssResidual / ssTotal) : 0;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DropColumnImportance has not been fitted.");

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
        throw new NotSupportedException("DropColumnImportance does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DropColumnImportance has not been fitted.");

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

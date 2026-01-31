using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Supervised;

/// <summary>
/// Linear Discriminant Analysis (LDA) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses LDA to identify features that maximize class separability by
/// analyzing the between-class and within-class scatter.
/// </para>
/// <para><b>For Beginners:</b> LDA finds directions in your data where different
/// classes are most separated. Features that contribute most to these
/// discriminating directions are the most useful for classification.
/// </para>
/// </remarks>
public class LDABasedSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _discriminantScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? DiscriminantScores => _discriminantScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public LDABasedSelector(
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
            "LDABasedSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        var y = new int[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = (int)Math.Round(NumOps.ToDouble(target[i]));
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Get unique classes
        var classes = y.Distinct().OrderBy(c => c).ToList();
        int nClasses = classes.Count;

        // Compute overall mean
        var overallMean = new double[p];
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                overallMean[j] += X[i, j];
            overallMean[j] /= n;
        }

        // Compute class means and within-class scatter
        var classMeans = new double[nClasses, p];
        var classCounts = new int[nClasses];
        for (int c = 0; c < nClasses; c++)
        {
            int classLabel = classes[c];
            for (int i = 0; i < n; i++)
            {
                if (y[i] == classLabel)
                {
                    classCounts[c]++;
                    for (int j = 0; j < p; j++)
                        classMeans[c, j] += X[i, j];
                }
            }
            for (int j = 0; j < p; j++)
                classMeans[c, j] /= (classCounts[c] + 1e-10);
        }

        // Compute discriminant scores for each feature (univariate Fisher ratio)
        _discriminantScores = new double[p];
        for (int j = 0; j < p; j++)
        {
            // Between-class variance
            double betweenVar = 0;
            for (int c = 0; c < nClasses; c++)
            {
                double diff = classMeans[c, j] - overallMean[j];
                betweenVar += classCounts[c] * diff * diff;
            }

            // Within-class variance
            double withinVar = 0;
            for (int c = 0; c < nClasses; c++)
            {
                int classLabel = classes[c];
                for (int i = 0; i < n; i++)
                {
                    if (y[i] == classLabel)
                    {
                        double diff = X[i, j] - classMeans[c, j];
                        withinVar += diff * diff;
                    }
                }
            }

            _discriminantScores[j] = betweenVar / (withinVar + 1e-10);
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _discriminantScores[j])
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
            throw new InvalidOperationException("LDABasedSelector has not been fitted.");

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
        throw new NotSupportedException("LDABasedSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LDABasedSelector has not been fitted.");

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

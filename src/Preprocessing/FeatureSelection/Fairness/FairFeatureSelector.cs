using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Fairness;

/// <summary>
/// Fair Feature Selection that balances predictive power with fairness constraints.
/// </summary>
/// <remarks>
/// <para>
/// Selects features that maximize predictive accuracy while minimizing correlation
/// with protected attributes to promote fair predictions.
/// </para>
/// <para><b>For Beginners:</b> In ML fairness, we want models that don't discriminate
/// based on protected attributes like race or gender. This selector chooses features
/// that are good for prediction but have low correlation with a specified protected
/// attribute. This helps build fairer models by removing proxies for protected groups.
/// </para>
/// </remarks>
public class FairFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _fairnessWeight;
    private readonly int _protectedAttributeIndex;

    private double[]? _fairnessScores;
    private double[]? _predictiveScores;
    private double[]? _combinedScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double FairnessWeight => _fairnessWeight;
    public int ProtectedAttributeIndex => _protectedAttributeIndex;
    public double[]? FairnessScores => _fairnessScores;
    public double[]? PredictiveScores => _predictiveScores;
    public double[]? CombinedScores => _combinedScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public FairFeatureSelector(
        int nFeaturesToSelect = 10,
        int protectedAttributeIndex = 0,
        double fairnessWeight = 0.5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (fairnessWeight < 0 || fairnessWeight > 1)
            throw new ArgumentException("Fairness weight must be between 0 and 1.", nameof(fairnessWeight));
        if (protectedAttributeIndex < 0)
            throw new ArgumentException("Protected attribute index must be non-negative.", nameof(protectedAttributeIndex));

        _nFeaturesToSelect = nFeaturesToSelect;
        _fairnessWeight = fairnessWeight;
        _protectedAttributeIndex = protectedAttributeIndex;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FairFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");
        if (_protectedAttributeIndex >= data.Columns)
            throw new ArgumentException("Protected attribute index exceeds number of columns.");

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

        // Extract protected attribute
        var protectedAttr = new double[n];
        for (int i = 0; i < n; i++)
            protectedAttr[i] = X[i, _protectedAttributeIndex];

        double protectedMean = protectedAttr.Average();
        double protectedVar = 0;
        for (int i = 0; i < n; i++)
            protectedVar += (protectedAttr[i] - protectedMean) * (protectedAttr[i] - protectedMean);

        double targetMean = y.Average();
        double targetVar = 0;
        for (int i = 0; i < n; i++)
            targetVar += (y[i] - targetMean) * (y[i] - targetMean);

        _predictiveScores = new double[p];
        _fairnessScores = new double[p];
        _combinedScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            if (j == _protectedAttributeIndex)
            {
                // Protected attribute itself gets zero score
                _predictiveScores[j] = 0;
                _fairnessScores[j] = 0;
                _combinedScores[j] = 0;
                continue;
            }

            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            double featureMean = col.Average();
            double featureVar = 0;
            for (int i = 0; i < n; i++)
                featureVar += (col[i] - featureMean) * (col[i] - featureMean);

            // Compute correlation with target (predictive power)
            double covTarget = 0;
            for (int i = 0; i < n; i++)
                covTarget += (col[i] - featureMean) * (y[i] - targetMean);

            double denomTarget = Math.Sqrt(featureVar * targetVar);
            _predictiveScores[j] = denomTarget > 1e-10 ? Math.Abs(covTarget / denomTarget) : 0;

            // Compute correlation with protected attribute (fairness concern)
            double covProtected = 0;
            for (int i = 0; i < n; i++)
                covProtected += (col[i] - featureMean) * (protectedAttr[i] - protectedMean);

            double denomProtected = Math.Sqrt(featureVar * protectedVar);
            double protectedCorr = denomProtected > 1e-10 ? Math.Abs(covProtected / denomProtected) : 0;

            // Fairness score: higher when less correlated with protected attribute
            _fairnessScores[j] = 1.0 - protectedCorr;

            // Combined score: balance predictive power with fairness
            _combinedScores[j] = (1 - _fairnessWeight) * _predictiveScores[j] + _fairnessWeight * _fairnessScores[j];
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p - 1); // Exclude protected attribute
        _selectedIndices = Enumerable.Range(0, p)
            .Where(j => j != _protectedAttributeIndex)
            .OrderByDescending(j => _combinedScores[j])
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
            throw new InvalidOperationException("FairFeatureSelector has not been fitted.");

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
        throw new NotSupportedException("FairFeatureSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FairFeatureSelector has not been fitted.");

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

    /// <summary>
    /// Computes the disparate impact ratio for a specific feature.
    /// </summary>
    public double GetDisparateImpact(Matrix<T> data, int featureIndex)
    {
        if (featureIndex >= data.Columns || featureIndex < 0)
            throw new ArgumentException("Invalid feature index.");

        int n = data.Rows;

        // Split by protected attribute (threshold at median)
        var protectedVals = new double[n];
        var featureVals = new double[n];

        for (int i = 0; i < n; i++)
        {
            protectedVals[i] = NumOps.ToDouble(data[i, _protectedAttributeIndex]);
            featureVals[i] = NumOps.ToDouble(data[i, featureIndex]);
        }

        Array.Sort(protectedVals.ToArray());
        double median = protectedVals[n / 2];

        // Compute mean feature value for each group
        double sum0 = 0, count0 = 0;
        double sum1 = 0, count1 = 0;

        for (int i = 0; i < n; i++)
        {
            double prot = NumOps.ToDouble(data[i, _protectedAttributeIndex]);
            double feat = NumOps.ToDouble(data[i, featureIndex]);

            if (prot <= median)
            {
                sum0 += feat;
                count0++;
            }
            else
            {
                sum1 += feat;
                count1++;
            }
        }

        double mean0 = count0 > 0 ? sum0 / count0 : 0;
        double mean1 = count1 > 0 ? sum1 / count1 : 0;

        // Disparate impact ratio (closer to 1 is fairer)
        if (Math.Abs(mean1) < 1e-10 && Math.Abs(mean0) < 1e-10)
            return 1.0;
        if (Math.Abs(mean1) < 1e-10)
            return 0;

        return Math.Min(mean0 / mean1, mean1 / mean0);
    }
}

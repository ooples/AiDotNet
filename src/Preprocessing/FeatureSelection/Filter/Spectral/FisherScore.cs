using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Spectral;

/// <summary>
/// Fisher Score for class-separability based feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Fisher Score measures how well a feature separates different classes.
/// It's the ratio of between-class variance to within-class variance.
/// Higher scores indicate better class discrimination.
/// </para>
/// <para><b>For Beginners:</b> A good feature for classification should have
/// values that are similar within each class but different across classes.
/// Fisher Score captures this by comparing spread between vs within groups.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FisherScore<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _fisherScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FisherScores => _fisherScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public FisherScore(int nFeaturesToSelect = 10, int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FisherScore requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Group samples by class
        var classGroups = new Dictionary<double, List<int>>();
        for (int i = 0; i < n; i++)
        {
            double y = NumOps.ToDouble(target[i]);
            if (!classGroups.ContainsKey(y))
                classGroups[y] = new List<int>();
            classGroups[y].Add(i);
        }

        int nClasses = classGroups.Count;
        if (nClasses < 2)
            throw new ArgumentException("FisherScore requires at least 2 classes.");

        _fisherScores = new double[p];

        // Compute global mean for each feature
        var globalMeans = new double[p];
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                globalMeans[j] += NumOps.ToDouble(data[i, j]);
            globalMeans[j] /= n;
        }

        for (int j = 0; j < p; j++)
        {
            double betweenClassVar = 0;
            double withinClassVar = 0;

            foreach (var kvp in classGroups)
            {
                var classIndices = kvp.Value;
                int nk = classIndices.Count;

                // Compute class mean
                double classMean = 0;
                foreach (int i in classIndices)
                    classMean += NumOps.ToDouble(data[i, j]);
                classMean /= nk;

                // Between-class variance contribution
                betweenClassVar += nk * Math.Pow(classMean - globalMeans[j], 2);

                // Within-class variance contribution
                foreach (int i in classIndices)
                {
                    double diff = NumOps.ToDouble(data[i, j]) - classMean;
                    withinClassVar += diff * diff;
                }
            }

            _fisherScores[j] = withinClassVar > 1e-10 ? betweenClassVar / withinClassVar : 0;
        }

        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _fisherScores
            .Select((fs, idx) => (Score: fs, Index: idx))
            .OrderByDescending(x => x.Score)
            .Take(nToSelect)
            .Select(x => x.Index)
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
            throw new InvalidOperationException("FisherScore has not been fitted.");

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
        throw new NotSupportedException("FisherScore does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FisherScore has not been fitted.");

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

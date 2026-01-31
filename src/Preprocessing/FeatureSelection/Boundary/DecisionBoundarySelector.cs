using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Boundary;

/// <summary>
/// Decision Boundary based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on how clearly they separate classes at the decision
/// boundary, focusing on features that create clean class separations.
/// </para>
/// <para><b>For Beginners:</b> A decision boundary is where the classifier switches
/// from predicting one class to another. Features that create clear, simple boundaries
/// are easier for classifiers to use effectively.
/// </para>
/// </remarks>
public class DecisionBoundarySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _boundaryClarityScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? BoundaryClarityScores => _boundaryClarityScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public DecisionBoundarySelector(
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
            "DecisionBoundarySelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        var classes = y.Distinct().OrderBy(c => c).ToList();
        var classIndices = new Dictionary<int, List<int>>();
        foreach (var c in classes)
            classIndices[c] = Enumerable.Range(0, n).Where(i => y[i] == c).ToList();

        _boundaryClarityScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            // For each pair of classes, find the best threshold and measure clarity
            double totalClarity = 0;
            int pairs = 0;

            for (int ci = 0; ci < classes.Count; ci++)
            {
                for (int cj = ci + 1; cj < classes.Count; cj++)
                {
                    var class1Vals = classIndices[classes[ci]].Select(i => col[i]).ToList();
                    var class2Vals = classIndices[classes[cj]].Select(i => col[i]).ToList();

                    // Find optimal threshold
                    double class1Mean = class1Vals.Average();
                    double class2Mean = class2Vals.Average();
                    double threshold = (class1Mean + class2Mean) / 2;

                    // Count correct classifications
                    int correct = 0;
                    int total = class1Vals.Count + class2Vals.Count;

                    if (class1Mean < class2Mean)
                    {
                        correct += class1Vals.Count(v => v < threshold);
                        correct += class2Vals.Count(v => v >= threshold);
                    }
                    else
                    {
                        correct += class1Vals.Count(v => v >= threshold);
                        correct += class2Vals.Count(v => v < threshold);
                    }

                    // Clarity score = accuracy at boundary
                    double clarity = (double)correct / total;
                    totalClarity += clarity;
                    pairs++;
                }
            }

            _boundaryClarityScores[j] = pairs > 0 ? totalClarity / pairs : 0.5;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _boundaryClarityScores[j])
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
            throw new InvalidOperationException("DecisionBoundarySelector has not been fitted.");

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
        throw new NotSupportedException("DecisionBoundarySelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DecisionBoundarySelector has not been fitted.");

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

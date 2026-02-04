using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Text;

/// <summary>
/// Chi-Square test for text classification feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Applies chi-square test to measure the association between terms and
/// document classes. Terms with high chi-square values are more discriminative.
/// </para>
/// <para><b>For Beginners:</b> This test finds words that appear significantly
/// more often in one class than expected by chance. These discriminative words
/// are the best features for text classification.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ChiSquareText<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly bool _binarize;

    private double[]? _chiSquareScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? ChiSquareScores => _chiSquareScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ChiSquareText(
        int nFeaturesToSelect = 100,
        bool binarize = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _binarize = binarize;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ChiSquareText requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Get unique classes
        var classes = new HashSet<double>();
        for (int i = 0; i < n; i++)
            classes.Add(NumOps.ToDouble(target[i]));
        var classList = classes.OrderBy(x => x).ToList();
        int nClasses = classList.Count;

        _chiSquareScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            double chiSq = 0;

            foreach (double c in classList)
            {
                // Contingency table:
                // a = term present, class = c
                // b = term present, class != c
                // c = term absent, class = c
                // d = term absent, class != c

                int a = 0, b = 0, tc = 0, d = 0;

                for (int i = 0; i < n; i++)
                {
                    double value = NumOps.ToDouble(data[i, j]);
                    bool termPresent = _binarize ? value > 0 : value > 0.5;
                    bool isClass = Math.Abs(NumOps.ToDouble(target[i]) - c) < 1e-10;

                    if (termPresent && isClass) a++;
                    else if (termPresent && !isClass) b++;
                    else if (!termPresent && isClass) tc++;
                    else d++;
                }

                // Chi-square contribution for this class
                int n1 = a + b;  // term present
                int n0 = tc + d;  // term absent
                int c1 = a + tc;  // class = c
                int c0 = b + d;   // class != c

                // Expected values
                double eA = (double)n1 * c1 / n;
                double eB = (double)n1 * c0 / n;
                double eC = (double)n0 * c1 / n;
                double eD = (double)n0 * c0 / n;

                if (eA > 0) chiSq += Math.Pow(a - eA, 2) / eA;
                if (eB > 0) chiSq += Math.Pow(b - eB, 2) / eB;
                if (eC > 0) chiSq += Math.Pow(tc - eC, 2) / eC;
                if (eD > 0) chiSq += Math.Pow(d - eD, 2) / eD;
            }

            _chiSquareScores[j] = chiSq / nClasses;  // Average across classes
        }

        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _chiSquareScores
            .Select((s, idx) => (Score: s, Index: idx))
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
            throw new InvalidOperationException("ChiSquareText has not been fitted.");

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
        throw new NotSupportedException("ChiSquareText does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ChiSquareText has not been fitted.");

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

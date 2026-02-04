using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Forward Selection (Sequential Forward Selection) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Starts with an empty set and greedily adds one feature at a time that provides
/// the best improvement until the desired number of features is reached.
/// </para>
/// <para><b>For Beginners:</b> Forward Selection is like building a team one person
/// at a time. At each step, you try adding each remaining candidate and keep the one
/// who helps the most. You continue until your team is the desired size. Simple but
/// can miss feature interactions that only appear together.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ForwardSelection<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly Func<Matrix<T>, Vector<T>, int[], double>? _scoringFunction;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ForwardSelection(
        int nFeaturesToSelect = 10,
        Func<Matrix<T>, Vector<T>, int[], double>? scoringFunction = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _scoringFunction = scoringFunction;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ForwardSelection requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;
        int nToSelect = Math.Min(_nFeaturesToSelect, p);

        var selected = new List<int>();
        var remaining = Enumerable.Range(0, p).ToHashSet();
        _featureScores = new double[p];

        while (selected.Count < nToSelect && remaining.Count > 0)
        {
            int bestFeature = -1;
            double bestScore = double.MinValue;

            foreach (int candidate in remaining)
            {
                var testSet = selected.Concat(new[] { candidate }).ToArray();
                double score = EvaluateSubset(data, target, testSet);

                if (score > bestScore)
                {
                    bestScore = score;
                    bestFeature = candidate;
                }
            }

            if (bestFeature >= 0)
            {
                selected.Add(bestFeature);
                remaining.Remove(bestFeature);
                _featureScores[bestFeature] = bestScore;
            }
            else
            {
                break;
            }
        }

        _selectedIndices = selected.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private double EvaluateSubset(Matrix<T> data, Vector<T> target, int[] features)
    {
        if (_scoringFunction is not null)
            return _scoringFunction(data, target, features);

        return DefaultScoring(data, target, features);
    }

    private double DefaultScoring(Matrix<T> data, Vector<T> target, int[] features)
    {
        // Use R-squared as default scoring
        int n = data.Rows;
        int p = features.Length;

        if (p == 0 || n <= p)
            return 0;

        // Simple linear regression R-squared
        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        // Use correlation-based approximation for speed
        double totalCorr = 0;
        foreach (int j in features)
        {
            double xMean = 0;
            for (int i = 0; i < n; i++)
                xMean += NumOps.ToDouble(data[i, j]);
            xMean /= n;

            double ssXY = 0, ssXX = 0, ssYY = 0;
            for (int i = 0; i < n; i++)
            {
                double dx = NumOps.ToDouble(data[i, j]) - xMean;
                double dy = NumOps.ToDouble(target[i]) - yMean;
                ssXY += dx * dy;
                ssXX += dx * dx;
                ssYY += dy * dy;
            }

            if (ssXX > 1e-10 && ssYY > 1e-10)
            {
                double corr = ssXY / Math.Sqrt(ssXX * ssYY);
                totalCorr += corr * corr;
            }
        }

        return totalCorr;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ForwardSelection has not been fitted.");

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
        throw new NotSupportedException("ForwardSelection does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ForwardSelection has not been fitted.");

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

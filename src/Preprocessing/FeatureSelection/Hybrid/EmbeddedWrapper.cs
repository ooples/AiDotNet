using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Hybrid;

/// <summary>
/// Embedded-Wrapper hybrid feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Embedded-Wrapper combines embedded methods (like regularization) with wrapper
/// search. It uses embedded feature importance to guide the wrapper search,
/// making the search more efficient.
/// </para>
/// <para><b>For Beginners:</b> This is like having an expert advisor (embedded method)
/// suggest which features might be important, then carefully verifying those
/// suggestions (wrapper). The advisor speeds up the search while the verification
/// ensures accuracy.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class EmbeddedWrapper<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _lambda;
    private readonly int _maxWrapperIterations;

    private double[]? _embeddedImportance;
    private double[]? _wrapperScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Lambda => _lambda;
    public int MaxWrapperIterations => _maxWrapperIterations;
    public double[]? EmbeddedImportance => _embeddedImportance;
    public double[]? WrapperScores => _wrapperScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public EmbeddedWrapper(
        int nFeaturesToSelect = 10,
        double lambda = 0.1,
        int maxWrapperIterations = 50,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (lambda < 0)
            throw new ArgumentException("Lambda must be non-negative.", nameof(lambda));

        _nFeaturesToSelect = nFeaturesToSelect;
        _lambda = lambda;
        _maxWrapperIterations = maxWrapperIterations;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "EmbeddedWrapper requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Stage 1: Embedded - compute L1-regularized importance
        _embeddedImportance = ComputeL1Importance(data, target);

        // Get initial candidates from embedded importance
        var candidates = _embeddedImportance
            .Select((imp, idx) => (Importance: imp, Index: idx))
            .OrderByDescending(x => x.Importance)
            .Take(Math.Min(_nFeaturesToSelect * 2, p))
            .Select(x => x.Index)
            .ToHashSet();

        // Stage 2: Wrapper - refine selection with guided search
        _wrapperScores = new double[p];
        var selected = GuidedForwardSelection(data, target, candidates);

        _selectedIndices = selected.OrderBy(x => x).ToArray();

        IsFitted = true;
    }

    private double[] ComputeL1Importance(Matrix<T> data, Vector<T> target)
    {
        int n = data.Rows;
        int p = data.Columns;

        // Standardize data
        var means = new double[p];
        var stds = new double[p];
        var X = new double[n, p];

        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                means[j] += NumOps.ToDouble(data[i, j]);
            means[j] /= n;

            for (int i = 0; i < n; i++)
            {
                double diff = NumOps.ToDouble(data[i, j]) - means[j];
                stds[j] += diff * diff;
            }
            stds[j] = Math.Sqrt(stds[j] / n);
            if (stds[j] < 1e-10) stds[j] = 1;

            for (int i = 0; i < n; i++)
                X[i, j] = (NumOps.ToDouble(data[i, j]) - means[j]) / stds[j];
        }

        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        var y = new double[n];
        for (int i = 0; i < n; i++)
            y[i] = NumOps.ToDouble(target[i]) - yMean;

        // Coordinate descent LASSO
        var coefficients = new double[p];
        const int maxIter = 100;

        for (int iter = 0; iter < maxIter; iter++)
        {
            for (int j = 0; j < p; j++)
            {
                double rho = 0;
                for (int i = 0; i < n; i++)
                {
                    double residual = y[i];
                    for (int k = 0; k < p; k++)
                    {
                        if (k != j)
                            residual -= X[i, k] * coefficients[k];
                    }
                    rho += X[i, j] * residual;
                }
                rho /= n;

                // Soft threshold
                coefficients[j] = SoftThreshold(rho, _lambda);
            }
        }

        return coefficients.Select(c => Math.Abs(c)).ToArray();
    }

    private double SoftThreshold(double z, double lambda)
    {
        if (z > lambda) return z - lambda;
        if (z < -lambda) return z + lambda;
        return 0;
    }

    private HashSet<int> GuidedForwardSelection(Matrix<T> data, Vector<T> target, HashSet<int> candidates)
    {
        var selected = new HashSet<int>();
        var available = new HashSet<int>(candidates);
        double currentScore = 0;

        // Order candidates by embedded importance for guided search
        var orderedCandidates = available
            .OrderByDescending(j => _embeddedImportance![j])
            .ToList();

        int iterations = 0;
        while (selected.Count < _nFeaturesToSelect &&
               available.Count > 0 &&
               iterations < _maxWrapperIterations)
        {
            int bestFeature = -1;
            double bestScore = currentScore;

            // Prioritize candidates with higher embedded importance
            foreach (int j in orderedCandidates)
            {
                if (!available.Contains(j)) continue;

                selected.Add(j);
                double score = EvaluateSubset(data, target, selected);
                selected.Remove(j);

                if (score > bestScore)
                {
                    bestScore = score;
                    bestFeature = j;
                }
            }

            if (bestFeature >= 0)
            {
                selected.Add(bestFeature);
                available.Remove(bestFeature);
                currentScore = bestScore;
                _wrapperScores![bestFeature] = bestScore;
            }
            else
            {
                break;
            }

            iterations++;
        }

        return selected;
    }

    private double EvaluateSubset(Matrix<T> data, Vector<T> target, HashSet<int> subset)
    {
        if (subset.Count == 0) return 0;

        int n = data.Rows;

        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        // Weighted sum of embedded importance and correlation
        double score = 0;
        foreach (int j in subset)
        {
            double corr = ComputeCorrelation(data, target, j, n, yMean);
            double importance = _embeddedImportance![j];
            score += corr * 0.5 + importance * 0.5;
        }

        return score / subset.Count;
    }

    private double ComputeCorrelation(Matrix<T> data, Vector<T> target, int j, int n, double yMean)
    {
        double xMean = 0;
        for (int i = 0; i < n; i++)
            xMean += NumOps.ToDouble(data[i, j]);
        xMean /= n;

        double sxy = 0, sxx = 0, syy = 0;
        for (int i = 0; i < n; i++)
        {
            double xDiff = NumOps.ToDouble(data[i, j]) - xMean;
            double yDiff = NumOps.ToDouble(target[i]) - yMean;
            sxy += xDiff * yDiff;
            sxx += xDiff * xDiff;
            syy += yDiff * yDiff;
        }

        return (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("EmbeddedWrapper has not been fitted.");

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
        throw new NotSupportedException("EmbeddedWrapper does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("EmbeddedWrapper has not been fitted.");

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

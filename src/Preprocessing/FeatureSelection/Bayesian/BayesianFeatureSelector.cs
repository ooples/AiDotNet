using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Bayesian;

/// <summary>
/// Bayesian Feature Selection using posterior probability.
/// </summary>
/// <remarks>
/// <para>
/// Uses Bayesian inference to compute the posterior probability that each
/// feature is relevant to the target. Features with high posterior probability
/// are selected.
/// </para>
/// <para><b>For Beginners:</b> Bayesian methods treat feature relevance as a
/// probability question. Instead of saying "feature X is important" or "not important",
/// we say "there's a 90% chance feature X is important." This gives us uncertainty
/// estimates and allows us to incorporate prior knowledge about which features
/// might be relevant.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class BayesianFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _priorProbability;
    private readonly double _threshold;

    private double[]? _posteriorProbabilities;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double PriorProbability => _priorProbability;
    public double[]? PosteriorProbabilities => _posteriorProbabilities;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public BayesianFeatureSelector(
        int nFeaturesToSelect = 10,
        double priorProbability = 0.5,
        double threshold = 0.5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (priorProbability <= 0 || priorProbability >= 1)
            throw new ArgumentException("Prior probability must be between 0 and 1 exclusive.", nameof(priorProbability));

        _nFeaturesToSelect = nFeaturesToSelect;
        _priorProbability = priorProbability;
        _threshold = threshold;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BayesianFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        _posteriorProbabilities = new double[p];

        // Compute likelihood ratio for each feature
        var targetArray = new double[n];
        for (int i = 0; i < n; i++)
            targetArray[i] = NumOps.ToDouble(target[i]);

        double positiveCount = targetArray.Count(y => y > 0.5);
        double negativeCount = n - positiveCount;

        for (int j = 0; j < p; j++)
        {
            // Compute feature statistics for positive and negative classes
            double sumPos = 0, sumNeg = 0;
            double sumSqPos = 0, sumSqNeg = 0;
            int countPos = 0, countNeg = 0;

            for (int i = 0; i < n; i++)
            {
                double value = NumOps.ToDouble(data[i, j]);
                if (targetArray[i] > 0.5)
                {
                    sumPos += value;
                    sumSqPos += value * value;
                    countPos++;
                }
                else
                {
                    sumNeg += value;
                    sumSqNeg += value * value;
                    countNeg++;
                }
            }

            if (countPos < 2 || countNeg < 2)
            {
                _posteriorProbabilities[j] = _priorProbability;
                continue;
            }

            double meanPos = sumPos / countPos;
            double meanNeg = sumNeg / countNeg;
            double varPos = (sumSqPos / countPos) - (meanPos * meanPos) + 1e-10;
            double varNeg = (sumSqNeg / countNeg) - (meanNeg * meanNeg) + 1e-10;

            // Compute Bayes factor using t-test approximation
            double pooledVar = ((countPos - 1) * varPos + (countNeg - 1) * varNeg) / (n - 2);
            double standardError = Math.Sqrt(pooledVar * (1.0 / countPos + 1.0 / countNeg)) + 1e-10;
            double tStatistic = Math.Abs(meanPos - meanNeg) / standardError;

            // Convert t-statistic to likelihood ratio approximation
            double likelihoodRatio = Math.Exp(tStatistic * tStatistic / 2);

            // Compute posterior using Bayes' rule
            double posteriorOdds = (_priorProbability / (1 - _priorProbability)) * likelihoodRatio;
            _posteriorProbabilities[j] = posteriorOdds / (1 + posteriorOdds);
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _posteriorProbabilities[j])
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
            throw new InvalidOperationException("BayesianFeatureSelector has not been fitted.");

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
        throw new NotSupportedException("BayesianFeatureSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BayesianFeatureSelector has not been fitted.");

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

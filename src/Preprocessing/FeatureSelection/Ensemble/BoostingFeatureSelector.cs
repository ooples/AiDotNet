using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Ensemble;

/// <summary>
/// Boosting-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses a boosting-inspired approach where samples are reweighted based on
/// prediction errors, focusing feature selection on harder-to-predict samples.
/// </para>
/// <para><b>For Beginners:</b> Like gradient boosting, this method focuses more
/// attention on the samples that are hardest to predict. Features that help
/// predict these difficult samples get higher importance scores, leading to
/// more discriminative feature selection.
/// </para>
/// </remarks>
public class BoostingFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nRounds;
    private readonly double _learningRate;

    private double[]? _boostingScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NRounds => _nRounds;
    public double[]? BoostingScores => _boostingScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public BoostingFeatureSelector(
        int nFeaturesToSelect = 10,
        int nRounds = 10,
        double learningRate = 0.1,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nRounds = nRounds;
        _learningRate = learningRate;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BoostingFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

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

        // Initialize sample weights uniformly
        var weights = new double[n];
        for (int i = 0; i < n; i++)
            weights[i] = 1.0 / n;

        _boostingScores = new double[p];
        var predictions = new double[n];

        for (int round = 0; round < _nRounds; round++)
        {
            // Find best feature for this round (weighted correlation)
            int bestFeature = -1;
            double bestScore = double.MinValue;
            double[] bestPred = new double[n];

            for (int j = 0; j < p; j++)
            {
                // Weighted mean
                double wxSum = 0, wySum = 0, wSum = 0;
                for (int i = 0; i < n; i++)
                {
                    wxSum += weights[i] * X[i, j];
                    wySum += weights[i] * y[i];
                    wSum += weights[i];
                }
                double xMean = wxSum / wSum;
                double yMean = wySum / wSum;

                // Weighted regression coefficient
                double sxy = 0, sxx = 0;
                for (int i = 0; i < n; i++)
                {
                    double xd = X[i, j] - xMean;
                    double yd = y[i] - yMean;
                    sxy += weights[i] * xd * yd;
                    sxx += weights[i] * xd * xd;
                }

                double beta = sxx > 1e-10 ? sxy / sxx : 0;

                // Compute predictions and weighted error
                var pred = new double[n];
                double weightedError = 0;
                for (int i = 0; i < n; i++)
                {
                    pred[i] = yMean + beta * (X[i, j] - xMean);
                    double err = (y[i] - pred[i]) * (y[i] - pred[i]);
                    weightedError += weights[i] * err;
                }

                double score = -weightedError;
                if (score > bestScore)
                {
                    bestScore = score;
                    bestFeature = j;
                    bestPred = pred;
                }
            }

            if (bestFeature >= 0)
            {
                _boostingScores[bestFeature] += 1.0 / (_nRounds - round);

                // Update predictions
                for (int i = 0; i < n; i++)
                    predictions[i] += _learningRate * (bestPred[i] - predictions[i]);

                // Update weights based on residuals
                double maxResidual = 0;
                for (int i = 0; i < n; i++)
                {
                    double residual = Math.Abs(y[i] - predictions[i]);
                    maxResidual = Math.Max(maxResidual, residual);
                }

                if (maxResidual > 1e-10)
                {
                    double weightSum = 0;
                    for (int i = 0; i < n; i++)
                    {
                        double residual = Math.Abs(y[i] - predictions[i]);
                        weights[i] = residual / maxResidual + 0.01;
                        weightSum += weights[i];
                    }
                    for (int i = 0; i < n; i++)
                        weights[i] /= weightSum;
                }
            }
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _boostingScores[j])
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
            throw new InvalidOperationException("BoostingFeatureSelector has not been fitted.");

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
        throw new NotSupportedException("BoostingFeatureSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BoostingFeatureSelector has not been fitted.");

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

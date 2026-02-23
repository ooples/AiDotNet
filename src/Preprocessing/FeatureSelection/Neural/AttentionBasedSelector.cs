using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Neural;

/// <summary>
/// Attention-based Feature Selection using neural attention mechanisms.
/// </summary>
/// <remarks>
/// <para>
/// Uses an attention mechanism to learn feature importance weights.
/// The attention scores indicate which features the model focuses on
/// when making predictions.
/// </para>
/// <para><b>For Beginners:</b> Attention is like a spotlight that tells
/// the model where to focus. Features that get more "spotlight" (higher
/// attention) are more important for making predictions. This method
/// learns those importance weights automatically from the data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class AttentionBasedSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _hiddenSize;
    private readonly int _nEpochs;
    private readonly double _learningRate;

    private double[]? _attentionScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? AttentionScores => _attentionScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public AttentionBasedSelector(
        int nFeaturesToSelect = 10,
        int hiddenSize = 32,
        int nEpochs = 100,
        double learningRate = 0.01,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _hiddenSize = hiddenSize;
        _nEpochs = nEpochs;
        _learningRate = learningRate;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "AttentionBasedSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var rand = RandomHelper.CreateSecureRandom();

        // Convert and normalize data
        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }
        NormalizeData(X, n, p);

        // Initialize attention network
        int hidden = Math.Min(_hiddenSize, p);
        var W1 = InitializeWeights(p, hidden, rand);
        var W2 = InitializeWeights(hidden, 1, rand);
        var Wpred = InitializeWeights(p, 1, rand);

        // Train with attention mechanism
        for (int epoch = 0; epoch < _nEpochs; epoch++)
        {
            for (int i = 0; i < n; i++)
            {
                // Compute attention scores
                var h = new double[hidden];
                for (int j = 0; j < hidden; j++)
                {
                    for (int k = 0; k < p; k++)
                        h[j] += X[i, k] * W1[k, j];
                    h[j] = Math.Tanh(h[j]);
                }

                var scores = new double[p];
                for (int j = 0; j < p; j++)
                {
                    for (int k = 0; k < hidden; k++)
                        scores[j] += h[k] * W2[k, 0] * W1[j, k];
                }

                // Softmax
                double maxScore = scores.Max();
                for (int j = 0; j < p; j++)
                    scores[j] = Math.Exp(scores[j] - maxScore);
                double sumScores = scores.Sum();
                for (int j = 0; j < p; j++)
                    scores[j] /= sumScores;

                // Weighted prediction
                double pred = 0;
                for (int j = 0; j < p; j++)
                    pred += scores[j] * X[i, j] * Wpred[j, 0];

                // Loss and gradients
                double error = pred - y[i];

                // Update weights (simplified gradient descent)
                for (int j = 0; j < p; j++)
                    Wpred[j, 0] -= _learningRate * error * scores[j] * X[i, j];

                for (int j = 0; j < p; j++)
                {
                    for (int k = 0; k < hidden; k++)
                    {
                        double grad = error * scores[j] * (1 - scores[j]) * h[k];
                        W1[j, k] -= _learningRate * grad * 0.01;
                    }
                }
            }
        }

        // Final attention scores
        _attentionScores = new double[p];
        for (int i = 0; i < n; i++)
        {
            var h = new double[hidden];
            for (int j = 0; j < hidden; j++)
            {
                for (int k = 0; k < p; k++)
                    h[j] += X[i, k] * W1[k, j];
                h[j] = Math.Tanh(h[j]);
            }

            var scores = new double[p];
            for (int j = 0; j < p; j++)
            {
                for (int k = 0; k < hidden; k++)
                    scores[j] += h[k] * W2[k, 0] * W1[j, k];
            }

            double maxScore = scores.Max();
            for (int j = 0; j < p; j++)
                scores[j] = Math.Exp(scores[j] - maxScore);
            double sumScores = scores.Sum();
            for (int j = 0; j < p; j++)
                _attentionScores[j] += scores[j] / sumScores;
        }

        for (int j = 0; j < p; j++)
            _attentionScores[j] /= n;

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _attentionScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private void NormalizeData(double[,] X, int n, int p)
    {
        for (int j = 0; j < p; j++)
        {
            double mean = 0, std = 0;
            for (int i = 0; i < n; i++)
                mean += X[i, j];
            mean /= n;

            for (int i = 0; i < n; i++)
                std += (X[i, j] - mean) * (X[i, j] - mean);
            std = Math.Sqrt(std / (n - 1)) + 1e-10;

            for (int i = 0; i < n; i++)
                X[i, j] = (X[i, j] - mean) / std;
        }
    }

    private double[,] InitializeWeights(int rows, int cols, Random rand)
    {
        var weights = new double[rows, cols];
        double scale = Math.Sqrt(2.0 / (rows + cols));
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                weights[i, j] = (rand.NextDouble() * 2 - 1) * scale;
        return weights;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AttentionBasedSelector has not been fitted.");

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
        throw new NotSupportedException("AttentionBasedSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AttentionBasedSelector has not been fitted.");

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

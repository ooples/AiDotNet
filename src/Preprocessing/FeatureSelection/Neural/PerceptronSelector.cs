using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Neural;

/// <summary>
/// Perceptron based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their weight magnitudes in a trained single-layer
/// perceptron, identifying features with the strongest learned connections.
/// </para>
/// <para><b>For Beginners:</b> A perceptron is the simplest neural network - just
/// one layer of weights connecting inputs to output. Features with larger weights
/// (positive or negative) are more important for the prediction.
/// </para>
/// </remarks>
public class PerceptronSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _epochs;
    private readonly double _learningRate;
    private readonly int? _randomState;

    private double[]? _weightImportance;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int Epochs => _epochs;
    public double LearningRate => _learningRate;
    public double[]? WeightImportance => _weightImportance;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public PerceptronSelector(
        int nFeaturesToSelect = 10,
        int epochs = 100,
        double learningRate = 0.01,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _epochs = epochs;
        _learningRate = learningRate;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "PerceptronSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Standardize data
        var means = new double[p];
        var stds = new double[p];
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                means[j] += X[i, j];
            means[j] /= n;

            for (int i = 0; i < n; i++)
                stds[j] += (X[i, j] - means[j]) * (X[i, j] - means[j]);
            stds[j] = Math.Sqrt(stds[j] / n);
            if (stds[j] < 1e-10) stds[j] = 1;
        }

        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] = (X[i, j] - means[j]) / stds[j];

        // Standardize target
        double yMean = y.Average();
        double yStd = Math.Sqrt(y.Select(v => (v - yMean) * (v - yMean)).Average());
        if (yStd < 1e-10) yStd = 1;
        for (int i = 0; i < n; i++)
            y[i] = (y[i] - yMean) / yStd;

        // Initialize weights
        var weights = new double[p];
        double bias = 0;
        double scale = Math.Sqrt(1.0 / p);
        for (int j = 0; j < p; j++)
            weights[j] = (rand.NextDouble() - 0.5) * 2 * scale;

        // Train perceptron (gradient descent)
        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            for (int i = 0; i < n; i++)
            {
                // Forward pass
                double prediction = bias;
                for (int j = 0; j < p; j++)
                    prediction += X[i, j] * weights[j];

                // Error
                double error = prediction - y[i];

                // Update weights
                for (int j = 0; j < p; j++)
                    weights[j] -= _learningRate * error * X[i, j];
                bias -= _learningRate * error;
            }
        }

        // Feature importance based on absolute weight
        _weightImportance = weights.Select(Math.Abs).ToArray();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _weightImportance[j])
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
            throw new InvalidOperationException("PerceptronSelector has not been fitted.");

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
        throw new NotSupportedException("PerceptronSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PerceptronSelector has not been fitted.");

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

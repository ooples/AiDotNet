using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Neural;

/// <summary>
/// Deep Feature Selection using a multi-layer neural network.
/// </summary>
/// <remarks>
/// <para>
/// Uses a deep neural network with L1 regularization on the first layer
/// to perform feature selection while learning non-linear relationships.
/// </para>
/// <para><b>For Beginners:</b> This method trains a multi-layer neural network
/// but penalizes the input layer weights. Features with larger input weights
/// after training are considered more important. The network can capture
/// complex non-linear patterns that simpler methods might miss.
/// </para>
/// </remarks>
public class DeepFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int[] _hiddenLayers;
    private readonly int _nEpochs;
    private readonly double _l1Penalty;

    private double[]? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureImportances => _featureImportances;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public DeepFeatureSelector(
        int nFeaturesToSelect = 10,
        int[]? hiddenLayers = null,
        int nEpochs = 100,
        double l1Penalty = 0.01,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _hiddenLayers = hiddenLayers ?? [64, 32];
        _nEpochs = nEpochs;
        _l1Penalty = l1Penalty;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "DeepFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var rand = RandomHelper.CreateSecureRandom();
        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Normalize
        var means = new double[p];
        var stds = new double[p];
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++) means[j] += X[i, j];
            means[j] /= n;
            for (int i = 0; i < n; i++) stds[j] += (X[i, j] - means[j]) * (X[i, j] - means[j]);
            stds[j] = Math.Sqrt(stds[j] / (n - 1)) + 1e-10;
            for (int i = 0; i < n; i++) X[i, j] = (X[i, j] - means[j]) / stds[j];
        }

        // Build layer sizes
        var layerSizes = new List<int> { p };
        foreach (int h in _hiddenLayers)
            layerSizes.Add(Math.Min(h, p));
        layerSizes.Add(1); // Output layer

        // Initialize weights
        var weights = new List<double[,]>();
        var biases = new List<double[]>();
        for (int l = 0; l < layerSizes.Count - 1; l++)
        {
            int inSize = layerSizes[l];
            int outSize = layerSizes[l + 1];
            double scale = Math.Sqrt(2.0 / inSize);

            var W = new double[inSize, outSize];
            var b = new double[outSize];
            for (int i = 0; i < inSize; i++)
                for (int j = 0; j < outSize; j++)
                    W[i, j] = (rand.NextDouble() * 2 - 1) * scale;
            weights.Add(W);
            biases.Add(b);
        }

        double lr = 0.001;

        for (int epoch = 0; epoch < _nEpochs; epoch++)
        {
            for (int i = 0; i < n; i++)
            {
                // Forward pass
                var activations = new List<double[]>();
                var preActivations = new List<double[]>();

                var input = new double[p];
                for (int j = 0; j < p; j++) input[j] = X[i, j];
                activations.Add(input);

                for (int l = 0; l < weights.Count; l++)
                {
                    int inSize = layerSizes[l];
                    int outSize = layerSizes[l + 1];
                    var pre = new double[outSize];
                    var act = new double[outSize];

                    for (int j = 0; j < outSize; j++)
                    {
                        pre[j] = biases[l][j];
                        for (int k = 0; k < inSize; k++)
                            pre[j] += activations[l][k] * weights[l][k, j];

                        // ReLU for hidden, linear for output
                        act[j] = (l < weights.Count - 1) ? Math.Max(0, pre[j]) : pre[j];
                    }
                    preActivations.Add(pre);
                    activations.Add(act);
                }

                // Compute loss gradient
                double pred = activations[^1][0];
                double error = pred - y[i];

                // Backward pass
                var deltas = new List<double[]>();
                deltas.Add(new double[] { error });

                for (int l = weights.Count - 2; l >= 0; l--)
                {
                    int outSize = layerSizes[l + 1];
                    int nextSize = layerSizes[l + 2];
                    var delta = new double[outSize];

                    for (int j = 0; j < outSize; j++)
                    {
                        double grad = 0;
                        for (int k = 0; k < nextSize; k++)
                            grad += deltas[0][k] * weights[l + 1][j, k];

                        // ReLU derivative
                        delta[j] = grad * (preActivations[l][j] > 0 ? 1 : 0);
                    }
                    deltas.Insert(0, delta);
                }

                // Update weights
                for (int l = 0; l < weights.Count; l++)
                {
                    int inSize = layerSizes[l];
                    int outSize = layerSizes[l + 1];

                    for (int j = 0; j < inSize; j++)
                    {
                        for (int k = 0; k < outSize; k++)
                        {
                            double grad = deltas[l][k] * activations[l][j];

                            // L1 regularization on first layer only
                            if (l == 0)
                                grad += _l1Penalty * Math.Sign(weights[l][j, k]);

                            weights[l][j, k] -= lr * grad;
                        }
                    }

                    for (int k = 0; k < outSize; k++)
                        biases[l][k] -= lr * deltas[l][k];
                }
            }
        }

        // Compute feature importance from first layer weights
        _featureImportances = new double[p];
        int firstLayerOut = layerSizes[1];
        for (int j = 0; j < p; j++)
            for (int k = 0; k < firstLayerOut; k++)
                _featureImportances[j] += Math.Abs(weights[0][j, k]);

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _featureImportances[j])
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
            throw new InvalidOperationException("DeepFeatureSelector has not been fitted.");

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
        throw new NotSupportedException("DeepFeatureSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DeepFeatureSelector has not been fitted.");

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

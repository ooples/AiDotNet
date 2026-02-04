using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Neural;

/// <summary>
/// Concrete Autoencoder Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses the Concrete Autoencoder which learns a discrete feature selection
/// mask using the Concrete/Gumbel-Softmax distribution for differentiable
/// feature selection.
/// </para>
/// <para><b>For Beginners:</b> The Concrete Autoencoder learns which features
/// to select as part of its training. It uses a special trick (Gumbel-Softmax)
/// that lets it learn a "soft" selection that gradually becomes "hard" (0 or 1)
/// as training progresses.
/// </para>
/// </remarks>
public class ConcreteAutoencoderSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _hiddenSize;
    private readonly int _nEpochs;
    private readonly double _initialTemperature;
    private readonly double _minTemperature;

    private double[]? _selectionProbabilities;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? SelectionProbabilities => _selectionProbabilities;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ConcreteAutoencoderSelector(
        int nFeaturesToSelect = 10,
        int hiddenSize = 32,
        int nEpochs = 200,
        double initialTemperature = 10.0,
        double minTemperature = 0.1,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _hiddenSize = hiddenSize;
        _nEpochs = nEpochs;
        _initialTemperature = initialTemperature;
        _minTemperature = minTemperature;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

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

        int k = Math.Min(_nFeaturesToSelect, p);
        int h = Math.Min(_hiddenSize, p);
        var rand = RandomHelper.CreateSecureRandom();

        // Selection logits (one for each feature, for each of k selections)
        var logits = new double[k, p];
        for (int s = 0; s < k; s++)
            for (int j = 0; j < p; j++)
                logits[s, j] = (rand.NextDouble() * 2 - 1) * 0.1;

        // Decoder weights
        var W2 = new double[k, p];
        double scale = Math.Sqrt(2.0 / (k + p));
        for (int i = 0; i < k; i++)
            for (int j = 0; j < p; j++)
                W2[i, j] = (rand.NextDouble() * 2 - 1) * scale;

        double lr = 0.01;

        for (int epoch = 0; epoch < _nEpochs; epoch++)
        {
            // Anneal temperature
            double temp = _initialTemperature * Math.Pow(_minTemperature / _initialTemperature, (double)epoch / _nEpochs);

            for (int i = 0; i < n; i++)
            {
                // Sample Concrete/Gumbel-Softmax selections
                var selections = new double[k, p];
                var selectedValues = new double[k];

                for (int s = 0; s < k; s++)
                {
                    // Gumbel-Softmax
                    var gumbels = new double[p];
                    double maxLogit = double.MinValue;
                    for (int j = 0; j < p; j++)
                    {
                        double u = rand.NextDouble();
                        gumbels[j] = logits[s, j] - Math.Log(-Math.Log(u + 1e-20) + 1e-20);
                        maxLogit = Math.Max(maxLogit, gumbels[j]);
                    }

                    // Softmax with temperature
                    double sumExp = 0;
                    for (int j = 0; j < p; j++)
                    {
                        selections[s, j] = Math.Exp((gumbels[j] - maxLogit) / temp);
                        sumExp += selections[s, j];
                    }
                    for (int j = 0; j < p; j++)
                    {
                        selections[s, j] /= sumExp;
                        selectedValues[s] += selections[s, j] * X[i, j];
                    }
                }

                // Decode
                var output = new double[p];
                for (int j = 0; j < p; j++)
                    for (int s = 0; s < k; s++)
                        output[j] += selectedValues[s] * W2[s, j];

                // Reconstruction error
                var error = new double[p];
                for (int j = 0; j < p; j++)
                    error[j] = output[j] - X[i, j];

                // Gradient for decoder
                for (int s = 0; s < k; s++)
                    for (int j = 0; j < p; j++)
                        W2[s, j] -= lr * error[j] * selectedValues[s];

                // Gradient for selection logits
                var gradSelected = new double[k];
                for (int s = 0; s < k; s++)
                    for (int j = 0; j < p; j++)
                        gradSelected[s] += error[j] * W2[s, j];

                for (int s = 0; s < k; s++)
                {
                    for (int j = 0; j < p; j++)
                    {
                        // Gradient of Gumbel-Softmax
                        double grad = gradSelected[s] * X[i, j] * selections[s, j] * (1 - selections[s, j]) / temp;
                        logits[s, j] -= lr * grad;
                    }
                }
            }
        }

        // Compute selection probabilities from final logits
        _selectionProbabilities = new double[p];
        for (int s = 0; s < k; s++)
        {
            double maxLogit = logits[s, 0];
            for (int j = 1; j < p; j++) maxLogit = Math.Max(maxLogit, logits[s, j]);

            double sumExp = 0;
            var probs = new double[p];
            for (int j = 0; j < p; j++)
            {
                probs[j] = Math.Exp(logits[s, j] - maxLogit);
                sumExp += probs[j];
            }
            for (int j = 0; j < p; j++)
                _selectionProbabilities[j] += probs[j] / sumExp;
        }

        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _selectionProbabilities[j])
            .Take(k)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ConcreteAutoencoderSelector has not been fitted.");

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
        throw new NotSupportedException("ConcreteAutoencoderSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ConcreteAutoencoderSelector has not been fitted.");

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

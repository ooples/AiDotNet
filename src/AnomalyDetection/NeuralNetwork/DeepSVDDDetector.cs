using AiDotNet.Tensors.Engines;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.NeuralNetwork;

/// <summary>
/// Detects anomalies using Deep SVDD (Support Vector Data Description).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Deep SVDD trains a neural network to map normal data points close to
/// a hypersphere center in the output space. Anomalies are points that map far from this center.
/// It combines deep learning with the classic SVDD concept.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Initialize network and compute center from initial encodings
/// 2. Train network to minimize distance of normal points to center
/// 3. Anomaly score is the distance to the center
/// </para>
/// <para>
/// <b>When to use:</b>
/// - One-class classification with deep learning
/// - When you have only normal examples for training
/// - Complex, high-dimensional data
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Hidden dimensions: 64
/// - Output dimensions: 32
/// - Learning rate: 0.001
/// - Epochs: 100
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Ruff, L., et al. (2018). "Deep One-Class Classification." ICML.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.SVM)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Deep One-Class Classification", "https://doi.org/10.48550/arXiv.1802.04365", Year = 2018, Authors = "Lukas Ruff, Robert A. Vandermeulen, Nico Görnitz, Lucas Deecke, Shoaib A. Siddiqui, Alexander Binder, Emmanuel Müller, Marius Kloft")]
public partial class DeepSVDDDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _hiddenDim;
    private readonly int _outputDim;
    private readonly int _epochs;
    private readonly double _learningRate;

    // Network weights
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _w1;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _b1;
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _w2;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _b2;
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _w3;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _b3;

    // Hypersphere center
    [Buffer]
    private Vector<T>? _center;
    private int _inputDim;

    // Normalization parameters
    [Buffer]
    private Vector<T>? _dataMeans;
    [Buffer]
    private Vector<T>? _dataStds;

    /// <summary>
    /// Gets the hidden layer dimensions.
    /// </summary>
    public int HiddenDim => _hiddenDim;

    /// <summary>
    /// Gets the output dimensions.
    /// </summary>
    public int OutputDim => _outputDim;

    /// <summary>
    /// Creates a new Deep SVDD anomaly detector.
    /// </summary>
    /// <param name="hiddenDim">Dimensions of hidden layers. Default is 64.</param>
    /// <param name="outputDim">Dimensions of output (representation). Default is 32.</param>
    /// <param name="epochs">Number of training epochs. Default is 100.</param>
    /// <param name="learningRate">Learning rate. Default is 0.001.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public DeepSVDDDetector(int hiddenDim = 64, int outputDim = 32, int epochs = 100,
        double learningRate = 0.001, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (hiddenDim < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(hiddenDim),
                "HiddenDim must be at least 1. Recommended is 64.");
        }

        if (outputDim < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(outputDim),
                "OutputDim must be at least 1. Recommended is 32.");
        }

        if (epochs < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(epochs),
                "Epochs must be at least 1. Recommended is 100.");
        }

        if (learningRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate),
                "Learning rate must be positive. Recommended is 0.001.");
        }

        _hiddenDim = hiddenDim;
        _outputDim = outputDim;
        _epochs = epochs;
        _learningRate = learningRate;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        _inputDim = X.Columns;

        // Normalize data and store normalization parameters
        var (normalizedData, means, stds) = NormalizeData(X);
        _dataMeans = means;
        _dataStds = stds;

        // Initialize network weights
        InitializeWeights();

        // Compute initial center
        _center = ComputeCenter(normalizedData);

        // Train network
        TrainNetwork(normalizedData);

        // Recompute center after training
        _center = ComputeCenter(normalizedData);

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private (Matrix<T> normalized, Vector<T> means, Vector<T> stds) NormalizeData(Matrix<T> data)
    {
        // Per-feature mean and population std (std below 1e-10 replaced by 1), then (x - mean) / std
        var engine = Engine;
        var values = AnomalyTensorOps.ToTensor(data);
        var (std, mean) = engine.TensorStdMean(values, new[] { 0 }, 0, false);
        var tiny = engine.TensorLessThan(std, NumOps.FromDouble(1e-10));
        var stds = engine.TensorWhere(tiny, Tensor<T>.CreateOnes(new[] { data.Columns }), std);
        var normalized = AnomalyTensorOps.DivideRow(AnomalyTensorOps.SubtractRow(values, mean), stds);
        return (normalized.ToMatrix(), mean.ToVector(), stds.ToVector());
    }

    private void InitializeWeights()
    {
        double scale1 = Math.Sqrt(2.0 / (_inputDim + _hiddenDim));
        double scale2 = Math.Sqrt(2.0 / (_hiddenDim + _hiddenDim));
        double scale3 = Math.Sqrt(2.0 / (_hiddenDim + _outputDim));

        // Biases start at zero
        _w1 = InitializeMatrix(_inputDim, _hiddenDim, scale1);
        _b1 = new Vector<T>(_hiddenDim);
        _w2 = InitializeMatrix(_hiddenDim, _hiddenDim, scale2);
        _b2 = new Vector<T>(_hiddenDim);
        _w3 = InitializeMatrix(_hiddenDim, _outputDim, scale3);
        _b3 = new Vector<T>(_outputDim);
    }

    private Matrix<T> InitializeMatrix(int rows, int cols, double scale)
    {
        var matrix = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double u1 = 1.0 - _random.NextDouble();
                double u2 = 1.0 - _random.NextDouble();
                double val = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2) * scale;
                matrix[i, j] = NumOps.FromDouble(val);
            }
        }
        return matrix;
    }

    // Mean network output over the rows of data.
    private Vector<T> ComputeCenter(Matrix<T> data)
    {
        var (_, _, output) = Forward(AnomalyTensorOps.ToTensor(data));
        return Engine.ReduceMean(output, new[] { 0 }, false).ToVector();
    }

    // Batched forward pass: h1 = ReLU(x W1 + b1), h2 = ReLU(h1 W2 + b2), output = h2 W3 + b3 (linear).
    private (Tensor<T> h1, Tensor<T> h2, Tensor<T> output) Forward(Tensor<T> x)
    {
        var w1 = _w1;
        var b1 = _b1;
        var w2 = _w2;
        var b2 = _b2;
        var w3 = _w3;
        var b3 = _b3;
        if (w1 == null || b1 == null || w2 == null || b2 == null || w3 == null || b3 == null)
        {
            throw new InvalidOperationException("Weights not initialized.");
        }

        var engine = Engine;
        int rows = x.Shape[0];
        var h1 = engine.TensorMax(engine.TensorAdd(engine.TensorMatMul(x, Tensor<T>.FromMatrix(w1)),
            AnomalyTensorOps.BroadcastRow(b1.AsTensor(), rows)), NumOps.Zero);
        var h2 = engine.TensorMax(engine.TensorAdd(engine.TensorMatMul(h1, Tensor<T>.FromMatrix(w2)),
            AnomalyTensorOps.BroadcastRow(b2.AsTensor(), rows)), NumOps.Zero);
        var output = engine.TensorAdd(engine.TensorMatMul(h2, Tensor<T>.FromMatrix(w3)),
            AnomalyTensorOps.BroadcastRow(b3.AsTensor(), rows));
        return (h1, h2, output);
    }

    private void TrainNetwork(Matrix<T> data)
    {
        var center = _center;
        if (_w1 == null || _b1 == null || _w2 == null || _b2 == null || _w3 == null || _b3 == null || center == null)
        {
            throw new InvalidOperationException("Weights or center not initialized.");
        }

        var engine = Engine;
        int n = data.Rows;
        int batchSize = Math.Min(32, n);
        var values = AnomalyTensorOps.ToTensor(data);
        T two = NumOps.FromDouble(2);

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            var indices = Enumerable.Range(0, n).OrderBy(_ => _random.NextDouble()).ToArray();

            for (int batch = 0; batch < n; batch += batchSize)
            {
                int actualBatchSize = Math.Min(batchSize, n - batch);
                var batchRows = new int[actualBatchSize];
                Array.Copy(indices, batch, batchRows, 0, actualBatchSize);
                var x = AnomalyTensorOps.SelectRows(values, batchRows);
                var (h1, h2, output) = Forward(x);

                // Gradient of the loss ||output - center||^2, summed over the batch, back-propagated through
                // the output layer and the two ReLU layers
                var dOutput = engine.TensorMultiplyScalar(AnomalyTensorOps.SubtractRow(output, center.AsTensor()), two);
                var w2 = Tensor<T>.FromMatrix(_w2);
                var w3 = Tensor<T>.FromMatrix(_w3);
                var dW3 = engine.TensorMatMul(engine.TensorTranspose(h2), dOutput);
                var dB3 = engine.ReduceSum(dOutput, new[] { 0 }, false);
                var dH2 = engine.TensorMultiply(engine.TensorMatMul(dOutput, engine.TensorTranspose(w3)),
                    engine.TensorGreaterThan(h2, NumOps.Zero));
                var dW2 = engine.TensorMatMul(engine.TensorTranspose(h1), dH2);
                var dB2 = engine.ReduceSum(dH2, new[] { 0 }, false);
                var dH1 = engine.TensorMultiply(engine.TensorMatMul(dH2, engine.TensorTranspose(w2)),
                    engine.TensorGreaterThan(h1, NumOps.Zero));
                var dW1 = engine.TensorMatMul(engine.TensorTranspose(x), dH1);
                var dB1 = engine.ReduceSum(dH1, new[] { 0 }, false);

                // Update weights: w = w - lr * g with lr = learningRate / batch size
                T lr = NumOps.FromDouble(_learningRate / actualBatchSize);
                _w1 = Step(Tensor<T>.FromMatrix(_w1), dW1, lr).ToMatrix();
                _b1 = Step(_b1.AsTensor(), dB1, lr).ToVector();
                _w2 = Step(w2, dW2, lr).ToMatrix();
                _b2 = Step(_b2.AsTensor(), dB2, lr).ToVector();
                _w3 = Step(w3, dW3, lr).ToMatrix();
                _b3 = Step(_b3.AsTensor(), dB3, lr).ToVector();
            }
        }
    }

    // weights - lr * gradients
    private Tensor<T> Step(Tensor<T> weights, Tensor<T> gradients, T lr)
        => Engine.TensorSubtract(weights, Engine.TensorMultiplyScalar(gradients, lr));

    /// <inheritdoc/>
    public override Vector<T> ScoreAnomalies(Matrix<T> X)
    {
        EnsureFitted();
        return ScoreAnomaliesInternal(X);
    }

    private Vector<T> ScoreAnomaliesInternal(Matrix<T> X)
    {
        ValidateInput(X);

        if (X.Columns != _inputDim)
        {
            throw new ArgumentException(
                $"Input has {X.Columns} features, but model was fitted with {_inputDim} features.",
                nameof(X));
        }

        var dataMeans = _dataMeans;
        var dataStds = _dataStds;
        var center = _center;
        if (dataMeans == null || dataStds == null || center == null)
        {
            throw new InvalidOperationException("Model not properly fitted. Normalization parameters or center missing.");
        }

        // Squared distance of the (normalized) network output to the center
        var engine = Engine;
        var normalized = AnomalyTensorOps.DivideRow(
            AnomalyTensorOps.SubtractRow(AnomalyTensorOps.ToTensor(X), dataMeans.AsTensor()), dataStds.AsTensor());
        var (_, _, output) = Forward(normalized);
        return engine.ReduceSum(engine.TensorSquare(AnomalyTensorOps.SubtractRow(output, center.AsTensor())),
            new[] { 1 }, false).ToVector();
    }
}

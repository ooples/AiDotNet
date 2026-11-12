using AiDotNet.Tensors.Engines;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.NeuralNetwork;

/// <summary>
/// Implements DevNet (Deep Anomaly Detection Network) for end-to-end anomaly scoring.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> DevNet learns to directly output anomaly scores using a deviation
/// network. It combines feature learning and anomaly scoring in a single network,
/// using reference points and deviation loss for training.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Learn feature representations through neural network
/// 2. Use Gaussian reference points to define normalcy
/// 3. Train with deviation loss to produce anomaly scores directly
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When you want end-to-end anomaly scoring
/// - Tabular data with known anomaly labels (semi-supervised)
/// - When reconstruction-based methods don't work well
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Hidden dimensions: 64
/// - Output dimensions: 1
/// - Epochs: 50
/// - Learning rate: 0.001
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Pang, G., Shen, C., and van den Hengel, A. (2019).
/// "Deep Anomaly Detection with Deviation Networks." KDD.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Deep Anomaly Detection with Deviation Networks", "https://doi.org/10.1145/3292500.3330871", Year = 2019, Authors = "Guansong Pang, Chunhua Shen, Anton van den Hengel")]
public partial class DevNetDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _hiddenDim;
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

    // Reference statistics (initialized in Fit, default to zero before that)
    private T _refMean;
    private T _refStd;
    private int _inputDim;

    // Normalization parameters
    [Buffer]
    private Vector<T>? _dataMeans;
    [Buffer]
    private Vector<T>? _dataStds;

    /// <summary>
    /// Gets the hidden dimensions.
    /// </summary>
    public int HiddenDim => _hiddenDim;

    /// <summary>
    /// Creates a new DevNet anomaly detector.
    /// </summary>
    /// <param name="hiddenDim">Dimensions of hidden layers. Default is 64.</param>
    /// <param name="epochs">Number of training epochs. Default is 50.</param>
    /// <param name="learningRate">Learning rate. Default is 0.001.</param>
    /// <param name="marginScale">Scale for deviation margin. Default is 5.0.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public DevNetDetector(int hiddenDim = 64, int epochs = 50,
        double learningRate = 0.001, double marginScale = 5.0,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (hiddenDim < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(hiddenDim),
                "Hidden dimensions must be at least 1. Recommended is 64.");
        }

        if (epochs < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(epochs),
                "Epochs must be at least 1. Recommended is 50.");
        }

        if (learningRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate),
                "Learning rate must be positive. Recommended is 0.001.");
        }

        if (marginScale <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(marginScale),
                "Margin scale must be positive. Recommended is 5.0.");
        }

        _hiddenDim = hiddenDim;
        _epochs = epochs;
        _learningRate = learningRate;

        // Initialize reference statistics to default values (will be set in Fit)
        _refMean = NumOps.Zero;
        _refStd = NumOps.One;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        _inputDim = X.Columns;

        // Normalize data
        var (normalizedData, means, stds) = NormalizeData(X);
        _dataMeans = means;
        _dataStds = stds;

        // Initialize reference statistics (Gaussian prior)
        _refMean = NumOps.Zero;
        _refStd = NumOps.One;

        // Initialize weights
        InitializeWeights();

        // Train network with deviation loss
        Train(normalizedData);

        // Calculate scores for training data
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
        double scale3 = Math.Sqrt(2.0 / (_hiddenDim + 1));

        // Biases start at zero
        _w1 = InitializeMatrix(_inputDim, _hiddenDim, scale1);
        _b1 = new Vector<T>(_hiddenDim);
        _w2 = InitializeMatrix(_hiddenDim, _hiddenDim, scale2);
        _b2 = new Vector<T>(_hiddenDim);
        _w3 = InitializeMatrix(_hiddenDim, 1, scale3);
        _b3 = new Vector<T>(1);
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

    private void Train(Matrix<T> data)
    {
        if (_w1 == null || _b1 == null || _w2 == null || _b2 == null || _w3 == null || _b3 == null)
        {
            throw new InvalidOperationException("Weights not initialized.");
        }

        var engine = Engine;
        int n = data.Rows;
        int batchSize = Math.Min(64, n);
        var values = AnomalyTensorOps.ToTensor(data);
        T two = NumOps.FromDouble(2);
        T clipMin = NumOps.FromDouble(-10.0);
        T clipMax = NumOps.FromDouble(10.0);

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            var indices = Enumerable.Range(0, n).OrderBy(_ => _random.NextDouble()).ToArray();

            for (int batch = 0; batch < n; batch += batchSize)
            {
                int actualBatchSize = Math.Min(batchSize, n - batch);
                var batchRows = new int[actualBatchSize];
                Array.Copy(indices, batch, batchRows, 0, actualBatchSize);
                var x = AnomalyTensorOps.SelectRows(values, batchRows);
                var (h1, h2, score) = Forward(x);

                // Reference scores refMean + refStd * z with z ~ N(0, 1) drawn by Box-Muller, one per sample
                // in batch order (same random draws as the per-sample loop)
                var gaussian = new double[actualBatchSize];
                for (int b = 0; b < actualBatchSize; b++)
                {
                    double u1 = 1.0 - _random.NextDouble();
                    double u2 = 1.0 - _random.NextDouble();
                    gaussian[b] = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                }

                var refScores = engine.TensorAddScalar(engine.TensorMultiplyScalar(
                    Tensor<double>.FromVector(Vector<double>.Wrap(gaussian), new[] { actualBatchSize, 1 }).Cast<T>(),
                    _refStd), _refMean);

                // Deviation loss (score - refScore)^2: gradient 2 * (score - refScore), clipped to [-10, 10], then
                // back-propagated through the output layer and the two ReLU layers
                var dScore = engine.TensorClamp(engine.TensorMultiplyScalar(engine.TensorSubtract(score, refScores), two),
                    clipMin, clipMax);
                var w2 = Tensor<T>.FromMatrix(_w2);
                var w3 = Tensor<T>.FromMatrix(_w3);
                var dW3 = engine.TensorMatMul(engine.TensorTranspose(h2), dScore);
                var dB3 = engine.ReduceSum(dScore, new[] { 0 }, false);
                var dH2 = engine.TensorMultiply(engine.TensorMatMul(dScore, engine.TensorTranspose(w3)),
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

    // Batched forward pass: h1 = ReLU(x W1 + b1), h2 = ReLU(h1 W2 + b2), score = h2 W3 + b3 ([rows, 1], linear).
    private (Tensor<T> h1, Tensor<T> h2, Tensor<T> score) Forward(Tensor<T> x)
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
        var score = engine.TensorAdd(engine.TensorMatMul(h2, Tensor<T>.FromMatrix(w3)),
            AnomalyTensorOps.BroadcastRow(b3.AsTensor(), rows));
        return (h1, h2, score);
    }

    /// <inheritdoc/>
    public override Vector<T> ScoreAnomalies(Matrix<T> X)
    {
        EnsureFitted();
        return ScoreAnomaliesInternal(X);
    }

    private Vector<T> ScoreAnomaliesInternal(Matrix<T> X)
    {
        ValidateInput(X);

        var dataMeans = _dataMeans;
        var dataStds = _dataStds;
        if (dataMeans == null || dataStds == null)
        {
            throw new InvalidOperationException("Model not properly fitted. Normalization parameters missing.");
        }

        if (X.Columns != _inputDim)
        {
            throw new ArgumentException($"Expected {_inputDim} features but got {X.Columns}.", nameof(X));
        }

        // Anomaly score |score - refMean| of the normalized rows (higher = more anomalous)
        var engine = Engine;
        var normalized = AnomalyTensorOps.DivideRow(
            AnomalyTensorOps.SubtractRow(AnomalyTensorOps.ToTensor(X), dataMeans.AsTensor()), dataStds.AsTensor());
        var (_, _, score) = Forward(normalized);
        return engine.TensorAbs(engine.TensorSubtractScalar(engine.Reshape(score, new[] { X.Rows }), _refMean)).ToVector();
    }
}

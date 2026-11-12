using AiDotNet.Tensors.Engines;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.NeuralNetwork;

/// <summary>
/// Implements GANomaly for anomaly detection using GAN-based reconstruction.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> GANomaly learns to encode, decode, and re-encode data.
/// Anomalies are detected when the encoding of the original differs from the
/// encoding of the reconstruction, indicating the model cannot properly represent the data.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Encoder maps input to latent space z
/// 2. Decoder reconstructs from z
/// 3. Second encoder re-encodes the reconstruction
/// 4. Anomaly score is the difference between original encoding and re-encoding
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Image anomaly detection
/// - When reconstruction error alone is insufficient
/// - Semi-supervised anomaly detection with only normal examples
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Latent dimensions: 32
/// - Hidden dimensions: 64
/// - Epochs: 100
/// - Learning rate: 0.0002
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Akcay, S., Atapour-Abarghouei, A., and Breckon, T. P. (2018).
/// "GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training." ACCV.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.GAN)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training", "https://doi.org/10.1007/978-3-030-20893-6_39", Year = 2018, Authors = "Samet Akcay, Amir Atapour-Abarghouei, Toby P. Breckon")]
public partial class GANomalyDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _latentDim;
    private readonly int _hiddenDim;
    private readonly int _epochs;
    private readonly double _learningRate;

    // Encoder weights
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _encW1;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _encB1;
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _encW2;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _encB2;

    // Decoder weights
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _decW1;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _decB1;
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _decW2;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _decB2;

    // Re-encoder weights (separate encoder for reconstruction)
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _reEncW1;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _reEncB1;
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _reEncW2;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _reEncB2;

    private int _inputDim;

    // Normalization parameters
    [Buffer]
    private Vector<T>? _dataMeans;
    [Buffer]
    private Vector<T>? _dataStds;

    /// <summary>
    /// Gets the latent dimensions.
    /// </summary>
    public int LatentDim => _latentDim;

    /// <summary>
    /// Gets the hidden dimensions.
    /// </summary>
    public int HiddenDim => _hiddenDim;

    /// <summary>
    /// Creates a new GANomaly anomaly detector.
    /// </summary>
    /// <param name="latentDim">Dimensions of latent space. Default is 32.</param>
    /// <param name="hiddenDim">Dimensions of hidden layers. Default is 64.</param>
    /// <param name="epochs">Number of training epochs. Default is 100.</param>
    /// <param name="learningRate">Learning rate. Default is 0.0002.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public GANomalyDetector(int latentDim = 32, int hiddenDim = 64, int epochs = 100,
        double learningRate = 0.0002, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (latentDim < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(latentDim),
                "Latent dimensions must be at least 1. Recommended is 32.");
        }

        if (hiddenDim < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(hiddenDim),
                "Hidden dimensions must be at least 1. Recommended is 64.");
        }

        if (epochs < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(epochs),
                "Epochs must be at least 1. Recommended is 100.");
        }

        if (learningRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate),
                "Learning rate must be positive. Recommended is 0.0002.");
        }

        _latentDim = latentDim;
        _hiddenDim = hiddenDim;
        _epochs = epochs;
        _learningRate = learningRate;
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

        // Initialize weights
        InitializeWeights();

        // Train
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
        double scale2 = Math.Sqrt(2.0 / (_hiddenDim + _latentDim));
        double scale3 = Math.Sqrt(2.0 / (_latentDim + _hiddenDim));
        double scale4 = Math.Sqrt(2.0 / (_hiddenDim + _inputDim));

        // Encoder: input -> hidden -> latent
        _encW1 = InitializeMatrix(_inputDim, _hiddenDim, scale1);
        _encB1 = new Vector<T>(_hiddenDim);
        _encW2 = InitializeMatrix(_hiddenDim, _latentDim, scale2);
        _encB2 = new Vector<T>(_latentDim);

        // Decoder: latent -> hidden -> input
        _decW1 = InitializeMatrix(_latentDim, _hiddenDim, scale3);
        _decB1 = new Vector<T>(_hiddenDim);
        _decW2 = InitializeMatrix(_hiddenDim, _inputDim, scale4);
        _decB2 = new Vector<T>(_inputDim);

        // Re-encoder: input -> hidden -> latent
        _reEncW1 = InitializeMatrix(_inputDim, _hiddenDim, scale1);
        _reEncB1 = new Vector<T>(_hiddenDim);
        _reEncW2 = InitializeMatrix(_hiddenDim, _latentDim, scale2);
        _reEncB2 = new Vector<T>(_latentDim);
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
        int n = data.Rows;
        int batchSize = Math.Min(32, n);
        var values = AnomalyTensorOps.ToTensor(data);

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            var indices = Enumerable.Range(0, n).OrderBy(_ => _random.NextDouble()).ToArray();

            for (int batch = 0; batch < n; batch += batchSize)
            {
                int actualBatchSize = Math.Min(batchSize, n - batch);
                var batchRows = new int[actualBatchSize];
                Array.Copy(indices, batch, batchRows, 0, actualBatchSize);

                // Forward pass, then gradients summed over the batch and a step of learningRate / batch size
                var forward = Forward(AnomalyTensorOps.SelectRows(values, batchRows));
                Backward(forward, NumOps.FromDouble(_learningRate / actualBatchSize));
            }
        }
    }

    // Activations of a batched encode -> decode -> re-encode pass (pre-activations kept for the LeakyReLU slopes).
    private sealed class ForwardResult
    {
        public required Tensor<T> X { get; init; }
        public required Tensor<T> EncHiddenPre { get; init; }
        public required Tensor<T> EncHidden { get; init; }
        public required Tensor<T> Z { get; init; }
        public required Tensor<T> DecHiddenPre { get; init; }
        public required Tensor<T> DecHidden { get; init; }
        public required Tensor<T> XRecon { get; init; }
        public required Tensor<T> ReEncHiddenPre { get; init; }
        public required Tensor<T> ReEncHidden { get; init; }
        public required Tensor<T> ZRecon { get; init; }
    }

    // Each sub-network is h = LeakyReLU(input W1 + b1, 0.2), output = h W2 + b2 (linear).
    private ForwardResult Forward(Tensor<T> x)
    {
        var encW1 = _encW1;
        var encB1 = _encB1;
        var encW2 = _encW2;
        var encB2 = _encB2;
        var decW1 = _decW1;
        var decB1 = _decB1;
        var decW2 = _decW2;
        var decB2 = _decB2;
        var reEncW1 = _reEncW1;
        var reEncB1 = _reEncB1;
        var reEncW2 = _reEncW2;
        var reEncB2 = _reEncB2;

        if (encW1 == null || encB1 == null || encW2 == null || encB2 == null ||
            decW1 == null || decB1 == null || decW2 == null || decB2 == null ||
            reEncW1 == null || reEncB1 == null || reEncW2 == null || reEncB2 == null)
        {
            throw new InvalidOperationException("Weights not initialized.");
        }

        var engine = Engine;
        T leakyAlpha = NumOps.FromDouble(0.2);
        var encHiddenPre = Affine(x, encW1, encB1);
        var encHidden = engine.LeakyReLU(encHiddenPre, leakyAlpha);
        var z = Affine(encHidden, encW2, encB2);
        var decHiddenPre = Affine(z, decW1, decB1);
        var decHidden = engine.LeakyReLU(decHiddenPre, leakyAlpha);
        var xRecon = Affine(decHidden, decW2, decB2);
        var reEncHiddenPre = Affine(xRecon, reEncW1, reEncB1);
        var reEncHidden = engine.LeakyReLU(reEncHiddenPre, leakyAlpha);
        var zRecon = Affine(reEncHidden, reEncW2, reEncB2);

        return new ForwardResult
        {
            X = x,
            EncHiddenPre = encHiddenPre,
            EncHidden = encHidden,
            Z = z,
            DecHiddenPre = decHiddenPre,
            DecHidden = decHidden,
            XRecon = xRecon,
            ReEncHiddenPre = reEncHiddenPre,
            ReEncHidden = reEncHidden,
            ZRecon = zRecon
        };
    }

    // input W + b (bias broadcast over the rows)
    private Tensor<T> Affine(Tensor<T> input, Matrix<T> weights, Vector<T> bias)
        => Engine.TensorAdd(Engine.TensorMatMul(input, Tensor<T>.FromMatrix(weights)),
            AnomalyTensorOps.BroadcastRow(bias.AsTensor(), input.Shape[0]));

    // LeakyReLU derivative: 0.2 where the pre-activation is negative, 1 elsewhere.
    private Tensor<T> LeakySlope(Tensor<T> preActivation)
        => Engine.TensorAddScalar(Engine.TensorMultiplyScalar(
            Engine.TensorLessThan(preActivation, NumOps.Zero), NumOps.FromDouble(-0.8)), NumOps.One);

    // Back-propagates the batch sum of the reconstruction loss ||x - xRecon||^2 and the latent consistency
    // loss ||z - zRecon||^2 (both with weight 1) and applies w = w - lr * g to every weight. The re-encoder
    // learns zRecon -> z; the encoder receives the reconstruction gradient through the decoder plus the
    // latent consistency gradient on z (the re-encoder path does not feed back into the decoder).
    private void Backward(ForwardResult forward, T lr)
    {
        var encW1 = _encW1;
        var encB1 = _encB1;
        var encW2 = _encW2;
        var encB2 = _encB2;
        var decW1 = _decW1;
        var decB1 = _decB1;
        var decW2 = _decW2;
        var decB2 = _decB2;
        var reEncW1 = _reEncW1;
        var reEncB1 = _reEncB1;
        var reEncW2 = _reEncW2;
        var reEncB2 = _reEncB2;

        if (encW1 == null || encB1 == null || encW2 == null || encB2 == null ||
            decW1 == null || decB1 == null || decW2 == null || decB2 == null ||
            reEncW1 == null || reEncB1 == null || reEncW2 == null || reEncB2 == null)
        {
            throw new InvalidOperationException("Weights not initialized.");
        }

        var engine = Engine;
        T two = NumOps.FromDouble(2);
        var axis0 = new[] { 0 };
        var encW2Tensor = Tensor<T>.FromMatrix(encW2);
        var decW1Tensor = Tensor<T>.FromMatrix(decW1);
        var decW2Tensor = Tensor<T>.FromMatrix(decW2);
        var reEncW2Tensor = Tensor<T>.FromMatrix(reEncW2);

        // === Backprop for reconstruction loss: ||x - xRecon||^2 ===
        var dXRecon = engine.TensorMultiplyScalar(engine.TensorSubtract(forward.XRecon, forward.X), two);
        var gDecW2 = engine.TensorMatMul(engine.TensorTranspose(forward.DecHidden), dXRecon);
        var gDecB2 = engine.ReduceSum(dXRecon, axis0, false);
        var dDecH1 = engine.TensorMultiply(engine.TensorMatMul(dXRecon, engine.TensorTranspose(decW2Tensor)),
            LeakySlope(forward.DecHiddenPre));
        var gDecW1 = engine.TensorMatMul(engine.TensorTranspose(forward.Z), dDecH1);
        var gDecB1 = engine.ReduceSum(dDecH1, axis0, false);
        var dZFromRecon = engine.TensorMatMul(dDecH1, engine.TensorTranspose(decW1Tensor));

        // === Backprop for latent consistency loss: ||z - zRecon||^2 ===
        var dZRecon = engine.TensorMultiplyScalar(engine.TensorSubtract(forward.ZRecon, forward.Z), two);
        var dZFromLatent = engine.TensorNegate(dZRecon);
        var gReEncW2 = engine.TensorMatMul(engine.TensorTranspose(forward.ReEncHidden), dZRecon);
        var gReEncB2 = engine.ReduceSum(dZRecon, axis0, false);
        var dReEncH1 = engine.TensorMultiply(engine.TensorMatMul(dZRecon, engine.TensorTranspose(reEncW2Tensor)),
            LeakySlope(forward.ReEncHiddenPre));
        var gReEncW1 = engine.TensorMatMul(engine.TensorTranspose(forward.XRecon), dReEncH1);
        var gReEncB1 = engine.ReduceSum(dReEncH1, axis0, false);

        // === Backprop through encoder ===
        var dZ = engine.TensorAdd(dZFromRecon, dZFromLatent);
        var gEncW2 = engine.TensorMatMul(engine.TensorTranspose(forward.EncHidden), dZ);
        var gEncB2 = engine.ReduceSum(dZ, axis0, false);
        var dEncH1 = engine.TensorMultiply(engine.TensorMatMul(dZ, engine.TensorTranspose(encW2Tensor)),
            LeakySlope(forward.EncHiddenPre));
        var gEncW1 = engine.TensorMatMul(engine.TensorTranspose(forward.X), dEncH1);
        var gEncB1 = engine.ReduceSum(dEncH1, axis0, false);

        _encW1 = Step(Tensor<T>.FromMatrix(encW1), gEncW1, lr).ToMatrix();
        _encB1 = Step(encB1.AsTensor(), gEncB1, lr).ToVector();
        _encW2 = Step(encW2Tensor, gEncW2, lr).ToMatrix();
        _encB2 = Step(encB2.AsTensor(), gEncB2, lr).ToVector();
        _decW1 = Step(decW1Tensor, gDecW1, lr).ToMatrix();
        _decB1 = Step(decB1.AsTensor(), gDecB1, lr).ToVector();
        _decW2 = Step(decW2Tensor, gDecW2, lr).ToMatrix();
        _decB2 = Step(decB2.AsTensor(), gDecB2, lr).ToVector();
        _reEncW1 = Step(Tensor<T>.FromMatrix(reEncW1), gReEncW1, lr).ToMatrix();
        _reEncB1 = Step(reEncB1.AsTensor(), gReEncB1, lr).ToVector();
        _reEncW2 = Step(reEncW2Tensor, gReEncW2, lr).ToMatrix();
        _reEncB2 = Step(reEncB2.AsTensor(), gReEncB2, lr).ToVector();
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

        var dataMeans = _dataMeans;
        var dataStds = _dataStds;
        if (dataMeans == null || dataStds == null)
        {
            throw new InvalidOperationException("Model not properly fitted. Normalization parameters missing.");
        }

        // Anomaly score: ||z - zRecon||^2 (latent consistency) after encode, decode, re-encode of the normalized rows
        var engine = Engine;
        var normalized = AnomalyTensorOps.DivideRow(
            AnomalyTensorOps.SubtractRow(AnomalyTensorOps.ToTensor(X), dataMeans.AsTensor()), dataStds.AsTensor());
        var forward = Forward(normalized);
        return engine.ReduceSum(engine.TensorSquare(engine.TensorSubtract(forward.Z, forward.ZRecon)),
            new[] { 1 }, false).ToVector();
    }
}

using AiDotNet.Tensors.Engines;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.NeuralNetwork;

/// <summary>
/// Detects anomalies using Variational Autoencoder (VAE).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A VAE is a generative neural network that learns to encode data into
/// a lower-dimensional probabilistic latent space and decode it back. Anomalies are points
/// that are poorly reconstructed or fall in low-probability regions of the latent space.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Train encoder to map data to latent distribution (mean + variance)
/// 2. Train decoder to reconstruct from latent samples
/// 3. Score combines reconstruction error and KL divergence
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Complex, high-dimensional data
/// - When you want probabilistic anomaly scores
/// - Image, text, or structured data anomalies
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Latent dimensions: 10
/// - Hidden dimensions: 64
/// - Learning rate: 0.001
/// - Epochs: 100
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Kingma, D.P., Welling, M. (2014). "Auto-Encoding Variational Bayes." ICLR.
/// An, J., Cho, S. (2015). "Variational Autoencoder based Anomaly Detection."
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Autoencoder)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Auto-Encoding Variational Bayes", "https://doi.org/10.48550/arXiv.1312.6114", Year = 2014, Authors = "Diederik P. Kingma, Max Welling")]
public partial class VAEDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _latentDim;
    private readonly int _hiddenDim;
    private readonly int _epochs;
    private readonly double _learningRate;

    // Encoder weights
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _encoderW1;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _encoderB1;
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _encoderWMean;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _encoderBMean;
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _encoderWLogVar;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _encoderBLogVar;

    // Decoder weights
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _decoderW1;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _decoderB1;
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _decoderW2;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _decoderB2;

    // Normalization parameters
    [Buffer]
    private Vector<T>? _dataMeans;
    [Buffer]
    private Vector<T>? _dataStds;

    private int _inputDim;

    /// <summary>
    /// Gets the latent space dimensions.
    /// </summary>
    public int LatentDim => _latentDim;

    /// <summary>
    /// Gets the hidden layer dimensions.
    /// </summary>
    public int HiddenDim => _hiddenDim;

    /// <summary>
    /// Creates a new VAE anomaly detector.
    /// </summary>
    /// <param name="latentDim">Dimensions of the latent space. Default is 10.</param>
    /// <param name="hiddenDim">Dimensions of hidden layers. Default is 64.</param>
    /// <param name="epochs">Number of training epochs. Default is 100.</param>
    /// <param name="learningRate">Learning rate. Default is 0.001.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public VAEDetector(int latentDim = 10, int hiddenDim = 64, int epochs = 100,
        double learningRate = 0.001, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (latentDim < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(latentDim),
                "LatentDim must be at least 1. Recommended is 10.");
        }

        if (hiddenDim < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(hiddenDim),
                "HiddenDim must be at least 1. Recommended is 64.");
        }

        if (epochs < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(epochs),
                "Epochs must be at least 1. Recommended is 100.");
        }

        if (learningRate <= 0 || double.IsNaN(learningRate) || double.IsInfinity(learningRate))
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate),
                "Learning rate must be a positive, finite number. Recommended is 0.001.");
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

        // Normalize data and store parameters
        var (normalizedData, means, stds) = NormalizeData(X);
        _dataMeans = means;
        _dataStds = stds;

        // Initialize weights
        InitializeWeights();

        // Train VAE
        TrainVAE(normalizedData);

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
        // Xavier initialization
        double scale1 = Math.Sqrt(2.0 / (_inputDim + _hiddenDim));
        double scale2 = Math.Sqrt(2.0 / (_hiddenDim + _latentDim));
        double scale3 = Math.Sqrt(2.0 / (_latentDim + _hiddenDim));
        double scale4 = Math.Sqrt(2.0 / (_hiddenDim + _inputDim));

        // Encoder
        _encoderW1 = InitializeMatrix(_inputDim, _hiddenDim, scale1);
        _encoderB1 = new Vector<T>(_hiddenDim);
        _encoderWMean = InitializeMatrix(_hiddenDim, _latentDim, scale2);
        _encoderBMean = new Vector<T>(_latentDim);
        _encoderWLogVar = InitializeMatrix(_hiddenDim, _latentDim, scale2);
        _encoderBLogVar = new Vector<T>(_latentDim);

        // Decoder
        _decoderW1 = InitializeMatrix(_latentDim, _hiddenDim, scale3);
        _decoderB1 = new Vector<T>(_hiddenDim);
        _decoderW2 = InitializeMatrix(_hiddenDim, _inputDim, scale4);
        _decoderB2 = new Vector<T>(_inputDim);
    }

    private Matrix<T> InitializeMatrix(int rows, int cols, double scale)
    {
        var matrix = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                // Box-Muller transform
                double u1 = 1.0 - _random.NextDouble();
                double u2 = 1.0 - _random.NextDouble();
                double val = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2) * scale;
                matrix[i, j] = NumOps.FromDouble(val);
            }
        }
        return matrix;
    }

    private void TrainVAE(Matrix<T> data)
    {
        int n = data.Rows;
        int batchSize = Math.Min(32, n);
        var values = AnomalyTensorOps.ToTensor(data);
        // Step learningRate / batch size: every batch is full except possibly the last one
        int lastBatchSize = batchSize > 0 && n % batchSize != 0 ? n % batchSize : batchSize;
        T fullBatchStep = batchSize > 0 ? NumOps.FromDouble(_learningRate / batchSize) : NumOps.Zero;
        T lastBatchStep = lastBatchSize > 0 ? NumOps.FromDouble(_learningRate / lastBatchSize) : NumOps.Zero;

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            // Shuffle data
            var indices = Enumerable.Range(0, n).OrderBy(_ => _random.NextDouble()).ToArray();

            for (int batch = 0; batch < n; batch += batchSize)
            {
                int actualBatchSize = Math.Min(batchSize, n - batch);
                var batchRows = new int[actualBatchSize];
                Array.Copy(indices, batch, batchRows, 0, actualBatchSize);
                var x = AnomalyTensorOps.SelectRows(values, batchRows);

                // Reparameterization noise epsilon ~ N(0, 1): one seeded engine draw per batch, seed taken from
                // the detector's Random
                var epsilon = Tensor<T>.CreateZeros(new[] { actualBatchSize, _latentDim });
                Engine.TensorNormalInPlace(epsilon, 0.0, 1.0, _random.Next());

                // Forward pass, then gradients summed over the batch and a step of learningRate / batch size
                var forward = Forward(x, epsilon);
                Backward(x, forward, actualBatchSize == batchSize ? fullBatchStep : lastBatchStep);
            }
        }
    }

    // Activations of a batched forward pass.
    private sealed class ForwardResult
    {
        public required Tensor<T> Hidden { get; init; }
        public required Tensor<T> Mean { get; init; }
        public required Tensor<T> LogVar { get; init; }
        public required Tensor<T> Std { get; init; }
        public required Tensor<T> Epsilon { get; init; }
        public required Tensor<T> Z { get; init; }
        public required Tensor<T> Hidden2 { get; init; }
        public required Tensor<T> Reconstruction { get; init; }
    }

    // Batched forward pass. Encoder: hidden = ReLU(x W1 + b1), mean = hidden WMean + bMean,
    // logVar = hidden WLogVar + bLogVar. Reparameterization z = mean + exp(0.5 * logVar) * epsilon.
    // Sampling belongs to TRAINING only. Scoring drew a fresh epsilon on every call, so the
    // reconstruction — and therefore the anomaly score — changed between identical Predict
    // calls and Predict_ShouldBeDeterministic failed. Per Kingma and Welling 2013 the
    // reparameterization trick exists to make the training objective differentiable; at
    // inference (epsilon null) the latent code is the distribution's mean, which is also what VAE-based
    // anomaly detection scores against. Decoder: hidden2 = ReLU(z D1 + c1), reconstruction = hidden2 D2 + c2.
    private ForwardResult Forward(Tensor<T> x, Tensor<T>? epsilon)
    {
        var encoderW1 = _encoderW1;
        var encoderB1 = _encoderB1;
        var encoderWMean = _encoderWMean;
        var encoderBMean = _encoderBMean;
        var encoderWLogVar = _encoderWLogVar;
        var encoderBLogVar = _encoderBLogVar;
        var decoderW1 = _decoderW1;
        var decoderB1 = _decoderB1;
        var decoderW2 = _decoderW2;
        var decoderB2 = _decoderB2;

        if (encoderW1 == null || encoderB1 == null || encoderWMean == null || encoderBMean == null ||
            encoderWLogVar == null || encoderBLogVar == null || decoderW1 == null || decoderB1 == null ||
            decoderW2 == null || decoderB2 == null)
        {
            throw new InvalidOperationException("Weights not initialized.");
        }

        var engine = Engine;
        int rows = x.Shape[0];
        var hidden = engine.TensorMax(Affine(x, encoderW1, encoderB1), NumOps.Zero);
        var mean = Affine(hidden, encoderWMean, encoderBMean);
        var logVar = Affine(hidden, encoderWLogVar, encoderBLogVar);
        var std = engine.TensorExp(engine.TensorMultiplyScalar(logVar, NumOps.FromDouble(0.5)));
        var noise = epsilon ?? Tensor<T>.CreateZeros(new[] { rows, _latentDim });
        var z = epsilon == null ? mean : engine.TensorAdd(mean, engine.TensorMultiply(std, noise));
        var hidden2 = engine.TensorMax(Affine(z, decoderW1, decoderB1), NumOps.Zero);
        var reconstruction = Affine(hidden2, decoderW2, decoderB2);

        return new ForwardResult
        {
            Hidden = hidden,
            Mean = mean,
            LogVar = logVar,
            Std = std,
            Epsilon = noise,
            Z = z,
            Hidden2 = hidden2,
            Reconstruction = reconstruction
        };
    }

    // input W + b (bias broadcast over the rows)
    private Tensor<T> Affine(Tensor<T> input, Matrix<T> weights, Vector<T> bias)
        => Engine.TensorAdd(Engine.TensorMatMul(input, Tensor<T>.FromMatrix(weights)),
            AnomalyTensorOps.BroadcastRow(bias.AsTensor(), input.Shape[0]));

    // Back-propagates the loss sum over the batch of ||reconstruction - x||^2 + KL(N(mean, exp(logVar)) || N(0, 1))
    // and applies w = w - lr * g to every weight.
    private void Backward(Tensor<T> x, ForwardResult forward, T lr)
    {
        var encoderW1 = _encoderW1;
        var encoderB1 = _encoderB1;
        var encoderWMean = _encoderWMean;
        var encoderBMean = _encoderBMean;
        var encoderWLogVar = _encoderWLogVar;
        var encoderBLogVar = _encoderBLogVar;
        var decoderW1 = _decoderW1;
        var decoderB1 = _decoderB1;
        var decoderW2 = _decoderW2;
        var decoderB2 = _decoderB2;

        if (encoderW1 == null || encoderB1 == null || encoderWMean == null || encoderBMean == null ||
            encoderWLogVar == null || encoderBLogVar == null || decoderW1 == null || decoderB1 == null ||
            decoderW2 == null || decoderB2 == null)
        {
            throw new InvalidOperationException("Weights not initialized.");
        }

        var engine = Engine;
        T half = NumOps.FromDouble(0.5);
        var axis0 = new[] { 0 };

        // Reconstruction loss gradient: d/dRecon MSE = 2 * (recon - x), back through the decoder output layer
        // and the hidden2 ReLU
        var dReconstruction = engine.TensorMultiplyScalar(engine.TensorSubtract(forward.Reconstruction, x), NumOps.FromDouble(2));
        var decoderW2Tensor = Tensor<T>.FromMatrix(decoderW2);
        var decoderW1Tensor = Tensor<T>.FromMatrix(decoderW1);
        var gDecoderW2 = engine.TensorMatMul(engine.TensorTranspose(forward.Hidden2), dReconstruction);
        var gDecoderB2 = engine.ReduceSum(dReconstruction, axis0, false);
        var dHidden2 = engine.TensorMultiply(engine.TensorMatMul(dReconstruction, engine.TensorTranspose(decoderW2Tensor)),
            engine.TensorGreaterThan(forward.Hidden2, NumOps.Zero));
        var gDecoderW1 = engine.TensorMatMul(engine.TensorTranspose(forward.Z), dHidden2);
        var gDecoderB1 = engine.ReduceSum(dHidden2, axis0, false);
        var dZ = engine.TensorMatMul(dHidden2, engine.TensorTranspose(decoderW1Tensor));

        // Reconstruction gradient through the reparameterization trick
        // (Kingma & Welling 2014 "Auto-Encoding Variational Bayes" §2.4):
        //   z = μ + σ · ε,    σ = exp(0.5 · logVar)
        // Exact gradients:
        //   ∂z/∂μ      = 1
        //   ∂z/∂logVar = ε · 0.5 · σ
        // using the SAME ε as the forward pass (taken as 0 where σ <= 1e-12). KL divergence gradient:
        // d/dMean = mean, d/dLogVar = 0.5 * (exp(logVar) - 1).
        var epsilon = engine.TensorWhere(engine.TensorGreaterThan(forward.Std, NumOps.FromDouble(1e-12)),
            forward.Epsilon, Tensor<T>.CreateZeros(new[] { x.Shape[0], _latentDim }));
        var dMean = engine.TensorAdd(dZ, forward.Mean);
        var dLogVar = engine.TensorAdd(
            engine.TensorMultiplyScalar(engine.TensorMultiply(dZ, engine.TensorMultiply(forward.Std, epsilon)), half),
            engine.TensorMultiplyScalar(engine.TensorSubtractScalar(engine.TensorExp(forward.LogVar), NumOps.One), half));

        // Backprop through encoder mean/logvar layers, the hidden ReLU and encoder layer 1
        var encoderWMeanTensor = Tensor<T>.FromMatrix(encoderWMean);
        var encoderWLogVarTensor = Tensor<T>.FromMatrix(encoderWLogVar);
        var hiddenT = engine.TensorTranspose(forward.Hidden);
        var gEncoderWMean = engine.TensorMatMul(hiddenT, dMean);
        var gEncoderWLogVar = engine.TensorMatMul(hiddenT, dLogVar);
        var gEncoderBMean = engine.ReduceSum(dMean, axis0, false);
        var gEncoderBLogVar = engine.ReduceSum(dLogVar, axis0, false);
        var dHidden = engine.TensorMultiply(engine.TensorAdd(
                engine.TensorMatMul(dMean, engine.TensorTranspose(encoderWMeanTensor)),
                engine.TensorMatMul(dLogVar, engine.TensorTranspose(encoderWLogVarTensor))),
            engine.TensorGreaterThan(forward.Hidden, NumOps.Zero));
        var gEncoderW1 = engine.TensorMatMul(engine.TensorTranspose(x), dHidden);
        var gEncoderB1 = engine.ReduceSum(dHidden, axis0, false);

        // Update encoder
        _encoderW1 = Step(Tensor<T>.FromMatrix(encoderW1), gEncoderW1, lr).ToMatrix();
        _encoderB1 = Step(encoderB1.AsTensor(), gEncoderB1, lr).ToVector();
        _encoderWMean = Step(encoderWMeanTensor, gEncoderWMean, lr).ToMatrix();
        _encoderBMean = Step(encoderBMean.AsTensor(), gEncoderBMean, lr).ToVector();
        _encoderWLogVar = Step(encoderWLogVarTensor, gEncoderWLogVar, lr).ToMatrix();
        _encoderBLogVar = Step(encoderBLogVar.AsTensor(), gEncoderBLogVar, lr).ToVector();

        // Update decoder
        _decoderW1 = Step(decoderW1Tensor, gDecoderW1, lr).ToMatrix();
        _decoderB1 = Step(decoderB1.AsTensor(), gDecoderB1, lr).ToVector();
        _decoderW2 = Step(decoderW2Tensor, gDecoderW2, lr).ToMatrix();
        _decoderB2 = Step(decoderB2.AsTensor(), gDecoderB2, lr).ToVector();
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

        // Score = mean squared reconstruction error + 0.1 * mean KL divergence -0.5 * (1 + logVar - mean^2 - exp(logVar))
        // of the normalized rows, with the latent code at its mean
        var engine = Engine;
        var normalized = AnomalyTensorOps.DivideRow(
            AnomalyTensorOps.SubtractRow(AnomalyTensorOps.ToTensor(X), dataMeans.AsTensor()), dataStds.AsTensor());
        var forward = Forward(normalized, null);
        var axis1 = new[] { 1 };
        var reconError = engine.ReduceMean(engine.TensorSquare(engine.TensorSubtract(normalized, forward.Reconstruction)), axis1, false);
        var klTerms = engine.TensorSubtract(
            engine.TensorSubtract(engine.TensorAddScalar(forward.LogVar, NumOps.One), engine.TensorSquare(forward.Mean)),
            engine.TensorExp(forward.LogVar));
        var klDiv = engine.TensorMultiplyScalar(engine.ReduceMean(klTerms, axis1, false), NumOps.FromDouble(-0.5));
        return engine.TensorAdd(reconError, engine.TensorMultiplyScalar(klDiv, NumOps.FromDouble(0.1))).ToVector();
    }
}

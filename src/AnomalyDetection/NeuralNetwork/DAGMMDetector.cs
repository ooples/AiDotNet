using AiDotNet.Tensors.Engines;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.NeuralNetwork;

/// <summary>
/// Implements DAGMM (Deep Autoencoding Gaussian Mixture Model) for anomaly detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> DAGMM combines an autoencoder with a Gaussian Mixture Model
/// in an end-to-end trainable architecture. The autoencoder learns compressed representations,
/// and the GMM learns the density of normal data in the latent space.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Compression network (autoencoder) learns latent representation
/// 2. Estimation network predicts GMM membership probabilities
/// 3. GMM models the distribution of latent codes + reconstruction features
/// 4. Anomaly score is the negative log-likelihood under the GMM
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Complex multivariate data
/// - When you want to model multiple modes of normal behavior
/// - When reconstruction error alone is insufficient
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Latent dimensions: 4
/// - Hidden dimensions: 64
/// - Number of mixtures: 4
/// - Epochs: 100
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Zong, B., et al. (2018).
/// "Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection." ICLR.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Autoencoder)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection",
    "https://openreview.net/forum?id=BJJLHbb0-",
    Year = 2018,
    Authors = "Bo Zong, Qi Song, Martin Renqiang Min, Wei Cheng, Cristian Lumezanu, Daeki Cho, Haifeng Chen")]
public partial class DAGMMDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _latentDim;
    private readonly int _hiddenDim;
    private readonly int _numMixtures;
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

    // Estimation network weights
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _estW1;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _estB1;
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _estW2;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _estB2;

    // GMM parameters
    private Tensor<T>? _phi; // Mixture weights [numMixtures]
    private Tensor<T>? _mu; // Means [numMixtures, zDim]
    private Tensor<T>? _sigma; // Covariances [numMixtures, zDim, zDim]

    private int _inputDim;
    private int _zDim; // latent + reconstruction features

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
    /// Gets the number of GMM components.
    /// </summary>
    public int NumMixtures => _numMixtures;

    /// <summary>
    /// Creates a new DAGMM anomaly detector.
    /// </summary>
    /// <param name="latentDim">Dimensions of latent space. Default is 4.</param>
    /// <param name="hiddenDim">Dimensions of hidden layers. Default is 64.</param>
    /// <param name="numMixtures">Number of GMM components. Default is 4.</param>
    /// <param name="epochs">Number of training epochs. Default is 100.</param>
    /// <param name="learningRate">Learning rate. Default is 0.0001.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public DAGMMDetector(int latentDim = 4, int hiddenDim = 64, int numMixtures = 4,
        int epochs = 100, double learningRate = 0.0001,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (latentDim < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(latentDim),
                "Latent dimensions must be at least 1. Recommended is 4.");
        }

        if (hiddenDim < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(hiddenDim),
                "Hidden dimensions must be at least 1. Recommended is 64.");
        }

        if (numMixtures < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(numMixtures),
                "Number of mixtures must be at least 1. Recommended is 4.");
        }

        if (epochs < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(epochs),
                "Epochs must be at least 1. Recommended is 100.");
        }

        if (learningRate <= 0 || double.IsNaN(learningRate) || double.IsInfinity(learningRate))
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate),
                "Learning rate must be a positive, finite number. Recommended is 0.0001.");
        }

        _latentDim = latentDim;
        _hiddenDim = hiddenDim;
        _numMixtures = numMixtures;
        _epochs = epochs;
        _learningRate = learningRate;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        _inputDim = X.Columns;
        _zDim = _latentDim + 2; // latent + euclidean distance + cosine similarity

        // Normalize data
        var (normalizedData, means, stds) = NormalizeData(X);
        _dataMeans = means;
        _dataStds = stds;

        // Initialize weights
        InitializeWeights();

        // Initialize GMM parameters
        InitializeGMM();

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
        double scale5 = Math.Sqrt(2.0 / (_zDim + _hiddenDim));
        double scale6 = Math.Sqrt(2.0 / (_hiddenDim + _numMixtures));

        // Encoder
        _encW1 = InitializeMatrix(_inputDim, _hiddenDim, scale1);
        _encB1 = new Vector<T>(_hiddenDim);
        _encW2 = InitializeMatrix(_hiddenDim, _latentDim, scale2);
        _encB2 = new Vector<T>(_latentDim);

        // Decoder
        _decW1 = InitializeMatrix(_latentDim, _hiddenDim, scale3);
        _decB1 = new Vector<T>(_hiddenDim);
        _decW2 = InitializeMatrix(_hiddenDim, _inputDim, scale4);
        _decB2 = new Vector<T>(_inputDim);

        // Estimation network
        _estW1 = InitializeMatrix(_zDim, _hiddenDim, scale5);
        _estB1 = new Vector<T>(_hiddenDim);
        _estW2 = InitializeMatrix(_hiddenDim, _numMixtures, scale6);
        _estB2 = new Vector<T>(_numMixtures);
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

    private void InitializeGMM()
    {
        // Equal mixture weights, means uniform in [-0.05, 0.05) (one seeded engine draw, seed taken from the
        // detector's Random), identity covariances
        var engine = Engine;
        var uniform = engine.TensorUniform<T>(new[] { _numMixtures, _zDim }, 0.0, 1.0, _random.Next());
        _phi = engine.Fill(_numMixtures, NumOps.FromDouble(1.0 / _numMixtures)).AsTensor();
        _mu = engine.TensorMultiplyScalar(engine.TensorSubtractScalar(uniform, NumOps.FromDouble(0.5)), NumOps.FromDouble(0.1));
        _sigma = engine.TensorTile(engine.Reshape(engine.TensorEye<T>(_zDim), new[] { 1, _zDim, _zDim }),
            new[] { _numMixtures, 1, 1 });
    }

    private void Train(Matrix<T> data)
    {
        int n = data.Rows;
        int batchSize = Math.Min(64, n);
        var values = AnomalyTensorOps.ToTensor(data);

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            var indices = Enumerable.Range(0, n).OrderBy(_ => _random.NextDouble()).ToArray();
            var allZ = new List<Tensor<T>>();
            var allGamma = new List<Tensor<T>>();

            for (int batch = 0; batch < n; batch += batchSize)
            {
                int actualBatchSize = Math.Min(batchSize, n - batch);
                var batchRows = new int[actualBatchSize];
                Array.Copy(indices, batch, batchRows, 0, actualBatchSize);

                // Forward pass with caching; zc and gamma are stored for the GMM update
                var forward = Forward(AnomalyTensorOps.SelectRows(values, batchRows));
                allZ.Add(forward.Zc);
                allGamma.Add(forward.Gamma);

                TrainStep(forward, actualBatchSize);
            }

            // Update GMM parameters using all samples
            if (allZ.Count > 0)
            {
                UpdateGMM(Engine.Concat(allZ, 0), Engine.Concat(allGamma, 0));
            }
        }
    }

    // Back-propagates the batch, summing per-sample gradients, then applies
    // w = w - (learningRate / batch size) * clip(g, -5, 5) to every weight.
    private void TrainStep(ForwardResult forward, int actualBatchSize)
    {
        var encW1 = _encW1;
        var encB1 = _encB1;
        var encW2 = _encW2;
        var encB2 = _encB2;
        var decW1 = _decW1;
        var decB1 = _decB1;
        var decW2 = _decW2;
        var decB2 = _decB2;
        var estW1 = _estW1;
        var estB1 = _estB1;
        var estW2 = _estW2;
        var estB2 = _estB2;

        if (encW1 == null || encB1 == null || encW2 == null || encB2 == null ||
            decW1 == null || decB1 == null || decW2 == null || decB2 == null ||
            estW1 == null || estB1 == null || estW2 == null || estB2 == null)
        {
            throw new InvalidOperationException("Weights not initialized.");
        }

        var engine = Engine;
        var axis0 = new[] { 0 };
        var encW2Tensor = Tensor<T>.FromMatrix(encW2);
        var decW1Tensor = Tensor<T>.FromMatrix(decW1);
        var decW2Tensor = Tensor<T>.FromMatrix(decW2);
        var estW2Tensor = Tensor<T>.FromMatrix(estW2);

        // Reconstruction loss gradient: dL/dxRecon = 2 * (xRecon - x) / inputDim, back through the decoder
        // (tanh hidden layer), the tanh latent layer and the encoder (tanh hidden layer)
        var dXRecon = engine.TensorMultiplyScalar(engine.TensorSubtract(forward.XRecon, forward.X),
            NumOps.FromDouble(2.0 / _inputDim));
        var dDecW2 = engine.TensorMatMul(engine.TensorTranspose(forward.DecH), dXRecon);
        var dDecB2 = engine.ReduceSum(dXRecon, axis0, false);
        var dDecH = engine.TensorMultiply(engine.TensorMatMul(dXRecon, engine.TensorTranspose(decW2Tensor)),
            TanhDerivative(forward.DecH));
        var dDecW1 = engine.TensorMatMul(engine.TensorTranspose(forward.Z), dDecH);
        var dDecB1 = engine.ReduceSum(dDecH, axis0, false);
        var dZ = engine.TensorMultiply(engine.TensorMatMul(dDecH, engine.TensorTranspose(decW1Tensor)),
            TanhDerivative(forward.Z));
        var dEncW2 = engine.TensorMatMul(engine.TensorTranspose(forward.EncH), dZ);
        var dEncB2 = engine.ReduceSum(dZ, axis0, false);
        var dEncH = engine.TensorMultiply(engine.TensorMatMul(dZ, engine.TensorTranspose(encW2Tensor)),
            TanhDerivative(forward.EncH));
        var dEncW1 = engine.TensorMatMul(engine.TensorTranspose(forward.X), dEncH);
        var dEncB1 = engine.ReduceSum(dEncH, axis0, false);

        // Backprop through estimation network
        // Compute energy-based gradient for end-to-end DAGMM training
        var phi = _phi;
        Tensor<T> dGamma;
        if (phi != null && _mu != null && _sigma != null)
        {
            // Soft responsibilities exp(l_k - max) / (sum + 1e-10) of the component log-likelihoods
            // l_k = log N(zc | mu_k, sigma_k) + log(phi_k + 1e-10) under the current GMM parameters
            int rows = forward.Zc.Shape[0];
            var logPhi = engine.TensorLog(engine.TensorAddScalar(phi, NumOps.FromDouble(1e-10)));
            var logLikelihood = engine.TensorAdd(ComponentLogDensities(forward.Zc),
                AnomalyTensorOps.BroadcastRow(logPhi, rows));
            var maxLogLik = engine.ReduceMax(logLikelihood, new[] { 1 }, false);
            var expShifted = engine.TensorExp(AnomalyTensorOps.SubtractColumn(logLikelihood, maxLogLik));
            var responsibilities = AnomalyTensorOps.DivideColumn(expShifted,
                engine.TensorAddScalar(engine.ReduceSum(expShifted, new[] { 1 }, false), NumOps.FromDouble(1e-10)));

            // Energy gradient: encourage gamma to match GMM responsibilities
            // dGamma[k] = (gamma[k] - responsibility[k]) is the gradient to minimize KL divergence
            // This provides end-to-end training signal from GMM to estimation network
            // (energy weight 0.1 relative to reconstruction)
            dGamma = engine.TensorMultiplyScalar(engine.TensorSubtract(forward.Gamma, responsibilities), NumOps.FromDouble(0.1));
        }
        else
        {
            // GMM not yet initialized, use uniform target as fallback
            dGamma = engine.TensorMultiplyScalar(
                engine.TensorSubtractScalar(forward.Gamma, NumOps.FromDouble(1.0 / _numMixtures)), NumOps.FromDouble(0.01));
        }

        var dEstW2 = engine.TensorMatMul(engine.TensorTranspose(forward.EstH), dGamma);
        var dEstB2 = engine.ReduceSum(dGamma, axis0, false);
        var dEstH = engine.TensorMultiply(engine.TensorMatMul(dGamma, engine.TensorTranspose(estW2Tensor)),
            TanhDerivative(forward.EstH));
        var dEstW1 = engine.TensorMatMul(engine.TensorTranspose(forward.Zc), dEstH);
        var dEstB1 = engine.ReduceSum(dEstH, axis0, false);

        // Apply weight updates with gradient clipping
        T lr = NumOps.FromDouble(_learningRate * (1.0 / actualBatchSize));
        _encW1 = ApplyGradients(Tensor<T>.FromMatrix(encW1), dEncW1, lr).ToMatrix();
        _encB1 = ApplyGradients(encB1.AsTensor(), dEncB1, lr).ToVector();
        _encW2 = ApplyGradients(encW2Tensor, dEncW2, lr).ToMatrix();
        _encB2 = ApplyGradients(encB2.AsTensor(), dEncB2, lr).ToVector();
        _decW1 = ApplyGradients(decW1Tensor, dDecW1, lr).ToMatrix();
        _decB1 = ApplyGradients(decB1.AsTensor(), dDecB1, lr).ToVector();
        _decW2 = ApplyGradients(decW2Tensor, dDecW2, lr).ToMatrix();
        _decB2 = ApplyGradients(decB2.AsTensor(), dDecB2, lr).ToVector();
        _estW1 = ApplyGradients(Tensor<T>.FromMatrix(estW1), dEstW1, lr).ToMatrix();
        _estB1 = ApplyGradients(estB1.AsTensor(), dEstB1, lr).ToVector();
        _estW2 = ApplyGradients(estW2Tensor, dEstW2, lr).ToMatrix();
        _estB2 = ApplyGradients(estB2.AsTensor(), dEstB2, lr).ToVector();
    }

    // weights - lr * clip(gradients, -5, 5)
    private Tensor<T> ApplyGradients(Tensor<T> weights, Tensor<T> gradients, T lr)
    {
        var engine = Engine;
        T clipValue = NumOps.FromDouble(5.0);
        var clipped = engine.TensorClamp(gradients, NumOps.Negate(clipValue), clipValue);
        return engine.TensorSubtract(weights, engine.TensorMultiplyScalar(clipped, lr));
    }

    // Tanh derivative from the activation: 1 - h^2.
    private Tensor<T> TanhDerivative(Tensor<T> activation)
        => Engine.ScalarMinusTensor(NumOps.One, Engine.TensorSquare(activation));

    // input W + b (bias broadcast over the rows)
    private Tensor<T> Affine(Tensor<T> input, Matrix<T> weights, Vector<T> bias)
        => Engine.TensorAdd(Engine.TensorMatMul(input, Tensor<T>.FromMatrix(weights)),
            AnomalyTensorOps.BroadcastRow(bias.AsTensor(), input.Shape[0]));

    // Activations of a batched forward pass.
    private sealed class ForwardResult
    {
        public required Tensor<T> X { get; init; }
        public required Tensor<T> EncH { get; init; }
        public required Tensor<T> Z { get; init; }
        public required Tensor<T> DecH { get; init; }
        public required Tensor<T> XRecon { get; init; }
        public required Tensor<T> Zc { get; init; }
        public required Tensor<T> EstH { get; init; }
        public required Tensor<T> Gamma { get; init; }
    }

    // Batched forward pass. Encoder: encH = tanh(x encW1 + encB1), z = tanh(encH encW2 + encB2). Decoder:
    // decH = tanh(z decW1 + decB1), xRecon = decH decW2 + decB2 (no activation). zc = [z, ||x - xRecon||,
    // cos(x, xRecon)] (cosine 0 unless both norms exceed 1e-10). Estimation: estH = tanh(zc estW1 + estB1),
    // gamma = softmax(estH estW2 + estB2).
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
        var estW1 = _estW1;
        var estB1 = _estB1;
        var estW2 = _estW2;
        var estB2 = _estB2;

        if (encW1 == null || encB1 == null || encW2 == null || encB2 == null ||
            decW1 == null || decB1 == null || decW2 == null || decB2 == null ||
            estW1 == null || estB1 == null || estW2 == null || estB2 == null)
        {
            throw new InvalidOperationException("Weights not initialized.");
        }

        var engine = Engine;
        int rows = x.Shape[0];
        var axis1 = new[] { 1 };
        var encH = engine.Tanh(Affine(x, encW1, encB1));
        var z = engine.Tanh(Affine(encH, encW2, encB2));
        var decH = engine.Tanh(Affine(z, decW1, decB1));
        var xRecon = Affine(decH, decW2, decB2);

        // Reconstruction features
        var eucDist = engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(engine.TensorSubtract(x, xRecon)), axis1, false));
        var dotProduct = engine.ReduceSum(engine.TensorMultiply(x, xRecon), axis1, false);
        var normX = engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(x), axis1, false));
        var normRecon = engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(xRecon), axis1, false));
        T epsNorm = NumOps.FromDouble(1e-10);
        var valid = engine.TensorMultiply(engine.TensorGreaterThan(normX, epsNorm), engine.TensorGreaterThan(normRecon, epsNorm));
        var ones = Tensor<T>.CreateOnes(new[] { rows });
        var cosSim = engine.TensorWhere(valid,
            engine.TensorDivide(dotProduct, engine.TensorWhere(valid, engine.TensorMultiply(normX, normRecon), ones)),
            Tensor<T>.CreateZeros(new[] { rows }));

        // Concatenate z with reconstruction features
        var zc = engine.Concat(new[]
        {
            z, engine.Reshape(eucDist, new[] { rows, 1 }), engine.Reshape(cosSim, new[] { rows, 1 })
        }, 1);

        var estH = engine.Tanh(Affine(zc, estW1, estB1));
        var gamma = engine.TensorSoftmax(Affine(estH, estW2, estB2), 1);

        return new ForwardResult
        {
            X = x,
            EncH = encH,
            Z = z,
            DecH = decH,
            XRecon = xRecon,
            Zc = zc,
            EstH = estH,
            Gamma = gamma
        };
    }

    // GMM update from the epoch's zc rows and memberships: phi_k = mean gamma_k; mu_k = sum gamma_k zc / sum gamma_k
    // (0 when sum gamma_k is 0); sigma_k = sum gamma_k (zc - mu_k)(zc - mu_k)^T / sum gamma_k (identity when sum
    // gamma_k is 0), with the diagonal floored at 1e-6 for numerical stability.
    private void UpdateGMM(Tensor<T> allZ, Tensor<T> allGamma)
    {
        if (_phi == null || _mu == null || _sigma == null)
        {
            throw new InvalidOperationException("GMM parameters not initialized.");
        }

        var engine = Engine;
        int n = allZ.Shape[0];
        int k = _numMixtures;
        int d = _zDim;

        var sumGamma = engine.ReduceSum(allGamma, new[] { 0 }, false);
        _phi = engine.TensorDivideScalar(sumGamma, NumOps.FromDouble(n));

        var positive = engine.TensorGreaterThan(sumGamma, NumOps.Zero);
        var safeSum = engine.TensorWhere(positive, sumGamma, Tensor<T>.CreateOnes(new[] { k }));
        var weighted = AnomalyTensorOps.DivideColumn(engine.TensorMatMul(engine.TensorTranspose(allGamma), allZ), safeSum);
        var mu = engine.TensorWhere(AnomalyTensorOps.BroadcastColumn(positive, d), weighted, Tensor<T>.CreateZeros(new[] { k, d }));
        _mu = mu;

        // diffs[c, i, :] = zc_i - mu_c; covariance = (gamma-weighted diffs)^T diffs / sum gamma
        var diffs = engine.TensorSubtract(
            engine.TensorBroadcastTo(engine.Reshape(allZ, new[] { 1, n, d }), new[] { k, n, d }),
            engine.TensorBroadcastTo(engine.Reshape(mu, new[] { k, 1, d }), new[] { k, n, d }));
        var gammaWeights = engine.TensorBroadcastTo(
            engine.Reshape(engine.TensorTranspose(allGamma), new[] { k, n, 1 }), new[] { k, n, d });
        var scatter = engine.TensorBatchMatMul(engine.TensorSwapAxes(engine.TensorMultiply(diffs, gammaWeights), 1, 2), diffs);
        var covariance = engine.TensorDivide(scatter,
            engine.TensorBroadcastTo(engine.Reshape(safeSum, new[] { k, 1, 1 }), new[] { k, d, d }));
        var identity = engine.TensorBroadcastTo(engine.Reshape(engine.TensorEye<T>(d), new[] { 1, d, d }), new[] { k, d, d });
        covariance = engine.TensorWhere(
            engine.TensorBroadcastTo(engine.Reshape(positive, new[] { k, 1, 1 }), new[] { k, d, d }), covariance, identity);

        // Add regularization to diagonal for numerical stability: diag = max(1e-6, diag)
        var diagonal = LinalgDispatch<T>.Diagonal(covariance);
        var raised = engine.TensorSubtract(engine.TensorClampMin(diagonal, NumOps.FromDouble(1e-6)), diagonal);
        _sigma = engine.TensorAdd(covariance, engine.TensorMultiply(identity,
            engine.TensorBroadcastTo(engine.Reshape(raised, new[] { k, 1, d }), new[] { k, d, d })));
    }

    // log N(zc_i | mu_k, sigma_k) for every row and component ([rows, numMixtures]): log|sigma| and the
    // Mahalanobis distance from the Cholesky factor (forward substitution), or, when sigma_k is not
    // positive definite, from the diagonal of sigma_k plus 1e-10.
    private Tensor<T> ComponentLogDensities(Tensor<T> zc)
    {
        var mu = _mu;
        var sigma = _sigma;
        if (mu == null || sigma == null)
        {
            throw new InvalidOperationException("GMM parameters not initialized.");
        }

        var engine = Engine;
        int rows = zc.Shape[0];
        var axis0 = new[] { 0 };
        var axis1 = new[] { 1 };
        var componentRows = new int[1];
        var logDets = new Tensor<T>[_numMixtures];
        var mahalanobis = new Tensor<T>[_numMixtures];
        T varianceFloor = NumOps.FromDouble(1e-10);

        for (int c = 0; c < _numMixtures; c++)
        {
            componentRows[0] = c;
            var sigmaK = engine.Reshape(AnomalyTensorOps.SelectRows(engine.Reshape(sigma, new[] { _numMixtures, _zDim * _zDim }),
                componentRows), new[] { _zDim, _zDim });
            var diff = AnomalyTensorOps.SubtractRow(zc,
                engine.Reshape(AnomalyTensorOps.SelectRows(mu, componentRows), new[] { _zDim }));

            var (factor, info) = LinalgDispatch<T>.CholeskyEx(sigmaK);
            Tensor<T> mahal;
            if (info.ToArray()[0] == 0)
            {
                // Log determinant from Cholesky: log|sigma| = 2 * sum(log(L_ii)); Mahalanobis = ||L^-1 (z - mu)||^2
                // via forward substitution
                var logDiagonal = engine.TensorLog(LinalgDispatch<T>.Diagonal(factor));
                logDets[c] = engine.ReduceSum(engine.TensorAdd(logDiagonal, logDiagonal), axis0, true);
                var y = LinalgDispatch<T>.SolveTriangular(factor, engine.TensorTranspose(diff), false);
                mahal = engine.ReduceSum(engine.TensorSquare(y), axis0, false);
            }
            else
            {
                // Diagonal fallback
                var variances = engine.TensorAddScalar(LinalgDispatch<T>.Diagonal(sigmaK), varianceFloor);
                logDets[c] = engine.ReduceSum(engine.TensorLog(variances), axis0, true);
                mahal = engine.ReduceSum(AnomalyTensorOps.DivideRow(engine.TensorSquare(diff), variances), axis1, false);
            }

            mahalanobis[c] = engine.Reshape(mahal, new[] { rows, 1 });
        }

        // -0.5 * (zDim * log(2 pi) + log|sigma_k| + mahalanobis)
        var logDet = engine.TensorAddScalar(engine.Concat(logDets, 0), NumOps.FromDouble(_zDim * Math.Log(2 * Math.PI)));
        return engine.TensorMultiplyScalar(
            engine.TensorAdd(engine.Concat(mahalanobis, 1), AnomalyTensorOps.BroadcastRow(logDet, rows)), NumOps.FromDouble(-0.5));
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

        if (X.Columns != _inputDim)
        {
            throw new ArgumentException(
                $"Input has {X.Columns} features but model was trained with {_inputDim} features.",
                nameof(X));
        }

        var dataMeans = _dataMeans;
        var dataStds = _dataStds;
        if (dataMeans == null || dataStds == null)
        {
            throw new InvalidOperationException("Model not properly fitted. Normalization parameters missing.");
        }

        // Energy (anomaly score) of the normalized rows: -log(sum_k phi_k N(zc | mu_k, sigma_k) + 1e-10)
        var phi = _phi ?? throw new InvalidOperationException("GMM parameters not initialized.");
        var engine = Engine;
        var normalized = AnomalyTensorOps.DivideRow(
            AnomalyTensorOps.SubtractRow(AnomalyTensorOps.ToTensor(X), dataMeans.AsTensor()), dataStds.AsTensor());
        // computed in log space: -logsumexp(log phi_k + log N_k, log 1e-10), so no density overflows T
        var forward = Forward(normalized);
        int rows = X.Rows;
        var weighted = engine.TensorAdd(ComponentLogDensities(forward.Zc),
            AnomalyTensorOps.BroadcastRow(engine.TensorLog(phi), rows));
        var floor = Tensor<T>.CreateDefault(new[] { rows, 1 }, NumOps.FromDouble(Math.Log(1e-10)));
        var logTotal = engine.TensorLogSumExp(engine.Concat(new[] { weighted, floor }, 1), 1, false);
        return engine.TensorNegate(logTotal).ToVector();
    }
}

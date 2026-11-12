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
/// 4. A discriminator D (h = LeakyReLU(x W1 + b1), p = sigmoid(h W2 + b2)) tells real rows from
///    reconstructions; its hidden layer h is the feature map f(.) of the adversarial loss
/// 5. The generator minimizes w_adv ||f(x) - f(x')||^2 + w_con |x - x'| + w_enc ||z - z'||^2 (element means,
///    w_adv = 1, w_con = 50, w_enc = 1) and the discriminator minimizes the binary cross-entropy of real rows
///    as 1 and reconstructions as 0, both with Adam (beta1 = 0.5, beta2 = 0.999) as in the reference code
/// 6. Anomaly score is the mean squared difference between the encoding and the re-encoding
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
/// - Learning rate: 0.0002 (Adam, beta1 = 0.5)
/// - Batch size: 64
/// - Loss weights: adversarial 1, contextual 50, encoder 1
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

    // Discriminator weights (feature layer and real/fake classifier)
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _discW1;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _discB1;
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _discW2;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _discB2;

    private readonly double _adversarialWeight;
    private readonly double _contextualWeight;
    private readonly double _encoderWeight;

    // Adam moments for the 16 parameters in TrainingParameters() order, and the step count
    private Tensor<T>[]? _adamM;
    private Tensor<T>[]? _adamV;
    private int _adamStep;

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
    /// <param name="adversarialWeight">Weight w_adv of the feature-matching adversarial loss. Default is 1.</param>
    /// <param name="contextualWeight">Weight w_con of the L1 reconstruction loss. Default is 50.</param>
    /// <param name="encoderWeight">Weight w_enc of the latent (encoder) loss. Default is 1.</param>
    public GANomalyDetector(int latentDim = 32, int hiddenDim = 64, int epochs = 100,
        double learningRate = 0.0002, double contamination = 0.1, int randomSeed = 42,
        double adversarialWeight = 1, double contextualWeight = 50, double encoderWeight = 1)
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

        ValidateLossWeight(adversarialWeight, nameof(adversarialWeight), 1);
        ValidateLossWeight(contextualWeight, nameof(contextualWeight), 50);
        ValidateLossWeight(encoderWeight, nameof(encoderWeight), 1);

        _latentDim = latentDim;
        _hiddenDim = hiddenDim;
        _epochs = epochs;
        _learningRate = learningRate;
        _adversarialWeight = adversarialWeight;
        _contextualWeight = contextualWeight;
        _encoderWeight = encoderWeight;
    }

    /// <summary>
    /// Creates a new GANomaly anomaly detector with the default loss weights (1 adversarial, 50 contextual,
    /// 1 encoder); keeps the shipped six-parameter signature binary compatible.
    /// </summary>
    /// <param name="latentDim">Dimensions of latent space.</param>
    /// <param name="hiddenDim">Dimensions of hidden layers.</param>
    /// <param name="epochs">Number of training epochs.</param>
    /// <param name="learningRate">Learning rate.</param>
    /// <param name="contamination">Expected proportion of anomalies.</param>
    /// <param name="randomSeed">Random seed for reproducibility.</param>
    public GANomalyDetector(int latentDim, int hiddenDim, int epochs, double learningRate, double contamination,
        int randomSeed)
        : this(latentDim, hiddenDim, epochs, learningRate, contamination, randomSeed, 1, 50, 1)
    {
    }

    // A loss weight must be finite and non-negative.
    private static void ValidateLossWeight(double weight, string name, double recommended)
    {
        if (double.IsNaN(weight) || double.IsInfinity(weight) || weight < 0)
        {
            throw new ArgumentOutOfRangeException(name,
                $"Loss weight must be finite and non-negative. Recommended is {recommended}.");
        }
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

        // Discriminator: input -> hidden (features) -> 1 (real/fake logit)
        _discW1 = InitializeMatrix(_inputDim, _hiddenDim, scale1);
        _discB1 = new Vector<T>(_hiddenDim);
        _discW2 = InitializeMatrix(_hiddenDim, 1, Math.Sqrt(2.0 / (_hiddenDim + 1)));
        _discB2 = new Vector<T>(1);

        _adamM = null;
        _adamV = null;
        _adamStep = 0;
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
        int batchSize = Math.Min(64, n);
        var values = AnomalyTensorOps.ToTensor(data);

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            var indices = Enumerable.Range(0, n).OrderBy(_ => _random.NextDouble()).ToArray();

            for (int batch = 0; batch < n; batch += batchSize)
            {
                int actualBatchSize = Math.Min(batchSize, n - batch);
                var batchRows = new int[actualBatchSize];
                Array.Copy(indices, batch, batchRows, 0, actualBatchSize);

                // Generator and discriminator gradients from the same forward pass (the reconstruction is
                // detached for the discriminator), then one Adam step on every parameter
                AdamUpdate(TrainingGradients(AnomalyTensorOps.SelectRows(values, batchRows)));
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

    // Discriminator activations: pre-activation and features of the hidden layer, and the real/fake logit.
    private sealed class DiscriminatorResult
    {
        public required Tensor<T> HiddenPre { get; init; }
        public required Tensor<T> Features { get; init; }
        public required Tensor<T> Logit { get; init; }
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

    private DiscriminatorResult Discriminate(Tensor<T> x)
    {
        var discW1 = _discW1;
        var discB1 = _discB1;
        var discW2 = _discW2;
        var discB2 = _discB2;
        if (discW1 == null || discB1 == null || discW2 == null || discB2 == null)
        {
            throw new InvalidOperationException("Discriminator weights not initialized.");
        }

        var hiddenPre = Affine(x, discW1, discB1);
        var features = Engine.LeakyReLU(hiddenPre, NumOps.FromDouble(0.2));
        return new DiscriminatorResult
        {
            HiddenPre = hiddenPre,
            Features = features,
            Logit = Affine(features, discW2, discB2)
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

    /// <summary>
    /// The generator loss of Akcay et al. (2018), Eq. 4: w_adv mean((f(x) - f(x'))^2) + w_con mean(|x - x'|)
    /// + w_enc mean((z - z')^2), each mean over all elements of the batch, with f the discriminator features.
    /// </summary>
    internal T GeneratorLoss(Tensor<T> x)
    {
        var engine = Engine;
        var forward = Forward(x);
        T adversarial = engine.TensorMean(engine.TensorSquare(engine.TensorSubtract(
            Discriminate(forward.XRecon).Features, Discriminate(x).Features)));
        T contextual = engine.TensorMean(engine.TensorAbs(engine.TensorSubtract(forward.XRecon, x)));
        T encoder = engine.TensorMean(engine.TensorSquare(engine.TensorSubtract(forward.ZRecon, forward.Z)));
        return NumOps.Add(NumOps.Add(NumOps.Multiply(NumOps.FromDouble(_adversarialWeight), adversarial),
            NumOps.Multiply(NumOps.FromDouble(_contextualWeight), contextual)),
            NumOps.Multiply(NumOps.FromDouble(_encoderWeight), encoder));
    }

    /// <summary>
    /// The discriminator loss of the reference code: (BCE(D(x), 1) + BCE(D(x'), 0)) / 2, each a batch mean.
    /// </summary>
    internal T DiscriminatorLoss(Tensor<T> x)
    {
        var engine = Engine;
        var forward = Forward(x);
        T real = engine.TensorMean(engine.Softplus(engine.TensorNegate(Discriminate(x).Logit)));
        T fake = engine.TensorMean(engine.Softplus(Discriminate(forward.XRecon).Logit));
        return NumOps.Multiply(NumOps.Add(real, fake), NumOps.FromDouble(0.5));
    }

    /// <summary>
    /// The parameters in a fixed order: encoder W1, b1, W2, b2, decoder W1, b1, W2, b2, re-encoder W1, b1,
    /// W2, b2, discriminator W1, b1, W2, b2.
    /// </summary>
    internal Tensor<T>[] TrainingParameters()
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
        var discW1 = _discW1;
        var discB1 = _discB1;
        var discW2 = _discW2;
        var discB2 = _discB2;

        if (encW1 == null || encB1 == null || encW2 == null || encB2 == null ||
            decW1 == null || decB1 == null || decW2 == null || decB2 == null ||
            reEncW1 == null || reEncB1 == null || reEncW2 == null || reEncB2 == null ||
            discW1 == null || discB1 == null || discW2 == null || discB2 == null)
        {
            throw new InvalidOperationException("Weights not initialized.");
        }

        return new[]
        {
            Tensor<T>.FromMatrix(encW1), encB1.AsTensor(), Tensor<T>.FromMatrix(encW2), encB2.AsTensor(),
            Tensor<T>.FromMatrix(decW1), decB1.AsTensor(), Tensor<T>.FromMatrix(decW2), decB2.AsTensor(),
            Tensor<T>.FromMatrix(reEncW1), reEncB1.AsTensor(), Tensor<T>.FromMatrix(reEncW2), reEncB2.AsTensor(),
            Tensor<T>.FromMatrix(discW1), discB1.AsTensor(), Tensor<T>.FromMatrix(discW2), discB2.AsTensor()
        };
    }

    /// <summary>
    /// Replaces the parameters, given in <see cref="TrainingParameters"/> order.
    /// </summary>
    internal void SetTrainingParameters(Tensor<T>[] parameters)
    {
        if (parameters is null || parameters.Length != 16)
        {
            throw new ArgumentException("Expected the 16 GANomaly parameters.", nameof(parameters));
        }

        _encW1 = parameters[0].ToMatrix();
        _encB1 = parameters[1].ToVector();
        _encW2 = parameters[2].ToMatrix();
        _encB2 = parameters[3].ToVector();
        _decW1 = parameters[4].ToMatrix();
        _decB1 = parameters[5].ToVector();
        _decW2 = parameters[6].ToMatrix();
        _decB2 = parameters[7].ToVector();
        _reEncW1 = parameters[8].ToMatrix();
        _reEncB1 = parameters[9].ToVector();
        _reEncW2 = parameters[10].ToMatrix();
        _reEncB2 = parameters[11].ToVector();
        _discW1 = parameters[12].ToMatrix();
        _discB1 = parameters[13].ToVector();
        _discW2 = parameters[14].ToMatrix();
        _discB2 = parameters[15].ToVector();
    }

    /// <summary>
    /// Gradients in <see cref="TrainingParameters"/> order: the 12 generator gradients of <see cref="GeneratorLoss"/>
    /// (the discriminator is held fixed) and the 4 discriminator gradients of <see cref="DiscriminatorLoss"/>
    /// (the reconstruction is detached, as in the reference code).
    /// </summary>
    internal Tensor<T>[] TrainingGradients(Tensor<T> x)
    {
        var p = TrainingParameters();
        var engine = Engine;
        var axis0 = new[] { 0 };
        int batch = x.Shape[0];
        var forward = Forward(x);
        var real = Discriminate(x);
        var fake = Discriminate(forward.XRecon);

        // === Generator: adversarial feature matching, w_adv mean((f(x') - f(x))^2) ===
        T adversarialScale = NumOps.FromDouble(2 * _adversarialWeight / ((double)batch * _hiddenDim));
        var dFakeFeatures = engine.TensorMultiplyScalar(engine.TensorSubtract(fake.Features, real.Features), adversarialScale);
        var dFakePre = engine.TensorMultiply(dFakeFeatures, LeakySlope(fake.HiddenPre));
        var dXReconAdversarial = engine.TensorMatMul(dFakePre, engine.TensorTranspose(p[12]));

        // === Contextual loss: w_con mean(|x' - x|) ===
        var dXReconContextual = engine.TensorMultiplyScalar(engine.TensorSign(engine.TensorSubtract(forward.XRecon, x)),
            NumOps.FromDouble(_contextualWeight / ((double)batch * _inputDim)));

        // === Encoder loss: w_enc mean((z' - z)^2), through the re-encoder back into the reconstruction ===
        var dZRecon = engine.TensorMultiplyScalar(engine.TensorSubtract(forward.ZRecon, forward.Z),
            NumOps.FromDouble(2 * _encoderWeight / ((double)batch * _latentDim)));
        var gReEncW2 = engine.TensorMatMul(engine.TensorTranspose(forward.ReEncHidden), dZRecon);
        var gReEncB2 = engine.ReduceSum(dZRecon, axis0, false);
        var dReEncH1 = engine.TensorMultiply(engine.TensorMatMul(dZRecon, engine.TensorTranspose(p[10])),
            LeakySlope(forward.ReEncHiddenPre));
        var gReEncW1 = engine.TensorMatMul(engine.TensorTranspose(forward.XRecon), dReEncH1);
        var gReEncB1 = engine.ReduceSum(dReEncH1, axis0, false);
        var dXReconEncoder = engine.TensorMatMul(dReEncH1, engine.TensorTranspose(p[8]));

        // === Decoder ===
        var dXRecon = engine.TensorAdd(engine.TensorAdd(dXReconAdversarial, dXReconContextual), dXReconEncoder);
        var gDecW2 = engine.TensorMatMul(engine.TensorTranspose(forward.DecHidden), dXRecon);
        var gDecB2 = engine.ReduceSum(dXRecon, axis0, false);
        var dDecH1 = engine.TensorMultiply(engine.TensorMatMul(dXRecon, engine.TensorTranspose(p[6])),
            LeakySlope(forward.DecHiddenPre));
        var gDecW1 = engine.TensorMatMul(engine.TensorTranspose(forward.Z), dDecH1);
        var gDecB1 = engine.ReduceSum(dDecH1, axis0, false);

        // === Encoder: decoder path plus the direct -dZRecon of the encoder loss ===
        var dZ = engine.TensorSubtract(engine.TensorMatMul(dDecH1, engine.TensorTranspose(p[4])), dZRecon);
        var gEncW2 = engine.TensorMatMul(engine.TensorTranspose(forward.EncHidden), dZ);
        var gEncB2 = engine.ReduceSum(dZ, axis0, false);
        var dEncH1 = engine.TensorMultiply(engine.TensorMatMul(dZ, engine.TensorTranspose(p[2])),
            LeakySlope(forward.EncHiddenPre));
        var gEncW1 = engine.TensorMatMul(engine.TensorTranspose(x), dEncH1);
        var gEncB1 = engine.ReduceSum(dEncH1, axis0, false);

        // === Discriminator: (mean softplus(-logit_real) + mean softplus(logit_fake)) / 2 ===
        T half = NumOps.FromDouble(0.5 / batch);
        var dRealLogit = engine.TensorMultiplyScalar(engine.TensorSubtractScalar(engine.TensorSigmoid(real.Logit), NumOps.One), half);
        var dFakeLogit = engine.TensorMultiplyScalar(engine.TensorSigmoid(fake.Logit), half);
        var gDiscW2 = engine.TensorAdd(engine.TensorMatMul(engine.TensorTranspose(real.Features), dRealLogit),
            engine.TensorMatMul(engine.TensorTranspose(fake.Features), dFakeLogit));
        var gDiscB2 = engine.TensorAdd(engine.ReduceSum(dRealLogit, axis0, false), engine.ReduceSum(dFakeLogit, axis0, false));
        var discW2Transposed = engine.TensorTranspose(p[14]);
        var dRealPre = engine.TensorMultiply(engine.TensorMatMul(dRealLogit, discW2Transposed), LeakySlope(real.HiddenPre));
        var dFakeDiscPre = engine.TensorMultiply(engine.TensorMatMul(dFakeLogit, discW2Transposed), LeakySlope(fake.HiddenPre));
        var gDiscW1 = engine.TensorAdd(engine.TensorMatMul(engine.TensorTranspose(x), dRealPre),
            engine.TensorMatMul(engine.TensorTranspose(forward.XRecon), dFakeDiscPre));
        var gDiscB1 = engine.TensorAdd(engine.ReduceSum(dRealPre, axis0, false), engine.ReduceSum(dFakeDiscPre, axis0, false));

        return new[]
        {
            gEncW1, gEncB1, gEncW2, gEncB2, gDecW1, gDecB1, gDecW2, gDecB2,
            gReEncW1, gReEncB1, gReEncW2, gReEncB2, gDiscW1, gDiscB1, gDiscW2, gDiscB2
        };
    }

    // Adam (beta1 = 0.5, beta2 = 0.999, eps = 1e-8, the reference code's optimizer) on every parameter:
    // m = b1 m + (1 - b1) g, v = b2 v + (1 - b2) g^2, w = w - lr (m / (1 - b1^t)) / (sqrt(v / (1 - b2^t)) + eps)
    private void AdamUpdate(Tensor<T>[] gradients)
    {
        const double beta1 = 0.5;
        const double beta2 = 0.999;
        var engine = Engine;
        var parameters = TrainingParameters();
        var adamM = _adamM;
        var adamV = _adamV;
        if (adamM is null || adamV is null)
        {
            adamM = new Tensor<T>[parameters.Length];
            adamV = new Tensor<T>[parameters.Length];
            for (int i = 0; i < parameters.Length; i++)
            {
                adamM[i] = Tensor<T>.CreateZeros(parameters[i].Shape.ToArray());
                adamV[i] = Tensor<T>.CreateZeros(parameters[i].Shape.ToArray());
            }

            _adamM = adamM;
            _adamV = adamV;
        }

        _adamStep++;
        T b1 = NumOps.FromDouble(beta1);
        T oneMinusB1 = NumOps.FromDouble(1 - beta1);
        T b2 = NumOps.FromDouble(beta2);
        T oneMinusB2 = NumOps.FromDouble(1 - beta2);
        T firstCorrection = NumOps.FromDouble(1 - Math.Pow(beta1, _adamStep));
        T secondCorrection = NumOps.FromDouble(1 - Math.Pow(beta2, _adamStep));
        T lr = NumOps.FromDouble(_learningRate);
        T eps = NumOps.FromDouble(1e-8);
        var updated = new Tensor<T>[parameters.Length];
        for (int i = 0; i < parameters.Length; i++)
        {
            adamM[i] = engine.TensorAdd(engine.TensorMultiplyScalar(adamM[i], b1), engine.TensorMultiplyScalar(gradients[i], oneMinusB1));
            adamV[i] = engine.TensorAdd(engine.TensorMultiplyScalar(adamV[i], b2),
                engine.TensorMultiplyScalar(engine.TensorSquare(gradients[i]), oneMinusB2));
            var mHat = engine.TensorDivideScalar(adamM[i], firstCorrection);
            var vHat = engine.TensorDivideScalar(adamV[i], secondCorrection);
            var step = engine.TensorDivide(mHat, engine.TensorAddScalar(engine.TensorSqrt(vHat), eps));
            updated[i] = engine.TensorSubtract(parameters[i], engine.TensorMultiplyScalar(step, lr));
        }

        SetTrainingParameters(updated);
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

        // Anomaly score (reference code): mean((z - zRecon)^2) over the latent dimensions after encode, decode,
        // re-encode of the normalized rows
        var engine = Engine;
        var normalized = AnomalyTensorOps.DivideRow(
            AnomalyTensorOps.SubtractRow(AnomalyTensorOps.ToTensor(X), dataMeans.AsTensor()), dataStds.AsTensor());
        var forward = Forward(normalized);
        return engine.ReduceMean(engine.TensorSquare(engine.TensorSubtract(forward.Z, forward.ZRecon)),
            new[] { 1 }, false).ToVector();
    }
}

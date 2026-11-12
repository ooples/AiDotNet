using AiDotNet.Tensors.Engines;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.NeuralNetwork;

/// <summary>
/// Implements AnoGAN (Anomaly Detection with Generative Adversarial Networks).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> AnoGAN trains a GAN to generate normal data, then detects
/// anomalies by finding the latent code that best reconstructs a query point.
/// Points that cannot be well-reconstructed are anomalies.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Train a GAN (Generator + Discriminator) on normal data
/// 2. For anomaly scoring, find z that minimizes reconstruction error
/// 3. Anomaly score combines reconstruction loss and discriminator feature loss
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Image anomaly detection
/// - When you have only normal examples for training
/// - High-dimensional data where reconstruction quality matters
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Latent dimensions: 64
/// - Hidden dimensions: 128
/// - Epochs: 100
/// - Learning rate: 0.0002
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Schlegl, T., Seeböck, P., Waldstein, S. M., Schmidt-Erfurth, U., and Langs, G. (2017).
/// "Unsupervised Anomaly Detection with Generative Adversarial Networks." IPMI.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.GAN)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Unsupervised Anomaly Detection with Generative Adversarial Networks", "https://doi.org/10.1007/978-3-319-59050-9_12", Year = 2017, Authors = "Thomas Schlegl, Philipp Seeböck, Sebastian M. Waldstein, Ursula Schmidt-Erfurth, Georg Langs")]
public partial class AnoGANDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _latentDim;
    private readonly int _hiddenDim;
    private readonly int _epochs;
    private readonly double _learningRate;
    private readonly int _inferenceSteps;

    // Generator weights
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _genW1;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _genB1;
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _genW2;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _genB2;
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _genW3;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _genB3;

    // Discriminator weights
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _discW1;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _discB1;
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _discW2;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _discB2;
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _discW3;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _discB3;

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
    /// Creates a new AnoGAN anomaly detector.
    /// </summary>
    /// <param name="latentDim">Dimensions of latent space. Default is 64.</param>
    /// <param name="hiddenDim">Dimensions of hidden layers. Default is 128.</param>
    /// <param name="epochs">Number of training epochs. Default is 100.</param>
    /// <param name="learningRate">Learning rate. Default is 0.0002.</param>
    /// <param name="inferenceSteps">Steps to find optimal z during scoring. Default is 100.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public AnoGANDetector(int latentDim = 64, int hiddenDim = 128, int epochs = 100,
        double learningRate = 0.0002, int inferenceSteps = 100,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (latentDim < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(latentDim),
                "Latent dimensions must be at least 1. Recommended is 64.");
        }

        if (hiddenDim < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(hiddenDim),
                "Hidden dimensions must be at least 1. Recommended is 128.");
        }

        if (epochs < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(epochs),
                "Epochs must be at least 1. Recommended is 100.");
        }

        if (inferenceSteps < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(inferenceSteps),
                "Inference steps must be at least 1. Recommended is 100.");
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
        _inferenceSteps = inferenceSteps;
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

        // Train GAN
        TrainGAN(normalizedData);

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
        // Generator: z -> hidden1 -> hidden2 -> output
        double genScale1 = Math.Sqrt(2.0 / (_latentDim + _hiddenDim));
        double genScale2 = Math.Sqrt(2.0 / (_hiddenDim + _hiddenDim));
        double genScale3 = Math.Sqrt(2.0 / (_hiddenDim + _inputDim));

        _genW1 = InitializeMatrix(_latentDim, _hiddenDim, genScale1);
        _genB1 = new Vector<T>(_hiddenDim);
        _genW2 = InitializeMatrix(_hiddenDim, _hiddenDim, genScale2);
        _genB2 = new Vector<T>(_hiddenDim);
        _genW3 = InitializeMatrix(_hiddenDim, _inputDim, genScale3);
        _genB3 = new Vector<T>(_inputDim);

        // Discriminator: input -> hidden1 -> hidden2 -> 1
        double discScale1 = Math.Sqrt(2.0 / (_inputDim + _hiddenDim));
        double discScale2 = Math.Sqrt(2.0 / (_hiddenDim + _hiddenDim));
        double discScale3 = Math.Sqrt(2.0 / (_hiddenDim + 1));

        _discW1 = InitializeMatrix(_inputDim, _hiddenDim, discScale1);
        _discB1 = new Vector<T>(_hiddenDim);
        _discW2 = InitializeMatrix(_hiddenDim, _hiddenDim, discScale2);
        _discB2 = new Vector<T>(_hiddenDim);
        _discW3 = InitializeMatrix(_hiddenDim, 1, discScale3);
        _discB3 = new Vector<T>(1);
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

    private void TrainGAN(Matrix<T> data)
    {
        int n = data.Rows;
        int batchSize = Math.Min(32, n);
        var values = AnomalyTensorOps.ToTensor(data);
        T lr = NumOps.FromDouble(_learningRate);

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            var indices = Enumerable.Range(0, n).OrderBy(_ => _random.NextDouble()).ToArray();

            for (int batch = 0; batch < n; batch += batchSize)
            {
                int actualBatchSize = Math.Min(batchSize, n - batch);

                // Train Discriminator: one SGD step per sample (real row, then a fake row generated from a
                // fresh latent code). The generator does not change during this phase, so the fakes are
                // generated together from latent codes drawn in sample order.
                var (_, _, fakeBatch) = Generate(SampleLatent(actualBatchSize));
                for (int b = 0; b < actualBatchSize; b++)
                {
                    var realData = AnomalyTensorOps.SelectRows(values, new[] { indices[batch + b] });
                    var fakeData = AnomalyTensorOps.SelectRows(fakeBatch, new[] { b });
                    UpdateDiscriminator(realData, fakeData, lr);
                }

                // Train Generator: one SGD step per sample, latent codes drawn in sample order
                var latentBatch = SampleLatent(actualBatchSize);
                for (int b = 0; b < actualBatchSize; b++)
                {
                    UpdateGenerator(AnomalyTensorOps.SelectRows(latentBatch, new[] { b }), lr);
                }
            }
        }
    }

    // [rows, latentDim] standard normal latent codes from the model's shared random stream.
    private Tensor<T> SampleLatent(int rows) => SampleLatent(_random, rows);

    /// <summary>
    /// Draws a latent code from the standard normal using <paramref name="rng"/>.
    /// </summary>
    /// <remarks>
    /// Training draws from the model's shared <c>_random</c>, where advancing the stream across
    /// steps is the point. INFERENCE must not: Schlegl et al. (2017) start the latent search from a
    /// random z and refine it by backpropagation, so the restart is part of the algorithm, but a
    /// detector constructed with an explicit <c>randomSeed</c> promises reproducible scores. Drawing
    /// the query-time restart from the shared stream broke that promise silently -
    /// <c>Predict_ShouldBeDeterministic</c> and <c>Clone_ShouldProduceSameScores</c> both failed
    /// because a second call to the same model on the same input resumed the stream further along.
    /// </remarks>
    private Tensor<T> SampleLatent(Random rng, int rows)
    {
        // One latent code per row, drawn in row order (latent unit by latent unit within a row)
        if (rows == 0)
        {
            return Tensor<T>.CreateZeros(new[] { 0, _latentDim });
        }

        var codes = new Tensor<T>[rows];
        for (int r = 0; r < rows; r++)
        {
            codes[r] = SampleLatent(rng).AsTensor();
        }

        return Engine.TensorStack(codes, 0);
    }

    /// <summary>
    /// Draws one latent code from the standard normal using <paramref name="rng"/>.
    /// </summary>
    private Vector<T> SampleLatent(Random rng)
    {
        var z = new Vector<T>(_latentDim);
        for (int i = 0; i < _latentDim; i++)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double val = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            z[i] = NumOps.FromDouble(val);
        }
        return z;
    }

    // Batched generator: h1 = LeakyReLU(z genW1 + genB1), h2 = LeakyReLU(h1 genW2 + genB2),
    // output = tanh(h2 genW3 + genB3).
    private (Tensor<T> h1, Tensor<T> h2, Tensor<T> output) Generate(Tensor<T> z)
    {
        var genW1 = _genW1;
        var genB1 = _genB1;
        var genW2 = _genW2;
        var genB2 = _genB2;
        var genW3 = _genW3;
        var genB3 = _genB3;

        if (genW1 == null || genB1 == null || genW2 == null || genB2 == null ||
            genW3 == null || genB3 == null)
        {
            throw new InvalidOperationException("Generator weights not initialized.");
        }

        var engine = Engine;
        T leakyAlpha = NumOps.FromDouble(0.2);
        var h1 = engine.LeakyReLU(Affine(z, genW1, genB1), leakyAlpha);
        var h2 = engine.LeakyReLU(Affine(h1, genW2, genB2), leakyAlpha);
        var output = engine.Tanh(Affine(h2, genW3, genB3));
        return (h1, h2, output);
    }

    // Batched discriminator: h1 = LeakyReLU(x discW1 + discB1), h2 = LeakyReLU(h1 discW2 + discB2) (the
    // features), output = sigmoid(h2 discW3 + discB3) = 1 / (1 + exp(-pre)), [rows, 1].
    private (Tensor<T> h1, Tensor<T> h2, Tensor<T> output) Discriminate(Tensor<T> x)
    {
        var discW1 = _discW1;
        var discB1 = _discB1;
        var discW2 = _discW2;
        var discB2 = _discB2;
        var discW3 = _discW3;
        var discB3 = _discB3;

        if (discW1 == null || discB1 == null || discW2 == null || discB2 == null ||
            discW3 == null || discB3 == null)
        {
            throw new InvalidOperationException("Discriminator weights not initialized.");
        }

        var engine = Engine;
        T leakyAlpha = NumOps.FromDouble(0.2);
        var h1 = engine.LeakyReLU(Affine(x, discW1, discB1), leakyAlpha);
        var h2 = engine.LeakyReLU(Affine(h1, discW2, discB2), leakyAlpha);
        var output = engine.TensorReciprocal(engine.TensorAddScalar(
            engine.TensorExp(engine.TensorNegate(Affine(h2, discW3, discB3))), NumOps.One));
        return (h1, h2, output);
    }

    // input W + b (bias broadcast over the rows)
    private Tensor<T> Affine(Tensor<T> input, Matrix<T> weights, Vector<T> bias)
        => Engine.TensorAdd(Engine.TensorMatMul(input, Tensor<T>.FromMatrix(weights)),
            AnomalyTensorOps.BroadcastRow(bias.AsTensor(), input.Shape[0]));

    // LeakyReLU derivative from the activation (post-activation sign == pre-activation sign): 0.2 where
    // negative, 1 elsewhere.
    private Tensor<T> LeakySlope(Tensor<T> activation)
        => Engine.TensorAddScalar(Engine.TensorMultiplyScalar(
            Engine.TensorLessThan(activation, NumOps.Zero), NumOps.FromDouble(-0.8)), NumOps.One);

    // weights - lr * gradients
    private Tensor<T> Step(Tensor<T> weights, Tensor<T> gradients, T lr)
        => Engine.TensorSubtract(weights, Engine.TensorMultiplyScalar(gradients, lr));

    private void UpdateDiscriminator(Tensor<T> realData, Tensor<T> fakeData, T lr)
    {
        // Forward pass for real data with cache
        var (realH1, realH2, realOut) = Discriminate(realData);
        // Forward pass for fake data with cache
        var (fakeH1, fakeH2, fakeOut) = Discriminate(fakeData);

        // Binary cross-entropy gradients
        var dRealOut = Engine.TensorSubtractScalar(realOut, NumOps.One);
        var dFakeOut = fakeOut;

        // Backprop through discriminator for real data
        BackpropDiscriminator(realData, realH1, realH2, dRealOut, lr);
        // Backprop through discriminator for fake data
        BackpropDiscriminator(fakeData, fakeH1, fakeH2, dFakeOut, lr);
    }

    // Gradients of the discriminator output (seed dOut, [rows, 1]) computed with the current weights,
    // then W -= lr * dW for every layer.
    private void BackpropDiscriminator(Tensor<T> x, Tensor<T> h1, Tensor<T> h2, Tensor<T> dOut, T lr)
    {
        var discW1 = _discW1;
        var discB1 = _discB1;
        var discW2 = _discW2;
        var discB2 = _discB2;
        var discW3 = _discW3;
        var discB3 = _discB3;

        if (discW1 == null || discB1 == null || discW2 == null || discB2 == null ||
            discW3 == null || discB3 == null)
        {
            throw new InvalidOperationException("Discriminator weights not initialized.");
        }

        var engine = Engine;
        var axis0 = new[] { 0 };
        var w2Tensor = Tensor<T>.FromMatrix(discW2);
        var w3Tensor = Tensor<T>.FromMatrix(discW3);

        // === Phase 1: Compute ALL gradients using original weights ===
        var dH2 = engine.TensorMultiply(engine.TensorMatMul(dOut, engine.TensorTranspose(w3Tensor)), LeakySlope(h2));
        var dH1 = engine.TensorMultiply(engine.TensorMatMul(dH2, engine.TensorTranspose(w2Tensor)), LeakySlope(h1));

        // === Phase 2: Compute weight gradients and apply updates ===
        _discW3 = Step(w3Tensor, engine.TensorMatMul(engine.TensorTranspose(h2), dOut), lr).ToMatrix();
        _discB3 = Step(discB3.AsTensor(), engine.ReduceSum(dOut, axis0, false), lr).ToVector();
        _discW2 = Step(w2Tensor, engine.TensorMatMul(engine.TensorTranspose(h1), dH2), lr).ToMatrix();
        _discB2 = Step(discB2.AsTensor(), engine.ReduceSum(dH2, axis0, false), lr).ToVector();
        _discW1 = Step(Tensor<T>.FromMatrix(discW1), engine.TensorMatMul(engine.TensorTranspose(x), dH1), lr).ToMatrix();
        _discB1 = Step(discB1.AsTensor(), engine.ReduceSum(dH1, axis0, false), lr).ToVector();
    }

    private void UpdateGenerator(Tensor<T> z, T lr)
    {
        // Forward pass through generator with cache
        var (h1, h2, fakeData) = Generate(z);

        // Forward pass through discriminator
        var (discH1, discH2, fakeOut) = Discriminate(fakeData);

        // Generator wants discriminator to output 1 for fake
        var dOut = Engine.TensorSubtractScalar(fakeOut, NumOps.One);

        // Backprop through discriminator to get gradient w.r.t. fakeData
        var discW3 = _discW3 ?? throw new InvalidOperationException("Discriminator weights not initialized.");
        var dFakeData = DiscriminatorInputGradient(discH1, discH2,
            Engine.TensorMatMul(dOut, Engine.TensorTranspose(Tensor<T>.FromMatrix(discW3))));

        // Backprop through generator
        BackpropGenerator(z, h1, h2, fakeData, dFakeData, lr);
    }

    // Gradients of the generator output (seed dOutput through the tanh) computed with the current weights,
    // then W -= lr * dW for every layer.
    private void BackpropGenerator(Tensor<T> z, Tensor<T> h1, Tensor<T> h2, Tensor<T> output, Tensor<T> dOutput, T lr)
    {
        var genW1 = _genW1;
        var genB1 = _genB1;
        var genW2 = _genW2;
        var genB2 = _genB2;
        var genW3 = _genW3;
        var genB3 = _genB3;

        if (genW1 == null || genB1 == null || genW2 == null || genB2 == null ||
            genW3 == null || genB3 == null)
        {
            throw new InvalidOperationException("Generator weights not initialized.");
        }

        var engine = Engine;
        var axis0 = new[] { 0 };
        var w2Tensor = Tensor<T>.FromMatrix(genW2);
        var w3Tensor = Tensor<T>.FromMatrix(genW3);

        // === Phase 1: Compute ALL gradients using original weights ===
        var dOutputPre = TanhGradient(output, dOutput);
        var dH2 = engine.TensorMultiply(engine.TensorMatMul(dOutputPre, engine.TensorTranspose(w3Tensor)), LeakySlope(h2));
        var dH1 = engine.TensorMultiply(engine.TensorMatMul(dH2, engine.TensorTranspose(w2Tensor)), LeakySlope(h1));

        // === Phase 2: Compute weight gradients and apply updates ===
        _genW3 = Step(w3Tensor, engine.TensorMatMul(engine.TensorTranspose(h2), dOutputPre), lr).ToMatrix();
        _genB3 = Step(genB3.AsTensor(), engine.ReduceSum(dOutputPre, axis0, false), lr).ToVector();
        _genW2 = Step(w2Tensor, engine.TensorMatMul(engine.TensorTranspose(h1), dH2), lr).ToMatrix();
        _genB2 = Step(genB2.AsTensor(), engine.ReduceSum(dH2, axis0, false), lr).ToVector();
        _genW1 = Step(Tensor<T>.FromMatrix(genW1), engine.TensorMatMul(engine.TensorTranspose(z), dH1), lr).ToMatrix();
        _genB1 = Step(genB1.AsTensor(), engine.ReduceSum(dH1, axis0, false), lr).ToVector();
    }

    // Gradient through the tanh output layer: dOutput * (1 - output^2).
    private Tensor<T> TanhGradient(Tensor<T> output, Tensor<T> dOutput)
        => Engine.TensorMultiply(dOutput, Engine.ScalarMinusTensor(NumOps.One, Engine.TensorSquare(output)));

    /// <summary>
    /// Gradient of a loss w.r.t. the generator input z, given its gradient <paramref name="dOutput"/> w.r.t.
    /// the generator output: h1 = LeakyReLU(z·W1+b1), h2 = LeakyReLU(h1·W2+b2), out = tanh(h2·W3+b3).
    /// </summary>
    private Tensor<T> GeneratorInputGradient(Tensor<T> h1, Tensor<T> h2, Tensor<T> output, Tensor<T> dOutput)
    {
        var genW1 = _genW1;
        var genW2 = _genW2;
        var genW3 = _genW3;
        if (genW1 == null || genW2 == null || genW3 == null)
        {
            throw new InvalidOperationException("Generator weights not initialized.");
        }

        var engine = Engine;
        var dPre3 = TanhGradient(output, dOutput);
        var dH2 = engine.TensorMultiply(engine.TensorMatMul(dPre3, engine.TensorTranspose(Tensor<T>.FromMatrix(genW3))), LeakySlope(h2));
        var dH1 = engine.TensorMultiply(engine.TensorMatMul(dH2, engine.TensorTranspose(Tensor<T>.FromMatrix(genW2))), LeakySlope(h1));
        return engine.TensorMatMul(dH1, engine.TensorTranspose(Tensor<T>.FromMatrix(genW1)));
    }

    /// <summary>
    /// Gradient w.r.t. the discriminator input, given the gradient <paramref name="dH2Seed"/> w.r.t. the
    /// post-activation features h2 (LeakyReLU derivatives on h2 and h1, then through W2 and W1).
    /// </summary>
    private Tensor<T> DiscriminatorInputGradient(Tensor<T> h1, Tensor<T> h2, Tensor<T> dH2Seed)
    {
        var discW1 = _discW1;
        var discW2 = _discW2;
        if (discW1 == null || discW2 == null)
        {
            throw new InvalidOperationException("Discriminator weights not initialized.");
        }

        var engine = Engine;
        var dH2 = engine.TensorMultiply(dH2Seed, LeakySlope(h2));
        var dH1 = engine.TensorMultiply(engine.TensorMatMul(dH2, engine.TensorTranspose(Tensor<T>.FromMatrix(discW2))), LeakySlope(h1));
        return engine.TensorMatMul(dH1, engine.TensorTranspose(Tensor<T>.FromMatrix(discW1)));
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
                $"Input has {X.Columns} features but model was trained on {_inputDim} features.",
                nameof(X));
        }

        var dataMeans = _dataMeans;
        var dataStds = _dataStds;
        if (dataMeans == null || dataStds == null)
        {
            throw new InvalidOperationException("Model not properly fitted. Normalization parameters missing.");
        }

        var engine = Engine;
        int n = X.Rows;
        var x = AnomalyTensorOps.DivideRow(
            AnomalyTensorOps.SubtractRow(AnomalyTensorOps.ToTensor(X), dataMeans.AsTensor()), dataStds.AsTensor());

        // Scoring-local RNG, re-seeded from the model's own seed on every call, so the latent
        // restarts are identical for identical input. The model's shared _random is deliberately
        // untouched here: scoring must not advance the training stream either. One latent restart per
        // row, drawn in row order.
        var scoringRng = RandomHelper.CreateSeededRandom(_randomSeed);
        var z = SampleLatent(scoringRng, n);

        // Find optimal z per row via gradient descent; the score is the lowest loss seen
        var bestLoss = Tensor<T>.CreateDefault(new[] { n }, NumOps.MaxValue);
        T zLr = NumOps.FromDouble(0.1);
        T regularization = NumOps.FromDouble(0.001);
        T featureWeight = NumOps.FromDouble(0.1);
        T featureGradientWeight = NumOps.FromDouble(0.2);
        T two = NumOps.FromDouble(2.0);
        var axis1 = new[] { 1 };

        // Discriminator features of the real input are constant across steps (x is fixed).
        var (_, featReal, _) = Discriminate(x);

        for (int step = 0; step < _inferenceSteps; step++)
        {
            // Forward with cached activations for the analytic backward pass.
            var (gH1, gH2, xGen) = Generate(z);
            var (discH1, featGen, _) = Discriminate(xGen);

            // Loss = ||x - xGen||^2 + 0.1 * ||featReal - featGen||^2 (feature matching)
            var reconLoss = engine.ReduceSum(engine.TensorSquare(engine.TensorSubtract(x, xGen)), axis1, false);
            var featLoss = engine.ReduceSum(engine.TensorSquare(engine.TensorSubtract(featReal, featGen)), axis1, false);
            var loss = engine.TensorAdd(reconLoss, engine.TensorMultiplyScalar(featLoss, featureWeight));
            bestLoss = engine.TensorWhere(engine.TensorLessThan(loss, bestLoss), loss, bestLoss);

            // EXACT analytic gradient of the loss w.r.t. z in ONE backward pass. This replaces
            // a central finite-difference estimate that ran 2 x latentDim full generator +
            // discriminator forwards PER step (hundreds of thousands of forwards per scoring
            // call) — the cause of the >5-minute test-host hang. Same loss, same descent.
            //
            // d(reconstruction)/dxGen = 2 (xGen - x); d(0.1 * feature-matching)/dh2 = 0.2 (featGen - featReal),
            // flowed back through the discriminator's hidden layers to xGen and added.
            var dxGen = engine.TensorAdd(
                engine.TensorMultiplyScalar(engine.TensorSubtract(xGen, x), two),
                DiscriminatorInputGradient(discH1, featGen,
                    engine.TensorMultiplyScalar(engine.TensorSubtract(featGen, featReal), featureGradientWeight)));

            // Backprop dxGen through the generator to obtain dLoss/dz.
            var dz = GeneratorInputGradient(gH1, gH2, xGen, dxGen);

            // Gradient descent on z with L2 regularization.
            z = engine.TensorSubtract(z, engine.TensorMultiplyScalar(
                engine.TensorAdd(dz, engine.TensorMultiplyScalar(z, regularization)), zLr));
        }

        return bestLoss.ToVector();
    }
}

using AiDotNet.Tensors.Engines;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
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
[ModelPaper("Unsupervised Anomaly Detection with Generative Adversarial Networks", "https://doi.org/10.1007/978-3-319-59050-9_12", Year = 2017, Authors = "Thomas Schlegl, Philipp Seeböck, Sebastian M. Waldstein, Ursula Schmidt-Erfurth, Georg Langs")]
public class AnoGANDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _latentDim;
    private readonly int _hiddenDim;
    private readonly int _epochs;
    private readonly double _learningRate;
    private readonly int _inferenceSteps;

    // Generator weights
    private Matrix<T>? _genW1;
    private Vector<T>? _genB1;
    private Matrix<T>? _genW2;
    private Vector<T>? _genB2;
    private Matrix<T>? _genW3;
    private Vector<T>? _genB3;

    // Discriminator weights
    private Matrix<T>? _discW1;
    private Vector<T>? _discB1;
    private Matrix<T>? _discW2;
    private Vector<T>? _discB2;
    private Matrix<T>? _discW3;
    private Vector<T>? _discB3;

    private int _inputDim;

    // Normalization parameters
    private Vector<T>? _dataMeans;
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
        int n = data.Rows;
        int d = data.Columns;

        var means = new Vector<T>(d);
        var stds = new Vector<T>(d);

        for (int j = 0; j < d; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                sum = NumOps.Add(sum, data[i, j]);
            }
            means[j] = NumOps.Divide(sum, NumOps.FromDouble(n));

            T variance = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                T diff = NumOps.Subtract(data[i, j], means[j]);
                variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
            }
            double stdVal = Math.Sqrt(NumOps.ToDouble(variance) / n);
            if (stdVal < 1e-10) stdVal = 1;
            stds[j] = NumOps.FromDouble(stdVal);
        }

        var normalized = new Matrix<T>(n, d);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
            {
                T diff = NumOps.Subtract(data[i, j], means[j]);
                normalized[i, j] = NumOps.Divide(diff, stds[j]);
            }
        }

        return (normalized, means, stds);
    }

    private void InitializeWeights()
    {
        // Generator: z -> hidden1 -> hidden2 -> output
        double genScale1 = Math.Sqrt(2.0 / (_latentDim + _hiddenDim));
        double genScale2 = Math.Sqrt(2.0 / (_hiddenDim + _hiddenDim));
        double genScale3 = Math.Sqrt(2.0 / (_hiddenDim + _inputDim));

        _genW1 = InitializeMatrix(_latentDim, _hiddenDim, genScale1);
        _genB1 = InitializeVector(_hiddenDim);
        _genW2 = InitializeMatrix(_hiddenDim, _hiddenDim, genScale2);
        _genB2 = InitializeVector(_hiddenDim);
        _genW3 = InitializeMatrix(_hiddenDim, _inputDim, genScale3);
        _genB3 = InitializeVector(_inputDim);

        // Discriminator: input -> hidden1 -> hidden2 -> 1
        double discScale1 = Math.Sqrt(2.0 / (_inputDim + _hiddenDim));
        double discScale2 = Math.Sqrt(2.0 / (_hiddenDim + _hiddenDim));
        double discScale3 = Math.Sqrt(2.0 / (_hiddenDim + 1));

        _discW1 = InitializeMatrix(_inputDim, _hiddenDim, discScale1);
        _discB1 = InitializeVector(_hiddenDim);
        _discW2 = InitializeMatrix(_hiddenDim, _hiddenDim, discScale2);
        _discB2 = InitializeVector(_hiddenDim);
        _discW3 = InitializeMatrix(_hiddenDim, 1, discScale3);
        _discB3 = InitializeVector(1);
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

    private Vector<T> InitializeVector(int size)
    {
        var vector = new Vector<T>(size);
        for (int i = 0; i < size; i++)
        {
            vector[i] = NumOps.Zero;
        }
        return vector;
    }

    private void TrainGAN(Matrix<T> data)
    {
        int n = data.Rows;
        int batchSize = Math.Min(32, n);

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            var indices = Enumerable.Range(0, n).OrderBy(_ => _random.NextDouble()).ToArray();

            for (int batch = 0; batch < n; batch += batchSize)
            {
                int actualBatchSize = Math.Min(batchSize, n - batch);

                // Train Discriminator
                for (int b = 0; b < actualBatchSize; b++)
                {
                    int idx = indices[batch + b];
                    var realData = data.GetRow(idx);

                    // Generate fake data
                    var z = SampleLatent();
                    var fakeData = Generate(z);

                    // Update discriminator
                    UpdateDiscriminator(realData, fakeData);
                }

                // Train Generator
                for (int b = 0; b < actualBatchSize; b++)
                {
                    var z = SampleLatent();
                    UpdateGenerator(z);
                }
            }
        }
    }

    private Vector<T> SampleLatent()
    {
        var z = new Vector<T>(_latentDim);
        for (int i = 0; i < _latentDim; i++)
        {
            double u1 = 1.0 - _random.NextDouble();
            double u2 = 1.0 - _random.NextDouble();
            double val = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            z[i] = NumOps.FromDouble(val);
        }
        return z;
    }

    private Vector<T> Generate(Vector<T> z)
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

        // Vectorized generator forward using Engine.TensorMatMul
        // Layer 1: h1 = LeakyReLU(z @ genW1 + genB1)
        var zTensor = Tensor<T>.FromVector(z).Reshape(1, _latentDim);
        var layer1 = Engine.TensorMatMul(zTensor, Tensor<T>.FromMatrix(genW1));
        layer1 = Engine.TensorBroadcastAdd(layer1, Tensor<T>.FromVector(genB1).Reshape(1, _hiddenDim));
        layer1 = Engine.LeakyReLU(layer1, NumOps.FromDouble(0.2));
        var h1 = layer1.Reshape(_hiddenDim).ToVector();

        // Layer 2: h2 = LeakyReLU(h1 @ genW2 + genB2)
        var layer2 = Engine.TensorMatMul(layer1, Tensor<T>.FromMatrix(genW2));
        layer2 = Engine.TensorBroadcastAdd(layer2, Tensor<T>.FromVector(genB2).Reshape(1, _hiddenDim));
        layer2 = Engine.LeakyReLU(layer2, NumOps.FromDouble(0.2));
        var h2 = layer2.Reshape(_hiddenDim).ToVector();

        // Output layer: output = tanh(h2 @ genW3 + genB3)
        var layer3 = Engine.TensorMatMul(layer2, Tensor<T>.FromMatrix(genW3));
        layer3 = Engine.TensorBroadcastAdd(layer3, Tensor<T>.FromVector(genB3).Reshape(1, _inputDim));
        layer3 = Engine.Tanh(layer3);
        var output = layer3.Reshape(_inputDim).ToVector();

        return output;
    }

    private (T output, Vector<T> features) Discriminate(Vector<T> x)
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

        // Vectorized discriminator forward
        // Layer 1: h1 = LeakyReLU(x @ discW1 + discB1)
        var xTensor = Tensor<T>.FromVector(x).Reshape(1, _inputDim);
        var dLayer1 = Engine.TensorMatMul(xTensor, Tensor<T>.FromMatrix(discW1));
        dLayer1 = Engine.TensorBroadcastAdd(dLayer1, Tensor<T>.FromVector(discB1).Reshape(1, _hiddenDim));
        dLayer1 = Engine.LeakyReLU(dLayer1, NumOps.FromDouble(0.2));
        var h1 = dLayer1.Reshape(_hiddenDim).ToVector();

        // Layer 2: h2 = LeakyReLU(h1 @ discW2 + discB2)
        var dLayer2 = Engine.TensorMatMul(dLayer1, Tensor<T>.FromMatrix(discW2));
        dLayer2 = Engine.TensorBroadcastAdd(dLayer2, Tensor<T>.FromVector(discB2).Reshape(1, _hiddenDim));
        dLayer2 = Engine.LeakyReLU(dLayer2, NumOps.FromDouble(0.2));
        var h2 = dLayer2.Reshape(_hiddenDim).ToVector();

        // Output layer: sigmoid(h2 @ discW3 + discB3)
        var dLayer3 = Engine.TensorMatMul(dLayer2, Tensor<T>.FromMatrix(discW3));
        dLayer3 = Engine.TensorBroadcastAdd(dLayer3, Tensor<T>.FromVector(discB3).Reshape(1, 1));
        double sigVal = Sigmoid(NumOps.ToDouble(dLayer3[0]));
        T output = NumOps.FromDouble(sigVal);

        return (output, h2);
    }

    private void UpdateDiscriminator(Vector<T> realData, Vector<T> fakeData)
    {
        double lr = _learningRate;

        // Forward pass for real data with cache
        var (realH1, realH2, realOut) = DiscriminateWithCache(realData);
        // Forward pass for fake data with cache
        var (fakeH1, fakeH2, fakeOut) = DiscriminateWithCache(fakeData);

        // Binary cross-entropy gradients
        double dRealOut = NumOps.ToDouble(realOut) - 1.0;
        double dFakeOut = NumOps.ToDouble(fakeOut);

        // Backprop through discriminator for real data
        BackpropDiscriminator(realData, realH1, realH2, dRealOut, lr);
        // Backprop through discriminator for fake data
        BackpropDiscriminator(fakeData, fakeH1, fakeH2, dFakeOut, lr);
    }

    private (Vector<T> h1, Vector<T> h2, T output) DiscriminateWithCache(Vector<T> x)
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

        // Layer 1
        var h1 = new Vector<T>(_hiddenDim);
        for (int j = 0; j < _hiddenDim; j++)
        {
            T sum = discB1[j];
            { var wCol_5 = new Vector<T>(_inputDim); for (int ii = 0; ii < _inputDim; ii++) wCol_5[ii] = discW1[ii, j]; sum = NumOps.Add(sum, Engine.DotProduct(x, wCol_5)); }
            double leakyVal = LeakyReLU(NumOps.ToDouble(sum));
            h1[j] = NumOps.FromDouble(leakyVal);
        }

        // Layer 2
        var h2 = new Vector<T>(_hiddenDim);
        for (int j = 0; j < _hiddenDim; j++)
        {
            T sum = discB2[j];
            { var wCol_6 = new Vector<T>(_hiddenDim); for (int ii = 0; ii < _hiddenDim; ii++) wCol_6[ii] = discW2[ii, j]; sum = NumOps.Add(sum, Engine.DotProduct(h1, wCol_6)); }
            double leakyVal = LeakyReLU(NumOps.ToDouble(sum));
            h2[j] = NumOps.FromDouble(leakyVal);
        }

        // Output layer
        T outSum = discB3[0];
        { var _e1 = new Vector<T>(_hiddenDim); for (int _i = 0; _i < _hiddenDim; _i++) _e1[_i] = discW3[_i, 0]; outSum = NumOps.Add(outSum, Engine.DotProduct(h2, _e1)); }
        double sigVal = Sigmoid(NumOps.ToDouble(outSum));
        T output = NumOps.FromDouble(sigVal);

        return (h1, h2, output);
    }

    private void BackpropDiscriminator(Vector<T> x, Vector<T> h1, Vector<T> h2, double dOut, double lr)
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

        T lrT = NumOps.FromDouble(lr);
        T dOutT = NumOps.FromDouble(dOut);
        T leakySlope = NumOps.FromDouble(0.2);

        // === Phase 1: Compute ALL gradients using original weights ===

        // Layer 3 gradient: dH2 = W3 @ dOut (W3 is [hiddenDim, 1], dOut is scalar)
        // dH2[i] = W3[i,0] * dOut — vectorized: column of W3 * scalar
        var w3Col = new Vector<T>(_hiddenDim);
        for (int i = 0; i < _hiddenDim; i++) w3Col[i] = discW3[i, 0];
        var dH2 = (Vector<T>)Engine.Multiply(w3Col, dOutT);

        // LeakyReLU derivative for h2
        for (int i = 0; i < _hiddenDim; i++)
        {
            if (NumOps.LessThan(h2[i], NumOps.Zero))
                dH2[i] = NumOps.Multiply(dH2[i], leakySlope);
        }

        // Layer 2 gradient: dH1 = W2^T @ dH2 — vectorized matmul
        var w2Tensor = Tensor<T>.FromMatrix(discW2);
        var dH2Tensor = Tensor<T>.FromVector(dH2).Reshape(_hiddenDim, 1);
        var dH1Tensor = Engine.TensorMatMul(w2Tensor, dH2Tensor).Reshape(_hiddenDim);
        var dH1 = dH1Tensor.ToVector();

        // LeakyReLU derivative for h1
        for (int i = 0; i < _hiddenDim; i++)
        {
            if (NumOps.LessThan(h1[i], NumOps.Zero))
                dH1[i] = NumOps.Multiply(dH1[i], leakySlope);
        }

        // === Phase 2: Compute weight gradients and apply updates ===

        // Layer 3 weight gradient: dW3 = h2 * dOut, then W3 -= lr * dW3
        var dW3 = (Vector<T>)Engine.Multiply(h2, dOutT);
        for (int i = 0; i < _hiddenDim; i++)
            discW3[i, 0] = NumOps.Subtract(discW3[i, 0], NumOps.Multiply(lrT, dW3[i]));
        discB3[0] = NumOps.Subtract(discB3[0], NumOps.Multiply(lrT, dOutT));

        // Layer 2 weight gradient: dW2 = h1 @ dH2^T (outer product), then W2 -= lr * dW2
        var h1Tensor = Tensor<T>.FromVector(h1).Reshape(_hiddenDim, 1);
        var dH2Row = Tensor<T>.FromVector(dH2).Reshape(1, _hiddenDim);
        var dW2 = Engine.TensorMatMul(h1Tensor, dH2Row);
        var w2Update = Engine.TensorMultiplyScalar(dW2, lrT);
        var updatedW2 = Engine.TensorSubtract(w2Tensor, w2Update);
        for (int i = 0; i < _hiddenDim; i++)
            for (int j = 0; j < _hiddenDim; j++)
                discW2[i, j] = updatedW2[i, j];

        // Layer 2 bias gradient: dB2 = dH2, then B2 -= lr * dB2
        var scaledDH2 = (Vector<T>)Engine.Multiply(dH2, lrT);
        for (int j = 0; j < _hiddenDim; j++)
            discB2[j] = NumOps.Subtract(discB2[j], scaledDH2[j]);

        // Layer 1 weight gradient: dW1 = x @ dH1^T (outer product), then W1 -= lr * dW1
        var xTensor = Tensor<T>.FromVector(x).Reshape(_inputDim, 1);
        var dH1Row = Tensor<T>.FromVector(dH1).Reshape(1, _hiddenDim);
        var dW1 = Engine.TensorMatMul(xTensor, dH1Row);
        var w1Tensor = Tensor<T>.FromMatrix(discW1);
        var w1Update = Engine.TensorMultiplyScalar(dW1, lrT);
        var updatedW1 = Engine.TensorSubtract(w1Tensor, w1Update);
        for (int i = 0; i < _inputDim; i++)
            for (int j = 0; j < _hiddenDim; j++)
                discW1[i, j] = updatedW1[i, j];

        // Layer 1 bias gradient: dB1 = dH1, then B1 -= lr * dB1
        var scaledDH1 = (Vector<T>)Engine.Multiply(dH1, lrT);
        for (int j = 0; j < _hiddenDim; j++)
            discB1[j] = NumOps.Subtract(discB1[j], scaledDH1[j]);
    }

    private void UpdateGenerator(Vector<T> z)
    {
        double lr = _learningRate;

        // Forward pass through generator with cache
        var (h1, h2, fakeData) = GenerateWithCache(z);

        // Forward pass through discriminator
        var (_, discH2, fakeOut) = DiscriminateWithCache(fakeData);

        // Generator wants discriminator to output 1 for fake
        double dOut = NumOps.ToDouble(fakeOut) - 1.0;

        // Backprop through discriminator to get gradient w.r.t. fakeData
        var dFakeData = BackpropDiscriminatorToInput(fakeData, discH2, dOut);

        // Backprop through generator
        BackpropGenerator(z, h1, h2, dFakeData, lr);
    }

    private (Vector<T> h1, Vector<T> h2, Vector<T> output) GenerateWithCache(Vector<T> z)
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

        // Layer 1
        var h1 = new Vector<T>(_hiddenDim);
        for (int j = 0; j < _hiddenDim; j++)
        {
            T sum = genB1[j];
            { var wCol_7 = new Vector<T>(_latentDim); for (int ii = 0; ii < _latentDim; ii++) wCol_7[ii] = genW1[ii, j]; sum = NumOps.Add(sum, Engine.DotProduct(z, wCol_7)); }
            double leakyVal = LeakyReLU(NumOps.ToDouble(sum));
            h1[j] = NumOps.FromDouble(leakyVal);
        }

        // Layer 2
        var h2 = new Vector<T>(_hiddenDim);
        for (int j = 0; j < _hiddenDim; j++)
        {
            T sum = genB2[j];
            { var wCol_8 = new Vector<T>(_hiddenDim); for (int ii = 0; ii < _hiddenDim; ii++) wCol_8[ii] = genW2[ii, j]; sum = NumOps.Add(sum, Engine.DotProduct(h1, wCol_8)); }
            double leakyVal = LeakyReLU(NumOps.ToDouble(sum));
            h2[j] = NumOps.FromDouble(leakyVal);
        }

        // Output layer (tanh for bounded output)
        var output = new Vector<T>(_inputDim);
        for (int j = 0; j < _inputDim; j++)
        {
            T sum = genB3[j];
            { var wCol_9 = new Vector<T>(_hiddenDim); for (int ii = 0; ii < _hiddenDim; ii++) wCol_9[ii] = genW3[ii, j]; sum = NumOps.Add(sum, Engine.DotProduct(h2, wCol_9)); }
            double tanhVal = Math.Tanh(NumOps.ToDouble(sum));
            output[j] = NumOps.FromDouble(tanhVal);
        }

        return (h1, h2, output);
    }

    private Vector<T> BackpropDiscriminatorToInput(Vector<T> x, Vector<T> h2, double dOut)
    {
        var discW1 = _discW1;
        var discB1 = _discB1;
        var discW2 = _discW2;
        var discW3 = _discW3;

        if (discW1 == null || discB1 == null || discW2 == null || discW3 == null)
        {
            throw new InvalidOperationException("Discriminator weights not initialized.");
        }

        T leakySlope = NumOps.FromDouble(0.2);

        // Gradient through output layer: dH2 = W3[:,0] * dOut — vectorized
        var w3ColVec = new Vector<T>(_hiddenDim);
        for (int i = 0; i < _hiddenDim; i++) w3ColVec[i] = discW3[i, 0];
        var dH2 = (Vector<T>)Engine.Multiply(w3ColVec, NumOps.FromDouble(dOut));

        // LeakyReLU derivative for h2
        for (int i = 0; i < _hiddenDim; i++)
        {
            if (NumOps.LessThan(h2[i], NumOps.Zero))
                dH2[i] = NumOps.Multiply(dH2[i], leakySlope);
        }

        // Recompute h1: h1 = LeakyReLU(x @ W1 + b1) — vectorized matmul
        var xTensor = Tensor<T>.FromVector(x).Reshape(1, _inputDim);
        var w1Tensor = Tensor<T>.FromMatrix(discW1);
        var h1Pre = Engine.TensorMatMul(xTensor, w1Tensor).Reshape(_hiddenDim);
        var h1PreVec = h1Pre.ToVector();
        var h1 = new Vector<T>(_hiddenDim);
        for (int j = 0; j < _hiddenDim; j++)
        {
            T val = NumOps.Add(h1PreVec[j], discB1[j]);
            h1[j] = NumOps.FromDouble(LeakyReLU(NumOps.ToDouble(val)));
        }

        // Gradient through layer 2: dH1 = W2 @ dH2 — vectorized matmul
        var w2Tensor = Tensor<T>.FromMatrix(discW2);
        var dH2Col = Tensor<T>.FromVector(dH2).Reshape(_hiddenDim, 1);
        var dH1 = Engine.TensorMatMul(w2Tensor, dH2Col).Reshape(_hiddenDim).ToVector();

        // LeakyReLU derivative for h1
        for (int i = 0; i < _hiddenDim; i++)
        {
            if (NumOps.LessThan(h1[i], NumOps.Zero))
                dH1[i] = NumOps.Multiply(dH1[i], leakySlope);
        }

        // Gradient through layer 1 to input: dX = W1 @ dH1 — vectorized matmul
        var dH1Col = Tensor<T>.FromVector(dH1).Reshape(_hiddenDim, 1);
        var dX = Engine.TensorMatMul(w1Tensor, dH1Col).Reshape(_inputDim).ToVector();

        return dX;
    }

    private void BackpropGenerator(Vector<T> z, Vector<T> h1, Vector<T> h2, Vector<T> dOutput, double lr)
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

        T lrT = NumOps.FromDouble(lr);
        T leakySlope = NumOps.FromDouble(0.2);

        // Recompute output layer: output = tanh(h2 @ W3 + b3) — vectorized matmul
        var h2Tensor = Tensor<T>.FromVector(h2).Reshape(1, _hiddenDim);
        var w3Tensor = Tensor<T>.FromMatrix(genW3);
        var outputPre = Engine.TensorMatMul(h2Tensor, w3Tensor).Reshape(_inputDim);
        var output = new Vector<T>(_inputDim);
        var dOutputPre = new Vector<T>(_inputDim);
        for (int j = 0; j < _inputDim; j++)
        {
            T pre = NumOps.Add(outputPre[j], genB3[j]);
            double tanhVal = Math.Tanh(NumOps.ToDouble(pre));
            output[j] = NumOps.FromDouble(tanhVal);
            double tanhDeriv = 1 - tanhVal * tanhVal;
            dOutputPre[j] = NumOps.Multiply(dOutput[j], NumOps.FromDouble(tanhDeriv));
        }

        // === Phase 1: Compute ALL gradients using original weights ===

        // Layer 3 gradient: dH2 = W3 @ dOutputPre — vectorized matmul
        var dOutPreCol = Tensor<T>.FromVector(dOutputPre).Reshape(_inputDim, 1);
        var dH2 = Engine.TensorMatMul(w3Tensor, dOutPreCol).Reshape(_hiddenDim).ToVector();

        // LeakyReLU derivative for h2
        for (int i = 0; i < _hiddenDim; i++)
        {
            if (NumOps.LessThan(h2[i], NumOps.Zero))
                dH2[i] = NumOps.Multiply(dH2[i], leakySlope);
        }

        // Layer 2 gradient: dH1 = W2 @ dH2 — vectorized matmul
        var w2Tensor = Tensor<T>.FromMatrix(genW2);
        var dH2Col = Tensor<T>.FromVector(dH2).Reshape(_hiddenDim, 1);
        var dH1 = Engine.TensorMatMul(w2Tensor, dH2Col).Reshape(_hiddenDim).ToVector();

        // LeakyReLU derivative for h1
        for (int i = 0; i < _hiddenDim; i++)
        {
            if (NumOps.LessThan(h1[i], NumOps.Zero))
                dH1[i] = NumOps.Multiply(dH1[i], leakySlope);
        }

        // === Phase 2: Compute weight gradients and apply updates ===

        // Layer 3: dW3 = h2 @ dOutputPre^T, W3 -= lr * dW3
        var h2Col = Tensor<T>.FromVector(h2).Reshape(_hiddenDim, 1);
        var dOutPreRow = Tensor<T>.FromVector(dOutputPre).Reshape(1, _inputDim);
        var dW3 = Engine.TensorMatMul(h2Col, dOutPreRow);
        var updatedW3 = Engine.TensorSubtract(w3Tensor, Engine.TensorMultiplyScalar(dW3, lrT));
        for (int i = 0; i < _hiddenDim; i++)
            for (int j = 0; j < _inputDim; j++)
                genW3[i, j] = updatedW3[i, j];
        var scaledDOutPre = (Vector<T>)Engine.Multiply(dOutputPre, lrT);
        for (int j = 0; j < _inputDim; j++)
            genB3[j] = NumOps.Subtract(genB3[j], scaledDOutPre[j]);

        // Layer 2: dW2 = h1 @ dH2^T, W2 -= lr * dW2
        var h1Col = Tensor<T>.FromVector(h1).Reshape(_hiddenDim, 1);
        var dH2Row = Tensor<T>.FromVector(dH2).Reshape(1, _hiddenDim);
        var dW2 = Engine.TensorMatMul(h1Col, dH2Row);
        var updatedW2 = Engine.TensorSubtract(w2Tensor, Engine.TensorMultiplyScalar(dW2, lrT));
        for (int i = 0; i < _hiddenDim; i++)
            for (int j = 0; j < _hiddenDim; j++)
                genW2[i, j] = updatedW2[i, j];
        var scaledDH2 = (Vector<T>)Engine.Multiply(dH2, lrT);
        for (int j = 0; j < _hiddenDim; j++)
            genB2[j] = NumOps.Subtract(genB2[j], scaledDH2[j]);

        // Layer 1: dW1 = z @ dH1^T, W1 -= lr * dW1
        var zCol = Tensor<T>.FromVector(z).Reshape(_latentDim, 1);
        var dH1Row = Tensor<T>.FromVector(dH1).Reshape(1, _hiddenDim);
        var dW1 = Engine.TensorMatMul(zCol, dH1Row);
        var w1Tensor = Tensor<T>.FromMatrix(genW1);
        var updatedW1 = Engine.TensorSubtract(w1Tensor, Engine.TensorMultiplyScalar(dW1, lrT));
        for (int i = 0; i < _latentDim; i++)
            for (int j = 0; j < _hiddenDim; j++)
                genW1[i, j] = updatedW1[i, j];
        var scaledDH1 = (Vector<T>)Engine.Multiply(dH1, lrT);
        for (int j = 0; j < _hiddenDim; j++)
            genB1[j] = NumOps.Subtract(genB1[j], scaledDH1[j]);
    }

    private static double LeakyReLU(double x, double alpha = 0.2)
    {
        return x >= 0 ? x : alpha * x;
    }

    private static double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-Math.Max(-500, Math.Min(500, x))));
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

        var scores = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            // Normalize
            var x = new Vector<T>(_inputDim);
            for (int j = 0; j < _inputDim; j++)
            {
                T diff = NumOps.Subtract(X[i, j], dataMeans[j]);
                x[j] = NumOps.Divide(diff, dataStds[j]);
            }

            // Find optimal z via gradient descent
            var z = SampleLatent();
            var bestZ = new Vector<T>(_latentDim);
            for (int j = 0; j < _latentDim; j++) bestZ[j] = z[j];

            T bestLoss = NumOps.MaxValue;
            double zLr = 0.1;

            for (int step = 0; step < _inferenceSteps; step++)
            {
                var xGen = Generate(z);
                var (_, featGen) = Discriminate(xGen);
                var (_, featReal) = Discriminate(x);

                // Reconstruction loss
                T reconLoss = NumOps.Zero;
                for (int j = 0; j < _inputDim; j++)
                {
                    T diff = NumOps.Subtract(x[j], xGen[j]);
                    reconLoss = NumOps.Add(reconLoss, NumOps.Multiply(diff, diff));
                }

                // Feature matching loss
                T featLoss = NumOps.Zero;
                for (int j = 0; j < _hiddenDim; j++)
                {
                    T diff = NumOps.Subtract(featReal[j], featGen[j]);
                    featLoss = NumOps.Add(featLoss, NumOps.Multiply(diff, diff));
                }

                T loss = NumOps.Add(reconLoss, NumOps.Multiply(NumOps.FromDouble(0.1), featLoss));

                if (NumOps.LessThan(loss, bestLoss))
                {
                    bestLoss = loss;
                    for (int j = 0; j < _latentDim; j++) bestZ[j] = z[j];
                }

                // Compute gradient of loss w.r.t. z using finite differences
                double eps = 1e-4;
                var dz = new Vector<T>(_latentDim);

                for (int j = 0; j < _latentDim; j++)
                {
                    // Perturb z[j] positively
                    T origVal = z[j];
                    z[j] = NumOps.Add(z[j], NumOps.FromDouble(eps));
                    var xGenPlus = Generate(z);
                    var (_, featGenPlus) = Discriminate(xGenPlus);

                    T lossPlus = NumOps.Zero;
                    for (int k = 0; k < _inputDim; k++)
                    {
                        T diff = NumOps.Subtract(x[k], xGenPlus[k]);
                        lossPlus = NumOps.Add(lossPlus, NumOps.Multiply(diff, diff));
                    }
                    for (int k = 0; k < _hiddenDim; k++)
                    {
                        T diff = NumOps.Subtract(featReal[k], featGenPlus[k]);
                        lossPlus = NumOps.Add(lossPlus, NumOps.Multiply(NumOps.FromDouble(0.1), NumOps.Multiply(diff, diff)));
                    }

                    // Perturb z[j] negatively
                    z[j] = NumOps.Subtract(origVal, NumOps.FromDouble(eps));
                    var xGenMinus = Generate(z);
                    var (_, featGenMinus) = Discriminate(xGenMinus);

                    T lossMinus = NumOps.Zero;
                    for (int k = 0; k < _inputDim; k++)
                    {
                        T diff = NumOps.Subtract(x[k], xGenMinus[k]);
                        lossMinus = NumOps.Add(lossMinus, NumOps.Multiply(diff, diff));
                    }
                    for (int k = 0; k < _hiddenDim; k++)
                    {
                        T diff = NumOps.Subtract(featReal[k], featGenMinus[k]);
                        lossMinus = NumOps.Add(lossMinus, NumOps.Multiply(NumOps.FromDouble(0.1), NumOps.Multiply(diff, diff)));
                    }

                    // Restore z[j] and compute gradient
                    z[j] = origVal;
                    double gradVal = (NumOps.ToDouble(lossPlus) - NumOps.ToDouble(lossMinus)) / (2 * eps);
                    dz[j] = NumOps.FromDouble(gradVal);
                }

                // Update z using gradient descent with regularization
                for (int j = 0; j < _latentDim; j++)
                {
                    double gradWithReg = NumOps.ToDouble(dz[j]) + 0.001 * NumOps.ToDouble(z[j]);
                    z[j] = NumOps.Subtract(z[j], NumOps.FromDouble(zLr * gradWithReg));
                }
            }

            scores[i] = bestLoss;
        }

        return scores;
    }
}

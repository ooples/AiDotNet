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

        // Layer 1
        var h1 = new Vector<T>(_hiddenDim);
        for (int j = 0; j < _hiddenDim; j++)
        {
            T sum = genB1[j];
            for (int i = 0; i < _latentDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(z[i], genW1[i, j]));
            }
            double leakyVal = LeakyReLU(NumOps.ToDouble(sum));
            h1[j] = NumOps.FromDouble(leakyVal);
        }

        // Layer 2
        var h2 = new Vector<T>(_hiddenDim);
        for (int j = 0; j < _hiddenDim; j++)
        {
            T sum = genB2[j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(h1[i], genW2[i, j]));
            }
            double leakyVal = LeakyReLU(NumOps.ToDouble(sum));
            h2[j] = NumOps.FromDouble(leakyVal);
        }

        // Output layer (tanh for bounded output)
        var output = new Vector<T>(_inputDim);
        for (int j = 0; j < _inputDim; j++)
        {
            T sum = genB3[j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(h2[i], genW3[i, j]));
            }
            double tanhVal = Math.Tanh(NumOps.ToDouble(sum));
            output[j] = NumOps.FromDouble(tanhVal);
        }

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

        // Layer 1
        var h1 = new Vector<T>(_hiddenDim);
        for (int j = 0; j < _hiddenDim; j++)
        {
            T sum = discB1[j];
            for (int i = 0; i < _inputDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(x[i], discW1[i, j]));
            }
            double leakyVal = LeakyReLU(NumOps.ToDouble(sum));
            h1[j] = NumOps.FromDouble(leakyVal);
        }

        // Layer 2 (feature layer)
        var h2 = new Vector<T>(_hiddenDim);
        for (int j = 0; j < _hiddenDim; j++)
        {
            T sum = discB2[j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(h1[i], discW2[i, j]));
            }
            double leakyVal = LeakyReLU(NumOps.ToDouble(sum));
            h2[j] = NumOps.FromDouble(leakyVal);
        }

        // Output layer (sigmoid for probability)
        T outputSum = discB3[0];
        for (int i = 0; i < _hiddenDim; i++)
        {
            outputSum = NumOps.Add(outputSum, NumOps.Multiply(h2[i], discW3[i, 0]));
        }
        double sigVal = Sigmoid(NumOps.ToDouble(outputSum));
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
            for (int i = 0; i < _inputDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(x[i], discW1[i, j]));
            }
            double leakyVal = LeakyReLU(NumOps.ToDouble(sum));
            h1[j] = NumOps.FromDouble(leakyVal);
        }

        // Layer 2
        var h2 = new Vector<T>(_hiddenDim);
        for (int j = 0; j < _hiddenDim; j++)
        {
            T sum = discB2[j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(h1[i], discW2[i, j]));
            }
            double leakyVal = LeakyReLU(NumOps.ToDouble(sum));
            h2[j] = NumOps.FromDouble(leakyVal);
        }

        // Output layer
        T outSum = discB3[0];
        for (int i = 0; i < _hiddenDim; i++)
        {
            outSum = NumOps.Add(outSum, NumOps.Multiply(h2[i], discW3[i, 0]));
        }
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

        // Gradient through output layer - compute gradient using ORIGINAL weights before updating
        var dH2 = new Vector<T>(_hiddenDim);
        for (int i = 0; i < _hiddenDim; i++)
        {
            // Capture original weight for gradient computation
            T origW3 = discW3[i, 0];
            dH2[i] = NumOps.Multiply(origW3, NumOps.FromDouble(dOut));
            // Now update the weight
            T grad = NumOps.Multiply(h2[i], NumOps.FromDouble(dOut));
            discW3[i, 0] = NumOps.Subtract(discW3[i, 0], NumOps.FromDouble(lr * NumOps.ToDouble(grad)));
        }
        discB3[0] = NumOps.Subtract(discB3[0], NumOps.FromDouble(lr * dOut));

        // LeakyReLU derivative for h2
        for (int i = 0; i < _hiddenDim; i++)
        {
            if (NumOps.ToDouble(h2[i]) < 0)
                dH2[i] = NumOps.Multiply(dH2[i], NumOps.FromDouble(0.2));
        }

        // Gradient through layer 2 - compute gradient using ORIGINAL weights before updating
        var dH1 = new Vector<T>(_hiddenDim);
        for (int i = 0; i < _hiddenDim; i++)
        {
            dH1[i] = NumOps.Zero;
            for (int j = 0; j < _hiddenDim; j++)
            {
                // Capture original weight for gradient computation
                T origW2 = discW2[i, j];
                dH1[i] = NumOps.Add(dH1[i], NumOps.Multiply(origW2, dH2[j]));
                // Now update the weight
                T grad = NumOps.Multiply(h1[i], dH2[j]);
                discW2[i, j] = NumOps.Subtract(discW2[i, j], NumOps.FromDouble(lr * NumOps.ToDouble(grad)));
            }
        }
        for (int j = 0; j < _hiddenDim; j++)
        {
            discB2[j] = NumOps.Subtract(discB2[j], NumOps.FromDouble(lr * NumOps.ToDouble(dH2[j])));
        }

        // LeakyReLU derivative for h1
        for (int i = 0; i < _hiddenDim; i++)
        {
            if (NumOps.ToDouble(h1[i]) < 0)
                dH1[i] = NumOps.Multiply(dH1[i], NumOps.FromDouble(0.2));
        }

        // Gradient through layer 1
        for (int i = 0; i < _inputDim; i++)
        {
            for (int j = 0; j < _hiddenDim; j++)
            {
                T grad = NumOps.Multiply(x[i], dH1[j]);
                discW1[i, j] = NumOps.Subtract(discW1[i, j], NumOps.FromDouble(lr * NumOps.ToDouble(grad)));
            }
        }
        for (int j = 0; j < _hiddenDim; j++)
        {
            discB1[j] = NumOps.Subtract(discB1[j], NumOps.FromDouble(lr * NumOps.ToDouble(dH1[j])));
        }
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
            for (int i = 0; i < _latentDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(z[i], genW1[i, j]));
            }
            double leakyVal = LeakyReLU(NumOps.ToDouble(sum));
            h1[j] = NumOps.FromDouble(leakyVal);
        }

        // Layer 2
        var h2 = new Vector<T>(_hiddenDim);
        for (int j = 0; j < _hiddenDim; j++)
        {
            T sum = genB2[j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(h1[i], genW2[i, j]));
            }
            double leakyVal = LeakyReLU(NumOps.ToDouble(sum));
            h2[j] = NumOps.FromDouble(leakyVal);
        }

        // Output layer (tanh for bounded output)
        var output = new Vector<T>(_inputDim);
        for (int j = 0; j < _inputDim; j++)
        {
            T sum = genB3[j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(h2[i], genW3[i, j]));
            }
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

        // Gradient through output layer
        var dH2 = new Vector<T>(_hiddenDim);
        for (int i = 0; i < _hiddenDim; i++)
        {
            dH2[i] = NumOps.Multiply(discW3[i, 0], NumOps.FromDouble(dOut));
        }

        // LeakyReLU derivative for h2
        for (int i = 0; i < _hiddenDim; i++)
        {
            if (NumOps.ToDouble(h2[i]) < 0)
                dH2[i] = NumOps.Multiply(dH2[i], NumOps.FromDouble(0.2));
        }

        // Recompute h1
        var h1 = new Vector<T>(_hiddenDim);
        for (int j = 0; j < _hiddenDim; j++)
        {
            T sum = discB1[j];
            for (int i = 0; i < _inputDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(x[i], discW1[i, j]));
            }
            double leakyVal = LeakyReLU(NumOps.ToDouble(sum));
            h1[j] = NumOps.FromDouble(leakyVal);
        }

        // Gradient through layer 2
        var dH1 = new Vector<T>(_hiddenDim);
        for (int i = 0; i < _hiddenDim; i++)
        {
            dH1[i] = NumOps.Zero;
            for (int j = 0; j < _hiddenDim; j++)
            {
                dH1[i] = NumOps.Add(dH1[i], NumOps.Multiply(discW2[i, j], dH2[j]));
            }
        }

        // LeakyReLU derivative for h1
        for (int i = 0; i < _hiddenDim; i++)
        {
            if (NumOps.ToDouble(h1[i]) < 0)
                dH1[i] = NumOps.Multiply(dH1[i], NumOps.FromDouble(0.2));
        }

        // Gradient through layer 1 to input
        var dX = new Vector<T>(_inputDim);
        for (int i = 0; i < _inputDim; i++)
        {
            dX[i] = NumOps.Zero;
            for (int j = 0; j < _hiddenDim; j++)
            {
                dX[i] = NumOps.Add(dX[i], NumOps.Multiply(discW1[i, j], dH1[j]));
            }
        }

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

        // Recompute output layer values for tanh derivative
        var output = new Vector<T>(_inputDim);
        for (int j = 0; j < _inputDim; j++)
        {
            T sum = genB3[j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(h2[i], genW3[i, j]));
            }
            double tanhVal = Math.Tanh(NumOps.ToDouble(sum));
            output[j] = NumOps.FromDouble(tanhVal);
        }

        // Apply tanh derivative
        var dOutputPre = new Vector<T>(_inputDim);
        for (int j = 0; j < _inputDim; j++)
        {
            double outVal = NumOps.ToDouble(output[j]);
            double tanhDeriv = 1 - outVal * outVal;
            dOutputPre[j] = NumOps.Multiply(dOutput[j], NumOps.FromDouble(tanhDeriv));
        }

        // Gradient through output layer - compute gradient using ORIGINAL weights before updating
        var dH2 = new Vector<T>(_hiddenDim);
        for (int i = 0; i < _hiddenDim; i++)
        {
            dH2[i] = NumOps.Zero;
            for (int j = 0; j < _inputDim; j++)
            {
                // Capture original weight for gradient computation
                T origW3 = genW3[i, j];
                dH2[i] = NumOps.Add(dH2[i], NumOps.Multiply(origW3, dOutputPre[j]));
                // Now update the weight
                T grad = NumOps.Multiply(h2[i], dOutputPre[j]);
                genW3[i, j] = NumOps.Subtract(genW3[i, j], NumOps.FromDouble(lr * NumOps.ToDouble(grad)));
            }
        }
        for (int j = 0; j < _inputDim; j++)
        {
            genB3[j] = NumOps.Subtract(genB3[j], NumOps.FromDouble(lr * NumOps.ToDouble(dOutputPre[j])));
        }

        // LeakyReLU derivative for h2
        for (int i = 0; i < _hiddenDim; i++)
        {
            if (NumOps.ToDouble(h2[i]) < 0)
                dH2[i] = NumOps.Multiply(dH2[i], NumOps.FromDouble(0.2));
        }

        // Gradient through layer 2 - compute gradient using ORIGINAL weights before updating
        var dH1 = new Vector<T>(_hiddenDim);
        for (int i = 0; i < _hiddenDim; i++)
        {
            dH1[i] = NumOps.Zero;
            for (int j = 0; j < _hiddenDim; j++)
            {
                // Capture original weight for gradient computation
                T origW2 = genW2[i, j];
                dH1[i] = NumOps.Add(dH1[i], NumOps.Multiply(origW2, dH2[j]));
                // Now update the weight
                T grad = NumOps.Multiply(h1[i], dH2[j]);
                genW2[i, j] = NumOps.Subtract(genW2[i, j], NumOps.FromDouble(lr * NumOps.ToDouble(grad)));
            }
        }
        for (int j = 0; j < _hiddenDim; j++)
        {
            genB2[j] = NumOps.Subtract(genB2[j], NumOps.FromDouble(lr * NumOps.ToDouble(dH2[j])));
        }

        // LeakyReLU derivative for h1
        for (int i = 0; i < _hiddenDim; i++)
        {
            if (NumOps.ToDouble(h1[i]) < 0)
                dH1[i] = NumOps.Multiply(dH1[i], NumOps.FromDouble(0.2));
        }

        // Gradient through layer 1
        for (int i = 0; i < _latentDim; i++)
        {
            for (int j = 0; j < _hiddenDim; j++)
            {
                T grad = NumOps.Multiply(z[i], dH1[j]);
                genW1[i, j] = NumOps.Subtract(genW1[i, j], NumOps.FromDouble(lr * NumOps.ToDouble(grad)));
            }
        }
        for (int j = 0; j < _hiddenDim; j++)
        {
            genB1[j] = NumOps.Subtract(genB1[j], NumOps.FromDouble(lr * NumOps.ToDouble(dH1[j])));
        }
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

            T bestLoss = NumOps.FromDouble(double.MaxValue);
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

                if (NumOps.ToDouble(loss) < NumOps.ToDouble(bestLoss))
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

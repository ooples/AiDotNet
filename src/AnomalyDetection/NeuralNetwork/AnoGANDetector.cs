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
    private double[,]? _genW1;
    private double[]? _genB1;
    private double[,]? _genW2;
    private double[]? _genB2;
    private double[,]? _genW3;
    private double[]? _genB3;

    // Discriminator weights
    private double[,]? _discW1;
    private double[]? _discB1;
    private double[,]? _discW2;
    private double[]? _discB2;
    private double[,]? _discW3;
    private double[]? _discB3;

    private int _inputDim;

    // Normalization parameters
    private double[]? _dataMeans;
    private double[]? _dataStds;

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

        // Convert to double array
        var data = new double[n][];
        for (int i = 0; i < n; i++)
        {
            data[i] = new double[_inputDim];
            for (int j = 0; j < _inputDim; j++)
            {
                data[i][j] = NumOps.ToDouble(X[i, j]);
            }
        }

        // Normalize data
        var (normalizedData, means, stds) = NormalizeData(data);
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

    private (double[][] normalized, double[] means, double[] stds) NormalizeData(double[][] data)
    {
        int n = data.Length;
        int d = data[0].Length;

        var means = new double[d];
        var stds = new double[d];

        for (int j = 0; j < d; j++)
        {
            for (int i = 0; i < n; i++)
            {
                means[j] += data[i][j];
            }
            means[j] /= n;

            for (int i = 0; i < n; i++)
            {
                stds[j] += Math.Pow(data[i][j] - means[j], 2);
            }
            stds[j] = Math.Sqrt(stds[j] / n);
            if (stds[j] < 1e-10) stds[j] = 1;
        }

        var normalized = new double[n][];
        for (int i = 0; i < n; i++)
        {
            normalized[i] = new double[d];
            for (int j = 0; j < d; j++)
            {
                normalized[i][j] = (data[i][j] - means[j]) / stds[j];
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
        _genB1 = new double[_hiddenDim];
        _genW2 = InitializeMatrix(_hiddenDim, _hiddenDim, genScale2);
        _genB2 = new double[_hiddenDim];
        _genW3 = InitializeMatrix(_hiddenDim, _inputDim, genScale3);
        _genB3 = new double[_inputDim];

        // Discriminator: input -> hidden1 -> hidden2 -> 1
        double discScale1 = Math.Sqrt(2.0 / (_inputDim + _hiddenDim));
        double discScale2 = Math.Sqrt(2.0 / (_hiddenDim + _hiddenDim));
        double discScale3 = Math.Sqrt(2.0 / (_hiddenDim + 1));

        _discW1 = InitializeMatrix(_inputDim, _hiddenDim, discScale1);
        _discB1 = new double[_hiddenDim];
        _discW2 = InitializeMatrix(_hiddenDim, _hiddenDim, discScale2);
        _discB2 = new double[_hiddenDim];
        _discW3 = InitializeMatrix(_hiddenDim, 1, discScale3);
        _discB3 = new double[1];
    }

    private double[,] InitializeMatrix(int rows, int cols, double scale)
    {
        var matrix = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double u1 = 1.0 - _random.NextDouble();
                double u2 = 1.0 - _random.NextDouble();
                matrix[i, j] = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2) * scale;
            }
        }
        return matrix;
    }

    private void TrainGAN(double[][] data)
    {
        int n = data.Length;
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
                    var realData = data[idx];

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

    private double[] SampleLatent()
    {
        var z = new double[_latentDim];
        for (int i = 0; i < _latentDim; i++)
        {
            double u1 = 1.0 - _random.NextDouble();
            double u2 = 1.0 - _random.NextDouble();
            z[i] = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }
        return z;
    }

    private double[] Generate(double[] z)
    {
        // Layer 1
        var h1 = new double[_hiddenDim];
        for (int j = 0; j < _hiddenDim; j++)
        {
            h1[j] = _genB1![j];
            for (int i = 0; i < _latentDim; i++)
            {
                h1[j] += z[i] * _genW1![i, j];
            }
            h1[j] = LeakyReLU(h1[j]);
        }

        // Layer 2
        var h2 = new double[_hiddenDim];
        for (int j = 0; j < _hiddenDim; j++)
        {
            h2[j] = _genB2![j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                h2[j] += h1[i] * _genW2![i, j];
            }
            h2[j] = LeakyReLU(h2[j]);
        }

        // Output layer (tanh for bounded output)
        var output = new double[_inputDim];
        for (int j = 0; j < _inputDim; j++)
        {
            output[j] = _genB3![j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                output[j] += h2[i] * _genW3![i, j];
            }
            output[j] = Math.Tanh(output[j]);
        }

        return output;
    }

    private (double output, double[] features) Discriminate(double[] x)
    {
        // Layer 1
        var h1 = new double[_hiddenDim];
        for (int j = 0; j < _hiddenDim; j++)
        {
            h1[j] = _discB1![j];
            for (int i = 0; i < _inputDim; i++)
            {
                h1[j] += x[i] * _discW1![i, j];
            }
            h1[j] = LeakyReLU(h1[j]);
        }

        // Layer 2 (feature layer)
        var h2 = new double[_hiddenDim];
        for (int j = 0; j < _hiddenDim; j++)
        {
            h2[j] = _discB2![j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                h2[j] += h1[i] * _discW2![i, j];
            }
            h2[j] = LeakyReLU(h2[j]);
        }

        // Output layer (sigmoid for probability)
        double output = _discB3![0];
        for (int i = 0; i < _hiddenDim; i++)
        {
            output += h2[i] * _discW3![i, 0];
        }
        output = Sigmoid(output);

        return (output, h2);
    }

    private void UpdateDiscriminator(double[] realData, double[] fakeData)
    {
        double lr = _learningRate;

        // Forward pass for real data with cache
        var (realH1, realH2, realOut) = DiscriminateWithCache(realData);
        // Forward pass for fake data with cache
        var (fakeH1, fakeH2, fakeOut) = DiscriminateWithCache(fakeData);

        // Binary cross-entropy gradients: d/dz sigmoid_cross_entropy = sigmoid(z) - target
        // For real data, target = 1: gradient = realOut - 1
        // For fake data, target = 0: gradient = fakeOut - 0 = fakeOut
        double dRealOut = realOut - 1.0;
        double dFakeOut = fakeOut;

        // Backprop through discriminator for real data
        BackpropDiscriminator(realData, realH1, realH2, dRealOut, lr);
        // Backprop through discriminator for fake data
        BackpropDiscriminator(fakeData, fakeH1, fakeH2, dFakeOut, lr);
    }

    private (double[] h1, double[] h2, double output) DiscriminateWithCache(double[] x)
    {
        // Layer 1
        var h1Pre = new double[_hiddenDim];
        var h1 = new double[_hiddenDim];
        for (int j = 0; j < _hiddenDim; j++)
        {
            h1Pre[j] = _discB1![j];
            for (int i = 0; i < _inputDim; i++)
            {
                h1Pre[j] += x[i] * _discW1![i, j];
            }
            h1[j] = LeakyReLU(h1Pre[j]);
        }

        // Layer 2
        var h2Pre = new double[_hiddenDim];
        var h2 = new double[_hiddenDim];
        for (int j = 0; j < _hiddenDim; j++)
        {
            h2Pre[j] = _discB2![j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                h2Pre[j] += h1[i] * _discW2![i, j];
            }
            h2[j] = LeakyReLU(h2Pre[j]);
        }

        // Output layer
        double outPre = _discB3![0];
        for (int i = 0; i < _hiddenDim; i++)
        {
            outPre += h2[i] * _discW3![i, 0];
        }
        double output = Sigmoid(outPre);

        return (h1, h2, output);
    }

    private void BackpropDiscriminator(double[] x, double[] h1, double[] h2, double dOut, double lr)
    {
        // Gradient through output layer
        var dH2 = new double[_hiddenDim];
        for (int i = 0; i < _hiddenDim; i++)
        {
            _discW3![i, 0] -= lr * h2[i] * dOut;
            dH2[i] = _discW3[i, 0] * dOut;
        }
        _discB3![0] -= lr * dOut;

        // LeakyReLU derivative for h2
        for (int i = 0; i < _hiddenDim; i++)
        {
            if (h2[i] < 0) dH2[i] *= 0.2;
        }

        // Gradient through layer 2
        var dH1 = new double[_hiddenDim];
        for (int i = 0; i < _hiddenDim; i++)
        {
            for (int j = 0; j < _hiddenDim; j++)
            {
                _discW2![i, j] -= lr * h1[i] * dH2[j];
                dH1[i] += _discW2[i, j] * dH2[j];
            }
        }
        for (int j = 0; j < _hiddenDim; j++)
        {
            _discB2![j] -= lr * dH2[j];
        }

        // LeakyReLU derivative for h1
        for (int i = 0; i < _hiddenDim; i++)
        {
            if (h1[i] < 0) dH1[i] *= 0.2;
        }

        // Gradient through layer 1
        for (int i = 0; i < _inputDim; i++)
        {
            for (int j = 0; j < _hiddenDim; j++)
            {
                _discW1![i, j] -= lr * x[i] * dH1[j];
            }
        }
        for (int j = 0; j < _hiddenDim; j++)
        {
            _discB1![j] -= lr * dH1[j];
        }
    }

    private void UpdateGenerator(double[] z)
    {
        double lr = _learningRate;

        // Forward pass through generator with cache
        var (h1, h2, fakeData) = GenerateWithCache(z);

        // Forward pass through discriminator
        var (_, discH2, fakeOut) = DiscriminateWithCache(fakeData);

        // Generator wants discriminator to output 1 for fake
        // Gradient: d/dz BCE(sigmoid(z), 1) = sigmoid(z) - 1
        double dOut = fakeOut - 1.0;

        // Backprop through discriminator to get gradient w.r.t. fakeData
        var dFakeData = BackpropDiscriminatorToInput(fakeData, discH2, dOut);

        // Backprop through generator
        BackpropGenerator(z, h1, h2, dFakeData, lr);
    }

    private (double[] h1, double[] h2, double[] output) GenerateWithCache(double[] z)
    {
        // Layer 1
        var h1 = new double[_hiddenDim];
        for (int j = 0; j < _hiddenDim; j++)
        {
            h1[j] = _genB1![j];
            for (int i = 0; i < _latentDim; i++)
            {
                h1[j] += z[i] * _genW1![i, j];
            }
            h1[j] = LeakyReLU(h1[j]);
        }

        // Layer 2
        var h2 = new double[_hiddenDim];
        for (int j = 0; j < _hiddenDim; j++)
        {
            h2[j] = _genB2![j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                h2[j] += h1[i] * _genW2![i, j];
            }
            h2[j] = LeakyReLU(h2[j]);
        }

        // Output layer (tanh for bounded output)
        var output = new double[_inputDim];
        for (int j = 0; j < _inputDim; j++)
        {
            output[j] = _genB3![j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                output[j] += h2[i] * _genW3![i, j];
            }
            output[j] = Math.Tanh(output[j]);
        }

        return (h1, h2, output);
    }

    private double[] BackpropDiscriminatorToInput(double[] x, double[] h2, double dOut)
    {
        // Backprop through discriminator without updating weights
        // to get gradient w.r.t. input

        // Gradient through output layer
        var dH2 = new double[_hiddenDim];
        for (int i = 0; i < _hiddenDim; i++)
        {
            dH2[i] = _discW3![i, 0] * dOut;
        }

        // LeakyReLU derivative for h2
        for (int i = 0; i < _hiddenDim; i++)
        {
            if (h2[i] < 0) dH2[i] *= 0.2;
        }

        // We need h1 to continue backprop - recompute it
        var h1 = new double[_hiddenDim];
        for (int j = 0; j < _hiddenDim; j++)
        {
            h1[j] = _discB1![j];
            for (int i = 0; i < _inputDim; i++)
            {
                h1[j] += x[i] * _discW1![i, j];
            }
            h1[j] = LeakyReLU(h1[j]);
        }

        // Gradient through layer 2
        var dH1 = new double[_hiddenDim];
        for (int i = 0; i < _hiddenDim; i++)
        {
            for (int j = 0; j < _hiddenDim; j++)
            {
                dH1[i] += _discW2![i, j] * dH2[j];
            }
        }

        // LeakyReLU derivative for h1
        for (int i = 0; i < _hiddenDim; i++)
        {
            if (h1[i] < 0) dH1[i] *= 0.2;
        }

        // Gradient through layer 1 to input
        var dX = new double[_inputDim];
        for (int i = 0; i < _inputDim; i++)
        {
            for (int j = 0; j < _hiddenDim; j++)
            {
                dX[i] += _discW1![i, j] * dH1[j];
            }
        }

        return dX;
    }

    private void BackpropGenerator(double[] z, double[] h1, double[] h2, double[] dOutput, double lr)
    {
        // Tanh derivative: d/dx tanh(x) = 1 - tanh(x)^2
        // We need the pre-tanh values, but we stored the post-tanh values
        // We can recover: if output = tanh(pre), then pre = atanh(output)
        // Alternatively, dOutput * (1 - output^2) gives us the gradient through tanh

        // Recompute output layer pre-activation and tanh values
        var outputPre = new double[_inputDim];
        var output = new double[_inputDim];
        for (int j = 0; j < _inputDim; j++)
        {
            outputPre[j] = _genB3![j];
            for (int i = 0; i < _hiddenDim; i++)
            {
                outputPre[j] += h2[i] * _genW3![i, j];
            }
            output[j] = Math.Tanh(outputPre[j]);
        }

        // Apply tanh derivative
        var dOutputPre = new double[_inputDim];
        for (int j = 0; j < _inputDim; j++)
        {
            dOutputPre[j] = dOutput[j] * (1 - output[j] * output[j]);
        }

        // Gradient through output layer
        var dH2 = new double[_hiddenDim];
        for (int i = 0; i < _hiddenDim; i++)
        {
            for (int j = 0; j < _inputDim; j++)
            {
                _genW3![i, j] -= lr * h2[i] * dOutputPre[j];
                dH2[i] += _genW3[i, j] * dOutputPre[j];
            }
        }
        for (int j = 0; j < _inputDim; j++)
        {
            _genB3![j] -= lr * dOutputPre[j];
        }

        // LeakyReLU derivative for h2
        for (int i = 0; i < _hiddenDim; i++)
        {
            if (h2[i] < 0) dH2[i] *= 0.2;
        }

        // Gradient through layer 2
        var dH1 = new double[_hiddenDim];
        for (int i = 0; i < _hiddenDim; i++)
        {
            for (int j = 0; j < _hiddenDim; j++)
            {
                _genW2![i, j] -= lr * h1[i] * dH2[j];
                dH1[i] += _genW2[i, j] * dH2[j];
            }
        }
        for (int j = 0; j < _hiddenDim; j++)
        {
            _genB2![j] -= lr * dH2[j];
        }

        // LeakyReLU derivative for h1
        for (int i = 0; i < _hiddenDim; i++)
        {
            if (h1[i] < 0) dH1[i] *= 0.2;
        }

        // Gradient through layer 1
        for (int i = 0; i < _latentDim; i++)
        {
            for (int j = 0; j < _hiddenDim; j++)
            {
                _genW1![i, j] -= lr * z[i] * dH1[j];
            }
        }
        for (int j = 0; j < _hiddenDim; j++)
        {
            _genB1![j] -= lr * dH1[j];
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
            var x = new double[_inputDim];
            for (int j = 0; j < _inputDim; j++)
            {
                x[j] = (NumOps.ToDouble(X[i, j]) - dataMeans[j]) / dataStds[j];
            }

            // Find optimal z via gradient descent
            var z = SampleLatent();
            var bestZ = (double[])z.Clone();
            double bestLoss = double.MaxValue;
            double zLr = 0.1; // Learning rate for z optimization

            for (int step = 0; step < _inferenceSteps; step++)
            {
                var xGen = Generate(z);
                var (_, featGen) = Discriminate(xGen);
                var (_, featReal) = Discriminate(x);

                // Reconstruction loss
                double reconLoss = 0;
                for (int j = 0; j < _inputDim; j++)
                {
                    reconLoss += Math.Pow(x[j] - xGen[j], 2);
                }

                // Feature matching loss
                double featLoss = 0;
                for (int j = 0; j < _hiddenDim; j++)
                {
                    featLoss += Math.Pow(featReal[j] - featGen[j], 2);
                }

                double loss = reconLoss + 0.1 * featLoss;
                if (loss < bestLoss)
                {
                    bestLoss = loss;
                    Array.Copy(z, bestZ, _latentDim);
                }

                // Compute gradient of loss w.r.t. z using finite differences
                // dL/dz_j = (L(z + eps*e_j) - L(z - eps*e_j)) / (2*eps)
                double eps = 1e-4;
                var dz = new double[_latentDim];

                for (int j = 0; j < _latentDim; j++)
                {
                    // Perturb z[j] positively
                    z[j] += eps;
                    var xGenPlus = Generate(z);
                    var (_, featGenPlus) = Discriminate(xGenPlus);
                    double lossPlus = 0;
                    for (int k = 0; k < _inputDim; k++)
                    {
                        lossPlus += Math.Pow(x[k] - xGenPlus[k], 2);
                    }
                    for (int k = 0; k < _hiddenDim; k++)
                    {
                        lossPlus += 0.1 * Math.Pow(featReal[k] - featGenPlus[k], 2);
                    }

                    // Perturb z[j] negatively
                    z[j] -= 2 * eps;
                    var xGenMinus = Generate(z);
                    var (_, featGenMinus) = Discriminate(xGenMinus);
                    double lossMinus = 0;
                    for (int k = 0; k < _inputDim; k++)
                    {
                        lossMinus += Math.Pow(x[k] - xGenMinus[k], 2);
                    }
                    for (int k = 0; k < _hiddenDim; k++)
                    {
                        lossMinus += 0.1 * Math.Pow(featReal[k] - featGenMinus[k], 2);
                    }

                    // Restore z[j] and compute gradient
                    z[j] += eps;
                    dz[j] = (lossPlus - lossMinus) / (2 * eps);
                }

                // Update z using gradient descent with regularization
                for (int j = 0; j < _latentDim; j++)
                {
                    z[j] -= zLr * (dz[j] + 0.001 * z[j]); // Gradient + L2 regularization
                }
            }

            scores[i] = NumOps.FromDouble(bestLoss);
        }

        return scores;
    }
}

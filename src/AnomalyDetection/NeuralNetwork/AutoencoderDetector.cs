using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.NeuralNetwork;

/// <summary>
/// Implements an Autoencoder-based method for anomaly detection using reconstruction error.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> An autoencoder is a neural network that learns to compress data into
/// a smaller representation and then reconstruct it. Normal data can be reconstructed well,
/// but anomalies (which the autoencoder hasn't learned to represent) will have high reconstruction error.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Training a simple autoencoder (encoder-decoder network) on the data
/// 2. For each data point, computing the reconstruction error (how different the output is from input)
/// 3. Points with high reconstruction error are likely anomalies
/// </para>
/// <para>
/// <b>When to use:</b> Autoencoder-based detection is particularly effective for:
/// - High-dimensional data
/// - Data with complex patterns
/// - When you want the detector to automatically learn important features
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Encoding dimension: Auto (input_dim/2)
/// - Epochs: 50
/// - Learning rate: 0.01
/// - Batch size: 32
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// This implementation uses a simple fully-connected autoencoder with one hidden layer.
/// For more complex scenarios, consider using a deeper architecture.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Autoencoder)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("Reducing the Dimensionality of Data with Neural Networks", "https://doi.org/10.1126/science.1127647")]
public partial class AutoencoderDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _encodingDim;
    private readonly int _epochs;
    private readonly double _learningRate;
    private readonly int _batchSize;

    // Weights for the simple autoencoder
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _encoderWeights;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _encoderBias;
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _decoderWeights;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _decoderBias;

    private int _inputDim;

    /// <summary>
    /// Gets the encoding dimension (bottleneck size).
    /// </summary>
    public int EncodingDim => _encodingDim > 0 ? _encodingDim : Math.Max(1, _inputDim / 2);

    /// <summary>
    /// Gets the number of training epochs.
    /// </summary>
    public int Epochs => _epochs;

    /// <summary>
    /// Gets the learning rate.
    /// </summary>
    public double LearningRate => _learningRate;

    /// <summary>
    /// Creates a new Autoencoder-based anomaly detector.
    /// </summary>
    /// <param name="encodingDim">
    /// The dimension of the encoded (compressed) representation. Default is 0, which
    /// means it will be automatically set to input_dim/2.
    /// Smaller values create more compression, potentially losing detail but better at detecting global anomalies.
    /// </param>
    /// <param name="epochs">
    /// The number of training epochs. Default is 50.
    /// More epochs may improve the reconstruction quality but increase training time.
    /// </param>
    /// <param name="learningRate">
    /// The learning rate for training. Default is 0.01.
    /// Lower values may need more epochs but can find better solutions.
    /// </param>
    /// <param name="batchSize">
    /// The batch size for training. Default is 32.
    /// </param>
    /// <param name="contamination">
    /// The expected proportion of anomalies in the data. Default is 0.1 (10%).
    /// </param>
    /// <param name="randomSeed">
    /// Random seed for reproducibility. Default is 42.
    /// </param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The key parameters to tune are:
    /// - encodingDim: How much to compress the data (smaller = more compression)
    /// - epochs: How long to train (more = better reconstruction, but slower)
    ///
    /// Start with defaults and adjust if needed. If the detector misses obvious anomalies,
    /// try more epochs or a smaller encoding dimension.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when encodingDim is negative, epochs is less than 1, learningRate is not positive,
    /// or batchSize is less than 1.
    /// </exception>
    public AutoencoderDetector(
        int encodingDim = 0,
        int epochs = 50,
        double learningRate = 0.01,
        int batchSize = 32,
        double contamination = 0.1,
        int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (encodingDim < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(encodingDim),
                "Encoding dimension must be non-negative. Use 0 for auto-detection.");
        }

        if (epochs < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(epochs),
                "Number of epochs must be at least 1.");
        }

        if (learningRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate),
                "Learning rate must be positive.");
        }

        if (batchSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(batchSize),
                "Batch size must be at least 1.");
        }

        _encodingDim = encodingDim;
        _epochs = epochs;
        _learningRate = learningRate;
        _batchSize = batchSize;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        _inputDim = X.Columns;
        int effectiveEncodingDim = _encodingDim > 0 ? _encodingDim : Math.Max(1, _inputDim / 2);

        // Initialize weights using Xavier initialization
        InitializeWeights(_inputDim, effectiveEncodingDim);

        // Train the autoencoder
        Train(X);

        // Calculate reconstruction errors for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
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

        if (_inputDim != 0 && X.Columns != _inputDim)
        {
            throw new ArgumentException(
                $"Input has {X.Columns} features, but model was fitted with {_inputDim} features.",
                nameof(X));
        }

        // Mean squared reconstruction error per row: higher reconstruction error = more anomalous
        var engine = Engine;
        var input = AnomalyTensorOps.ToTensor(X);
        var (_, _, reconstruction) = ForwardPass(input);
        return engine.TensorDivideScalar(engine.ReduceSum(engine.TensorSquare(engine.TensorSubtract(input, reconstruction)),
            new[] { 1 }, false), NumOps.FromDouble(X.Columns)).ToVector();
    }

    private void InitializeWeights(int inputDim, int encodingDim)
    {
        // Xavier initialization: weights ~ N(0, sqrt(2/(fan_in + fan_out)))
        double encoderStd = Math.Sqrt(2.0 / (inputDim + encodingDim));
        double decoderStd = Math.Sqrt(2.0 / (encodingDim + inputDim));

        _encoderWeights = new Matrix<T>(inputDim, encodingDim);
        _encoderBias = new Vector<T>(encodingDim);
        _decoderWeights = new Matrix<T>(encodingDim, inputDim);
        _decoderBias = new Vector<T>(inputDim);

        // Biases start at zero
        InitializeWeightMatrix(_encoderWeights, encoderStd);
        InitializeWeightMatrix(_decoderWeights, decoderStd);
    }

    private void InitializeWeightMatrix(Matrix<T> weights, double stdDev)
    {
        for (int i = 0; i < weights.Rows; i++)
        {
            for (int j = 0; j < weights.Columns; j++)
            {
                weights[i, j] = NumOps.FromDouble(SampleGaussian(0, stdDev));
            }
        }
    }

    private double SampleGaussian(double mean, double stdDev)
    {
        // Box-Muller transform
        double u1 = 1.0 - _random.NextDouble();
        double u2 = 1.0 - _random.NextDouble();
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return mean + stdDev * randStdNormal;
    }

    private void Train(Matrix<T> X)
    {
        var engine = Engine;
        int n = X.Rows;
        if (_encoderWeights == null || _encoderBias == null || _decoderWeights == null || _decoderBias == null)
        {
            throw new InvalidOperationException("Weights not initialized.");
        }

        var data = AnomalyTensorOps.ToTensor(X);
        T lr = NumOps.FromDouble(_learningRate);

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            // Shuffle indices (Fisher-Yates)
            var indices = Enumerable.Range(0, n).ToArray();
            for (int i = n - 1; i > 0; i--)
            {
                int j = _random.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }

            // Mini-batch gradient descent on the MSE (gradients summed over the batch, then averaged)
            for (int batchStart = 0; batchStart < n; batchStart += _batchSize)
            {
                int currentBatchSize = Math.Min(batchStart + _batchSize, n) - batchStart;
                var batchRows = new int[currentBatchSize];
                Array.Copy(indices, batchStart, batchRows, 0, currentBatchSize);
                var input = AnomalyTensorOps.SelectRows(data, batchRows);
                var (encoded, activated, reconstruction) = ForwardPass(input);
                var (encoderWeightsGrad, encoderBiasGrad, decoderWeightsGrad, decoderBiasGrad) =
                    ComputeGradients(input, encoded, activated, reconstruction);

                T batchSizeT = NumOps.FromDouble(currentBatchSize);
                _encoderWeights = Update(AnomalyTensorOps.ToTensor(_encoderWeights), encoderWeightsGrad, lr, batchSizeT).ToMatrix();
                _encoderBias = Update(_encoderBias.AsTensor(), encoderBiasGrad, lr, batchSizeT).ToVector();
                _decoderWeights = Update(AnomalyTensorOps.ToTensor(_decoderWeights), decoderWeightsGrad, lr, batchSizeT).ToMatrix();
                _decoderBias = Update(_decoderBias.AsTensor(), decoderBiasGrad, lr, batchSizeT).ToVector();
            }
        }
    }

    // Encoder pre-activation, ReLU code and linear reconstruction for a batch of rows.
    private (Tensor<T> encoded, Tensor<T> activated, Tensor<T> reconstruction) ForwardPass(Tensor<T> input)
    {
        var encoderWeights = _encoderWeights;
        var encoderBias = _encoderBias;
        var decoderWeights = _decoderWeights;
        var decoderBias = _decoderBias;
        if (encoderWeights == null || encoderBias == null || decoderWeights == null || decoderBias == null)
        {
            throw new InvalidOperationException("Weights not initialized.");
        }

        var engine = Engine;
        int rows = input.Shape[0];
        var encoded = engine.TensorAdd(engine.TensorMatMul(input, AnomalyTensorOps.ToTensor(encoderWeights)),
            AnomalyTensorOps.BroadcastRow(encoderBias.AsTensor(), rows));
        var activated = engine.TensorMax(encoded, NumOps.Zero);
        var reconstruction = engine.TensorAdd(engine.TensorMatMul(activated, AnomalyTensorOps.ToTensor(decoderWeights)),
            AnomalyTensorOps.BroadcastRow(decoderBias.AsTensor(), rows));
        return (encoded, activated, reconstruction);
    }

    // Gradients of the summed squared error over a batch: output error d_output = reconstruction - input
    // (MSE derivative), back-propagated through the decoder and the encoder ReLU.
    private (Tensor<T> encoderWeightsGrad, Tensor<T> encoderBiasGrad, Tensor<T> decoderWeightsGrad, Tensor<T> decoderBiasGrad)
        ComputeGradients(Tensor<T> input, Tensor<T> encoded, Tensor<T> activated, Tensor<T> reconstruction)
    {
        var decoderWeights = _decoderWeights ?? throw new InvalidOperationException("Weights not initialized.");
        var engine = Engine;
        var outputError = engine.TensorSubtract(reconstruction, input);
        var decoderWeightsGrad = engine.TensorMatMul(engine.TensorTranspose(activated), outputError);
        var decoderBiasGrad = engine.ReduceSum(outputError, new[] { 0 }, false);
        var hiddenError = engine.TensorMultiply(
            engine.TensorMatMul(outputError, engine.TensorTranspose(AnomalyTensorOps.ToTensor(decoderWeights))),
            engine.TensorGreaterThan(encoded, NumOps.Zero));
        var encoderWeightsGrad = engine.TensorMatMul(engine.TensorTranspose(input), hiddenError);
        var encoderBiasGrad = engine.ReduceSum(hiddenError, new[] { 0 }, false);
        return (encoderWeightsGrad, encoderBiasGrad, decoderWeightsGrad, decoderBiasGrad);
    }

    // weights - lr * (grad / batchSize)
    private Tensor<T> Update(Tensor<T> weights, Tensor<T> grad, T lr, T batchSize)
    {
        var engine = Engine;
        return engine.TensorSubtract(weights, engine.TensorMultiplyScalar(engine.TensorDivideScalar(grad, batchSize), lr));
    }

    /// <summary>
    /// Gets the encoded representation of the input data.
    /// </summary>
    /// <param name="X">The input data matrix.</param>
    /// <returns>The encoded representations.</returns>
    public Matrix<T> Encode(Matrix<T> X)
    {
        EnsureFitted();
        ValidateInput(X);

        if (_inputDim != 0 && X.Columns != _inputDim)
        {
            throw new ArgumentException(
                $"Input has {X.Columns} features, but model was fitted with {_inputDim} features.",
                nameof(X));
        }

        var encoderWeights = _encoderWeights;
        if (encoderWeights == null)
        {
            throw new InvalidOperationException("Weights not initialized.");
        }

        var (_, activated, _) = ForwardPass(AnomalyTensorOps.ToTensor(X));
        return activated.ToMatrix();
    }
}

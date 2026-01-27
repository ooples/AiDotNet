using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.OutlierRemoval;

/// <summary>
/// Implements an Autoencoder-based method for outlier detection using reconstruction error.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> An autoencoder is a neural network that learns to compress data into
/// a smaller representation and then reconstruct it. Normal data can be reconstructed well,
/// but outliers (which the autoencoder hasn't learned to represent) will have high reconstruction error.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Training a simple autoencoder (encoder-decoder network) on the data
/// 2. For each data point, computing the reconstruction error (how different the output is from input)
/// 3. Points with high reconstruction error are likely outliers
/// </para>
/// <para>
/// <b>When to use:</b> Autoencoder-based detection is particularly effective for:
/// - High-dimensional data
/// - Data with complex patterns
/// - When you want the detector to automatically learn important features
/// </para>
/// <para>
/// This implementation uses a simple fully-connected autoencoder with one hidden layer.
/// For more complex scenarios, consider using a deeper architecture.
/// </para>
/// </remarks>
public class AutoencoderOutlierDetector<T> : OutlierDetectorBase<T>
{
    private readonly int _encodingDim;
    private readonly int _epochs;
    private readonly double _learningRate;
    private readonly int _batchSize;

    // Weights for the simple autoencoder
    private Matrix<T>? _encoderWeights;
    private Vector<T>? _encoderBias;
    private Matrix<T>? _decoderWeights;
    private Vector<T>? _decoderBias;

    private int _inputDim;

    /// <summary>
    /// Creates a new Autoencoder-based outlier detector.
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
    /// The expected proportion of outliers in the data. Default is 0.1 (10%).
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
    /// Start with defaults and adjust if needed. If the detector misses obvious outliers,
    /// try more epochs or a smaller encoding dimension.
    /// </para>
    /// </remarks>
    public AutoencoderOutlierDetector(
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
        var trainingScores = DecisionFunctionInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    /// <inheritdoc/>
    public override Vector<T> DecisionFunction(Matrix<T> X)
    {
        EnsureFitted();
        return DecisionFunctionInternal(X);
    }

    private Vector<T> DecisionFunctionInternal(Matrix<T> X)
    {
        ValidateInput(X);

        var scores = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            var input = X.GetRow(i);
            var reconstruction = Reconstruct(input);
            T error = ComputeReconstructionError(input, reconstruction);

            // Negate error so lower values = outliers (consistent with our Predict method)
            // Higher reconstruction error = more anomalous
            scores[i] = NumOps.Negate(error);
        }

        return scores;
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

        // Initialize encoder weights
        for (int i = 0; i < inputDim; i++)
        {
            for (int j = 0; j < encodingDim; j++)
            {
                _encoderWeights[i, j] = NumOps.FromDouble(SampleGaussian(0, encoderStd));
            }
        }

        // Initialize encoder bias to zero
        for (int i = 0; i < encodingDim; i++)
        {
            _encoderBias[i] = NumOps.Zero;
        }

        // Initialize decoder weights
        for (int i = 0; i < encodingDim; i++)
        {
            for (int j = 0; j < inputDim; j++)
            {
                _decoderWeights[i, j] = NumOps.FromDouble(SampleGaussian(0, decoderStd));
            }
        }

        // Initialize decoder bias to zero
        for (int i = 0; i < inputDim; i++)
        {
            _decoderBias[i] = NumOps.Zero;
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
        int n = X.Rows;

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            // Shuffle indices
            var indices = Enumerable.Range(0, n).ToArray();
            for (int i = n - 1; i > 0; i--)
            {
                int j = _random.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }

            // Process mini-batches
            for (int batchStart = 0; batchStart < n; batchStart += _batchSize)
            {
                int batchEnd = Math.Min(batchStart + _batchSize, n);
                int currentBatchSize = batchEnd - batchStart;

                // Accumulate gradients
                var encoderWeightsGrad = new Matrix<T>(_encoderWeights!.Rows, _encoderWeights.Columns);
                var encoderBiasGrad = new Vector<T>(_encoderBias!.Length);
                var decoderWeightsGrad = new Matrix<T>(_decoderWeights!.Rows, _decoderWeights.Columns);
                var decoderBiasGrad = new Vector<T>(_decoderBias!.Length);

                for (int b = batchStart; b < batchEnd; b++)
                {
                    int idx = indices[b];
                    var input = X.GetRow(idx);

                    // Forward pass
                    var (encoded, activated, reconstruction) = ForwardPass(input);

                    // Backward pass
                    ComputeGradients(input, encoded, activated, reconstruction,
                        encoderWeightsGrad, encoderBiasGrad, decoderWeightsGrad, decoderBiasGrad);
                }

                // Update weights (average gradients over batch)
                T batchSizeT = NumOps.FromDouble(currentBatchSize);
                T lr = NumOps.FromDouble(_learningRate);

                UpdateWeights(_encoderWeights, encoderWeightsGrad, lr, batchSizeT);
                UpdateBias(_encoderBias, encoderBiasGrad, lr, batchSizeT);
                UpdateWeights(_decoderWeights, decoderWeightsGrad, lr, batchSizeT);
                UpdateBias(_decoderBias, decoderBiasGrad, lr, batchSizeT);
            }
        }
    }

    private (Vector<T> encoded, Vector<T> activated, Vector<T> reconstruction) ForwardPass(Vector<T> input)
    {
        // Encoder: encoded = ReLU(input * W_enc + b_enc)
        var encoded = new Vector<T>(_encoderWeights!.Columns);

        for (int j = 0; j < _encoderWeights.Columns; j++)
        {
            T sum = _encoderBias![j];
            for (int i = 0; i < _encoderWeights.Rows; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(input[i], _encoderWeights[i, j]));
            }

            encoded[j] = sum;
        }

        // Apply ReLU activation
        var activated = new Vector<T>(encoded.Length);
        for (int i = 0; i < encoded.Length; i++)
        {
            activated[i] = NumOps.GreaterThan(encoded[i], NumOps.Zero) ? encoded[i] : NumOps.Zero;
        }

        // Decoder: reconstruction = activated * W_dec + b_dec (no activation for output)
        var reconstruction = new Vector<T>(_decoderWeights!.Columns);

        for (int j = 0; j < _decoderWeights.Columns; j++)
        {
            T sum = _decoderBias![j];
            for (int i = 0; i < _decoderWeights.Rows; i++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(activated[i], _decoderWeights[i, j]));
            }

            reconstruction[j] = sum;
        }

        return (encoded, activated, reconstruction);
    }

    private void ComputeGradients(
        Vector<T> input,
        Vector<T> encoded,
        Vector<T> activated,
        Vector<T> reconstruction,
        Matrix<T> encoderWeightsGrad,
        Vector<T> encoderBiasGrad,
        Matrix<T> decoderWeightsGrad,
        Vector<T> decoderBiasGrad)
    {
        // Output error: d_output = reconstruction - input (MSE derivative)
        var outputError = new Vector<T>(reconstruction.Length);
        for (int i = 0; i < reconstruction.Length; i++)
        {
            outputError[i] = NumOps.Subtract(reconstruction[i], input[i]);
        }

        // Decoder gradients
        for (int i = 0; i < _decoderWeights!.Rows; i++)
        {
            for (int j = 0; j < _decoderWeights.Columns; j++)
            {
                T grad = NumOps.Multiply(activated[i], outputError[j]);
                decoderWeightsGrad[i, j] = NumOps.Add(decoderWeightsGrad[i, j], grad);
            }
        }

        for (int j = 0; j < _decoderBias!.Length; j++)
        {
            decoderBiasGrad[j] = NumOps.Add(decoderBiasGrad[j], outputError[j]);
        }

        // Backpropagate to encoder
        var hiddenError = new Vector<T>(_decoderWeights.Rows);

        for (int i = 0; i < _decoderWeights.Rows; i++)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < _decoderWeights.Columns; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(outputError[j], _decoderWeights[i, j]));
            }

            // Apply ReLU derivative
            hiddenError[i] = NumOps.GreaterThan(encoded[i], NumOps.Zero) ? sum : NumOps.Zero;
        }

        // Encoder gradients
        for (int i = 0; i < _encoderWeights!.Rows; i++)
        {
            for (int j = 0; j < _encoderWeights.Columns; j++)
            {
                T grad = NumOps.Multiply(input[i], hiddenError[j]);
                encoderWeightsGrad[i, j] = NumOps.Add(encoderWeightsGrad[i, j], grad);
            }
        }

        for (int j = 0; j < _encoderBias!.Length; j++)
        {
            encoderBiasGrad[j] = NumOps.Add(encoderBiasGrad[j], hiddenError[j]);
        }
    }

    private void UpdateWeights(Matrix<T> weights, Matrix<T> grad, T lr, T batchSize)
    {
        for (int i = 0; i < weights.Rows; i++)
        {
            for (int j = 0; j < weights.Columns; j++)
            {
                T avgGrad = NumOps.Divide(grad[i, j], batchSize);
                T update = NumOps.Multiply(lr, avgGrad);
                weights[i, j] = NumOps.Subtract(weights[i, j], update);
            }
        }
    }

    private void UpdateBias(Vector<T> bias, Vector<T> grad, T lr, T batchSize)
    {
        for (int i = 0; i < bias.Length; i++)
        {
            T avgGrad = NumOps.Divide(grad[i], batchSize);
            T update = NumOps.Multiply(lr, avgGrad);
            bias[i] = NumOps.Subtract(bias[i], update);
        }
    }

    private Vector<T> Reconstruct(Vector<T> input)
    {
        var (_, _, reconstruction) = ForwardPass(input);
        return reconstruction;
    }

    private T ComputeReconstructionError(Vector<T> original, Vector<T> reconstruction)
    {
        // Mean Squared Error
        T sum = NumOps.Zero;

        for (int i = 0; i < original.Length; i++)
        {
            T diff = NumOps.Subtract(original[i], reconstruction[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }

        return NumOps.Divide(sum, NumOps.FromDouble(original.Length));
    }
}

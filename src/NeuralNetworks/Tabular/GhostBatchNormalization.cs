using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Implements Ghost Batch Normalization, a regularization technique used in TabNet
/// that applies batch normalization to virtual mini-batches within each actual batch.
/// </summary>
/// <remarks>
/// <para>
/// Ghost Batch Normalization divides each training batch into smaller "virtual batches"
/// and computes separate normalization statistics for each. This provides a regularization
/// effect similar to using smaller batch sizes without the computational overhead.
/// </para>
/// <para>
/// <b>For Beginners:</b> Batch Normalization helps neural networks train faster by
/// normalizing the inputs to each layer. Ghost Batch Normalization takes this further
/// by adding controlled randomness through virtual batches.
///
/// Imagine you have a batch of 256 samples:
/// - Standard Batch Norm: Computes mean/variance over all 256 samples
/// - Ghost Batch Norm (virtual size 64): Computes 4 separate mean/variance calculations,
///   one for each group of 64 samples
///
/// This variation in statistics acts as regularization, helping prevent overfitting.
/// It's particularly effective for tabular data where overfitting is common.
/// </para>
/// <para>
/// Reference: "TabNet: Attentive Interpretable Tabular Learning" (Arik &amp; Pfister, AAAI 2021)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class GhostBatchNormalization<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _numFeatures;
    private readonly int _virtualBatchSize;
    private readonly double _momentum;
    private readonly double _epsilon;

    // Learnable parameters
    private Vector<T> _gamma; // Scale parameter
    private Vector<T> _beta;  // Shift parameter

    // Running statistics for inference
    private Vector<T> _runningMean;
    private Vector<T> _runningVar;

    // Gradients
    private Vector<T>? _gammaGrad;
    private Vector<T>? _betaGrad;

    // Cache for backward pass
    private Tensor<T>? _inputCache;
    private Tensor<T>? _normalizedCache;
    private Vector<T>[]? _batchMeans;
    private Vector<T>[]? _batchVars;
    private int _numVirtualBatches;

    /// <summary>
    /// Gets the name of this layer.
    /// </summary>
    public string Name => "GhostBatchNormalization";

    /// <summary>
    /// Gets whether this layer supports training.
    /// </summary>
    public bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the GhostBatchNormalization class.
    /// </summary>
    /// <param name="numFeatures">The number of features (channels) to normalize.</param>
    /// <param name="virtualBatchSize">The size of each virtual batch. Default is 128.</param>
    /// <param name="momentum">The momentum for running statistics. Default is 0.02.</param>
    /// <param name="epsilon">Small constant for numerical stability. Default is 1e-5.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When creating Ghost Batch Normalization:
    /// - numFeatures: Should match the number of features in your data
    /// - virtualBatchSize: Smaller = more regularization (try 32-128)
    /// - momentum: How quickly running stats adapt (smaller = slower adaptation)
    /// - epsilon: Prevents division by zero (rarely needs changing)
    /// </para>
    /// </remarks>
    public GhostBatchNormalization(
        int numFeatures,
        int virtualBatchSize = 128,
        double momentum = 0.02,
        double epsilon = 1e-5)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _numFeatures = numFeatures;
        _virtualBatchSize = virtualBatchSize;
        _momentum = momentum;
        _epsilon = epsilon;

        // Initialize learnable parameters
        _gamma = new Vector<T>(_numFeatures);
        _beta = new Vector<T>(_numFeatures);

        // Initialize running statistics
        _runningMean = new Vector<T>(_numFeatures);
        _runningVar = new Vector<T>(_numFeatures);

        // Initialize gamma to 1 and beta to 0
        for (int i = 0; i < _numFeatures; i++)
        {
            _gamma[i] = _numOps.One;
            _beta[i] = _numOps.Zero;
            _runningMean[i] = _numOps.Zero;
            _runningVar[i] = _numOps.One;
        }
    }

    /// <summary>
    /// Performs the forward pass through the Ghost Batch Normalization layer.
    /// </summary>
    /// <param name="input">The input tensor of shape [batch_size, num_features].</param>
    /// <returns>The normalized output tensor.</returns>
    /// <remarks>
    /// <para>
    /// During training, the input is divided into virtual batches and each is normalized
    /// separately. During inference, the running statistics are used instead.
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input)
    {
        if (input.Rank != 2)
        {
            throw new ArgumentException($"Expected 2D input [batch_size, features], got {input.Rank}D", nameof(input));
        }

        int batchSize = input.Shape[0];
        int features = input.Shape[1];

        if (features != _numFeatures)
        {
            throw new ArgumentException($"Expected {_numFeatures} features, got {features}", nameof(input));
        }

        var output = new Tensor<T>(input.Shape);

        // Determine number of virtual batches
        _numVirtualBatches = Math.Max(1, (batchSize + _virtualBatchSize - 1) / _virtualBatchSize);
        int actualVirtualSize = (batchSize + _numVirtualBatches - 1) / _numVirtualBatches;

        _inputCache = input;
        _normalizedCache = new Tensor<T>(input.Shape);
        _batchMeans = new Vector<T>[_numVirtualBatches];
        _batchVars = new Vector<T>[_numVirtualBatches];

        // Process each virtual batch
        for (int vb = 0; vb < _numVirtualBatches; vb++)
        {
            int startIdx = vb * actualVirtualSize;
            int endIdx = Math.Min(startIdx + actualVirtualSize, batchSize);
            int virtualSize = endIdx - startIdx;

            if (virtualSize <= 0) continue;

            // Compute mean for this virtual batch
            var mean = new Vector<T>(_numFeatures);
            for (int f = 0; f < features; f++)
            {
                var sum = _numOps.Zero;
                for (int b = startIdx; b < endIdx; b++)
                {
                    sum = _numOps.Add(sum, input[b * features + f]);
                }
                mean[f] = _numOps.Divide(sum, _numOps.FromDouble(virtualSize));
            }

            // Compute variance for this virtual batch
            var variance = new Vector<T>(_numFeatures);
            for (int f = 0; f < features; f++)
            {
                var sumSq = _numOps.Zero;
                for (int b = startIdx; b < endIdx; b++)
                {
                    var diff = _numOps.Subtract(input[b * features + f], mean[f]);
                    sumSq = _numOps.Add(sumSq, _numOps.Multiply(diff, diff));
                }
                variance[f] = _numOps.Divide(sumSq, _numOps.FromDouble(virtualSize));
            }

            _batchMeans[vb] = mean;
            _batchVars[vb] = variance;

            // Normalize and apply scale/shift for this virtual batch
            for (int b = startIdx; b < endIdx; b++)
            {
                for (int f = 0; f < features; f++)
                {
                    // Normalize: (x - mean) / sqrt(var + eps)
                    var normalized = _numOps.Divide(
                        _numOps.Subtract(input[b * features + f], mean[f]),
                        _numOps.FromDouble(Math.Sqrt(_numOps.ToDouble(variance[f]) + _epsilon)));

                    _normalizedCache[b * features + f] = normalized;

                    // Scale and shift: gamma * normalized + beta
                    output[b * features + f] = _numOps.Add(
                        _numOps.Multiply(_gamma[f], normalized),
                        _beta[f]);
                }
            }

            // Update running statistics
            for (int f = 0; f < features; f++)
            {
                _runningMean[f] = _numOps.Add(
                    _numOps.Multiply(_numOps.FromDouble(1 - _momentum), _runningMean[f]),
                    _numOps.Multiply(_numOps.FromDouble(_momentum), mean[f]));

                _runningVar[f] = _numOps.Add(
                    _numOps.Multiply(_numOps.FromDouble(1 - _momentum), _runningVar[f]),
                    _numOps.Multiply(_numOps.FromDouble(_momentum), variance[f]));
            }
        }

        return output;
    }

    /// <summary>
    /// Performs the forward pass using running statistics (inference mode).
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The normalized output tensor.</returns>
    public Tensor<T> ForwardInference(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int features = input.Shape[1];
        var output = new Tensor<T>(input.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < features; f++)
            {
                var normalized = _numOps.Divide(
                    _numOps.Subtract(input[b * features + f], _runningMean[f]),
                    _numOps.FromDouble(Math.Sqrt(_numOps.ToDouble(_runningVar[f]) + _epsilon)));

                output[b * features + f] = _numOps.Add(
                    _numOps.Multiply(_gamma[f], normalized),
                    _beta[f]);
            }
        }

        return output;
    }

    /// <summary>
    /// Performs the backward pass to compute gradients.
    /// </summary>
    /// <param name="gradOutput">The gradient flowing back from the next layer.</param>
    /// <returns>The gradient with respect to the input.</returns>
    public Tensor<T> Backward(Tensor<T> gradOutput)
    {
        if (_inputCache == null || _normalizedCache == null || _batchMeans == null || _batchVars == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        int batchSize = gradOutput.Shape[0];
        int features = gradOutput.Shape[1];
        var gradInput = new Tensor<T>(gradOutput.Shape);

        // Initialize gradients
        _gammaGrad = new Vector<T>(_numFeatures);
        _betaGrad = new Vector<T>(_numFeatures);

        int actualVirtualSize = (batchSize + _numVirtualBatches - 1) / _numVirtualBatches;

        // Process each virtual batch
        for (int vb = 0; vb < _numVirtualBatches; vb++)
        {
            int startIdx = vb * actualVirtualSize;
            int endIdx = Math.Min(startIdx + actualVirtualSize, batchSize);
            int virtualSize = endIdx - startIdx;

            if (virtualSize <= 0) continue;

            var mean = _batchMeans[vb];
            var variance = _batchVars[vb];

            for (int f = 0; f < features; f++)
            {
                // Compute gradients for gamma and beta
                var dGamma = _numOps.Zero;
                var dBeta = _numOps.Zero;

                for (int b = startIdx; b < endIdx; b++)
                {
                    dGamma = _numOps.Add(dGamma, _numOps.Multiply(gradOutput[b * features + f], _normalizedCache[b * features + f]));
                    dBeta = _numOps.Add(dBeta, gradOutput[b * features + f]);
                }

                _gammaGrad[f] = _numOps.Add(_gammaGrad[f], dGamma);
                _betaGrad[f] = _numOps.Add(_betaGrad[f], dBeta);

                // Compute gradient with respect to input
                var std = _numOps.FromDouble(Math.Sqrt(_numOps.ToDouble(variance[f]) + _epsilon));
                var invStd = _numOps.Divide(_numOps.One, std);

                // Sum terms for batch gradient
                var sumDy = _numOps.Zero;
                var sumDyXhat = _numOps.Zero;

                for (int b = startIdx; b < endIdx; b++)
                {
                    var dy = _numOps.Multiply(gradOutput[b * features + f], _gamma[f]);
                    sumDy = _numOps.Add(sumDy, dy);
                    sumDyXhat = _numOps.Add(sumDyXhat, _numOps.Multiply(dy, _normalizedCache[b * features + f]));
                }

                var n = _numOps.FromDouble(virtualSize);

                for (int b = startIdx; b < endIdx; b++)
                {
                    var dy = _numOps.Multiply(gradOutput[b * features + f], _gamma[f]);
                    var xhat = _normalizedCache[b * features + f];

                    // dx = (1/std) * (dy - mean(dy) - xhat * mean(dy * xhat))
                    var dx = _numOps.Multiply(invStd,
                        _numOps.Subtract(dy,
                            _numOps.Add(
                                _numOps.Divide(sumDy, n),
                                _numOps.Multiply(xhat, _numOps.Divide(sumDyXhat, n)))));

                    gradInput[b * features + f] = dx;
                }
            }
        }

        return gradInput;
    }

    /// <summary>
    /// Gets the learnable parameters of this layer.
    /// </summary>
    /// <returns>A vector containing gamma and beta parameters.</returns>
    public Vector<T> GetParameters()
    {
        var parameters = new Vector<T>(_numFeatures * 2);
        for (int i = 0; i < _numFeatures; i++)
        {
            parameters[i] = _gamma[i];
            parameters[_numFeatures + i] = _beta[i];
        }
        return parameters;
    }

    /// <summary>
    /// Sets the learnable parameters of this layer.
    /// </summary>
    /// <param name="parameters">A vector containing gamma and beta parameters.</param>
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != _numFeatures * 2)
        {
            throw new ArgumentException($"Expected {_numFeatures * 2} parameters, got {parameters.Length}");
        }

        for (int i = 0; i < _numFeatures; i++)
        {
            _gamma[i] = parameters[i];
            _beta[i] = parameters[_numFeatures + i];
        }
    }

    /// <summary>
    /// Gets the parameter gradients from the last backward pass.
    /// </summary>
    /// <returns>A vector containing gamma and beta gradients.</returns>
    public Vector<T> GetParameterGradients()
    {
        if (_gammaGrad == null || _betaGrad == null)
        {
            return new Vector<T>(_numFeatures * 2);
        }

        var gradients = new Vector<T>(_numFeatures * 2);
        for (int i = 0; i < _numFeatures; i++)
        {
            gradients[i] = _gammaGrad[i];
            gradients[_numFeatures + i] = _betaGrad[i];
        }
        return gradients;
    }

    /// <summary>
    /// Resets the gradients to zero.
    /// </summary>
    public void ResetGradients()
    {
        _gammaGrad = null;
        _betaGrad = null;
    }

    /// <summary>
    /// Gets the output shape given an input shape.
    /// </summary>
    /// <param name="inputShape">The input shape.</param>
    /// <returns>The output shape (same as input for normalization layers).</returns>
    public int[] GetOutputShape(int[] inputShape)
    {
        return inputShape;
    }

    /// <summary>
    /// Gets the number of trainable parameters in this layer.
    /// </summary>
    public int ParameterCount => _numFeatures * 2;

    /// <summary>
    /// Gets the scale (gamma) parameters.
    /// </summary>
    public Vector<T> Gamma => _gamma;

    /// <summary>
    /// Gets the shift (beta) parameters.
    /// </summary>
    public Vector<T> Beta => _beta;

    /// <summary>
    /// Gets the running mean statistics.
    /// </summary>
    public Vector<T> RunningMean => _runningMean;

    /// <summary>
    /// Gets the running variance statistics.
    /// </summary>
    public Vector<T> RunningVar => _runningVar;
}

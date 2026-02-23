using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Piecewise Linear Encoding for numerical features in tabular models like TabM.
/// </summary>
/// <remarks>
/// <para>
/// Piecewise linear encoding transforms numerical features into a richer representation
/// by computing activations based on learned bin boundaries. Each feature is encoded
/// as a combination of linear pieces, allowing the model to learn non-linear relationships.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of this like creating "bins" for each number:
/// - A feature value of 25 might activate "20-30" bin strongly
/// - It might partially activate neighboring bins too
/// - This gives the model more ways to understand numerical values
///
/// It's similar to how histograms work, but with soft (differentiable) boundaries.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class PiecewiseLinearEncoding<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly Random _random;

    private readonly int _numFeatures;
    private readonly int _numBins;

    // Learnable bin boundaries for each feature
    private Tensor<T> _binBoundaries;
    private Tensor<T> _binBoundaryGradients;

    // Cached values for backward pass
    private Tensor<T>? _inputCache;
    private Tensor<T>? _outputCache;

    /// <summary>
    /// Gets the output dimension (numFeatures * numBins).
    /// </summary>
    public int OutputDimension => _numFeatures * _numBins;

    /// <summary>
    /// Gets the number of parameters.
    /// </summary>
    public int ParameterCount => _numFeatures * (_numBins - 1);

    /// <summary>
    /// Initializes piecewise linear encoding.
    /// </summary>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="numBins">Number of bins per feature.</param>
    public PiecewiseLinearEncoding(int numFeatures, int numBins = 16)
    {
        if (numFeatures < 1)
            throw new ArgumentException("Must have at least 1 feature", nameof(numFeatures));
        if (numBins < 2)
            throw new ArgumentException("Must have at least 2 bins", nameof(numBins));

        _numFeatures = numFeatures;
        _numBins = numBins;
        _random = RandomHelper.CreateSecureRandom();

        // Initialize bin boundaries (numBins - 1 boundaries per feature)
        _binBoundaries = new Tensor<T>([numFeatures, numBins - 1]);
        _binBoundaryGradients = new Tensor<T>([numFeatures, numBins - 1]);

        InitializeBoundaries();
    }

    private void InitializeBoundaries()
    {
        // Initialize boundaries as evenly spaced quantiles
        for (int f = 0; f < _numFeatures; f++)
        {
            for (int b = 0; b < _numBins - 1; b++)
            {
                // Spread boundaries from -2 to 2 (assuming standardized input)
                double boundary = -2.0 + 4.0 * (b + 1) / _numBins;
                _binBoundaries[f * (_numBins - 1) + b] = NumOps.FromDouble(boundary);
            }
        }
    }

    /// <summary>
    /// Encodes numerical features using piecewise linear representation.
    /// </summary>
    /// <param name="input">Input features with shape [batchSize, numFeatures].</param>
    /// <returns>Encoded features with shape [batchSize, numFeatures * numBins].</returns>
    public Tensor<T> Forward(Tensor<T> input)
    {
        _inputCache = input;

        int batchSize = input.Shape[0];
        var output = new Tensor<T>([batchSize, _numFeatures * _numBins]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < _numFeatures; f++)
            {
                var value = input[b * _numFeatures + f];
                EncodeFeature(value, f, output, b);
            }
        }

        _outputCache = output;
        return output;
    }

    private void EncodeFeature(T value, int featureIdx, Tensor<T> output, int batchIdx)
    {
        int outputOffset = batchIdx * _numFeatures * _numBins + featureIdx * _numBins;
        int boundaryOffset = featureIdx * (_numBins - 1);

        // First bin: value - boundary[0] (clamped to [0, 1])
        var firstBoundary = _binBoundaries[boundaryOffset];
        var firstActivation = NumOps.Subtract(value, firstBoundary);
        firstActivation = ClampToUnitInterval(firstActivation);
        output[outputOffset] = firstActivation;

        // Middle bins: min(value - boundary[i-1], boundary[i] - value) (clamped)
        for (int bin = 1; bin < _numBins - 1; bin++)
        {
            var lowerBound = _binBoundaries[boundaryOffset + bin - 1];
            var upperBound = _binBoundaries[boundaryOffset + bin];

            var lowerDiff = NumOps.Subtract(value, lowerBound);
            var upperDiff = NumOps.Subtract(upperBound, value);
            var activation = Min(lowerDiff, upperDiff);
            activation = ClampToUnitInterval(activation);
            output[outputOffset + bin] = activation;
        }

        // Last bin: boundary[last] - value (clamped to [0, 1])
        var lastBoundary = _binBoundaries[boundaryOffset + _numBins - 2];
        var lastActivation = NumOps.Subtract(lastBoundary, value);
        lastActivation = ClampToUnitInterval(lastActivation);
        output[outputOffset + _numBins - 1] = lastActivation;
    }

    private T ClampToUnitInterval(T value)
    {
        if (NumOps.Compare(value, NumOps.Zero) < 0)
            return NumOps.Zero;
        if (NumOps.Compare(value, NumOps.One) > 0)
            return NumOps.One;
        return value;
    }

    private T Min(T a, T b)
    {
        return NumOps.Compare(a, b) < 0 ? a : b;
    }

    /// <summary>
    /// Computes gradients for the backward pass.
    /// </summary>
    /// <param name="gradient">Gradient with respect to output.</param>
    /// <returns>Gradient with respect to input.</returns>
    public Tensor<T> Backward(Tensor<T> gradient)
    {
        if (_inputCache == null)
            throw new InvalidOperationException("Forward must be called before backward");

        int batchSize = gradient.Shape[0];
        var inputGrad = new Tensor<T>([batchSize, _numFeatures]);

        // Reset boundary gradients
        for (int i = 0; i < _binBoundaryGradients.Length; i++)
        {
            _binBoundaryGradients[i] = NumOps.Zero;
        }

        // Compute gradients (simplified - full implementation would track bin assignments)
        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < _numFeatures; f++)
            {
                var grad = NumOps.Zero;
                int gradOffset = b * _numFeatures * _numBins + f * _numBins;

                for (int bin = 0; bin < _numBins; bin++)
                {
                    grad = NumOps.Add(grad, gradient[gradOffset + bin]);
                }

                inputGrad[b * _numFeatures + f] = grad;
            }
        }

        return inputGrad;
    }

    /// <summary>
    /// Updates the bin boundaries.
    /// </summary>
    /// <param name="learningRate">The learning rate.</param>
    public void UpdateParameters(T learningRate)
    {
        for (int i = 0; i < _binBoundaries.Length; i++)
        {
            _binBoundaries[i] = NumOps.Subtract(
                _binBoundaries[i],
                NumOps.Multiply(learningRate, _binBoundaryGradients[i]));
        }
    }

    /// <summary>
    /// Resets internal state.
    /// </summary>
    public void ResetState()
    {
        _inputCache = null;
        _outputCache = null;

        for (int i = 0; i < _binBoundaryGradients.Length; i++)
        {
            _binBoundaryGradients[i] = NumOps.Zero;
        }
    }
}

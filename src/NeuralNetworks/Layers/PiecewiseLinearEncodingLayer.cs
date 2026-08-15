// File-level, deliberately: two Tensors namespaces in the project's global usings also define a
// TensorLayout, so [TensorLayout(...)] only binds when this import shadows them from a nearer scope.
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Extensions;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

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
// Rank 2 only, and that comes from ForwardTraced rather than from the base constructor. The base is
// handed [numFeatures] -> [numFeatures * numBins], but the forward reads `int batchSize = input.Shape[0]`
// and indexes `input[b * _numFeatures + f]`, so the tensor it actually consumes is [batch, features].
// A rank-1 input would be interpreted as numFeatures separate batches of one element - not a case this
// layer handles - so no rank-1 layout is declared.
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class PiecewiseLinearEncodingLayer<T> : LayerBase<T>, IShapeContract
{
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

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Hand-written rather than generated because the feature axis is REPLACED, not carried: the encoding
    /// widens every scalar feature into <c>_numBins</c> activations. Taken straight from
    /// <c>ForwardTraced</c>, which allocates
    /// <c>TensorAllocator.Rent&lt;T&gt;([batchSize, _numFeatures * _numBins])</c>.
    /// </para>
    /// <para>
    /// <c>Fixed</c> rather than <c>Scaled(Features, _numBins, 1)</c> on purpose. The output width does not
    /// depend on the incoming feature count at all - the layer writes exactly
    /// <c>_numFeatures * _numBins</c> values per row using its OWN <c>_numFeatures</c>, so a mismatched
    /// input would be silently mis-encoded rather than produce a proportionally sized output. Fixing the
    /// size to <see cref="OutputDimension"/> reports what the layer actually emits.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank != 2 || _numFeatures <= 0 || _numBins <= 0) return null;

        return new[]
        {
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(OutputDimension)),
        };
    }

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes piecewise linear encoding.
    /// </summary>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="numBins">Number of bins per feature.</param>
    public PiecewiseLinearEncodingLayer(int numFeatures, int numBins = 16)
        : base([numFeatures], [numFeatures * numBins])
    {
        if (numFeatures < 1)
            throw new ArgumentException("Must have at least 1 feature", nameof(numFeatures));
        if (numBins < 2)
            throw new ArgumentException("Must have at least 2 bins", nameof(numBins));

        _numFeatures = numFeatures;
        _numBins = numBins;

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
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        _inputCache = input;

        int batchSize = input.Shape[0];
        var output = TensorAllocator.Rent<T>([batchSize, _numFeatures * _numBins]);

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

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        _binBoundaries = Engine.TensorSubtract(_binBoundaries,
            Engine.TensorMultiplyScalar(_binBoundaryGradients, learningRate));
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _inputCache = null;
        _outputCache = null;

        Engine.TensorFill(_binBoundaryGradients, NumOps.Zero);
    }

}

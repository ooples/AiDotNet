using AiDotNet.Helpers;
using AiDotNet.Autodiff;
using AiDotNet.Attributes;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Interacting Layer for AutoInt architecture.
/// </summary>
/// <remarks>
/// <para>
/// The interacting layer is the core component of AutoInt that learns high-order feature
/// interactions through multi-head self-attention. Each layer captures different orders
/// of interactions between features.
/// </para>
/// <para>
/// <b>For Beginners:</b> The interacting layer helps discover relationships between features:
/// - 1st layer: "age relates to income"
/// - 2nd layer: "age + income together relate to credit score"
/// - 3rd layer: "age + income + credit score relate to loan approval"
///
/// Each layer builds on the previous to capture more complex patterns.
/// The attention mechanism learns which feature combinations are important.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
// AutoInt operates on EMBEDDED TABULAR FIELDS, not a sequence: ForwardTraced reads
// [batchSize, numFeatures, embDim] off input.Shape and attends across the numFeatures axis. That axis
// is the set of table columns, which is why it is Other rather than Time - there is no ordering to it,
// the same reason GraphAttentionLayer names its node axis Other. The trailing axis is the per-field
// embedding width, the layer's Features.
// Shape-preserving on all three axes: attention is over the field axis so it cannot change its length,
// and ProjectOutput ends with `projected.Reshape(batchSize, numFeatures, _embeddingDim)` - the output
// projection maps _attentionDim back DOWN to the embedding width it started at, which is also what lets
// the residual add work when useResidual is set.
// Rank 3 only: the forward pass indexes Shape[0..2] unconditionally, so nothing else is accepted.
// Same rank and same roles both directions, so OutputAxesFor is generated as Same on every axis.
[TensorLayout(TensorAxis.Batch, TensorAxis.Other, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input,
    Note = "Field embeddings: the middle axis is the tabular fields AutoInt attends across.")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Other, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class InteractingLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _embeddingDim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _attentionDim;
    private readonly bool _useResidual;

    // Multi-head attention parameters
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _queryWeights;   // [embeddingDim, attentionDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _keyWeights;     // [embeddingDim, attentionDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _valueWeights;   // [embeddingDim, attentionDim]

    // Output projection (combines heads)
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _outputWeights;  // [attentionDim, embeddingDim]

    // Residual projection (if dimensions don't match)
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T>? _residualWeights;  // [embeddingDim, embeddingDim] if needed

    // Gradients
    [Scratch]
    private Tensor<T> _queryWeightsGrad;
    [Scratch]
    private Tensor<T> _keyWeightsGrad;
    [Scratch]
    private Tensor<T> _valueWeightsGrad;
    [Scratch]
    private Tensor<T> _outputWeightsGrad;
    [Scratch]
    private Tensor<T>? _residualWeightsGrad;

    // Cached values
    [Scratch]
    private Tensor<T>? _attentionScoresCache;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>Construction state: the 'initScale' the layer was built with.</summary>
    private readonly double _initScale;

    /// <summary>
    /// Initializes an interacting layer.
    /// </summary>
    /// <param name="embeddingDim">Input/output embedding dimension.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="attentionDim">Dimension for attention (default: embeddingDim).</param>
    /// <param name="useResidual">Whether to use residual connections.</param>
    /// <param name="initScale">Initialization scale.</param>
    public InteractingLayer(
        int embeddingDim,
        int numHeads = 2,
        int? attentionDim = null,
        bool useResidual = true,
        double initScale = 0.02)
        : base([embeddingDim], [embeddingDim])
    {
        if (embeddingDim <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingDim));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if ((attentionDim ?? embeddingDim) <= 0 || (attentionDim ?? embeddingDim) % numHeads != 0)
            throw new ArgumentException("Attention dimension must be positive and divisible by numHeads.", nameof(attentionDim));
        _initScale = initScale;
        _embeddingDim = embeddingDim;
        _numHeads = numHeads;
        _attentionDim = attentionDim ?? embeddingDim;
        _headDim = _attentionDim / numHeads;
        _useResidual = useResidual;

        // Initialize attention weights
        _queryWeights = new Tensor<T>([embeddingDim, _attentionDim]);
        _keyWeights = new Tensor<T>([embeddingDim, _attentionDim]);
        _valueWeights = new Tensor<T>([embeddingDim, _attentionDim]);
        _outputWeights = new Tensor<T>([_attentionDim, embeddingDim]);

        // Initialize gradients
        _queryWeightsGrad = new Tensor<T>([embeddingDim, _attentionDim]);
        _keyWeightsGrad = new Tensor<T>([embeddingDim, _attentionDim]);
        _valueWeightsGrad = new Tensor<T>([embeddingDim, _attentionDim]);
        _outputWeightsGrad = new Tensor<T>([_attentionDim, embeddingDim]);

        // Initialize residual projection if needed
        if (_useResidual && _attentionDim != embeddingDim)
        {
            _residualWeights = new Tensor<T>([embeddingDim, embeddingDim]);
            _residualWeightsGrad = new Tensor<T>([embeddingDim, embeddingDim]);
        }

        InitializeWeights(initScale);
    }

    private void InitializeWeights(double scale)
    {
        // Xavier/Glorot initialization
        double queryKeyScale = scale / Math.Sqrt(_embeddingDim);
        double outputScale = scale / Math.Sqrt(_attentionDim);

        for (int i = 0; i < _queryWeights.Length; i++)
        {
            _queryWeights[i] = NumOps.FromDouble(Random.NextGaussian() * queryKeyScale);
        }

        for (int i = 0; i < _keyWeights.Length; i++)
        {
            _keyWeights[i] = NumOps.FromDouble(Random.NextGaussian() * queryKeyScale);
        }

        for (int i = 0; i < _valueWeights.Length; i++)
        {
            _valueWeights[i] = NumOps.FromDouble(Random.NextGaussian() * queryKeyScale);
        }

        for (int i = 0; i < _outputWeights.Length; i++)
        {
            _outputWeights[i] = NumOps.FromDouble(Random.NextGaussian() * outputScale);
        }

        if (_residualWeights != null)
        {
            for (int i = 0; i < _residualWeights.Length; i++)
            {
                _residualWeights[i] = NumOps.FromDouble(Random.NextGaussian() * scale);
            }
        }
    }

    /// <summary>
    /// Forward pass through the interacting layer.
    /// </summary>
    /// <param name="input">Input embeddings [batchSize, numFeatures, embeddingDim].</param>
    /// <returns>Feature interactions [batchSize, numFeatures, embeddingDim].</returns>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 3 || input.Shape[2] != _embeddingDim)
            throw new ArgumentException("Expected [batch, fields, embeddingDim] input.", nameof(input));
        int batchSize = input.Shape[0];
        int numFeatures = input.Shape[1];
        var queries = ProjectInput(input, _queryWeights, batchSize, numFeatures, _embeddingDim, _attentionDim);
        var keys = ProjectInput(input, _keyWeights, batchSize, numFeatures, _embeddingDim, _attentionDim);
        var values = ProjectInput(input, _valueWeights, batchSize, numFeatures, _embeddingDim, _attentionDim);

        // Split heads explicitly; scalar copies here disconnect Q/K/V from the gradient tape.
        var headShape = new[] { batchSize, numFeatures, _numHeads, _headDim };
        var q = Engine.TensorPermute(Engine.Reshape(queries, headShape), new[] { 0, 2, 1, 3 });
        var k = Engine.TensorPermute(Engine.Reshape(keys, headShape), new[] { 0, 2, 1, 3 });
        var v = Engine.TensorPermute(Engine.Reshape(values, headShape), new[] { 0, 2, 1, 3 });
        var context = Engine.ScaledDotProductAttention(q, k, v, mask: null,
            scale: 1.0 / Math.Sqrt(_headDim), out var scores);
        // Preserve the public [batch, fields, fields] interpretation by averaging the heads.
        if (ShouldCacheForBackward)
            SaveForBackward(ref _attentionScoresCache, Engine.ReduceMean(scores, new[] { 1 }, keepDims: false));
        else
            _attentionScoresCache = null;
        var attended = Engine.Reshape(Engine.TensorPermute(context, new[] { 0, 2, 1, 3 }),
            new[] { batchSize, numFeatures, _attentionDim });
        var output = ProjectInput(attended, _outputWeights, batchSize, numFeatures, _attentionDim, _embeddingDim);
        if (_useResidual)
            output = AddResidual(input, output, batchSize, numFeatures, _embeddingDim);
        return Engine.ReLU(output);
    }

    private Tensor<T> ProjectInput(Tensor<T> input, Tensor<T> weights,
        int batchSize, int numFeatures, int inputDim, int outputDim)
    {
        var flat = Engine.Reshape(input, new[] { batchSize * numFeatures, inputDim });
        return Engine.Reshape(Engine.TensorMatMul(flat, weights), new[] { batchSize, numFeatures, outputDim });
    }

    private Tensor<T> AddResidual(Tensor<T> input, Tensor<T> output,
        int batchSize, int numFeatures, int embDim)
    {
        if (_residualWeights != null)
        {
            // Project residual if dimensions differ
            var flat = Engine.Reshape(input, new[] { batchSize * numFeatures, embDim });
            var projected = Engine.TensorMatMul(flat, _residualWeights);
            var residual = Engine.Reshape(projected, new[] { batchSize, numFeatures, embDim });
            return Engine.TensorAdd(output, residual);
        }

        // Direct addition
        return Engine.TensorAdd(output, input);
    }

    /// <summary>
    /// Gets attention scores for interpretability.
    /// </summary>
    public Tensor<T>? GetAttentionScores() => _attentionScoresCache;

    /// <summary>
    /// Updates parameters using gradient descent.
    /// </summary>
    public override void UpdateParameters(T learningRate)
    {
        // w = w - lr * grad  =>  Engine.TensorSubtract(w, Engine.TensorMultiplyScalar(grad, lr))
        Engine.TensorSubtractInPlace(_queryWeights, Engine.TensorMultiplyScalar(_queryWeightsGrad, learningRate));
        Engine.TensorSubtractInPlace(_keyWeights, Engine.TensorMultiplyScalar(_keyWeightsGrad, learningRate));
        Engine.TensorSubtractInPlace(_valueWeights, Engine.TensorMultiplyScalar(_valueWeightsGrad, learningRate));
        Engine.TensorSubtractInPlace(_outputWeights, Engine.TensorMultiplyScalar(_outputWeightsGrad, learningRate));

        if (_residualWeights != null && _residualWeightsGrad != null)
        {
            Engine.TensorSubtractInPlace(_residualWeights, Engine.TensorMultiplyScalar(_residualWeightsGrad, learningRate));
        }

        // Register trainable parameters for tape-based autodiff
        RegisterTrainableParameter(_queryWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_keyWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_valueWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputWeights, PersistentTensorRole.Weights);

    }

    private static void CopyTensorToVector(Tensor<T> tensor, Vector<T> vector, ref int offset)
    {
        for (int i = 0; i < tensor.Length; i++)
        {
            vector[offset++] = tensor[i];
        }
    }

    /// <summary>
    /// Resets internal state.
    /// </summary>
    public override void ResetState()
    {
        _attentionScoresCache = null;

        Engine.TensorFill(_queryWeightsGrad, NumOps.Zero);
        Engine.TensorFill(_keyWeightsGrad, NumOps.Zero);
        Engine.TensorFill(_valueWeightsGrad, NumOps.Zero);
        Engine.TensorFill(_outputWeightsGrad, NumOps.Zero);
        if (_residualWeightsGrad != null)
        {
            Engine.TensorFill(_residualWeightsGrad, NumOps.Zero);
        }
    }
}

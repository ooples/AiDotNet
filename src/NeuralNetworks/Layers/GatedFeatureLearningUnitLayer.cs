using AiDotNet.ActivationFunctions;
// File-level, deliberately: two Tensors namespaces in the project's global usings also define a
// TensorLayout, so [TensorLayout(...)] only binds when this import shadows them from a nearer scope.
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Gated Feature Learning Unit (GFLU) for GANDALF architecture.
/// </summary>
/// <remarks>
/// <para>
/// The GFLU is the core building block of GANDALF that performs feature selection
/// and transformation through a gating mechanism. It learns which features are
/// important and how to transform them.
/// </para>
/// <para>
/// <b>For Beginners:</b> GFLU works like a smart filter:
/// 1. Look at all features and decide which ones matter (gating)
/// 2. Transform the selected features
/// 3. Combine them for the next layer
///
/// The "gate" is like a dimmer switch that can turn features on/off or anywhere in between.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
// Both branches are FullyConnectedLayer<T>(outputDim) over the SAME input, and ForwardTraced combines
// them elementwise (`Engine.TensorMultiply(transformed, gate)`), so the output shape is just the
// fully-connected one - the gate cannot change it, only scale it. That fixes the accepted ranks to the
// two the inner layers handle: rank 1, and rank 2 with a leading batch. OnFirstForward's
// `_inputDim = input.Shape[rank - 1]` confirms the LAST axis is the feature axis.
[TensorLayout(TensorAxis.Batch, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class GatedFeatureLearningUnitLayer<T> : LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// HAND-WRITTEN because the output width is configuration: <c>_outputDim</c>, the constructor
    /// argument that both inner projections are built with and that OnFirstForward hands to
    /// <c>ResolveShapes(new[] { _inputDim }, new[] { _outputDim })</c>.
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (_outputDim <= 0) return null;

        var features = new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_outputDim));

        return inputRank switch
        {
            1 => new[] { features },
            2 => new[]
            {
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                features,
            },
            _ => null,
        };
    }

    private int _inputDim;
    private readonly int _outputDim;

    // Feature transformation
    private readonly FullyConnectedLayer<T> _featureTransform;

    // Gating mechanism
    private readonly FullyConnectedLayer<T> _gateTransform;

    // Cached values
    private Tensor<T>? _inputCache;
    private Tensor<T>? _transformedCache;
    private Tensor<T>? _gateCache;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a Gated Feature Learning Unit.
    /// </summary>
    /// <param name="inputDim">Input dimension.</param>
    /// <param name="outputDim">Output dimension.</param>
    public GatedFeatureLearningUnitLayer(int outputDim)
        : base(new[] { -1 }, new[] { outputDim })
    {
        _inputDim = -1;
        _outputDim = outputDim;

        _featureTransform = new FullyConnectedLayer<T>(outputDim, new ReLUActivation<T>() as IActivationFunction<T>);
        _gateTransform = new FullyConnectedLayer<T>(outputDim, (IActivationFunction<T>?)null);
    }

    /// <inheritdoc/>
    protected override void OnFirstForward(Tensor<T> input)
    {
        int rank = input.Shape.Length;
        if (rank < 1)
            throw new ArgumentException(
                $"GatedFeatureLearningUnitLayer requires rank>=1 input; got rank {rank}.", nameof(input));
        _inputDim = input.Shape[rank - 1];
        ResolveShapes(new[] { _inputDim }, new[] { _outputDim });
    }

    /// <summary>
    /// Forward pass through the GFLU.
    /// </summary>
    /// <param name="input">Input tensor [batchSize, inputDim].</param>
    /// <returns>Gated output [batchSize, outputDim].</returns>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        EnsureInitializedFromInput(input);
        _inputCache = input;

        // Transform features
        var transformed = _featureTransform.Forward(input);
        _transformedCache = transformed;

        // Compute gate values with sigmoid
        var gateLogits = _gateTransform.Forward(input);
        var gate = Engine.Sigmoid(gateLogits);
        _gateCache = gate;

        // Apply gate: output = transformed * gate
        return Engine.TensorMultiply(transformed, gate);
    }

    /// <summary>
    /// Gets the current gate values (for interpretability).
    /// </summary>
    public Tensor<T>? GetGateValues() => _gateCache;

    /// <summary>
    /// Gets feature importance based on gate activation magnitudes.
    /// </summary>
    /// <returns>Average gate activation per output dimension.</returns>
    public Vector<T> GetFeatureImportance()
    {
        if (_gateCache == null)
        {
            throw new InvalidOperationException("Forward must be called first");
        }

        int batchSize = _gateCache.Shape[0];
        // Mean across the batch axis: ReduceMean over axis 0 of the [B, outputDim]
        // gate cache. Replaces the per-output-dim scalar accumulation loop with
        // one Engine call.
        var gateCache2D = Engine.Reshape(_gateCache, new[] { batchSize, _outputDim });
        var meanTensor = Engine.ReduceMean(gateCache2D, new[] { 0 }, keepDims: false);
        return meanTensor.ToVector();
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        _featureTransform.UpdateParameters(learningRate);
        _gateTransform.UpdateParameters(learningRate);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _inputCache = null;
        _transformedCache = null;
        _gateCache = null;
        _featureTransform.ResetState();
        _gateTransform.ResetState();
    }

}

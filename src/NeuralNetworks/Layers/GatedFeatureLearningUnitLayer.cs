using AiDotNet.ActivationFunctions;
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
public class GatedFeatureLearningUnitLayer<T> : LayerBase<T>
{
    private readonly int _inputDim;
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

    /// <inheritdoc/>
    public override int ParameterCount => _featureTransform.ParameterCount + _gateTransform.ParameterCount;

    /// <summary>
    /// Initializes a Gated Feature Learning Unit.
    /// </summary>
    /// <param name="inputDim">Input dimension.</param>
    /// <param name="outputDim">Output dimension.</param>
    public GatedFeatureLearningUnitLayer(int inputDim, int outputDim)
        : base([inputDim], [outputDim])
    {
        _inputDim = inputDim;
        _outputDim = outputDim;

        // Feature transformation with ReLU
        _featureTransform = new FullyConnectedLayer<T>(
            inputDim, outputDim, new ReLUActivation<T>() as IActivationFunction<T>);

        // Gate transformation (no activation, sigmoid applied manually)
        _gateTransform = new FullyConnectedLayer<T>(
            inputDim, outputDim, (IActivationFunction<T>?)null);
    }

    /// <summary>
    /// Forward pass through the GFLU.
    /// </summary>
    /// <param name="input">Input tensor [batchSize, inputDim].</param>
    /// <returns>Gated output [batchSize, outputDim].</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
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

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var featureParams = _featureTransform.GetParameters();
        var gateParams = _gateTransform.GetParameters();
        var result = new Vector<T>(featureParams.Length + gateParams.Length);
        int offset = 0;
        for (int i = 0; i < featureParams.Length; i++)
            result[offset++] = featureParams[i];
        for (int i = 0; i < gateParams.Length; i++)
            result[offset++] = gateParams[i];
        return result;
    }
}

using AiDotNet.ActivationFunctions;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

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
public class GatedFeatureLearningUnit<T> : LayerBase<T>
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
    public override bool SupportsJitCompilation => false;

    /// <inheritdoc/>
    public override int ParameterCount => _featureTransform.ParameterCount + _gateTransform.ParameterCount;

    /// <summary>
    /// Initializes a Gated Feature Learning Unit.
    /// </summary>
    /// <param name="inputDim">Input dimension.</param>
    /// <param name="outputDim">Output dimension.</param>
    public GatedFeatureLearningUnit(int inputDim, int outputDim)
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
    /// Backward pass through the GFLU.
    /// </summary>
    /// <param name="gradient">Gradient from upstream [batchSize, outputDim].</param>
    /// <returns>Gradient with respect to input [batchSize, inputDim].</returns>
    public override Tensor<T> Backward(Tensor<T> gradient)
    {
        if (_transformedCache == null || _gateCache == null)
        {
            throw new InvalidOperationException("Forward must be called before backward");
        }

        // Gradient for transformed: dL/dtransformed = dL/dout * gate
        var transformedGrad = Engine.TensorMultiply(gradient, _gateCache);

        // Gradient for gate: dL/dgate = dL/dout * transformed
        var gateGrad = Engine.TensorMultiply(gradient, _transformedCache);

        // Gradient through sigmoid: dL/dlogits = dL/dgate * gate * (1 - gate)
        var ones = Tensor<T>.CreateDefault(_gateCache.Shape, NumOps.One);
        var sigmoidDeriv = Engine.TensorMultiply(_gateCache, Engine.TensorSubtract(ones, _gateCache));
        var gateLogitsGrad = Engine.TensorMultiply(gateGrad, sigmoidDeriv);

        // Backprop through both layers
        var inputGrad1 = _featureTransform.Backward(transformedGrad);
        var inputGrad2 = _gateTransform.Backward(gateLogitsGrad);

        // Sum gradients
        return Engine.TensorAdd(inputGrad1, inputGrad2);
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
        var importance = new Vector<T>(_outputDim);

        for (int d = 0; d < _outputDim; d++)
        {
            var sum = NumOps.Zero;
            for (int b = 0; b < batchSize; b++)
            {
                sum = NumOps.Add(sum, _gateCache[b * _outputDim + d]);
            }
            importance[d] = NumOps.Divide(sum, NumOps.FromDouble(batchSize));
        }

        return importance;
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

    /// <inheritdoc/>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        return inputNode;
    }
}

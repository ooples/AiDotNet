using AiDotNet.ActivationFunctions;
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
public class GatedFeatureLearningUnit<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly Random _random;

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

    /// <summary>
    /// Gets the total parameter count.
    /// </summary>
    public int ParameterCount => _featureTransform.ParameterCount + _gateTransform.ParameterCount;

    /// <summary>
    /// Initializes a Gated Feature Learning Unit.
    /// </summary>
    /// <param name="inputDim">Input dimension.</param>
    /// <param name="outputDim">Output dimension.</param>
    public GatedFeatureLearningUnit(int inputDim, int outputDim)
    {
        _inputDim = inputDim;
        _outputDim = outputDim;
        _random = RandomHelper.CreateSecureRandom();

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
    public Tensor<T> Forward(Tensor<T> input)
    {
        _inputCache = input;

        // Transform features
        var transformed = _featureTransform.Forward(input);
        _transformedCache = transformed;

        // Compute gate values with sigmoid
        var gateLogits = _gateTransform.Forward(input);
        var gate = ApplySigmoid(gateLogits);
        _gateCache = gate;

        // Apply gate: output = transformed * gate
        var output = new Tensor<T>(transformed.Shape);
        for (int i = 0; i < transformed.Length; i++)
        {
            output[i] = NumOps.Multiply(transformed[i], gate[i]);
        }

        return output;
    }

    private Tensor<T> ApplySigmoid(Tensor<T> input)
    {
        var output = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Length; i++)
        {
            var negX = NumOps.Negate(input[i]);
            var expNegX = NumOps.Exp(negX);
            var onePlusExp = NumOps.Add(NumOps.One, expNegX);
            output[i] = NumOps.Divide(NumOps.One, onePlusExp);
        }
        return output;
    }

    /// <summary>
    /// Backward pass through the GFLU.
    /// </summary>
    /// <param name="gradient">Gradient from upstream [batchSize, outputDim].</param>
    /// <returns>Gradient with respect to input [batchSize, inputDim].</returns>
    public Tensor<T> Backward(Tensor<T> gradient)
    {
        if (_transformedCache == null || _gateCache == null)
        {
            throw new InvalidOperationException("Forward must be called before backward");
        }

        // Gradient for transformed: dL/dtransformed = dL/dout * gate
        var transformedGrad = new Tensor<T>(_transformedCache.Shape);
        for (int i = 0; i < gradient.Length; i++)
        {
            transformedGrad[i] = NumOps.Multiply(gradient[i], _gateCache[i]);
        }

        // Gradient for gate: dL/dgate = dL/dout * transformed
        var gateGrad = new Tensor<T>(_gateCache.Shape);
        for (int i = 0; i < gradient.Length; i++)
        {
            gateGrad[i] = NumOps.Multiply(gradient[i], _transformedCache[i]);
        }

        // Gradient through sigmoid: dL/dlogits = dL/dgate * gate * (1 - gate)
        var gateLogitsGrad = new Tensor<T>(_gateCache.Shape);
        for (int i = 0; i < _gateCache.Length; i++)
        {
            var sigmoidDeriv = NumOps.Multiply(_gateCache[i],
                NumOps.Subtract(NumOps.One, _gateCache[i]));
            gateLogitsGrad[i] = NumOps.Multiply(gateGrad[i], sigmoidDeriv);
        }

        // Backprop through both layers
        var inputGrad1 = _featureTransform.Backward(transformedGrad);
        var inputGrad2 = _gateTransform.Backward(gateLogitsGrad);

        // Sum gradients
        var inputGrad = new Tensor<T>(inputGrad1.Shape);
        for (int i = 0; i < inputGrad.Length; i++)
        {
            inputGrad[i] = NumOps.Add(inputGrad1[i], inputGrad2[i]);
        }

        return inputGrad;
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

    /// <summary>
    /// Updates parameters.
    /// </summary>
    public void UpdateParameters(T learningRate)
    {
        _featureTransform.UpdateParameters(learningRate);
        _gateTransform.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Resets internal state.
    /// </summary>
    public void ResetState()
    {
        _inputCache = null;
        _transformedCache = null;
        _gateCache = null;
        _featureTransform.ResetState();
        _gateTransform.ResetState();
    }
}

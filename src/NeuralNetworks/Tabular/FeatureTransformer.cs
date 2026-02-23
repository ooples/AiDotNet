using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Implements the Feature Transformer block used in TabNet architecture.
/// </summary>
/// <remarks>
/// <para>
/// The Feature Transformer processes selected features at each decision step using shared and
/// step-specific layers. It employs a GLU (Gated Linear Unit) mechanism for non-linear transformations
/// combined with Ghost Batch Normalization for regularization.
/// </para>
/// <para>
/// <b>For Beginners:</b> The Feature Transformer is like a smart processor that takes selected
/// features and transforms them into useful representations.
///
/// Key concepts:
/// - **Shared Layers**: Parameters shared across all decision steps (helps learn common patterns)
/// - **Step-Specific Layers**: Parameters unique to each step (learns step-specific patterns)
/// - **GLU (Gated Linear Unit)**: A gating mechanism that controls information flow
/// - **Residual Connections**: Helps with gradient flow during training
///
/// Think of it as a two-part filter:
/// 1. One part decides what information to keep (the "gate")
/// 2. The other part provides the actual information
/// 3. The final output is the product of both
///
/// This architecture allows TabNet to learn complex feature interactions effectively.
/// </para>
/// <para>
/// Reference: "TabNet: Attentive Interpretable Tabular Learning" (Arik &amp; Pfister, AAAI 2021)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FeatureTransformer<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _inputDim;
    private readonly int _outputDim;
    private readonly int _numSharedLayers;
    private readonly int _numStepSpecificLayers;
    private readonly int _virtualBatchSize;
    private readonly double _momentum;
    private readonly double _epsilon;

    // Shared layers (used across all decision steps)
    private readonly List<FullyConnectedLayer<T>> _sharedFCLayers;
    private readonly List<GhostBatchNormalization<T>> _sharedBNLayers;

    // Step-specific layers (unique to each transformer instance)
    private readonly List<FullyConnectedLayer<T>> _stepFCLayers;
    private readonly List<GhostBatchNormalization<T>> _stepBNLayers;

    // Cache for backward pass
    private Tensor<T>? _inputCache;
    private readonly List<Tensor<T>> _intermediateOutputs = [];

    /// <summary>
    /// Gets whether this layer supports training.
    /// </summary>
    public bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the FeatureTransformer class.
    /// </summary>
    /// <param name="inputDim">The dimension of the input features.</param>
    /// <param name="outputDim">The dimension of the output (typically same as inputDim for residual).</param>
    /// <param name="sharedLayers">Existing shared layers to reuse across decision steps (null creates new).</param>
    /// <param name="sharedBNLayers">Existing shared batch norm layers (null creates new).</param>
    /// <param name="numSharedLayers">Number of shared fully connected layers.</param>
    /// <param name="numStepSpecificLayers">Number of step-specific fully connected layers.</param>
    /// <param name="virtualBatchSize">Virtual batch size for Ghost Batch Normalization.</param>
    /// <param name="momentum">Momentum for batch normalization.</param>
    /// <param name="epsilon">Epsilon for numerical stability.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When creating a Feature Transformer:
    /// - inputDim: How many features come in
    /// - outputDim: How many values come out (usually same as input for residual connections)
    /// - sharedLayers: Pass existing shared layers to reuse them across decision steps
    /// - numSharedLayers: How many layers are shared (typically 2)
    /// - numStepSpecificLayers: How many layers are unique to this step (typically 2)
    ///
    /// The shared layers help the model learn common feature patterns across all decision steps,
    /// while step-specific layers allow each step to specialize.
    /// </para>
    /// </remarks>
    public FeatureTransformer(
        int inputDim,
        int outputDim,
        List<FullyConnectedLayer<T>>? sharedLayers = null,
        List<GhostBatchNormalization<T>>? sharedBNLayers = null,
        int numSharedLayers = 2,
        int numStepSpecificLayers = 2,
        int virtualBatchSize = 128,
        double momentum = 0.02,
        double epsilon = 1e-5)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _inputDim = inputDim;
        _outputDim = outputDim;
        _numSharedLayers = numSharedLayers;
        _numStepSpecificLayers = numStepSpecificLayers;
        _virtualBatchSize = virtualBatchSize;
        _momentum = momentum;
        _epsilon = epsilon;

        // Initialize or reuse shared layers
        if (sharedLayers != null && sharedBNLayers != null)
        {
            _sharedFCLayers = sharedLayers;
            _sharedBNLayers = sharedBNLayers;
        }
        else
        {
            _sharedFCLayers = [];
            _sharedBNLayers = [];
            InitializeSharedLayers();
        }

        // Initialize step-specific layers
        _stepFCLayers = [];
        _stepBNLayers = [];
        InitializeStepSpecificLayers();
    }

    /// <summary>
    /// Initializes the shared layers.
    /// </summary>
    private void InitializeSharedLayers()
    {
        int currentDim = _inputDim;
        int hiddenDim = _outputDim * 2; // GLU doubles the dimension, then halves it

        for (int i = 0; i < _numSharedLayers; i++)
        {
            // FC layer with doubled output for GLU (half for value, half for gate)
            var fc = new FullyConnectedLayer<T>(currentDim, hiddenDim, (IActivationFunction<T>?)null);
            _sharedFCLayers.Add(fc);

            // Ghost Batch Normalization
            var bn = new GhostBatchNormalization<T>(hiddenDim, _virtualBatchSize, _momentum, _epsilon);
            _sharedBNLayers.Add(bn);

            currentDim = _outputDim; // After GLU, dimension is halved
        }
    }

    /// <summary>
    /// Initializes the step-specific layers.
    /// </summary>
    private void InitializeStepSpecificLayers()
    {
        int currentDim = _numSharedLayers > 0 ? _outputDim : _inputDim;
        int hiddenDim = _outputDim * 2;

        for (int i = 0; i < _numStepSpecificLayers; i++)
        {
            var fc = new FullyConnectedLayer<T>(currentDim, hiddenDim, (IActivationFunction<T>?)null);
            _stepFCLayers.Add(fc);

            var bn = new GhostBatchNormalization<T>(hiddenDim, _virtualBatchSize, _momentum, _epsilon);
            _stepBNLayers.Add(bn);

            currentDim = _outputDim;
        }
    }

    /// <summary>
    /// Gets the shared fully connected layers for reuse in other FeatureTransformers.
    /// </summary>
    public List<FullyConnectedLayer<T>> SharedFCLayers => _sharedFCLayers;

    /// <summary>
    /// Gets the shared batch normalization layers for reuse.
    /// </summary>
    public List<GhostBatchNormalization<T>> SharedBNLayers => _sharedBNLayers;

    /// <summary>
    /// Applies the GLU (Gated Linear Unit) activation.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, 2 * dim].</param>
    /// <returns>Output tensor of shape [batch, dim].</returns>
    /// <remarks>
    /// <para>
    /// GLU splits the input in half: one half is the value, the other is the gate.
    /// The gate is passed through sigmoid, then multiplied with the value.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> GLU is like a smart filter:
    /// - Split the input into two halves
    /// - One half contains the actual values
    /// - The other half decides how much of each value to keep (the "gate")
    /// - The gate uses sigmoid to produce values between 0 and 1
    /// - Final output = values × gate
    ///
    /// This allows the network to selectively amplify or suppress different features.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplyGLU(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int fullDim = input.Shape[1];
        int halfDim = fullDim / 2;

        var output = new Tensor<T>([batchSize, halfDim]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int d = 0; d < halfDim; d++)
            {
                // First half is the value
                var value = input[b * fullDim + d];
                // Second half is the gate
                var gate = input[b * fullDim + halfDim + d];

                // Sigmoid on gate
                var sigmoidGate = _numOps.Divide(
                    _numOps.One,
                    _numOps.Add(_numOps.One, _numOps.Exp(_numOps.Negate(gate))));

                // GLU: value * sigmoid(gate)
                output[b * halfDim + d] = _numOps.Multiply(value, sigmoidGate);
            }
        }

        return output;
    }

    /// <summary>
    /// Performs the forward pass through the Feature Transformer.
    /// </summary>
    /// <param name="input">The input tensor of shape [batch_size, input_dim].</param>
    /// <returns>The transformed output tensor of shape [batch_size, output_dim].</returns>
    public Tensor<T> Forward(Tensor<T> input)
    {
        _inputCache = input;
        _intermediateOutputs.Clear();

        var current = input;

        // Process through shared layers
        for (int i = 0; i < _numSharedLayers; i++)
        {
            var fcOutput = _sharedFCLayers[i].Forward(current);
            var bnOutput = _sharedBNLayers[i].Forward(fcOutput);
            var gluOutput = ApplyGLU(bnOutput);

            // Residual connection (if dimensions match)
            if (i > 0 && current.Shape[1] == gluOutput.Shape[1])
            {
                gluOutput = AddTensors(gluOutput, current);
                // Scale by sqrt(0.5) for stability
                gluOutput = ScaleTensor(gluOutput, Math.Sqrt(0.5));
            }

            _intermediateOutputs.Add(gluOutput);
            current = gluOutput;
        }

        // Process through step-specific layers
        for (int i = 0; i < _numStepSpecificLayers; i++)
        {
            var fcOutput = _stepFCLayers[i].Forward(current);
            var bnOutput = _stepBNLayers[i].Forward(fcOutput);
            var gluOutput = ApplyGLU(bnOutput);

            // Residual connection
            if (current.Shape[1] == gluOutput.Shape[1])
            {
                gluOutput = AddTensors(gluOutput, current);
                gluOutput = ScaleTensor(gluOutput, Math.Sqrt(0.5));
            }

            _intermediateOutputs.Add(gluOutput);
            current = gluOutput;
        }

        return current;
    }

    /// <summary>
    /// Helper method to add two tensors element-wise.
    /// </summary>
    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        var result = new Tensor<T>(a.Shape);
        for (int i = 0; i < a.Length; i++)
        {
            result[i] = _numOps.Add(a[i], b[i]);
        }
        return result;
    }

    /// <summary>
    /// Helper method to scale a tensor by a scalar value.
    /// </summary>
    private Tensor<T> ScaleTensor(Tensor<T> tensor, double scale)
    {
        var result = new Tensor<T>(tensor.Shape);
        var scaleT = _numOps.FromDouble(scale);
        for (int i = 0; i < tensor.Length; i++)
        {
            result[i] = _numOps.Multiply(tensor[i], scaleT);
        }
        return result;
    }

    /// <summary>
    /// Performs the backward pass through the Feature Transformer.
    /// </summary>
    /// <param name="outputGradient">The gradient flowing back from the next layer.</param>
    /// <returns>The gradient with respect to the input.</returns>
    public Tensor<T> Backward(Tensor<T> outputGradient)
    {
        var currentGrad = outputGradient;

        // Backward through step-specific layers (in reverse order)
        for (int i = _numStepSpecificLayers - 1; i >= 0; i--)
        {
            // TODO: Implement GLU backward and BN backward properly
            // For now, simplified backward pass
            currentGrad = _stepBNLayers[i].Backward(currentGrad);
            currentGrad = _stepFCLayers[i].Backward(currentGrad);
        }

        // Backward through shared layers (in reverse order)
        for (int i = _numSharedLayers - 1; i >= 0; i--)
        {
            currentGrad = _sharedBNLayers[i].Backward(currentGrad);
            currentGrad = _sharedFCLayers[i].Backward(currentGrad);
        }

        return currentGrad;
    }

    /// <summary>
    /// Gets all trainable parameters of this layer.
    /// </summary>
    public Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        // Collect from shared layers
        foreach (var fc in _sharedFCLayers)
        {
            var p = fc.GetParameters();
            for (int i = 0; i < p.Length; i++) allParams.Add(p[i]);
        }
        foreach (var bn in _sharedBNLayers)
        {
            var p = bn.GetParameters();
            for (int i = 0; i < p.Length; i++) allParams.Add(p[i]);
        }

        // Collect from step-specific layers
        foreach (var fc in _stepFCLayers)
        {
            var p = fc.GetParameters();
            for (int i = 0; i < p.Length; i++) allParams.Add(p[i]);
        }
        foreach (var bn in _stepBNLayers)
        {
            var p = bn.GetParameters();
            for (int i = 0; i < p.Length; i++) allParams.Add(p[i]);
        }

        var result = new Vector<T>(allParams.Count);
        for (int i = 0; i < allParams.Count; i++)
        {
            result[i] = allParams[i];
        }
        return result;
    }

    /// <summary>
    /// Sets the trainable parameters of this layer.
    /// </summary>
    public void SetParameters(Vector<T> parameters)
    {
        int offset = 0;

        // Set shared layers
        foreach (var fc in _sharedFCLayers)
        {
            var count = fc.ParameterCount;
            var p = new Vector<T>(count);
            for (int i = 0; i < count; i++) p[i] = parameters[offset + i];
            fc.SetParameters(p);
            offset += count;
        }
        foreach (var bn in _sharedBNLayers)
        {
            var count = bn.ParameterCount;
            var p = new Vector<T>(count);
            for (int i = 0; i < count; i++) p[i] = parameters[offset + i];
            bn.SetParameters(p);
            offset += count;
        }

        // Set step-specific layers
        foreach (var fc in _stepFCLayers)
        {
            var count = fc.ParameterCount;
            var p = new Vector<T>(count);
            for (int i = 0; i < count; i++) p[i] = parameters[offset + i];
            fc.SetParameters(p);
            offset += count;
        }
        foreach (var bn in _stepBNLayers)
        {
            var count = bn.ParameterCount;
            var p = new Vector<T>(count);
            for (int i = 0; i < count; i++) p[i] = parameters[offset + i];
            bn.SetParameters(p);
            offset += count;
        }
    }

    /// <summary>
    /// Gets the parameter gradients from the last backward pass.
    /// </summary>
    public Vector<T> GetParameterGradients()
    {
        var allGrads = new List<T>();

        foreach (var fc in _sharedFCLayers)
        {
            var g = fc.GetParameterGradients();
            for (int i = 0; i < g.Length; i++) allGrads.Add(g[i]);
        }
        foreach (var bn in _sharedBNLayers)
        {
            var g = bn.GetParameterGradients();
            for (int i = 0; i < g.Length; i++) allGrads.Add(g[i]);
        }
        foreach (var fc in _stepFCLayers)
        {
            var g = fc.GetParameterGradients();
            for (int i = 0; i < g.Length; i++) allGrads.Add(g[i]);
        }
        foreach (var bn in _stepBNLayers)
        {
            var g = bn.GetParameterGradients();
            for (int i = 0; i < g.Length; i++) allGrads.Add(g[i]);
        }

        var result = new Vector<T>(allGrads.Count);
        for (int i = 0; i < allGrads.Count; i++)
        {
            result[i] = allGrads[i];
        }
        return result;
    }

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public int ParameterCount
    {
        get
        {
            int count = 0;
            foreach (var fc in _sharedFCLayers) count += fc.ParameterCount;
            foreach (var bn in _sharedBNLayers) count += bn.ParameterCount;
            foreach (var fc in _stepFCLayers) count += fc.ParameterCount;
            foreach (var bn in _stepBNLayers) count += bn.ParameterCount;
            return count;
        }
    }

    /// <summary>
    /// Gets the input shape for this layer.
    /// </summary>
    public int[] GetInputShape() => [_inputDim];

    /// <summary>
    /// Gets the output shape for this layer.
    /// </summary>
    public int[] GetOutputShape() => [_outputDim];

    /// <summary>
    /// Gets the weights tensor (not applicable for this composite layer).
    /// </summary>
    public Tensor<T>? GetWeights() => null;

    /// <summary>
    /// Gets the biases tensor (not applicable for this composite layer).
    /// </summary>
    public Tensor<T>? GetBiases() => null;

    /// <summary>
    /// Updates the parameters using the specified learning rate.
    /// </summary>
    public void UpdateParameters(T learningRate)
    {
        foreach (var fc in _sharedFCLayers) fc.UpdateParameters(learningRate);
        foreach (var fc in _stepFCLayers) fc.UpdateParameters(learningRate);
        // Note: BN layers update running stats during forward, not here
    }

    /// <summary>
    /// Updates the parameters using the specified parameter values.
    /// </summary>
    public void UpdateParameters(Vector<T> parameters)
    {
        SetParameters(parameters);
    }

    /// <summary>
    /// Clears accumulated gradients.
    /// </summary>
    public void ClearGradients()
    {
        foreach (var fc in _sharedFCLayers) fc.ClearGradients();
        foreach (var bn in _sharedBNLayers) bn.ResetGradients();
        foreach (var fc in _stepFCLayers) fc.ClearGradients();
        foreach (var bn in _stepBNLayers) bn.ResetGradients();
    }

    /// <summary>
    /// Resets the internal state.
    /// </summary>
    public void ResetState()
    {
        _inputCache = null;
        _intermediateOutputs.Clear();
        foreach (var fc in _sharedFCLayers) fc.ResetState();
        foreach (var fc in _stepFCLayers) fc.ResetState();
    }

    /// <summary>
    /// Sets training mode.
    /// </summary>
    public void SetTrainingMode(bool isTraining)
    {
        foreach (var fc in _sharedFCLayers) fc.SetTrainingMode(isTraining);
        foreach (var fc in _stepFCLayers) fc.SetTrainingMode(isTraining);
    }

}

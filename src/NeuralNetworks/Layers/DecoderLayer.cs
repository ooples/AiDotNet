#pragma warning disable CS0649, CS0414, CS0169
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a Decoder Layer in a Transformer architecture.
/// </summary>
/// <remarks>
/// <para>
/// The Decoder Layer is a key component in sequence-to-sequence models, particularly in Transformer architectures.
/// It processes the target sequence and incorporates information from the encoder output.
/// </para>
/// <para><b>For Beginners:</b> The Decoder Layer helps in generating output sequences (like translations) 
/// by considering both what it has generated so far and the information from the input sequence.
/// 
/// It's like writing a story where you:
/// 1. Look at what you've written so far (self-attention)
/// 2. Refer back to your source material (cross-attention to encoder output)
/// 3. Think about how to continue the story (feed-forward network)
/// 
/// This process helps in creating coherent and context-aware outputs.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
[LayerCategory(LayerCategory.Transformer)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, Cost = ComputeCost.High, TestInputShape = "1, 4, 4", TestConstructorArgs = "4, 4, 8, (AiDotNet.Interfaces.IActivationFunction<double>?)null")]
public class DecoderLayer<T> : LayerBase<T>
{

    /// <summary>
    /// The self-attention mechanism of the decoder layer.
    /// </summary>
    private readonly AttentionLayer<T> _selfAttention;

    /// <summary>
    /// The cross-attention mechanism that attends to the encoder output.
    /// </summary>
    private readonly AttentionLayer<T> _crossAttention;

    /// <summary>
    /// The feed-forward neural network component of the decoder layer.
    /// </summary>
    private readonly FeedForwardLayer<T> _feedForward1;
    private readonly FeedForwardLayer<T> _feedForward2;

    /// <summary>
    /// Layer normalization applied after self-attention.
    /// </summary>
    private readonly LayerNormalizationLayer<T> _norm1;

    /// <summary>
    /// Layer normalization applied after cross-attention.
    /// </summary>
    private readonly LayerNormalizationLayer<T> _norm2;

    /// <summary>
    /// Layer normalization applied after the feed-forward network.
    /// </summary>
    private readonly LayerNormalizationLayer<T> _norm3;

    /// <summary>
    /// Stores the last input tensor processed by the layer.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the last encoder output tensor used by the layer.
    /// </summary>
    private Tensor<T>? _lastEncoderOutput;

    // GPU cached tensors for backward pass
    private Tensor<T>? _gpuDecoderInput;
    private Tensor<T>? _gpuEncoderOutput;
    private Tensor<T>? _gpuNormalized1;
    private Tensor<T>? _gpuNormalized2;
    private Tensor<T>? _gpuResidual1;
    private Tensor<T>? _gpuResidual2;

    /// <summary>
    /// Gets the size of the input features for this layer.
    /// </summary>
    public int InputSize { get; private set; }

    /// <summary>
    /// Stores the gradient with respect to the input from the last backward pass.
    /// </summary>
    private Tensor<T>? _lastInputGradient;

    /// <summary>
    /// Stores the gradient with respect to the encoder output from the last backward pass.
    /// </summary>
    private Tensor<T>? _lastEncoderOutputGradient;

    /// <summary>
    /// Tracks whether the last input was originally 2D (and thus reshaped to 3D).
    /// </summary>
    private bool _inputWas2D = false;

    /// <summary>
    /// Stores the original input shape for restoring higher-rank tensor output.
    /// </summary>
    private int[]? _originalInputShape;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    public override void SetParameters(Vector<T> parameters)
    {
        int idx = 0;
        void Set(ILayer<T> layer) { int c = layer.ParameterCount; layer.SetParameters(parameters.Slice(idx, c)); idx += c; }
        Set(_selfAttention); Set(_crossAttention); Set(_feedForward1); Set(_feedForward2);
        Set(_norm1); Set(_norm2); Set(_norm3);
    }

    public override Vector<T> GetParameterGradients()
    {
        return Vector<T>.Concatenate(
            _selfAttention.GetParameterGradients(), _crossAttention.GetParameterGradients(),
            _feedForward1.GetParameterGradients(), _feedForward2.GetParameterGradients(),
            _norm1.GetParameterGradients(), _norm2.GetParameterGradients(), _norm3.GetParameterGradients());
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _selfAttention.ClearGradients(); _crossAttention.ClearGradients();
        _feedForward1.ClearGradients(); _feedForward2.ClearGradients();
        _norm1.ClearGradients(); _norm2.ClearGradients(); _norm3.ClearGradients();
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Initializes a new instance of the DecoderLayer class with scalar activation.
    /// </summary>
    /// <param name="inputSize">The size of the input features.</param>
    /// <param name="attentionSize">The size of the attention mechanism.</param>
    /// <param name="feedForwardSize">The size of the feed-forward network.</param>
    /// <param name="activation">The scalar activation function to use. If null, ReLUActivation is used.</param>
    /// <param name="engine">The computation engine for vectorized operations. Defaults to CPU if not specified.</param>
    public DecoderLayer(int inputSize, int attentionSize, int feedForwardSize, IActivationFunction<T>? activation = null)
        : base([inputSize], [inputSize], activation ?? new ReLUActivation<T>())
    {
        _selfAttention = new AttentionLayer<T>(inputSize, attentionSize, (IVectorActivationFunction<T>?)null);
        _crossAttention = new AttentionLayer<T>(inputSize, attentionSize, activation);

        // Standard transformer FFN: Linear(input -> ff) + activation + Linear(ff -> input)
        _feedForward1 = new FeedForwardLayer<T>(inputSize, feedForwardSize, activation);
        _feedForward2 = new FeedForwardLayer<T>(feedForwardSize, inputSize, (IActivationFunction<T>?)null);

        InputSize = inputSize;

        _norm1 = new LayerNormalizationLayer<T>(inputSize);
        _norm2 = new LayerNormalizationLayer<T>(inputSize);
        _norm3 = new LayerNormalizationLayer<T>(inputSize);

        RegisterSubLayer(_selfAttention);
        RegisterSubLayer(_crossAttention);
        RegisterSubLayer(_feedForward1);
        RegisterSubLayer(_feedForward2);
        RegisterSubLayer(_norm1);
        RegisterSubLayer(_norm2);
        RegisterSubLayer(_norm3);
    }

    /// <summary>
    /// Initializes a new instance of the DecoderLayer class with vector activation.
    /// </summary>
    /// <param name="inputSize">The size of the input features.</param>
    /// <param name="attentionSize">The size of the attention mechanism.</param>
    /// <param name="feedForwardSize">The size of the feed-forward network.</param>
    /// <param name="activation">The vector activation function to use. If null, ReLUActivation is used.</param>
    /// <param name="engine">The computation engine for vectorized operations. Defaults to CPU if not specified.</param>
    public DecoderLayer(int inputSize, int attentionSize, int feedForwardSize, IVectorActivationFunction<T>? activation = null)
        : base([inputSize], [inputSize], activation ?? new ReLUActivation<T>())
    {
        _selfAttention = new AttentionLayer<T>(inputSize, attentionSize, (IVectorActivationFunction<T>?)null);
        _crossAttention = new AttentionLayer<T>(inputSize, attentionSize, activation);

        // Standard transformer FFN: Linear(input -> ff) + activation + Linear(ff -> input)
        _feedForward1 = new FeedForwardLayer<T>(inputSize, feedForwardSize, activation);
        _feedForward2 = new FeedForwardLayer<T>(feedForwardSize, inputSize, (IVectorActivationFunction<T>?)null);

        InputSize = inputSize;

        _norm1 = new LayerNormalizationLayer<T>(inputSize);
        _norm2 = new LayerNormalizationLayer<T>(inputSize);
        _norm3 = new LayerNormalizationLayer<T>(inputSize);

        RegisterSubLayer(_selfAttention);
        RegisterSubLayer(_crossAttention);
        RegisterSubLayer(_feedForward1);
        RegisterSubLayer(_feedForward2);
        RegisterSubLayer(_norm1);
        RegisterSubLayer(_norm2);
        RegisterSubLayer(_norm3);
    }

    /// <summary>
    /// Performs the forward pass of the decoder layer.
    /// </summary>
    /// <param name="inputs">An array of input tensors. The first tensor is the decoder input, the second is the encoder output, and the third (optional) is the attention mask.</param>
    /// <returns>The output tensor after processing through the decoder layer.</returns>
    /// <exception cref="ArgumentException">Thrown when the number of input tensors or their shapes are invalid.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method processes the inputs through the decoder layer. It expects
    /// two or three input tensors: the decoder's own input, the encoder's output, and optionally an attention mask.
    /// The method combines these inputs, processes them through the layer, and returns the final output.
    /// The attention mask, if provided, helps control which parts of the input sequence the layer should focus on.</para>
    /// </remarks>
    public override Tensor<T> Forward(params Tensor<T>[] inputs)
    {
        if (inputs.Length < 2 || inputs.Length > 3)
            throw new ArgumentException("DecoderLayer requires two or three input tensors: decoder input, encoder output, and optionally an attention mask.");

        var decoderInput = inputs[0];
        var encoderOutput = inputs[1];
        Tensor<T>? attentionMask = inputs.Length == 3 ? inputs[2] : null;

        // Handle any rank >= 2: last 2 dims are [seq, features], earlier dims are batch-like
        int rank = decoderInput.Shape.Length;
        _inputWas2D = rank == 2;
        _originalInputShape = decoderInput.Shape.ToArray();
        Tensor<T> input3D, encoderOutput3D;

        if (_inputWas2D)
        {
            // 2D input: [batch, features] -> [batch, 1, features]
            if (decoderInput.Shape[1] != InputSize)
                throw new ArgumentException($"Decoder input dimension {decoderInput.Shape[1]} does not match expected InputSize {InputSize}.");
            if (encoderOutput.Shape[encoderOutput.Shape.Length - 1] != InputSize)
                throw new ArgumentException($"Encoder output last dimension does not match expected InputSize {InputSize}.");

            int batchSize = decoderInput.Shape[0];
            input3D = decoderInput.Reshape(batchSize, 1, InputSize);

            // Handle encoder output reshaping
            int encRank = encoderOutput.Shape.Length;
            if (encRank == 2)
                encoderOutput3D = encoderOutput.Reshape(encoderOutput.Shape[0], 1, InputSize);
            else if (encRank == 3)
                encoderOutput3D = encoderOutput;
            else
            {
                int encFlatBatch = 1;
                for (int d = 0; d < encRank - 2; d++)
                    encFlatBatch *= encoderOutput.Shape[d];
                encoderOutput3D = encoderOutput.Reshape(encFlatBatch, encoderOutput.Shape[encRank - 2], encoderOutput.Shape[encRank - 1]);
            }
        }
        else if (rank == 3)
        {
            // 3D input: standard format
            if (decoderInput.Shape[2] != InputSize)
                throw new ArgumentException($"Decoder input dimension {decoderInput.Shape[2]} does not match expected InputSize {InputSize}.");

            input3D = decoderInput;

            // Handle encoder output
            int encRank = encoderOutput.Shape.Length;
            if (encRank == 3)
                encoderOutput3D = encoderOutput;
            else
            {
                int encFlatBatch = 1;
                for (int d = 0; d < encRank - 2; d++)
                    encFlatBatch *= encoderOutput.Shape[d];
                encoderOutput3D = encoderOutput.Reshape(encFlatBatch, encoderOutput.Shape[encRank - 2], encoderOutput.Shape[encRank - 1]);
            }
        }
        else
        {
            // Higher rank: flatten leading dimensions
            if (decoderInput.Shape[rank - 1] != InputSize)
                throw new ArgumentException($"Decoder input last dimension {decoderInput.Shape[rank - 1]} does not match expected InputSize {InputSize}.");

            int flatBatch = 1;
            for (int d = 0; d < rank - 2; d++)
                flatBatch *= decoderInput.Shape[d];
            input3D = decoderInput.Reshape(flatBatch, decoderInput.Shape[rank - 2], decoderInput.Shape[rank - 1]);

            // Handle encoder output
            int encRank = encoderOutput.Shape.Length;
            int encFlatBatch = 1;
            for (int d = 0; d < encRank - 2; d++)
                encFlatBatch *= encoderOutput.Shape[d];
            encoderOutput3D = encoderOutput.Reshape(encFlatBatch, encoderOutput.Shape[encRank - 2], encoderOutput.Shape[encRank - 1]);
        }

        return ForwardInternal(input3D, encoderOutput3D, attentionMask);
    }

    /// <summary>
    /// Performs the internal forward pass of the decoder layer.
    /// </summary>
    /// <param name="input">The decoder input tensor.</param>
    /// <param name="encoderOutput">The encoder output tensor.</param>
    /// <param name="attentionMask">Optional attention mask tensor.</param>
    /// <returns>The output tensor after processing through the decoder layer.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method represents the core processing of the decoder layer.
    /// It applies self-attention, cross-attention with the encoder output, and a feed-forward network,
    /// with layer normalization applied between each step. The attention mask, if provided, helps
    /// control which parts of the input sequence the layer should focus on.</para>
    /// </remarks>
    private Tensor<T> ForwardInternal(Tensor<T> input, Tensor<T> encoderOutput, Tensor<T>? attentionMask = null)
    {
        _lastInput = input;
        _lastEncoderOutput = encoderOutput;

        // Get dimensions from input and encoder output
        int batchSize = input.Shape[0];
        int decoderSeqLen = input.Shape[1];
        int encoderSeqLen = encoderOutput.Shape[1];

        // Self-attention mask: [batchSize, decoder_seqLen, decoder_seqLen]
        // Each decoder position can attend to all decoder positions (or causal subset)
        Tensor<T> selfAttnMask = attentionMask ?? Tensor<T>.CreateDefault([batchSize, decoderSeqLen, decoderSeqLen], NumOps.Zero);

        // Cross-attention mask: [batchSize, decoder_seqLen, encoder_seqLen]
        // Each decoder position attends to encoder positions
        Tensor<T> crossAttnMask = Tensor<T>.CreateDefault([batchSize, decoderSeqLen, encoderSeqLen], NumOps.Zero);

        // Self-attention (decoder attending to itself)
        var selfAttentionOutput = _selfAttention.Forward(input, selfAttnMask);
        var normalized1 = _norm1.Forward(input.Add(selfAttentionOutput));

        // Cross-attention (decoder attending to encoder output)
        var crossAttentionOutput = _crossAttention.Forward(normalized1, encoderOutput, crossAttnMask);
        var normalized2 = _norm2.Forward(normalized1.Add(crossAttentionOutput));

        // Standard transformer FFN: Linear(input -> ff) + activation + Linear(ff -> input)
        var ffExpanded = _feedForward1.Forward(normalized2);
        var feedForwardOutput = _feedForward2.Forward(ffExpanded);
        var output = _norm3.Forward(normalized2.Add(feedForwardOutput));

        // Restore original tensor shape
        if (_inputWas2D)
        {
            output = output.Reshape(batchSize, output.Shape[2]);
        }
        else if (_originalInputShape != null && _originalInputShape.Length > 3)
        {
            // Restore original batch dimensions for higher-rank input
            var outputShape = new int[_originalInputShape.Length];
            for (int d = 0; d < _originalInputShape.Length - 2; d++)
                outputShape[d] = _originalInputShape[d];
            outputShape[_originalInputShape.Length - 2] = output.Shape[1];
            outputShape[_originalInputShape.Length - 1] = output.Shape[2];
            output = output.Reshape(outputShape);
        }

        return output;
    }

    /// <summary>
    /// Performs the GPU-resident forward pass of the decoder layer.
    /// </summary>
    /// <param name="inputs">The GPU input tensors: [decoderInput, encoderOutput].</param>
    /// <returns>The GPU output tensor after self-attention, cross-attention, and FFN.</returns>
    /// <remarks>
    /// All computations stay on GPU. Chains: SelfAttention → Norm1 → CrossAttention → Norm2 → FFN → Norm3.
    /// </remarks>
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length < 2)
            throw new ArgumentException("DecoderLayer requires two inputs: [decoderInput, encoderOutput]", nameof(inputs));

        var gpuEngine = Engine as DirectGpuTensorEngine;
        if (gpuEngine == null)
            throw new InvalidOperationException("ForwardGpu requires a DirectGpuTensorEngine.");

        var decoderInput = inputs[0];
        var encoderOutput = inputs[1];

        // 1. Self-attention: decoder attends to itself
        var selfAttentionOutput = _selfAttention.ForwardGpu(decoderInput);

        // 2. First residual connection + layer norm
        var residual1 = gpuEngine.AddGpu(decoderInput, selfAttentionOutput);
        var normalized1 = _norm1.ForwardGpu(residual1);

        // 3. Cross-attention: decoder attends to encoder output
        var crossAttentionOutput = _crossAttention.ForwardGpu(normalized1, encoderOutput);

        // 4. Second residual connection + layer norm
        var residual2 = gpuEngine.AddGpu(normalized1, crossAttentionOutput);
        var normalized2 = _norm2.ForwardGpu(residual2);

        // 5. Feed-forward network: two linear layers
        var ffHidden = _feedForward1.ForwardGpu(normalized2);
        var ffOutput = _feedForward2.ForwardGpu(ffHidden);

        // 6. Third residual connection + layer norm
        var residual3 = gpuEngine.AddGpu(normalized2, ffOutput);
        var output = _norm3.ForwardGpu(residual3);

        // Cache state for backward pass only during training
        if (IsTrainingMode)
        {
            _gpuDecoderInput = decoderInput;
            _gpuEncoderOutput = encoderOutput;
            _gpuNormalized1 = normalized1;
            _gpuNormalized2 = normalized2;
            _gpuResidual1 = residual1;
            _gpuResidual2 = residual2;
            _lastInput = decoderInput;
            _lastEncoderOutput = encoderOutput;
        }

        return output;
    }

    // LastBackwardGradients removed — tape-based autodiff handles gradients.

    // BackwardInternal removed — tape-based autodiff handles decoder gradients.

    /// <summary>
    /// Updates the layer's parameters based on the computed gradients and a learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the update.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method applies the calculated gradients to the layer's parameters,
    /// effectively "learning" from the training data. The learning rate determines how big of a step
    /// we take in the direction suggested by the gradients.</para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        _selfAttention.UpdateParameters(learningRate);
        _crossAttention.UpdateParameters(learningRate);
        _feedForward1.UpdateParameters(learningRate);
        _feedForward2.UpdateParameters(learningRate);
        _norm1.UpdateParameters(learningRate);
        _norm2.UpdateParameters(learningRate);
        _norm3.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Updates layer parameters using GPU-resident optimizer.
    /// </summary>
    /// <param name="config">The GPU optimizer configuration.</param>
    /// <remarks>
    /// <para>
    /// This method delegates to each sublayer's UpdateParametersGpu method.
    /// All sublayers (self-attention, cross-attention, layer norms, feed-forward) are updated.
    /// </para>
    /// </remarks>
    public override void UpdateParametersGpu(IGpuOptimizerConfig config)
    {
        // Update parameters for each sub-layer using GPU optimizer
        _selfAttention.UpdateParametersGpu(config);
        _crossAttention.UpdateParametersGpu(config);
        _feedForward1.UpdateParametersGpu(config);
        _feedForward2.UpdateParametersGpu(config);
        _norm1.UpdateParametersGpu(config);
        _norm2.UpdateParametersGpu(config);
        _norm3.UpdateParametersGpu(config);
    }

    /// <summary>
    /// Retrieves the current parameters of the layer.
    /// </summary>
    /// <returns>A vector containing all the parameters of the layer.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method collects all the parameters from the various components
    /// of the decoder layer (self-attention, cross-attention, feed-forward network, and layer normalizations)
    /// and combines them into a single vector. This is useful for operations that need to work with all
    /// parameters at once, such as optimization algorithms.</para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Use Vector<T>.Concatenate for efficient parameter collection
        var selfAttnParams = _selfAttention.GetParameters();
        var crossAttnParams = _crossAttention.GetParameters();
        var ff1Params = _feedForward1.GetParameters();
        var ff2Params = _feedForward2.GetParameters();
        var norm1Params = _norm1.GetParameters();
        var norm2Params = _norm2.GetParameters();
        var norm3Params = _norm3.GetParameters();

        return Vector<T>.Concatenate(
            Vector<T>.Concatenate(
                Vector<T>.Concatenate(
                    Vector<T>.Concatenate(selfAttnParams, crossAttnParams),
                    Vector<T>.Concatenate(ff1Params, ff2Params)),
                norm1Params),
            Vector<T>.Concatenate(norm2Params, norm3Params));
    }

    /// <summary>
    /// Updates the layer's parameters with the provided values.
    /// </summary>
    /// <param name="parameters">A vector containing new parameter values.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method takes a vector of new parameter values and distributes
    /// them to the various components of the decoder layer. It's the opposite of GetParameters() and is
    /// typically used after an optimization step to update the layer with improved parameter values.</para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int index = 0;
        index = UpdateComponentParameters(_selfAttention, parameters, index);
        index = UpdateComponentParameters(_crossAttention, parameters, index);
        index = UpdateComponentParameters(_feedForward1, parameters, index);
        index = UpdateComponentParameters(_feedForward2, parameters, index);
        index = UpdateComponentParameters(_norm1, parameters, index);
        index = UpdateComponentParameters(_norm2, parameters, index);
        UpdateComponentParameters(_norm3, parameters, index);
    }

    /// <summary>
    /// Updates the parameters of a specific component within the decoder layer.
    /// </summary>
    /// <param name="component">The component to update.</param>
    /// <param name="parameters">The full vector of parameters for the entire layer.</param>
    /// <param name="startIndex">The starting index in the parameters vector for this component.</param>
    /// <returns>The next starting index after updating this component's parameters.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This helper method is used by UpdateParameters to update each
    /// individual component of the decoder layer. It extracts the relevant portion of the parameter
    /// vector for a specific component and applies those parameters to that component.</para>
    /// </remarks>
    private int UpdateComponentParameters(LayerBase<T> component, Vector<T> parameters, int startIndex)
    {
        int paramCount = component.ParameterCount;

        // Use Engine.TensorSlice to extract component parameters without manual loops
        var paramsTensor = new Tensor<T>([parameters.Length], parameters);
        var componentParamsTensor = Engine.TensorSlice(paramsTensor, [startIndex], [paramCount]);
        var componentParams = new Vector<T>(componentParamsTensor.ToArray());

        component.UpdateParameters(componentParams);

        return startIndex + paramCount;
    }

    /// <summary>
    /// Resets the state of the decoder layer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method clears any stored state in the decoder layer and its components.
    /// It's typically called between processing different sequences or at the start of a new epoch in training.
    /// Resetting the state ensures that information from previous inputs doesn't affect the processing of new, unrelated inputs.</para>
    /// </remarks>
    public override void ResetState()
    {
        _lastInput = null;
        _lastEncoderOutput = null;
        _gpuDecoderInput = null;
        _gpuEncoderOutput = null;
        _gpuNormalized1 = null;
        _gpuNormalized2 = null;
        _gpuResidual1 = null;
        _gpuResidual2 = null;
        _selfAttention.ResetState();
        _crossAttention.ResetState();
        _feedForward1.ResetState();
        _feedForward2.ResetState();
        _norm1.ResetState();
        _norm2.ResetState();
        _norm3.ResetState();
    }

    /// <summary>
    /// Single-input forward pass uses the input as both decoder input and encoder output.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> When only one input is provided, the decoder attends to itself by reusing the input
    /// for both decoder and encoder streams. Use the overload that accepts multiple tensors to supply
    /// a separate encoder output when available.</para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        return Forward(input, input);
    }


    /// <summary>
    /// Gets the total number of trainable parameters in the layer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This property calculates and returns the total number of parameters
    /// in the decoder layer by summing the parameter counts of all its components. This is useful for
    /// understanding the complexity of the layer and for certain optimization techniques.</para>
    /// </remarks>
    public override int ParameterCount =>
        _selfAttention.ParameterCount +
        _crossAttention.ParameterCount +
        _feedForward1.ParameterCount + _feedForward2.ParameterCount +
        _norm1.ParameterCount +
        _norm2.ParameterCount +
        _norm3.ParameterCount;

}

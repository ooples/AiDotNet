using AiDotNet.ActivationFunctions;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements a Dense Block from the DenseNet architecture.
/// </summary>
/// <remarks>
/// <para>
/// A Dense Block is the core building block of DenseNet. It contains multiple layers where
/// each layer receives feature maps from ALL preceding layers (dense connectivity).
/// This creates strong gradient flow and feature reuse throughout the network.
/// </para>
/// <para>
/// Architecture of a Dense Block with n layers:
/// <code>
/// Input (k0 channels)
///   ↓
/// Layer 1: BN → ReLU → Conv1x1 → BN → ReLU → Conv3x3 → Output1 (k channels)
///   ↓ concat
/// [Input, Output1] (k0 + k channels)
///   ↓
/// Layer 2: BN → ReLU → Conv1x1 → BN → ReLU → Conv3x3 → Output2 (k channels)
///   ↓ concat
/// [Input, Output1, Output2] (k0 + 2k channels)
///   ↓
/// ... (continues for n layers)
///   ↓
/// Final: [Input, Output1, ..., OutputN] (k0 + n*k channels)
/// </code>
/// Where k is the growth rate (number of channels added per layer).
/// </para>
/// <para>
/// <b>For Beginners:</b> Dense connectivity means each layer can directly access
/// features from all previous layers, promoting feature reuse and reducing
/// the need for redundant feature learning.
///
/// Key benefits:
/// - Strong gradient flow (helps with training very deep networks)
/// - Feature reuse (each layer can use features from all previous layers)
/// - Fewer parameters (layers can be narrow since they share features)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DenseBlock<T> : LayerBase<T>
{
    private readonly List<DenseBlockLayer<T>> _layers;
    private readonly int _numLayers;
    private readonly int _growthRate;
    private readonly int _inputChannels;
    private List<Tensor<T>>? _layerOutputs;

    // GPU cached tensors for backward pass
    private List<IGpuTensor<T>>? _gpuFeatureMaps;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer has a GPU implementation.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Gets the number of layers in this dense block.
    /// </summary>
    public int NumLayers => _numLayers;

    /// <summary>
    /// Gets the growth rate (channels added per layer).
    /// </summary>
    public int GrowthRate => _growthRate;

    /// <summary>
    /// Gets the number of output channels (inputChannels + numLayers * growthRate).
    /// </summary>
    public int OutputChannels => _inputChannels + _numLayers * _growthRate;

    /// <summary>
    /// Initializes a new instance of the <see cref="DenseBlock{T}"/> class.
    /// </summary>
    /// <param name="inputChannels">The number of input channels.</param>
    /// <param name="numLayers">The number of layers in the dense block.</param>
    /// <param name="growthRate">The number of channels each layer adds (k in the paper).</param>
    /// <param name="inputHeight">The input feature map height.</param>
    /// <param name="inputWidth">The input feature map width.</param>
    /// <param name="bnMomentum">Batch normalization momentum (default: 0.1).</param>
    public DenseBlock(
        int inputChannels,
        int numLayers,
        int growthRate,
        int inputHeight,
        int inputWidth,
        double bnMomentum = 0.1)
        : base(
            inputShape: [inputChannels, inputHeight, inputWidth],
            outputShape: [inputChannels + numLayers * growthRate, inputHeight, inputWidth])
    {
        _inputChannels = inputChannels;
        _numLayers = numLayers;
        _growthRate = growthRate;
        _layers = new List<DenseBlockLayer<T>>(numLayers);

        // Create layers with increasing input channels
        int currentChannels = inputChannels;
        for (int i = 0; i < numLayers; i++)
        {
            _layers.Add(new DenseBlockLayer<T>(
                currentChannels, growthRate, inputHeight, inputWidth, bnMomentum));
            currentChannels += growthRate; // Each layer adds growthRate channels
        }
    }

    /// <summary>
    /// Performs the forward pass of the Dense Block.
    /// </summary>
    /// <param name="input">The input tensor [B, C, H, W].</param>
    /// <returns>The output tensor with all layer outputs concatenated.</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _layerOutputs = new List<Tensor<T>>(_numLayers + 1) { input };

        // Current feature maps (accumulated)
        var currentFeatures = input;

        foreach (var layer in _layers)
        {
            // Each layer takes ALL previous features as input
            var layerOutput = layer.Forward(currentFeatures);
            _layerOutputs.Add(layerOutput);

            // Concatenate new features with existing features along channel dimension
            currentFeatures = ConcatenateChannels(currentFeatures, layerOutput);
        }

        return currentFeatures;
    }

    /// <summary>
    /// Performs the forward pass on GPU, keeping data GPU-resident.
    /// </summary>
    /// <param name="inputs">The input tensors (expects single input).</param>
    /// <returns>The output tensor on GPU.</returns>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires a DirectGpuTensorEngine.");

        var currentFeatures = inputs[0];

        // Cache feature maps for backward pass during training
        if (IsTrainingMode)
        {
            _gpuFeatureMaps = new List<IGpuTensor<T>>(_numLayers + 1) { currentFeatures };
        }

        foreach (var layer in _layers)
        {
            // Each layer takes ALL previous features as input
            var layerOutput = layer.ForwardGpu(currentFeatures);

            // Concatenate new features with existing features along channel dimension (axis 1)
            currentFeatures = gpuEngine.ConcatGpu(new[] { currentFeatures, layerOutput }, 1);

            // Cache for backward pass
            if (IsTrainingMode)
            {
                _gpuFeatureMaps!.Add(currentFeatures);
            }
        }

        return currentFeatures;
    }

    /// <summary>
    /// Performs the backward pass of the Dense Block.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the output.</param>
    /// <returns>The gradient of the loss with respect to the input.</returns>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_layerOutputs is null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Start with the full gradient (for all concatenated features)
        var currentGrad = outputGradient;

        // Process layers in reverse order
        for (int i = _numLayers - 1; i >= 0; i--)
        {
            var layer = _layers[i];

            // Calculate input channels to this layer
            int inputChannelsToLayer = _inputChannels + i * _growthRate;

            // Split gradient: [grad for previous layers, grad for this layer's output]
            var (prevGrad, layerGrad) = SplitGradient(currentGrad, inputChannelsToLayer, _growthRate);

            // Backward through this layer
            var layerInputGrad = layer.Backward(layerGrad);

            // Accumulate gradients (add to previous gradient)
            currentGrad = AddGradients(prevGrad, layerInputGrad);
        }

        // The remaining gradient is for the original input
        return currentGrad;
    }

    /// <summary>
    /// Performs GPU-accelerated backward pass for the Dense Block.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Processes layers in reverse order, splitting gradients along channel dimension
    /// and accumulating gradients through dense connections.
    /// </para>
    /// </remarks>
    /// <param name="outputGradient">GPU tensor containing gradient of loss with respect to output.</param>
    /// <returns>GPU tensor containing gradient with respect to input.</returns>
    public override IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        if (_gpuFeatureMaps == null)
            throw new InvalidOperationException("ForwardGpu must be called before BackwardGpu.");

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("BackwardGpu requires DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend == null)
            throw new InvalidOperationException("GPU backend unavailable.");

        // Start with the full gradient (for all concatenated features)
        var currentGrad = outputGradient;

        // Process layers in reverse order
        for (int i = _numLayers - 1; i >= 0; i--)
        {
            var layer = _layers[i];

            // Calculate input channels to this layer
            int inputChannelsToLayer = _inputChannels + i * _growthRate;

            // Get batch and spatial dimensions from gradient
            int batch = currentGrad.Shape[0];
            int height = currentGrad.Shape.Length > 2 ? currentGrad.Shape[2] : 1;
            int width = currentGrad.Shape.Length > 3 ? currentGrad.Shape[3] : 1;

            // Split gradient: [grad for previous layers, grad for this layer's output]
            // Use SliceGpu along channel dimension (axis 1)
            var prevGrad = gpuEngine.SliceGpu(currentGrad, 1, 0, inputChannelsToLayer);
            var layerGrad = gpuEngine.SliceGpu(currentGrad, 1, inputChannelsToLayer, inputChannelsToLayer + _growthRate);

            // Backward through this layer
            var layerInputGrad = layer.BackwardGpu(layerGrad);

            // Accumulate gradients (add to previous gradient)
            currentGrad = gpuEngine.AddGpu(prevGrad, layerInputGrad);
        }

        // The remaining gradient is for the original input
        return currentGrad;
    }

    /// <summary>
    /// Updates the parameters of all sub-layers.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    public override void UpdateParameters(T learningRate)
    {
        foreach (var layer in _layers)
        {
            layer.UpdateParameters(learningRate);
        }
    }

    /// <summary>
    /// Gets all trainable parameters from the block.
    /// </summary>
    public override Vector<T> GetParameters()
    {
        return new Vector<T>(_layers.SelectMany(l => l.GetParameters().ToArray()).ToArray());
    }

    /// <summary>
    /// Sets all trainable parameters from the given parameter vector.
    /// </summary>
    /// <param name="parameters">The parameter vector containing all layer parameters.</param>
    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var layer in _layers)
        {
            int count = layer.GetParameters().Length;
            layer.SetParameters(parameters.SubVector(offset, count));
            offset += count;
        }
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["InputChannels"] = _inputChannels.ToString();
        metadata["NumLayers"] = _numLayers.ToString();
        metadata["GrowthRate"] = _growthRate.ToString();
        return metadata;
    }

    /// <summary>
    /// Resets the internal state of the block.
    /// </summary>
    public override void ResetState()
    {
        _layerOutputs = null;
        _gpuFeatureMaps = null;
        foreach (var layer in _layers)
        {
            layer.ResetState();
        }
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    public override bool SupportsJitCompilation
    {
        get
        {
            // Check all sub-layers support JIT
            foreach (var layer in _layers)
            {
                if (!layer.SupportsJitCompilation)
                    return false;
            }
            return true;
        }
    }

    /// <summary>
    /// Exports the computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node representing the DenseBlock.</returns>
    /// <remarks>
    /// <para>
    /// This method builds a computation graph representing the DenseBlock with dense connectivity:
    /// Each layer's output is concatenated with all previous features along the channel dimension.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes is null)
        {
            throw new ArgumentNullException(nameof(inputNodes));
        }

        if (InputShape is null || InputShape.Length == 0)
        {
            throw new InvalidOperationException("Layer input shape not configured.");
        }

        // Create symbolic input node with batch dimension
        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        // Current features (accumulated via concatenation)
        var currentFeatures = inputNode;

        // Process each layer with dense connectivity
        for (int i = 0; i < _layers.Count; i++)
        {
            var layer = _layers[i];

            // Build the layer's computation graph using the current accumulated features
            var layerOutput = layer.BuildComputationGraph(currentFeatures, $"layer{i}_");

            // Concatenate new features with existing features along channel dimension (axis 1)
            var nodesToConcat = new List<ComputationNode<T>> { currentFeatures, layerOutput };
            currentFeatures = TensorOperations<T>.Concat(nodesToConcat, axis: 1);
        }

        return currentFeatures;
    }

    #region Helper Methods

    /// <summary>
    /// Concatenates two tensors along the channel dimension (dim=0 for CHW, dim=1 for NCHW).
    /// </summary>
    private Tensor<T> ConcatenateChannels(Tensor<T> a, Tensor<T> b)
    {
        // Handle 3D tensors (CHW format)
        if (a.Shape.Length == 3)
        {
            int channelsA = a.Shape[0];
            int channelsB = b.Shape[0];
            int height = a.Shape[1];
            int width = a.Shape[2];

            int totalChannels = channelsA + channelsB;
            var result = new Tensor<T>([totalChannels, height, width]);

            int spatialSize = height * width;

            // Copy first tensor
            for (int c = 0; c < channelsA; c++)
            {
                for (int hw = 0; hw < spatialSize; hw++)
                {
                    int srcIdx = c * spatialSize + hw;
                    int dstIdx = c * spatialSize + hw;
                    result.Data[dstIdx] = a.Data[srcIdx];
                }
            }

            // Copy second tensor
            for (int c = 0; c < channelsB; c++)
            {
                for (int hw = 0; hw < spatialSize; hw++)
                {
                    int srcIdx = c * spatialSize + hw;
                    int dstIdx = (channelsA + c) * spatialSize + hw;
                    result.Data[dstIdx] = b.Data[srcIdx];
                }
            }

            return result;
        }

        // Handle 4D tensors (NCHW format)
        int batch = a.Shape[0];
        int channelsA4D = a.Shape[1];
        int channelsB4D = b.Shape[1];
        int height4D = a.Shape[2];
        int width4D = a.Shape[3];

        int totalChannels4D = channelsA4D + channelsB4D;
        var result4D = new Tensor<T>([batch, totalChannels4D, height4D, width4D]);

        // Copy first tensor
        int spatialSize4D = height4D * width4D;
        for (int n = 0; n < batch; n++)
        {
            for (int c = 0; c < channelsA4D; c++)
            {
                for (int hw = 0; hw < spatialSize4D; hw++)
                {
                    int srcIdx = n * (channelsA4D * spatialSize4D) + c * spatialSize4D + hw;
                    int dstIdx = n * (totalChannels4D * spatialSize4D) + c * spatialSize4D + hw;
                    result4D.Data[dstIdx] = a.Data[srcIdx];
                }
            }

            // Copy second tensor
            for (int c = 0; c < channelsB4D; c++)
            {
                for (int hw = 0; hw < spatialSize4D; hw++)
                {
                    int srcIdx = n * (channelsB4D * spatialSize4D) + c * spatialSize4D + hw;
                    int dstIdx = n * (totalChannels4D * spatialSize4D) + (channelsA4D + c) * spatialSize4D + hw;
                    result4D.Data[dstIdx] = b.Data[srcIdx];
                }
            }
        }

        return result4D;
    }

    /// <summary>
    /// Splits a gradient tensor along the channel dimension (dim=0 for CHW, dim=1 for NCHW).
    /// </summary>
    private (Tensor<T> first, Tensor<T> second) SplitGradient(Tensor<T> grad, int firstChannels, int secondChannels)
    {
        // Handle 3D tensors (CHW format)
        if (grad.Shape.Length == 3)
        {
            int height = grad.Shape[1];
            int width = grad.Shape[2];

            var first = new Tensor<T>([firstChannels, height, width]);
            var second = new Tensor<T>([secondChannels, height, width]);

            int spatialSize = height * width;

            for (int c = 0; c < firstChannels; c++)
            {
                for (int hw = 0; hw < spatialSize; hw++)
                {
                    int srcIdx = c * spatialSize + hw;
                    int dstIdx = c * spatialSize + hw;
                    first.Data[dstIdx] = grad.Data[srcIdx];
                }
            }

            for (int c = 0; c < secondChannels; c++)
            {
                for (int hw = 0; hw < spatialSize; hw++)
                {
                    int srcIdx = (firstChannels + c) * spatialSize + hw;
                    int dstIdx = c * spatialSize + hw;
                    second.Data[dstIdx] = grad.Data[srcIdx];
                }
            }

            return (first, second);
        }

        // Handle 4D tensors (NCHW format)
        int batch = grad.Shape[0];
        int height4D = grad.Shape[2];
        int width4D = grad.Shape[3];

        var first4D = new Tensor<T>([batch, firstChannels, height4D, width4D]);
        var second4D = new Tensor<T>([batch, secondChannels, height4D, width4D]);

        int totalChannels4D = firstChannels + secondChannels;
        int spatialSize4D = height4D * width4D;

        for (int n = 0; n < batch; n++)
        {
            for (int c = 0; c < firstChannels; c++)
            {
                for (int hw = 0; hw < spatialSize4D; hw++)
                {
                    int srcIdx = n * (totalChannels4D * spatialSize4D) + c * spatialSize4D + hw;
                    int dstIdx = n * (firstChannels * spatialSize4D) + c * spatialSize4D + hw;
                    first4D.Data[dstIdx] = grad.Data[srcIdx];
                }
            }

            for (int c = 0; c < secondChannels; c++)
            {
                for (int hw = 0; hw < spatialSize4D; hw++)
                {
                    int srcIdx = n * (totalChannels4D * spatialSize4D) + (firstChannels + c) * spatialSize4D + hw;
                    int dstIdx = n * (secondChannels * spatialSize4D) + c * spatialSize4D + hw;
                    second4D.Data[dstIdx] = grad.Data[srcIdx];
                }
            }
        }

        return (first4D, second4D);
    }

    /// <summary>
    /// Adds two gradient tensors of the same shape element-wise.
    /// </summary>
    private Tensor<T> AddGradients(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorAdd(a, b);
    }

    #endregion
}

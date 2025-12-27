using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;

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

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => true;

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
        var parameters = new List<T>();
        foreach (var layer in _layers)
        {
            parameters.AddRange(layer.GetParameters().ToArray());
        }
        return new Vector<T>(parameters.ToArray());
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
        foreach (var layer in _layers)
        {
            layer.ResetState();
        }
    }

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

    /// <inheritdoc />
    public override Autodiff.ComputationNode<T> ExportComputationGraph(List<Autodiff.ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException(
            "DenseBlock does not support JIT compilation. Use the standard Forward/Backward API instead.");
    }

    #region Helper Methods

    /// <summary>
    /// Concatenates two tensors along the channel dimension (dim=1 for NCHW).
    /// </summary>
    private Tensor<T> ConcatenateChannels(Tensor<T> a, Tensor<T> b)
    {
        int batch = a.Shape[0];
        int channelsA = a.Shape[1];
        int channelsB = b.Shape[1];
        int height = a.Shape[2];
        int width = a.Shape[3];

        int totalChannels = channelsA + channelsB;
        var result = new Tensor<T>([batch, totalChannels, height, width]);

        // Copy first tensor
        int spatialSize = height * width;
        for (int n = 0; n < batch; n++)
        {
            for (int c = 0; c < channelsA; c++)
            {
                for (int hw = 0; hw < spatialSize; hw++)
                {
                    int srcIdx = n * (channelsA * spatialSize) + c * spatialSize + hw;
                    int dstIdx = n * (totalChannels * spatialSize) + c * spatialSize + hw;
                    result.Data[dstIdx] = a.Data[srcIdx];
                }
            }

            // Copy second tensor
            for (int c = 0; c < channelsB; c++)
            {
                for (int hw = 0; hw < spatialSize; hw++)
                {
                    int srcIdx = n * (channelsB * spatialSize) + c * spatialSize + hw;
                    int dstIdx = n * (totalChannels * spatialSize) + (channelsA + c) * spatialSize + hw;
                    result.Data[dstIdx] = b.Data[srcIdx];
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Splits a gradient tensor along the channel dimension.
    /// </summary>
    private (Tensor<T> first, Tensor<T> second) SplitGradient(Tensor<T> grad, int firstChannels, int secondChannels)
    {
        int batch = grad.Shape[0];
        int height = grad.Shape[2];
        int width = grad.Shape[3];

        var first = new Tensor<T>([batch, firstChannels, height, width]);
        var second = new Tensor<T>([batch, secondChannels, height, width]);

        int totalChannels = firstChannels + secondChannels;
        int spatialSize = height * width;

        for (int n = 0; n < batch; n++)
        {
            for (int c = 0; c < firstChannels; c++)
            {
                for (int hw = 0; hw < spatialSize; hw++)
                {
                    int srcIdx = n * (totalChannels * spatialSize) + c * spatialSize + hw;
                    int dstIdx = n * (firstChannels * spatialSize) + c * spatialSize + hw;
                    first.Data[dstIdx] = grad.Data[srcIdx];
                }
            }

            for (int c = 0; c < secondChannels; c++)
            {
                for (int hw = 0; hw < spatialSize; hw++)
                {
                    int srcIdx = n * (totalChannels * spatialSize) + (firstChannels + c) * spatialSize + hw;
                    int dstIdx = n * (secondChannels * spatialSize) + c * spatialSize + hw;
                    second.Data[dstIdx] = grad.Data[srcIdx];
                }
            }
        }

        return (first, second);
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

/// <summary>
/// A single layer within a DenseBlock: BN-ReLU-Conv1x1-BN-ReLU-Conv3x3.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal class DenseBlockLayer<T> : LayerBase<T>
{
    private readonly BatchNormalizationLayer<T> _bn1;
    private readonly ConvolutionalLayer<T> _conv1x1;
    private readonly BatchNormalizationLayer<T> _bn2;
    private readonly ConvolutionalLayer<T> _conv3x3;
    private readonly IActivationFunction<T> _relu;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _bn1Out;
    private Tensor<T>? _relu1Out;
    private Tensor<T>? _conv1Out;
    private Tensor<T>? _bn2Out;
    private Tensor<T>? _relu2Out;

    public override bool SupportsTraining => true;

    public DenseBlockLayer(int inputChannels, int growthRate, int height, int width, double bnMomentum = 0.1)
        : base([inputChannels, height, width], [growthRate, height, width])
    {
        _relu = new ReLUActivation<T>();

        // Bottleneck layer: 1x1 conv to reduce channels (4 * growthRate is standard)
        int bottleneckChannels = 4 * growthRate;

        _bn1 = new BatchNormalizationLayer<T>(inputChannels);
        _conv1x1 = new ConvolutionalLayer<T>(
            inputDepth: inputChannels,
            outputDepth: bottleneckChannels,
            kernelSize: 1,
            inputHeight: height,
            inputWidth: width,
            stride: 1,
            padding: 0,
            activation: new IdentityActivation<T>());

        _bn2 = new BatchNormalizationLayer<T>(bottleneckChannels);
        _conv3x3 = new ConvolutionalLayer<T>(
            inputDepth: bottleneckChannels,
            outputDepth: growthRate,
            kernelSize: 3,
            inputHeight: height,
            inputWidth: width,
            stride: 1,
            padding: 1,
            activation: new IdentityActivation<T>());
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        // BN-ReLU-Conv1x1
        _bn1Out = _bn1.Forward(input);
        _relu1Out = _relu.Activate(_bn1Out);
        _conv1Out = _conv1x1.Forward(_relu1Out);

        // BN-ReLU-Conv3x3
        _bn2Out = _bn2.Forward(_conv1Out);
        _relu2Out = _relu.Activate(_bn2Out);
        var output = _conv3x3.Forward(_relu2Out);

        return output;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput is null || _bn1Out is null || _relu1Out is null ||
            _conv1Out is null || _bn2Out is null || _relu2Out is null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Backward through conv3x3
        var grad = _conv3x3.Backward(outputGradient);

        // Backward through ReLU2
        grad = ApplyReluDerivative(_bn2Out, grad);

        // Backward through BN2
        grad = _bn2.Backward(grad);

        // Backward through conv1x1
        grad = _conv1x1.Backward(grad);

        // Backward through ReLU1
        grad = ApplyReluDerivative(_bn1Out, grad);

        // Backward through BN1
        grad = _bn1.Backward(grad);

        return grad;
    }

    private Tensor<T> ApplyReluDerivative(Tensor<T> input, Tensor<T> grad)
    {
        var result = new T[grad.Data.Length];
        for (int i = 0; i < grad.Data.Length; i++)
        {
            // ReLU derivative: 1 if x > 0, else 0
            result[i] = NumOps.GreaterThan(input.Data[i], NumOps.Zero)
                ? grad.Data[i]
                : NumOps.Zero;
        }
        return new Tensor<T>(grad.Shape, new Vector<T>(result));
    }

    public override void UpdateParameters(T learningRate)
    {
        _bn1.UpdateParameters(learningRate);
        _conv1x1.UpdateParameters(learningRate);
        _bn2.UpdateParameters(learningRate);
        _conv3x3.UpdateParameters(learningRate);
    }

    public override Vector<T> GetParameters()
    {
        var parameters = new List<T>();
        parameters.AddRange(_bn1.GetParameters().ToArray());
        parameters.AddRange(_conv1x1.GetParameters().ToArray());
        parameters.AddRange(_bn2.GetParameters().ToArray());
        parameters.AddRange(_conv3x3.GetParameters().ToArray());
        return new Vector<T>(parameters.ToArray());
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;

        int count = _bn1.GetParameters().Length;
        _bn1.SetParameters(parameters.SubVector(offset, count));
        offset += count;

        count = _conv1x1.GetParameters().Length;
        _conv1x1.SetParameters(parameters.SubVector(offset, count));
        offset += count;

        count = _bn2.GetParameters().Length;
        _bn2.SetParameters(parameters.SubVector(offset, count));
        offset += count;

        count = _conv3x3.GetParameters().Length;
        _conv3x3.SetParameters(parameters.SubVector(offset, count));
    }

    public override void ResetState()
    {
        _lastInput = null;
        _bn1Out = null;
        _relu1Out = null;
        _conv1Out = null;
        _bn2Out = null;
        _relu2Out = null;

        _bn1.ResetState();
        _conv1x1.ResetState();
        _bn2.ResetState();
        _conv3x3.ResetState();
    }

    public override bool SupportsJitCompilation => false;

    public override Autodiff.ComputationNode<T> ExportComputationGraph(List<Autodiff.ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException("DenseBlockLayer does not support JIT compilation.");
    }
}

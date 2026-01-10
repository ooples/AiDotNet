using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents an input layer that passes input data through unchanged to the next layer in the neural network.
/// </summary>
/// <remarks>
/// <para>
/// The Input Layer serves as the entry point for data into a neural network. Unlike other layers, 
/// it doesn't transform the data or learn any parameters; it simply validates and passes the input
/// through to the next layer. This layer establishes the dimensionality of the input data for the
/// entire network.
/// </para>
/// <para><b>For Beginners:</b> This layer is like the doorway to your neural network.
/// 
/// Think of the InputLayer as:
/// - The entrance where your data first enters the neural network
/// - A way to tell the network what shape your data has
/// - A pass-through that doesn't change your data
/// 
/// For example, if you're processing images that are 28x28 pixels, you would use an InputLayer
/// with inputSize=784 (28×28) to tell the network about the size of each image.
/// 
/// Unlike other layers, the InputLayer doesn't learn or transform anything - it just
/// passes your data into the network.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class InputLayer<T> : LayerBase<T>
{
    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// <c>false</c> because the input layer has no trainable parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the layer can be trained through backpropagation.
    /// The InputLayer always returns false because it contains no trainable parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can learn from data.
    /// 
    /// A value of false means:
    /// - The layer doesn't have any values that can be adjusted during training
    /// - It will behave exactly the same before and after training
    /// - It doesn't participate in the learning process
    /// 
    /// The input layer doesn't need to learn because its only job is to feed data into the network.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => false;

    /// <inheritdoc/>
    protected override bool SupportsGpuExecution => true;

    /// <inheritdoc/>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        return inputs[0];
    }

    /// <summary>
    /// Computes the gradient of the loss with respect to the input on the GPU.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The same output gradient, unchanged.</returns>
    /// <remarks>
    /// InputLayer is an identity operation, so the gradient passes through unchanged.
    /// </remarks>
    public override IGpuTensor<T> BackwardGpu(IGpuTensor<T> outputGradient)
    {
        return outputGradient;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="InputLayer{T}"/> class with the specified input size.
    /// </summary>
    /// <param name="inputSize">The size of the input vector.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new Input Layer with the specified input size. The layer uses an identity
    /// activation function, meaning it does not transform the input data in any way.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new doorway to your neural network with a specific size.
    /// 
    /// When creating an Input Layer, you specify:
    /// - inputSize: How many features your data has (like pixels in an image or properties in a dataset)
    /// 
    /// This size tells the rest of the network what dimensions to expect for input data.
    /// For example, if your data has 10 features, you would set inputSize=10.
    /// 
    /// The layer is automatically set up to pass data through without changing it.
    /// </para>
    /// </remarks>
    public InputLayer(int inputSize)
        : base([inputSize], [inputSize], new IdentityActivation<T>() as IActivationFunction<T>)
    {
    }

    /// <summary>
    /// Performs the forward pass of the input layer, simply returning the input unchanged.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The same input tensor, unchanged.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the input layer. Since the input layer does not transform
    /// the data, it simply returns the input tensor as is.
    /// </para>
    /// <para><b>For Beginners:</b> This method passes your data into the neural network without changing it.
    /// 
    /// The forward pass:
    /// - Takes in your data
    /// - Passes it through unchanged
    /// - Sends it to the next layer
    /// 
    /// This simple pass-through behavior is all that's needed for an input layer,
    /// as its purpose is just to feed data into the network.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        return input;
    }

    /// <summary>
    /// Performs the backward pass of the input layer, simply returning the output gradient unchanged.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The same output gradient, unchanged.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the input layer. Since the input layer does not transform
    /// the data and has no parameters to learn, it simply passes the gradient back unchanged.
    /// </para>
    /// <para><b>For Beginners:</b> This method passes error information backward through the network.
    ///
    /// During the backward pass:
    /// - The layer receives information about how its output contributed to errors
    /// - Since this layer doesn't change anything, it passes this information back unchanged
    /// - There are no parameters to update
    ///
    /// This method is still needed even though the layer doesn't learn, because
    /// it's part of the backpropagation process that allows the entire network to learn.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Manual backward pass implementation (optimized).
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The same output gradient, unchanged.</returns>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        return outputGradient;
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The same output gradient, unchanged.</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation to compute gradients. For InputLayer,
    /// since it performs identity operation (passthrough), the gradient is simply passed through.
    /// This implementation exists for consistency with other layers and verification purposes.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        // InputLayer is an identity operation: output = input
        // The gradient of identity is 1, so: d(loss)/d(input) = d(loss)/d(output) * 1 = d(loss)/d(output)
        // Therefore, we simply return the output gradient unchanged
        return outputGradient;
    }

    /// <summary>
    /// Update parameters is a no-op for the input layer since it has no trainable parameters.
    /// </summary>
    /// <param name="learningRate">The learning rate (unused in this layer).</param>
    /// <remarks>
    /// <para>
    /// This method is implemented as required by the LayerBase interface but does nothing for the InputLayer
    /// since it has no parameters to update.
    /// </para>
    /// <para><b>For Beginners:</b> This method exists but does nothing because there's nothing to update.
    /// 
    /// Since the input layer:
    /// - Has no weights or biases
    /// - Doesn't transform the data
    /// - Doesn't learn from training
    /// 
    /// This method is included only because all layers must have this method,
    /// but it doesn't actually do anything for the input layer.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // Input layer has no parameters to update
    }

    /// <summary>
    /// Returns an empty vector since the input layer has no trainable parameters.
    /// </summary>
    /// <returns>An empty vector.</returns>
    /// <remarks>
    /// <para>
    /// This method returns an empty vector since the InputLayer has no trainable parameters.
    /// It is implemented as required by the LayerBase interface.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns an empty list because there are no parameters.
    /// 
    /// Since the input layer:
    /// - Has no weights or biases
    /// - Doesn't have any learnable values
    /// 
    /// This method returns an empty vector to indicate there are no parameters.
    /// Other layers would return their weights and biases here.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // InputLayer has no trainable parameters
        return Vector<T>.Empty();
    }

    /// <summary>
    /// Reset state is a no-op for the input layer since it maintains no state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method is implemented as required by the LayerBase interface but does nothing for the InputLayer
    /// since it maintains no state between forward and backward passes.
    /// </para>
    /// <para><b>For Beginners:</b> This method exists but does nothing because there's no state to reset.
    /// 
    /// Since the input layer:
    /// - Doesn't store any information between processing steps
    /// - Doesn't keep track of previous inputs or outputs
    /// - Has no memory that needs to be cleared
    /// 
    /// This method is included only because all layers must have this method,
    /// but it doesn't actually do anything for the input layer.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // InputLayer has no state to reset
    }

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        return inputNode; // Identity - pass through unchanged
    }

    public override bool SupportsJitCompilation => true; // Always supports JIT (identity operation)
}

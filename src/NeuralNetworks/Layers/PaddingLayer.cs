using AiDotNet.Autodiff;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a layer that adds padding to the input tensor.
/// </summary>
/// <remarks>
/// <para>
/// The PaddingLayer adds a specified amount of padding around the edges of the input tensor.
/// This is commonly used in convolutional neural networks to preserve spatial dimensions
/// after convolution operations or to provide additional context at the boundaries of the input.
/// The padding is added symmetrically on both sides of each dimension of the input tensor.
/// </para>
/// <para><b>For Beginners:</b> This layer adds extra space around the edges of your data.
/// 
/// Think of it like adding a frame around a picture:
/// - You have an image (your input data)
/// - The padding adds extra space around all sides of the image
/// - The padding is filled with zeros by default
/// 
/// This is useful for:
/// - Preserving the size of images when applying convolutions
/// - Preventing loss of information at the edges of the data
/// - Giving convolutional filters more context at the boundaries
/// 
/// For example, if you have a 28×28 image and add padding of 2 pixels on all sides,
/// you get a 32×32 image with your original data in the center and zeros around the edges.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class PaddingLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The amount of padding to add to each dimension of the input tensor.
    /// </summary>
    /// <remarks>
    /// This array specifies the padding amount for each dimension. Each value represents
    /// the number of zeros to add on both sides of the corresponding dimension.
    /// </remarks>
    private readonly int[] _padding;

    /// <summary>
    /// The input tensor from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// This field stores the input tensor from the most recent forward pass, which is needed
    /// during the backward pass for gradient calculation.
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>true</c> because the PaddingLayer supports backpropagation, even though it has no parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the layer supports backpropagation during training. Although
    /// the PaddingLayer has no trainable parameters, it still supports the backward pass to propagate
    /// gradients to previous layers.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can participate in the training process.
    /// 
    /// A value of true means:
    /// - The layer can pass gradient information backward during training
    /// - It's part of the learning process, even though it doesn't have learnable parameters
    /// 
    /// While this layer doesn't have weights or biases that get updated during training,
    /// it still needs to properly handle gradients to ensure that layers before it
    /// can learn correctly.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="PaddingLayer{T}"/> class with the specified input shape,
    /// padding, and a scalar activation function.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor.</param>
    /// <param name="padding">The amount of padding to add to each dimension.</param>
    /// <param name="activationFunction">The activation function to apply after processing. Defaults to Identity if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a PaddingLayer with the specified input shape and padding amounts.
    /// The output shape is calculated by adding twice the padding amount to each dimension of the input shape.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor sets up the layer with the necessary dimensions and padding values.
    /// 
    /// When creating a PaddingLayer, you need to specify:
    /// - inputShape: The shape of your input data (e.g., [32, 28, 28, 3] for 32 images of size 28×28 with 3 color channels)
    /// - padding: How much padding to add to each dimension (e.g., [0, 2, 2, 0] to add 2 pixels of padding around the height and width)
    /// - activationFunction: The function that processes the final output (optional)
    /// 
    /// The padding is applied to both sides of each dimension, so a padding of [0, 2, 2, 0]
    /// would add 4 pixels to the height (2 at the top and 2 at the bottom) and
    /// 4 pixels to the width (2 on the left and 2 on the right).
    /// </para>
    /// </remarks>
    public PaddingLayer(int[] inputShape, int[] padding, IActivationFunction<T>? activationFunction = null)
        : base(inputShape, CalculateOutputShape(inputShape, padding), activationFunction ?? new IdentityActivation<T>())
    {
        _padding = padding;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="PaddingLayer{T}"/> class with the specified input shape,
    /// padding, and a vector activation function.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor.</param>
    /// <param name="padding">The amount of padding to add to each dimension.</param>
    /// <param name="vectorActivationFunction">The vector activation function to apply after processing. Defaults to Identity if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a PaddingLayer with the specified input shape and padding amounts.
    /// The output shape is calculated by adding twice the padding amount to each dimension of the input shape.
    /// This overload accepts a vector activation function, which operates on entire vectors rather than
    /// individual elements.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor sets up the layer with a vector-based activation function.
    /// 
    /// A vector activation function:
    /// - Operates on entire groups of numbers at once, rather than one at a time
    /// - Can capture relationships between different elements in the output
    /// - Defaults to the Identity function, which doesn't change the values
    /// 
    /// This constructor is useful when you need more complex activation patterns
    /// that consider the relationships between different values after padding.
    /// </para>
    /// </remarks>
    public PaddingLayer(int[] inputShape, int[] padding, IVectorActivationFunction<T>? vectorActivationFunction = null)
        : base(inputShape, CalculateOutputShape(inputShape, padding), vectorActivationFunction ?? new IdentityActivation<T>())
    {
        _padding = padding;
    }

    /// <summary>
    /// Calculates the output shape of the padding layer based on the input shape and padding amounts.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor.</param>
    /// <param name="padding">The amount of padding to add to each dimension.</param>
    /// <returns>The calculated output shape for the padding layer.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates the output shape by adding twice the padding amount to each dimension
    /// of the input shape. This reflects the fact that padding is added symmetrically on both sides
    /// of each dimension.
    /// </para>
    /// <para><b>For Beginners:</b> This method figures out the shape of the data after padding is added.
    /// 
    /// The calculation is straightforward:
    /// - For each dimension of the input shape
    /// - Add twice the padding value for that dimension
    /// 
    /// For example:
    /// - Input shape: [32, 28, 28, 3]
    /// - Padding: [0, 2, 2, 0]
    /// - Output shape: [32, 32, 32, 3]
    /// 
    /// We add 2*2=4 to the height (28+4=32) and width (28+4=32) dimensions,
    /// while leaving the batch size (32) and channels (3) unchanged.
    /// </para>
    /// </remarks>
    private static int[] CalculateOutputShape(int[] inputShape, int[] padding)
    {
        int[] outputShape = new int[inputShape.Length];
        for (int i = 0; i < inputShape.Length; i++)
        {
            outputShape[i] = inputShape[i] + 2 * padding[i];
        }
        return outputShape;
    }

    /// <summary>
    /// Performs the forward pass of the padding layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after padding and activation.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the padding layer. It creates a new tensor with
    /// the padded dimensions, copies the input data to the appropriate positions in the padded tensor,
    /// and applies the activation function to the result. The input tensor is cached for use during
    /// the backward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This method performs the actual padding operation.
    /// 
    /// During the forward pass:
    /// - The method creates a new, larger tensor to hold the padded data
    /// - It copies the original data to the center of this new tensor
    /// - The areas around the edges are implicitly filled with zeros
    /// - Finally, it applies the activation function to the result
    /// 
    /// For example, with a 3×3 image and padding of 1:
    /// - The output is a 5×5 image
    /// - The original 3×3 data is in the center
    /// - The outer border of width 1 is filled with zeros
    /// 
    /// The method also saves the input for later use in backpropagation.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        if (_padding.Length != input.Shape.Length)
            throw new ArgumentException("Padding array length must match input dimensions.");

        // Assume BHWC format: padding order [batch, height, width, channels]
        var paddedOutput = Engine.Pad(input, _padding[1], _padding[1], _padding[2], _padding[2], NumOps.Zero);
        return ApplyActivation(paddedOutput);
    }

    /// <summary>
    /// Performs the backward pass of the padding layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the padding layer, which is used during training to propagate
    /// error gradients back through the network. It extracts the gradients corresponding to the original input
    /// positions from the output gradient tensor, ignoring the gradients in the padded regions. The method
    /// applies the activation function derivative to the result.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how changes in the input would affect the final output.
    ///
    /// During the backward pass:
    /// - The layer receives gradients for the entire padded output tensor
    /// - It extracts only the gradients corresponding to the original input area
    /// - The gradients in the padded regions are ignored (since they don't correspond to any input)
    ///
    /// This is essentially the reverse of the forward pass:
    /// - Forward: copy input to center of larger padded tensor
    /// - Backward: extract central region of gradient tensor that corresponds to the original input
    ///
    /// This allows the network to learn as if the padding wasn't there,
    /// while still benefiting from the additional context it provides.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Performs the backward pass using manual gradient computation (optimized implementation).
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        if (_padding.Length != _lastInput.Shape.Length)
            throw new ArgumentException("Padding array length must match input dimensions.");

        var inputGradient = Engine.PadBackward(outputGradient, _padding[1], _padding[2], _lastInput.Shape);
        return ApplyActivationDerivative(_lastInput, inputGradient);
    }

    /// <summary>
    /// Performs the backward pass using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    /// <remarks>
    /// <para>
    /// This method computes gradients using the same computation as BackwardManual to ensure
    /// identical results. Both paths use the same indexing logic for extracting the center
    /// region from the padded gradient tensor.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        if (_padding.Length != _lastInput.Shape.Length)
            throw new ArgumentException("Padding array length must match input dimensions.");

        var inputGradient = Engine.PadBackward(outputGradient, _padding[1], _padding[2], _lastInput.Shape);
        return ApplyActivationDerivative(_lastInput, inputGradient);
    }

    /// <summary>
    /// Updates the parameters of the padding layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method is part of the training process, but since PaddingLayer has no trainable parameters,
    /// this method does nothing.
    /// </para>
    /// <para><b>For Beginners:</b> This method would normally update a layer's internal values during training.
    /// 
    /// However, since PaddingLayer just performs a fixed operation (adding zeros around the edges) and doesn't
    /// have any internal values that can be learned or adjusted, this method is empty.
    /// 
    /// This is unlike layers such as Dense or Convolutional layers, which have weights and biases
    /// that get updated during training.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // No parameters to update in a padding layer
    }

    /// <summary>
    /// Gets all trainable parameters from the padding layer as a single vector.
    /// </summary>
    /// <returns>An empty vector since PaddingLayer has no trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters from the layer as a single vector. Since PaddingLayer
    /// has no trainable parameters, it returns an empty vector.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns all the learnable values in the layer.
    /// 
    /// Since PaddingLayer:
    /// - Only performs a fixed operation (adding zeros around the edges)
    /// - Has no weights, biases, or other learnable parameters
    /// - The method returns an empty list
    /// 
    /// This is different from layers like Dense layers, which would return their weights and biases.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // PaddingLayer has no trainable parameters
        return Vector<T>.Empty();
    }

    /// <summary>
    /// Resets the internal state of the padding layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the padding layer, including the cached input tensor.
    /// This is useful when starting to process a new sequence or batch of data.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - Stored input from previous processing is cleared
    /// - The layer forgets any information from previous data batches
    /// 
    /// This is important for:
    /// - Processing a new, unrelated batch of data
    /// - Ensuring clean state before a new training epoch
    /// - Preventing information from one batch affecting another
    /// 
    /// While the PaddingLayer doesn't maintain long-term state across samples,
    /// clearing these cached values helps with memory management and ensuring a clean processing pipeline.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward pass
        _lastInput = null;
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

        return TensorOperations<T>.Pad(inputNode, _padding);
    }

    public override bool SupportsJitCompilation => true;
}

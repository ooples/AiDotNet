namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A layer that adds multiple input tensors element-wise and optionally applies an activation function.
/// </summary>
/// <remarks>
/// <para>
/// The AddLayer combines multiple tensors of identical shape by adding their values element-wise. This is useful for 
/// implementing residual connections, skip connections, or any architecture that requires combining information from 
/// multiple sources. After adding the inputs, an optional activation function can be applied to the result.
/// </para>
/// <para><b>For Beginners:</b> This layer adds together multiple inputs of the same shape.
/// 
/// Think of this layer as performing element-wise addition:
/// - If you have two 3×3 matrices, it adds corresponding elements together
/// - All inputs must have exactly the same dimensions
/// - After adding, it can optionally apply an activation function
/// 
/// This is commonly used in:
/// - Residual networks (ResNets) where outputs from earlier layers are added to later layers
/// - Skip connections that help information flow more directly through deep networks
/// - Any situation where you want to combine information from multiple sources
/// 
/// For example, if you have two feature maps from different parts of a network,
/// this layer lets you combine them by adding their values together.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (like float, double, etc.)</typeparam>
public class AddLayer<T> : LayerBase<T>
{
    /// <summary>
    /// Stores the input tensors from the most recent forward pass for use in the backward pass.
    /// </summary>
    /// <remarks>
    /// This field caches the input tensors from the most recent call to Forward(). During backpropagation,
    /// these cached inputs are used to calculate the gradients. The field is nullable and will be null
    /// until Forward() is called at least once.
    /// </remarks>
    private Tensor<T>[]? _lastInputs;
    
    /// <summary>
    /// Stores the output tensor from the most recent forward pass for use in the backward pass.
    /// </summary>
    /// <remarks>
    /// This field caches the output tensor from the most recent call to Forward(). During backpropagation,
    /// this cached output is used to calculate the gradient of the activation function. The field is nullable
    /// and will be null until Forward() is called at least once.
    /// </remarks>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Indicates whether this layer has trainable parameters.
    /// </summary>
    /// <value>Always returns false because addition layers don't have parameters to train.</value>
    /// <remarks>
    /// <para>
    /// This property overrides the base class property to specify that addition layers do not have trainable parameters.
    /// Trainable parameters are values within a layer that are adjusted during the training process to minimize the loss
    /// function. Since addition layers simply add their inputs together without any adjustable parameters, this property
    /// always returns false.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you that addition layers don't learn or change during training.
    /// 
    /// While layers like Dense layers have weights that get updated during training,
    /// addition layers just perform a fixed mathematical operation (addition) that never changes.
    /// 
    /// This property helps the training system know that it doesn't need to
    /// update anything in this layer during the training process.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => false;

    /// <summary>
    /// Creates a new addition layer with the specified input shapes and an optional scalar activation function.
    /// </summary>
    /// <param name="inputShapes">An array of input shapes, where each shape is an array of integers representing the dimensions.</param>
    /// <param name="activationFunction">An optional activation function to apply after addition. If null, an identity function is used.</param>
    /// <exception cref="ArgumentException">Thrown when fewer than two input shapes are provided or when the shapes are not identical.</exception>
    /// <remarks>
    /// <para>
    /// This constructor creates an addition layer that expects multiple inputs with the specified shapes. All input shapes must be
    /// identical because element-wise addition requires tensors of the same dimensions. If an activation function is provided,
    /// it will be applied to the result of the addition. If no activation function is provided, an identity function is used,
    /// which means the output will be the same as the sum of the inputs.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates an addition layer that expects inputs of specific shapes.
    /// 
    /// When creating an AddLayer:
    /// - You specify the shapes of all inputs (they must all be the same shape)
    /// - You can optionally provide an activation function to apply after addition
    /// - If no activation function is provided, the output will just be the sum of inputs
    /// 
    /// For example:
    /// ```csharp
    /// // Create an AddLayer for combining two 28×28 feature maps with ReLU activation
    /// var addLayer = new AddLayer<float>(
    ///     new[] { new[] { 32, 28, 28, 64 }, new[] { 32, 28, 28, 64 } },
    ///     new ReLU<float>()
    /// );
    /// ```
    /// 
    /// The first parameter is an array of input shapes, where each shape describes
    /// the dimensions of one input tensor (like [batchSize, height, width, channels]).
    /// </para>
    /// </remarks>
    public AddLayer(int[][] inputShapes, IActivationFunction<T>? activationFunction = null)
        : base(inputShapes, inputShapes[0], activationFunction ?? new IdentityActivation<T>())
    {
        ValidateInputShapes(inputShapes);
    }

    /// <summary>
    /// Creates a new addition layer with the specified input shapes and an optional vector activation function.
    /// </summary>
    /// <param name="inputShapes">An array of input shapes, where each shape is an array of integers representing the dimensions.</param>
    /// <param name="vectorActivationFunction">An optional vector activation function to apply after addition. If null, an identity function is used.</param>
    /// <exception cref="ArgumentException">Thrown when fewer than two input shapes are provided or when the shapes are not identical.</exception>
    /// <remarks>
    /// <para>
    /// This constructor creates an addition layer that expects multiple inputs with the specified shapes. All input shapes must be
    /// identical because element-wise addition requires tensors of the same dimensions. If a vector activation function is provided,
    /// it will be applied to the result of the addition. If no activation function is provided, an identity function is used,
    /// which means the output will be the same as the sum of the inputs.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates an addition layer that applies a vector activation function after addition.
    /// 
    /// This constructor is similar to the other one, but it accepts a vector activation function
    /// instead of a scalar one. Vector<double> activation functions (like Softmax) need to consider
    /// multiple values together, while scalar functions (like ReLU) process each value independently.
    /// 
    /// For example:
    /// ```csharp
    /// // Create an AddLayer for combining two feature vectors with Softmax activation
    /// var addLayer = new AddLayer<float>(
    ///     new[] { new[] { 32, 10 }, new[] { 32, 10 } },
    ///     new Softmax<float>()
    /// );
    /// ```
    /// 
    /// Use this constructor when you need an activation function that operates on
    /// entire vectors rather than individual elements.
    /// </para>
    /// </remarks>
    public AddLayer(int[][] inputShapes, IVectorActivationFunction<T>? vectorActivationFunction = null)
        : base(inputShapes, inputShapes[0], vectorActivationFunction ?? new IdentityActivation<T>())
    {
        ValidateInputShapes(inputShapes);
    }

    /// <summary>
    /// Validates that there are at least two input shapes and that all shapes are identical.
    /// </summary>
    /// <param name="inputShapes">The array of input shapes to validate.</param>
    /// <exception cref="ArgumentException">Thrown when fewer than two input shapes are provided or when the shapes are not identical.</exception>
    /// <remarks>
    /// This private helper method checks that the input shapes meet the requirements for the AddLayer:
    /// there must be at least two input shapes (since addition requires at least two operands), and all
    /// shapes must be identical (since element-wise addition requires tensors of the same dimensions).
    /// </remarks>
    private static void ValidateInputShapes(int[][] inputShapes)
    {
        if (inputShapes.Length < 2)
        {
            throw new ArgumentException("AddLayer requires at least two input tensors.", nameof(inputShapes));
        }

        var firstShape = inputShapes[0];
        for (int i = 1; i < inputShapes.Length; i++)
        {
            if (!firstShape.SequenceEqual(inputShapes[i]))
            {
                throw new ArgumentException("All input shapes must be identical for AddLayer.", nameof(inputShapes));
            }
        }
    }

    /// <summary>
    /// This method is not supported for AddLayer, which requires multiple inputs.
    /// </summary>
    /// <param name="input">A single input tensor.</param>
    /// <returns>Never returns as this method always throws an exception.</returns>
    /// <exception cref="NotSupportedException">Always thrown because AddLayer requires multiple inputs.</exception>
    /// <remarks>
    /// <para>
    /// This method overrides the base class method but is not supported for AddLayer, which requires multiple inputs.
    /// Instead, use the Forward(params Tensor<double>&lt;T&gt;[] inputs) method, which accepts multiple input tensors.
    /// </para>
    /// <para><b>For Beginners:</b> This method is not supported because addition requires multiple inputs.
    /// 
    /// Since addition requires at least two operands, this layer doesn't support
    /// the single-input Forward method. If you try to use it, you'll get an error.
    /// 
    /// Instead, use the version of Forward that accepts multiple inputs:
    /// ```csharp
    /// var output = addLayer.Forward(input1, input2);
    /// ```
    /// 
    /// This design ensures that the layer is used correctly - you can't accidentally
    /// try to add just one tensor.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        throw new NotSupportedException("AddLayer requires multiple inputs. Use Forward(params Tensor<T>[] inputs) instead.");
    }

    /// <summary>
    /// Processes multiple input tensors by adding them element-wise and optionally applying an activation function.
    /// </summary>
    /// <param name="inputs">An array of input tensors to add together.</param>
    /// <returns>The result of adding the input tensors and applying the activation function.</returns>
    /// <exception cref="ArgumentException">Thrown when fewer than two input tensors are provided.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass for the addition layer. It adds all input tensors element-wise,
    /// then applies the activation function (if any) to the result. The input tensors and the output are stored
    /// for later use in the backward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This method adds multiple input tensors together element by element.
    /// 
    /// During the forward pass, this method:
    /// 1. Checks that you've provided at least two input tensors
    /// 2. Saves the inputs for later use in backpropagation
    /// 3. Creates a copy of the first input tensor
    /// 4. Adds each of the other input tensors to it
    /// 5. Applies the activation function (if any)
    /// 6. Saves and returns the result
    /// 
    /// For example, with inputs [1, 2, 3] and [4, 5, 6]:
    /// - The addition gives [5, 7, 9]
    /// - If using ReLU activation, the output remains [5, 7, 9]
    /// - If using a different activation, it would transform these values
    /// 
    /// This operation combines information from multiple sources in your network.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(params Tensor<T>[] inputs)
    {
        if (inputs.Length < 2)
        {
            throw new ArgumentException("AddLayer requires at least two input tensors.", nameof(inputs));
        }

        _lastInputs = inputs;

        var result = inputs[0].Clone();
        for (int i = 1; i < inputs.Length; i++)
        {
            result = result.Add(inputs[i]);
        }

        _lastOutput = ApplyActivation(result);
        return _lastOutput;
    }

    /// <summary>
    /// Calculates how changes in the output affect the inputs during training.
    /// </summary>
    /// <param name="outputGradient">How much the network's error changes with respect to this layer's output.</param>
    /// <returns>How much the network's error changes with respect to this layer's first input.</returns>
    /// <exception cref="InvalidOperationException">Thrown if called before Forward method.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass for the addition layer. It first applies the derivative of the
    /// activation function to the output gradient, then copies this gradient for each input. In an addition operation,
    /// the gradient with respect to each input is the same as the gradient with respect to the output (after accounting
    /// for the activation function). This method returns the gradient for the first input only, as required by the
    /// interface, but internally it calculates gradients for all inputs.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how the error gradient flows backward through this layer.
    /// 
    /// During backpropagation, this method:
    /// 1. Checks that Forward() was called first
    /// 2. Calculates how the gradient changes due to the activation function (if any)
    /// 3. Creates a copy of this gradient for each input
    /// 4. Returns the gradient for the first input
    /// 
    /// For addition, the gradient flows equally to all inputs. This means if the output
    /// needs to change by some amount, each input contributes equally to that change.
    /// 
    /// Note: This method only returns the gradient for the first input due to interface
    /// constraints. In a real network, you would need to handle returning all gradients
    /// to their respective sources.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInputs == null || _lastOutput == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        Tensor<T> gradientWithActivation;
        if (UsingVectorActivation && VectorActivation != null)
        {
            gradientWithActivation = VectorActivation.Derivative(_lastOutput).Multiply(outputGradient);
        }
        else if (ScalarActivation != null)
        {
            gradientWithActivation = _lastOutput.Transform((x, i) => NumOps.Multiply(ScalarActivation.Derivative(x), outputGradient[i]));
        }
        else
        {
            gradientWithActivation = outputGradient;
        }

        var inputGradients = new Tensor<T>[_lastInputs.Length];
        for (int i = 0; i < _lastInputs.Length; i++)
        {
            inputGradients[i] = gradientWithActivation.Clone();
        }

        return inputGradients[0];
    }

    /// <summary>
    /// Updates the layer's internal parameters during training.
    /// </summary>
    /// <param name="learningRate">How quickly the network should learn from new data.</param>
    /// <remarks>
    /// <para>
    /// This method is called during the training process after the forward and backward passes have been completed.
    /// For layers with trainable parameters, this method would update those parameters based on the gradients
    /// calculated during backpropagation and the provided learning rate. However, since addition layers have
    /// no trainable parameters, this method does nothing.
    /// </para>
    /// <para><b>For Beginners:</b> This method would update the layer's internal values during training, but addition layers have nothing to update.
    /// 
    /// In neural networks, training involves adjusting parameters to reduce errors.
    /// This method is where those adjustments happen, but addition layers don't have
    /// any adjustable parameters, so this method is empty.
    /// 
    /// For comparison:
    /// - In a Dense layer, this would update weights and biases
    /// - In a BatchNorm layer, this would update scale and shift parameters
    /// - In this AddLayer, there's nothing to update
    /// 
    /// The learning rate parameter controls how big the updates would be if there
    /// were any parameters to update - higher values mean bigger changes.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // No parameters to update in this layer
    }

    /// <summary>
    /// Gets all trainable parameters of this layer as a flat vector.
    /// </summary>
    /// <returns>An empty vector since addition layers have no trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method returns all trainable parameters of the layer as a flat vector. For layers with trainable
    /// parameters, this would involve reshaping multi-dimensional parameters (like weight matrices) into a
    /// one-dimensional vector. However, since addition layers have no trainable parameters, this method
    /// returns an empty vector.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns all the layer's trainable values as a single list, but addition layers have none.
    /// 
    /// Some operations in neural networks need to work with all parameters at once:
    /// - Saving and loading models
    /// - Applying regularization (techniques to prevent overfitting)
    /// - Using advanced optimization algorithms
    /// 
    /// This method provides those parameters as a single vector, but since
    /// addition layers don't have any trainable parameters, it returns an empty vector.
    /// 
    /// For comparison:
    /// - A Dense layer with 100 inputs and 10 outputs would return a vector with 1,010 values
    ///   (1,000 weights + 10 biases)
    /// - This AddLayer returns an empty vector with 0 values
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Add layers don't have parameters, so return an empty vector
        return Vector<T>.Empty();
    }

    /// <summary>
    /// Clears the layer's memory of previous inputs and outputs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the layer by clearing the cached input tensors and output tensor.
    /// The addition layer stores the inputs and output from the most recent forward pass to use during the backward
    /// pass for calculating gradients. Resetting this state is useful when starting to process new data or when you
    /// want to ensure the layer behaves deterministically.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory of previous calculations.
    /// 
    /// During training, the layer remembers the inputs and output from the last forward pass
    /// to help with backpropagation calculations. This method makes the layer "forget" those values.
    /// 
    /// You might need to reset state:
    /// - When starting a new batch of training data
    /// - Between training epochs
    /// - When switching from training to testing
    /// - When you want to ensure consistent behavior
    /// 
    /// For addition layers, this simply clears the saved input and output tensors.
    /// 
    /// This helps ensure that processing one batch doesn't accidentally affect
    /// the processing of the next batch.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        _lastInputs = null;
        _lastOutput = null;
    }
}
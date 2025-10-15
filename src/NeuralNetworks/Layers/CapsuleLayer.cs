namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a capsule neural network layer that encapsulates groups of neurons to better preserve spatial information.
/// </summary>
/// <remarks>
/// <para>
/// A capsule layer is a specialized neural network layer that groups neurons into "capsules," where each capsule 
/// represents a specific entity or feature. Unlike traditional neural networks that use scalar outputs, capsules 
/// output vectors whose length represents the probability of the entity's existence and whose orientation encodes 
/// the entity's properties. Capsule layers use dynamic routing between capsules, which helps preserve hierarchical 
/// relationships between features and improves the network's ability to recognize objects from different viewpoints.
/// </para>
/// <para><b>For Beginners:</b> A capsule layer is an advanced type of neural network layer that works differently 
/// from standard layers.
/// 
/// Traditional neural network layers use single numbers to represent features, but capsule layers use groups of numbers
/// (vectors) that can capture more information:
/// 
/// - Each "capsule" is a group of neurons that work together
/// - The length of a capsule's output tells you how likely something exists
/// - The direction of a capsule's output tells you about its properties (like position, size, rotation)
/// 
/// For example, if detecting faces in images:
/// - A traditional network might have neurons that detect eyes, nose, mouth separately
/// - A capsule network would understand how these parts relate to each other spatially
/// 
/// This helps the network recognize objects even when they're viewed from different angles or positions,
/// which is something traditional networks struggle with.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class CapsuleLayer<T> : LayerBase<T>
{
    private readonly int _numCapsules;
    private readonly int _capsuleDimension;
    private readonly int _numRoutingIterations;
    private Tensor<T> _transformationMatrix = default!;
    private Vector<T> _bias = default!;
    private Tensor<T>? _transformationMatrixGradient;
    private Vector<T>? _biasGradient;
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastCouplingCoefficients;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>true</c> as capsule layers have trainable parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property returns true, indicating that the capsule layer can be trained through backpropagation.
    /// Capsule layers contain trainable parameters (transformation matrices and biases) that are adjusted
    /// during the training process to minimize the network's error.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you that this layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer contains values (parameters) that will change during training
    /// - It will improve its performance as it sees more examples
    /// - It participates in the learning process of the neural network
    /// 
    /// Capsule layers always support training because they contain transformation matrices
    /// and bias values that need to be learned from data.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="CapsuleLayer{T}"/> class with specified dimensions and routing iterations.
    /// </summary>
    /// <param name="inputCapsules">The number of input capsules.</param>
    /// <param name="inputDimension">The dimension of each input capsule.</param>
    /// <param name="numCapsules">The number of output capsules.</param>
    /// <param name="capsuleDimension">The dimension of each output capsule.</param>
    /// <param name="numRoutingIterations">The number of dynamic routing iterations to perform.</param>
    /// <param name="activationFunction">The activation function to apply. Defaults to squash activation if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new capsule layer with the specified dimensions and routing parameters. It initializes
    /// the transformation matrix and bias vector with appropriate values. The transformation matrix is used to convert
    /// input capsules to output capsules, and the bias is added to each output capsule. The number of routing iterations
    /// determines how many times the dynamic routing algorithm is executed during the forward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new capsule layer with specific settings.
    /// 
    /// When creating a capsule layer, you need to specify:
    /// - How many input capsules there are (from the previous layer)
    /// - How many numbers each input capsule contains
    /// - How many output capsules you want this layer to create
    /// - How many numbers each output capsule should contain
    /// - How many routing iterations to perform (more iterations = more accurate but slower)
    /// 
    /// The "routing" is the special process that capsule networks use to determine which 
    /// higher-level capsules should receive information from lower-level capsules.
    /// 
    /// Think of it like this: if you see an eye, nose, and mouth, the routing process
    /// helps decide if they should be grouped together as a face.
    /// </para>
    /// </remarks>
    public CapsuleLayer(int inputCapsules, int inputDimension, int numCapsules, int capsuleDimension, int numRoutingIterations, IActivationFunction<T>? activationFunction = null)
        : base([inputCapsules, inputDimension], [numCapsules, capsuleDimension], activationFunction ?? new SquashActivation<T>())
    {
        _numCapsules = numCapsules;
        _capsuleDimension = capsuleDimension;
        _numRoutingIterations = numRoutingIterations;

        _transformationMatrix = new Tensor<T>([inputCapsules, inputDimension, numCapsules, capsuleDimension]);
        _bias = new Vector<T>(numCapsules * capsuleDimension);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes the layer's transformation matrix and bias parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the transformation matrix using a scaled random initialization, which helps
    /// with training convergence. The scaling factor is calculated based on the total number of elements in
    /// the transformation matrix. The bias vector is initialized to zeros, which is a common practice for
    /// neural network bias terms.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the initial values for the layer's parameters.
    /// 
    /// Before training begins, we need to:
    /// - Fill the transformation matrix with small random values (not too big, not too small)
    /// - Set all bias values to zero
    /// 
    /// This initialization is important because:
    /// - Starting with the right range of random values helps the network learn faster
    /// - If values are too large, the network might have trouble learning
    /// - If values are too small, the network might learn too slowly
    /// 
    /// The scaling factor (the "2.0 / totalElements" part) is a common technique to keep
    /// the initial values in a good range based on the size of the network.
    /// </para>
    /// </remarks>
    private void InitializeParameters()
    {
        int totalElements = _transformationMatrix.Shape.Aggregate(1, (acc, dim) => acc * dim);
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / totalElements));
        InitializeTensor(_transformationMatrix, scale);

        for (int i = 0; i < _bias.Length; i++)
        {
            _bias[i] = NumOps.Zero;
        }
    }

    /// <summary>
    /// Initializes a tensor with scaled random values.
    /// </summary>
    /// <param name="tensor">The tensor to initialize.</param>
    /// <param name="scale">The scale factor for the random values.</param>
    /// <remarks>
    /// <para>
    /// This method fills the given tensor with random values drawn from a uniform distribution
    /// centered at zero (-0.5 to 0.5) and scaled by the provided scale factor. This type of
    /// initialization helps with the training process by preventing issues like vanishing or
    /// exploding gradients.
    /// </para>
    /// <para><b>For Beginners:</b> This method fills a tensor with random values of an appropriate size.
    /// 
    /// When filling the tensor:
    /// - We generate random numbers between -0.5 and 0.5
    /// - We multiply each by a scaling factor to control their magnitude
    /// - The result is a tensor filled with small random values
    /// 
    /// This randomness is essential for neural networks - if all values started the same,
    /// the network wouldn't be able to learn different features.
    /// </para>
    /// </remarks>
    private void InitializeTensor(Tensor<T> tensor, T scale)
    {
        for (int i = 0; i < tensor.Shape.Aggregate(1, (acc, dim) => acc * dim); i++)
        {
            tensor.SetFlatIndex(i, NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale));
        }
    }

    /// <summary>
    /// Performs the forward pass of the capsule layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after capsule processing.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the capsule layer, including the dynamic routing algorithm.
    /// The input capsules are first transformed using the transformation matrix. Then, dynamic routing is performed
    /// for the specified number of iterations to determine the coupling coefficients between input and output capsules.
    /// These coefficients are used to compute the weighted sum for each output capsule, which is then passed through
    /// the squash activation function to produce the final output.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes the input data through the capsule layer.
    /// 
    /// The forward pass has several steps:
    /// 
    /// 1. Transform input capsules using the transformation matrix
    /// 2. Start with equal connections between all input and output capsules
    /// 3. Perform dynamic routing:
    ///    - Calculate weighted sums for each output capsule
    ///    - Add the bias values
    ///    - Apply the "squash" activation function to ensure vector lengths are between 0 and 1
    ///    - Update the connection strengths based on how well input and output capsules agree
    /// 4. Repeat the routing process multiple times to refine the connections
    /// 
    /// The "squash" function is special to capsule networks - it preserves the direction of a vector
    /// but adjusts its length to be between 0 and 1 (representing a probability).
    /// 
    /// Dynamic routing is what makes capsule networks unique - it's how they learn to group
    /// lower-level features into higher-level concepts.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];
        int inputCapsules = input.Shape[1];
        int inputDimension = input.Shape[2];

        // Reshape input for matrix multiplication
        var reshapedInput = input.Reshape(batchSize * inputCapsules, inputDimension);

        // Perform transformation
        var transformedInput = reshapedInput.Multiply(_transformationMatrix);
        transformedInput = transformedInput.Reshape(batchSize, inputCapsules, _numCapsules, _capsuleDimension);

        // Initialize coupling coefficients
        var couplingCoefficients = new Tensor<T>([batchSize, inputCapsules, _numCapsules]);
        couplingCoefficients.Fill(NumOps.FromDouble(1.0 / _numCapsules));

        // Declare output tensor outside the loop
        Tensor<T> output = null!;

        // Perform dynamic routing
        for (int i = 0; i < _numRoutingIterations; i++)
        {
            var weightedSum = new Tensor<T>([batchSize, _numCapsules, _capsuleDimension]);
            for (int b = 0; b < batchSize; b++)
            {
                for (int j = 0; j < inputCapsules; j++)
                {
                    for (int k = 0; k < _numCapsules; k++)
                    {
                        for (int d = 0; d < _capsuleDimension; d++)
                        {
                            weightedSum[b, k, d] = NumOps.Add(weightedSum[b, k, d], 
                                NumOps.Multiply(couplingCoefficients[b, j, k], transformedInput[b, j, k, d]));
                        }
                    }
                }
            }

            // Apply bias after the weighted sum
            for (int b = 0; b < batchSize; b++)
            {
                for (int k = 0; k < _numCapsules; k++)
                {
                    for (int d = 0; d < _capsuleDimension; d++)
                    {
                        weightedSum[b, k, d] = NumOps.Add(weightedSum[b, k, d], _bias[k * _capsuleDimension + d]);
                    }
                }
            }

            // Apply squash activation
            output = ApplyActivation(weightedSum);

            // Update coupling coefficients
            if (i < _numRoutingIterations - 1)
            {
                for (int b = 0; b < batchSize; b++)
                {
                    for (int j = 0; j < inputCapsules; j++)
                    {
                        for (int k = 0; k < _numCapsules; k++)
                        {
                            T agreement = NumOps.Zero;
                            for (int d = 0; d < _capsuleDimension; d++)
                            {
                                agreement = NumOps.Add(agreement, 
                                    NumOps.Multiply(transformedInput[b, j, k, d], output[b, k, d]));
                            }
                            couplingCoefficients[b, j, k] = NumOps.Add(couplingCoefficients[b, j, k], agreement);
                        }
                    }
                }

                couplingCoefficients = ApplySoftmax(couplingCoefficients);
            }
        }

        _lastOutput = output;
        _lastCouplingCoefficients = couplingCoefficients;

        return _lastOutput;
    }

    /// <summary>
    /// Performs the backward pass of the capsule layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the capsule layer, which is used during training to propagate
    /// error gradients back through the network. It computes the gradients of the loss with respect to the layer's
    /// parameters (transformation matrix and bias) and the layer's input. The gradients are stored internally and
    /// used during the parameter update step.
    /// </para>
    /// <para><b>For Beginners:</b> This method is used during training to calculate how the layer's inputs 
    /// and parameters should change to reduce errors.
    ///
    /// The backward pass:
    /// 1. Takes in gradients (directions of improvement) from the next layer
    /// 2. Applies the derivative of the activation function
    /// 3. Calculates how much each parameter (transformation matrix and bias) contributed to the error
    /// 4. Calculates how the input contributed to the error, to pass gradients to the previous layer
    /// 
    /// During this process, the method:
    /// - Creates gradient tensors for the transformation matrix and bias
    /// - Uses the coupling coefficients (connection strengths) calculated during the forward pass
    /// - Produces gradients that will be used to update the parameters
    /// 
    /// This is part of the "backpropagation" algorithm that helps neural networks learn.
    /// The error flows backward through the network, and each layer determines how it
    /// should change to reduce that error.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _lastCouplingCoefficients == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int inputCapsules = _lastInput.Shape[1];
        int inputDimension = _lastInput.Shape[2];

        var activationGradient = ApplyActivationDerivative(_lastOutput, outputGradient);

        _transformationMatrixGradient = new Tensor<T>([inputCapsules, inputDimension, _numCapsules, _capsuleDimension]);
        _biasGradient = new Vector<T>(_numCapsules * _capsuleDimension);
        var inputGradient = new Tensor<T>(_lastInput.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < inputCapsules; i++)
            {
                for (int j = 0; j < _numCapsules; j++)
                {
                    for (int d = 0; d < _capsuleDimension; d++)
                    {
                        T grad = outputGradient[b, j, d];
                        T coeff = _lastCouplingCoefficients[b, i, j];

                        // Update bias gradient
                        _biasGradient[j * _capsuleDimension + d] = NumOps.Add(
                            _biasGradient[j * _capsuleDimension + d],
                            grad
                        );

                        for (int k = 0; k < inputDimension; k++)
                        {
                            T input = _lastInput[b, i, k];
                            _transformationMatrixGradient[i, k, j, d] = NumOps.Add(
                                _transformationMatrixGradient[i, k, j, d], 
                                NumOps.Multiply(NumOps.Multiply(grad, coeff), input)
                            );
                            inputGradient[b, i, k] = NumOps.Add(
                                inputGradient[b, i, k], 
                                NumOps.Multiply(NumOps.Multiply(grad, coeff), _transformationMatrix[i, k, j, d])
                            );
                        }
                    }
                }
            }
        }

        return inputGradient;
    }

    /// <summary>
    /// Updates the layer's parameters using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <exception cref="InvalidOperationException">Thrown when update is called before backward.</exception>
    /// <remarks>
    /// <para>
    /// This method updates the layer's parameters (transformation matrix and bias) based on the gradients
    /// calculated during the backward pass. The learning rate controls the size of the parameter updates.
    /// The update is performed by subtracting the scaled gradients from the current parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the layer's internal values during training.
    /// 
    /// After calculating the gradients in the backward pass:
    /// - This method applies those changes to the transformation matrix and bias
    /// - The learning rate controls how big each update step is
    /// - Smaller learning rates mean slower but more stable learning
    /// - Larger learning rates mean faster but potentially unstable learning
    /// 
    /// The formula is simple: new_value = old_value - (gradient * learning_rate)
    /// 
    /// This is how the layer "learns" from data over time, gradually improving its ability
    /// to make accurate predictions or classifications.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_transformationMatrixGradient == null || _biasGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _transformationMatrix = _transformationMatrix.Subtract(_transformationMatrixGradient.Multiply(learningRate));
        _bias = _bias.Subtract(_biasGradient.Multiply(learningRate));
    }

    /// <summary>
    /// Applies the softmax activation function to the input tensor.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The tensor after softmax activation.</returns>
    /// <remarks>
    /// <para>
    /// This method applies the softmax activation function to the input tensor, which converts a vector of
    /// real numbers into a probability distribution. The softmax function is used to normalize the coupling
    /// coefficients so that they sum to 1 for each input capsule.
    /// </para>
    /// <para><b>For Beginners:</b> This method turns any set of numbers into probabilities that add up to 1.
    /// 
    /// The softmax function:
    /// - Takes a group of numbers of any size
    /// - Transforms them so they're all positive
    /// - Scales them so they sum to exactly 1
    /// 
    /// For example, if the input is [2, 1, 0]:
    /// - After softmax: [0.67, 0.24, 0.09] (they sum to 1)
    /// 
    /// In capsule networks, this is used to ensure that the connection strengths
    /// from each input capsule to all output capsules sum to 1, representing a
    /// probability distribution of where the information should flow.
    /// </para>
    /// </remarks>
    private static Tensor<T> ApplySoftmax(Tensor<T> input)
    {
        var softmax = new SoftmaxActivation<T>();
        return softmax.Activate(input);
    }

    /// <summary>
    /// Gets all trainable parameters from the layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters from the layer and combines them into a single vector.
    /// It flattens the transformation matrix and concatenates it with the bias vector. This is useful for
    /// optimization algorithms that operate on all parameters at once, or for saving and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values from the layer into a single list.
    /// 
    /// The parameters:
    /// - Include all values from the transformation matrix and bias
    /// - Are combined into a single long list (vector)
    /// - Represent everything this layer has learned
    /// 
    /// This is useful for:
    /// - Saving the model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques that need access to all parameters
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Flatten the transformation matrix and concatenate with bias
        int matrixSize = _transformationMatrix.Shape.Aggregate(1, (acc, dim) => acc * dim);
        var parameters = new Vector<T>(matrixSize + _bias.Length);
        
        // Copy transformation matrix parameters
        for (int i = 0; i < matrixSize; i++)
        {
            parameters[i] = _transformationMatrix.GetFlatIndexValue(i);
        }
        
        // Copy bias parameters
        for (int i = 0; i < _bias.Length; i++)
        {
            parameters[matrixSize + i] = _bias[i];
        }
        
        return parameters;
    }

    /// <summary>
    /// Sets the trainable parameters for the layer.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets the trainable parameters for the layer from a single vector. It extracts the appropriate
    /// portions of the input vector for the transformation matrix and bias. This is useful for loading saved model
    /// weights or for implementing optimization algorithms that operate on all parameters at once.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the learnable values in the layer.
    /// 
    /// When setting parameters:
    /// - The input must be a vector with the correct length
    /// - The first part of the vector is used for the transformation matrix
    /// - The second part of the vector is used for the bias
    /// 
    /// This is useful for:
    /// - Loading a previously saved model
    /// - Transferring parameters from another model
    /// - Testing different parameter values
    /// 
    /// An error is thrown if the input vector doesn't have the expected number of parameters.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        int matrixSize = _transformationMatrix.Shape.Aggregate(1, (acc, dim) => acc * dim);
        
        if (parameters.Length != matrixSize + _bias.Length)
            throw new ArgumentException($"Expected {matrixSize + _bias.Length} parameters, but got {parameters.Length}");
        
        // Set transformation matrix parameters
        for (int i = 0; i < matrixSize; i++)
        {
            _transformationMatrix.SetFlatIndex(i, parameters[i]);
        }
        
        // Set bias parameters
        for (int i = 0; i < _bias.Length; i++)
        {
            _bias[i] = parameters[matrixSize + i];
        }
    }

    /// <summary>
    /// Resets the internal state of the capsule layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the capsule layer, including the cached inputs, outputs,
    /// coupling coefficients, and gradients. This is useful when starting to process a new sequence or batch
    /// after training on a previous one.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's temporary memory to start fresh.
    /// 
    /// When resetting the state:
    /// - Stored inputs and outputs are cleared
    /// - Calculated gradients are cleared
    /// - Coupling coefficients (connection strengths) are cleared
    /// - The layer forgets any information from previous batches
    /// 
    /// This is important for:
    /// - Processing a new, unrelated batch of data
    /// - Preventing information from one batch affecting another
    /// - Preparing the layer for a new training step
    /// 
    /// Note that this doesn't reset the learned parameters (transformation matrix and bias),
    /// just the temporary information used during a single forward/backward pass.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastCouplingCoefficients = null;
        _transformationMatrixGradient = null;
        _biasGradient = null;
    }
}
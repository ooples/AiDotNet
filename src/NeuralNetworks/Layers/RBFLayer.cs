namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a Radial Basis Function (RBF) layer for neural networks.
/// </summary>
/// <remarks>
/// <para>
/// The RBF layer implements a type of artificial neural network that uses radial basis functions as 
/// activation functions. Each neuron in this layer has a center point in the input space and responds
/// most strongly to inputs near that center. The response decreases as the distance from the center
/// increases, controlled by the width parameter of each neuron.
/// </para>
/// <para><b>For Beginners:</b> This layer works like a collection of specialized detectors.
/// 
/// Think of each neuron in this layer as a spotlight:
/// - Each spotlight has a specific location (center) in the input space
/// - Each spotlight has a certain brightness range (width)
/// - When input comes in, spotlights that are close to that input light up brightly
/// - Spotlights far from the input barely light up at all
/// 
/// For example, if you're recognizing handwritten digits:
/// - One spotlight might be positioned to detect curved lines (like in "8")
/// - Another might detect vertical lines (like in "1")
/// - When a "3" comes in, the spotlights for curves light up strongly, while others stay dim
/// 
/// This layer is particularly good at classification problems and function approximation
/// where the relationship between inputs and outputs is complex or non-linear.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class RBFLayer<T> : LayerBase<T>
{
    /// <summary>
    /// Matrix<double> storing the center positions of each RBF neuron in the input space.
    /// </summary>
    /// <remarks>
    /// This matrix has dimensions [outputSize, inputSize], where each row represents the coordinates
    /// of a center point for one RBF neuron. These centers are the primary trainable parameters of
    /// the layer and determine where in the input space each neuron responds most strongly.
    /// </remarks>
    private Matrix<T> _centers = default!;

    /// <summary>
    /// Vector<double> storing the width parameters for each RBF neuron.
    /// </summary>
    /// <remarks>
    /// This vector has length outputSize, where each element controls how quickly the response of
    /// the corresponding RBF neuron decreases as the distance from its center increases. Larger
    /// width values mean the neuron responds more broadly, while smaller values make the response
    /// more focused around the center.
    /// </remarks>
    private Vector<T> _widths = default!;

    /// <summary>
    /// The radial basis function implementation used to compute neuron activations.
    /// </summary>
    /// <remarks>
    /// This interface provides methods to compute the activation of an RBF neuron based on the
    /// distance from the center, as well as derivatives for these computations needed during
    /// backpropagation. Common implementations include Gaussian, Multiquadric, and Inverse Quadratic
    /// functions, each providing different response patterns.
    /// </remarks>
    private IRadialBasisFunction<T> _rbf = default!;

    /// <summary>
    /// Stores the input tensor from the most recent forward pass for use in backpropagation.
    /// </summary>
    /// <remarks>
    /// This cached input is needed during the backward pass to compute gradients. It holds the
    /// batch of input vectors that were processed in the most recent forward pass. The tensor
    /// is null before the first forward pass or after a reset.
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the output tensor from the most recent forward pass for use in backpropagation.
    /// </summary>
    /// <remarks>
    /// This cached output is needed during the backward pass to compute certain derivatives.
    /// It holds the batch of output vectors that were produced in the most recent forward pass.
    /// The tensor is null before the first forward pass or after a reset.
    /// </remarks>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Stores the gradients of the loss with respect to the center parameters.
    /// </summary>
    /// <remarks>
    /// This matrix holds the accumulated gradients for all center parameters during the backward pass.
    /// It has the same dimensions as the _centers matrix and is used to update the centers during
    /// the parameter update step. The matrix is null before the first backward pass or after a reset.
    /// </remarks>
    private Matrix<T>? _centersGradient;

    /// <summary>
    /// Stores the gradients of the loss with respect to the width parameters.
    /// </summary>
    /// <remarks>
    /// This vector holds the accumulated gradients for all width parameters during the backward pass.
    /// It has the same length as the _widths vector and is used to update the widths during
    /// the parameter update step. The vector is null before the first backward pass or after a reset.
    /// </remarks>
    private Vector<T>? _widthsGradient;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>true</c> for RBF layers, indicating that the layer can be trained through backpropagation.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that the RBF layer has trainable parameters (centers and widths) that
    /// can be optimized during the training process using backpropagation. The gradients of these parameters
    /// are calculated during the backward pass and used to update the parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer has values (centers and widths) that can be adjusted during training
    /// - It will improve its performance as it sees more data
    /// - It participates in the learning process of the neural network
    /// 
    /// When you train a neural network containing this layer, the centers and widths will 
    /// automatically adjust to better match the patterns in your specific data.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="RBFLayer{T}"/> class with specified dimensions and radial basis function.
    /// </summary>
    /// <param name="inputSize">The size of the input to the layer.</param>
    /// <param name="outputSize">The size of the output from the layer (number of RBF neurons).</param>
    /// <param name="rbf">The radial basis function to use for computing neuron activations.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new RBF layer with the specified dimensions and radial basis function.
    /// The centers are initialized randomly around the origin, and the widths are initialized with random
    /// values between 0 and 1. The scale of the random initialization for centers depends on the layer dimensions.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new RBF layer for your neural network.
    /// 
    /// When you create this layer, you specify:
    /// - inputSize: How many numbers come into the layer (features of your data)
    /// - outputSize: How many RBF neurons to create (more neurons can capture more complex patterns)
    /// - rbf: The specific mathematical function that determines how neurons respond to input
    /// 
    /// For example, if you're analyzing images with 784 pixels and want 100 different pattern detectors,
    /// you might use inputSize=784 and outputSize=100.
    /// 
    /// Common radial basis functions include Gaussian (bell-shaped), Multiquadric, and Inverse Quadratic.
    /// Each creates a different pattern of responsiveness around the neuron centers.
    /// </para>
    /// </remarks>
    public RBFLayer(int inputSize, int outputSize, IRadialBasisFunction<T> rbf)
        : base([inputSize], [outputSize])
    {
        _centers = new Matrix<T>(outputSize, inputSize);
        _widths = new Vector<T>(outputSize);
        _rbf = rbf;

        InitializeParameters();
    }

    /// <summary>
    /// Performs the forward pass of the RBF layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after RBF processing.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the RBF layer. For each neuron (center), it calculates
    /// the Euclidean distance between the input vector and the center, then applies the radial basis function
    /// to this distance to produce the neuron's activation. The input and output are cached for use during
    /// the backward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes your data through the RBF neurons.
    /// 
    /// During the forward pass:
    /// 1. For each input example, the layer measures how far it is from each neuron's center
    /// 2. The distance is plugged into the radial basis function (like a mathematical formula)
    /// 3. The result determines how strongly each neuron activates
    /// 4. Neurons with centers close to the input activate strongly; distant ones activate weakly
    /// 
    /// This is like asking "How similar is this input to each of my known patterns?" The output
    /// tells you the similarity scores for each pattern.
    /// 
    /// The layer saves the input and output for later use during training.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];

        var output = new Tensor<T>([batchSize, _centers.Rows]);

        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < _centers.Rows; j++)
            {
                T distance = CalculateDistance(input.GetVector(i), _centers.GetRow(j));
                output[i, j] = _rbf.Compute(distance);
            }
        }

        _lastOutput = output;
        return output;
    }

    /// <summary>
    /// Performs the backward pass of the RBF layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the RBF layer, which is used during training
    /// to propagate error gradients back through the network. It calculates the gradients of the loss
    /// with respect to the centers and widths (to update the layer's parameters) and with respect to
    /// the input (to propagate back to previous layers).
    /// </para>
    /// <para><b>For Beginners:</b> This method is used during training to calculate how the layer should change to reduce errors.
    /// 
    /// During the backward pass:
    /// 1. The error gradient from the next layer is received
    /// 2. The layer calculates how each center should move to reduce the error
    /// 3. The layer calculates how each width should change to reduce the error
    /// 4. The layer calculates how the previous layer's output should change
    /// 
    /// This is like saying "Based on the mistakes we made, how should we adjust our pattern detectors
    /// to be more accurate next time?" The gradients tell us both how to update this layer and
    /// how to guide the previous layers.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int inputSize = _lastInput.Shape[1];
        int outputSize = _centers.Rows;

        _centersGradient = new Matrix<T>(outputSize, inputSize);
        _widthsGradient = new Vector<T>(outputSize);

        var inputGradient = new Tensor<T>(_lastInput.Shape);

        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                T distance = CalculateDistance(_lastInput.GetVector(i), _centers.GetRow(j));
                T rbfDerivative = _rbf.ComputeDerivative(distance);

                for (int k = 0; k < inputSize; k++)
                {
                    T inputDiff = NumOps.Subtract(_lastInput[i, k], _centers[j, k]);
                    T gradient = NumOps.Multiply(outputGradient[i, j], rbfDerivative);
                    T centerGradient = NumOps.Multiply(gradient, inputDiff);

                    _centersGradient[j, k] = NumOps.Add(_centersGradient[j, k], centerGradient);
                    inputGradient[i, k] = NumOps.Add(inputGradient[i, k], NumOps.Negate(centerGradient));
                }

                _widthsGradient[j] = NumOps.Add(_widthsGradient[j], 
                    NumOps.Multiply(outputGradient[i, j], _rbf.ComputeWidthDerivative(distance)));
            }
        }

        return inputGradient;
    }

    /// <summary>
    /// Updates the parameters of the RBF layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <exception cref="InvalidOperationException">Thrown when UpdateParameters is called before Backward.</exception>
    /// <remarks>
    /// <para>
    /// This method updates the centers and widths of the RBF layer based on the gradients
    /// calculated during the backward pass. The learning rate controls the size of the parameter
    /// updates. This method should be called after the backward pass to apply the calculated updates.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the layer's internal values during training.
    /// 
    /// When updating parameters:
    /// 1. The center positions are adjusted based on their gradients
    /// 2. The widths are adjusted based on their gradients
    /// 3. The learning rate controls how big each update step is
    /// 
    /// Imagine each RBF neuron as a spotlight:
    /// - Updating the centers moves where the spotlight is pointing
    /// - Updating the widths changes how broad or narrow the spotlight beam is
    /// 
    /// Smaller learning rates mean slower but more stable learning, while larger learning rates
    /// mean faster but potentially unstable learning.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_centersGradient == null || _widthsGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        for (int i = 0; i < _centers.Rows; i++)
        {
            for (int j = 0; j < _centers.Columns; j++)
            {
                _centers[i, j] = NumOps.Subtract(_centers[i, j], 
                    NumOps.Multiply(learningRate, _centersGradient[i, j]));
            }
            _widths[i] = NumOps.Subtract(_widths[i], 
                NumOps.Multiply(learningRate, _widthsGradient[i]));
        }
    }

    /// <summary>
    /// Gets all trainable parameters of the RBF layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters (centers and widths).</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters (centers and widths) of the RBF layer as a
    /// single vector. The centers are stored first, followed by the widths. This is useful for optimization
    /// algorithms that operate on all parameters at once, or for saving and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values from the RBF layer.
    /// 
    /// The parameters:
    /// - Are the centers and widths that the RBF layer learns during training
    /// - Control where and how widely each neuron responds to inputs
    /// - Are returned as a single list (vector)
    /// 
    /// This is useful for:
    /// - Saving the model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques that need access to all parameters
    /// 
    /// The centers are stored first in the vector, followed by all the width values.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Calculate total number of parameters
        int totalParams = _centers.Rows * _centers.Columns + _widths.Length;
    
        var parameters = new Vector<T>(totalParams);
        int index = 0;
    
        // Copy centers
        for (int i = 0; i < _centers.Rows; i++)
        {
            for (int j = 0; j < _centers.Columns; j++)
            {
                parameters[index++] = _centers[i, j];
            }
        }
    
        // Copy widths
        for (int i = 0; i < _widths.Length; i++)
        {
            parameters[index++] = _widths[i];
        }
    
        return parameters;
    }

    /// <summary>
    /// Sets the trainable parameters of the RBF layer.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters (centers and widths) to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets the trainable parameters (centers and widths) of the RBF layer from a single vector.
    /// The vector should contain the center values first, followed by the width values. This is useful for loading
    /// saved model weights or for implementing optimization algorithms that operate on all parameters at once.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the centers and widths in the RBF layer.
    /// 
    /// When setting parameters:
    /// - The input must be a vector with the correct total length
    /// - The first part of the vector is used for the centers (positions of the neurons)
    /// - The second part of the vector is used for the widths (how broadly each neuron responds)
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
        int totalParams = _centers.Rows * _centers.Columns + _widths.Length;
    
        if (parameters.Length != totalParams)
        {
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");
        }
    
        int index = 0;
    
        // Set centers
        for (int i = 0; i < _centers.Rows; i++)
        {
            for (int j = 0; j < _centers.Columns; j++)
            {
                _centers[i, j] = parameters[index++];
            }
        }
    
        // Set widths
        for (int i = 0; i < _widths.Length; i++)
        {
            _widths[i] = parameters[index++];
        }
    }

    /// <summary>
    /// Resets the internal state of the RBF layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the RBF layer, including the cached inputs and outputs
    /// from the forward pass, and the gradients from the backward pass. This is useful when starting to
    /// process a new sequence or batch of data.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - Stored inputs and outputs from previous calculations are cleared
    /// - Calculated gradients are cleared
    /// - The layer forgets any information from previous batches
    /// 
    /// This is important for:
    /// - Processing a new, unrelated batch of data
    /// - Preventing information from one batch affecting another
    /// - Starting a new training episode
    /// 
    /// The centers and widths (the learned parameters) are not reset,
    /// only the temporary state information.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastOutput = null;
        _centersGradient = null;
        _widthsGradient = null;
    }

    /// <summary>
    /// Initializes the centers and widths of the RBF layer with random values.
    /// </summary>
    /// <remarks>
    /// This private method initializes the centers with random values scaled by the input and output dimensions,
    /// and initializes the widths with random values between 0 and 1. This provides a good starting point for
    /// training the RBF layer.
    /// </remarks>
    private void InitializeParameters()
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_centers.Rows + _centers.Columns)));
        for (int i = 0; i < _centers.Rows; i++)
        {
            for (int j = 0; j < _centers.Columns; j++)
            {
                _centers[i, j] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
            }

            _widths[i] = NumOps.FromDouble(Random.NextDouble());
        }
    }

    /// <summary>
    /// Calculates the Euclidean distance between an input vector and a center vector.
    /// </summary>
    /// <param name="x">The input vector.</param>
    /// <param name="center">The center vector.</param>
    /// <returns>The Euclidean distance between the vectors.</returns>
    /// <remarks>
    /// This private method calculates the Euclidean distance between an input vector and a center vector,
    /// which is the square root of the sum of squared differences between corresponding components.
    /// This distance is used to determine how strongly each RBF neuron activates in response to an input.
    /// </remarks>
    private T CalculateDistance(Vector<T> x, Vector<T> center)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < x.Length; i++)
        {
            T diff = NumOps.Subtract(x[i], center[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }

        return NumOps.Sqrt(sum);
    }
}
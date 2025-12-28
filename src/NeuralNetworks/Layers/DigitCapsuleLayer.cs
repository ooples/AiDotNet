namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a digit capsule layer that implements the dynamic routing algorithm between capsules.
/// </summary>
/// <remarks>
/// <para>
/// A digit capsule layer extends the concept of traditional neural networks by using groups of neurons
/// (capsules) that encapsulate various properties of entities. This implementation is based on the 
/// CapsNet architecture proposed by Hinton et al., which uses a dynamic routing algorithm to determine 
/// how lower-level capsules should send their output to higher-level capsules.
/// </para>
/// <para><b>For Beginners:</b> A capsule layer is a special type of neural network layer that groups neurons together.
/// 
/// Think of regular neural networks as looking at individual puzzle pieces (like detecting edges or corners). 
/// A capsule network looks at how these pieces fit together to form objects.
/// 
/// For example, in image recognition:
/// - Regular neurons might detect a wheel, a window, and a door
/// - Capsules understand that these parts can make up a car, and how those parts relate to each other
/// 
/// This layer specifically handles digit recognition, taking information from previous capsule layers
/// and determining which digit is most likely present in the input.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class DigitCapsuleLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The weight tensor connecting input capsules to output capsules.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This 4D tensor stores the transformation matrices that map the input capsule vectors to
    /// the output capsule vectors. The dimensions are [inputCapsules, numClasses, inputCapsuleDimension, outputCapsuleDimension].
    /// </para>
    /// <para><b>For Beginners:</b> These weights are how the network learns to transform input information into output predictions.
    /// 
    /// Imagine translating between languages:
    /// - The weights act like a translation dictionary
    /// - They convert information from one "language" (input capsules) to another (output capsules)
    /// - During training, these weights are adjusted to make better predictions
    /// </para>
    /// </remarks>
    private Tensor<T> _weights;

    /// <summary>
    /// Gradients for the weight tensor, used during backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the gradients of the loss with respect to each weight. These gradients are used
    /// to update the weights during training. It has the same shape as the weight tensor.
    /// </para>
    /// <para><b>For Beginners:</b> This stores information about how to adjust the weights to improve accuracy.
    /// 
    /// When the network makes a mistake:
    /// - The gradients indicate which weights need to change and by how much
    /// - Larger gradients mean bigger adjustments are needed
    /// - They're like a "report card" for each weight showing what needs improvement
    /// </para>
    /// </remarks>
    private Tensor<T>? _weightsGradient;

    /// <summary>
    /// The input tensor from the last forward pass, saved for backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the input received during the last forward pass. It is necessary for computing
    /// gradients during the backward pass (backpropagation).
    /// </para>
    /// <para><b>For Beginners:</b> This remembers what input data was processed during the latest prediction.
    /// 
    /// It's like keeping your work when solving a math problem:
    /// - The network needs to know what input values it used
    /// - During training, it looks back at this input to understand its mistakes
    /// - This helps it learn how to adjust its weights correctly
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;

    /// <summary>
    /// The output tensor from the last forward pass, saved for backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the output produced during the last forward pass. It is used during
    /// backpropagation to compute certain gradients.
    /// </para>
    /// <para><b>For Beginners:</b> This remembers what predictions the layer made in its latest run.
    /// 
    /// When training the network:
    /// - It needs to remember what it predicted
    /// - It compares these predictions with the correct answers
    /// - This helps it understand how to improve for next time
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// The coupling coefficients from the last routing iteration, saved for backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the coupling coefficients computed during the dynamic routing process.
    /// These coefficients determine how much each input capsule contributes to each output capsule.
    /// </para>
    /// <para><b>For Beginners:</b> This tracks how strongly different input features connect to each output class.
    /// 
    /// Think of it like voting:
    /// - Each input capsule "votes" for different output capsules
    /// - These numbers record how strongly each input voted for each output
    /// - Higher values mean a stronger connection between an input and output
    /// 
    /// For example, if detecting numbers, an input feature that looks like a loop might strongly 
    /// "vote" for digits like 0, 6, 8, and 9.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastCouplings;

    /// <summary>
    /// The number of capsules in the input layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the number of capsules in the input layer that feed into this layer.
    /// Each capsule represents a group of neurons that encode a specific entity or feature.
    /// </para>
    /// <para><b>For Beginners:</b> This is how many groups of features the layer receives as input.
    /// 
    /// For example, in an image recognition task:
    /// - Each input capsule might represent a different feature (edges, corners, textures)
    /// - More input capsules mean the network can detect more features
    /// - These features are combined to recognize more complex patterns
    /// </para>
    /// </remarks>
    private readonly int _inputCapsules;

    /// <summary>
    /// The dimension (number of values) of each input capsule vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the dimension of each input capsule vector. Each value in the vector
    /// represents a different property or attribute of the entity that the capsule detects.
    /// </para>
    /// <para><b>For Beginners:</b> This is how much information each input feature group contains.
    /// 
    /// If each input capsule is a feature:
    /// - The dimension is how many aspects of that feature we track
    /// - For instance, a feature might have properties like size, orientation, and position
    /// - More dimensions allow for more detailed feature representation
    /// </para>
    /// </remarks>
    private readonly int _inputCapsuleDimension;

    /// <summary>
    /// The number of classes (output capsules) that this layer can identify.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the number of classes that this layer can identify, which corresponds
    /// to the number of output capsules. For digit recognition, this would typically be 10 (for digits 0-9).
    /// </para>
    /// <para><b>For Beginners:</b> This is how many different categories the layer can classify inputs into.
    /// 
    /// For example:
    /// - In digit recognition, there would be 10 classes (digits 0-9)
    /// - In letter recognition, there might be 26 classes (A-Z)
    /// - Each class gets its own output capsule to represent that category
    /// </para>
    /// </remarks>
    private readonly int _numClasses;

    /// <summary>
    /// The dimension (number of values) of each output capsule vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the dimension of each output capsule vector. The length of this vector
    /// represents the probability of the entity's existence, while its orientation represents the entity's properties.
    /// </para>
    /// <para><b>For Beginners:</b> This is how much detail each output prediction contains.
    /// 
    /// In capsule networks:
    /// - The length of an output vector shows how confident the network is about a prediction
    /// - The direction of the vector encodes properties like position, size, or orientation
    /// - More dimensions allow for capturing more properties about what was detected
    /// </para>
    /// </remarks>
    private readonly int _outputCapsuleDimension;

    /// <summary>
    /// The number of iterations to use in the dynamic routing algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field determines how many iterations of the dynamic routing algorithm to perform.
    /// More iterations can lead to better routing decisions but increase computational cost.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how carefully the network considers connections between features and classes.
    /// 
    /// Think of it like deliberating before making a decision:
    /// - More routing iterations means more careful consideration of all evidence
    /// - With each iteration, the network refines which inputs are important for each output
    /// - Typically 3-5 iterations provides a good balance of accuracy and speed
    /// </para>
    /// </remarks>
    private readonly int _routingIterations;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>true</c> because this layer has trainable parameters (weights).
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that the digit capsule layer supports training through backpropagation.
    /// The layer has trainable weights that are updated during the training process.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you that this layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer can adjust its internal values during training
    /// - It will improve its performance as it sees more data
    /// - It has weights that are updated to make better predictions over time
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="DigitCapsuleLayer{T}"/> class.
    /// </summary>
    /// <param name="inputCapsules">The number of capsules in the input layer.</param>
    /// <param name="inputCapsuleDimension">The dimension of each input capsule vector.</param>
    /// <param name="numClasses">The number of classes (output capsules) that this layer can identify.</param>
    /// <param name="outputCapsuleDimension">The dimension of each output capsule vector.</param>
    /// <param name="routingIterations">The number of iterations to use in the dynamic routing algorithm.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new digit capsule layer with the specified parameters. It initializes the
    /// weight tensor with small random values scaled according to the dimensions of the input and output capsules.
    /// The layer uses a squash activation function, which is specific to capsule networks.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the capsule layer with the specific details it needs.
    /// 
    /// When creating a digit capsule layer, you need to specify:
    /// - How many input feature groups there are (inputCapsules)
    /// - How detailed each input feature is (inputCapsuleDimension)
    /// - How many categories to classify into (numClasses)
    /// - How detailed each output prediction should be (outputCapsuleDimension)
    /// - How carefully to analyze connections between inputs and outputs (routingIterations)
    /// 
    /// For example, to recognize handwritten digits, you might use 10 output classes (digits 0-9)
    /// with a moderate number of routing iterations (3-5) for good performance.
    /// </para>
    /// </remarks>
    public DigitCapsuleLayer(int inputCapsules, int inputCapsuleDimension, int numClasses, int outputCapsuleDimension, int routingIterations)
        : base([inputCapsules, inputCapsuleDimension], [numClasses, outputCapsuleDimension], new SquashActivation<T>() as IActivationFunction<T>)
    {
        _inputCapsules = inputCapsules;
        _inputCapsuleDimension = inputCapsuleDimension;
        _numClasses = numClasses;
        _outputCapsuleDimension = outputCapsuleDimension;
        _routingIterations = routingIterations;
        _weights = new Tensor<T>([inputCapsules, numClasses, inputCapsuleDimension, outputCapsuleDimension]);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes the weight parameters with small random values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the weight tensor with small random values. The values are scaled by a factor
    /// that depends on the input size, which helps in achieving good convergence during training.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the initial connection strengths with random starting values.
    /// 
    /// Before training begins:
    /// - The network needs some starting values for its weights
    /// - Random values give it an unbiased starting point
    /// - The scaling makes sure these values aren't too large or too small
    /// 
    /// It's like shuffling cards before a game - it ensures a fair and random starting point
    /// before the network begins learning.
    /// </para>
    /// </remarks>
    private void InitializeParameters()
    {
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_inputCapsules * _inputCapsuleDimension)));

        // Calculate total elements for flat tensor initialization
        int totalElements = _inputCapsules * _numClasses * _inputCapsuleDimension * _outputCapsuleDimension;

        // Create flat random tensor [0, 1] directly as 1D, shift to [-0.5, 0.5], scale
        var randomTensor = Tensor<T>.CreateRandom(totalElements);
        var halfTensor = new Tensor<T>([totalElements]);
        halfTensor.Fill(NumOps.FromDouble(0.5));
        var shifted = Engine.TensorSubtract(randomTensor, halfTensor);
        var scaled = Engine.TensorMultiplyScalar(shifted, scale);

        // Copy to weights tensor - reshape maintains the same underlying data
        _weights = scaled.Reshape(_weights.Shape);
    }

    /// <summary>
    /// Performs the forward pass of the digit capsule layer using dynamic routing.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after capsule routing.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the digit capsule layer using the dynamic routing algorithm.
    /// It transforms input capsules into predictions for each class, then iteratively refines the routing
    /// coefficients to determine how strongly each input capsule should connect to each output capsule.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the layer makes its predictions based on the input data.
    /// 
    /// The forward pass works in these steps:
    /// 
    /// 1. Transform each input feature into predictions for each possible class
    /// 2. Start with equal connection strengths between all inputs and outputs
    /// 3. For several iterations:
    ///    - Calculate how strongly each input connects to each output
    ///    - Update these connections based on how well inputs agree with outputs
    ///    - Recalculate the output predictions using the updated connections
    /// 
    /// This process is like having experts (input capsules) vote on different outcomes (classes),
    /// then gradually giving more weight to experts who agree with the consensus for each outcome.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Store original shape for any-rank tensor support
        _originalInputShape = input.Shape;
        int rank = input.Shape.Length;

        // Handle any-rank tensor: collapse to 2D for processing
        Tensor<T> processInput;
        int batchSize;

        if (rank == 1)
        {
            // 1D: add batch dim
            batchSize = 1;
            processInput = input.Reshape([1, input.Shape[0]]);
        }
        else if (rank == 2)
        {
            // Standard 2D
            batchSize = input.Shape[0];
            processInput = input;
        }
        else
        {
            // Higher-rank: collapse leading dims into batch
            int flatBatch = 1;
            for (int d = 0; d < rank - 1; d++)
                flatBatch *= input.Shape[d];
            batchSize = flatBatch;
            processInput = input.Reshape([flatBatch, input.Shape[rank - 1]]);
        }

        _lastInput = processInput;

        // Compute predictions u_hat_ij = W_ij * u_i using elementwise multiply + reduce-sum
        // This approach keeps all dimensions aligned and avoids BatchMatMul shape issues

        // Input: [B, I, D_in] -> expand to [B, I, 1, D_in, 1] for broadcasting
        var inputExpanded = input.Reshape([batchSize, _inputCapsules, 1, _inputCapsuleDimension, 1]);

        // Weights: [I, C, D_in, D_out] -> expand to [1, I, C, D_in, D_out] for broadcasting
        var weightsExpanded = _weights.Reshape([1, _inputCapsules, _numClasses, _inputCapsuleDimension, _outputCapsuleDimension]);

        // Elementwise multiply broadcasts to [B, I, C, D_in, D_out]
        var multiplied = Engine.TensorMultiply(weightsExpanded, inputExpanded);

        // Sum over D_in (axis 3) to get predictions: [B, I, C, D_out]
        var predictions = Engine.ReduceSum(multiplied, new[] { 3 }, keepDims: false);

        var couplings = new Tensor<T>([batchSize, _inputCapsules, _numClasses]);
        couplings.Fill(NumOps.Zero);

        var output = new Tensor<T>([batchSize, _numClasses, _outputCapsuleDimension]);

        var softmaxActivation = new SoftmaxActivation<T>();
        for (int iteration = 0; iteration < _routingIterations; iteration++)
        {
            var routingWeights = softmaxActivation.Activate(couplings); // [B,I,C]

            // weightedSum = sum_i routing * predictions
            var routingExpanded = routingWeights.Reshape([batchSize, _inputCapsules, _numClasses, 1]);
            var weightedPred = Engine.TensorMultiply(predictions, routingExpanded);
            var weightedSum = Engine.ReduceSum(weightedPred, new[] { 1 }, keepDims: false); // [B, C, outDim]

            // activation
            var activated = ApplyActivation(weightedSum);
            output = activated;

            if (iteration < _routingIterations - 1)
            {
                // couplings += predictions · output
                var outputExpanded = output.Reshape([batchSize, 1, _numClasses, _outputCapsuleDimension]);
                var dot = Engine.ReduceSum(Engine.TensorMultiply(predictions, outputExpanded), new[] { 3 }, keepDims: false); // [B,I,C]
                couplings = Engine.TensorAdd(couplings, dot);
            }
        }

        _lastOutput = output;
        _lastCouplings = couplings;

        // Restore output to match input rank pattern for any-rank tensor support
        if (_originalInputShape != null && _originalInputShape.Length != 3)
        {
            if (_originalInputShape.Length == 1)
            {
                // Was 1D, return [numClasses * outputCapsuleDimension]
                return output.Reshape([_numClasses * _outputCapsuleDimension]);
            }
            else if (_originalInputShape.Length == 2)
            {
                // Was 2D [batch, features], return [batch, numClasses * outputCapsuleDimension]
                int origBatch = _originalInputShape[0];
                return output.Reshape([origBatch, _numClasses * _outputCapsuleDimension]);
            }
            else
            {
                // Higher-rank: restore leading dimensions
                var newShape = new int[_originalInputShape.Length];
                for (int d = 0; d < _originalInputShape.Length - 1; d++)
                    newShape[d] = _originalInputShape[d];
                newShape[_originalInputShape.Length - 1] = _numClasses * _outputCapsuleDimension;
                return output.Reshape(newShape);
            }
        }

        return output;
    }

    /// <summary>
    /// Performs the backward pass of the digit capsule layer to compute gradients.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass (backpropagation) of the digit capsule layer. It computes
    /// the gradients of the loss with respect to the layer's weights and inputs, which are used to update
    /// the weights during training.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the layer learns from its mistakes during training.
    /// 
    /// The backward pass:
    /// 1. Receives information about how the network's prediction was wrong
    /// 2. Calculates how each weight contributed to this error
    /// 3. Determines how to adjust the weights to reduce the error next time
    /// 4. Passes error information back to previous layers
    /// 
    /// It's like figuring out which ingredients in a recipe need to be adjusted after tasting
    /// the finished dish, then sharing that feedback with those who prepared the ingredients.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Manual backward pass implementation using optimized gradient calculations.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _lastCouplings == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];

        var activationGradient = ApplyActivationDerivative(_lastOutput, outputGradient);

        _weightsGradient = new Tensor<T>(_weights.Shape);
        var inputGradient = new Tensor<T>(_lastInput.Shape);

        var softmaxActivation = new SoftmaxActivation<T>();
        var routingWeights = softmaxActivation.Activate(_lastCouplings);
        var routingWeightsGradient = softmaxActivation.Derivative(_lastCouplings);

        // Tensorized prediction gradients: [B, I, C, outDim]
        var predGrad = new Tensor<T>([batchSize, _inputCapsules, _numClasses, _outputCapsuleDimension]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < _inputCapsules; i++)
            {
                for (int j = 0; j < _numClasses; j++)
                {
                    var pg = activationGradient.SubTensor(b, j).Multiply(routingWeights[b, i, j]);
                    for (int l = 0; l < _outputCapsuleDimension; l++)
                    {
                        predGrad[b, i, j, l] = pg[l];
                    }
                }
            }
        }

        // Weight gradients: sum over batch of outer(inputCapsule, predGrad)
        for (int i = 0; i < _inputCapsules; i++)
        {
            for (int j = 0; j < _numClasses; j++)
            {
                var accum = new Tensor<T>([_inputCapsuleDimension, _outputCapsuleDimension]);
                for (int b = 0; b < batchSize; b++)
                {
                    var inputCapsule = _lastInput.SubTensor(b, i).Reshape([_inputCapsuleDimension, 1]);
                    var pg = predGrad.SubTensor(b, i, j).Reshape([1, _outputCapsuleDimension]);
                    var outer = Engine.TensorMatMul(inputCapsule, pg);
                    accum = Engine.TensorAdd(accum, outer);
                }
                for (int k = 0; k < _inputCapsuleDimension; k++)
                    for (int l = 0; l < _outputCapsuleDimension; l++)
                        _weightsGradient[i, j, k, l] = accum[k, l];
            }
        }

        // Input gradient accumulation
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < _inputCapsules; i++)
            {
                var gradVec = new Tensor<T>([_inputCapsuleDimension, 1]);
                for (int j = 0; j < _numClasses; j++)
                {
                    var weightsMat = _weights.SubTensor(i, j);
                    var pg = predGrad.SubTensor(b, i, j).Reshape([_outputCapsuleDimension, 1]);
                    gradVec = Engine.TensorAdd(gradVec, Engine.TensorMatMul(weightsMat, pg));

                    // coupling gradient contribution
                    var outputCapsule = _lastOutput.SubTensor(b, j);
                    var actGradCapsule = activationGradient.SubTensor(b, j);
                    T dot = NumOps.Zero;
                    for (int l = 0; l < _outputCapsuleDimension; l++)
                    {
                        dot = NumOps.Add(dot, NumOps.Multiply(outputCapsule[l], actGradCapsule[l]));
                    }
                    dot = NumOps.Multiply(dot, routingWeightsGradient[b, i, j]);
                    var couplingVec = new Tensor<T>([_outputCapsuleDimension, 1]);
                    couplingVec.Fill(dot);
                    gradVec = Engine.TensorAdd(gradVec, Engine.TensorMatMul(weightsMat, couplingVec));
                }

                for (int k = 0; k < _inputCapsuleDimension; k++)
                {
                    inputGradient[b, i, k] = gradVec[k, 0];
                }
            }
        }

        // Restore gradient to original input shape for any-rank support
        if (_originalInputShape != null && _originalInputShape.Length != 3)
        {
            return inputGradient.Reshape(_originalInputShape);
        }

        return inputGradient;
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation with unrolled routing.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // 1. Create variables
        var inputNode = Autodiff.TensorOperations<T>.Variable(_lastInput, "input", requiresGradient: true);
        var weightsNode = Autodiff.TensorOperations<T>.Variable(_weights, "weights", requiresGradient: true);

        int batchSize = _lastInput.Shape[0];
        int inputCapsules = _inputCapsules;
        int numClasses = _numClasses;
        int inputDim = _inputCapsuleDimension;
        int outputDim = _outputCapsuleDimension;

        // 2. Compute Predictions: input @ weights
        // Input: [B, I, D_in] -> Reshape to [B, I, 1, D_in] to broadcast over C
        var inputReshaped = Autodiff.TensorOperations<T>.Reshape(inputNode, batchSize, inputCapsules, 1, 1, inputDim);

        // Weights: [I, C, D_in, D_out] -> Reshape to [1, I, C, D_in, D_out] to broadcast over B
        var weightsReshaped = Autodiff.TensorOperations<T>.Reshape(weightsNode, 1, inputCapsules, numClasses, inputDim, outputDim);

        // Result: [B, I, C, 1, D_out] (Batched MatMul handles broadcasting [B, I, 1] vs [1, I, C])
        var predictionsRaw = Autodiff.TensorOperations<T>.MatrixMultiply(inputReshaped, weightsReshaped);

        // Reshape to [B, I, C, D_out]
        var predictions = Autodiff.TensorOperations<T>.Reshape(predictionsRaw, batchSize, inputCapsules, numClasses, outputDim);

        // 3. Dynamic Routing
        // Initialize couplings to zero [B, I, C]
        var couplingsTensor = new Tensor<T>(new int[] { batchSize, inputCapsules, numClasses });
        couplingsTensor.Fill(NumOps.Zero);
        var couplings = Autodiff.TensorOperations<T>.Constant(couplingsTensor, "couplings");

        Autodiff.ComputationNode<T> output = predictions; // Placeholder

        for (int iter = 0; iter < _routingIterations; iter++)
        {
            // Softmax over classes (axis 2) -> [B, I, C]
            var routingWeights = Autodiff.TensorOperations<T>.Softmax(couplings, axis: 2);

            // Reshape routing weights to [B, I, C, 1] for broadcasting
            var routingWeightsBroad = Autodiff.TensorOperations<T>.Reshape(routingWeights, batchSize, inputCapsules, numClasses, 1);

            // Weighted predictions: predictions * routing
            var weightedPredictions = Autodiff.TensorOperations<T>.ElementwiseMultiply(predictions, routingWeightsBroad);

            // Sum over input capsules (axis 1) -> [B, C, D_out]
            var weightedSum = Autodiff.TensorOperations<T>.Sum(weightedPredictions, new int[] { 1 }, keepDims: false);

            // Squash activation
            // v = ||s||^2 / (1 + ||s||^2) * s / ||s||
            // ||s||^2 = sum(s^2, axis=-1)
            var s2 = Autodiff.TensorOperations<T>.Square(weightedSum);
            var normSq = Autodiff.TensorOperations<T>.Sum(s2, new int[] { 2 }, keepDims: true); // [B, C, 1]
            var norm = Autodiff.TensorOperations<T>.Sqrt(normSq);

            var one = Autodiff.TensorOperations<T>.Constant(Tensor<T>.CreateDefault(new int[] { 1 }, NumOps.One));
            var scale = Autodiff.TensorOperations<T>.Divide(normSq, Autodiff.TensorOperations<T>.Add(one, normSq));
            var unitVec = Autodiff.TensorOperations<T>.Divide(weightedSum, norm);

            // output = scale * unitVec
            output = Autodiff.TensorOperations<T>.ElementwiseMultiply(scale, unitVec);

            // Update couplings (agreement)
            if (iter < _routingIterations - 1)
            {
                // Agreement = predictions . output
                // predictions [B, I, C, D], output [B, C, D] -> reshape output to [B, 1, C, D]
                var outputBroad = Autodiff.TensorOperations<T>.Reshape(output, batchSize, 1, numClasses, outputDim);

                // Elementwise multiply -> [B, I, C, D]
                var agreementRaw = Autodiff.TensorOperations<T>.ElementwiseMultiply(predictions, outputBroad);

                // Sum over D (axis 3) -> [B, I, C]
                var agreement = Autodiff.TensorOperations<T>.Sum(agreementRaw, new int[] { 3 }, keepDims: false);

                couplings = Autodiff.TensorOperations<T>.Add(couplings, agreement);
            }
        }

        // 4. Set Gradient
        output.Gradient = outputGradient;

        // 5. Backward
        output.Backward();

        // 6. Store Gradients
        _weightsGradient = weightsNode.Gradient;

        var inputGradient = inputNode.Gradient ?? throw new InvalidOperationException("Gradient computation failed.");

        // Restore gradient to original input shape for any-rank support
        if (_originalInputShape != null && _originalInputShape.Length != 3)
        {
            return inputGradient.Reshape(_originalInputShape);
        }

        return inputGradient;
    }


    /// <summary>
    /// Updates the layer's weights using the calculated gradients and the specified learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <exception cref="InvalidOperationException">Thrown when update is called before backward.</exception>
    /// <remarks>
    /// <para>
    /// This method updates the layer's weights based on the gradients calculated during the backward pass.
    /// The learning rate determines the size of the parameter updates. Smaller learning rates lead to more
    /// stable but slower training, while larger learning rates can lead to faster but potentially unstable training.
    /// </para>
    /// <para><b>For Beginners:</b> This method actually adjusts the weights based on what was learned.
    /// 
    /// After figuring out what changes need to be made:
    /// - The network adjusts each weight by a small amount
    /// - The learning rate controls how big this adjustment is
    /// - Too small: learning happens very slowly
    /// - Too large: learning becomes unstable
    /// 
    /// It's like adjusting a recipe based on taste - you don't want to add too much salt at once,
    /// but you also don't want to add just one grain at a time.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_weightsGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _weights = _weights.Subtract(_weightsGradient.Multiply(learningRate));
    }

    /// <summary>
    /// Gets all trainable parameters of the layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters (weights) of the layer as a single vector.
    /// This is useful for optimization algorithms that operate on all parameters at once, or for saving
    /// and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the layer's learnable values into a single list.
    /// 
    /// The parameters:
    /// - Are all the weight values that the network learns
    /// - Are flattened into a single long list (vector)
    /// - Can be saved to disk or loaded from a previous training session
    /// 
    /// This allows you to:
    /// - Save a trained model for later use
    /// - Transfer a model's knowledge to another identical model
    /// - Share trained models with others
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Use ToArray() for production-grade parameter extraction
        return new Vector<T>(_weights.ToArray());
    }

    /// <summary>
    /// Sets the trainable parameters of the layer from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets all trainable parameters (weights) of the layer from a single vector.
    /// This is useful for loading saved model weights or for implementing optimization algorithms
    /// that operate on all parameters at once.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the layer's learnable values from a provided list.
    /// 
    /// When setting parameters:
    /// - The input must be a vector with the exact right length
    /// - Each value in the vector corresponds to a specific weight in the layer
    /// - This allows loading previously trained weights
    /// 
    /// Use cases include:
    /// - Restoring a saved model
    /// - Using pre-trained weights
    /// - Testing specific weight configurations
    /// 
    /// The method throws an error if the provided vector doesn't contain exactly the right number of values.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != _weights.Length)
        {
            throw new ArgumentException($"Expected {_weights.Length} parameters, but got {parameters.Length}");
        }

        // Use Tensor.FromVector for production-grade parameter setting
        _weights = Tensor<T>.FromVector(parameters, _weights.Shape);
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the layer by clearing all cached values from forward
    /// and backward passes. This is useful when starting to process a new batch of data or when
    /// implementing stateful recurrent networks.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - All stored inputs, outputs, and intermediate values are cleared
    /// - The layer forgets previous data it processed
    /// - This prepares it for processing new, unrelated data
    /// 
    /// It's like wiping a whiteboard clean before starting a new calculation.
    /// This ensures that information from one batch of data doesn't affect the
    /// processing of another, unrelated batch.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward and backward passes
        _lastInput = null;
        _lastOutput = null;
        _lastCouplings = null;
        _weightsGradient = null;
    }

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (inputNodes.Count == 0)
            throw new ArgumentException("At least one input node is required.", nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        var input = inputNodes[0];

        // Create weight tensor as constant node [inputCapsules, numClasses, inputCapsuleDimension, outputCapsuleDimension]
        var weightsTensor = new Tensor<T>(
            new[] { _inputCapsules, _numClasses, _inputCapsuleDimension, _outputCapsuleDimension },
            _weights.ToVector());
        var weightsNode = TensorOperations<T>.Constant(weightsTensor, "DigitCapsWeights");

        // Transform input capsules to predictions for each class
        // For each input capsule i and class j: predictions[i,j] = input[i] @ weights[i,j]
        var predictions = TensorOperations<T>.MatrixMultiply(input, weightsNode);

        // Initialize coupling coefficients to zero using Fill
        var couplingsTensor = new Tensor<T>(new[] { _inputCapsules, _numClasses });
        couplingsTensor.Fill(NumOps.Zero);
        var couplings = TensorOperations<T>.Constant(couplingsTensor, "InitialCouplings");

        ComputationNode<T> output = predictions;

        // Unroll routing iterations
        for (int iter = 0; iter < _routingIterations; iter++)
        {
            // Apply softmax to couplings along numClasses dimension
            var routingWeights = TensorOperations<T>.Softmax(couplings, axis: 1);

            // Weighted sum for each class: output[j] = sum_i(routingWeights[i,j] * predictions[i,j])
            var weighted = TensorOperations<T>.ElementwiseMultiply(predictions, routingWeights);
            var weightedSum = TensorOperations<T>.Sum(weighted, [0]); // Sum over inputCapsules

            // Apply squash activation: v = ||s||^2 / (1 + ||s||^2) * s / ||s||
            var squaredNorm = TensorOperations<T>.Sum(TensorOperations<T>.Square(weightedSum), [1]);
            var oneTensor = new Tensor<T>(new[] { 1 }, new Vector<T>(new[] { NumOps.One }));
            var oneNode = TensorOperations<T>.Constant(oneTensor, "One");
            var normPlusOne = TensorOperations<T>.Add(squaredNorm, oneNode);
            var scaleFactor = TensorOperations<T>.Divide(squaredNorm, normPlusOne);
            var norm = TensorOperations<T>.Sqrt(squaredNorm);
            var normalizedVec = TensorOperations<T>.Divide(weightedSum, norm);
            output = TensorOperations<T>.ElementwiseMultiply(normalizedVec, scaleFactor);

            // Update couplings if not last iteration
            if (iter < _routingIterations - 1)
            {
                // Agreement: dot product between predictions and output for each input capsule/class pair
                var agreement = TensorOperations<T>.Sum(
                    TensorOperations<T>.ElementwiseMultiply(predictions, output), [2]);
                couplings = TensorOperations<T>.Add(couplings, agreement);
            }
        }

        return output;
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    /// <value>
    /// <c>true</c> because DigitCapsuleLayer uses dynamic routing with a fixed number of iterations
    /// that can be unrolled into a static computation graph.
    /// </value>
    public override bool SupportsJitCompilation => true;

}

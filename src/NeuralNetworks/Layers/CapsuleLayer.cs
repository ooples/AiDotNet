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
public class CapsuleLayer<T> : LayerBase<T>, IAuxiliaryLossLayer<T>
{
    /// <summary>
    /// Gets or sets whether auxiliary loss (routing entropy regularization) should be used during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Routing entropy regularization encourages diversity in the routing coefficients by penalizing
    /// low entropy distributions. This prevents routing from becoming too deterministic and helps
    /// the capsule layer learn more robust features.
    /// </para>
    /// <para><b>For Beginners:</b> Routing regularization helps capsules make better decisions.
    ///
    /// In capsule networks:
    /// - Routing coefficients decide how much information flows from lower to higher capsules
    /// - If routing becomes too "certain" (all weight on one capsule), it might miss important patterns
    /// - Entropy regularization encourages routing to consider multiple options
    ///
    /// Think of it like this:
    /// - Without regularization: "This is 100% a face, ignore everything else"
    /// - With regularization: "This is probably a face (80%), but could be other things (20%)"
    ///
    /// This helps the network:
    /// - Learn more robust features
    /// - Avoid overconfidence
    /// - Generalize better to new examples
    /// </para>
    /// </remarks>
    public bool UseAuxiliaryLoss { get; set; } = false;

    /// <summary>
    /// Gets or sets the weight for the routing entropy auxiliary loss.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This weight controls how much the routing entropy regularization contributes to the total loss.
    /// The total loss is: main_loss + (auxiliary_weight * entropy_loss).
    /// Typical values range from 0.001 to 0.01.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much the network should encourage diverse routing.
    ///
    /// The weight determines the balance between:
    /// - Task accuracy (main loss)
    /// - Routing diversity (entropy loss)
    ///
    /// Common values:
    /// - 0.005 (default): Balanced routing diversity
    /// - 0.001-0.003: Light diversity enforcement
    /// - 0.008-0.01: Strong diversity enforcement
    ///
    /// Higher values make routing more diverse but might reduce task performance.
    /// Lower values allow more deterministic routing but might lead to overconfidence.
    /// </para>
    /// </remarks>
    public T AuxiliaryLossWeight { get; set; }

    private T _lastRoutingEntropyLoss;

    private readonly int _numCapsules;
    private readonly int _capsuleDimension;
    private readonly int _numRoutingIterations;
    private Tensor<T> _transformationMatrix;
    private Tensor<T> _bias;
    private Tensor<T>? _transformationMatrixGradient;
    private Tensor<T>? _biasGradient;
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;
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
    /// <param name="engine">The computation engine for vectorized operations. Defaults to CPU if not specified.</param>
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
        if (numRoutingIterations < 1)
        {
            throw new ArgumentException("Number of routing iterations must be at least 1.", nameof(numRoutingIterations));
        }

        AuxiliaryLossWeight = NumOps.FromDouble(0.005);
        _lastRoutingEntropyLoss = NumOps.Zero;

        _numCapsules = numCapsules;
        _capsuleDimension = capsuleDimension;
        _numRoutingIterations = numRoutingIterations;

        _transformationMatrix = new Tensor<T>([inputCapsules, inputDimension, numCapsules, capsuleDimension]);
        _bias = new Tensor<T>([numCapsules * capsuleDimension]);

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

        // Initialize bias to zero using Fill
        _bias.Fill(NumOps.Zero);
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
        // For multi-dimensional tensors, create random and apply transformation
        int totalElements = tensor.Shape.Aggregate(1, (acc, dim) => acc * dim);

        // Create a flat random tensor [0, 1]
        var randomTensor = Tensor<T>.CreateRandom(totalElements, 1).Reshape([totalElements]);

        // Shift to [-0.5, 0.5] range: random - 0.5
        var halfTensor = new Tensor<T>([totalElements]);
        halfTensor.Fill(NumOps.FromDouble(0.5));
        var shifted = Engine.TensorSubtract(randomTensor, halfTensor);

        // Scale by the scale factor
        var scaled = Engine.TensorMultiplyScalar(shifted, scale);

        // Copy values to original tensor using flat index setter to preserve tensor.Shape
        // Note: Array.Copy into ToArray() doesn't work because ToArray() returns a copy
        for (int i = 0; i < totalElements; i++)
        {
            tensor[i] = scaled[i];
        }
    }

    /// <summary>
    /// Computes the auxiliary loss for routing entropy regularization.
    /// </summary>
    /// <returns>The computed routing entropy auxiliary loss.</returns>
    /// <remarks>
    /// <para>
    /// This method computes the entropy of the routing coefficients. Low entropy means the routing
    /// is very deterministic (concentrating on one capsule), while high entropy means it's more
    /// distributed across multiple capsules. We penalize low entropy to encourage diverse routing.
    /// Entropy: H = -Σ(p * log(p)) where p are the routing coefficients.
    /// We use negative entropy as loss since we want to maximize entropy (minimize -H).
    /// </para>
    /// <para><b>For Beginners:</b> This calculates how diverse the routing decisions are.
    ///
    /// Routing entropy works by:
    /// 1. Looking at the routing coefficients (how information flows between capsules)
    /// 2. Measuring how "spread out" these coefficients are
    /// 3. Penalizing routing that's too concentrated on one capsule
    /// 4. Encouraging routing that considers multiple capsules
    ///
    /// Entropy is a measure of uncertainty/diversity:
    /// - Low entropy: Very certain, concentrated (e.g., [0.99, 0.01, 0.00])
    /// - High entropy: Uncertain, diverse (e.g., [0.33, 0.33, 0.34])
    ///
    /// By encouraging higher entropy, we prevent the network from becoming overconfident
    /// and help it learn more robust features that work in different situations.
    /// </para>
    /// </remarks>
    public T ComputeAuxiliaryLoss()
    {
        if (!UseAuxiliaryLoss || _lastCouplingCoefficients == null)
        {
            _lastRoutingEntropyLoss = NumOps.Zero;
            return NumOps.Zero;
        }

        // VECTORIZED: Compute negative entropy of routing coefficients using tensor ops
        // Entropy: H = -Σ(p * log(p)) per distribution

        // Clamp values to avoid log(0)
        T epsilon = NumOps.FromDouble(1e-10);
        var epsilonTensor = new Tensor<T>(_lastCouplingCoefficients.Shape);
        epsilonTensor.Fill(epsilon);

        // p_clamped = max(p, epsilon) - element-wise
        var pClamped = Engine.TensorMax(_lastCouplingCoefficients, epsilonTensor);

        // log_p = log(p_clamped)
        var logP = Engine.TensorLog(pClamped);

        // p * log(p)
        var pLogP = Engine.TensorMultiply(_lastCouplingCoefficients, logP);

        // Sum all and negate to get negative entropy
        var sumPLogP = Engine.ReduceSum(pLogP, Enumerable.Range(0, pLogP.Shape.Length).ToArray(), keepDims: false);
        T totalPLogP = sumPLogP.GetFlat(0);

        // Average across all distributions (total elements / distribution size)
        int flatSize = _lastCouplingCoefficients.Shape.Aggregate(1, (acc, dim) => acc * dim);
        int distributionSize = _numCapsules;
        int numDistributions = flatSize / distributionSize;

        T totalNegativeEntropy;
        if (numDistributions > 0)
        {
            totalNegativeEntropy = NumOps.Divide(NumOps.Negate(totalPLogP), NumOps.FromDouble(numDistributions));
        }
        else
        {
            totalNegativeEntropy = NumOps.Zero;
        }

        // Store unweighted loss for diagnostics
        _lastRoutingEntropyLoss = totalNegativeEntropy;

        // Return weighted auxiliary loss
        return NumOps.Multiply(AuxiliaryLossWeight, totalNegativeEntropy);
    }

    /// <summary>
    /// Gets diagnostic information about the routing entropy auxiliary loss.
    /// </summary>
    /// <returns>A dictionary containing diagnostic information about routing regularization.</returns>
    /// <remarks>
    /// <para>
    /// This method returns detailed diagnostics about the routing entropy regularization, including
    /// the computed entropy loss, weight applied, and whether the feature is enabled.
    /// This information is useful for monitoring training progress and debugging.
    /// </para>
    /// <para><b>For Beginners:</b> This provides information about how routing regularization is working.
    ///
    /// The diagnostics include:
    /// - Total routing entropy loss (how concentrated routing is)
    /// - Weight applied to the entropy loss
    /// - Whether routing regularization is enabled
    /// - Number of routing iterations being used
    ///
    /// This helps you:
    /// - Monitor if routing is becoming too deterministic
    /// - Debug issues with capsule layer learning
    /// - Understand the impact of entropy regularization on routing
    ///
    /// You can use this information to adjust the auxiliary loss weight or
    /// routing iterations for better results.
    /// </para>
    /// </remarks>
    public Dictionary<string, string> GetAuxiliaryLossDiagnostics()
    {
        return new Dictionary<string, string>
        {
            { "TotalRoutingEntropyLoss", $"{_lastRoutingEntropyLoss}" },
            { "EntropyWeight", $"{AuxiliaryLossWeight}" },
            { "UseRoutingRegularization", UseAuxiliaryLoss.ToString() },
            { "NumRoutingIterations", _numRoutingIterations.ToString() },
            { "RoutingCoefficientsCached", (_lastCouplingCoefficients != null).ToString() }
        };
    }

    /// <summary>
    /// Gets diagnostic information about this component's state and behavior.
    /// Overrides <see cref="LayerBase{T}.GetDiagnostics"/> to include auxiliary loss diagnostics.
    /// </summary>
    /// <returns>
    /// A dictionary containing diagnostic metrics including both base layer diagnostics and
    /// auxiliary loss diagnostics from <see cref="GetAuxiliaryLossDiagnostics"/>.
    /// </returns>
    public override Dictionary<string, string> GetDiagnostics()
    {
        var diagnostics = base.GetDiagnostics();

        // Merge auxiliary loss diagnostics
        var auxDiagnostics = GetAuxiliaryLossDiagnostics();
        foreach (var kvp in auxDiagnostics)
        {
            diagnostics[kvp.Key] = kvp.Value;
        }

        return diagnostics;
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
        // Store original shape for any-rank tensor support
        _originalInputShape = input.Shape;
        int rank = input.Shape.Length;

        // Handle any-rank tensor: need at least 2D [capsules, dim]
        Tensor<T> input3D;
        int batchSize;
        int inputCapsules;
        int inputDimension;

        if (rank == 2)
        {
            // 2D: [capsules, dim] -> add batch dim
            batchSize = 1;
            inputCapsules = input.Shape[0];
            inputDimension = input.Shape[1];
            input3D = input.Reshape([1, inputCapsules, inputDimension]);
        }
        else if (rank == 3)
        {
            // Standard 3D: [batch, capsules, dim]
            batchSize = input.Shape[0];
            inputCapsules = input.Shape[1];
            inputDimension = input.Shape[2];
            input3D = input;
        }
        else if (rank > 3)
        {
            // Higher-rank: collapse leading dims into batch
            int flatBatch = 1;
            for (int d = 0; d < rank - 2; d++)
                flatBatch *= input.Shape[d];
            batchSize = flatBatch;
            inputCapsules = input.Shape[rank - 2];
            inputDimension = input.Shape[rank - 1];
            input3D = input.Reshape([flatBatch, inputCapsules, inputDimension]);
        }
        else
        {
            throw new ArgumentException($"CapsuleLayer requires at least 2D input, got {rank}D");
        }

        _lastInput = input3D;

        // Reshape input for matrix multiplication
        var reshapedInput = input3D.Reshape(batchSize * inputCapsules, inputDimension);

        // Perform transformation
        var transformedInput = reshapedInput.Multiply(_transformationMatrix);
        transformedInput = transformedInput.Reshape(batchSize, inputCapsules, _numCapsules, _capsuleDimension);

        // Initialize coupling coefficients
        var couplingCoefficients = new Tensor<T>([batchSize, inputCapsules, _numCapsules]);
        couplingCoefficients.Fill(NumOps.FromDouble(1.0 / _numCapsules));

        // Declare output tensor outside the loop
        Tensor<T>? output = null;

        // Perform dynamic routing
        for (int i = 0; i < _numRoutingIterations; i++)
        {
            // === FULLY VECTORIZED Weighted Sum using tensor operations ===
            // transformedInput: [batchSize, inputCapsules, numCapsules, capsuleDimension]
            // couplingCoefficients: [batchSize, inputCapsules, numCapsules]
            // weightedSum: [batchSize, numCapsules, capsuleDimension]

            // Reshape coupling coefficients to broadcast: [batchSize, inputCapsules, numCapsules, 1]
            var coefExpanded = couplingCoefficients.Reshape([batchSize, inputCapsules, _numCapsules, 1]);

            // Element-wise multiply to weight the capsules
            var weighted = Engine.TensorMultiply(transformedInput, coefExpanded);

            // Sum over input capsules (axis 1) to get weighted sum: [batchSize, numCapsules, capsuleDimension]
            var weightedSum = Engine.ReduceSum(weighted, new[] { 1 }, keepDims: false);

            // === VECTORIZED Bias Addition ===
            // Reshape bias from [numCapsules * capsuleDimension] to [1, numCapsules, capsuleDimension]
            var biasReshaped = _bias.Reshape([1, _numCapsules, _capsuleDimension]);
            weightedSum = Engine.TensorAdd(weightedSum, biasReshaped);

            // Apply squash activation
            output = ApplyActivation(weightedSum);

            // Update coupling coefficients
            if (i < _numRoutingIterations - 1)
            {
                // === FULLY VECTORIZED Agreement Calculation ===
                // Agreement = dot product between transformedInput and output for each (batch, inputCapsule, outputCapsule)
                // transformedInput: [batchSize, inputCapsules, numCapsules, capsuleDimension]
                // output: [batchSize, numCapsules, capsuleDimension]
                // Need to compute: agreement[b, j, k] = sum_d(transformedInput[b, j, k, d] * output[b, k, d])

                // Reshape output to broadcast: [batchSize, 1, numCapsules, capsuleDimension]
                var outputExpanded = output.Reshape([batchSize, 1, _numCapsules, _capsuleDimension]);

                // Element-wise multiply
                var agreementProduct = Engine.TensorMultiply(transformedInput, outputExpanded);

                // Sum over capsule dimension (axis 3) to get agreement: [batchSize, inputCapsules, numCapsules]
                var agreement = Engine.ReduceSum(agreementProduct, new[] { 3 }, keepDims: false);

                // Update coupling coefficients
                couplingCoefficients = Engine.TensorAdd(couplingCoefficients, agreement);
                couplingCoefficients = ApplySoftmax(couplingCoefficients);
            }
        }

        // output is guaranteed to be non-null because _numRoutingIterations is validated to be >= 1
        if (output == null)
            throw new InvalidOperationException("Output tensor was not initialized during forward pass.");

        // Restore original batch dimensions for any-rank support
        if (_originalInputShape != null && _originalInputShape.Length > 3)
        {
            // Output shape: [...leadingDims, numCapsules, capsuleDimension]
            int[] newShape = new int[_originalInputShape.Length];
            for (int d = 0; d < _originalInputShape.Length - 2; d++)
                newShape[d] = _originalInputShape[d];
            newShape[_originalInputShape.Length - 2] = _numCapsules;
            newShape[_originalInputShape.Length - 1] = _capsuleDimension;
            output = output!.Reshape(newShape);
        }
        else if (_originalInputShape != null && _originalInputShape.Length == 2)
        {
            // 2D input -> 2D output (remove batch dim)
            output = output!.Reshape([_numCapsules, _capsuleDimension]);
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
        if (_lastInput == null || _lastOutput == null || _lastCouplingCoefficients == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int inputCapsules = _lastInput.Shape[1];
        int inputDimension = _lastInput.Shape[2];

        var activationGradient = ApplyActivationDerivative(_lastOutput, outputGradient);

        // === FULLY VECTORIZED Gradient Computation ===

        // outputGradient: [batchSize, numCapsules, capsuleDimension]

        // Bias gradient: sum over batch dimension
        // Reshape outputGradient to [batchSize, numCapsules * capsuleDimension]
        var gradReshaped = outputGradient.Reshape([batchSize, _numCapsules * _capsuleDimension]);
        // Sum over batch: [numCapsules * capsuleDimension]
        _biasGradient = Engine.ReduceSum(gradReshaped, new[] { 0 }, keepDims: false);

        // Transformation matrix gradient:
        // grad_W[i, k, j, d] = sum_b(lastInput[b, i, k] * coupling[b, i, j] * outputGrad[b, j, d])
        // This requires outer product operations

        // _lastInput: [batchSize, inputCapsules, inputDimension]
        // _lastCouplingCoefficients: [batchSize, inputCapsules, numCapsules]
        // outputGradient: [batchSize, numCapsules, capsuleDimension]

        // Expand dimensions for broadcasting:
        // input: [batchSize, inputCapsules, inputDimension, 1, 1]
        // coef: [batchSize, inputCapsules, 1, numCapsules, 1]
        // grad: [batchSize, 1, 1, numCapsules, capsuleDimension]

        var inputExpanded = _lastInput.Reshape([batchSize, inputCapsules, inputDimension, 1, 1]);
        var coefExpanded = _lastCouplingCoefficients.Reshape([batchSize, inputCapsules, 1, _numCapsules, 1]);
        var gradExpanded = outputGradient.Reshape([batchSize, 1, 1, _numCapsules, _capsuleDimension]);

        // Element-wise multiply all together
        var inputCoef = Engine.TensorMultiply(inputExpanded, coefExpanded);
        var gradProduct = Engine.TensorMultiply(inputCoef, gradExpanded);

        // Sum over batch dimension to get transformation gradient: [inputCapsules, inputDimension, numCapsules, capsuleDimension]
        _transformationMatrixGradient = Engine.ReduceSum(gradProduct, new[] { 0 }, keepDims: false);

        // Input gradient:
        // grad_input[b, i, k] = sum_j,d(coupling[b, i, j] * outputGrad[b, j, d] * W[i, k, j, d])
        // This is: (coupling * outputGrad) @ W^T summed appropriately

        // Reshape for computation:
        // coupling: [batchSize, inputCapsules, numCapsules, 1]
        // grad: [batchSize, 1, numCapsules, capsuleDimension]
        // W: [inputCapsules, inputDimension, numCapsules, capsuleDimension]

        var coefForInput = _lastCouplingCoefficients.Reshape([batchSize, inputCapsules, _numCapsules, 1]);
        var gradForInput = outputGradient.Reshape([batchSize, 1, _numCapsules, _capsuleDimension]);

        // Multiply coupling with gradient
        var coefGrad = Engine.TensorMultiply(coefForInput, gradForInput); // [batchSize, inputCapsules, numCapsules, capsuleDimension]

        // Now multiply with transformation matrix and sum
        // W: [inputCapsules, inputDimension, numCapsules, capsuleDimension]
        // coefGrad: [batchSize, inputCapsules, numCapsules, capsuleDimension]
        // Need: sum over j,d of (coefGrad[b, i, j, d] * W[i, k, j, d])

        // Reshape coefGrad to [batchSize, inputCapsules, 1, numCapsules, capsuleDimension]
        var coefGradExpanded = coefGrad.Reshape([batchSize, inputCapsules, 1, _numCapsules, _capsuleDimension]);
        // Reshape W to [1, inputCapsules, inputDimension, numCapsules, capsuleDimension]
        var wExpanded = _transformationMatrix.Reshape([1, inputCapsules, inputDimension, _numCapsules, _capsuleDimension]);

        // Element-wise multiply
        var inputGradProduct = Engine.TensorMultiply(coefGradExpanded, wExpanded);

        // Sum over numCapsules and capsuleDimension: [batchSize, inputCapsules, inputDimension]
        var inputGradient = Engine.ReduceSum(inputGradProduct, new[] { 3, 4 }, keepDims: false);

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
        var weightsNode = Autodiff.TensorOperations<T>.Variable(_transformationMatrix, "weights", requiresGradient: true);
        var biasNode = Autodiff.TensorOperations<T>.Variable(_bias, "bias", requiresGradient: true);

        int batchSize = _lastInput.Shape[0];
        int inputCapsules = _lastInput.Shape[1];
        int inputDim = _lastInput.Shape[2];
        int numCapsules = _numCapsules;
        int capsuleDim = _capsuleDimension;

        // 2. Prepare Weights: [I, D_in, O, D_out] -> [I, O, D_in, D_out]
        var weightsPermuted = Autodiff.TensorOperations<T>.Permute(weightsNode, 0, 2, 1, 3);

        // 3. Compute Predictions: input @ weights
        // Input: [B, I, D_in] -> [B, I, 1, D_in]
        var inputReshaped = Autodiff.TensorOperations<T>.Reshape(inputNode, batchSize, inputCapsules, 1, inputDim);

        // Weights: [I, O, D_in, D_out] -> [1, I, O, D_in, D_out]
        var weightsReshaped = Autodiff.TensorOperations<T>.Reshape(weightsPermuted, 1, inputCapsules, numCapsules, inputDim, capsuleDim);

        // Result: [B, I, O, 1, D_out]
        var predictionsRaw = Autodiff.TensorOperations<T>.MatrixMultiply(inputReshaped, weightsReshaped);

        // Reshape to [B, I, O, D_out]
        var predictions = Autodiff.TensorOperations<T>.Reshape(predictionsRaw, batchSize, inputCapsules, numCapsules, capsuleDim);

        // 4. Dynamic Routing
        // Initialize couplings to uniform (1/O) as per Forward, but Constant
        // Note: Forward used Fill(1.0/O). BackwardManual used Softmax(0)=Uniform?
        // Standard CapsNet starts with 0 logits (softmax(0) = uniform).
        // Forward uses Fill(1.0/O) which sums to 1.
        // If we use Softmax in loop, we should start with 0 logits.
        // If Forward uses fixed coefficients, we should match.
        // Forward: `couplingCoefficients.Fill(NumOps.FromDouble(1.0 / _numCapsules));`
        // Loop starts. `weighted = ... * coupling`.
        // Then `Update coupling`.
        // It does NOT Softmax the initial couplings in the first iteration?
        // Let's check Forward logic carefully.
        // "Initialize coupling coefficients... Perform dynamic routing... for (int i = 0; i < _numRoutingIterations; i++)"
        // Inside loop: `var weighted = ... couplingCoefficients ...`
        // `if (i < ... - 1) { ... couplingCoefficients = ApplySoftmax(couplingCoefficients); }`
        // It applies Softmax AFTER update.
        // So initial coefficients are used AS IS.
        // 1/N sums to 1. So it works.
        // But for autodiff, if we want to learn routing, we usually softmax logits.
        // Here we have fixed initial values.

        // We'll create a Constant node for initial couplings.
        var couplingsTensor = new Tensor<T>(new int[] { batchSize, inputCapsules, numCapsules });
        couplingsTensor.Fill(NumOps.FromDouble(1.0 / numCapsules));
        var couplings = Autodiff.TensorOperations<T>.Constant(couplingsTensor, "couplings");

        Autodiff.ComputationNode<T> output = predictions; // Placeholder

        for (int iter = 0; iter < _numRoutingIterations; iter++)
        {
            // Note: Forward doesn't Softmax at start of loop, only at end of previous.
            // So use 'couplings' directly.

            // Reshape couplings to [B, I, O, 1]
            var couplingsBroad = Autodiff.TensorOperations<T>.Reshape(couplings, batchSize, inputCapsules, numCapsules, 1);

            // Weighted predictions
            var weightedPredictions = Autodiff.TensorOperations<T>.ElementwiseMultiply(predictions, couplingsBroad);

            // Sum over input capsules (axis 1) -> [B, O, D_out]
            var weightedSum = Autodiff.TensorOperations<T>.Sum(weightedPredictions, new int[] { 1 }, keepDims: false);

            // Add Bias
            // Bias [O * D]. Reshape to [1, O, D] for broadcast.
            // Wait, _bias is flattened?
            // Constructor: _bias = new Tensor<T>([numCapsules * capsuleDimension]);
            // We need to reshape it to [numCapsules, capsuleDimension] to match weightedSum.
            var biasReshaped = Autodiff.TensorOperations<T>.Reshape(biasNode, 1, numCapsules, capsuleDim);
            var withBias = Autodiff.TensorOperations<T>.Add(weightedSum, biasReshaped);

            // Squash activation
            var s2 = Autodiff.TensorOperations<T>.Square(withBias);
            var normSq = Autodiff.TensorOperations<T>.Sum(s2, new int[] { 2 }, keepDims: true); // [B, O, 1]
            var norm = Autodiff.TensorOperations<T>.Sqrt(normSq);

            // Create epsilon constant for numerical stability (prevent division by zero)
            var epsilon = Autodiff.TensorOperations<T>.Constant(Tensor<T>.CreateDefault(new int[] { 1 }, NumOps.FromDouble(1e-8)));
            var one = Autodiff.TensorOperations<T>.Constant(Tensor<T>.CreateDefault(new int[] { 1 }, NumOps.One));
            var scale = Autodiff.TensorOperations<T>.Divide(normSq, Autodiff.TensorOperations<T>.Add(one, normSq));
            // Use norm + epsilon as denominator to avoid division by zero when norm is zero
            var stableNorm = Autodiff.TensorOperations<T>.Add(norm, epsilon);
            var unitVec = Autodiff.TensorOperations<T>.Divide(withBias, stableNorm);

            output = Autodiff.TensorOperations<T>.ElementwiseMultiply(scale, unitVec);

            // Update couplings if not last iteration
            if (iter < _numRoutingIterations - 1)
            {
                // Agreement = predictions . output
                var outputBroad = Autodiff.TensorOperations<T>.Reshape(output, batchSize, 1, numCapsules, capsuleDim);
                var agreementRaw = Autodiff.TensorOperations<T>.ElementwiseMultiply(predictions, outputBroad);
                var agreement = Autodiff.TensorOperations<T>.Sum(agreementRaw, new int[] { 3 }, keepDims: false);

                // Update and Softmax
                var rawCouplings = Autodiff.TensorOperations<T>.Add(couplings, agreement);
                couplings = Autodiff.TensorOperations<T>.Softmax(rawCouplings, axis: 2);
            }
        }

        // 5. Set Gradient
        output.Gradient = outputGradient;

        // 6. Backward
        output.Backward();

        // 7. Store Gradients
        // _biasGradient is flattened - use default zero tensor if gradient is null
        _biasGradient = biasNode.Gradient ?? Tensor<T>.CreateDefault(_bias.Shape, NumOps.Zero);

        // _transformationMatrixGradient needs [I, D_in, O, D_out]
        // weightsPermuted.Gradient is [I, O, D_in, D_out]
        // Permute back to [I, D_in, O, D_out] (0, 2, 1, 3)
        var gradPermuted = weightsPermuted.Gradient;
        if (gradPermuted != null)
        {
            _transformationMatrixGradient = gradPermuted.Transpose(new int[] { 0, 2, 1, 3 });
        }
        else
        {
            _transformationMatrixGradient = Tensor<T>.CreateDefault(_transformationMatrix.Shape, NumOps.Zero);
        }

        return inputNode.Gradient ?? throw new InvalidOperationException("Gradient computation failed.");
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

        // Use Engine operations for GPU/CPU acceleration
        var scaledTransformGrad = Engine.TensorMultiplyScalar(_transformationMatrixGradient, learningRate);
        _transformationMatrix = Engine.TensorSubtract(_transformationMatrix, scaledTransformGrad);

        var scaledBiasGrad = Engine.TensorMultiplyScalar(_biasGradient, learningRate);
        _bias = Engine.TensorSubtract(_bias, scaledBiasGrad);
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
        // Use Vector.Concatenate for production-grade parameter extraction
        return Vector<T>.Concatenate(
            new Vector<T>(_transformationMatrix.ToArray()),
            new Vector<T>(_bias.ToArray())
        );
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
        int biasSize = _bias.Length;

        if (parameters.Length != matrixSize + biasSize)
            throw new ArgumentException($"Expected {matrixSize + biasSize} parameters, but got {parameters.Length}");

        // Set parameters without hot-path conversions
        _transformationMatrix = new Tensor<T>(_transformationMatrix.Shape, parameters.Slice(0, matrixSize));
        _bias = new Tensor<T>([biasSize], parameters.Slice(matrixSize, biasSize));
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

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (inputNodes.Count == 0)
            throw new ArgumentException("At least one input node is required.", nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        var input = inputNodes[0];
        int inputCapsules = InputShape[0];
        int inputDimension = InputShape[1];

        // Create weight tensor as constant node
        var transformTensor = new Tensor<T>(
            new[] { _transformationMatrix.Shape[0], _transformationMatrix.Shape[1], _transformationMatrix.Shape[2] },
            _transformationMatrix.ToVector());
        var transformationMatrixNode = TensorOperations<T>.Constant(transformTensor, "CapsuleTransformMatrix");

        // Bias is already a Tensor<T>, use directly
        var biasNode = TensorOperations<T>.Constant(_bias, "CapsuleBias");

        // Reshape input for matrix multiplication: [batchSize * inputCapsules, inputDimension]
        var reshapedInput = TensorOperations<T>.Reshape(input, [inputCapsules, inputDimension]);

        // Transform input capsules: predictions = input @ transformationMatrix
        // This gives us [inputCapsules, numCapsules, capsuleDimension]
        var predictions = TensorOperations<T>.MatrixMultiply(reshapedInput, transformationMatrixNode);

        // Initialize coupling coefficients as uniform: 1/numCapsules using Fill
        var uniformCoeff = NumOps.FromDouble(1.0 / _numCapsules);
        var couplingsTensor = new Tensor<T>(new[] { inputCapsules, _numCapsules });
        couplingsTensor.Fill(uniformCoeff);
        var couplings = TensorOperations<T>.Constant(couplingsTensor, "InitialCouplings");

        ComputationNode<T> output = predictions;

        // Unroll routing iterations
        for (int iter = 0; iter < _numRoutingIterations; iter++)
        {
            // Apply softmax to couplings along numCapsules dimension
            var routingWeights = TensorOperations<T>.Softmax(couplings, axis: 1);

            // Weighted sum: weightedSum[j] = sum_i(couplings[i,j] * predictions[i,j])
            // This is element-wise multiply then sum over input capsules
            var weighted = TensorOperations<T>.ElementwiseMultiply(predictions, routingWeights);
            var weightedSum = TensorOperations<T>.Sum(weighted, [0]); // Sum over inputCapsules

            // Add bias
            var withBias = TensorOperations<T>.Add(weightedSum, biasNode);

            // Apply squash activation: v = ||s||^2 / (1 + ||s||^2) * s / ||s||
            // This normalizes vectors to have length <= 1
            var squaredNorm = TensorOperations<T>.Sum(TensorOperations<T>.Square(withBias), [1]);
            var oneTensor = new Tensor<T>(new[] { 1 }, new Vector<T>(new[] { NumOps.One }));
            var oneNode = TensorOperations<T>.Constant(oneTensor, "One");
            // Create epsilon constant for numerical stability (prevent division by zero)
            var epsTensor = new Tensor<T>(new[] { 1 }, new Vector<T>(new[] { NumOps.FromDouble(1e-8) }));
            var epsNode = TensorOperations<T>.Constant(epsTensor, "Epsilon");
            var normPlusOne = TensorOperations<T>.Add(squaredNorm, oneNode);
            var scaleFactor = TensorOperations<T>.Divide(squaredNorm, normPlusOne);
            var norm = TensorOperations<T>.Sqrt(squaredNorm);
            // Use norm + epsilon as denominator to avoid division by zero when norm is zero
            var stableNorm = TensorOperations<T>.Add(norm, epsNode);
            var normalizedVec = TensorOperations<T>.Divide(withBias, stableNorm);
            output = TensorOperations<T>.ElementwiseMultiply(normalizedVec, scaleFactor);

            // Update couplings if not last iteration
            if (iter < _numRoutingIterations - 1)
            {
                // Agreement: predictions dot output for each input capsule
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
    /// <c>true</c> because CapsuleLayer uses dynamic routing with a fixed number of iterations
    /// that can be unrolled into a static computation graph.
    /// </value>
    public override bool SupportsJitCompilation => true;

}

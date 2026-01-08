using AiDotNet.Autodiff;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a spatial pooler layer inspired by hierarchical temporal memory (HTM) principles.
/// </summary>
/// <remarks>
/// <para>
/// A spatial pooler is a key component in HTM systems that converts input patterns into sparse distributed
/// representations (SDRs). It maps input space to a new representation that captures the spatial structure
/// of the input while maintaining semantic similarity. The spatial pooler creates these representations
/// by selecting a small subset of active columns based on their connection strengths to the input.
/// </para>
/// <para><b>For Beginners:</b> This layer helps convert input data into a format that's easier for neural networks to learn from.
/// 
/// Think of it like a translator that:
/// - Takes dense input information (where many values can be active)
/// - Converts it to a sparse representation (where only a few values are active)
/// - Preserves the important patterns and relationships in the data
/// 
/// Benefits include:
/// - Better handling of noisy or incomplete data
/// - More efficient representation of information
/// - Improved ability to recognize patterns
/// 
/// For example, when processing images, a spatial pooler might identify the most important features
/// while ignoring background noise or variations that don't matter for classification.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class SpatialPoolerLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The size of the input vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field specifies the number of elements in the input vector, representing the dimensionality of the input space.
    /// </para>
    /// <para><b>For Beginners:</b> This is the number of values in the data that goes into this layer.
    /// 
    /// For example:
    /// - If processing images, this might be the total number of pixels
    /// - If processing text, this might be the vocabulary size
    /// 
    /// Larger input sizes can capture more detailed information but require more memory and processing power.
    /// </para>
    /// </remarks>
    private readonly int InputSize;

    /// <summary>
    /// The number of columns in the spatial pooler.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field specifies the number of columns in the spatial pooler, which determines the dimensionality of the
    /// output space. Each column corresponds to a feature detector that responds to specific patterns in the input.
    /// </para>
    /// <para><b>For Beginners:</b> This is the number of "feature detectors" in the layer.
    /// 
    /// Each column acts like a pattern detector that looks for specific features in the input.
    /// The layer produces a sparse output where only a small percentage of these columns are active
    /// at any given time.
    /// 
    /// More columns allow the layer to detect more distinct patterns, but require more computation.
    /// </para>
    /// </remarks>
    private readonly int ColumnCount;

    /// <summary>
    /// The threshold that determines the sparsity of the output.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field specifies the threshold used to determine which columns become active. Only columns with
    /// activation values above this threshold will be activated in the output. This controls the sparsity
    /// of the output representation.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how "picky" the layer is about activating columns.
    /// 
    /// A higher threshold means:
    /// - Fewer columns will be active (more sparse output)
    /// - Only the strongest patterns will be detected
    /// - The representation will be more selective
    /// 
    /// A lower threshold means:
    /// - More columns will be active (less sparse output)
    /// - Weaker patterns will also be detected
    /// - The representation will be more inclusive
    /// 
    /// Finding the right balance is important for the network to learn effectively.
    /// </para>
    /// </remarks>
    private readonly double SparsityThreshold;

    /// <summary>
    /// The connection strengths between input elements and columns.
    /// </summary>
    private Tensor<T> Connections;

    /// <summary>
    /// Gradient of the connections computed during backpropagation.
    /// </summary>
    private Tensor<T>? _connectionsGradient;

    /// <summary>
    /// Stores the input tensor from the most recent forward pass or learning step.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field caches the input to the layer during the forward pass or learning step, which is needed during
    /// parameter updates. It is cleared when ResetState() is called.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the layer's short-term memory of what input it received.
    ///
    /// The layer needs to remember what input it processed so that it can properly update
    /// its connections during learning. This temporary storage is cleared between batches
    /// or when you explicitly reset the layer.
    /// </para>
    /// </remarks>
    private Tensor<T>? LastInput;

    /// <summary>
    /// Stores the output tensor from the most recent forward pass or learning step.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field caches the output from the layer during the forward pass or learning step, which is needed during
    /// parameter updates. It is cleared when ResetState() is called.
    /// </para>
    /// <para><b>For Beginners:</b> This is the layer's memory of what output it produced.
    ///
    /// The layer needs to remember which columns were activated so that it can update
    /// the connections appropriately during learning. This temporary storage is cleared
    /// between batches or when you explicitly reset the layer.
    /// </para>
    /// </remarks>
    private Tensor<T>? LastOutput;

    /// <summary>
    /// The learning rate used during the learning process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field specifies the rate at which connection strengths are adjusted during learning. A higher learning rate
    /// leads to faster adaptation but potentially less stability, while a lower learning rate provides more stability
    /// but slower adaptation.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how quickly the layer updates its connections when learning.
    /// 
    /// Think of it as the size of steps taken during learning:
    /// - Larger value (e.g., 0.1): Takes bigger steps, learns faster but might overshoot
    /// - Smaller value (e.g., 0.001): Takes smaller steps, learns more slowly but more precisely
    /// 
    /// The default value of 0.01 provides a balance between learning speed and stability.
    /// </para>
    /// </remarks>
    private readonly double LearningRate = 0.01;

    /// <summary>
    /// The factor that controls boosting of inactive columns.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field specifies the boost factor used to encourage activation of columns that are rarely active.
    /// Boosting helps to ensure that all columns participate in representing the input space, preventing a small
    /// subset of columns from dominating.
    /// </para>
    /// <para><b>For Beginners:</b> This helps make sure all columns get a chance to be useful.
    /// 
    /// If some columns are never active, they're not helping the network learn.
    /// The boost factor gives a small advantage to columns that haven't been active recently,
    /// making it more likely they'll respond to some input patterns.
    /// 
    /// Think of it like giving extra attention to team members who haven't had a chance
    /// to contribute yet.
    /// </para>
    /// </remarks>
    private readonly double BoostFactor = 0.005;

    /// <summary>
    /// Gets a value indicating whether this layer supports training through backpropagation.
    /// </summary>
    /// <value>
    /// Always returns <c>true</c> as spatial pooler layers have trainable parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that the spatial pooler layer can be trained. The layer contains trainable parameters
    /// (connection strengths) that are updated during the training process.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you that the layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer contains numbers (parameters) that can be adjusted during training
    /// - It will improve its performance as it sees more examples
    /// - It participates in the learning process of the neural network
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="SpatialPoolerLayer{T}"/> class.
    /// </summary>
    /// <param name="inputSize">The size of the input vector.</param>
    /// <param name="columnCount">The number of columns in the spatial pooler.</param>
    /// <param name="sparsityThreshold">The threshold that determines the sparsity of the output.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a spatial pooler layer with the specified input size, column count, and sparsity threshold.
    /// It initializes the connection matrix with random values and sets up the input and output shapes for the layer.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up a new spatial pooler layer with your specified settings.
    /// 
    /// When creating a spatial pooler, you need to specify:
    /// - inputSize: How many values are in each input (e.g., number of pixels in an image)
    /// - columnCount: How many feature detectors to create (more means more detailed representation)
    /// - sparsityThreshold: How selective the layer should be (higher means fewer active columns)
    /// 
    /// The constructor automatically initializes the connections between inputs and columns
    /// with random values that will be adjusted during learning.
    /// </para>
    /// </remarks>
    public SpatialPoolerLayer(int inputSize, int columnCount, double sparsityThreshold)
        : base([inputSize], [columnCount])
    {
        InputSize = inputSize;
        ColumnCount = columnCount;
        SparsityThreshold = sparsityThreshold;
        Connections = new Tensor<T>([inputSize, columnCount]);

        InitializeConnections();
    }

    /// <summary>
    /// Initializes the connection strengths between input elements and columns with random values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the connection matrix with random values between 0 and 1. These initial
    /// random connections provide a starting point for the learning process.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the initial random connection strengths.
    /// 
    /// Before learning begins, the layer needs some starting connection values:
    /// - Each connection is assigned a random value between 0 and 1
    /// - These random values give the network a starting point
    /// - During learning, these values will be adjusted to better detect patterns
    /// 
    /// Good initialization helps the network learn more effectively from the beginning.
    /// </para>
    /// </remarks>
    private void InitializeConnections()
    {
        // Initialize connections with random values using tensor ops
        Connections = new Tensor<T>([InputSize, ColumnCount], Vector<T>.CreateRandom(InputSize * ColumnCount, 0.0, 1.0));
    }

    /// <summary>
    /// Performs the forward pass of the spatial pooler layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after processing through the spatial pooler.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the spatial pooler layer. It computes the overlap between
    /// the input and each column's connections, and then activates columns whose overlap exceeds the sparsity
    /// threshold. The result is a sparse representation where active columns are set to 1 and inactive columns
    /// are set to 0.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes data through the spatial pooler.
    /// 
    /// During the forward pass:
    /// - The input tensor is converted to a vector for easier processing
    /// - For each column, the layer calculates how strongly it responds to the input
    /// - Columns with a strong enough response (above the sparsity threshold) are activated (set to 1)
    /// - All other columns remain inactive (set to 0)
    /// - The result is a sparse output where only a small percentage of columns are active
    /// 
    /// This sparse representation helps the network focus on the most important features
    /// and ignore noise or irrelevant details.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Flatten to 1D tensor if needed
        LastInput = input.Shape.Length == 1
            ? input
            : input.Reshape([input.Length]);

        // Use Engine tensor operations: output = Connections^T @ input
        var inputTensor = LastInput.Reshape([InputSize, 1]);

        var connectionsT = Engine.TensorTranspose(Connections);
        var activations = Engine.TensorMatMul(connectionsT, inputTensor);

        // Apply threshold to get sparse binary output
        T threshold = NumOps.FromDouble(SparsityThreshold);
        var outputMask = Engine.TensorGreaterThan(activations, threshold);
        var output = outputMask.Reshape([ColumnCount]);

        LastOutput = output;
        return output;
    }

    /// <summary>
    /// Performs the GPU-accelerated forward pass for the spatial pooler layer.
    /// </summary>
    /// <param name="inputs">The GPU tensor inputs. First element is the input activation.</param>
    /// <returns>A GPU tensor containing the sparse binary output.</returns>
    /// <remarks>
    /// The spatial pooler converts input patterns into sparse distributed representations.
    /// This method processes the input using GPU operations for matrix multiplication and thresholding.
    /// </remarks>
    public override IGpuTensor<T> ForwardGpu(params IGpuTensor<T>[] inputs)
    {
        if (inputs == null || inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        var input = inputs[0];

        // Validate GPU engine availability
        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires a DirectGpuTensorEngine.");

        var backend = gpuEngine.GetBackend();
        if (backend == null)
            throw new InvalidOperationException("GPU backend is not available.");

        // Flatten to 1D if needed
        var flatInput = input.Shape.Length == 1 ? input : gpuEngine.ReshapeGpu(input, [input.ElementCount]);

        // Reshape to [1, InputSize] for matrix multiply (batch of 1)
        var inputReshaped = gpuEngine.ReshapeGpu(flatInput, [1, InputSize]);

        // Matrix multiply: [1, InputSize] @ [InputSize, ColumnCount] = [1, ColumnCount]
        var activations = gpuEngine.BatchedMatMulGpu(inputReshaped, Connections);

        // Apply threshold to get sparse binary output
        var thresholdFloat = (float)SparsityThreshold;
        var outputMask = gpuEngine.GreaterThanScalarGpu(activations, thresholdFloat);

        // Reshape to [ColumnCount]
        var output = gpuEngine.ReshapeGpu(outputMask, [ColumnCount]);

        // Dispose intermediates (except output)
        if (input.Shape.Length != 1)
            flatInput.Dispose();
        inputReshaped.Dispose();
        activations.Dispose();
        outputMask.Dispose();

        return output;
    }

    /// <summary>
    /// Performs a learning step on the spatial pooler using the provided input.
    /// </summary>
    /// <param name="input">The input vector to learn from.</param>
    /// <remarks>
    /// <para>
    /// This method implements the learning algorithm for the spatial pooler. It updates the connection strengths
    /// based on the current input and output. For active columns, connections to active input elements are
    /// strengthened and connections to inactive input elements are weakened. For inactive columns, a small
    /// boost is applied to help them become active for other patterns.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the spatial pooler to recognize patterns.
    /// 
    /// During learning:
    /// 1. The input is processed to determine which columns activate
    /// 2. For active columns:
    ///    - Connections to active input elements are strengthened
    ///    - Connections to inactive input elements are weakened
    /// 3. For inactive columns:
    ///    - A small boost is applied to connections to active input elements
    ///    - This gives inactive columns a better chance of activating in the future
    /// 4. All connections are normalized to keep their values balanced
    /// 
    /// Over time, this process helps the spatial pooler learn to recognize important patterns
    /// in the input data and represent them effectively.
    /// </para>
    /// </remarks>
    public void Learn(Vector<T> input)
    {
        // Convert to tensor and call Forward which stores in LastInput/LastOutput
        var inputTensor = new Tensor<T>(new[] { InputSize }, input);
        Forward(inputTensor);

        if (LastInput == null || LastOutput == null)
            return;

        T lr = NumOps.FromDouble(LearningRate);
        T bf = NumOps.FromDouble(BoostFactor);

        // Strengthen active columns: Connections += lr * (input - Connections) * activeMask
        var activeMask = LastOutput.Reshape([1, ColumnCount]);
        var inputRow = LastInput.Reshape([InputSize, 1]);
        var onesCol = new Tensor<T>([1, ColumnCount]);
        onesCol.Fill(NumOps.One);
        var inputExpanded = Engine.TensorMatMul(inputRow, onesCol); // [InputSize, ColumnCount]
        var deltaActive = Engine.TensorMultiplyScalar(Engine.TensorSubtract(inputExpanded, Connections), lr);
        Connections = Engine.TensorAdd(Connections, Engine.TensorMultiply(deltaActive, activeMask));

        // Boost inactive columns: Connections += bf * input * inactiveMask
        var inactiveMask = Engine.TensorSubtract(onesCol, activeMask);
        var boostTensor = Engine.TensorMultiplyScalar(inputExpanded, bf);
        Connections = Engine.TensorAdd(Connections, Engine.TensorMultiply(boostTensor, inactiveMask));

        NormalizeConnections();
    }

    /// <summary>
    /// Normalizes the connection strengths to ensure all columns have balanced total connection strength.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method normalizes the connection strengths for each column so that they sum to 1. This ensures that
    /// all columns have the same total connection strength, preventing some columns from dominating due to
    /// having generally higher connection values.
    /// </para>
    /// <para><b>For Beginners:</b> This method keeps the connection strengths balanced.
    /// 
    /// Normalization ensures that:
    /// - The sum of all connection strengths for each column equals 1
    /// - No column can become dominant just by having larger connection values overall
    /// - The relative strength of connections within a column is preserved
    /// 
    /// Think of it like ensuring each column has the same total "budget" of connection strength,
    /// but it can distribute that budget differently across its connections based on what it learns.
    /// </para>
    /// </remarks>
    private void NormalizeConnections()
    {
        var colSums = Engine.ReduceSum(Connections, new[] { 0 }, keepDims: true);
        var safeSums = Engine.TensorMax(colSums, NumOps.FromDouble(1e-12));
        Connections = Engine.TensorDivide(Connections, safeSums);
    }

    /// <summary>
    /// Performs the backward pass of the spatial pooler layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the spatial pooler layer, which is used during training to propagate
    /// error gradients back through the network. It computes the gradient of the loss with respect to the input
    /// by propagating the output gradient through the connection matrix.
    /// </para>
    /// <para><b>For Beginners:</b> This method is used during training to calculate how the layer's input
    /// should change to reduce errors.
    /// 
    /// During the backward pass:
    /// - The layer receives gradients (error signals) from the next layer
    /// - It uses the connection strengths to determine how these errors should be distributed to the input
    /// - Each input element receives a weighted sum of gradients based on its connections to the columns
    /// 
    /// This process allows error information to flow backward through the network during training,
    /// enabling all layers to learn from the overall network performance.
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
        // Use Engine tensor operations: inputGrad = Connections @ outputGrad
        // Connections is [InputSize, ColumnCount], no transpose needed
        // outputGradient expected shape [ColumnCount]
        var gradTensor = outputGradient.Shape.Length == 1
            ? outputGradient.Reshape([ColumnCount, 1])
            : outputGradient;

        var inputGradTensor = Engine.TensorMatMul(Connections, gradTensor);

        return inputGradTensor.Reshape([InputSize]);
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation with a straight-through estimator for the threshold.
    /// It constructs the computation graph to compute gradients for both input and connections.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (LastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // 1. Variables
        var inputNode = Autodiff.TensorOperations<T>.Variable(LastInput, "input", requiresGradient: true);
        var connectionsNode = Autodiff.TensorOperations<T>.Variable(Connections, "connections", requiresGradient: true);

        // 2. Graph Construction (mirrors Forward/Export)
        // Transpose connections: [ColumnCount, InputSize]
        var connectionsTransposed = Autodiff.TensorOperations<T>.Transpose(connectionsNode);

        // Reshape input for matrix multiplication: [InputSize, 1]
        var inputReshaped = Autodiff.TensorOperations<T>.Reshape(inputNode, InputSize, 1);

        // activation = Connections^T @ input
        var activation = Autodiff.TensorOperations<T>.MatrixMultiply(connectionsTransposed, inputReshaped);
        var activationFlat = Autodiff.TensorOperations<T>.Reshape(activation, ColumnCount);

        // Apply straight-through threshold for sparse binary output
        var output = Autodiff.TensorOperations<T>.StraightThroughThreshold(activationFlat, SparsityThreshold);

        // Apply layer activation if present
        var finalOutput = ApplyActivationToGraph(output);

        // 3. Set Gradient and Backward
        finalOutput.Gradient = outputGradient;
        finalOutput.Backward();

        // 4. Store Gradients
        _connectionsGradient = connectionsNode.Gradient;

        return inputNode.Gradient ?? throw new InvalidOperationException("Gradient computation failed.");
    }


    /// <summary>
    /// Updates the parameters of the layer using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method updates the connection strengths based on the most recent input and output from a forward pass.
    /// It strengthens connections between active input elements and active columns, then normalizes the connections
    /// to maintain balanced total connection strength.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the layer's connections during standard neural network training.
    /// 
    /// When updating parameters:
    /// - If there's no record of previous input/output, the method does nothing
    /// - Otherwise, it strengthens connections between active inputs and active columns
    /// - The learning rate controls how big each update step is
    /// - Finally, connections are normalized to keep their values balanced
    /// 
    /// This method is similar to the Learn method but is designed to work within the standard
    /// neural network training framework, where gradient information flows through the network.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // 1. Standard Backpropagation (Discriminative Training)
        if (_connectionsGradient != null)
        {
            var delta = Engine.TensorMultiplyScalar(_connectionsGradient, learningRate);
            Connections = Engine.TensorSubtract(Connections, delta);

            // Maintain connection constraints
            NormalizeConnections();
            return;
        }

        // 2. Hebbian Learning (HTM Training)
        if (LastOutput == null || LastInput == null)
            return;

        T lr = NumOps.FromDouble(LearningRate);
        T bf = NumOps.FromDouble(BoostFactor);

        for (int i = 0; i < ColumnCount; i++)
        {
            if (NumOps.Equals(LastOutput[i], NumOps.One))
            {
                // Strengthen connections for active columns
                for (int j = 0; j < InputSize; j++)
                {
                    T delta = NumOps.Multiply(lr, NumOps.Subtract(LastInput[j], Connections[j, i]));
                    Connections[j, i] = NumOps.Add(Connections[j, i], delta);
                }
            }
        }

        // Boost inactive columns
        for (int i = 0; i < ColumnCount; i++)
        {
            if (NumOps.Equals(LastOutput[i], NumOps.Zero))
            {
                for (int j = 0; j < InputSize; j++)
                {
                    T boost = NumOps.Multiply(bf, LastInput[j]);
                    Connections[j, i] = NumOps.Add(Connections[j, i], boost);
                }
            }
        }

        NormalizeConnections();
    }

    /// <summary>
    /// Gets all trainable parameters of the layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters of the layer (connection strengths) and combines them
    /// into a single vector. This is useful for optimization algorithms that operate on all parameters at once,
    /// or for saving and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values from the layer into a single list.
    /// 
    /// The parameters:
    /// - Are the connection strengths between input elements and columns
    /// - Are converted from a matrix to a single long list (vector)
    /// - Can be used to save the state of the layer or apply optimization techniques
    /// 
    /// This method is useful for:
    /// - Saving the model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques that need access to all parameters
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Use Tensor.ToArray() to efficiently convert to vector
        var flatConnections = new Vector<T>(Connections.ToArray());
        return flatConnections;
    }

    /// <summary>
    /// Sets the trainable parameters of the layer from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets the trainable parameters of the layer (connection strengths) from a single vector.
    /// It expects the vector to contain the parameters in the same order as they are retrieved by GetParameters().
    /// This is useful for loading saved model weights or for implementing optimization algorithms that operate
    /// on all parameters at once.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the connections in the layer from a single list.
    /// 
    /// When setting parameters:
    /// - The input must be a vector with exactly the right number of values
    /// - The values are distributed back into the connection matrix
    /// - The order must match how they were stored in GetParameters()
    /// 
    /// This method is useful for:
    /// - Loading a previously saved model
    /// - Transferring parameters from another model
    /// - Testing different parameter values
    /// 
    /// An error is thrown if the input vector doesn't have the expected number of parameters.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != InputSize * ColumnCount)
        {
            throw new ArgumentException($"Expected {InputSize * ColumnCount} parameters, but got {parameters.Length}");
        }

        // Restore connections without hot-path conversions
        Connections = new Tensor<T>([InputSize, ColumnCount], parameters);
    }

    /// <summary>
    /// Resets the internal state of the spatial pooler layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the spatial pooler layer, including the cached inputs and outputs.
    /// This is useful when starting to process a new batch or when implementing stateful networks that need to be
    /// reset between sequences.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - Stored inputs and outputs from previous passes are cleared
    /// - The connection strengths are not affected, so the layer retains what it has learned
    /// 
    /// This is important for:
    /// - Processing a new batch of unrelated data
    /// - Preventing information from one batch affecting another
    /// - Starting a new training episode
    /// 
    /// Think of it like clearing your short-term memory while keeping your long-term knowledge intact.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values
        LastInput = null;
        LastOutput = null;
        _connectionsGradient = null;
    }

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        if (inputNodes.Count == 0)
            throw new ArgumentException("At least one input node is required.", nameof(inputNodes));

        // SpatialPoolerLayer JIT uses straight-through estimator for thresholding:
        // 1. Compute overlap: activation = Connections^T @ input
        // 2. Apply threshold: output = StraightThroughThreshold(activation, sparsityThreshold)
        //
        // The straight-through estimator allows gradients to flow through the discrete threshold
        // operation during backpropagation.

        var input = inputNodes[0];

        // Connections is already a Tensor<T>, use it directly
        var connectionsNode = TensorOperations<T>.Constant(Connections, "sp_connections");

        // Transpose connections for multiplication: [ColumnCount, InputSize]
        var connectionsTransposed = TensorOperations<T>.Transpose(connectionsNode);

        // Reshape input for matrix multiplication
        var inputReshaped = TensorOperations<T>.Reshape(input, InputSize, 1);

        // activation = Connections^T @ input
        var activation = TensorOperations<T>.MatrixMultiply(connectionsTransposed, inputReshaped);
        var activationFlat = TensorOperations<T>.Reshape(activation, ColumnCount);

        // Apply straight-through threshold for sparse binary output
        var output = TensorOperations<T>.StraightThroughThreshold(activationFlat, SparsityThreshold);

        // Apply layer activation if present
        output = ApplyActivationToGraph(output);

        return output;
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    /// <value>
    /// Always <c>true</c>. SpatialPoolerLayer uses straight-through estimator for JIT compilation.
    /// </value>
    /// <remarks>
    /// <para>
    /// JIT compilation for SpatialPooler uses a straight-through estimator for the threshold
    /// operation. The forward pass produces sparse binary activations (0 or 1), but gradients
    /// pass through unchanged during backpropagation. This enables differentiable training
    /// while maintaining the sparse output characteristics.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

}

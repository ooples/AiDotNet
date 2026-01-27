using AiDotNet.Autodiff;
using AiDotNet.Extensions;

namespace AiDotNet.LoRA;

/// <summary>
/// Implements Low-Rank Adaptation (LoRA) layer for parameter-efficient fine-tuning of neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// LoRA works by decomposing weight updates into two low-rank matrices A and B, where the actual update
/// is computed as B * A. This dramatically reduces the number of trainable parameters compared to
/// fine-tuning all weights directly.
/// </para>
/// <para><b>For Beginners:</b> LoRA is a technique that makes it much cheaper to adapt large neural networks
/// to new tasks. Instead of updating all the weights in a layer (which can be millions of parameters),
/// LoRA adds two small matrices that work together to approximate the needed changes.
///
/// Think of it like this:
/// - Traditional fine-tuning: Adjusting every single knob on a massive control panel
/// - LoRA: Using just a few master controls that influence many knobs at once
///
/// The key insight is that the changes needed for fine-tuning often lie in a "low-rank" space,
/// meaning we don't need full freedom to adjust every parameter independently.
///
/// Key parameters:
/// - Rank (r): Controls how many "master controls" you have. Higher rank = more flexibility but more parameters
/// - Alpha: A scaling factor that controls how much influence the LoRA adaptation has
///
/// For example, adapting a layer with 1000x1000 weights (1M parameters) using LoRA with rank=8 only
/// requires 8x1000 + 8x1000 = 16,000 parameters (98.4% reduction!).
/// </para>
/// </remarks>
public class LoRALayer<T> : LayerBase<T>
{
    /// <summary>
    /// Low-rank matrix A with dimensions (inputSize × rank).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Matrix A is the first part of the low-rank decomposition. It projects the input from
    /// inputSize dimensions down to rank dimensions. This matrix is initialized with random values
    /// and trained during fine-tuning.
    /// </para>
    /// <para><b>For Beginners:</b> This is the first of two small matrices that work together.
    /// Think of it as compressing the input data into a smaller representation before expanding it again.
    /// </para>
    /// </remarks>
    private Matrix<T> _loraA;

    /// <summary>
    /// Low-rank matrix B with dimensions (rank × outputSize).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Matrix B is the second part of the low-rank decomposition. It projects from the rank dimensions
    /// back up to outputSize dimensions. This matrix is initialized to zero so that at the start of
    /// training, the LoRA layer has no effect on the base model's behavior.
    /// </para>
    /// <para><b>For Beginners:</b> This is the second matrix that expands the compressed data back
    /// to full size. It starts at zero so the adapted model initially behaves exactly like the original.
    /// </para>
    /// </remarks>
    private Matrix<T> _loraB;

    /// <summary>
    /// The rank of the low-rank decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The rank determines the dimensionality of the intermediate representation. Lower ranks mean
    /// fewer parameters but less expressiveness. Typical values range from 1 to 64, with 8 being
    /// a common choice.
    /// </para>
    /// <para><b>For Beginners:</b> The rank is like the number of "compression channels" you use.
    /// Higher rank = more flexibility but more parameters to train. It's a trade-off between
    /// efficiency and capability.
    /// </para>
    /// </remarks>
    private readonly int _rank;

    /// <summary>
    /// Scaling factor for the LoRA contribution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Alpha controls how much the LoRA adaptation influences the final output. The actual scaling
    /// applied is alpha/rank, which helps normalize the contribution across different rank values.
    /// Typical values for alpha are in the range of the rank (e.g., alpha = 16 with rank = 8).
    /// </para>
    /// <para><b>For Beginners:</b> This controls how strongly the LoRA adaptation affects the output.
    /// It's like a volume knob for the adaptations. The formula alpha/rank automatically adjusts
    /// so that different rank values produce similar strength adaptations.
    /// </para>
    /// </remarks>
    private readonly T _alpha;

    /// <summary>
    /// Computed scaling factor (alpha / rank) used during forward pass.
    /// </summary>
    private readonly T _scaling;

    /// <summary>
    /// Gradients for matrix A computed during backpropagation.
    /// </summary>
    private Matrix<T>? _loraAGradient;

    /// <summary>
    /// Gradients for matrix B computed during backpropagation.
    /// </summary>
    private Matrix<T>? _loraBGradient;

    /// <summary>
    /// Stored input from the forward pass, needed for gradient computation.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stored pre-activation output from the forward pass, needed for activation derivative computation.
    /// </summary>
    private Tensor<T>? _lastPreActivation;

    /// <summary>
    /// Gets the total number of trainable parameters (elements in A and B matrices).
    /// </summary>
    public override int ParameterCount => (_loraA.Rows * _loraA.Columns) + (_loraB.Rows * _loraB.Columns);

    /// <summary>
    /// Gets whether this layer supports training (always true for LoRA).
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new LoRA layer with the specified dimensions and hyperparameters.
    /// </summary>
    /// <param name="inputSize">The number of input features.</param>
    /// <param name="outputSize">The number of output features.</param>
    /// <param name="rank">The rank of the low-rank decomposition (must be positive and less than min(inputSize, outputSize)).</param>
    /// <param name="alpha">The scaling factor for LoRA contributions (typically similar to rank value).</param>
    /// <param name="activationFunction">Optional activation function to apply after the LoRA transformation.</param>
    /// <exception cref="ArgumentException">Thrown when rank is not positive or exceeds min(inputSize, outputSize).</exception>
    /// <remarks>
    /// <para>
    /// The LoRA matrices are initialized as follows:
    /// - Matrix A: Random values from a Gaussian distribution (similar to Kaiming initialization)
    /// - Matrix B: Zero initialization (so LoRA starts with no effect)
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new LoRA layer. You specify the input and output sizes
    /// (which should match the layer you're adapting), the rank (how much compression), and alpha
    /// (how strong the adaptation is).
    ///
    /// The initialization is carefully chosen:
    /// - Matrix A gets random values (so training can start moving in useful directions)
    /// - Matrix B starts at zero (so initially, LoRA doesn't change anything)
    /// </para>
    /// </remarks>
    public LoRALayer(int inputSize, int outputSize, int rank, double alpha = -1, IActivationFunction<T>? activationFunction = null)
        : base(new[] { inputSize }, new[] { outputSize }, activationFunction ?? new IdentityActivation<T>())
    {
        if (inputSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(inputSize), "Input size must be positive");
        }

        if (outputSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(outputSize), "Output size must be positive");
        }

        if (rank <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(rank), "Rank must be positive");
        }

        if (rank > Math.Min(inputSize, outputSize))
        {
            throw new ArgumentOutOfRangeException(nameof(rank), $"Rank ({rank}) cannot exceed min(inputSize, outputSize) = {Math.Min(inputSize, outputSize)}");
        }

        _rank = rank;

        // Default alpha to rank if not specified
        _alpha = alpha > 0 ? NumOps.FromDouble(alpha) : NumOps.FromDouble(rank);
        _scaling = NumOps.Divide(_alpha, NumOps.FromDouble(rank));

        // Initialize LoRA matrices
        // Matrix A: Random initialization (Gaussian with std = 1/sqrt(rank))
        _loraA = new Matrix<T>(inputSize, rank);
        T stddev = NumOps.Sqrt(NumOps.Divide(NumOps.One, NumOps.FromDouble(rank)));
        for (int i = 0; i < _loraA.Rows; i++)
        {
            for (int j = 0; j < _loraA.Columns; j++)
            {
                _loraA[i, j] = NumOps.Multiply(NumOps.FromDouble(Random.NextGaussian()), stddev);
            }
        }

        // Matrix B: Zero initialization (so LoRA has no effect initially)
        _loraB = new Matrix<T>(rank, outputSize);
        for (int i = 0; i < _loraB.Rows; i++)
        {
            for (int j = 0; j < _loraB.Columns; j++)
            {
                _loraB[i, j] = NumOps.Zero;
            }
        }

        // Initialize parameter vector
        Parameters = new Vector<T>(ParameterCount);
        UpdateParametersFromMatrices();
    }

    /// <summary>
    /// Performs the forward pass through the LoRA layer.
    /// </summary>
    /// <param name="input">Input tensor of shape [batchSize, inputSize].</param>
    /// <returns>Output tensor of shape [batchSize, outputSize].</returns>
    /// <remarks>
    /// <para>
    /// The forward pass computes: output = input * A * B * scaling
    /// where scaling = alpha / rank.
    /// </para>
    /// <para><b>For Beginners:</b> This processes data through the LoRA layer. The input is:
    /// 1. Multiplied by matrix A (compressing to rank dimensions)
    /// 2. Multiplied by matrix B (expanding back to output dimensions)
    /// 3. Scaled by alpha/rank (controlling the strength)
    ///
    /// The result represents the adaptation that gets added to the base layer's output.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input.Clone();

        // Get batch size and validate input shape
        int batchSize = input.Shape[0];
        int inputSize = input.Shape.Length > 1 ? input.Shape[1] : input.Length;

        if (inputSize != _loraA.Rows)
        {
            throw new ArgumentException($"Input size {inputSize} does not match expected input size {_loraA.Rows}");
        }

        // Convert input to matrix [batchSize, inputSize]
        // Handle any-rank tensors by flattening all dimensions after the first into inputSize
        Matrix<T> inputMatrix = new Matrix<T>(batchSize, inputSize);
        if (input.Shape.Length == 1)
        {
            // 1D input: treat as single batch
            for (int j = 0; j < inputSize; j++)
            {
                inputMatrix[0, j] = input[j];
            }
        }
        else if (input.Shape.Length == 2)
        {
            // 2D input: standard batch x features
            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < inputSize; j++)
                {
                    inputMatrix[i, j] = input[i, j];
                }
            }
        }
        else
        {
            // Higher rank: flatten all dims after first into the second dimension
            var flatInput = input.Reshape([batchSize, inputSize]);
            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < inputSize; j++)
                {
                    inputMatrix[i, j] = flatInput[i, j];
                }
            }
        }

        // Compute: input * A (result: [batchSize, rank])
        Matrix<T> intermediate = inputMatrix.Multiply(_loraA);

        // Compute: intermediate * B (result: [batchSize, outputSize])
        Matrix<T> output = intermediate.Multiply(_loraB);

        // Apply scaling
        output = output.Multiply(_scaling);

        // Convert back to tensor
        Vector<T> outputData = new Vector<T>(batchSize * _loraB.Columns);
        int idx = 0;
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < _loraB.Columns; j++)
            {
                outputData[idx++] = output[i, j];
            }
        }

        int[] outputShape = new int[] { batchSize, _loraB.Columns };
        Tensor<T> result = new Tensor<T>(outputShape, outputData);

        // Store pre-activation for gradient computation
        _lastPreActivation = result.Clone();

        // Apply activation if specified
        if (ScalarActivation != null)
        {
            result = ApplyActivation(result);
        }

        return result;
    }

    /// <summary>
    /// Performs the backward pass through the LoRA layer.
    /// </summary>
    /// <param name="outputGradient">Gradient flowing back from the next layer.</param>
    /// <returns>Gradient to pass to the previous layer.</returns>
    /// <remarks>
    /// <para>
    /// The backward pass computes gradients for both LoRA matrices and propagates gradients back to the input.
    /// Gradients are computed as:
    /// - dL/dB = A^T * input^T * outputGradient * scaling
    /// - dL/dA = input^T * outputGradient * B^T * scaling
    /// - dL/dinput = outputGradient * B^T * A^T * scaling
    /// </para>
    /// <para><b>For Beginners:</b> This is where learning happens! The backward pass:
    /// 1. Figures out how to adjust matrix A and B to reduce error
    /// 2. Passes gradients back to earlier layers so they can learn too
    ///
    /// It uses calculus (specifically, the chain rule) to figure out how each parameter
    /// contributed to the error.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass");
        }

        // Get dimensions
        int batchSize = _lastInput.Shape[0];
        int inputSize = _lastInput.Shape.Length > 1 ? _lastInput.Shape[1] : _lastInput.Length;
        int outputSize = _loraB.Columns;

        // Apply activation gradient if needed using pre-activation values
        if (ScalarActivation != null && _lastPreActivation != null)
        {
            outputGradient = ApplyActivationDerivative(_lastPreActivation, outputGradient);
        }

        // Convert tensors to matrices
        Matrix<T> inputMatrix = new Matrix<T>(batchSize, inputSize);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                inputMatrix[i, j] = _lastInput[i * inputSize + j];
            }
        }

        Matrix<T> gradMatrix = new Matrix<T>(batchSize, outputSize);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                gradMatrix[i, j] = outputGradient[i * outputSize + j];
            }
        }

        // Compute gradients for B: dL/dB = (input * A)^T * outputGradient * scaling
        Matrix<T> inputTimesA = inputMatrix.Multiply(_loraA);  // [batchSize, rank]
        _loraBGradient = inputTimesA.Transpose().Multiply(gradMatrix).Multiply(_scaling);  // [rank, outputSize]

        // Compute gradients for A: dL/dA = input^T * (outputGradient * B^T) * scaling
        Matrix<T> gradTimesB = gradMatrix.Multiply(_loraB.Transpose());  // [batchSize, rank]
        _loraAGradient = inputMatrix.Transpose().Multiply(gradTimesB).Multiply(_scaling);  // [inputSize, rank]

        // Compute input gradients: dL/dinput = outputGradient * B^T * A^T * scaling
        Matrix<T> inputGradient = gradMatrix.Multiply(_loraB.Transpose()).Multiply(_loraA.Transpose()).Multiply(_scaling);

        // Convert back to tensor
        Vector<T> inputGradData = new Vector<T>(batchSize * inputSize);
        int idx = 0;
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                inputGradData[idx++] = inputGradient[i, j];
            }
        }

        // Update parameter gradients vector
        UpdateParameterGradients();

        return new Tensor<T>(new[] { batchSize, inputSize }, inputGradData);
    }

    /// <summary>
    /// Updates the layer's parameters using the specified learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    public override void UpdateParameters(T learningRate)
    {
        if (_loraAGradient == null || _loraBGradient == null)
        {
            return;
        }

        // Update matrix A
        for (int i = 0; i < _loraA.Rows; i++)
        {
            for (int j = 0; j < _loraA.Columns; j++)
            {
                T update = NumOps.Multiply(_loraAGradient[i, j], learningRate);
                _loraA[i, j] = NumOps.Subtract(_loraA[i, j], update);
            }
        }

        // Update matrix B
        for (int i = 0; i < _loraB.Rows; i++)
        {
            for (int j = 0; j < _loraB.Columns; j++)
            {
                T update = NumOps.Multiply(_loraBGradient[i, j], learningRate);
                _loraB[i, j] = NumOps.Subtract(_loraB[i, j], update);
            }
        }

        // Update parameter vector
        UpdateParametersFromMatrices();
    }

    /// <summary>
    /// Gets the current parameters as a vector.
    /// </summary>
    /// <returns>Vector containing all LoRA parameters (A and B matrices flattened).</returns>
    public override Vector<T> GetParameters()
    {
        return Parameters.Clone();
    }

    /// <summary>
    /// Sets the layer parameters from a vector.
    /// </summary>
    /// <param name="parameters">Vector containing all LoRA parameters.</param>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException($"Expected {ParameterCount} parameters, got {parameters.Length}");
        }

        Parameters = parameters.Clone();
        UpdateMatricesFromParameters();
    }

    /// <summary>
    /// Updates the parameter vector from the current matrix values.
    /// </summary>
    private void UpdateParametersFromMatrices()
    {
        int idx = 0;

        // Pack matrix A
        for (int i = 0; i < _loraA.Rows; i++)
        {
            for (int j = 0; j < _loraA.Columns; j++)
            {
                Parameters[idx++] = _loraA[i, j];
            }
        }

        // Pack matrix B
        for (int i = 0; i < _loraB.Rows; i++)
        {
            for (int j = 0; j < _loraB.Columns; j++)
            {
                Parameters[idx++] = _loraB[i, j];
            }
        }
    }

    /// <summary>
    /// Updates the matrices from the parameter vector.
    /// </summary>
    private void UpdateMatricesFromParameters()
    {
        int idx = 0;

        // Unpack matrix A
        for (int i = 0; i < _loraA.Rows; i++)
        {
            for (int j = 0; j < _loraA.Columns; j++)
            {
                _loraA[i, j] = Parameters[idx++];
            }
        }

        // Unpack matrix B
        for (int i = 0; i < _loraB.Rows; i++)
        {
            for (int j = 0; j < _loraB.Columns; j++)
            {
                _loraB[i, j] = Parameters[idx++];
            }
        }
    }

    /// <summary>
    /// Updates the parameter gradients vector from the matrix gradients.
    /// </summary>
    private void UpdateParameterGradients()
    {
        if (_loraAGradient == null || _loraBGradient == null)
        {
            return;
        }

        ParameterGradients = new Vector<T>(ParameterCount);
        int idx = 0;

        // Pack matrix A gradients
        for (int i = 0; i < _loraAGradient.Rows; i++)
        {
            for (int j = 0; j < _loraAGradient.Columns; j++)
            {
                ParameterGradients[idx++] = _loraAGradient[i, j];
            }
        }

        // Pack matrix B gradients
        for (int i = 0; i < _loraBGradient.Rows; i++)
        {
            for (int j = 0; j < _loraBGradient.Columns; j++)
            {
                ParameterGradients[idx++] = _loraBGradient[i, j];
            }
        }
    }

    /// <summary>
    /// Merges the LoRA weights into a dense weight matrix that can be added to a base layer.
    /// </summary>
    /// <returns>The merged weight matrix (inputSize × outputSize) representing the full LoRA contribution.</returns>
    /// <remarks>
    /// <para>
    /// This computes the full weight matrix W_lora = A * B * scaling, which can then be added to the
    /// base layer's weights. This is useful for deployment when you want to merge the adaptation
    /// back into the base model for inference efficiency.
    /// </para>
    /// <para><b>For Beginners:</b> This "bakes in" the LoRA adaptation into a regular weight matrix.
    /// Instead of storing two small matrices (A and B) and computing them during inference,
    /// you can merge them into one larger matrix and add it to the original weights.
    ///
    /// This is like converting assembly instructions back into a final product - once you're done
    /// training, you can simplify the model for faster inference.
    /// </para>
    /// </remarks>
    public Matrix<T> MergeWeights()
    {
        // Compute W_lora = A * B * scaling
        // A: [inputSize, rank], B: [rank, outputSize]
        // Result: [inputSize, outputSize] - matches DenseLayer's industry standard convention
        Matrix<T> merged = _loraA.Multiply(_loraB).Multiply(_scaling);
        return merged;
    }

    /// <summary>
    /// Gets the rank of this LoRA layer.
    /// </summary>
    public int Rank => _rank;

    /// <summary>
    /// Gets the alpha scaling factor.
    /// </summary>
    public T Alpha => _alpha;

    /// <summary>
    /// Gets the computed scaling factor (alpha / rank).
    /// </summary>
    public T Scaling => _scaling;

    /// <summary>
    /// Gets matrix A (for inspection or advanced use cases).
    /// </summary>
    public Matrix<T> GetMatrixA() => _loraA.Clone();

    /// <summary>
    /// Gets matrix B (for inspection or advanced use cases).
    /// </summary>
    public Matrix<T> GetMatrixB() => _loraB.Clone();

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For LoRA layers, this clears the stored input from the last forward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This clears the layer's memory of the last input it processed.
    /// It's like hitting a reset button before processing a new, unrelated batch of data.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        _lastInput = null;
        _lastPreActivation = null;
        _loraAGradient = null;
        _loraBGradient = null;
    }

    /// <summary>
    /// Gets whether this LoRA layer supports JIT compilation.
    /// </summary>
    /// <value>True if the LoRA matrices are initialized.</value>
    /// <remarks>
    /// <para>
    /// LoRA layers support JIT compilation when their matrices (A and B) are properly initialized.
    /// The JIT-compiled version computes output = input * A * B * scaling using optimized tensor operations.
    /// </para>
    /// <para><b>For Beginners:</b> JIT compilation makes the LoRA layer run faster by converting
    /// its math operations into optimized native code. This is especially beneficial for inference
    /// when you want maximum speed.
    ///
    /// The layer can be JIT compiled as long as it has been initialized, which happens automatically
    /// when the layer is created.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => _loraA != null && _loraB != null;

    /// <summary>
    /// Exports the computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to which input nodes will be added.</param>
    /// <returns>The output computation node representing the LoRA transformation.</returns>
    /// <exception cref="ArgumentNullException">Thrown when inputNodes is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when matrices are not initialized.</exception>
    /// <remarks>
    /// <para>
    /// The computation graph implements: output = input * A * B * scaling
    /// where:
    /// - A is the low-rank projection matrix (inputSize × rank)
    /// - B is the reconstruction matrix (rank × outputSize)
    /// - scaling = alpha / rank
    /// </para>
    /// <para><b>For Beginners:</b> This exports the LoRA computation as a graph of operations
    /// that can be optimized and compiled to fast native code.
    ///
    /// The graph represents:
    /// 1. Input → multiply by matrix A (compress to low rank)
    /// 2. Result → multiply by matrix B (expand to output size)
    /// 3. Result → multiply by scaling factor
    ///
    /// The JIT compiler can then fuse these operations and apply optimizations like SIMD vectorization.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (_loraA == null || _loraB == null)
            throw new InvalidOperationException("LoRA matrices not initialized. Initialize the layer first.");

        int inputSize = _loraA.Rows;
        int outputSize = _loraB.Columns;

        // Create input placeholder with symbolic batch dimension
        var inputPlaceholder = new Tensor<T>(new int[] { 1, inputSize });
        var inputNode = TensorOperations<T>.Variable(inputPlaceholder, "lora_input");

        // Create constant nodes for matrix A [inputSize, rank]
        var matrixATensor = new Tensor<T>(new int[] { _loraA.Rows, _loraA.Columns }, _loraA);
        var matrixANode = TensorOperations<T>.Constant(matrixATensor, "lora_A");

        // Create constant node for matrix B [rank, outputSize]
        var matrixBTensor = new Tensor<T>(new int[] { _loraB.Rows, _loraB.Columns }, _loraB);
        var matrixBNode = TensorOperations<T>.Constant(matrixBTensor, "lora_B");

        // Create constant node for scaling factor
        var scalingTensor = new Tensor<T>(new int[] { 1 }, new Vector<T>(new[] { _scaling }));
        var scalingNode = TensorOperations<T>.Constant(scalingTensor, "lora_scaling");

        // Add input nodes
        inputNodes.Add(inputNode);
        inputNodes.Add(matrixANode);
        inputNodes.Add(matrixBNode);
        inputNodes.Add(scalingNode);

        // Build computation graph: output = input * A * B * scaling
        // Step 1: input * A -> [batch, rank]
        var intermediateNode = TensorOperations<T>.MatrixMultiply(inputNode, matrixANode);

        // Step 2: intermediate * B -> [batch, outputSize]
        var preScaledNode = TensorOperations<T>.MatrixMultiply(intermediateNode, matrixBNode);

        // Step 3: Apply scaling (element-wise multiply by scalar)
        var outputNode = TensorOperations<T>.ElementwiseMultiply(preScaledNode, scalingNode);

        // Apply activation using the inherited LayerBase method which properly delegates
        // to the activation function's ApplyToGraph method (Open/Closed Principle)
        var activatedOutput = ApplyActivationToGraph(outputNode);

        return activatedOutput;
    }
}

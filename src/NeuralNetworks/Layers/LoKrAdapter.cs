using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// LoKr (Low-Rank Kronecker Product Adaptation) adapter for parameter-efficient fine-tuning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// LoKr uses Kronecker products instead of standard matrix multiplication for low-rank adaptation.
/// Instead of computing ΔW = A × B (standard LoRA), LoKr computes ΔW = A ⊗ B where ⊗ is the
/// Kronecker product. This is particularly efficient for very large weight matrices.
/// </para>
/// <para><b>Kronecker Product Definition:</b>
/// For matrices A (m×n) and B (p×q), the Kronecker product A ⊗ B is an (m×p) × (n×q) matrix:
///
/// A ⊗ B = [a₁₁B  a₁₂B  ...  a₁ₙB]
///         [a₂₁B  a₂₂B  ...  a₂ₙB]
///         [  ⋮     ⋮    ⋱    ⋮  ]
///         [aₘ₁B  aₘ₂B  ...  aₘₙB]
///
/// Each element aᵢⱼ of A is multiplied by the entire matrix B, creating a block structure.
/// </para>
/// <para><b>For Beginners:</b> LoKr is a variant of LoRA that uses a different mathematical operation
/// called the Kronecker product. Think of it this way:
///
/// - Standard LoRA: Multiplies two small matrices (like 1000×8 and 8×1000) to approximate changes
/// - LoKr: Uses Kronecker product of two even smaller matrices (like 50×4 and 20×4) to create the same size output
///
/// The Kronecker product creates a larger matrix by taking every element of the first matrix and
/// multiplying it by the entire second matrix. This creates a block pattern that's very efficient
/// for representing certain types of structured transformations.
///
/// <b>When to use LoKr vs standard LoRA:</b>
/// - LoKr is better for very wide or very deep layers (e.g., 10000×10000 weight matrices)
/// - LoKr can achieve similar expressiveness with fewer parameters than LoRA
/// - Standard LoRA is simpler and works well for typical layer sizes
///
/// <b>Parameter Efficiency Example:</b>
/// For a 1000×1000 weight matrix with rank r=8:
/// - Standard LoRA: 1000×8 + 8×1000 = 16,000 parameters
/// - LoKr: 50×4 + 20×4 = 200 + 80 = 280 parameters (57x fewer!)
///   (where 50×20 = 1000 for both dimensions)
/// </para>
/// </remarks>
public class LoKrAdapter<T> : LoRAAdapterBase<T>
{
    /// <summary>
    /// First Kronecker factor matrix A with dimensions (m × n).
    /// </summary>
    /// <remarks>
    /// This is one of the two matrices used in the Kronecker product decomposition.
    /// </remarks>
    private Matrix<T> _matrixA;

    /// <summary>
    /// Second Kronecker factor matrix B with dimensions (p × q).
    /// </summary>
    /// <remarks>
    /// This is the second matrix used in the Kronecker product decomposition.
    /// The Kronecker product A ⊗ B produces a (m×p) × (n×q) matrix.
    /// </remarks>
    private Matrix<T> _matrixB;

    /// <summary>
    /// Scaling factor for the LoKr contribution.
    /// </summary>
    private readonly T _alpha;

    /// <summary>
    /// Computed scaling factor (alpha / effective_rank) used during forward pass.
    /// </summary>
    private readonly T _scaling;

    /// <summary>
    /// Gradients for matrix A computed during backpropagation.
    /// </summary>
    private Matrix<T>? _gradientA;

    /// <summary>
    /// Gradients for matrix B computed during backpropagation.
    /// </summary>
    private Matrix<T>? _gradientB;

    /// <summary>
    /// Stored input from the forward pass, needed for gradient computation.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Dimensions for matrix A (m, n).
    /// </summary>
    private readonly (int m, int n) _dimsA;

    /// <summary>
    /// Dimensions for matrix B (p, q).
    /// </summary>
    private readonly (int p, int q) _dimsB;

    /// <summary>
    /// Gets the total number of trainable parameters (elements in A and B matrices).
    /// </summary>
    public override int ParameterCount => (_matrixA.Rows * _matrixA.Columns) + (_matrixB.Rows * _matrixB.Columns);

    /// <summary>
    /// Initializes a new LoKr adapter wrapping an existing layer.
    /// </summary>
    /// <param name="baseLayer">The layer to adapt with LoKr.</param>
    /// <param name="rank">The effective rank of the decomposition (used to determine factor matrix sizes).</param>
    /// <param name="alpha">The LoKr scaling factor (defaults to rank if negative).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer's parameters during training.</param>
    /// <exception cref="ArgumentNullException">Thrown when baseLayer is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the base layer doesn't have 1D input/output shapes.</exception>
    /// <remarks>
    /// <para>
    /// The LoKr matrices are initialized as follows:
    /// - Matrix A: Random values from a Gaussian distribution
    /// - Matrix B: Zero initialization (so LoKr starts with no effect)
    ///
    /// The dimensions of A and B are chosen such that A ⊗ B produces a matrix that can be applied
    /// to the layer's weights. For a layer with inputSize and outputSize, we factor these dimensions
    /// to create A (m×n) and B (p×q) where m×p = outputSize and n×q = inputSize.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a LoKr adapter for a layer. The rank parameter determines
    /// how the weight matrix is factored into two smaller matrices. Lower rank = fewer parameters but
    /// less flexibility.
    ///
    /// The adapter automatically figures out the best sizes for matrices A and B based on your layer's
    /// input and output sizes and the rank you specify.
    /// </para>
    /// </remarks>
    public LoKrAdapter(ILayer<T> baseLayer, int rank, double alpha = -1, bool freezeBaseLayer = true)
        : base(baseLayer, rank, alpha, freezeBaseLayer)
    {
        // Validate base layer has single-dimensional input/output
        if (baseLayer.GetInputShape().Length != 1 || baseLayer.GetOutputShape().Length != 1)
        {
            throw new ArgumentException("LoKrAdapter only supports layers with 1D input/output shapes", nameof(baseLayer));
        }

        int inputSize = baseLayer.GetInputShape()[0];
        int outputSize = baseLayer.GetOutputShape()[0];

        // Factor the dimensions to create Kronecker factors
        // We want m*p = outputSize and n*q = inputSize, with balanced factors
        _dimsA = FactorDimension(outputSize, rank);
        _dimsB = (outputSize / _dimsA.m, inputSize / _dimsA.n);

        // Verify factorization is valid
        if (_dimsA.m * _dimsB.p != outputSize || _dimsA.n * _dimsB.q != inputSize)
        {
            throw new ArgumentException(
                $"Cannot factor dimensions for LoKr: outputSize={outputSize}, inputSize={inputSize}, rank={rank}. " +
                "Try a different rank value or use dimensions that are more easily factorizable.");
        }

        // Initialize matrices
        _matrixA = new Matrix<T>(_dimsA.m, _dimsA.n);
        _matrixB = new Matrix<T>(_dimsB.p, _dimsB.q);

        // Default alpha to rank if not specified
        _alpha = alpha > 0 ? NumOps.FromDouble(alpha) : NumOps.FromDouble(rank);
        int effectiveRank = _dimsA.n * _dimsB.q;
        _scaling = NumOps.Divide(_alpha, NumOps.FromDouble(effectiveRank));

        // Initialize matrix A with random values (Gaussian with std = 1/sqrt(effectiveRank))
        T stddev = NumOps.Sqrt(NumOps.Divide(NumOps.One, NumOps.FromDouble(effectiveRank)));
        for (int i = 0; i < _matrixA.Rows; i++)
        {
            for (int j = 0; j < _matrixA.Columns; j++)
            {
                double u1 = Random.NextDouble();
                double u2 = Random.NextDouble();
                double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                _matrixA[i, j] = NumOps.Multiply(NumOps.FromDouble(randStdNormal), stddev);
            }
        }

        // Initialize matrix B with zeros (so LoKr has no effect initially)
        for (int i = 0; i < _matrixB.Rows; i++)
        {
            for (int j = 0; j < _matrixB.Columns; j++)
            {
                _matrixB[i, j] = NumOps.Zero;
            }
        }

        // Initialize parameter vector
        Parameters = new Vector<T>(ParameterCount);
        UpdateParametersFromMatrices();
    }

    /// <summary>
    /// Factors a dimension into two factors based on the desired rank.
    /// </summary>
    /// <param name="size">The dimension to factor.</param>
    /// <param name="rank">The desired effective rank.</param>
    /// <returns>Two factors (m, n) such that their product approximates size.</returns>
    /// <remarks>
    /// This tries to create balanced factors for better numerical stability.
    /// </remarks>
    private static (int m, int n) FactorDimension(int size, int rank)
    {
        // Try to find balanced factors based on rank
        // We want m and n such that m*p ≈ size and n is related to rank
        int n = Math.Min(rank, (int)Math.Sqrt(size));
        int m = size / n;

        // Adjust if not evenly divisible
        while (size % m != 0 && m > 1)
        {
            m--;
        }
        n = size / m;

        return (m, n);
    }

    /// <summary>
    /// Computes the Kronecker product of two matrices.
    /// </summary>
    /// <param name="a">First matrix (m × n).</param>
    /// <param name="b">Second matrix (p × q).</param>
    /// <returns>Kronecker product A ⊗ B of size (m×p) × (n×q).</returns>
    /// <remarks>
    /// <para>
    /// The Kronecker product creates a block matrix where each element a[i,j] is multiplied
    /// by the entire matrix B. The result has a characteristic block structure.
    /// </para>
    /// <para><b>For Beginners:</b> The Kronecker product is like creating a grid of copies of matrix B,
    /// where each copy is scaled by a different element from matrix A. If A is 2×2 and B is 3×3,
    /// the result is a 6×6 matrix with 4 blocks (each 3×3).
    /// </para>
    /// </remarks>
    private Matrix<T> KroneckerProduct(Matrix<T> a, Matrix<T> b)
    {
        int m = a.Rows;
        int n = a.Columns;
        int p = b.Rows;
        int q = b.Columns;

        Matrix<T> result = new Matrix<T>(m * p, n * q);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                T aij = a[i, j];
                for (int k = 0; k < p; k++)
                {
                    for (int l = 0; l < q; l++)
                    {
                        result[i * p + k, j * q + l] = NumOps.Multiply(aij, b[k, l]);
                    }
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Performs the forward pass through both base and LoKr layers.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Sum of base layer output and LoKr output.</returns>
    /// <remarks>
    /// <para>
    /// The forward pass computes: output = base_layer(input) + (A ⊗ B) * input * scaling
    /// </para>
    /// <para><b>For Beginners:</b> This runs the input through both the original layer and the
    /// LoKr adaptation layer (using Kronecker product), then adds their outputs together.
    /// The result is the original behavior plus the learned Kronecker-factored adaptation.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input.Clone();

        // Forward through base layer
        Tensor<T> baseOutput = _baseLayer.Forward(input);

        // Compute Kronecker product delta = A ⊗ B
        Matrix<T> kronDelta = KroneckerProduct(_matrixA, _matrixB);

        // Apply to input: delta * input
        int batchSize = input.Shape[0];
        int inputSize = input.Shape.Length > 1 ? input.Shape[1] : input.Length;
        int outputSize = kronDelta.Rows;

        // Convert input to matrix [batchSize, inputSize]
        Matrix<T> inputMatrix = new Matrix<T>(batchSize, inputSize);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                inputMatrix[i, j] = input[i * inputSize + j];
            }
        }

        // Compute: input * kronDelta^T (because kronDelta is outputSize × inputSize)
        Matrix<T> deltaOutput = inputMatrix.Multiply(kronDelta.Transpose());

        // Apply scaling
        deltaOutput = deltaOutput.Multiply(_scaling);

        // Convert LoKr output to tensor and add to base output
        Tensor<T> result = new Tensor<T>(baseOutput.Shape);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                int idx = i * outputSize + j;
                result[idx] = NumOps.Add(baseOutput[idx], deltaOutput[i, j]);
            }
        }

        return result;
    }

    /// <summary>
    /// Performs the backward pass through both layers.
    /// </summary>
    /// <param name="outputGradient">Gradient flowing back from the next layer.</param>
    /// <returns>Gradient to pass to the previous layer.</returns>
    /// <remarks>
    /// <para>
    /// The backward pass computes gradients through the Kronecker product using the vec-trick
    /// for efficient gradient computation. The gradients are:
    /// - dL/dA uses the Kronecker structure to extract A-specific gradients
    /// - dL/dB uses the Kronecker structure to extract B-specific gradients
    /// - Input gradients flow through both paths and are summed
    /// </para>
    /// <para><b>For Beginners:</b> This figures out how to improve both the base layer and the
    /// LoKr matrices (A and B). It uses the special structure of the Kronecker product to
    /// efficiently compute gradients without having to work with the full Kronecker product matrix.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass");
        }

        // Backward through base layer
        Tensor<T> baseInputGrad = _baseLayer.Backward(outputGradient);

        // Compute gradients for LoKr matrices using Kronecker product properties
        int batchSize = _lastInput.Shape[0];
        int inputSize = _lastInput.Shape.Length > 1 ? _lastInput.Shape[1] : _lastInput.Length;
        int outputSize = outputGradient.Shape.Length > 1 ? outputGradient.Shape[1] : outputGradient.Length;

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

        // Use vec-trick for Kronecker gradient computation
        // For ΔW = A ⊗ B, the gradients are computed by reshaping and using Kronecker properties
        _gradientA = KroneckerGradientA(inputMatrix, gradMatrix, _matrixB);
        _gradientB = KroneckerGradientB(inputMatrix, gradMatrix, _matrixA);

        // Scale gradients
        _gradientA = _gradientA.Multiply(_scaling);
        _gradientB = _gradientB.Multiply(_scaling);

        // Compute input gradients through Kronecker product
        Matrix<T> kronDelta = KroneckerProduct(_matrixA, _matrixB);
        Matrix<T> loraInputGrad = gradMatrix.Multiply(kronDelta).Multiply(_scaling);

        // Sum input gradients from both paths
        Tensor<T> inputGrad = new Tensor<T>(baseInputGrad.Shape);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                int idx = i * inputSize + j;
                inputGrad[idx] = NumOps.Add(baseInputGrad[idx], loraInputGrad[i, j]);
            }
        }

        // Update parameter gradients vector
        UpdateParameterGradientsFromMatrices();

        return inputGrad;
    }

    /// <summary>
    /// Computes the gradient for matrix A using Kronecker product properties.
    /// </summary>
    /// <param name="input">Input matrix [batchSize, inputSize].</param>
    /// <param name="outputGrad">Output gradient matrix [batchSize, outputSize].</param>
    /// <param name="matrixB">The B matrix in the Kronecker product.</param>
    /// <returns>Gradient for matrix A.</returns>
    /// <remarks>
    /// Uses the vec-trick: vec(A ⊗ B) = (I_m ⊗ B) vec(A), which allows efficient gradient computation.
    /// </remarks>
    private Matrix<T> KroneckerGradientA(Matrix<T> input, Matrix<T> outputGrad, Matrix<T> matrixB)
    {
        int batchSize = input.Rows;
        Matrix<T> gradA = new Matrix<T>(_dimsA.m, _dimsA.n);

        // Reshape output gradient into blocks and compute gradient for A
        // This uses the property that ∂(A ⊗ B)/∂A can be computed efficiently
        for (int i = 0; i < _dimsA.m; i++)
        {
            for (int j = 0; j < _dimsA.n; j++)
            {
                T sum = NumOps.Zero;

                for (int batch = 0; batch < batchSize; batch++)
                {
                    // Extract the corresponding block from output gradient
                    for (int p = 0; p < _dimsB.p; p++)
                    {
                        for (int q = 0; q < _dimsB.q; q++)
                        {
                            int outRow = i * _dimsB.p + p;
                            int inCol = j * _dimsB.q + q;

                            T grad = outputGrad[batch, outRow];
                            T inp = input[batch, inCol];
                            T b = matrixB[p, q];

                            sum = NumOps.Add(sum, NumOps.Multiply(NumOps.Multiply(grad, inp), b));
                        }
                    }
                }

                gradA[i, j] = sum;
            }
        }

        return gradA;
    }

    /// <summary>
    /// Computes the gradient for matrix B using Kronecker product properties.
    /// </summary>
    /// <param name="input">Input matrix [batchSize, inputSize].</param>
    /// <param name="outputGrad">Output gradient matrix [batchSize, outputSize].</param>
    /// <param name="matrixA">The A matrix in the Kronecker product.</param>
    /// <returns>Gradient for matrix B.</returns>
    /// <remarks>
    /// Uses the vec-trick for efficient gradient computation through the Kronecker structure.
    /// </remarks>
    private Matrix<T> KroneckerGradientB(Matrix<T> input, Matrix<T> outputGrad, Matrix<T> matrixA)
    {
        int batchSize = input.Rows;
        Matrix<T> gradB = new Matrix<T>(_dimsB.p, _dimsB.q);

        // Compute gradient for B using Kronecker product properties
        for (int p = 0; p < _dimsB.p; p++)
        {
            for (int q = 0; q < _dimsB.q; q++)
            {
                T sum = NumOps.Zero;

                for (int batch = 0; batch < batchSize; batch++)
                {
                    // Extract the corresponding elements using Kronecker structure
                    for (int i = 0; i < _dimsA.m; i++)
                    {
                        for (int j = 0; j < _dimsA.n; j++)
                        {
                            int outRow = i * _dimsB.p + p;
                            int inCol = j * _dimsB.q + q;

                            T grad = outputGrad[batch, outRow];
                            T inp = input[batch, inCol];
                            T a = matrixA[i, j];

                            sum = NumOps.Add(sum, NumOps.Multiply(NumOps.Multiply(grad, inp), a));
                        }
                    }
                }

                gradB[p, q] = sum;
            }
        }

        return gradB;
    }

    /// <summary>
    /// Updates the layer's parameters using the specified learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    public override void UpdateParameters(T learningRate)
    {
        // Always update LoKr matrices
        if (_gradientA != null && _gradientB != null)
        {
            UpdateMatricesWithGradients(learningRate);
        }

        // Update base layer if not frozen
        if (!_freezeBaseLayer)
        {
            _baseLayer.UpdateParameters(learningRate);
        }

        // Update parameter vector
        UpdateParametersFromMatrices();
    }

    /// <summary>
    /// Updates matrices A and B using their gradients.
    /// </summary>
    private void UpdateMatricesWithGradients(T learningRate)
    {
        if (_gradientA == null || _gradientB == null)
        {
            return;
        }

        // Update matrix A
        for (int i = 0; i < _matrixA.Rows; i++)
        {
            for (int j = 0; j < _matrixA.Columns; j++)
            {
                T update = NumOps.Multiply(_gradientA[i, j], learningRate);
                _matrixA[i, j] = NumOps.Subtract(_matrixA[i, j], update);
            }
        }

        // Update matrix B
        for (int i = 0; i < _matrixB.Rows; i++)
        {
            for (int j = 0; j < _matrixB.Columns; j++)
            {
                T update = NumOps.Multiply(_gradientB[i, j], learningRate);
                _matrixB[i, j] = NumOps.Subtract(_matrixB[i, j], update);
            }
        }
    }

    /// <summary>
    /// Merges the LoKr adaptation into the base layer and returns the merged layer.
    /// </summary>
    /// <returns>A new layer with LoKr weights merged into the base layer's weights.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the base layer type is not DenseLayer or FullyConnectedLayer.</exception>
    /// <remarks>
    /// <para>
    /// This computes the full Kronecker product A ⊗ B and adds it to the base layer's weights.
    /// </para>
    /// <para><b>For Beginners:</b> This "bakes in" your LoKr adaptation to create a regular layer.
    /// It computes the full Kronecker product matrix and adds it to the original weights, creating
    /// a single merged layer that's faster for inference.
    /// </para>
    /// </remarks>
    public override ILayer<T> MergeToOriginalLayer()
    {
        // Support both DenseLayer and FullyConnectedLayer
        DenseLayer<T>? denseBase = _baseLayer as DenseLayer<T>;
        FullyConnectedLayer<T>? fcBase = _baseLayer as FullyConnectedLayer<T>;

        if (denseBase == null && fcBase == null)
        {
            throw new InvalidOperationException("LoKrAdapter only supports DenseLayer or FullyConnectedLayer base layers");
        }

        // Compute full Kronecker product
        Matrix<T> kronWeights = KroneckerProduct(_matrixA, _matrixB);

        // Apply scaling
        kronWeights = kronWeights.Multiply(_scaling);

        // Get base layer parameters
        Vector<T> baseParams = _baseLayer.GetParameters();

        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        int weightCount = inputSize * outputSize;

        // Create new parameters with merged weights
        Vector<T> mergedParams = new Vector<T>(baseParams.Length);

        // Merge weights (kronWeights is outputSize × inputSize, same as base weights)
        for (int i = 0; i < weightCount; i++)
        {
            int row = i / inputSize;
            int col = i % inputSize;
            mergedParams[i] = NumOps.Add(baseParams[i], kronWeights[row, col]);
        }

        // Copy biases unchanged
        for (int i = weightCount; i < baseParams.Length; i++)
        {
            mergedParams[i] = baseParams[i];
        }

        // Create a new dense layer with merged parameters
        DenseLayer<T> mergedLayer = new DenseLayer<T>(inputSize, outputSize, (IActivationFunction<T>?)null);
        mergedLayer.SetParameters(mergedParams);

        return mergedLayer;
    }

    /// <summary>
    /// Gets the current parameters as a vector.
    /// </summary>
    /// <returns>Vector containing parameters (LoKr only if base is frozen, otherwise both).</returns>
    public override Vector<T> GetParameters()
    {
        return Parameters.Clone();
    }

    /// <summary>
    /// Sets the layer parameters from a vector.
    /// </summary>
    /// <param name="parameters">Vector containing parameters.</param>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException($"Expected {ParameterCount} parameters, got {parameters.Length}", nameof(parameters));
        }

        Parameters = parameters.Clone();
        UpdateMatricesFromParameters();
    }

    /// <summary>
    /// Updates the parameter vector from the current matrix states.
    /// </summary>
    private void UpdateParametersFromMatrices()
    {
        int idx = 0;

        // Pack matrix A
        for (int i = 0; i < _matrixA.Rows; i++)
        {
            for (int j = 0; j < _matrixA.Columns; j++)
            {
                Parameters[idx++] = _matrixA[i, j];
            }
        }

        // Pack matrix B
        for (int i = 0; i < _matrixB.Rows; i++)
        {
            for (int j = 0; j < _matrixB.Columns; j++)
            {
                Parameters[idx++] = _matrixB[i, j];
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
        for (int i = 0; i < _matrixA.Rows; i++)
        {
            for (int j = 0; j < _matrixA.Columns; j++)
            {
                _matrixA[i, j] = Parameters[idx++];
            }
        }

        // Unpack matrix B
        for (int i = 0; i < _matrixB.Rows; i++)
        {
            for (int j = 0; j < _matrixB.Columns; j++)
            {
                _matrixB[i, j] = Parameters[idx++];
            }
        }
    }

    /// <summary>
    /// Updates the parameter gradients vector from the matrix gradients.
    /// </summary>
    private void UpdateParameterGradientsFromMatrices()
    {
        if (_gradientA == null || _gradientB == null)
        {
            return;
        }

        ParameterGradients = new Vector<T>(ParameterCount);
        int idx = 0;

        // Pack matrix A gradients
        for (int i = 0; i < _gradientA.Rows; i++)
        {
            for (int j = 0; j < _gradientA.Columns; j++)
            {
                ParameterGradients[idx++] = _gradientA[i, j];
            }
        }

        // Pack matrix B gradients
        for (int i = 0; i < _gradientB.Rows; i++)
        {
            for (int j = 0; j < _gradientB.Columns; j++)
            {
                ParameterGradients[idx++] = _gradientB[i, j];
            }
        }
    }

    /// <summary>
    /// Resets the internal state of the adapter.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This clears the memory of the last input and gradients.
    /// It's useful when starting to process a completely new, unrelated batch of data.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        base.ResetState();
        _lastInput = null;
        _gradientA = null;
        _gradientB = null;
    }

    /// <summary>
    /// Gets the dimensions of matrix A.
    /// </summary>
    public (int m, int n) MatrixADimensions => _dimsA;

    /// <summary>
    /// Gets the dimensions of matrix B.
    /// </summary>
    public (int p, int q) MatrixBDimensions => _dimsB;

    /// <summary>
    /// Gets matrix A (for inspection or advanced use cases).
    /// </summary>
    public Matrix<T> GetMatrixA() => _matrixA.Clone();

    /// <summary>
    /// Gets matrix B (for inspection or advanced use cases).
    /// </summary>
    public Matrix<T> GetMatrixB() => _matrixB.Clone();
}

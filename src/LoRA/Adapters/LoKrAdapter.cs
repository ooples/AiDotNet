using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Extensions;
using AiDotNet.Interfaces;

namespace AiDotNet.LoRA.Adapters;

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
[AutoParameters]
public partial class LoKrAdapter<T> : LoRAAdapterBase<T>
{
    /// <summary>
    /// First Kronecker factor matrix A with dimensions (m × n).
    /// </summary>
    /// <remarks>
    /// This is one of the two matrices used in the Kronecker product decomposition.
    /// </remarks>
    [TrainableParameter]
    private Tensor<T> _matrixA;

    /// <summary>
    /// Second Kronecker factor matrix B with dimensions (p × q).
    /// </summary>
    /// <remarks>
    /// This is the second matrix used in the Kronecker product decomposition.
    /// The Kronecker product A ⊗ B produces a (m×p) × (n×q) matrix.
    /// </remarks>
    [TrainableParameter]
    private Tensor<T> _matrixB;

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
    [Scratch]
    private Matrix<T>? _gradientA;

    /// <summary>
    /// Gradients for matrix B computed during backpropagation.
    /// </summary>
    [Scratch]
    private Matrix<T>? _gradientB;

    /// <summary>
    /// Dimensions for matrix A (m, n).
    /// </summary>
    private readonly (int m, int n) _dimsA;

    /// <summary>
    /// Dimensions for matrix B (p, q).
    /// </summary>
    private readonly (int p, int q) _dimsB;

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
        int outputSize = baseLayer.GetOutputLayerShape().RequireConcrete("Sizing a LoRA adapter's low-rank factors")[0];

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
        _matrixA = new Tensor<T>([_dimsA.m, _dimsA.n]);
        _matrixB = new Tensor<T>([_dimsB.p, _dimsB.q]);

        // Default alpha to rank if not specified
        _alpha = alpha > 0 ? NumOps.FromDouble(alpha) : NumOps.FromDouble(rank);
        int effectiveRank = _dimsA.n * _dimsB.q;
        _scaling = NumOps.Divide(_alpha, NumOps.FromDouble(effectiveRank));

        // Initialize matrix A with random values (Gaussian with std = 1/sqrt(effectiveRank))
        T stddev = NumOps.Sqrt(NumOps.Divide(NumOps.One, NumOps.FromDouble(effectiveRank)));
        for (int i = 0; i < _matrixA.Shape[0]; i++)
        {
            for (int j = 0; j < _matrixA.Shape[1]; j++)
            {
                _matrixA[i, j] = NumOps.Multiply(NumOps.FromDouble(Random.NextGaussian()), stddev);
            }
        }

        // Matrix B starts at zero (a fresh tensor is zero-filled), so LoKr has no effect initially


        // LoKr replaces the standard LoRA factors with its Kronecker factors.
        FreezeSubLayerParameters(_loraLayer);
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

        // result[i * p + k, j * q + l] = a[i, j] * b[k, l]
        Matrix<T> result = Engine.TensorKron(Tensor<T>.FromMatrix(a), Tensor<T>.FromMatrix(b)).ToMatrix();

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
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        // Forward through base layer
        Tensor<T> baseOutput = _baseLayer.Forward(input);

        // Use vec-trick to avoid materializing full Kronecker product
        // For (B ⊗ A) vec(X) = vec(A * X * B^T)
        int batchSize = input.Shape[0];
        int inputSize = input.Shape.Length > 1 ? input.Shape[1] : input.Length;

        // Dimensions: A is [p, q], B is [m, n]
        // Kronecker product (B ⊗ A) would be [(m*p), (n*q)]
        int p = _matrixA.Shape[0];
        int q = _matrixA.Shape[1];
        int m = _matrixB.Shape[0];
        int n = _matrixB.Shape[1];

        // inputSize should equal n*q, outputSize should equal m*p
        if (inputSize != n * q)
        {
            throw new InvalidOperationException($"Input size {inputSize} doesn't match expected {n * q} for Kronecker dimensions");
        }

        int outputSize = m * p;

        // Delta weight is W = A ⊗ B ([p*m, q*n], the same layout MergeToOriginalLayer and the
        // Kronecker gradients use), so per sample y = W x is Y = A * X * B^T with
        // X = reshape(x, [q, n]) and y = vec_rowmajor(Y) ([p, m]). Batched:
        //   X * B^T over all samples -> [batch, q, m] -> [q, batch * m]; A * that -> [p, batch, m].
        Tensor<T> xb = Engine.TensorMatMul(
            Engine.Reshape(input, [batchSize * q, n]), Engine.TensorTranspose(_matrixB));
        Tensor<T> xbByQ = Engine.Reshape(
            Engine.TensorPermute(Engine.Reshape(xb, [batchSize, q, m]), [1, 0, 2]), [q, batchSize * m]);
        Tensor<T> y = Engine.TensorMatMul(_matrixA, xbByQ);
        Tensor<T> deltaOutput = Engine.TensorMultiplyScalar(
            Engine.Reshape(Engine.TensorPermute(Engine.Reshape(y, [p, batchSize, m]), [1, 0, 2]), [batchSize, outputSize]),
            _scaling);

        // Add LoKr output to base output
        Tensor<T> result = Engine.TensorAdd(baseOutput, Engine.Reshape(deltaOutput, baseOutput.Shape.ToArray()));

        return result;
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

        // Compute full outer product: G = outputGrad^T @ input via Engine.TensorMatMul
        // G has shape [outputSize, inputSize] = [mA*pB, nA*qB]
        var gradT = Tensor<T>.FromMatrix(outputGrad).Transpose(new[] { 1, 0 });
        var inputT = Tensor<T>.FromMatrix(input);
        var fullGrad = Engine.TensorMatMul(gradT, inputT); // [mA*pB, nA*qB]

        // Contract with B to get gradA: gradA[i,j] = sum_p,q(fullGrad[i*pB+p, j*qB+q] * B[p,q])
        Tensor<T> blocks = Engine.Reshape(
            Engine.TensorPermute(Engine.Reshape(fullGrad, [_dimsA.m, _dimsB.p, _dimsA.n, _dimsB.q]), [0, 2, 1, 3]),
            [_dimsA.m * _dimsA.n, _dimsB.p * _dimsB.q]);
        Matrix<T> gradA = Engine.Reshape(
            Engine.TensorMatMul(blocks, Engine.Reshape(Tensor<T>.FromMatrix(matrixB), [_dimsB.p * _dimsB.q, 1])),
            [_dimsA.m, _dimsA.n]).ToMatrix();

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
        // Compute full outer product: G = outputGrad^T @ input via Engine.TensorMatMul
        // G has shape [outputSize, inputSize] = [mA*pB, nA*qB]
        var gradT = Tensor<T>.FromMatrix(outputGrad).Transpose(new[] { 1, 0 });
        var inputT = Tensor<T>.FromMatrix(input);
        var fullGrad = Engine.TensorMatMul(gradT, inputT); // [mA*pB, nA*qB]

        // Contract with A to get gradB: gradB[p,q] = sum_i,j(fullGrad[i*pB+p, j*qB+q] * A[i,j])
        Tensor<T> blocks = Engine.Reshape(
            Engine.TensorPermute(Engine.Reshape(fullGrad, [_dimsA.m, _dimsB.p, _dimsA.n, _dimsB.q]), [1, 3, 0, 2]),
            [_dimsB.p * _dimsB.q, _dimsA.m * _dimsA.n]);
        Matrix<T> gradB = Engine.Reshape(
            Engine.TensorMatMul(blocks, Engine.Reshape(Tensor<T>.FromMatrix(matrixA), [_dimsA.m * _dimsA.n, 1])),
            [_dimsB.p, _dimsB.q]).ToMatrix();

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

        // A -= learningRate * dA, B -= learningRate * dB
        Engine.TensorSubtractInPlace(_matrixA, Tensor<T>.FromMatrix(Engine.MatrixMultiplyScalar(_gradientA, learningRate)));
        Engine.TensorSubtractInPlace(_matrixB, Tensor<T>.FromMatrix(Engine.MatrixMultiplyScalar(_gradientB, learningRate)));
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
        Matrix<T> kronWeights = KroneckerProduct(_matrixA.ToMatrix(), _matrixB.ToMatrix());

        // Apply scaling
        kronWeights = kronWeights.Multiply(_scaling);

        // Get base layer parameters
        Vector<T> baseParams = _baseLayer.GetParameters();

        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputLayerShape().RequireConcrete("Sizing a LoRA adapter's low-rank factors")[0];
        int weightCount = inputSize * outputSize;

        // Merge weights (kronWeights is outputSize × inputSize). FullyConnectedLayer stores its weights
        // output-major, DenseLayer input-major ([inputSize, outputSize]), so DenseLayer uses the
        // transpose. Biases are kept unchanged.
        bool isOutputMajor = _baseLayer is FullyConnectedLayer<T>;
        Vector<T> mergedParams = AddFlatWeightsToParameters(
            baseParams, isOutputMajor ? kronWeights : kronWeights.Transpose(), weightCount);

        // Use helper method to clone base layer and preserve activation function
        return CreateMergedLayerWithClone(mergedParams);
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
    public Matrix<T> GetMatrixA() => _matrixA.ToMatrix();

    /// <summary>
    /// Gets matrix B (for inspection or advanced use cases).
    /// </summary>
    public Matrix<T> GetMatrixB() => _matrixB.ToMatrix();
}

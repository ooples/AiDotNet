using AiDotNet.Interfaces;

namespace AiDotNet.LoRA.Adapters;

/// <summary>
/// Implements MoRA (High-Rank Updating for Parameter-Efficient Fine-Tuning) adapter.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// <b>Paper Reference:</b> "MoRA: High-Rank Updating for Parameter-Efficient Fine-Tuning"
/// by Ting Jiang, Shaohan Huang, et al. (arXiv:2405.12130, May 2024)
/// </para>
/// <para>
/// MoRA addresses a fundamental limitation of LoRA: the low-rank constraint restricts the model's
/// ability to learn and memorize new knowledge. While LoRA uses two rectangular matrices (A and B)
/// to create low-rank updates, MoRA uses a single square matrix M combined with non-parameter-sharing
/// operators to achieve high-rank updates while maintaining the same parameter count.
/// </para>
/// <para><b>Key Innovations:</b>
///
/// 1. <b>High-Rank Updates</b>: Unlike LoRA's rank-r updates (r &lt;&lt; d), MoRA achieves rank-r̂
///    updates where r̂ can equal the full dimension d, enabling the model to learn richer representations.
///
/// 2. <b>Square Matrix M</b>: Instead of LoRA's A (d×r) and B (r×d) matrices, MoRA uses a single
///    square matrix M (r×r) where r = sqrt(d×d / 2). For the same parameter count as LoRA,
///    MoRA achieves much higher effective rank.
///
/// 3. <b>Non-Parameter-Sharing Operators</b>: MoRA uses rotation, permutation, or other linear
///    transformations that don't add trainable parameters but enable dimension compression
///    and decompression around the square matrix M.
///
/// 4. <b>Input Compression / Output Decompression</b>: The architecture is:
///    - Compress: Input (d) to Compressed (r) via rotation/permutation
///    - Transform: Compressed (r) to Transformed (r) via trainable matrix M
///    - Decompress: Transformed (r) to Output (d) via inverse rotation/permutation
/// </para>
/// <para><b>Architecture Comparison:</b>
///
/// LoRA: W = W₀ + BA where A ∈ ℝ^(d×r), B ∈ ℝ^(r×d)
/// - Parameters: 2dr
/// - Rank: r (low-rank constraint)
/// - Typical r: 8-64
///
/// MoRA: W = W₀ + R_d^(-1) M R_c where M ∈ ℝ^(r×r)
/// - Parameters: r²
/// - Rank: min(r, d) (can be full-rank)
/// - For same param count as LoRA: r = sqrt(2dr), so rank ≈ sqrt(2dr)
/// - Example: LoRA with r=8, d=1024 has 16,384 params and rank 8
///            MoRA with same params: r=128, rank 128 (16× higher!)
/// </para>
/// <para><b>Performance (from paper):</b>
///
/// Compared to LoRA on various tasks:
/// - <b>Memory-Intensive Tasks</b>: MoRA significantly outperforms LoRA
///   * Continual Pretraining: ~15% better perplexity
///   * Instruction Tuning: ~8% better accuracy on knowledge-intensive QA
/// - <b>Reasoning Tasks</b>: MoRA performs comparably to LoRA
///   * Mathematical Reasoning: Similar performance (within 1-2%)
/// - <b>Parameter Efficiency</b>: Same parameter count as LoRA
/// - <b>Training Speed</b>: Slightly slower than LoRA due to rotation operations (≈5-10% overhead)
/// </para>
/// <para><b>When to Use MoRA vs LoRA:</b>
///
/// Use MoRA when:
/// - Task requires memorizing new facts or knowledge
/// - Domain adaptation with significant vocabulary changes
/// - Continual learning scenarios
/// - You need the model to "remember" rather than just "adapt"
///
/// Use LoRA when:
/// - Task is primarily reasoning or pattern recognition
/// - Minimal new knowledge acquisition needed
/// - Training speed is critical
/// - Standard parameter-efficient fine-tuning is sufficient
/// </para>
/// <para><b>Implementation Details:</b>
///
/// This implementation uses rotation matrices as the non-parameter-sharing operators:
/// - Compression R_c: Projects input from dimension d to dimension r
/// - Decompression R_d: Projects from dimension r back to dimension d
/// - These are generated using random orthogonal matrices (Gram-Schmidt orthogonalization)
/// - They remain fixed during training (non-trainable)
///
/// Alternative operators mentioned in the paper (not implemented here):
/// - RoPE-based rotations (Rotary Position Embeddings)
/// - Random permutations
/// - Structured rotations (e.g., Hadamard transforms)
/// </para>
/// <para><b>For Beginners:</b> MoRA is like an upgraded version of LoRA that can learn
/// more complex changes to a model while using the same amount of memory.
///
/// Think of it like this:
/// - LoRA is like having 2 small notebooks to write changes (matrices A and B)
/// - MoRA is like having 1 square notebook plus a compression/decompression scheme
///
/// The key insight: By compressing the input, applying changes in compressed space,
/// and then decompressing, MoRA can make higher-rank updates that capture more
/// complex patterns. This is especially useful when you're teaching the model
/// entirely new facts or concepts, not just adapting its existing knowledge.
///
/// Example: If you're fine-tuning a model to learn medical terminology, MoRA
/// will be better at memorizing the new terms, while LoRA might be better at
/// learning to reason about medical cases using existing knowledge.
/// </para>
/// </remarks>
public class MoRAAdapter<T> : LoRAAdapterBase<T>
{
    /// <summary>
    /// Square matrix M for high-rank adaptation (r×r dimensions).
    /// </summary>
    /// <remarks>
    /// This is the core trainable component of MoRA. Unlike LoRA's rectangular matrices,
    /// M is square with dimensions (r×r), enabling higher-rank updates.
    /// </remarks>
    private Matrix<T> _matrixM;

    /// <summary>
    /// Compression matrix that reduces input dimension from d to r (non-trainable).
    /// </summary>
    /// <remarks>
    /// This is a non-trainable orthogonal matrix that compresses the input.
    /// It's generated once during initialization using Gram-Schmidt orthogonalization and remains fixed.
    /// </remarks>
    private readonly Matrix<T> _compressionMatrix;

    /// <summary>
    /// Decompression matrix that expands dimension from r back to d (non-trainable).
    /// </summary>
    /// <remarks>
    /// This is a non-trainable orthogonal matrix that decompresses the output.
    /// In this implementation, it's the transpose of the compression matrix.
    /// </remarks>
    private readonly Matrix<T> _decompressionMatrix;

    /// <summary>
    /// The dimension of the square matrix M.
    /// </summary>
    /// <remarks>
    /// For MoRA, this is calculated to match the parameter count of LoRA.
    /// If LoRA uses 2dr parameters, MoRA uses r² = 2dr, so r̂ = sqrt(2dr).
    /// This gives MoRA a much higher effective rank than LoRA.
    /// </remarks>
    private readonly int _squareRank;

    /// <summary>
    /// Gradients for matrix M computed during backpropagation.
    /// </summary>
    private Matrix<T>? _matrixMGradient;

    /// <summary>
    /// Stored input from the forward pass, needed for gradient computation.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Cached compressed input from forward pass.
    /// </summary>
    private Matrix<T>? _lastCompressed;

    /// <summary>
    /// Gets the effective rank of the MoRA adaptation.
    /// </summary>
    /// <remarks>
    /// This is the dimension of the square matrix M, which determines the
    /// maximum rank of the updates MoRA can make. Unlike LoRA where this
    /// is typically 8-64, MoRA can achieve ranks of 128+ with the same
    /// parameter count.
    /// </remarks>
    public int SquareRank => _squareRank;

    public MoRAAdapter(ILayer<T> baseLayer, int rank, double alpha = 1.0, bool freezeBaseLayer = true)
        : base(baseLayer, rank, alpha, freezeBaseLayer)
    {
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];

        if (inputSize != outputSize)
        {
            throw new ArgumentException(
                $"MoRA requires square layers (input size = output size). Got input={inputSize}, output={outputSize}. " +
                "For non-square layers, use LoRA instead.", nameof(baseLayer));
        }

        int dimension = inputSize;
        _squareRank = (int)Math.Sqrt(2.0 * dimension * rank);

        if (_squareRank < 1)
        {
            _squareRank = 1;
        }

        if (_squareRank > dimension)
        {
            _squareRank = dimension;
        }

        _matrixM = new Matrix<T>(_squareRank, _squareRank);
        InitializeMatrixM();

        _compressionMatrix = GenerateOrthogonalMatrix(dimension, _squareRank);
        _decompressionMatrix = _compressionMatrix.Transpose();

        // CRITICAL: Reallocate Parameters and ParameterGradients now that _squareRank is set
        // The base constructor allocated them when _squareRank was 0, creating zero-length buffers
        RebuildParameterSnapshot();
    }

    private void InitializeMatrixM()
    {
        T stddev = NumOps.Sqrt(NumOps.Divide(NumOps.One, NumOps.FromDouble(_squareRank)));

        for (int i = 0; i < _matrixM.Rows; i++)
        {
            for (int j = 0; j < _matrixM.Columns; j++)
            {
                double u1 = Random.NextDouble();
                double u2 = Random.NextDouble();
                double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                _matrixM[i, j] = NumOps.Multiply(NumOps.FromDouble(randStdNormal), stddev);
            }
        }
    }

    /// <summary>
    /// Reallocates and repopulates the Parameters and ParameterGradients vectors.
    /// </summary>
    /// <remarks>
    /// Called after _squareRank and _matrixM are initialized to fix the zero-length
    /// buffers allocated by the base constructor when _squareRank was still 0.
    /// This ensures ParameterCount matches the actual Parameters buffer length.
    /// </remarks>
    private void RebuildParameterSnapshot()
    {
        int paramCount = ParameterCount;
        Parameters = new Vector<T>(paramCount);
        ParameterGradients = new Vector<T>(paramCount);

        UpdateParametersFromLayers();
    }

    /// <summary>
    /// Overrides the base parameter packing to use the MoRA matrix M instead of the placeholder LoRA layer.
    /// This ensures that the public parameter surface is consistent with ParameterCount.
    /// </summary>
    protected override void UpdateParametersFromLayers()
    {
        int idx = 0;

        // Pack base layer parameters if not frozen
        if (!_freezeBaseLayer)
        {
            Vector<T> baseParams = _baseLayer.GetParameters();
            for (int i = 0; i < baseParams.Length; i++)
            {
                Parameters[idx++] = baseParams[i];
            }
        }

        // If _matrixM is not initialized, do nothing.
        // RebuildParameterSnapshot will be called later to correctly pack the parameters.
        if (_matrixM == null)
        {
            return;
        }

        // Pack _matrixM parameters
        for (int row = 0; row < _matrixM.Rows; row++)
        {
            for (int col = 0; col < _matrixM.Columns; col++)
            {
                if (idx < Parameters.Length)
                {
                    Parameters[idx++] = _matrixM[row, col];
                }
            }
        }
    }

    private Matrix<T> GenerateOrthogonalMatrix(int rows, int cols)
    {
        Matrix<T> randomMatrix = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double u1 = Random.NextDouble();
                double u2 = Random.NextDouble();
                double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                randomMatrix[i, j] = NumOps.FromDouble(randStdNormal);
            }
        }

        Matrix<T> orthogonal = new Matrix<T>(rows, cols);

        for (int j = 0; j < cols; j++)
        {
            Vector<T> column = new Vector<T>(rows);
            for (int i = 0; i < rows; i++)
            {
                column[i] = randomMatrix[i, j];
            }

            for (int k = 0; k < j; k++)
            {
                Vector<T> prevColumn = new Vector<T>(rows);
                for (int i = 0; i < rows; i++)
                {
                    prevColumn[i] = orthogonal[i, k];
                }

                T dotProduct = NumOps.Zero;
                for (int i = 0; i < rows; i++)
                {
                    dotProduct = NumOps.Add(dotProduct, NumOps.Multiply(column[i], prevColumn[i]));
                }

                for (int i = 0; i < rows; i++)
                {
                    column[i] = NumOps.Subtract(column[i], NumOps.Multiply(dotProduct, prevColumn[i]));
                }
            }

            T norm = NumOps.Zero;
            for (int i = 0; i < rows; i++)
            {
                norm = NumOps.Add(norm, NumOps.Multiply(column[i], column[i]));
            }
            norm = NumOps.Sqrt(norm);

            if (NumOps.GreaterThan(norm, NumOps.FromDouble(1e-10)))
            {
                for (int i = 0; i < rows; i++)
                {
                    orthogonal[i, j] = NumOps.Divide(column[i], norm);
                }
            }
            else
            {
                for (int i = 0; i < rows; i++)
                {
                    orthogonal[i, j] = i == j ? NumOps.One : NumOps.Zero;
                }
            }
        }

        return orthogonal;
    }

    /// <summary>
    /// Creates a minimal placeholder LoRA layer to satisfy base class requirements.
    /// </summary>
    /// <remarks>
    /// <para><b>IMPORTANT:</b> MoRA does NOT use the standard LoRA layer architecture.
    /// This method creates a minimal LoRALayer with rank=1 only to satisfy the LoRAAdapterBase
    /// contract, but it is never used in MoRA's Forward, Backward, or UpdateParameters methods.
    /// </para>
    /// <para>
    /// MoRA uses its own square matrix M combined with compression/decompression matrices instead
    /// of the standard A/B low-rank decomposition. The actual MoRA logic is implemented directly
    /// in the overridden methods using _matrixM, _compressionMatrix, and _decompressionMatrix.
    /// </para>
    /// <para>
    /// This design choice maintains compatibility with LoRAAdapterBase while avoiding the overhead
    /// of a full-rank unused LoRA layer. Future refactoring could make the LoRA layer optional
    /// in LoRAAdapterBase or have MoRAAdapter extend LayerBase directly.
    /// </para>
    /// </remarks>
    protected override LoRALayer<T> CreateLoRALayer(int rank, double alpha)
    {
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        // Minimal rank=1 to minimize memory overhead of unused layer
        return new LoRALayer<T>(inputSize, outputSize, 1, alpha);
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input.Clone();
        Tensor<T> baseOutput = _baseLayer.Forward(input);

        int batchSize = input.Shape[0];
        int dimension = input.Shape.Length > 1 ? input.Shape[1] : input.Length;

        Matrix<T> inputMatrix = new Matrix<T>(batchSize, dimension);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < dimension; j++)
            {
                inputMatrix[i, j] = input[i * dimension + j];
            }
        }

        Matrix<T> compressed = inputMatrix.Multiply(_compressionMatrix);
        _lastCompressed = compressed;

        Matrix<T> transformed = compressed.Multiply(_matrixM);
        Matrix<T> decompressed = transformed.Multiply(_decompressionMatrix);

        T scalingFactor = NumOps.FromDouble(Alpha);
        decompressed = decompressed.Multiply(scalingFactor);

        Tensor<T> moraOutput = new Tensor<T>(baseOutput.Shape);
        int idx = 0;
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < dimension; j++)
            {
                moraOutput[idx] = decompressed[i, j];
                idx++;
            }
        }

        Tensor<T> result = new Tensor<T>(baseOutput.Shape);
        for (int i = 0; i < baseOutput.Length; i++)
        {
            result[i] = NumOps.Add(baseOutput[i], moraOutput[i]);
        }

        return result;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastCompressed == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass");
        }

        int batchSize = _lastInput.Shape[0];
        int dimension = _lastInput.Shape.Length > 1 ? _lastInput.Shape[1] : _lastInput.Length;

        Matrix<T> gradMatrix = new Matrix<T>(batchSize, dimension);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < dimension; j++)
            {
                gradMatrix[i, j] = outputGradient[i * dimension + j];
            }
        }

        T scalingFactor = NumOps.FromDouble(Alpha);
        Matrix<T> gradTransformed = gradMatrix.Multiply(_decompressionMatrix.Transpose()).Multiply(scalingFactor);
        _matrixMGradient = _lastCompressed.Transpose().Multiply(gradTransformed);
        Matrix<T> gradCompressed = gradTransformed.Multiply(_matrixM.Transpose());
        Matrix<T> moraInputGradient = gradCompressed.Multiply(_compressionMatrix.Transpose());

        Tensor<T> baseInputGrad = _baseLayer.Backward(outputGradient);

        Tensor<T> inputGrad = new Tensor<T>(_lastInput.Shape);
        int idx = 0;
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < dimension; j++)
            {
                T moraGrad = moraInputGradient[i, j];
                inputGrad[idx] = NumOps.Add(baseInputGrad[idx], moraGrad);
                idx++;
            }
        }

        return inputGrad;
    }

    public override void UpdateParameters(T learningRate)
    {
        if (_matrixMGradient == null)
        {
            return;
        }

        for (int i = 0; i < _matrixM.Rows; i++)
        {
            for (int j = 0; j < _matrixM.Columns; j++)
            {
                T update = NumOps.Multiply(_matrixMGradient[i, j], learningRate);
                _matrixM[i, j] = NumOps.Subtract(_matrixM[i, j], update);
            }
        }

        if (!_freezeBaseLayer)
        {
            _baseLayer.UpdateParameters(learningRate);
        }

        // Rebuild Parameters buffer to reflect updated _matrixM and _baseLayer
        RebuildParameterSnapshot();
    }

    /// <summary>
    /// Gets the current parameter values (base layer + MoRA matrix M).
    /// </summary>
    /// <returns>A cloned vector containing all parameters.</returns>
    /// <remarks>
    /// <para>
    /// Since MoRA does not use the standard LoRA layer architecture, this method overrides
    /// the base implementation to pack parameters from the base layer (if not frozen) and
    /// the square matrix M directly.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        return Parameters.Clone();
    }

    /// <summary>
    /// Sets the parameter values (base layer + MoRA matrix M).
    /// </summary>
    /// <param name="parameters">Parameter vector to set.</param>
    /// <exception cref="ArgumentException">Thrown if parameter count doesn't match ParameterCount.</exception>
    /// <remarks>
    /// <para>
    /// This method unpacks the parameter vector into the base layer (if not frozen) and
    /// the square matrix M. The parameter layout is:
    /// - Base layer parameters (if !_freezeBaseLayer): [0 .. baseLayerParamCount)
    /// - Matrix M parameters (row-major): [baseLayerParamCount .. ParameterCount)
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException($"Expected {ParameterCount} parameters, but got {parameters.Length}", nameof(parameters));
        }

        // Clone into Parameters buffer
        Parameters = parameters.Clone();

        int idx = 0;

        // Unpack base layer parameters if not frozen
        if (!_freezeBaseLayer)
        {
            int baseParamCount = _baseLayer.ParameterCount;
            Vector<T> baseParams = new Vector<T>(baseParamCount);
            for (int i = 0; i < baseParamCount; i++)
            {
                baseParams[i] = parameters[idx++];
            }
            _baseLayer.SetParameters(baseParams);
        }

        // Unpack matrix M parameters (row-major order)
        for (int i = 0; i < _matrixM.Rows; i++)
        {
            for (int j = 0; j < _matrixM.Columns; j++)
            {
                _matrixM[i, j] = parameters[idx++];
            }
        }
    }

    public override int ParameterCount
    {
        get
        {
            // During base class construction, _squareRank is not yet initialized (it's 0).
            // In this phase, we need to return a parameter count that satisfies the base class,
            // which includes the base layer's parameters and the placeholder LoRA layer's parameters.
            if (_squareRank == 0)
            {
                int baseLayerParams = (_baseLayer != null && !_freezeBaseLayer) ? _baseLayer.ParameterCount : 0;
                // The _loraLayer is created in CreateLoRALayer, so it should be available.
                // Its parameter count is needed for the base class's internal parameter management.
                // CreateLoRALayer uses rank=1 for the placeholder LoRA layer.
                int loraLayerParams = _loraLayer?.ParameterCount ?? (GetInputShape()[0] * 1 + GetOutputShape()[0] * 1);
                return baseLayerParams + loraLayerParams;
            }
            else
            {
                // After MoRAAdapter's constructor has run and _squareRank is initialized,
                // the actual trainable parameters are from _matrixM and the base layer (if not frozen).
                int moraParams = _squareRank * _squareRank;
                int baseParams = (_baseLayer != null && !_freezeBaseLayer) ? _baseLayer.ParameterCount : 0;
                return baseParams + moraParams;
            }
        }
    }

    public override ILayer<T> MergeToOriginalLayer()
    {
        // Compute full MoRA adaptation: R_d * M * R_c^T
        Matrix<T> temp = _matrixM.Multiply(_compressionMatrix.Transpose());
        Matrix<T> fullAdaptation = _decompressionMatrix.Multiply(temp);

        T scalingFactor = NumOps.FromDouble(Alpha);
        fullAdaptation = fullAdaptation.Multiply(scalingFactor);

        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];

        // Get the original base layer weights
        DenseLayer<T>? denseBase = _baseLayer as DenseLayer<T>;
        FullyConnectedLayer<T>? fcBase = _baseLayer as FullyConnectedLayer<T>;

        if (denseBase == null && fcBase == null)
        {
            throw new InvalidOperationException("MoRAAdapter merging only supports DenseLayer or FullyConnectedLayer");
        }

        // Get base layer parameters (weights + biases)
        Vector<T> baseParams = _baseLayer.GetParameters();
        int weightCount = inputSize * outputSize;

        // Create merged parameters starting with base layer parameters
        Vector<T> mergedParams = baseParams.Clone();

        // Transpose fullAdaptation to get [outputSize, inputSize]
        Matrix<T> adaptationWeights = fullAdaptation.Transpose();

        // Add MoRA adaptation to the weight portion: W_merged = W_base + α * ΔW
        for (int i = 0; i < weightCount; i++)
        {
            int row = i / inputSize;
            int col = i % inputSize;
            mergedParams[i] = NumOps.Add(mergedParams[i], adaptationWeights[row, col]);
        }
        // Note: Biases remain unchanged (indices weightCount to end)

        // Use helper method to clone base layer and preserve activation function
        return CreateMergedLayerWithClone(mergedParams);
    }

    public override void ResetState()
    {
        _baseLayer.ResetState();
        _lastInput = null;
        _lastCompressed = null;
        _matrixMGradient = null;
    }
}

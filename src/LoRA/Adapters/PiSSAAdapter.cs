using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Enums.AlgorithmTypes;
using AiDotNet.Interfaces;

namespace AiDotNet.LoRA.Adapters;

/// <summary>
/// Principal Singular Values and Singular Vectors Adaptation (PiSSA) adapter for parameter-efficient fine-tuning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// PiSSA (NeurIPS 2024 Spotlight) improves upon standard LoRA by initializing adapter matrices with
/// principal components from Singular Value Decomposition (SVD) of pretrained weights, rather than
/// random initialization. This results in more effective use of the rank budget and faster convergence.
/// </para>
/// <para><b>Key Differences from Standard LoRA:</b>
/// - Standard LoRA: A initialized randomly, B initialized to zero
/// - PiSSA: A and B initialized from top-r singular vectors of pretrained weights
/// - Standard LoRA: All weights trainable
/// - PiSSA: Residual weights frozen, only top-r components trainable
/// </para>
/// <para><b>How PiSSA Works:</b>
/// 1. Perform SVD on pretrained weights: W = U Σ V^T
/// 2. Initialize adapter matrices from top-r components:
///    - A = V_r (top-r right singular vectors, dimensions: inputSize × rank)
///    - B = Σ_r * U_r^T (top-r left singular vectors scaled by singular values, dimensions: rank × outputSize)
/// 3. Freeze residual matrix: W_residual = W - (A*B)^T
/// 4. During training: output = W_residual * input + LoRA(input)
/// 5. Only B and A are updated; W_residual stays frozen
/// </para>
/// <para><b>Performance Benefits:</b>
/// PiSSA achieves superior performance compared to standard LoRA:
/// - GSM8K benchmark: 72.86% (PiSSA) vs 67.7% (LoRA)
/// - Better initialization captures important pretrained knowledge
/// - More effective gradient updates from the start
/// - Faster convergence with fewer training steps
/// </para>
/// <para><b>For Beginners:</b> Think of PiSSA as "smart LoRA initialization".
///
/// Standard LoRA starts from random:
/// - Random A matrix (like throwing darts blindfolded)
/// - Zero B matrix (starts with no effect)
/// - Learns everything from scratch
///
/// PiSSA starts from the most important parts of pretrained weights:
/// - A and B capture the top-r "principal directions" of the pretrained model
/// - Starts closer to the optimal solution
/// - Like starting a puzzle with the border pieces already connected
///
/// Example: If you have a pretrained language model with a 4096x4096 weight matrix,
/// PiSSA with rank=8 will:
/// 1. Find the top 8 most important patterns in those weights via SVD
/// 2. Put those patterns into A and B (making them trainable)
/// 3. Freeze the remaining "less important" patterns
/// 4. Train only the top 8 patterns to adapt to your task
///
/// This is much more efficient than starting from random and achieves better results!
/// </para>
/// <para><b>References:</b>
/// - Paper: "PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models"
/// - Venue: NeurIPS 2024 (Spotlight)
/// - Key Insight: SVD-based initialization > random initialization for low-rank adaptation
/// </para>
/// </remarks>
public partial class PiSSAAdapter<T> : LoRAAdapterBase<T>
{
    /// <summary>
    /// The frozen residual weights after removing top-r principal components.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This matrix represents W_residual = W - B*A, where W is the original pretrained weights
    /// and B*A is the top-r rank approximation. During training, this matrix remains frozen
    /// while only the adapter matrices (A and B) are updated.
    /// </para>
    /// <para><b>For Beginners:</b> This is the "leftover" part of the original weights.
    ///
    /// Think of the original weights as a complete picture:
    /// - The top-r components (in A and B) capture the main features
    /// - The residual is what's left after removing those main features
    /// - During training, we keep this residual fixed and only adjust the main features
    ///
    /// This is like keeping the background of a photo fixed while adjusting only the main subject.
    /// </para>
    /// </remarks>
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _residualWeights;

    /// <summary>
    /// Indicates whether the adapter was initialized from SVD of pretrained weights.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When true, this adapter was properly initialized using PiSSA's SVD-based initialization.
    /// When false, it falls back to standard LoRA random initialization (not recommended for PiSSA).
    /// </para>
    /// <para><b>For Beginners:</b> This flag tells you if the adapter is using PiSSA's smart initialization.
    ///
    /// True = properly initialized with SVD (recommended)
    /// False = using random initialization like standard LoRA (loses PiSSA benefits)
    /// </para>
    /// </remarks>
    private bool _initializedFromSVD;

    /// <summary>
    /// Gets the frozen residual weights matrix.
    /// </summary>
    /// <remarks>
    /// This matrix is computed during SVD initialization and remains frozen during training.
    /// Returns null if SVD initialization was not performed.
    /// </remarks>
    public Matrix<T>? ResidualWeights => _residualWeights?.Clone();

    /// <summary>
    /// Gets whether this adapter was initialized from SVD.
    /// </summary>
    /// <remarks>
    /// Returns true if InitializeFromSVD was called successfully, false otherwise.
    /// </remarks>
    public bool InitializedFromSVD => _initializedFromSVD;

    /// <summary>
    /// Initializes a new PiSSA adapter wrapping an existing layer.
    /// </summary>
    /// <param name="baseLayer">The layer to adapt with PiSSA.</param>
    /// <param name="rank">The rank of the low-rank decomposition.</param>
    /// <param name="alpha">The LoRA scaling factor (defaults to rank if negative).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer's parameters during training.</param>
    /// <exception cref="ArgumentNullException">Thrown when baseLayer is null.</exception>
    /// <remarks>
    /// <para>
    /// This constructor creates a PiSSA adapter. After construction, you should call
    /// InitializeFromSVD to properly initialize the adapter matrices from pretrained weights.
    /// Without SVD initialization, the adapter behaves like standard LoRA (not recommended).
    /// </para>
    /// <para><b>For Beginners:</b> This creates a PiSSA adapter for any layer type.
    ///
    /// Parameters:
    /// - baseLayer: The layer you want to adapt (Dense, Convolutional, etc.)
    /// - rank: How many principal components to use (typically 4-32)
    /// - alpha: Scaling factor for the adaptation strength
    /// - freezeBaseLayer: Usually true to freeze original weights
    ///
    /// Important: After creating the adapter, call InitializeFromSVD with the pretrained
    /// weights to get PiSSA's performance benefits. Otherwise, it's just regular LoRA.
    /// </para>
    /// </remarks>
    public PiSSAAdapter(ILayer<T> baseLayer, int rank, double alpha = -1, bool freezeBaseLayer = true)
        : base(baseLayer, rank, alpha, freezeBaseLayer)
    {
        _initializedFromSVD = false;
    }

    /// <summary>
    /// Initializes the adapter matrices from SVD of pretrained weights.
    /// </summary>
    /// <param name="pretrainedWeights">The pretrained weight matrix to decompose.</param>
    /// <param name="svdAlgorithm">The SVD algorithm to use (default: GolubReinsch).</param>
    /// <exception cref="ArgumentNullException">Thrown when pretrainedWeights is null.</exception>
    /// <exception cref="ArgumentException">Thrown when weight matrix dimensions don't match layer dimensions.</exception>
    /// <remarks>
    /// <para>
    /// This method performs the core PiSSA initialization:
    /// 1. Computes SVD: W = U Σ V^T
    /// 2. Extracts top-r components: U_r, Σ_r, V_r
    /// 3. Initializes A = V_r sqrt(Σ_r / s) (right singular vectors, s = alpha / rank)
    /// 4. Initializes B = sqrt(Σ_r / s) U_r^T (left singular vectors)
    /// 5. Computes residual: W_residual = W - U_r Σ_r V_r^T, so the adapter starts equal to W
    /// </para>
    /// <para><b>For Beginners:</b> This is where the magic happens!
    ///
    /// The method:
    /// 1. Takes your pretrained weights (like from a large language model)
    /// 2. Finds the most important patterns using SVD (mathematical technique)
    /// 3. Puts those patterns into the adapter matrices A and B
    /// 4. Saves the "leftover" patterns as frozen residual weights
    ///
    /// Think of it like:
    /// - Original weights = complete painting
    /// - SVD = identifying the main strokes vs. minor details
    /// - A and B = the main strokes (what we'll adjust)
    /// - Residual = the minor details (kept frozen)
    ///
    /// This initialization is what makes PiSSA better than LoRA - it starts from
    /// a smart place instead of random values.
    /// </para>
    /// </remarks>
    public void InitializeFromSVD(Matrix<T> pretrainedWeights, SvdAlgorithmType svdAlgorithm = SvdAlgorithmType.GolubReinsch)
    {
        if (pretrainedWeights == null)
        {
            throw new ArgumentNullException(nameof(pretrainedWeights));
        }

        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputLayerShape().RequireConcrete("Sizing a LoRA adapter's low-rank factors")[0];

        if (pretrainedWeights.Rows != outputSize || pretrainedWeights.Columns != inputSize)
        {
            throw new ArgumentException(
                $"Weight matrix dimensions ({pretrainedWeights.Rows}x{pretrainedWeights.Columns}) " +
                $"do not match layer dimensions ({outputSize}x{inputSize})",
                nameof(pretrainedWeights));
        }

        // Perform SVD: W = U Σ V^T
        SvdDecomposition<T> svd = new SvdDecomposition<T>(pretrainedWeights, svdAlgorithm);

        // Extract top-r singular values and vectors
        int r = Rank;

        // PiSSA (Meng et al., 2024) splits the principal singular values evenly between the factors and,
        // as in PEFT, divides them by the LoRA scaling first so that the scaled product the LoRA layer
        // applies, scaling * A * B, is exactly the rank-r principal part: with d = sqrt(S_r / scaling),
        //   A = V_r * diag(d)   (inputSize × rank)
        //   B = diag(d) * U_r^T (rank × outputSize)
        T scaling = _loraLayer.Scaling;
        Tensor<T> splitSingularValues = Engine.TensorSqrt(Engine.TensorDivideScalar(
            Tensor<T>.FromVector(svd.S.GetSubVector(0, r)), scaling));
        Matrix<T> matrixA = Engine.TensorMultiply(
            Tensor<T>.FromMatrix(Engine.MatrixTranspose(svd.Vt.GetSubMatrix(0, 0, r, inputSize))),
            Engine.Reshape(splitSingularValues, [1, r])).ToMatrix();
        Matrix<T> matrixB = Engine.TensorMultiply(
            Tensor<T>.FromMatrix(Engine.MatrixTranspose(svd.U.GetSubMatrix(0, 0, outputSize, r))),
            Engine.Reshape(splitSingularValues, [r, 1])).ToMatrix();

        // Compute the low-rank approximation: W_rank_r = B*A
        // Note: matrixA is [inputSize x r], matrixB is [r x outputSize]
        // So B*A would be [r x r], which is wrong. We need A*B^T for proper dimensions.
        // Actually, for PiSSA: output = W_residual * input + B * A * input
        // Where A: [inputSize x r], B: [r x outputSize]
        // So B*A: [r x outputSize] * [inputSize x r] - dimension mismatch!
        // Correct formulation: A is applied first (compresses input), then B (expands to output)
        // Let's recalculate: we need W ≈ B^T * A^T in weight space

        // For LoRA layer: input -> A -> (rank dims) -> B -> output
        // For weight reconstruction: W = B^T * A^T (both transposed)
        // Since LoRALayer stores A as [inputSize x rank] and B as [rank x outputSize]
        // The weight contribution is: W_lora = A * B (gives [inputSize x outputSize])
        // Then transposed to match DenseLayer format [outputSize x inputSize]

        // So we need: W = (A * B)^T + W_residual
        // Therefore: W_residual = W - (A * B)^T

        // The LoRA layer applies scaling * A * B, so that is the principal part to remove from W
        Matrix<T> lowRankApprox = Engine.MatrixMultiplyScalar(matrixA.Multiply(matrixB), scaling); // [inputSize x outputSize]
        Matrix<T> lowRankApproxTransposed = lowRankApprox.Transpose(); // [outputSize x inputSize]

        // Compute residual: W_residual = W - (B*A approximation)
        _residualWeights = Engine.MatrixSubtract(pretrainedWeights, lowRankApproxTransposed);

        // Set the LoRA layer's A and B matrices
        // Note: LoRALayer expects A: [inputSize x rank], B: [rank x outputSize]
        Vector<T> loraParams = Vector<T>.Concatenate(matrixA.ToRowVector(), matrixB.ToRowVector());

        _loraLayer.SetParameters(loraParams);
        _initializedFromSVD = true;
    }

    /// <summary>
    /// Creates a PiSSA adapter initialized from SVD of pretrained weights.
    /// </summary>
    /// <param name="baseLayer">The layer to adapt with PiSSA.</param>
    /// <param name="pretrainedWeights">The pretrained weight matrix to decompose.</param>
    /// <param name="rank">The rank of the low-rank decomposition.</param>
    /// <param name="alpha">The LoRA scaling factor (defaults to rank if negative).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer's parameters during training.</param>
    /// <param name="svdAlgorithm">The SVD algorithm to use (default: GolubReinsch).</param>
    /// <returns>A PiSSA adapter initialized from SVD.</returns>
    /// <remarks>
    /// <para>
    /// This static factory method creates and fully initializes a PiSSA adapter in one step.
    /// It combines construction and SVD initialization for convenience.
    /// </para>
    /// <para><b>For Beginners:</b> This is the recommended way to create a PiSSA adapter.
    ///
    /// Instead of:
    /// 1. Create adapter
    /// 2. Call InitializeFromSVD
    ///
    /// You can just:
    /// 1. Call this method with pretrained weights
    ///
    /// Example:
    /// var adapter = PiSSAAdapter.InitializeFromSVD(myLayer, pretrainedWeights, rank: 8);
    /// // Ready to train!
    /// </para>
    /// </remarks>
    public static PiSSAAdapter<T> InitializeFromSVD(
        ILayer<T> baseLayer,
        Matrix<T> pretrainedWeights,
        int rank,
        double alpha = -1,
        bool freezeBaseLayer = true,
        SvdAlgorithmType svdAlgorithm = SvdAlgorithmType.GolubReinsch)
    {
        PiSSAAdapter<T> adapter = new PiSSAAdapter<T>(baseLayer, rank, alpha, freezeBaseLayer);
        adapter.InitializeFromSVD(pretrainedWeights, svdAlgorithm);
        return adapter;
    }

    /// <summary>
    /// Performs the forward pass using residual weights plus trainable PiSSA adaptation.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output tensor computed as: residual_output + lora_output.</returns>
    /// <remarks>
    /// <para>
    /// If initialized from SVD, the forward pass computes:
    /// output = W_residual * input + LoRA(input)
    ///
    /// If not initialized from SVD (falls back to standard LoRA):
    /// output = base_layer(input) + LoRA(input)
    /// </para>
    /// <para><b>For Beginners:</b> This runs input through the adapter.
    ///
    /// With proper PiSSA initialization:
    /// - First applies frozen residual weights (the "less important" parts)
    /// - Then adds the trainable adaptation (the "important" parts from A and B)
    /// - Result combines both for the final output
    ///
    /// Without SVD initialization (not recommended):
    /// - Falls back to standard LoRA behavior
    /// - Uses base layer output + LoRA correction
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (!_initializedFromSVD || _residualWeights == null)
        {
            // Fall back to standard LoRA behavior if not initialized from SVD.
            // ForwardTraced, NOT Forward: LayerBase.Forward is the non-virtual recording wrapper that
            // dispatches to ForwardTraced, so base.Forward(input) would come straight back here and
            // recurse until the stack overflows. LoRAAdapterBase overrides ForwardTraced, so this is
            // the base implementation that call was always meant to reach.
            return base.ForwardTraced(input);
        }

        // Get batch size and validate input shape
        int batchSize = input.Shape[0];
        int inputSize = input.Shape.Length > 1 ? input.Shape[1] : input.Length;

        if (inputSize != _residualWeights.Columns)
        {
            throw new ArgumentException(
                $"Input size {inputSize} does not match residual weights columns {_residualWeights.Columns}",
                nameof(input));
        }

        // View input as [batchSize, inputSize]
        Tensor<T> inputMatrix = Engine.Reshape(input, [batchSize, inputSize]);

        // Compute residual output: input * W_residual^T -> [batchSize, outputSize]
        Tensor<T> residualOutput = Engine.TensorMatMul(
            inputMatrix, Tensor<T>.FromMatrix(Engine.MatrixTranspose(_residualWeights)));

        // Compute LoRA output
        Tensor<T> loraOutput = _loraLayer.Forward(input);

        // Sum the outputs, plus the base layer's bias (the merged layer keeps it, so the adapter must too)
        int outputSize = _residualWeights.Rows;
        Tensor<T> result = Engine.TensorAdd(residualOutput, Engine.Reshape(loraOutput, [batchSize, outputSize]));
        Tensor<T>? biasRows = BaseBiasRows(_baseLayer.GetParameters(), outputSize * inputSize, outputSize, batchSize);
        if (biasRows is not null)
        {
            result = Engine.TensorAdd(result, biasRows);
        }

        return result;
    }

    /// <summary>
    /// Merges the PiSSA adaptation into the original layer.
    /// </summary>
    /// <returns>A new layer with PiSSA weights merged back into a single weight matrix.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the adapter was not initialized from SVD.</exception>
    /// <remarks>
    /// <para>
    /// This method reconstructs the full weight matrix by combining:
    /// W_merged = W_residual + (A * B)^T
    ///
    /// This allows you to deploy the adapted model without the PiSSA overhead.
    /// </para>
    /// <para><b>For Beginners:</b> This "bakes in" the PiSSA adaptation.
    ///
    /// After training:
    /// - You have: frozen residual weights + trained A and B matrices
    /// - Merging combines them: residual + A*B = final weights
    /// - Result: a single regular layer with all improvements included
    ///
    /// Benefits:
    /// - Faster inference (no need to compute residual + LoRA separately)
    /// - Simpler deployment (just one layer)
    /// - Compatible with systems that don't support LoRA/PiSSA
    ///
    /// Example:
    /// var mergedLayer = adapter.MergeToOriginalLayer();
    /// // Now you have a standard layer with PiSSA improvements built in!
    /// </para>
    /// </remarks>
    public override ILayer<T> MergeToOriginalLayer()
    {
        if (!_initializedFromSVD || _residualWeights == null)
        {
            throw new InvalidOperationException(
                "Cannot merge PiSSA adapter that was not initialized from SVD. " +
                "Call InitializeFromSVD before merging.");
        }

        // Get the LoRA weight contribution: (A * B)^T
        Matrix<T> loraWeights = _loraLayer.MergeWeights(); // Already transposed

        // Merge: W_final = W_residual + LoRA_weights
        int outputSize = _residualWeights.Rows;
        int inputSize = _residualWeights.Columns;

        // MergeWeights() is [inputSize, outputSize] (A * B), so its transpose lines up with the
        // [outputSize, inputSize] residual, matching the forward pass (input * W_residual^T + input * A * B).
        Matrix<T> mergedWeights = Engine.MatrixAdd(_residualWeights, Engine.MatrixTranspose(loraWeights));

        // The residual is output-major; DenseLayer's flat weights are input-major.
        if (_baseLayer is DenseLayer<T>)
            mergedWeights = Engine.MatrixTranspose(mergedWeights);

        // Create parameters vector: [merged weights (row-major), biases unchanged]
        Vector<T> baseParams = _baseLayer.GetParameters();
        int weightCount = outputSize * inputSize;
        int biasCount = baseParams.Length - weightCount;
        Vector<T> mergedParams = biasCount > 0
            ? Vector<T>.Concatenate(mergedWeights.ToRowVector(), baseParams.GetSubVector(weightCount, biasCount))
            : mergedWeights.ToRowVector();

        // Use helper method to clone base layer and preserve activation function
        return CreateMergedLayerWithClone(mergedParams);
    }
}

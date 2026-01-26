using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Enums.AlgorithmTypes;
using AiDotNet.Interfaces;

namespace AiDotNet.LoRA.Adapters;

/// <summary>
/// LoRA-XS (Extremely Small) adapter for ultra-parameter-efficient fine-tuning using SVD with trainable scaling matrix.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// LoRA-XS achieves extreme parameter efficiency by leveraging SVD of pretrained weights to create frozen
/// orthonormal bases (U and V matrices), with only a small r×r trainable matrix R positioned between them.
/// This architecture reduces parameter count to r² instead of 2nr (standard LoRA), achieving 100x+ reduction
/// while matching or exceeding full fine-tuning performance.
/// </para>
/// <para><b>Architecture Comparison:</b>
/// - Standard LoRA: W' = W + BA, where A ∈ ℝ^(d×r), B ∈ ℝ^(r×d) (2dr parameters)
/// - LoRA-XS: W' = W + U_r Σ_r R V_r^T, where only R ∈ ℝ^(r×r) is trainable (r² parameters)
/// - U_r and V_r are frozen orthonormal bases from SVD of pretrained W
/// - Σ_r is the frozen diagonal matrix of top-r singular values
/// </para>
/// <para><b>Key Innovation:</b>
/// Instead of training both A and B matrices (standard LoRA), LoRA-XS:
/// 1. Computes SVD of pretrained weights: W = U Σ V^T
/// 2. Freezes U_r (top-r left singular vectors) and V_r^T (top-r right singular vectors)
/// 3. Freezes Σ_r (top-r singular values as diagonal matrix)
/// 4. Trains only R (r×r mixing matrix) that interpolates between frozen bases
/// 5. Parameter count is independent of hidden dimensions: only r² trainable parameters
/// </para>
/// <para><b>Performance Metrics (from paper):</b>
///
/// RoBERTa-large on GLUE (6 tasks):
/// - LoRA-XS (rank 16): 88.03% avg accuracy, 24.6K parameters
/// - Standard LoRA (rank 16): Similar accuracy, 100x more parameters
/// - Full fine-tuning: 88.0% avg accuracy, ~125M parameters per task
///
/// LLaMA2-7B on Commonsense Reasoning:
/// - LoRA-XS: 80.5% avg accuracy, 3.67M parameters
/// - Standard LoRA: 77.6% avg accuracy, 56M parameters (15x more)
///
/// Mistral-7B on GSM8K (Math Reasoning):
/// - LoRA-XS: 70.35% accuracy, 3.67M parameters
/// - Standard LoRA: 67.70% accuracy, 168M parameters (46x more)
///
/// GPT-3 Personalization (1M models):
/// - LoRA-XS: 96GB total storage
/// - Standard LoRA: 144TB total storage (1500x reduction)
/// </para>
/// <para><b>Mathematical Formulation:</b>
/// Forward pass computes:
///   output = (W + U_r Σ_r R V_r^T) * input
///          = W * input + (U_r Σ_r) * (R * (V_r^T * input))
///
/// Where:
/// - W is frozen pretrained weights
/// - U_r ∈ ℝ^(d_out × r): frozen left singular vectors (orthonormal columns)
/// - Σ_r ∈ ℝ^(r × r): frozen diagonal matrix of singular values
/// - R ∈ ℝ^(r × r): trainable mixing matrix (only trainable component!)
/// - V_r^T ∈ ℝ^(r × d_in): frozen right singular vectors (orthonormal rows)
/// </para>
/// <para><b>Why This Works:</b>
/// The SVD provides an optimal orthonormal basis for representing weight updates. By freezing
/// these bases and training only the mixing matrix R, LoRA-XS achieves:
/// - Drastically fewer parameters (r² vs 2dr)
/// - Better generalization (constrained to pretrained subspace)
/// - Faster convergence (optimal basis from initialization)
/// - No inference overhead (can be merged back into W)
/// - Scalable personalization (parameter count independent of model size)
/// </para>
/// <para><b>For Beginners:</b> Think of LoRA-XS as "ultra-compressed LoRA".
///
/// Imagine you have a large language model with huge weight matrices (e.g., 4096×4096):
///
/// Standard LoRA (rank 8):
/// - Creates two matrices: A (4096×8) and B (8×4096)
/// - Total parameters: 4096*8 + 8*4096 = 65,536 parameters
/// - Both matrices are trainable
///
/// LoRA-XS (rank 8):
/// - Decomposes pretrained weights with SVD into U, Σ, V
/// - Keeps top 8 singular vectors (U_8, Σ_8, V_8) FROZEN
/// - Trains only R matrix: 8×8 = 64 parameters
/// - Achieves similar or better performance with 1000x fewer parameters!
///
/// It's like having two fixed "coordinate systems" from the pretrained model,
/// and you only train a small "rotation matrix" between them. The fixed coordinate
/// systems capture the pretrained knowledge, while the rotation matrix adapts to your task.
///
/// Example workflow:
/// 1. Load pretrained model weights W
/// 2. Compute SVD: W = U Σ V^T
/// 3. Extract top-r components: U_r, Σ_r, V_r
/// 4. Create LoRA-XS adapter with these frozen bases
/// 5. Train only the tiny R matrix (64 params for rank 8)
/// 6. Deploy with merged weights: W' = W + U_r Σ_r R V_r^T
/// </para>
/// <para><b>References:</b>
/// - Paper: "LoRA-XS: Low-Rank Adaptation with Extremely Small Number of Parameters"
/// - arXiv: 2405.17604 (May 2024)
/// - GitHub: MohammadrezaBanaei/LoRA-XS
/// - Key Innovation: Parameter count O(r²) instead of O(dr), enabling extreme efficiency
/// </para>
/// </remarks>
public class LoRAXSAdapter<T> : LoRAAdapterBase<T>
{
    /// <summary>
    /// Frozen left singular vectors (U_r) from SVD of pretrained weights.
    /// Shape: [outputSize, rank]
    /// </summary>
    /// <remarks>
    /// <para>
    /// These are the top-r left singular vectors from the SVD decomposition of pretrained weights.
    /// They form an orthonormal basis for the output space and remain frozen during training.
    /// </para>
    /// <para><b>For Beginners:</b> This matrix contains the most important "output patterns" from
    /// the pretrained model. It's like having a fixed set of "building blocks" that the model
    /// learned during pretraining. We keep these fixed and only learn how to combine them.
    /// </para>
    /// </remarks>
    private Matrix<T>? _frozenU;

    /// <summary>
    /// Frozen singular values (diagonal of Σ_r) from SVD of pretrained weights.
    /// Length: rank
    /// </summary>
    /// <remarks>
    /// <para>
    /// These are the top-r singular values from the SVD decomposition. They represent the
    /// importance/strength of each corresponding singular vector pair. Stored as a vector
    /// representing the diagonal of Σ_r matrix.
    /// </para>
    /// <para><b>For Beginners:</b> These numbers tell you how important each "pattern" is.
    /// Larger values mean more important patterns. We keep the top-r most important ones
    /// and use them to scale the contributions during forward pass.
    /// </para>
    /// </remarks>
    private Vector<T>? _frozenSigma;

    /// <summary>
    /// Frozen right singular vectors transposed (V_r^T) from SVD of pretrained weights.
    /// Shape: [rank, inputSize]
    /// </summary>
    /// <remarks>
    /// <para>
    /// These are the top-r right singular vectors (transposed) from the SVD decomposition.
    /// They form an orthonormal basis for the input space and remain frozen during training.
    /// </para>
    /// <para><b>For Beginners:</b> This matrix contains the most important "input patterns" from
    /// the pretrained model. Like U, these are fixed building blocks. Together, U and V define
    /// the coordinate system in which we'll make small adjustments via the R matrix.
    /// </para>
    /// </remarks>
    private Matrix<T>? _frozenVt;

    /// <summary>
    /// Trainable r×r mixing matrix R - the ONLY trainable parameters in LoRA-XS.
    /// Shape: [rank, rank]
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the core trainable component of LoRA-XS. It's a small r×r matrix that learns
    /// how to mix/interpolate between the frozen singular vector bases. The forward pass computes:
    /// adaptation = U_r * Σ_r * R * V_r^T, where only R is updated during training.
    /// </para>
    /// <para><b>For Beginners:</b> This tiny matrix (e.g., 8×8 = 64 parameters for rank 8) is
    /// what actually gets trained! It learns how to "rotate" or "mix" between the frozen patterns
    /// in U and V to adapt to your specific task. This is where all the magic happens with
    /// minimal parameters.
    /// </para>
    /// </remarks>
    private Matrix<T> _trainableR;

    /// <summary>
    /// Gradient of the trainable R matrix computed during backpropagation.
    /// </summary>
    private Matrix<T>? _trainableRGradient;

    /// <summary>
    /// Intermediate result from forward pass: V_r^T * input
    /// Cached for use in backward pass.
    /// </summary>
    private Tensor<T>? _cachedVtInput;

    /// <summary>
    /// Intermediate result from forward pass: R * (V_r^T * input)
    /// Cached for use in backward pass.
    /// </summary>
    private Tensor<T>? _cachedRVtInput;

    /// <summary>
    /// Intermediate result from forward pass: Σ_r * R * (V_r^T * input)
    /// Cached for use in backward pass.
    /// </summary>
    private Tensor<T>? _cachedSigmaRVtInput;

    /// <summary>
    /// Indicates whether the adapter was initialized from SVD of pretrained weights.
    /// </summary>
    private bool _initializedFromSVD;

    /// <summary>
    /// Gets whether this adapter was initialized from SVD.
    /// </summary>
    /// <remarks>
    /// Returns true if InitializeFromSVD was called successfully. Without SVD initialization,
    /// LoRA-XS loses its key advantages and effectively becomes a very limited random adapter.
    /// </remarks>
    public bool InitializedFromSVD => _initializedFromSVD;

    /// <summary>
    /// Gets the frozen U matrix (left singular vectors).
    /// </summary>
    public Matrix<T>? FrozenU => _frozenU?.Clone();

    /// <summary>
    /// Gets the frozen singular values.
    /// </summary>
    public Vector<T>? FrozenSigma => _frozenSigma?.Clone();

    /// <summary>
    /// Gets the frozen V^T matrix (right singular vectors transposed).
    /// </summary>
    public Matrix<T>? FrozenVt => _frozenVt?.Clone();

    /// <summary>
    /// Gets the trainable R matrix.
    /// </summary>
    public Matrix<T> TrainableR => _trainableR.Clone();

    /// <summary>
    /// Gets the total number of trainable parameters (only r² for the R matrix).
    /// </summary>
    /// <remarks>
    /// <para>
    /// CRITICAL: Returns full base LoRA layer parameter count to match base constructor expectations.
    /// Even though only the R matrix (rank²) is trainable in LoRA-XS, the base constructor
    /// allocates Parameters buffer based on this count and packs the underlying LoRA layer.
    /// </para>
    /// <para>
    /// LoRA-XS only trains the rank×rank R matrix, so ParameterCount returns rank².
    /// The frozen U, Σ, and V matrices are not trainable parameters.
    /// </para>
    /// </remarks>
    public override int ParameterCount
    {
        get
        {
            // Handle case where R matrix hasn't been initialized yet
            // (called from base constructor before derived constructor runs)
            if (_trainableR == null)
            {
                return Rank * Rank;
            }

            return _trainableR.Rows * _trainableR.Rows;
        }
    }

    /// <summary>
    /// Initializes a new LoRA-XS adapter wrapping an existing layer.
    /// </summary>
    /// <param name="baseLayer">The layer to adapt with LoRA-XS.</param>
    /// <param name="rank">The rank of the SVD decomposition (number of singular values to use).</param>
    /// <param name="alpha">The LoRA scaling factor (defaults to rank if negative).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer's parameters during training (always true for LoRA-XS).</param>
    /// <exception cref="ArgumentNullException">Thrown when baseLayer is null.</exception>
    /// <remarks>
    /// <para>
    /// This constructor creates a LoRA-XS adapter. After construction, you MUST call
    /// InitializeFromSVD to properly initialize the frozen bases and trainable R matrix.
    /// Without SVD initialization, the adapter cannot function as intended.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a LoRA-XS adapter for your layer.
    ///
    /// Important steps:
    /// 1. Create the adapter with this constructor
    /// 2. Call InitializeFromSVD with your pretrained weights
    /// 3. Start training (only the tiny R matrix gets updated!)
    ///
    /// The rank parameter determines the size:
    /// - rank = 4: Only 16 trainable parameters (4×4)
    /// - rank = 8: Only 64 trainable parameters (8×8)
    /// - rank = 16: Only 256 trainable parameters (16×16)
    ///
    /// Compare this to standard LoRA which would have thousands or millions of parameters!
    /// </para>
    /// </remarks>
    public LoRAXSAdapter(ILayer<T> baseLayer, int rank, double alpha = -1, bool freezeBaseLayer = true)
        : base(baseLayer, rank, alpha, freezeBaseLayer: true) // Always freeze base layer for LoRA-XS
    {
        // Initialize trainable R matrix to identity (neutral starting point)
        _trainableR = new Matrix<T>(rank, rank);
        for (int i = 0; i < rank; i++)
        {
            for (int j = 0; j < rank; j++)
            {
                _trainableR[i, j] = (i == j) ? NumOps.One : NumOps.Zero;
            }
        }

        _initializedFromSVD = false;

        // Update parameters to reflect only R matrix
        Parameters = new Vector<T>(ParameterCount);
        UpdateParametersFromR();
    }

    /// <summary>
    /// Initializes the adapter from SVD of pretrained weights.
    /// </summary>
    /// <param name="pretrainedWeights">The pretrained weight matrix to decompose. Shape: [outputSize, inputSize]</param>
    /// <param name="svdAlgorithm">The SVD algorithm to use (default: GolubReinsch).</param>
    /// <exception cref="ArgumentNullException">Thrown when pretrainedWeights is null.</exception>
    /// <exception cref="ArgumentException">Thrown when weight matrix dimensions don't match layer dimensions.</exception>
    /// <remarks>
    /// <para>
    /// This method performs the core LoRA-XS initialization:
    /// 1. Computes full SVD: W = U Σ V^T
    /// 2. Extracts top-r components: U_r (outputSize × r), Σ_r (r diagonal values), V_r^T (r × inputSize)
    /// 3. Freezes U_r, Σ_r, and V_r^T as orthonormal bases
    /// 4. Initializes trainable R matrix to identity (neutral transformation)
    /// 5. During training: only R is updated, U/Σ/V remain frozen
    /// </para>
    /// <para><b>For Beginners:</b> This is where LoRA-XS gets initialized properly!
    ///
    /// What happens:
    /// 1. Takes your pretrained weights (e.g., from a language model layer)
    /// 2. Uses SVD to find the top-r most important patterns (like finding main themes in data)
    /// 3. Saves these patterns as frozen "coordinate systems" (U and V)
    /// 4. Saves their importance scores (Σ, the singular values)
    /// 5. Creates a small R matrix that will learn to adapt between these coordinates
    ///
    /// After this, when you train:
    /// - The frozen patterns (U, Σ, V) don't change
    /// - Only the tiny R matrix learns
    /// - This is why you only train r² parameters instead of millions!
    ///
    /// Example: For a 4096×4096 weight matrix with rank=8:
    /// - Freezes 4096×8 U matrix (32,768 values, but frozen)
    /// - Freezes 8 singular values
    /// - Freezes 8×4096 V^T matrix (32,768 values, but frozen)
    /// - Trains only 8×8 R matrix (64 parameters!)
    /// </para>
    /// </remarks>
    public void InitializeFromSVD(Matrix<T> pretrainedWeights, SvdAlgorithmType svdAlgorithm = SvdAlgorithmType.GolubReinsch)
    {
        if (pretrainedWeights == null)
        {
            throw new ArgumentNullException(nameof(pretrainedWeights));
        }

        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];

        if (pretrainedWeights.Rows != outputSize || pretrainedWeights.Columns != inputSize)
        {
            throw new ArgumentException(
                $"Pretrained weight matrix dimensions ({pretrainedWeights.Rows}×{pretrainedWeights.Columns}) " +
                $"do not match layer dimensions ({outputSize}×{inputSize})",
                nameof(pretrainedWeights));
        }

        // Perform SVD: W = U Σ V^T
        var svd = new SvdDecomposition<T>(pretrainedWeights, svdAlgorithm);

        // Extract top-r singular vectors and values
        int rank = Rank;

        // Extract U_r: top-r left singular vectors (columns of U)
        _frozenU = new Matrix<T>(outputSize, rank);
        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < rank; j++)
            {
                _frozenU[i, j] = svd.U[i, j];
            }
        }

        // Extract Σ_r: top-r singular values (diagonal elements)
        _frozenSigma = new Vector<T>(rank);
        for (int i = 0; i < rank; i++)
        {
            _frozenSigma[i] = svd.S[i];
        }

        // Extract V_r^T: top-r right singular vectors (rows of V^T)
        _frozenVt = new Matrix<T>(rank, inputSize);
        for (int i = 0; i < rank; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                _frozenVt[i, j] = svd.Vt[i, j];
            }
        }

        _initializedFromSVD = true;
    }

    /// <summary>
    /// Performs the forward pass through the LoRA-XS adapter.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Sum of base layer output and LoRA-XS adaptation.</returns>
    /// <remarks>
    /// <para>
    /// The forward pass computes:
    ///   output = base_layer(input) + U_r * Σ_r * R * V_r^T * input * scaling
    ///
    /// Steps:
    /// 1. x1 = V_r^T * input (project input onto frozen right singular vectors)
    /// 2. x2 = R * x1 (apply trainable mixing matrix)
    /// 3. x3 = Σ_r * x2 (scale by frozen singular values)
    /// 4. x4 = U_r * x3 (project onto frozen left singular vectors)
    /// 5. output = base_output + scaling * x4
    /// </para>
    /// <para><b>For Beginners:</b> This is how data flows through LoRA-XS:
    ///
    /// 1. Run input through the original layer (base layer)
    /// 2. Also run through LoRA-XS path:
    ///    - Project input using V (fixed patterns from pretraining)
    ///    - Mix with R matrix (the ONLY thing that's learning!)
    ///    - Scale by Σ (importance weights, fixed)
    ///    - Project back using U (fixed output patterns)
    /// 3. Add the two results together
    ///
    /// Think of it like: original output + small learned adjustment
    /// The adjustment is constrained to the most important pretrained patterns!
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        if (!_initializedFromSVD)
        {
            throw new InvalidOperationException(
                "LoRA-XS adapter must be initialized with InitializeFromSVD before use. " +
                "Call InitializeFromSVD(pretrainedWeights) to set up frozen bases.");
        }

        // Forward through base layer
        Tensor<T> baseOutput = _baseLayer.Forward(input);

        // LoRA-XS forward pass: U_r * Σ_r * R * V_r^T * input
        // Step 1: x1 = V_r^T * input [rank × batchSize]
        _cachedVtInput = MatrixVectorMultiply(_frozenVt!, input);

        // Step 2: x2 = R * x1 [rank × batchSize]
        _cachedRVtInput = MatrixVectorMultiply(_trainableR, _cachedVtInput);

        // Step 3: x3 = Σ_r * x2 (diagonal multiplication) [rank × batchSize]
        _cachedSigmaRVtInput = ApplySigmaScaling(_frozenSigma!, _cachedRVtInput);

        // Step 4: x4 = U_r * x3 [outputSize × batchSize]
        Tensor<T> loraOutput = MatrixVectorMultiply(_frozenU!, _cachedSigmaRVtInput);

        // Apply LoRA scaling factor
        T scaling = NumOps.Divide(NumOps.FromDouble(Alpha), NumOps.FromDouble(Rank));
        loraOutput = ScaleTensor(loraOutput, scaling);

        // Sum the outputs: output = base + lora_adaptation
        Tensor<T> result = new Tensor<T>(baseOutput.Shape);
        for (int i = 0; i < baseOutput.Length; i++)
        {
            result[i] = NumOps.Add(baseOutput[i], loraOutput[i]);
        }

        return result;
    }

    /// <summary>
    /// Performs the backward pass through the LoRA-XS adapter.
    /// </summary>
    /// <param name="outputGradient">Gradient flowing back from the next layer.</param>
    /// <returns>Gradient to pass to the previous layer.</returns>
    /// <remarks>
    /// <para>
    /// The backward pass computes gradients for the trainable R matrix and propagates gradients back.
    ///
    /// Gradient computation:
    ///   dL/dR = (Σ_r * U_r^T * outputGrad) * (V_r^T * input)^T * scaling
    ///   dL/dinput = base_grad + V_r * R^T * Σ_r * U_r^T * outputGrad * scaling
    ///
    /// Note: U, Σ, and V are frozen, so no gradients computed for them.
    /// </para>
    /// <para><b>For Beginners:</b> This is backpropagation for LoRA-XS!
    ///
    /// What happens:
    /// 1. Gradients flow back from the next layer
    /// 2. We compute how to adjust R matrix to reduce error
    ///    (U, Σ, V are frozen so we don't compute gradients for them)
    /// 3. We pass gradients back to the previous layer
    ///
    /// The key: only R learns! This is why training is so efficient.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (!_initializedFromSVD)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass");
        }

        T scaling = NumOps.Divide(NumOps.FromDouble(Alpha), NumOps.FromDouble(Rank));

        // Scale the gradient by the LoRA scaling factor
        Tensor<T> scaledGrad = ScaleTensor(outputGradient, scaling);

        // Backward through base layer (always needed for input gradients)
        Tensor<T> baseInputGrad = _baseLayer.Backward(outputGradient);

        // Backward through LoRA-XS path
        // Current flow: U_r * Σ_r * R * V_r^T
        // Gradient flow (backward): V_r * R^T * Σ_r * U_r^T

        // Step 1: grad_x3 = U_r^T * scaledGrad [rank × batchSize]
        Tensor<T> gradX3 = MatrixVectorMultiply(_frozenU!.Transpose(), scaledGrad);

        // Step 2: grad_x2 = Σ_r * grad_x3 (diagonal multiplication) [rank × batchSize]
        Tensor<T> gradX2 = ApplySigmaScaling(_frozenSigma!, gradX3);

        // Step 3: Compute gradient for R: dL/dR = grad_x2 * _cachedVtInput^T [rank × rank]
        _trainableRGradient = ComputeMatrixGradient(gradX2, _cachedVtInput!);

        // Step 4: grad_x1 = R^T * grad_x2 [rank × batchSize]
        Tensor<T> gradX1 = MatrixVectorMultiply(_trainableR.Transpose(), gradX2);

        // Step 5: input_grad_lora = V_r^T^T * grad_x1 = V_r * grad_x1 [inputSize × batchSize]
        Tensor<T> loraInputGrad = MatrixVectorMultiply(_frozenVt!.Transpose(), gradX1);

        // Sum input gradients from base and LoRA paths
        Tensor<T> inputGrad = new Tensor<T>(loraInputGrad.Shape);
        for (int i = 0; i < loraInputGrad.Length; i++)
        {
            inputGrad[i] = NumOps.Add(loraInputGrad[i], baseInputGrad[i]);
        }

        // Update parameter gradients vector
        UpdateParameterGradientsFromR();

        return inputGrad;
    }

    /// <summary>
    /// Updates the trainable R matrix using the specified learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    /// <remarks>
    /// Only the R matrix is updated; U, Σ, and V remain frozen.
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_trainableRGradient == null)
        {
            return;
        }

        // Update R matrix: R = R - learningRate * dR
        for (int i = 0; i < _trainableR.Rows; i++)
        {
            for (int j = 0; j < _trainableR.Columns; j++)
            {
                T update = NumOps.Multiply(_trainableRGradient[i, j], learningRate);
                _trainableR[i, j] = NumOps.Subtract(_trainableR[i, j], update);
            }
        }

        // Base layer is always frozen in LoRA-XS
        // Update parameter vector
        UpdateParametersFromR();
    }

    /// <summary>
    /// Gets the current parameters as a vector (only R matrix elements).
    /// </summary>
    /// <returns>Vector containing R matrix flattened row-major.</returns>
    public override Vector<T> GetParameters()
    {
        return Parameters.Clone();
    }

    /// <summary>
    /// Sets the layer parameters from a vector (R matrix only).
    /// </summary>
    /// <param name="parameters">Vector containing R matrix elements.</param>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException(
                $"Expected {ParameterCount} parameters (R matrix: {Rank}×{Rank}), got {parameters.Length}",
                nameof(parameters));
        }

        Parameters = parameters.Clone();
        UpdateRFromParameters();
    }

    /// <summary>
    /// Merges the LoRA-XS adaptation into the base layer and returns the merged layer.
    /// </summary>
    /// <returns>A new layer with LoRA-XS weights merged into base weights.</returns>
    /// <remarks>
    /// <para>
    /// Computes: W' = W + U_r * Σ_r * R * V_r^T * scaling
    /// This allows deployment without the adapter overhead.
    /// </para>
    /// <para><b>For Beginners:</b> This "bakes in" your LoRA-XS training.
    ///
    /// After training the R matrix, you can merge it back into the original weights:
    /// - Original weights + learned adaptation = new merged weights
    /// - Deployed model runs at full speed (no adapter overhead)
    /// - You can discard the adapter structure after merging
    ///
    /// This is one of the key advantages: ultra-efficient training, normal-speed inference!
    /// </para>
    /// </remarks>
    public override ILayer<T> MergeToOriginalLayer()
    {
        if (!_initializedFromSVD)
        {
            throw new InvalidOperationException(
                "Cannot merge LoRA-XS adapter that was not initialized from SVD. " +
                "Call InitializeFromSVD first.");
        }

        if (_frozenU == null || _frozenSigma == null || _frozenVt == null)
        {
            throw new InvalidOperationException(
                "LoRA-XS adapter SVD components are not properly initialized.");
        }

        // Support both DenseLayer and FullyConnectedLayer
        DenseLayer<T>? denseBase = _baseLayer as DenseLayer<T>;
        FullyConnectedLayer<T>? fcBase = _baseLayer as FullyConnectedLayer<T>;

        if (denseBase == null && fcBase == null)
        {
            throw new InvalidOperationException(
                "LoRA-XS adapter merging only supports DenseLayer or FullyConnectedLayer base layers. " +
                $"Got: {_baseLayer.GetType().Name}");
        }

        // Compute LoRA-XS weight delta: delta = U_r * Σ_r * R * V_r^T * scaling
        // Where scaling = alpha / rank
        int outputSize = _frozenU.Rows;
        int inputSize = _frozenVt.Columns;
        int rank = Rank;
        double scaling = Alpha / rank;

        // Step 1: Compute R * V_r^T (rank × inputSize)
        var RVt = new Matrix<T>(rank, inputSize);
        for (int i = 0; i < rank; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                T sum = NumOps.Zero;
                for (int k = 0; k < rank; k++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(_trainableR[i, k], _frozenVt[k, j]));
                }
                RVt[i, j] = sum;
            }
        }

        // Step 2: Compute Σ_r * (R * V_r^T) - diagonal scaling (rank × inputSize)
        var SigmaRVt = new Matrix<T>(rank, inputSize);
        for (int i = 0; i < rank; i++)
        {
            T sigma = _frozenSigma[i];
            for (int j = 0; j < inputSize; j++)
            {
                SigmaRVt[i, j] = NumOps.Multiply(sigma, RVt[i, j]);
            }
        }

        // Step 3: Compute U_r * (Σ_r * R * V_r^T) (outputSize × inputSize)
        var loraWeights = new Matrix<T>(outputSize, inputSize);
        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                T sum = NumOps.Zero;
                for (int k = 0; k < rank; k++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(_frozenU[i, k], SigmaRVt[k, j]));
                }
                // Apply scaling factor
                loraWeights[i, j] = NumOps.Multiply(sum, NumOps.FromDouble(scaling));
            }
        }

        // Get base layer parameters
        Vector<T> baseParams = _baseLayer.GetParameters();
        int weightCount = inputSize * outputSize;

        // Create new parameters with merged weights
        Vector<T> mergedParams = new Vector<T>(baseParams.Length);

        // Merge weights: baseWeight + loraWeight
        for (int i = 0; i < weightCount; i++)
        {
            int row = i / inputSize;
            int col = i % inputSize;
            mergedParams[i] = NumOps.Add(baseParams[i], loraWeights[row, col]);
        }

        // Copy biases unchanged (LoRA doesn't modify biases)
        for (int i = weightCount; i < baseParams.Length; i++)
        {
            mergedParams[i] = baseParams[i];
        }

        // Use helper method to clone base layer and preserve activation function
        return CreateMergedLayerWithClone(mergedParams);
    }

    /// <summary>
    /// Resets the internal state of the adapter.
    /// </summary>
    public override void ResetState()
    {
        _baseLayer.ResetState();
        _cachedVtInput = null;
        _cachedRVtInput = null;
        _cachedSigmaRVtInput = null;
        _trainableRGradient = null;
    }

    // ========== Helper Methods ==========

    /// <summary>
    /// Multiplies a matrix by a tensor (treating tensor as batch of vectors).
    /// </summary>
    /// <param name="matrix">Matrix to multiply (m × n).</param>
    /// <param name="tensor">Input tensor (batchSize, n) or (batchSize × n).</param>
    /// <returns>Result tensor (batchSize, m) or (batchSize × m).</returns>
    private Tensor<T> MatrixVectorMultiply(Matrix<T> matrix, Tensor<T> tensor)
    {
        // Determine batch size and vector size from tensor
        int batchSize = tensor.Shape[0];
        int vectorSize = tensor.Shape.Length > 1 ? tensor.Shape[1] : tensor.Length / batchSize;

        if (vectorSize != matrix.Columns)
        {
            throw new ArgumentException(
                $"Matrix columns ({matrix.Columns}) must match tensor vector size ({vectorSize})");
        }

        int outputSize = matrix.Rows;
        Vector<T> resultData = new Vector<T>(batchSize * outputSize);

        // Perform batched matrix-vector multiplication
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < outputSize; i++)
            {
                T sum = NumOps.Zero;
                for (int j = 0; j < vectorSize; j++)
                {
                    int inputIdx = b * vectorSize + j;
                    sum = NumOps.Add(sum, NumOps.Multiply(matrix[i, j], tensor[inputIdx]));
                }
                resultData[b * outputSize + i] = sum;
            }
        }

        return new Tensor<T>(new[] { batchSize, outputSize }, resultData);
    }

    /// <summary>
    /// Applies diagonal scaling by singular values: Σ * x
    /// </summary>
    /// <param name="sigma">Singular values vector (length rank).</param>
    /// <param name="tensor">Input tensor (batchSize, rank).</param>
    /// <returns>Scaled tensor (batchSize, rank).</returns>
    private Tensor<T> ApplySigmaScaling(Vector<T> sigma, Tensor<T> tensor)
    {
        int batchSize = tensor.Shape[0];
        int rank = sigma.Length;

        Vector<T> resultData = new Vector<T>(batchSize * rank);

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < rank; i++)
            {
                int idx = b * rank + i;
                resultData[idx] = NumOps.Multiply(sigma[i], tensor[idx]);
            }
        }

        return new Tensor<T>(new[] { batchSize, rank }, resultData);
    }

    /// <summary>
    /// Scales all elements of a tensor by a scalar value.
    /// </summary>
    private Tensor<T> ScaleTensor(Tensor<T> tensor, T scalar)
    {
        Tensor<T> result = new Tensor<T>(tensor.Shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            result[i] = NumOps.Multiply(tensor[i], scalar);
        }
        return result;
    }

    /// <summary>
    /// Computes gradient matrix: grad = left * right^T
    /// </summary>
    /// <param name="left">Left tensor (batchSize, m).</param>
    /// <param name="right">Right tensor (batchSize, n).</param>
    /// <returns>Gradient matrix (m × n).</returns>
    private Matrix<T> ComputeMatrixGradient(Tensor<T> left, Tensor<T> right)
    {
        int batchSize = left.Shape[0];
        int m = left.Shape.Length > 1 ? left.Shape[1] : left.Length / batchSize;
        int n = right.Shape.Length > 1 ? right.Shape[1] : right.Length / batchSize;

        Matrix<T> gradient = new Matrix<T>(m, n);

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    int leftIdx = b * m + i;
                    int rightIdx = b * n + j;
                    T product = NumOps.Multiply(left[leftIdx], right[rightIdx]);
                    gradient[i, j] = NumOps.Add(gradient[i, j], product);
                }
            }
        }

        return gradient;
    }

    /// <summary>
    /// Updates the parameter vector from the R matrix.
    /// </summary>
    private void UpdateParametersFromR()
    {
        int idx = 0;
        for (int i = 0; i < _trainableR.Rows; i++)
        {
            for (int j = 0; j < _trainableR.Columns; j++)
            {
                Parameters[idx++] = _trainableR[i, j];
            }
        }
    }

    /// <summary>
    /// Overrides base class parameter packing to prevent buffer overrun during base constructor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The base class constructor calls UpdateParametersFromLayers() which tries to pack
    /// _loraLayer.GetParameters() (size 2*d*r). However, LoRAXSAdapter's ParameterCount
    /// returns Rank*Rank (much smaller) before _trainableR is initialized.
    /// This override guards against that early call and delegates to UpdateParametersFromR
    /// once the R matrix is ready.
    /// </para>
    /// </remarks>
    protected override void UpdateParametersFromLayers()
    {
        // Guard against being called from base constructor before _trainableR is initialized
        if (_trainableR == null)
        {
            return;
        }

        // LoRA-XS only stores R matrix parameters, not the underlying _loraLayer
        UpdateParametersFromR();
    }

    /// <summary>
    /// Updates the R matrix from the parameter vector.
    /// </summary>
    private void UpdateRFromParameters()
    {
        int idx = 0;
        for (int i = 0; i < _trainableR.Rows; i++)
        {
            for (int j = 0; j < _trainableR.Columns; j++)
            {
                _trainableR[i, j] = Parameters[idx++];
            }
        }
    }

    /// <summary>
    /// Updates the parameter gradients vector from R matrix gradient.
    /// </summary>
    private void UpdateParameterGradientsFromR()
    {
        if (_trainableRGradient == null)
        {
            return;
        }

        ParameterGradients = new Vector<T>(ParameterCount);
        int idx = 0;
        for (int i = 0; i < _trainableRGradient.Rows; i++)
        {
            for (int j = 0; j < _trainableRGradient.Columns; j++)
            {
                ParameterGradients[idx++] = _trainableRGradient[i, j];
            }
        }
    }
}

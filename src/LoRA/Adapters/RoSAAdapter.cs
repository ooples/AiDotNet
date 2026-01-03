using AiDotNet.Extensions;
using AiDotNet.Interfaces;

namespace AiDotNet.LoRA.Adapters;

/// <summary>
/// RoSA (Robust Adaptation) adapter for parameter-efficient fine-tuning with improved robustness to distribution shifts.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// RoSA (Robust Adaptation) extends standard LoRA by combining two complementary components:
/// 1. Low-rank component (standard LoRA): Captures common, structured patterns in adaptations
/// 2. Sparse component: Captures specific, rare, or outlier patterns that low-rank cannot represent
/// </para>
/// <para>
/// <b>Mathematical Formulation:</b>
/// Given input x and pre-trained weights W, RoSA computes:
/// - Low-rank component: L = (alpha/rank) * B * A * x
/// - Sparse component: S = W_sparse * x (where W_sparse is highly sparse)
/// - Final output: y = W*x + L + S
///
/// The sparse component is maintained through magnitude-based pruning, keeping only the
/// most significant weights and zeroing out the rest. This creates a sparse matrix that
/// captures specific patterns while remaining parameter-efficient.
/// </para>
/// <para>
/// <b>Research Context:</b>
/// RoSA was introduced in January 2024 as a robust alternative to standard LoRA.
/// The key insight is that low-rank approximations work well for common patterns but
/// struggle with distribution shifts and rare patterns. By adding a sparse component,
/// RoSA can capture outliers and domain-specific patterns without significantly
/// increasing parameter count.
///
/// In experiments on domain adaptation tasks, RoSA showed:
/// - Better generalization to new domains (+5-10% over standard LoRA)
/// - More robust to distribution shifts
/// - Ability to capture both global patterns (low-rank) and local exceptions (sparse)
/// - Only modest increase in parameters (typically 5-15% more than pure LoRA)
/// </para>
/// <para>
/// <b>For Beginners:</b> RoSA is like LoRA with a safety net for unusual cases.
///
/// Think of it this way:
/// - Low-rank LoRA is like learning general rules ("most images of cats have pointed ears")
/// - Sparse component is like remembering specific exceptions ("this one cat breed has round ears")
/// - Together they make a robust model that handles both common and rare cases
///
/// Why RoSA is more robust:
/// - Low-rank component: Efficient for common patterns across domains
/// - Sparse component: Handles outliers and domain-specific quirks
/// - Result: Better performance when test data differs from training data
///
/// When to use RoSA over standard LoRA:
/// - When you expect distribution shifts (train on news, test on social media)
/// - When your data has outliers or rare patterns that matter
/// - When you need robustness more than absolute parameter efficiency
/// - When adapting to multiple related but distinct domains
///
/// Trade-offs vs standard LoRA:
/// + More robust to distribution shifts
/// + Better handles rare patterns
/// + More flexible adaptation
/// - Slightly more parameters (sparse component adds ~5-15%)
/// - Slightly more computation (extra sparse matrix multiply)
/// - Requires tuning sparsity ratio
/// </para>
/// <para>
/// <b>Reference:</b>
/// "RoSA: Robust Adaptation through Sparse Regularization"
/// January 2024
/// </para>
/// </remarks>
public class RoSAAdapter<T> : LoRAAdapterBase<T>
{
    /// <summary>
    /// Sparse weight matrix that captures specific/rare patterns.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This matrix has the same dimensions as the base layer's weights but is highly sparse
    /// (typically 90-99% zeros). It's maintained through magnitude-based pruning during training.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the "exception handler" of RoSA.
    /// Most of its values are zero, but the few non-zero values capture specific patterns
    /// that the low-rank component can't represent efficiently.
    /// </para>
    /// </remarks>
    private Matrix<T> _sparseWeights;

    /// <summary>
    /// Gradients for the sparse weight component, computed during backpropagation.
    /// </summary>
    private Matrix<T>? _sparseGradients;

    /// <summary>
    /// Cached input matrix from forward pass, needed for computing sparse weight gradients.
    /// </summary>
    /// <remarks>
    /// Stores the input activations in matrix form [batchSize, inputSize] to enable proper
    /// gradient computation: dL/dW_sparse[i,j] = sum_batch(gradMatrix[b,i] * input[b,j]) / batchSize.
    /// </remarks>
    private Matrix<T>? _cachedInputMatrix;

    /// <summary>
    /// Threshold for magnitude-based pruning of sparse weights.
    /// Weights with magnitude below this threshold are set to zero.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This threshold controls the sparsity of the sparse component. Lower values
    /// result in more non-zero weights (less sparse), higher values result in
    /// fewer non-zero weights (more sparse).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like a "minimum importance" cutoff.
    /// If a weight's importance is below this value, we zero it out to maintain
    /// sparsity. Typical values: 0.001 to 0.1
    /// </para>
    /// </remarks>
    public double SparseThreshold { get; set; }

    /// <summary>
    /// Target sparsity ratio (fraction of zeros in sparse component).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This value controls how sparse the sparse component should be.
    /// - 0.0 = no sparsity (all weights can be non-zero)
    /// - 0.5 = 50% of weights are zero
    /// - 0.95 = 95% of weights are zero (very sparse)
    /// - 0.99 = 99% of weights are zero (extremely sparse)
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the target percentage of zeros we want.
    /// Higher values (like 0.95) mean fewer non-zero weights, which keeps the
    /// model efficient. Lower values mean more flexibility but more parameters.
    ///
    /// Typical values:
    /// - 0.90 (90% zeros): More flexible, for complex domains
    /// - 0.95 (95% zeros): Good balance (recommended starting point)
    /// - 0.99 (99% zeros): Very efficient, for simple adaptations
    /// </para>
    /// </remarks>
    public double SparsityRatio { get; set; }

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// RoSA parameters include:
    /// - Base layer parameters (if not frozen)
    /// - LoRA parameters (rank * (inputSize + outputSize))
    /// - Non-zero sparse parameters (varies based on sparsity)
    ///
    /// For parameter counting, we report the full sparse matrix size, but in practice
    /// only the non-zero elements need to be stored and updated.
    /// </para>
    /// </remarks>
    public override int ParameterCount
    {
        get
        {
            int baseCount = (_baseLayer != null && !_freezeBaseLayer) ? _baseLayer.ParameterCount : 0;
            int loraCount = _loraLayer != null ? _loraLayer.ParameterCount : 0;
            // CRITICAL: Compute sparse count from layer dimensions when _sparseWeights is null
            // Returning 0 causes base constructor to allocate too-small buffer
            int sparseCount = _sparseWeights != null
                ? (_sparseWeights.Rows * _sparseWeights.Columns)
                : (GetOutputShape()[0] * GetInputShape()[0]);
            return baseCount + loraCount + sparseCount;
        }
    }

    /// <summary>
    /// Initializes a new RoSA adapter wrapping an existing layer.
    /// </summary>
    /// <param name="baseLayer">The layer to adapt with RoSA.</param>
    /// <param name="rank">The rank of the low-rank LoRA decomposition.</param>
    /// <param name="alpha">The LoRA scaling factor (defaults to rank if negative).</param>
    /// <param name="sparsityRatio">Target sparsity ratio (0.0 to 1.0, typically 0.9-0.99).</param>
    /// <param name="sparseThreshold">Magnitude threshold for pruning sparse weights (typically 0.001-0.1).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer's parameters during training.</param>
    /// <exception cref="ArgumentNullException">Thrown when baseLayer is null.</exception>
    /// <exception cref="ArgumentException">Thrown when sparsityRatio is not between 0 and 1.</exception>
    /// <remarks>
    /// <para>
    /// The constructor initializes the RoSA adapter by:
    /// 1. Setting up the standard LoRA components (via base constructor)
    /// 2. Initializing the sparse weight matrix (starts with small random values)
    /// 3. Applying initial pruning to enforce sparsity
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This creates a RoSA adapter around your existing layer.
    ///
    /// Parameters:
    /// - baseLayer: The layer you want to fine-tune efficiently and robustly
    /// - rank: How much compression for the low-rank component (lower = fewer parameters)
    /// - alpha: Scaling factor for LoRA contribution (usually equals rank)
    /// - sparsityRatio: How sparse the sparse component should be (0.95 = 95% zeros)
    /// - sparseThreshold: Minimum importance for keeping a sparse weight (0.01 is typical)
    /// - freezeBaseLayer: Usually true - we only train LoRA + sparse, not base weights
    ///
    /// Example: For a 1000x1000 layer with rank=8 and sparsityRatio=0.95:
    /// - Base layer: 1,000,000 parameters (frozen)
    /// - LoRA: 16,000 parameters (8 * (1000 + 1000))
    /// - Sparse: ~50,000 parameters (5% of 1,000,000)
    /// - Total trainable: ~66,000 parameters (vs 1M for full fine-tuning!)
    /// </para>
    /// </remarks>
    public RoSAAdapter(
        ILayer<T> baseLayer,
        int rank,
        double alpha = -1,
        double sparsityRatio = 0.95,
        double sparseThreshold = 0.01,
        bool freezeBaseLayer = true)
        : base(baseLayer, rank, alpha, freezeBaseLayer)
    {
        if (sparsityRatio < 0.0 || sparsityRatio >= 1.0)
        {
            throw new ArgumentException("Sparsity ratio must be between 0.0 and 1.0 (exclusive of 1.0)", nameof(sparsityRatio));
        }

        SparsityRatio = sparsityRatio;
        SparseThreshold = sparseThreshold;

        // Initialize sparse weights
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        _sparseWeights = new Matrix<T>(outputSize, inputSize);

        // Initialize with small random values (will be pruned)
        InitializeSparseWeights();

        // Apply initial pruning to enforce sparsity
        PruneSparseWeights();

        // Update parameters to include sparse component
        Parameters = new Vector<T>(ParameterCount);
        UpdateParametersFromComponents();
    }

    /// <summary>
    /// Initializes sparse weights with small random values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The sparse weights are initialized with small random values drawn from a
    /// normal distribution with standard deviation 0.01. These values will be
    /// pruned based on magnitude to enforce sparsity.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This gives the sparse component a random starting point.
    /// Most of these values will be pruned (set to zero) immediately, but this
    /// initialization ensures we start with a diverse set of potential patterns.
    /// </para>
    /// </remarks>
    private void InitializeSparseWeights()
    {
        Random random = RandomHelper.CreateSecureRandom();
        for (int i = 0; i < _sparseWeights.Rows; i++)
        {
            for (int j = 0; j < _sparseWeights.Columns; j++)
            {
                // Small random initialization
                double value = random.NextGaussian(0.0, 0.01);
                _sparseWeights[i, j] = NumOps.FromDouble(value);
            }
        }
    }

    /// <summary>
    /// Prunes sparse weights based on magnitude to maintain target sparsity.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method implements magnitude-based pruning:
    /// 1. Computes magnitude of all sparse weights
    /// 2. Determines threshold based on target sparsity ratio
    /// 3. Sets weights below threshold to zero
    ///
    /// This ensures the sparse component maintains its sparsity during training.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is like cleaning up the sparse component.
    ///
    /// We keep only the most important weights:
    /// 1. Look at all the weights and their magnitudes
    /// 2. Sort them by importance (magnitude)
    /// 3. Keep the top X% (based on sparsity ratio)
    /// 4. Zero out the rest
    ///
    /// Example with sparsity ratio 0.95:
    /// - We have 1000 weights
    /// - We want 95% zeros (950 zeros, 50 non-zeros)
    /// - Keep the 50 largest magnitudes
    /// - Set the other 950 to zero
    ///
    /// This is called periodically during training to maintain sparsity.
    /// </para>
    /// </remarks>
    public void PruneSparseWeights()
    {
        int rows = _sparseWeights.Rows;
        int cols = _sparseWeights.Columns;
        int totalWeights = rows * cols;

        // Collect magnitudes
        List<(int row, int col, double magnitude)> magnitudes = new List<(int, int, double)>();
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double mag = Math.Abs(Convert.ToDouble(_sparseWeights[i, j]));
                magnitudes.Add((i, j, mag));
            }
        }

        // Sort by magnitude (descending)
        magnitudes.Sort((a, b) => b.magnitude.CompareTo(a.magnitude));

        // Determine number of non-zero weights to keep
        int keepCount = (int)((1.0 - SparsityRatio) * totalWeights);
        keepCount = Math.Max(1, keepCount); // Keep at least one weight

        // Also consider threshold-based pruning
        double adaptiveThreshold = SparseThreshold;
        if (keepCount < magnitudes.Count)
        {
            // Use the larger of: fixed threshold or magnitude of keepCount-th element
            adaptiveThreshold = Math.Max(SparseThreshold, magnitudes[keepCount].magnitude);
        }

        // Apply pruning: zero out weights below threshold
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double mag = Math.Abs(Convert.ToDouble(_sparseWeights[i, j]));
                if (mag < adaptiveThreshold)
                {
                    _sparseWeights[i, j] = NumOps.Zero;
                }
            }
        }
    }

    /// <summary>
    /// Gets the current sparsity of the sparse component.
    /// </summary>
    /// <returns>The fraction of zeros in the sparse weight matrix (0.0 to 1.0).</returns>
    /// <remarks>
    /// <para>
    /// This method computes the actual sparsity by counting zero and near-zero elements.
    /// The result can be compared to SparsityRatio to see how well pruning is working.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This tells you what percentage of the sparse component is actually zero.
    ///
    /// If you set SparsityRatio to 0.95, this should return close to 0.95 after pruning.
    /// If it's much lower, you might need to adjust the threshold or pruning frequency.
    ///
    /// Example return values:
    /// - 0.95 = 95% zeros (good for target of 0.95)
    /// - 0.80 = 80% zeros (less sparse than target)
    /// - 0.99 = 99% zeros (more sparse than target)
    /// </para>
    /// </remarks>
    public double GetSparsity()
    {
        int totalWeights = _sparseWeights.Rows * _sparseWeights.Columns;
        int zeroCount = 0;
        double epsilon = 1e-10;

        for (int i = 0; i < _sparseWeights.Rows; i++)
        {
            for (int j = 0; j < _sparseWeights.Columns; j++)
            {
                double val = Math.Abs(Convert.ToDouble(_sparseWeights[i, j]));
                if (val < epsilon)
                {
                    zeroCount++;
                }
            }
        }

        return (double)zeroCount / totalWeights;
    }

    /// <summary>
    /// Performs the forward pass through RoSA adapter.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output combining base layer, low-rank LoRA, and sparse components.</returns>
    /// <remarks>
    /// <para>
    /// The RoSA forward pass computes:
    /// 1. Base output: y_base = base_layer(input)
    /// 2. LoRA output: y_lora = lora_layer(input)
    /// 3. Sparse output: y_sparse = input @ sparse_weights^T
    /// 4. Final output: y = y_base + y_lora + y_sparse
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is where all three components work together.
    ///
    /// Think of it as three parallel processing paths:
    /// - Base layer: Original pre-trained knowledge (usually frozen)
    /// - LoRA component: Low-rank corrections for common patterns
    /// - Sparse component: Specific corrections for rare patterns
    ///
    /// All three outputs are added together to get the final result.
    /// This combination gives RoSA its robustness: the low-rank handles
    /// common patterns efficiently, while sparse handles outliers.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // 1. Forward through base layer
        Tensor<T> baseOutput = _baseLayer.Forward(input);

        // 2. Forward through LoRA layer (low-rank component)
        Tensor<T> loraOutput = _loraLayer.Forward(input);

        // 3. Forward through sparse component
        // Compute: sparse_output = input @ sparse_weights^T
        int batchSize = input.Shape[0];
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];

        // Convert input to matrix
        Matrix<T> inputMatrix = new Matrix<T>(batchSize, inputSize);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                inputMatrix[i, j] = input[i * inputSize + j];
            }
        }

        // Cache input matrix for gradient computation in backward pass
        _cachedInputMatrix = inputMatrix.Clone();

        // Multiply by sparse weights: [batchSize, inputSize] @ [inputSize, outputSize]
        Matrix<T> sparseOutputMatrix = inputMatrix.Multiply(_sparseWeights.Transpose());

        // Convert to tensor
        Vector<T> sparseOutputData = new Vector<T>(batchSize * outputSize);
        int idx = 0;
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                sparseOutputData[idx++] = sparseOutputMatrix[i, j];
            }
        }
        Tensor<T> sparseOutput = new Tensor<T>(new[] { batchSize, outputSize }, sparseOutputData);

        // 4. Sum all three outputs
        Tensor<T> result = new Tensor<T>(baseOutput.Shape);
        for (int i = 0; i < baseOutput.Length; i++)
        {
            T sum = NumOps.Add(baseOutput[i], loraOutput[i]);
            sum = NumOps.Add(sum, sparseOutput[i]);
            result[i] = sum;
        }

        return result;
    }

    /// <summary>
    /// Performs the backward pass through RoSA adapter.
    /// </summary>
    /// <param name="outputGradient">Gradient flowing back from the next layer.</param>
    /// <returns>Gradient to pass to the previous layer.</returns>
    /// <remarks>
    /// <para>
    /// The backward pass computes gradients for all three components:
    /// 1. LoRA component (via LoRA layer's backward)
    /// 2. Sparse component (direct gradient computation)
    /// 3. Base layer (if not frozen)
    ///
    /// Gradients are accumulated and input gradients are summed.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is where RoSA learns from errors.
    ///
    /// The backward pass tells each component how to improve:
    /// - LoRA component: Update low-rank matrices A and B
    /// - Sparse component: Update the sparse weight matrix
    /// - Base layer: Update if not frozen (usually frozen)
    ///
    /// After this, UpdateParameters() will apply the learning using these gradients.
    /// The sparse gradients will be pruned to maintain sparsity.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        int batchSize = outputGradient.Shape[0];
        int outputSize = GetOutputShape()[0];
        int inputSize = GetInputShape()[0];

        // 1. Backward through LoRA layer
        Tensor<T> loraInputGrad = _loraLayer.Backward(outputGradient);

        // 2. Backward through base layer
        Tensor<T> baseInputGrad = _baseLayer.Backward(outputGradient);

        // 3. Compute gradients for sparse component
        // Sparse gradient: dL/dW_sparse = output_gradient^T @ input
        // Convert output gradient to matrix
        Matrix<T> gradMatrix = new Matrix<T>(batchSize, outputSize);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                gradMatrix[i, j] = outputGradient[i * outputSize + j];
            }
        }

        // Compute sparse weight gradients using cached input from forward pass
        // Formula: dL/dW_sparse[i,j] = sum_over_batch(gradMatrix[b,i] * input[b,j]) / batchSize
        if (_cachedInputMatrix == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass");
        }

        _sparseGradients = new Matrix<T>(outputSize, inputSize);

        // Proper gradient computation with input activation
        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                T gradSum = NumOps.Zero;
                for (int b = 0; b < batchSize; b++)
                {
                    // Multiply output gradient by input activation
                    T term = NumOps.Multiply(gradMatrix[b, i], _cachedInputMatrix[b, j]);
                    gradSum = NumOps.Add(gradSum, term);
                }
                // Average over batch
                _sparseGradients[i, j] = NumOps.Divide(gradSum, NumOps.FromDouble(batchSize));
            }
        }

        // 4. Compute input gradient for sparse component
        // input_grad_sparse = output_gradient @ sparse_weights
        Matrix<T> sparseInputGradMatrix = gradMatrix.Multiply(_sparseWeights);

        // Convert to tensor
        Vector<T> sparseInputGradData = new Vector<T>(batchSize * inputSize);
        int idx = 0;
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                sparseInputGradData[idx++] = sparseInputGradMatrix[i, j];
            }
        }
        Tensor<T> sparseInputGrad = new Tensor<T>(new[] { batchSize, inputSize }, sparseInputGradData);

        // 5. Sum input gradients from all three paths
        Tensor<T> inputGrad = new Tensor<T>(loraInputGrad.Shape);
        for (int i = 0; i < loraInputGrad.Length; i++)
        {
            T sum = NumOps.Add(loraInputGrad[i], baseInputGrad[i]);
            sum = NumOps.Add(sum, sparseInputGrad[i]);
            inputGrad[i] = sum;
        }

        // 6. Pack parameter gradients for optimizer
        // Order: [base_grads (if not frozen) | lora_grads | sparse_grads]
        ParameterGradients = new Vector<T>(ParameterCount);
        int gradIdx = 0;

        // Pack base layer gradients (if not frozen)
        if (!_freezeBaseLayer)
        {
            Vector<T> baseGrads = _baseLayer.GetParameterGradients();
            for (int i = 0; i < baseGrads.Length; i++)
            {
                ParameterGradients[gradIdx++] = baseGrads[i];
            }
        }

        // Pack LoRA gradients
        Vector<T> loraGrads = _loraLayer.GetParameterGradients();
        for (int i = 0; i < loraGrads.Length; i++)
        {
            ParameterGradients[gradIdx++] = loraGrads[i];
        }

        // Pack sparse weight gradients
        if (_sparseGradients != null)
        {
            for (int i = 0; i < _sparseGradients.Rows; i++)
            {
                for (int j = 0; j < _sparseGradients.Columns; j++)
                {
                    ParameterGradients[gradIdx++] = _sparseGradients[i, j];
                }
            }
        }

        return inputGrad;
    }

    /// <summary>
    /// Updates parameters using the specified learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method updates all trainable components:
    /// 1. LoRA layer (always)
    /// 2. Sparse weights (always, then prunes to maintain sparsity)
    /// 3. Base layer (only if not frozen)
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This applies the learning from the backward pass.
    ///
    /// For each component:
    /// - Use the gradients to update parameters
    /// - For sparse weights: update, then prune to maintain sparsity
    /// - This ensures we're always learning while keeping the model efficient
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // 1. Update LoRA layer (always)
        _loraLayer.UpdateParameters(learningRate);

        // 2. Update sparse weights (always)
        if (_sparseGradients != null)
        {
            for (int i = 0; i < _sparseWeights.Rows; i++)
            {
                for (int j = 0; j < _sparseWeights.Columns; j++)
                {
                    T update = NumOps.Multiply(_sparseGradients[i, j], learningRate);
                    _sparseWeights[i, j] = NumOps.Subtract(_sparseWeights[i, j], update);
                }
            }

            // Prune sparse weights to maintain sparsity
            PruneSparseWeights();
        }

        // 3. Update base layer (only if not frozen)
        if (!_freezeBaseLayer)
        {
            _baseLayer.UpdateParameters(learningRate);
        }

        // Update parameter vector
        UpdateParametersFromComponents();
    }

    /// <summary>
    /// Gets the current parameters as a vector.
    /// </summary>
    /// <returns>Vector containing all parameters (base if not frozen, LoRA, sparse).</returns>
    public override Vector<T> GetParameters()
    {
        return Parameters.Clone();
    }

    /// <summary>
    /// Sets the layer parameters from a vector.
    /// </summary>
    /// <param name="parameters">Vector containing all parameters.</param>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException($"Expected {ParameterCount} parameters, got {parameters.Length}", nameof(parameters));
        }

        Parameters = parameters.Clone();
        UpdateComponentsFromParameters();
    }

    /// <summary>
    /// Updates the parameter vector from the current component states.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method packs parameters from all components into a single vector:
    /// [base_params (if not frozen) | lora_params | sparse_weights]
    /// </para>
    /// </remarks>
    private void UpdateParametersFromComponents()
    {
        int idx = 0;

        // Pack base layer parameters (if not frozen)
        if (!_freezeBaseLayer)
        {
            Vector<T> baseParams = _baseLayer.GetParameters();
            for (int i = 0; i < baseParams.Length; i++)
            {
                Parameters[idx++] = baseParams[i];
            }
        }

        // Pack LoRA parameters
        Vector<T> loraParams = _loraLayer.GetParameters();
        for (int i = 0; i < loraParams.Length; i++)
        {
            Parameters[idx++] = loraParams[i];
        }

        // Pack sparse weights (row-major order)
        for (int i = 0; i < _sparseWeights.Rows; i++)
        {
            for (int j = 0; j < _sparseWeights.Columns; j++)
            {
                Parameters[idx++] = _sparseWeights[i, j];
            }
        }
    }

    /// <summary>
    /// Updates the components from the parameter vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method unpacks the parameter vector and distributes values to all components:
    /// [base_params (if not frozen) | lora_params | sparse_weights]
    /// </para>
    /// </remarks>
    private void UpdateComponentsFromParameters()
    {
        int idx = 0;

        // Unpack base layer parameters (if not frozen)
        if (!_freezeBaseLayer)
        {
            int baseParamCount = _baseLayer.ParameterCount;
            Vector<T> baseParams = new Vector<T>(baseParamCount);
            for (int i = 0; i < baseParamCount; i++)
            {
                baseParams[i] = Parameters[idx++];
            }
            _baseLayer.SetParameters(baseParams);
        }

        // Unpack LoRA parameters
        int loraParamCount = _loraLayer.ParameterCount;
        Vector<T> loraParams = new Vector<T>(loraParamCount);
        for (int i = 0; i < loraParamCount; i++)
        {
            loraParams[i] = Parameters[idx++];
        }
        _loraLayer.SetParameters(loraParams);

        // Unpack sparse weights (row-major order)
        for (int i = 0; i < _sparseWeights.Rows; i++)
        {
            for (int j = 0; j < _sparseWeights.Columns; j++)
            {
                _sparseWeights[i, j] = Parameters[idx++];
            }
        }
    }

    /// <summary>
    /// Merges the RoSA adaptation into the base layer and returns the merged layer.
    /// </summary>
    /// <returns>A new layer with both LoRA and sparse weights merged into the base layer's weights.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the base layer type is not supported for merging.</exception>
    /// <remarks>
    /// <para>
    /// This method creates a final layer by merging both components:
    /// - Merged weights: W' = W_base + W_lora + W_sparse
    /// where W_lora = (alpha/rank) * B * A
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This "bakes in" both the LoRA and sparse adaptations for deployment.
    ///
    /// After training with RoSA, you can create a single efficient layer by:
    /// 1. Computing the LoRA weight contribution (B * A)
    /// 2. Adding the sparse weights
    /// 3. Adding both to the base weights
    /// 4. Creating a new layer with the merged weights
    ///
    /// The result is a standard layer that has all the adaptations built in:
    /// - Faster inference (no need for three separate computations)
    /// - Simpler deployment (single layer instead of adapter)
    /// - Same behavior as the RoSA adapter
    /// - Compatible with any system (doesn't need RoSA support)
    ///
    /// Trade-off: You lose the ability to adjust LoRA/sparse contributions separately,
    /// but gain inference speed and simplicity.
    /// </para>
    /// </remarks>
    public override ILayer<T> MergeToOriginalLayer()
    {
        // Support DenseLayer and FullyConnectedLayer
        DenseLayer<T>? denseBase = _baseLayer as DenseLayer<T>;
        FullyConnectedLayer<T>? fcBase = _baseLayer as FullyConnectedLayer<T>;

        if (denseBase == null && fcBase == null)
        {
            throw new InvalidOperationException("RoSAAdapter merging only supports DenseLayer or FullyConnectedLayer base layers");
        }

        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];

        // Get base layer parameters
        Vector<T> baseParams = _baseLayer.GetParameters();
        int weightCount = inputSize * outputSize;

        // Get LoRA weight contribution
        Matrix<T> loraWeights = _loraLayer.MergeWeights();

        // Create merged parameters (weights + biases)
        Vector<T> mergedParams = new Vector<T>(baseParams.Length);

        // Merge weights: W' = W_base + W_lora + W_sparse
        for (int i = 0; i < weightCount; i++)
        {
            int row = i / inputSize;
            int col = i % inputSize;

            T baseWeight = baseParams[i];
            T loraWeight = loraWeights[row, col];
            T sparseWeight = _sparseWeights[row, col];

            // Sum all three components
            T merged = NumOps.Add(baseWeight, loraWeight);
            merged = NumOps.Add(merged, sparseWeight);
            mergedParams[i] = merged;
        }

        // Copy biases unchanged (LoRA and sparse don't modify biases)
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
        _loraLayer.ResetState();
        _sparseGradients = null;
        _cachedInputMatrix = null;
    }
}

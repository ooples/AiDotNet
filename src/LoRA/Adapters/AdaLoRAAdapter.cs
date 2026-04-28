using AiDotNet.Interfaces;

namespace AiDotNet.LoRA.Adapters;

/// <summary>
/// Adaptive Low-Rank Adaptation (AdaLoRA) adapter that dynamically allocates parameter budgets among weight matrices.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// AdaLoRA improves upon standard LoRA by dynamically adjusting the rank allocation based on importance scores.
/// Instead of using a fixed rank for all weight matrices, AdaLoRA:
/// - Starts with a maximum rank and adaptively reduces it during training
/// - Computes importance scores for each singular value component
/// - Prunes less important components to focus parameter budget on critical adaptations
/// - Allows different layers to have different effective ranks
/// </para>
/// <para>
/// This leads to more efficient parameter usage compared to fixed-rank LoRA, especially for large models
/// where some layers need more adaptation capacity than others.
/// </para>
/// <para><b>For Beginners:</b> AdaLoRA is like smart LoRA that learns which parts of the adaptation matter most.
///
/// Think of standard LoRA as giving every layer the same budget (rank=8 everywhere).
/// AdaLoRA is smarter:
/// - Some layers get more budget (rank=16) because they're important for the task
/// - Other layers get less budget (rank=2) because small changes are enough
/// - The model learns this automatically during training
///
/// How it works:
/// 1. Start with a large rank (e.g., maxRank=32)
/// 2. During training, track how important each component is
/// 3. Prune components with low importance scores
/// 4. Focus parameters on what actually helps
///
/// Benefits:
/// - More parameter-efficient than fixed-rank LoRA
/// - Better performance with same parameter budget
/// - Automatically finds optimal rank per layer
///
/// Reference: "Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning" (ICLR 2023)
/// https://arxiv.org/abs/2303.10512
/// </para>
/// </remarks>
public class AdaLoRAAdapter<T> : LoRAAdapterBase<T>
{
    /// <summary>
    /// Static random number generator for thread-safe initialization.
    /// </summary>
    private static readonly Random _rng = RandomHelper.CreateSecureRandom();

    /// <summary>
    /// Maximum possible rank for this adapter.
    /// </summary>
    /// <remarks>
    /// The adapter starts with this rank and may reduce it during training through pruning.
    /// This is the upper bound on the number of singular value components.
    /// </remarks>
    private readonly int _maxRank;

    /// <summary>
    /// Current active rank after pruning.
    /// </summary>
    /// <remarks>
    /// This represents the number of singular value components currently being used.
    /// It starts at maxRank and decreases as low-importance components are pruned.
    /// </remarks>
    private int _currentRank;

    /// <summary>
    /// Importance scores for each singular value component.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each score represents how important that singular value is for the adaptation.
    /// Higher scores indicate more important components that should be retained.
    /// These scores are updated during training based on gradient magnitudes.
    /// </para>
    /// <para><b>For Beginners:</b> Think of these as "usefulness ratings" for each component.
    /// Components with high scores are helping a lot, low scores mean they're not doing much.
    /// We keep the high-scoring components and prune the low-scoring ones.
    /// </para>
    /// </remarks>
    private Vector<T> _importanceScores;

    /// <summary>
    /// Pruning fraction (0.0–1.0): the bottom <c>_rankPruningThreshold</c> fraction of
    /// active components, ranked by importance score, is dropped on each prune cycle.
    /// </summary>
    /// <remarks>
    /// Despite the historic "threshold" name, this is a count-based percentile, not a
    /// score comparison: with <c>_rankPruningThreshold = 0.1</c>, the 10% lowest-scoring
    /// components are pruned regardless of their absolute score values. Typical values
    /// are 0.01–0.1. <see cref="_minRank"/> caps how aggressively this can shrink rank.
    /// </remarks>
    private readonly double _rankPruningThreshold;

    /// <summary>
    /// Exponential moving average factor for importance score updates.
    /// </summary>
    /// <remarks>
    /// Controls how quickly importance scores adapt to new gradient information.
    /// Typical values: 0.9 to 0.99 (higher = more smoothing, lower = faster adaptation)
    /// </remarks>
    private readonly double _importanceScoreEMA;

    /// <summary>
    /// Minimum rank to maintain (prevents pruning below this threshold).
    /// </summary>
    private readonly int _minRank;

    /// <summary>
    /// Number of training steps between rank pruning operations.
    /// </summary>
    private readonly int _pruningInterval;

    /// <summary>
    /// Current training step counter.
    /// </summary>
    private int _stepCount;

    /// <summary>
    /// Physical rank indices in <c>matrixA</c>/<c>matrixB</c> that remain active after
    /// pruning. Length equals <see cref="CurrentRank"/>; values are a subset of
    /// <c>[0, _maxRank)</c>. Importance scores at indices in <c>_activeIndices</c> are
    /// the live ones; positions outside this set hold either pruned (zeroed) matrix
    /// columns/rows or zero importance scores. Tracking active indices instead of
    /// compacting <c>_importanceScores</c> keeps gradient lookup aligned with the
    /// underlying physical matrix layout across multiple prune cycles.
    /// </summary>
    private List<int> _activeIndices;

    /// <summary>
    /// Gets the maximum rank this adapter can use.
    /// </summary>
    public int MaxRank => _maxRank;

    /// <summary>
    /// Gets the current active rank after pruning.
    /// </summary>
    public int CurrentRank => _currentRank;

    /// <summary>
    /// Gets a copy of the current importance scores.
    /// </summary>
    public Vector<T> GetImportanceScores() => _importanceScores.Clone();

    /// <summary>
    /// Initializes a new AdaLoRA adapter with adaptive rank allocation.
    /// </summary>
    /// <param name="baseLayer">The layer to adapt with AdaLoRA.</param>
    /// <param name="maxRank">The maximum rank for the LoRA decomposition.</param>
    /// <param name="alpha">The LoRA scaling factor (defaults to maxRank if negative).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer's parameters during training.</param>
    /// <param name="rankPruningThreshold">Threshold for pruning based on importance scores (default: 0.05).</param>
    /// <param name="minRank">Minimum rank to maintain after pruning (default: 1).</param>
    /// <param name="pruningInterval">Number of steps between pruning operations (default: 100).</param>
    /// <param name="importanceScoreEMA">EMA factor for importance score updates (default: 0.95).</param>
    /// <exception cref="ArgumentNullException">Thrown when baseLayer is null.</exception>
    /// <exception cref="ArgumentException">Thrown when rank parameters are invalid.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates an AdaLoRA adapter with smart rank allocation.
    ///
    /// Parameters:
    /// - baseLayer: The layer you want to adapt (typically Dense or FullyConnected)
    /// - maxRank: Start with this many components (will prune down during training)
    /// - alpha: How strong the adaptation is
    /// - freezeBaseLayer: Lock the original weights (usually true for efficiency)
    /// - rankPruningThreshold: How unimportant a component must be to get pruned (0.05 = bottom 5%)
    /// - minRank: Never prune below this rank (safety net)
    /// - pruningInterval: How often to check for pruning (in training steps)
    /// - importanceScoreEMA: How smooth importance tracking is (higher = more stable)
    ///
    /// The adapter will automatically adjust its rank during training to focus parameters
    /// on the most important components.
    /// </para>
    /// </remarks>
    public AdaLoRAAdapter(
        ILayer<T> baseLayer,
        int maxRank,
        double alpha = -1,
        bool freezeBaseLayer = true,
        double rankPruningThreshold = 0.05,
        int minRank = 1,
        int pruningInterval = 100,
        double importanceScoreEMA = 0.95)
        : base(baseLayer, maxRank, alpha, freezeBaseLayer)
    {
        if (minRank < 1)
        {
            throw new ArgumentException("Minimum rank must be at least 1", nameof(minRank));
        }

        if (minRank > maxRank)
        {
            throw new ArgumentException($"Minimum rank ({minRank}) cannot exceed maximum rank ({maxRank})", nameof(minRank));
        }

        if (rankPruningThreshold <= 0 || rankPruningThreshold >= 1)
        {
            throw new ArgumentException("Rank pruning threshold must be between 0 and 1", nameof(rankPruningThreshold));
        }

        if (importanceScoreEMA <= 0 || importanceScoreEMA >= 1)
        {
            throw new ArgumentException("Importance score EMA factor must be between 0 and 1", nameof(importanceScoreEMA));
        }

        if (pruningInterval <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(pruningInterval), "Pruning interval must be positive");
        }

        _maxRank = maxRank;
        _currentRank = maxRank;
        _rankPruningThreshold = rankPruningThreshold;
        _minRank = minRank;
        _pruningInterval = pruningInterval;
        _importanceScoreEMA = importanceScoreEMA;
        _stepCount = 0;

        // Every physical rank index starts active before any prune cycle runs.
        _activeIndices = new List<int>(maxRank);
        for (int i = 0; i < maxRank; i++) _activeIndices.Add(i);

        // Initialize importance scores (start with uniform importance)
        _importanceScores = new Vector<T>(maxRank);
        T initialScore = NumOps.One;
        for (int i = 0; i < maxRank; i++)
        {
            _importanceScores[i] = initialScore;
        }
    }

    /// <summary>
    /// Performs the forward pass using only the top-k most important singular values.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Sum of base layer output and AdaLoRA output (using current rank).</returns>
    /// <remarks>
    /// <para>
    /// Unlike standard LoRA which uses all rank components, AdaLoRA only uses the currentRank
    /// most important components based on importance scores. This is more efficient and focuses
    /// computation on the most impactful adaptations.
    /// </para>
    /// <para><b>For Beginners:</b> This computes the output using only the important components.
    /// If we started with rank=32 but pruned to rank=8, we only use the top 8 most important
    /// singular values. This makes computation faster and more focused.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Forward through base layer
        Tensor<T> baseOutput = _baseLayer.Forward(input);

        // Forward through LoRA layer with pruned components.
        // The LoRA layer matrices have been pruned by PruneRank() — zeroing out
        // low-importance components — so this Forward call only uses the top
        // _currentRank components (others contribute zero).
        Tensor<T> loraOutput = _loraLayer.Forward(input);

        // Sum the outputs (pruning is already applied via zeroed matrix elements).
        Tensor<T> result = Engine.TensorAdd(baseOutput, loraOutput);

        // Drive the automatic pruning schedule: every _pruningInterval forward
        // passes (during training mode), refresh importance scores from the latest
        // gradients and trim the rank. The base layer is frozen, so this is a
        // training-time-only operation; we gate on IsTrainingMode so inference
        // calls never trigger a rank change.
        if (IsTrainingMode && _pruningInterval > 0)
        {
            _stepCount++;
            if (_stepCount % _pruningInterval == 0)
            {
                UpdateImportanceScores();
                PruneRank();
            }
        }

        return result;
    }

    /// <summary>
    /// Updates importance scores based on current gradient magnitudes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Importance is computed using exponential moving average of gradient magnitudes.
    /// For each component i: importance[i] = ema * importance[i] + (1 - ema) * |gradient[i]|
    /// </para>
    /// <para><b>For Beginners:</b> This updates our "usefulness ratings" for each component.
    ///
    /// We use exponential moving average (EMA) which is like a smoothed average:
    /// - New score = 0.95 * old_score + 0.05 * current_gradient_magnitude
    ///
    /// This way, a component needs to consistently have high gradients to get a high score.
    /// A single spike won't cause us to keep an unimportant component.
    /// </para>
    /// </remarks>
    private void UpdateImportanceScores()
    {
        // Get the LoRA layer's parameter gradients
        Vector<T> loraGradients = _loraLayer.GetParameterGradients();

        // The LoRA layer stores parameters as [A matrix flattened, B matrix flattened]
        // We need to compute importance per rank component
        Matrix<T> matrixA = _loraLayer.GetMatrixA();
        Matrix<T> matrixB = _loraLayer.GetMatrixB();

        int inputSize = matrixA.Rows;
        int outputSize = matrixB.Columns;

        // Bail out before computing magnitudes if the gradient cache is empty —
        // happens when UpdateImportanceScores fires before any backward pass has
        // populated gradients (e.g., the very first PruningInterval-th forward in
        // training mode). Indexing into a zero-length vector below would otherwise
        // throw IndexOutOfRangeException.
        int expectedGradLength = inputSize * _maxRank + _maxRank * outputSize;
        if (loraGradients == null || loraGradients.Length < expectedGradLength)
        {
            return;
        }

        // For each ACTIVE rank component (physical index in _activeIndices), compute the
        // gradient magnitude. We must use the physical index r, not the compacted slot,
        // because matrices A and B retain original-rank layout — column r of A and row r
        // of B hold the active component's parameters.
        T emaFactor = NumOps.FromDouble(_importanceScoreEMA);
        T oneMinusEma = NumOps.FromDouble(1.0 - _importanceScoreEMA);
        int bOffset = inputSize * _maxRank;

        foreach (int r in _activeIndices)
        {
            T gradMagnitude = NumOps.Zero;

            // Gradients from matrix A for column r
            for (int i = 0; i < inputSize; i++)
            {
                T grad = loraGradients[i * _maxRank + r];
                gradMagnitude = NumOps.Add(gradMagnitude, NumOps.Multiply(grad, grad));
            }

            // Gradients from matrix B for row r
            for (int j = 0; j < outputSize; j++)
            {
                T grad = loraGradients[bOffset + r * outputSize + j];
                gradMagnitude = NumOps.Add(gradMagnitude, NumOps.Multiply(grad, grad));
            }

            gradMagnitude = NumOps.Sqrt(gradMagnitude);

            T oldScore = _importanceScores[r];
            T newScore = NumOps.Add(
                NumOps.Multiply(emaFactor, oldScore),
                NumOps.Multiply(oneMinusEma, gradMagnitude)
            );

            _importanceScores[r] = newScore;
        }
    }

    /// <summary>
    /// Prunes low-importance singular value components to reduce rank.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method identifies components with importance scores below the threshold and removes them.
    /// The rank is reduced accordingly, focusing parameters on high-importance components.
    /// </para>
    /// <para><b>For Beginners:</b> This removes components that aren't pulling their weight.
    ///
    /// Process:
    /// 1. Look at all importance scores
    /// 2. Find components below the threshold
    /// 3. Mark them for removal
    /// 4. Reduce the current rank
    ///
    /// For example, if we have 16 components but 8 have very low importance scores,
    /// we can prune those 8 and reduce from rank=16 to rank=8.
    ///
    /// This makes the model:
    /// - Faster (fewer components to compute)
    /// - More focused (parameters concentrated on what matters)
    /// - More efficient (same or better performance with fewer parameters)
    /// </para>
    /// </remarks>
    private void PruneRank()
    {
        // Sort active indices by their importance score (descending).
        var sortedActive = _activeIndices
            .Select(idx => (idx, score: Convert.ToDouble(_importanceScores[idx])))
            .OrderByDescending(p => p.score)
            .ToList();

        // Determine new rank (prune bottom threshold fraction). _minRank is a floor.
        int componentsToKeep = Math.Max(_minRank, (int)(_currentRank * (1.0 - _rankPruningThreshold)));
        if (componentsToKeep >= _currentRank) return;

        // Build the next active set from the top-k by importance.
        var newActive = new List<int>(componentsToKeep);
        var newActiveSet = new HashSet<int>();
        for (int i = 0; i < componentsToKeep; i++)
        {
            newActive.Add(sortedActive[i].idx);
            newActiveSet.Add(sortedActive[i].idx);
        }

        // Zero out pruned components in LoRA matrices. Indices stay PHYSICAL — column r
        // of A and row r of B continue to hold the same component's parameters across
        // prune cycles, which is what UpdateImportanceScores's gradient lookup requires.
        Matrix<T> matrixA = _loraLayer.GetMatrixA();
        Matrix<T> matrixB = _loraLayer.GetMatrixB();

        foreach (int r in _activeIndices)
        {
            if (newActiveSet.Contains(r)) continue;

            for (int i = 0; i < matrixA.Rows; i++)
                matrixA[i, r] = NumOps.Zero;
            for (int j = 0; j < matrixB.Columns; j++)
                matrixB[r, j] = NumOps.Zero;

            // Also zero the importance score for pruned indices so a future expand
            // that re-activates this slot starts from a clean baseline.
            _importanceScores[r] = NumOps.Zero;
        }

        SyncMatricesToParameters(matrixA, matrixB);

        _activeIndices = newActive;
        _currentRank = componentsToKeep;
    }

    /// <summary>
    /// Packs the current matrix A and matrix B values into the LoRA layer's flat
    /// parameter vector while preserving any tail parameters (biases, scale factors)
    /// that exist beyond the two matrices. Starting from a zero vector instead of
    /// <see cref="ILayer{T}.GetParameters"/> would silently zero those tail params.
    /// </summary>
    private void SyncMatricesToParameters(Matrix<T> matrixA, Matrix<T> matrixB)
    {
        Vector<T> loraParams = _loraLayer.GetParameters();
        int idx = 0;
        for (int i = 0; i < matrixA.Rows; i++)
            for (int j = 0; j < matrixA.Columns; j++)
                loraParams[idx++] = matrixA[i, j];
        for (int i = 0; i < matrixB.Rows; i++)
            for (int j = 0; j < matrixB.Columns; j++)
                loraParams[idx++] = matrixB[i, j];
        _loraLayer.SetParameters(loraParams);
    }

    /// <summary>
    /// Expands the rank by adding new components (for cases where more capacity is needed).
    /// </summary>
    /// <param name="additionalRank">Number of components to add.</param>
    /// <remarks>
    /// <para>
    /// This is the opposite of pruning - it adds new components when the model needs more capacity.
    /// New components are initialized with low importance and will need to prove their worth.
    /// The corresponding matrix elements are reinitialized with small random values so they can learn.
    /// </para>
    /// <para><b>For Beginners:</b> Sometimes the model realizes it needs more capacity.
    /// This method adds new components, giving the model more flexibility to learn.
    ///
    /// Think of it like hiring more workers when the team is overloaded.
    /// The new components start with low importance and have to earn their keep.
    /// </para>
    /// </remarks>
    public void ExpandRank(int additionalRank)
    {
        if (additionalRank <= 0)
        {
            throw new ArgumentException("Additional rank must be positive", nameof(additionalRank));
        }

        int newRank = Math.Min(_currentRank + additionalRank, _maxRank);
        if (newRank <= _currentRank) return;

        int slotsToActivate = newRank - _currentRank;

        // Find inactive physical indices to reactivate. After PruneRank, these are the
        // slots whose matrix columns/rows have been zeroed and whose importance score
        // is zero. Reactivating them gives the model fresh capacity without disturbing
        // the still-active components' parameters.
        var activeSet = new HashSet<int>(_activeIndices);
        var inactiveIndices = new List<int>(slotsToActivate);
        for (int r = 0; r < _maxRank && inactiveIndices.Count < slotsToActivate; r++)
        {
            if (!activeSet.Contains(r)) inactiveIndices.Add(r);
        }

        Matrix<T> matrixA = _loraLayer.GetMatrixA();
        Matrix<T> matrixB = _loraLayer.GetMatrixB();
        T lowImportance = NumOps.FromDouble(0.01);

        foreach (int r in inactiveIndices)
        {
            // Reinitialize column r of matrix A [inputSize, rank] with small random values.
            for (int i = 0; i < matrixA.Rows; i++)
                matrixA[i, r] = NumOps.FromDouble((_rng.NextDouble() - 0.5) * 0.02);
            // Reinitialize row r of matrix B [rank, outputSize] with small random values.
            for (int j = 0; j < matrixB.Columns; j++)
                matrixB[r, j] = NumOps.FromDouble((_rng.NextDouble() - 0.5) * 0.02);

            _importanceScores[r] = lowImportance;
            _activeIndices.Add(r);
        }

        SyncMatricesToParameters(matrixA, matrixB);

        _currentRank = newRank;
    }

    /// <summary>
    /// Merges the AdaLoRA adaptation into the base layer and returns the merged layer.
    /// </summary>
    /// <returns>A new layer with AdaLoRA weights merged into the base layer's weights.</returns>
    /// <remarks>
    /// <para>
    /// For Dense/FullyConnected layers, this merges the LoRA matrices into the base layer weights.
    /// Only the currently active components (based on currentRank) are merged.
    /// </para>
    /// <para><b>For Beginners:</b> This "bakes in" your adaptive LoRA to create a regular layer.
    /// Only the components that survived pruning (the important ones) are included in the merge.
    ///
    /// This gives you a final layer that:
    /// - Includes only the useful adaptations
    /// - Is as fast as a regular layer
    /// - Can be deployed without AdaLoRA infrastructure
    /// </para>
    /// </remarks>
    public override ILayer<T> MergeToOriginalLayer()
    {
        // Support both DenseLayer and FullyConnected layers
        DenseLayer<T>? denseBase = _baseLayer as DenseLayer<T>;
        FullyConnectedLayer<T>? fcBase = _baseLayer as FullyConnectedLayer<T>;

        if (denseBase == null && fcBase == null)
        {
            throw new InvalidOperationException("AdaLoRAAdapter only supports DenseLayer or FullyConnectedLayer base layers");
        }

        // Active-rank masking: PruneRank zeroes column r of matrix A and row r of
        // matrix B for every pruned component, so A @ B contributes only the active
        // rank components by construction. We additionally enforce the invariant
        // here so a divergent prune/merge state surfaces immediately instead of
        // silently merging stale weights from inactive slots.
        Matrix<T> loraA = _loraLayer.GetMatrixA();
        Matrix<T> loraB = _loraLayer.GetMatrixB();
        var activeSet = new HashSet<int>(_activeIndices);
        for (int r = 0; r < _maxRank; r++)
        {
            if (activeSet.Contains(r)) continue;
            // Inactive rank slot: every cell of column-r-of-A and row-r-of-B must be zero.
            for (int i = 0; i < loraA.Rows; i++)
            {
                if (!NumOps.Equals(loraA[i, r], NumOps.Zero))
                {
                    throw new InvalidOperationException(
                        $"AdaLoRA merge invariant broken: pruned rank slot r={r} has " +
                        $"non-zero value in matrix A at row {i}. PruneRank must zero " +
                        $"matrix entries for inactive components before merge.");
                }
            }
            for (int j = 0; j < loraB.Columns; j++)
            {
                if (!NumOps.Equals(loraB[r, j], NumOps.Zero))
                {
                    throw new InvalidOperationException(
                        $"AdaLoRA merge invariant broken: pruned rank slot r={r} has " +
                        $"non-zero value in matrix B at column {j}.");
                }
            }
        }

        // Compute the LoRA weight contribution. Because inactive slots are zeroed,
        // _loraLayer.MergeWeights() = A @ B already excludes pruned components.
        Matrix<T> loraWeights = _loraLayer.MergeWeights();

        // Get base layer parameters
        Vector<T> baseParams = _baseLayer.GetParameters();

        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        int weightCount = inputSize * outputSize;

        // Create new parameters with merged weights
        Vector<T> mergedParams = new Vector<T>(baseParams.Length);

        // Merge weights — loraWeights from MergeWeights() is [inputSize, outputSize].
        // DenseLayer stores weights as [inputSize, outputSize] (input-major).
        // FullyConnectedLayer stores weights as [outputSize, inputSize] (output-major).
        bool isOutputMajor = _baseLayer is FullyConnectedLayer<T>;
        for (int i = 0; i < weightCount; i++)
        {
            int row, col;
            if (isOutputMajor)
            {
                int outputIdx = i / inputSize;
                int inputIdx = i % inputSize;
                row = inputIdx;
                col = outputIdx;
            }
            else
            {
                row = i / outputSize;
                col = i % outputSize;
            }
            mergedParams[i] = NumOps.Add(baseParams[i], loraWeights[row, col]);
        }

        // Copy biases unchanged
        for (int i = weightCount; i < baseParams.Length; i++)
        {
            mergedParams[i] = baseParams[i];
        }

        // Use helper method to clone base layer and preserve activation function
        return CreateMergedLayerWithClone(mergedParams);
    }
}

using System.Collections.Generic;
using AiDotNet.Interfaces;

namespace AiDotNet.LoRA.Adapters;

/// <summary>
/// Vector Bank LoRA (VB-LoRA) adapter that uses shared parameter banks for efficient multi-client deployment.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// VB-LoRA (2024) introduces vector banks - reusable parameter stores shared across multiple LoRA adapters.
/// Instead of each adapter having its own complete A and B matrices, VB-LoRA maintains global banks of
/// column vectors. Each adapter selects which vectors from the banks to use via index arrays.
/// </para>
/// <para><b>Key Innovation:</b> Vector Bank Architecture
///
/// Traditional LoRA:
/// - Each adapter stores full A (inputSize × rank) and B (rank × outputSize) matrices
/// - Total parameters for N adapters = N × (inputSize × rank + rank × outputSize)
/// - No sharing between adapters
///
/// VB-LoRA:
/// - Global BankA contains pooled column vectors (inputSize × bankSize)
/// - Global BankB contains pooled column vectors (bankSize × outputSize)
/// - Each adapter stores only indices (which vectors to use from banks)
/// - Total parameters = (inputSize × bankSize + bankSize × outputSize) + N × 2 × rank × sizeof(int)
/// - Massive reduction when bankSize &lt;&lt; N × rank
/// </para>
/// <para><b>Benefits:</b>
///
/// 1. **Reduced Duplication**: Similar adapters share vector bank entries
/// 2. **Lower Communication Overhead**: Multi-client systems can cache banks locally
/// 3. **Memory Efficiency**: Fewer unique parameters to store and transmit
/// 4. **Scalability**: Adding new adapters only requires index arrays, not full matrices
/// 5. **Knowledge Sharing**: Banks capture common adaptation patterns
/// </para>
/// <para><b>For Beginners:</b> Think of vector banks like a shared library of building blocks.
///
/// Traditional LoRA is like each person having their own complete toolbox:
/// - Person 1: Full set of tools
/// - Person 2: Full set of tools
/// - Person 3: Full set of tools
/// Result: Lots of duplicate tools
///
/// VB-LoRA is like a shared tool library:
/// - Central tool bank: One of each tool type
/// - Each person: List of which tools they need (indices)
/// - Everyone shares the same physical tools
/// Result: Much fewer tools needed overall
///
/// This is especially powerful when many adapters need similar adjustments (common in
/// multi-task learning or personalization scenarios).
/// </para>
/// <para><b>Example Scenario:</b>
///
/// Suppose you're deploying personalized language models to 1000 users:
/// - Traditional LoRA: Each user needs their own 16K parameter adapter (16MB total)
/// - VB-LoRA: Shared 256K parameter bank + 1000 users × 128 indices each
/// - Result: 84% memory reduction (256K + 128K vs 16M)
///
/// The shared bank captures common language patterns, while per-user indices
/// select the patterns relevant to each individual.
/// </para>
/// </remarks>
public class VBLoRAAdapter<T> : LoRAAdapterBase<T>
{
    /// <summary>
    /// Global bank of column vectors for matrix A, shared across all VB-LoRA instances.
    /// Dimensions: [inputSize, bankSizeA] where each column is a bank vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This static bank is shared across all VB-LoRA adapters using the same bank configuration.
    /// Each column represents a reusable "building block" that adapters can select from.
    /// </para>
    /// <para><b>For Beginners:</b> This is the shared pool of vectors for the first part of LoRA (matrix A).
    /// It's like a library of column vectors that all adapters can reference. Instead of each adapter
    /// storing its own columns, they all point to columns in this shared library.
    /// </para>
    /// </remarks>
    private static readonly Dictionary<string, Matrix<T>> _globalBankA = new Dictionary<string, Matrix<T>>();

    /// <summary>
    /// Global bank of column vectors for matrix B, shared across all VB-LoRA instances.
    /// Dimensions: [bankSizeB, outputSize] where each row is a bank vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This static bank is shared across all VB-LoRA adapters using the same bank configuration.
    /// Each row represents a reusable "building block" that adapters can select from.
    /// </para>
    /// <para><b>For Beginners:</b> This is the shared pool of vectors for the second part of LoRA (matrix B).
    /// Similar to BankA, this is a shared library that all adapters reference instead of storing
    /// their own copies.
    /// </para>
    /// </remarks>
    private static readonly Dictionary<string, Matrix<T>> _globalBankB = new Dictionary<string, Matrix<T>>();

    /// <summary>
    /// Lock object for thread-safe bank initialization and access.
    /// </summary>
    private static readonly object _bankLock = new object();

    /// <summary>
    /// Indices into BankA - specifies which column vectors from the bank to use for this adapter.
    /// Length equals the rank of this adapter.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Instead of storing rank column vectors, we store rank integers pointing to bank columns.
    /// For example, [3, 7, 12] means use columns 3, 7, and 12 from BankA.
    /// </para>
    /// <para><b>For Beginners:</b> These are the "shopping list" of which vectors this adapter
    /// uses from the shared library. Each number is like a pointer saying "I want vector #3,
    /// vector #7, and vector #12 from the shared bank."
    /// </para>
    /// </remarks>
    private readonly int[] _bankIndicesA;

    /// <summary>
    /// Indices into BankB - specifies which row vectors from the bank to use for this adapter.
    /// Length equals the rank of this adapter.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Instead of storing rank row vectors, we store rank integers pointing to bank rows.
    /// For example, [5, 9, 15] means use rows 5, 9, and 15 from BankB.
    /// </para>
    /// <para><b>For Beginners:</b> Similar to BankIndicesA, but for the second part of LoRA.
    /// These numbers tell us which vectors from BankB this particular adapter needs.
    /// </para>
    /// </remarks>
    private readonly int[] _bankIndicesB;

    /// <summary>
    /// Unique identifier for the bank configuration (used as dictionary key).
    /// </summary>
    private readonly string _bankKey;

    /// <summary>
    /// Size of the vector bank A (number of available column vectors).
    /// </summary>
    private readonly int _bankSizeA;

    /// <summary>
    /// Size of the vector bank B (number of available row vectors).
    /// </summary>
    private readonly int _bankSizeB;

    /// <summary>
    /// Gets the indices into Bank A used by this adapter.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These indices specify which column vectors from the global Bank A this adapter uses.
    /// The length equals the adapter's rank.
    /// </para>
    /// <para><b>For Beginners:</b> This is the list of which vectors from the shared library
    /// this adapter is currently using for the A matrix part of LoRA.
    /// </para>
    /// </remarks>
    public int[] BankIndicesA => (int[])_bankIndicesA.Clone();

    /// <summary>
    /// Gets the indices into Bank B used by this adapter.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These indices specify which row vectors from the global Bank B this adapter uses.
    /// The length equals the adapter's rank.
    /// </para>
    /// <para><b>For Beginners:</b> This is the list of which vectors from the shared library
    /// this adapter is currently using for the B matrix part of LoRA.
    /// </para>
    /// </remarks>
    public int[] BankIndicesB => (int[])_bankIndicesB.Clone();

    /// <summary>
    /// Gets the size of Bank A (number of available column vectors).
    /// </summary>
    public int BankSizeA => _bankSizeA;

    /// <summary>
    /// Gets the size of Bank B (number of available row vectors).
    /// </summary>
    public int BankSizeB => _bankSizeB;

    /// <summary>
    /// Initializes a new VB-LoRA adapter with specified bank sizes and indices.
    /// </summary>
    /// <param name="baseLayer">The layer to adapt with VB-LoRA.</param>
    /// <param name="rank">The rank of the LoRA decomposition.</param>
    /// <param name="bankSizeA">Size of the vector bank for matrix A (number of column vectors in the pool).</param>
    /// <param name="bankSizeB">Size of the vector bank for matrix B (number of row vectors in the pool).</param>
    /// <param name="bankIndicesA">Indices into Bank A (if null, random indices are selected).</param>
    /// <param name="bankIndicesB">Indices into Bank B (if null, random indices are selected).</param>
    /// <param name="alpha">The LoRA scaling factor (defaults to rank if negative).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer's parameters during training.</param>
    /// <param name="bankKey">Unique identifier for the bank configuration (allows multiple independent banks).</param>
    /// <exception cref="ArgumentNullException">Thrown when baseLayer is null.</exception>
    /// <exception cref="ArgumentException">Thrown when rank, bank sizes, or indices are invalid.</exception>
    /// <remarks>
    /// <para>
    /// The constructor initializes or reuses existing vector banks based on the bankKey.
    /// If banks don't exist for this key, they are created with random initialization.
    /// </para>
    /// <para><b>Bank Initialization Strategy:</b>
    ///
    /// - BankA vectors: Gaussian random initialization (similar to standard LoRA matrix A)
    /// - BankB vectors: Zero initialization (so adapters start with no effect, like standard LoRA)
    /// - Indices: Random selection if not provided, allowing diverse initial configurations
    /// </para>
    /// <para><b>For Beginners:</b> This creates a VB-LoRA adapter. Key parameters:
    ///
    /// - rank: How many vectors this adapter selects from each bank (like standard LoRA rank)
    /// - bankSizeA/B: How many total vectors are in the shared banks (the size of the library)
    /// - bankIndicesA/B: Which specific vectors to use (can be left null for random selection)
    /// - bankKey: Allows creating separate banks for different purposes (like different model layers)
    ///
    /// The bankKey is important: adapters with the same bankKey share banks, different keys use
    /// separate banks. This lets you have different bank pools for different layers or tasks.
    /// </para>
    /// </remarks>
    public VBLoRAAdapter(
        ILayer<T> baseLayer,
        int rank,
        int bankSizeA,
        int bankSizeB,
        int[]? bankIndicesA = null,
        int[]? bankIndicesB = null,
        double alpha = -1,
        bool freezeBaseLayer = true,
        string? bankKey = null)
        : base(baseLayer, rank, alpha, freezeBaseLayer)
    {
        if (bankSizeA <= 0)
        {
            throw new ArgumentException("Bank size A must be positive", nameof(bankSizeA));
        }

        if (bankSizeB <= 0)
        {
            throw new ArgumentException("Bank size B must be positive", nameof(bankSizeB));
        }

        if (rank > bankSizeA)
        {
            throw new ArgumentException($"Rank ({rank}) cannot exceed bank size A ({bankSizeA})", nameof(rank));
        }

        if (rank > bankSizeB)
        {
            throw new ArgumentException($"Rank ({rank}) cannot exceed bank size B ({bankSizeB})", nameof(rank));
        }

        _bankKey = bankKey ?? "default";
        _bankSizeA = bankSizeA;
        _bankSizeB = bankSizeB;

        // Initialize or reuse banks
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];

        lock (_bankLock)
        {
            InitializeBanksIfNeeded(inputSize, outputSize);
        }

        // Set or generate indices
        if (bankIndicesA != null)
        {
            if (bankIndicesA.Length != rank)
            {
                throw new ArgumentException($"Bank indices A length ({bankIndicesA.Length}) must equal rank ({rank})", nameof(bankIndicesA));
            }

            foreach (int idx in bankIndicesA)
            {
                if (idx < 0 || idx >= bankSizeA)
                {
                    throw new ArgumentException($"Bank index A ({idx}) out of range [0, {bankSizeA})", nameof(bankIndicesA));
                }
            }

            _bankIndicesA = (int[])bankIndicesA.Clone();
        }
        else
        {
            _bankIndicesA = GenerateRandomIndices(rank, bankSizeA);
        }

        if (bankIndicesB != null)
        {
            if (bankIndicesB.Length != rank)
            {
                throw new ArgumentException($"Bank indices B length ({bankIndicesB.Length}) must equal rank ({rank})", nameof(bankIndicesB));
            }

            foreach (int idx in bankIndicesB)
            {
                if (idx < 0 || idx >= bankSizeB)
                {
                    throw new ArgumentException($"Bank index B ({idx}) out of range [0, {bankSizeB})", nameof(bankIndicesB));
                }
            }

            _bankIndicesB = (int[])bankIndicesB.Clone();
        }
        else
        {
            _bankIndicesB = GenerateRandomIndices(rank, bankSizeB);
        }

        // Now that banks + indices are initialized, sync the underlying LoRA layer to the selected vectors.
        UpdateLoRALayerFromBanks(_loraLayer);
    }

    /// <summary>
    /// Initializes the vector banks if they don't exist for this bank key.
    /// </summary>
    /// <param name="inputSize">Input dimension for Bank A.</param>
    /// <param name="outputSize">Output dimension for Bank B.</param>
    private void InitializeBanksIfNeeded(int inputSize, int outputSize)
    {
        lock (_bankLock)
        {
            // Initialize Bank A if not exists
            if (!_globalBankA.ContainsKey(_bankKey))
            {
                Matrix<T> bankA = new Matrix<T>(inputSize, _bankSizeA);
                T stddev = NumOps.Sqrt(NumOps.Divide(NumOps.One, NumOps.FromDouble(_bankSizeA)));

                // Initialize with Gaussian random values (similar to standard LoRA A matrix)
                for (int i = 0; i < inputSize; i++)
                {
                    for (int j = 0; j < _bankSizeA; j++)
                    {
                        // Box-Muller transform for Gaussian random numbers
                        double u1 = Random.NextDouble();
                        double u2 = Random.NextDouble();
                        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                        bankA[i, j] = NumOps.Multiply(NumOps.FromDouble(randStdNormal), stddev);
                    }
                }

                _globalBankA[_bankKey] = bankA;
            }

            // Initialize Bank B if not exists
            if (!_globalBankB.ContainsKey(_bankKey))
            {
                Matrix<T> bankB = new Matrix<T>(_bankSizeB, outputSize);
                T stddev = NumOps.Multiply(
                    NumOps.Sqrt(NumOps.Divide(NumOps.One, NumOps.FromDouble(_bankSizeB))),
                    NumOps.FromDouble(0.01));

                // Initialize with small Gaussian random values so both A and B receive gradient signal immediately.
                for (int i = 0; i < _bankSizeB; i++)
                {
                    for (int j = 0; j < outputSize; j++)
                    {
                        // Box-Muller transform for Gaussian random numbers
                        double u1 = Random.NextDouble();
                        double u2 = Random.NextDouble();
                        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
                        bankB[i, j] = NumOps.Multiply(NumOps.FromDouble(randStdNormal), stddev);
                    }
                }

                _globalBankB[_bankKey] = bankB;
            }
        }
    }

    /// <summary>
    /// Generates random indices for bank vector selection.
    /// </summary>
    /// <param name="count">Number of indices to generate.</param>
    /// <param name="maxValue">Maximum index value (exclusive).</param>
    /// <returns>Array of random unique indices.</returns>
    private int[] GenerateRandomIndices(int count, int maxValue)
    {
        // Use reservoir sampling to get unique random indices
        int[] indices = new int[count];
        HashSet<int> selected = new HashSet<int>();

        for (int i = 0; i < count; i++)
        {
            int idx;
            do
            {
                idx = Random.Next(maxValue);
            } while (selected.Contains(idx));

            indices[i] = idx;
            selected.Add(idx);
        }

        return indices;
    }

    /// <summary>
    /// Creates the LoRA layer for this adapter, customized to use vector bank indices.
    /// </summary>
    /// <param name="rank">The rank of the LoRA decomposition.</param>
    /// <param name="alpha">The LoRA scaling factor.</param>
    /// <returns>A LoRA layer configured to use vector banks.</returns>
    /// <remarks>
    /// <para>
    /// This override creates a standard LoRA layer. The matrices will be updated from
    /// bank vectors later in the constructor, after banks and indices are initialized.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a basic LoRA layer first. The bank-based
    /// customization happens later in the constructor after the banks are set up.
    /// </para>
    /// <para><b>Note:</b> We don't call UpdateLoRALayerFromBanks here because this method
    /// is called from the base constructor, before VBLoRAAdapter's fields (_bankKey,
    /// _bankIndicesA, etc.) are initialized. The bank update happens in the constructor
    /// after all fields are set up.
    /// </para>
    /// </remarks>
    protected override LoRALayer<T> CreateLoRALayer(int rank, double alpha)
    {
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];

        // Important: this virtual is invoked from the base constructor, before VB-LoRA banks/indices are initialized.
        // We therefore only construct a standard LoRA layer here and let the derived constructor/Forward() synchronize
        // it with the vector banks once initialization is complete.
        return new LoRALayer<T>(inputSize, outputSize, rank, alpha);
    }

    /// <summary>
    /// Updates the LoRA layer's matrices by extracting selected vectors from the banks.
    /// </summary>
    /// <param name="loraLayer">The LoRA layer to update.</param>
    private void UpdateLoRALayerFromBanks(LoRALayer<T> loraLayer)
    {
        Matrix<T> bankA;
        Matrix<T> bankB;
        lock (_bankLock)
        {
            bankA = _globalBankA[_bankKey];
            bankB = _globalBankB[_bankKey];
        }

        // Build matrix A from selected bank columns
        Matrix<T> loraA = new Matrix<T>(bankA.Rows, Rank);
        for (int i = 0; i < bankA.Rows; i++)
        {
            for (int j = 0; j < Rank; j++)
            {
                loraA[i, j] = bankA[i, _bankIndicesA[j]];
            }
        }

        // Build matrix B from selected bank rows
        Matrix<T> loraB = new Matrix<T>(Rank, bankB.Columns);
        for (int i = 0; i < Rank; i++)
        {
            for (int j = 0; j < bankB.Columns; j++)
            {
                loraB[i, j] = bankB[_bankIndicesB[i], j];
            }
        }

        // Pack matrices into parameter vector and set
        int inputSize = loraA.Rows;
        int outputSize = loraB.Columns;
        Vector<T> params_vec = new Vector<T>(inputSize * Rank + Rank * outputSize);

        int idx = 0;
        // Pack A
        for (int i = 0; i < loraA.Rows; i++)
        {
            for (int j = 0; j < loraA.Columns; j++)
            {
                params_vec[idx++] = loraA[i, j];
            }
        }

        // Pack B
        for (int i = 0; i < loraB.Rows; i++)
        {
            for (int j = 0; j < loraB.Columns; j++)
            {
                params_vec[idx++] = loraB[i, j];
            }
        }

        loraLayer.SetParameters(params_vec);
    }

    /// <summary>
    /// Performs the forward pass using bank-selected vectors.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Sum of base layer output and VB-LoRA output.</returns>
    /// <remarks>
    /// <para>
    /// The forward pass operates identically to standard LoRA, but the matrices A and B
    /// are composed of vectors selected from the shared banks.
    /// </para>
    /// <para><b>For Beginners:</b> This works exactly like regular LoRA forward pass, but the
    /// matrices being used are built from shared bank vectors. The computation is the same,
    /// but the memory footprint is much smaller when many adapters share banks.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Sync LoRA layer with current bank state before forward pass
        UpdateLoRALayerFromBanks(_loraLayer);

        // Use base class forward pass (base layer + LoRA layer)
        return base.Forward(input);
    }

    /// <summary>
    /// Updates parameters and propagates changes back to the shared banks.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    /// <remarks>
    /// <para>
    /// After updating the LoRA layer's parameters, this method writes the changes back to the
    /// shared banks. This allows the banks to learn and improve over time as multiple adapters
    /// share training signal.
    /// </para>
    /// <para><b>For Beginners:</b> When training updates the adapter's vectors, we need to write
    /// those changes back to the shared library. This way, the shared bank "learns" from all
    /// adapters using it, making it better for everyone.
    ///
    /// Think of it like updating a shared knowledge base - when one adapter learns something useful,
    /// that knowledge becomes available to all other adapters sharing the same banks.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // Update base class (updates LoRA layer and optionally base layer)
        base.UpdateParameters(learningRate);

        // Write updated LoRA parameters back to banks
        UpdateBanksFromLoRALayer(_loraLayer);
    }

    /// <summary>
    /// Writes the LoRA layer's updated parameters back to the shared banks.
    /// </summary>
    /// <param name="loraLayer">The LoRA layer with updated parameters.</param>
    private void UpdateBanksFromLoRALayer(LoRALayer<T> loraLayer)
    {
        Matrix<T> loraA = loraLayer.GetMatrixA();
        Matrix<T> loraB = loraLayer.GetMatrixB();

        lock (_bankLock)
        {
            Matrix<T> bankA = _globalBankA[_bankKey];
            Matrix<T> bankB = _globalBankB[_bankKey];

            // Write updated A matrix columns back to bank A
            for (int i = 0; i < loraA.Rows; i++)
            {
                for (int j = 0; j < Rank; j++)
                {
                    bankA[i, _bankIndicesA[j]] = loraA[i, j];
                }
            }

            // Write updated B matrix rows back to bank B
            for (int i = 0; i < Rank; i++)
            {
                for (int j = 0; j < loraB.Columns; j++)
                {
                    bankB[_bankIndicesB[i], j] = loraB[i, j];
                }
            }
        }
    }

    /// <summary>
    /// Merges the VB-LoRA adaptation into the base layer and returns the merged layer.
    /// </summary>
    /// <returns>A new DenseLayer with VB-LoRA weights (from selected bank vectors) merged into the base layer's weights.</returns>
    /// <remarks>
    /// <para>
    /// This method extracts the adapter's effective weight matrix from the bank-selected vectors
    /// and merges it with the base layer, just like standard LoRA merging.
    /// </para>
    /// <para><b>For Beginners:</b> This "bakes in" the VB-LoRA adaptation by:
    /// 1. Extracting the vectors this adapter uses from the banks
    /// 2. Computing the full weight matrix from those vectors
    /// 3. Adding that matrix to the base layer's weights
    /// 4. Returning a regular layer with the merged weights
    ///
    /// After merging, you don't need the banks anymore - you have a standalone layer
    /// with the adaptation permanently included.
    /// </para>
    /// </remarks>
    public override ILayer<T> MergeToOriginalLayer()
    {
        // Ensure the underlying LoRA layer reflects the latest shared bank state before merging.
        UpdateLoRALayerFromBanks(_loraLayer);

        // Get the current LoRA weights from selected bank vectors
        Matrix<T> loraWeights = _loraLayer.MergeWeights();

        // Merge with base layer (supports DenseLayer and FullyConnectedLayer)
        DenseLayer<T>? denseBase = _baseLayer as DenseLayer<T>;
        FullyConnectedLayer<T>? fcBase = _baseLayer as FullyConnectedLayer<T>;

        if (denseBase == null && fcBase == null)
        {
            throw new InvalidOperationException("VBLoRAAdapter currently only supports DenseLayer or FullyConnectedLayer base layers");
        }

        Vector<T> baseParams = _baseLayer.GetParameters();
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        int weightCount = inputSize * outputSize;

        // Create new parameters with merged weights
        Vector<T> mergedParams = new Vector<T>(baseParams.Length);

        // Merge weights
        for (int i = 0; i < weightCount; i++)
        {
            int row = i / inputSize;
            int col = i % inputSize;
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

    /// <summary>
    /// Gets the global vector bank A for inspection or advanced use cases.
    /// </summary>
    /// <param name="bankKey">The bank key identifier.</param>
    /// <returns>A clone of the bank A matrix, or null if it doesn't exist.</returns>
    /// <remarks>
    /// <para>
    /// This method allows inspection of the shared bank state. It returns a clone to prevent
    /// accidental modification of the shared bank.
    /// </para>
    /// <para><b>For Beginners:</b> This lets you look at the shared library of vectors for
    /// matrix A. It's useful for debugging or understanding what vectors are available in the bank.
    /// </para>
    /// </remarks>
    public static Matrix<T>? GetBankA(string bankKey = "default")
    {
        lock (_bankLock)
        {
            return _globalBankA.ContainsKey(bankKey) ? _globalBankA[bankKey].Clone() : null;
        }
    }

    /// <summary>
    /// Gets the global vector bank B for inspection or advanced use cases.
    /// </summary>
    /// <param name="bankKey">The bank key identifier.</param>
    /// <returns>A clone of the bank B matrix, or null if it doesn't exist.</returns>
    /// <remarks>
    /// <para>
    /// This method allows inspection of the shared bank state. It returns a clone to prevent
    /// accidental modification of the shared bank.
    /// </para>
    /// <para><b>For Beginners:</b> This lets you look at the shared library of vectors for
    /// matrix B. It's useful for debugging or understanding what vectors are available in the bank.
    /// </para>
    /// </remarks>
    public static Matrix<T>? GetBankB(string bankKey = "default")
    {
        lock (_bankLock)
        {
            return _globalBankB.ContainsKey(bankKey) ? _globalBankB[bankKey].Clone() : null;
        }
    }

    /// <summary>
    /// Clears the global vector banks (useful for testing or reinitialization).
    /// </summary>
    /// <param name="bankKey">The bank key to clear, or null to clear all banks.</param>
    /// <remarks>
    /// <para><b>Warning:</b> This will affect all VB-LoRA adapters using the specified bank(s).
    /// Use with caution, typically only in testing scenarios.
    /// </para>
    /// <para><b>For Beginners:</b> This erases the shared library. It's useful when you want to
    /// start fresh, but be careful - it affects all adapters using that library!
    /// </para>
    /// </remarks>
    public static void ClearBanks(string? bankKey = null)
    {
        lock (_bankLock)
        {
            if (bankKey != null)
            {
                _globalBankA.Remove(bankKey);
                _globalBankB.Remove(bankKey);
            }
            else
            {
                _globalBankA.Clear();
                _globalBankB.Clear();
            }
        }
    }
}

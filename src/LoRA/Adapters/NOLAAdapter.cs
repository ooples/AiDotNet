using AiDotNet.Attributes;
using AiDotNet.Helpers;
using System;
using AiDotNet.Interfaces;

namespace AiDotNet.LoRA.Adapters;

/// <summary>
/// Implements NOLA (Compressing LoRA using Linear Combination of Random Basis) adapter for extreme parameter efficiency.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// NOLA overcomes the rank-one lower bound in traditional LoRA by re-parameterizing the low-rank matrices
/// using linear combinations of randomly generated basis matrices. Instead of optimizing the full low-rank
/// matrices A and B, NOLA:
/// 1. Generates fixed random basis matrices using a deterministic seed
/// 2. Optimizes only scalar coefficients that linearly combine these basis matrices
/// 3. Regenerates basis matrices during forward/backward passes to minimize memory usage
/// </para>
/// <para>
/// This decouples the number of trainable parameters from both the choice of rank and the network architecture,
/// achieving compression ratios of 20x over standard LoRA without accuracy degradation.
/// </para>
/// <para><b>For Beginners:</b> NOLA is an extreme compression technique for LoRA that makes fine-tuning
/// even more efficient. Instead of storing and training two low-rank matrices (A and B), NOLA:
///
/// - Generates random "template" matrices on-the-fly (same random numbers every time due to fixed seed)
/// - Only trains small coefficients that control how much of each template to use
/// - Achieves 2-3x fewer parameters than LoRA while maintaining performance
///
/// Think of it like this:
/// - Traditional LoRA: You have 100 adjustable knobs (parameters)
/// - NOLA: You have 5 master controls that blend pre-defined settings
///
/// Key innovations:
/// 1. <b>Memory efficiency:</b> Random basis matrices are discarded after use and regenerated when needed
/// 2. <b>Parameter efficiency:</b> Only coefficients are trained, not full matrices
/// 3. <b>Performance:</b> Achieves similar or better results than LoRA with far fewer parameters
///
/// Example compression (1000x1000 layer, rank=8):
/// - LoRA: 16,000 parameters (1000×8 + 8×1000)
/// - NOLA with 100 basis: 200 parameters (100 coefficients for A + 100 for B) - 80x reduction!
///
/// On LLaMA-2 70B, NOLA achieves 20x compression over LoRA with no accuracy loss.
/// </para>
/// <para><b>Reference:</b> NOLA: Compressing LoRA using Linear Combination of Random Basis
/// (Koohpayegani et al., ICLR 2024) - https://arxiv.org/abs/2310.02556
/// </para>
/// </remarks>
[AutoParameters]
public partial class NOLAAdapter<T> : LoRAAdapterBase<T>
{
    /// <summary>
    /// Random number generator with fixed seed for reproducible basis generation.
    /// </summary>
    private readonly Random _basisGenerator;

    /// <summary>
    /// Number of random basis matrices to use for each low-rank matrix.
    /// </summary>
    private readonly int _numBasis;

    /// <summary>
    /// Trainable coefficients for matrix A basis combination (size: numBasis).
    /// </summary>
    [TrainableParameter]
    private Tensor<T> _coefficientsA;

    /// <summary>
    /// Trainable coefficients for matrix B basis combination (size: numBasis).
    /// </summary>
    [TrainableParameter]
    private Tensor<T> _coefficientsB;

    /// <summary>
    /// Gradients for coefficients A computed during backpropagation.
    /// </summary>
    [Scratch]
    private Vector<T>? _coefficientsAGradient;

    /// <summary>
    /// Gradients for coefficients B computed during backpropagation.
    /// </summary>
    [Scratch]
    private Vector<T>? _coefficientsBGradient;

    /// <summary>
    /// Seed for reproducible random basis generation.
    /// </summary>
    private readonly int _seed;

    /// <summary>
    /// Gets the number of basis matrices used for compression.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This determines the compression ratio. Fewer basis matrices = more compression but less flexibility.
    /// Typical values range from 10 to 100 depending on the task.
    /// </para>
    /// <para><b>For Beginners:</b> This is the number of "template" matrices we use. More templates
    /// give more flexibility but require more coefficients to train. It's the main knob for controlling
    /// the compression-accuracy trade-off in NOLA.
    /// </para>
    /// </remarks>
    public int NumBasis => _numBasis;

    /// <summary>
    /// Gets the compression ratio compared to standard LoRA.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Compression ratio = (LoRA parameters) / (NOLA parameters)
    /// Higher values indicate more extreme compression.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how much more efficient NOLA is compared to regular LoRA.
    /// For example, a compression ratio of 20 means NOLA uses 20 times fewer parameters!
    /// </para>
    /// </remarks>
    public double CompressionRatio
    {
        get
        {
            int inputSize = GetInputShape()[0];
            int outputSize = GetOutputLayerShape().RequireConcrete("Sizing a LoRA adapter's low-rank factors")[0];
            int loraParams = (inputSize * Rank) + (Rank * outputSize);
            int nolaParams = 2 * _numBasis;  // coefficients for A and B
            return (double)loraParams / nolaParams;
        }
    }

    /// <summary>
    /// Initializes a new NOLA adapter with the specified parameters.
    /// </summary>
    /// <param name="baseLayer">The layer to adapt with NOLA.</param>
    /// <param name="rank">The rank of the low-rank decomposition (determines basis matrix dimensions).</param>
    /// <param name="numBasis">Number of random basis matrices to use (controls compression ratio).</param>
    /// <param name="alpha">The LoRA scaling factor (defaults to rank if negative).</param>
    /// <param name="seed">Random seed for reproducible basis generation (default: 42).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer's parameters during training.</param>
    /// <exception cref="ArgumentNullException">Thrown when baseLayer is null.</exception>
    /// <exception cref="ArgumentException">Thrown when rank or numBasis are invalid.</exception>
    /// <remarks>
    /// <para>
    /// NOLA initialization:
    /// - Coefficients are initialized to zero (so NOLA starts with no effect, like LoRA)
    /// - Random basis matrices are generated on-demand during forward/backward passes
    /// - A fixed seed ensures reproducible basis generation across training
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new NOLA adapter. Important parameters:
    ///
    /// - baseLayer: The layer you want to make ultra-efficient to fine-tune
    /// - rank: Controls the "bottleneck" dimension (same as in LoRA)
    /// - numBasis: Controls compression (fewer = more compression, less flexibility)
    /// - seed: Ensures you get the same random "templates" every time
    ///
    /// Recommended values:
    /// - For extreme compression (20x): numBasis = rank / 2
    /// - For balanced compression (10x): numBasis = rank
    /// - For moderate compression (5x): numBasis = rank * 2
    ///
    /// Example: rank=8, numBasis=4 gives ~40x compression over full fine-tuning!
    /// </para>
    /// </remarks>
    public NOLAAdapter(
        ILayer<T> baseLayer,
        int rank,
        int numBasis,
        double alpha = -1,
        int seed = 42,
        bool freezeBaseLayer = true)
        : base(baseLayer, rank, alpha, freezeBaseLayer)
    {
        if (numBasis <= 0)
        {
            throw new ArgumentException("Number of basis matrices must be positive", nameof(numBasis));
        }

        _numBasis = numBasis;
        _seed = seed;
        _basisGenerator = RandomHelper.CreateSeededRandom(_seed);

        // Keep the initial adaptation zero while allowing coefficient gradients to start learning.
        _coefficientsA = new Tensor<T>([_numBasis]);
        // One nonzero A coefficient gives B a gradient while zero B preserves the initial base output.
        _coefficientsA[0] = NumOps.One;
        _coefficientsB = new Tensor<T>([_numBasis]); // fresh tensors are zero-filled


        // NOLA trains only its coefficient tensors; random bases are reproducibly derived from the
        // seed and the inherited standard LoRA factors are unused.
        FreezeSubLayerParameters(_loraLayer);
    }

    /// <summary>
    /// Generates a random basis matrix with the specified dimensions using the fixed seed.
    /// </summary>
    /// <param name="rows">Number of rows in the basis matrix.</param>
    /// <param name="cols">Number of columns in the basis matrix.</param>
    /// <param name="basisIndex">Index of the basis matrix (used to advance random state).</param>
    /// <returns>A random basis matrix with values in range [-1, 1].</returns>
    /// <remarks>
    /// <para>
    /// Basis matrices are generated using a uniform distribution in the range [-1, 1].
    /// The same seed ensures reproducibility across forward and backward passes.
    /// </para>
    /// <para><b>For Beginners:</b> This creates one of the random "template" matrices.
    /// By using a fixed seed, we always get the same template for a given index,
    /// which means we don't need to store them - we can regenerate them when needed!
    /// </para>
    /// </remarks>
    private Matrix<T> GenerateRandomBasis(int rows, int cols, int basisIndex)
    {
        // Reset random generator to get consistent basis for this index
        Random gen = RandomHelper.CreateSeededRandom(_seed + basisIndex);

        Matrix<T> basis = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                // Uniform distribution in [-1, 1]
                double value = gen.NextDouble() * 2.0 - 1.0;
                basis[i, j] = NumOps.FromDouble(value);
            }
        }
        return basis;
    }

    /// <summary>
    /// Reconstructs matrix A from linear combination of random basis matrices.
    /// </summary>
    /// <returns>Reconstructed matrix A (inputSize × rank).</returns>
    /// <remarks>
    /// <para>
    /// Computes: A = Σ(coefficient_i * basis_i) for all basis matrices.
    /// Each basis matrix is generated on-the-fly and discarded after use.
    /// </para>
    /// <para><b>For Beginners:</b> This creates the actual matrix A by blending all the
    /// random templates according to the learned coefficients. It's like mixing paint colors:
    /// each template is a color, and each coefficient controls how much of that color to use.
    /// </para>
    /// </remarks>
    private Tensor<T> CombineBases(int rows, int cols, int basisOffset, Tensor<T> coefficients)
    {
        Tensor<T> sum;
        // Random bases are constants: retaining each block on the forward tape defeats rematerialization.
        using (AiDotNet.Tensors.Helpers.TensorArena.Suspend())
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            int blockSize = BasisBlockSize(rows, cols);
            sum = new Tensor<T>([rows, cols]);
            for (int start = 0; start < _numBasis; start += blockSize)
            {
                int count = Math.Min(blockSize, _numBasis - start);
                var bases = GenerateBasisBlock(rows, cols, basisOffset + start, count);
                var weights = Engine.Reshape(Engine.TensorNarrow(coefficients, 0, start, count), [count, 1, 1]);
                sum = Engine.TensorAdd(sum, Engine.ReduceSum(Engine.TensorMultiply(bases, weights), [0], false));
            }
        }

        if (AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>.IsSuppressed) return sum;

        // The closure retains dimensions and the seed-bearing adapter, never generated basis tensors.
        return RegisterManualBackwardNode(sum, [coefficients],
            gradient => [BasisCoefficientGradient(gradient, rows, cols, basisOffset)]);
    }

    private static int BasisBlockSize(int rows, int cols)
        => (int)Math.Max(1, Math.Min(16, 65_536L / ((long)rows * cols)));

    private Tensor<T> GenerateBasisBlock(int rows, int cols, int basisOffset, int count)
    {
        var bases = new Tensor<T>[count];
        for (int b = 0; b < count; b++)
            bases[b] = Tensor<T>.FromMatrix(GenerateRandomBasis(rows, cols, basisOffset + b));
        return Engine.TensorStack(bases, 0);
    }

    private Tensor<T> BasisCoefficientGradient(Tensor<T> gradient, int rows, int cols, int basisOffset)
    {
        using var arenaSuspension = AiDotNet.Tensors.Helpers.TensorArena.Suspend();
        int blockSize = BasisBlockSize(rows, cols);
        var blocks = new List<Tensor<T>>();
        for (int start = 0; start < _numBasis; start += blockSize)
        {
            int count = Math.Min(blockSize, _numBasis - start);
            var bases = GenerateBasisBlock(rows, cols, basisOffset + start, count);
            // d(sum_i c_i basis_i)/dc_i is basis_i; contract with the incoming matrix gradient.
            var products = Engine.TensorMultiply(bases, Engine.Reshape(gradient, [1, rows, cols]));
            blocks.Add(Engine.ReduceSum(products, [1, 2], false));
        }
        return Engine.TensorConcatenate(blocks.ToArray(), 0);
    }


    private Tensor<T> ReconstructMatrixA()

    {
        int inputSize = GetInputShape()[0];
        // Linear combination of basis matrices: A = sum_b coefA[b] * basis_b
        Tensor<T> matrixA = CombineBases(inputSize, Rank, 0, _coefficientsA);

        return matrixA;
    }

    /// <summary>
    /// Reconstructs matrix B from linear combination of random basis matrices.
    /// </summary>
    /// <returns>Reconstructed matrix B (rank × outputSize).</returns>
    /// <remarks>
    /// <para>
    /// Computes: B = Σ(coefficient_i * basis_i) for all basis matrices.
    /// Each basis matrix is generated on-the-fly and discarded after use.
    /// </para>
    /// <para><b>For Beginners:</b> Same as ReconstructMatrixA, but for matrix B.
    /// Together, A and B form the complete NOLA adaptation.
    /// </para>
    /// </remarks>
    private Tensor<T> ReconstructMatrixB()
    {
        int outputSize = GetOutputLayerShape().RequireConcrete("Sizing a LoRA adapter's low-rank factors")[0];
        // Linear combination of basis matrices (offset by numBasis for B): B = sum_b coefB[b] * basis_b
        Tensor<T> matrixB = CombineBases(Rank, outputSize, _numBasis, _coefficientsB);

        return matrixB;
    }

    /// <summary>
    /// Performs the forward pass through both base and NOLA layers.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Sum of base layer output and NOLA output.</returns>
    /// <remarks>
    /// <para>
    /// The forward pass:
    /// 1. Reconstructs matrices A and B from coefficients and random basis
    /// 2. Computes NOLA output: input * A * B * scaling
    /// 3. Adds base layer output
    /// 4. Caches A and B for use in backward pass
    /// </para>
    /// <para><b>For Beginners:</b> This processes the input through both the original layer
    /// and the NOLA adaptation. The NOLA part:
    /// 1. Creates A and B matrices from the learned coefficients
    /// 2. Runs the input through A and B (compression then expansion)
    /// 3. Scales the result
    /// 4. Adds it to the base layer's output
    ///
    /// The result is the original behavior plus the ultra-compressed adaptation!
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        // Forward through base layer
        Tensor<T> baseOutput = _baseLayer.Forward(input);

        // Reconstruct NOLA matrices A and B
        Tensor<T> matrixA = ReconstructMatrixA();
        Tensor<T> matrixB = ReconstructMatrixB();

        // Compute NOLA contribution: input * A * B * scaling
        int batchSize = input.Shape[0];
        int inputSize = input.Shape.Length > 1 ? input.Shape[1] : input.Length;
        int outputSize = GetOutputLayerShape().RequireConcrete("Sizing a LoRA adapter's low-rank factors")[0];

        // Compute: input * A (result: [batchSize, rank]), then * B (result: [batchSize, outputSize])
        Tensor<T> intermediate = Engine.TensorMatMul(
            Engine.Reshape(input, [batchSize, inputSize]), matrixA);
        Tensor<T> nolaOutput = Engine.TensorMatMul(intermediate, matrixB);

        // Apply scaling (alpha / rank)
        T scaling = NumOps.Divide(
            NumOps.FromDouble(Alpha),
            NumOps.FromDouble(Rank));
        Tensor<T> nolaOutputTensor = Engine.TensorMultiplyScalar(nolaOutput, scaling);

        // Sum base and NOLA outputs
        Tensor<T> result = Engine.TensorAdd(baseOutput, nolaOutputTensor);

        return result;
    }


    /// <summary>
    /// Updates parameters using the specified learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    public override void UpdateParameters(T learningRate)
    {
        if (_coefficientsAGradient == null || _coefficientsBGradient == null)
        {
            return;
        }

        // Update coefficients for A and B: c -= learningRate * dc
        Engine.TensorSubtractInPlace(_coefficientsA, Tensor<T>.FromVector(Engine.Multiply(_coefficientsAGradient, learningRate)));
        Engine.TensorSubtractInPlace(_coefficientsB, Tensor<T>.FromVector(Engine.Multiply(_coefficientsBGradient, learningRate)));

        // Update base layer if not frozen
        if (!_freezeBaseLayer)
        {
            _baseLayer.UpdateParameters(learningRate);
        }

    }

    /// <summary>
    /// Merges the NOLA adaptation into the base layer and returns the merged layer.
    /// </summary>
    /// <returns>A new layer with NOLA weights merged into the base layer's weights.</returns>
    /// <remarks>
    /// <para>
    /// This reconstructs the full NOLA matrices A and B from coefficients, computes the
    /// merged weight matrix (A * B * scaling), and adds it to the base layer's weights.
    /// </para>
    /// <para><b>For Beginners:</b> This "bakes in" your NOLA adaptation to create a regular layer.
    /// It reconstructs the full A and B matrices from your learned coefficients and merges them
    /// into the base layer. The result is a standard layer with all adaptations built-in.
    /// </para>
    /// </remarks>
    public override ILayer<T> MergeToOriginalLayer()
    {
        // Reconstruct full matrices from coefficients
        Tensor<T> matrixA = ReconstructMatrixA();
        Tensor<T> matrixB = ReconstructMatrixB();

        // Compute merged weight matrix: A * B * scaling
        T scaling = NumOps.Divide(NumOps.FromDouble(Alpha), NumOps.FromDouble(Rank));
        Tensor<T> mergedWeight = Engine.TensorMultiplyScalar(Engine.TensorMatMul(matrixA, matrixB), scaling);

        // This requires knowledge of the base layer type to properly merge
        // For now, we'll throw an exception indicating this needs layer-specific implementation
        throw new NotSupportedException(
            "MergeToOriginalLayer requires layer-type-specific implementation. " +
            "Derived classes should override this method to handle their specific base layer type.");
    }

    /// <summary>
    /// Resets the internal state of the adapter.
    /// </summary>
    public override void ResetState()
    {
        base.ResetState();
        _coefficientsAGradient = null;
        _coefficientsBGradient = null;
    }

    /// <summary>
    /// Gets the current coefficient values for matrix A (for inspection).
    /// </summary>
    public Vector<T> GetCoefficientsA() => _coefficientsA.ToVector();

    /// <summary>
    /// Gets the current coefficient values for matrix B (for inspection).
    /// </summary>
    public Vector<T> GetCoefficientsB() => _coefficientsB.ToVector();
}

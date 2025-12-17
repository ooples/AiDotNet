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
public class NOLAAdapter<T> : LoRAAdapterBase<T>
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
    private Vector<T> _coefficientsA;

    /// <summary>
    /// Trainable coefficients for matrix B basis combination (size: numBasis).
    /// </summary>
    private Vector<T> _coefficientsB;

    /// <summary>
    /// Gradients for coefficients A computed during backpropagation.
    /// </summary>
    private Vector<T>? _coefficientsAGradient;

    /// <summary>
    /// Gradients for coefficients B computed during backpropagation.
    /// </summary>
    private Vector<T>? _coefficientsBGradient;

    /// <summary>
    /// Cached matrix A from last forward pass (used in backward pass).
    /// </summary>
    private Matrix<T>? _cachedMatrixA;

    /// <summary>
    /// Cached matrix B from last forward pass (used in backward pass).
    /// </summary>
    private Matrix<T>? _cachedMatrixB;

    /// <summary>
    /// Cached input from last forward pass (needed for gradient computation).
    /// </summary>
    private Tensor<T>? _lastInput;

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
            int outputSize = GetOutputShape()[0];
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

        // Initialize coefficients to zero (NOLA starts with no effect)
        _coefficientsA = new Vector<T>(_numBasis);
        _coefficientsB = new Vector<T>(_numBasis);
        for (int i = 0; i < _numBasis; i++)
        {
            _coefficientsA[i] = NumOps.Zero;
            _coefficientsB[i] = NumOps.Zero;
        }

        // Update parameter count to reflect NOLA compression
        // Parameters: coefficientsA + coefficientsB (+ base layer if not frozen)
        int nolaParams = 2 * _numBasis;
        Parameters = new Vector<T>(_freezeBaseLayer ? nolaParams : (_baseLayer.ParameterCount + nolaParams));
        UpdateParametersFromCoefficients();
    }

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    /// <remarks>
    /// For NOLA, this is just 2 * numBasis (coefficients for A and B), plus base layer parameters if not frozen.
    /// This is dramatically smaller than standard LoRA's (inputSize * rank) + (rank * outputSize).
    /// </remarks>
    public override int ParameterCount
    {
        get
        {
            // Guard against being called during base class construction before _numBasis is set
            if (_numBasis == 0 && _baseLayer != null)
            {
                return _freezeBaseLayer ? 0 : _baseLayer.ParameterCount;
            }

            int baseCount = (_baseLayer != null && !_freezeBaseLayer) ? _baseLayer.ParameterCount : 0;
            return baseCount + (2 * _numBasis);
        }
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
    private Matrix<T> ReconstructMatrixA()
    {
        int inputSize = GetInputShape()[0];
        Matrix<T> matrixA = new Matrix<T>(inputSize, Rank);

        // Linear combination of basis matrices
        for (int b = 0; b < _numBasis; b++)
        {
            Matrix<T> basis = GenerateRandomBasis(inputSize, Rank, b);
            T coef = _coefficientsA[b];

            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < Rank; j++)
                {
                    matrixA[i, j] = NumOps.Add(matrixA[i, j], NumOps.Multiply(basis[i, j], coef));
                }
            }
        }

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
    private Matrix<T> ReconstructMatrixB()
    {
        int outputSize = GetOutputShape()[0];
        Matrix<T> matrixB = new Matrix<T>(Rank, outputSize);

        // Linear combination of basis matrices
        for (int b = 0; b < _numBasis; b++)
        {
            Matrix<T> basis = GenerateRandomBasis(Rank, outputSize, _numBasis + b);  // Offset by numBasis for B
            T coef = _coefficientsB[b];

            for (int i = 0; i < Rank; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    matrixB[i, j] = NumOps.Add(matrixB[i, j], NumOps.Multiply(basis[i, j], coef));
                }
            }
        }

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
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Cache input for backward pass
        _lastInput = input.Clone();

        // Forward through base layer
        Tensor<T> baseOutput = _baseLayer.Forward(input);

        // Reconstruct NOLA matrices A and B
        _cachedMatrixA = ReconstructMatrixA();
        _cachedMatrixB = ReconstructMatrixB();

        // Compute NOLA contribution: input * A * B * scaling
        int batchSize = input.Shape[0];
        int inputSize = input.Shape.Length > 1 ? input.Shape[1] : input.Length;
        int outputSize = GetOutputShape()[0];

        // Convert input to matrix [batchSize, inputSize]
        Matrix<T> inputMatrix = new Matrix<T>(batchSize, inputSize);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                inputMatrix[i, j] = input[i * inputSize + j];
            }
        }

        // Compute: input * A (result: [batchSize, rank])
        Matrix<T> intermediate = inputMatrix.Multiply(_cachedMatrixA);

        // Compute: intermediate * B (result: [batchSize, outputSize])
        Matrix<T> nolaOutput = intermediate.Multiply(_cachedMatrixB);

        // Apply scaling (alpha / rank)
        T scaling = NumOps.Divide(
            NumOps.FromDouble(Alpha),
            NumOps.FromDouble(Rank));
        nolaOutput = nolaOutput.Multiply(scaling);

        // Convert to tensor
        Vector<T> nolaOutputData = new Vector<T>(batchSize * outputSize);
        int idx = 0;
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                nolaOutputData[idx++] = nolaOutput[i, j];
            }
        }
        Tensor<T> nolaOutputTensor = new Tensor<T>(new[] { batchSize, outputSize }, nolaOutputData);

        // Sum base and NOLA outputs
        Tensor<T> result = new Tensor<T>(baseOutput.Shape);
        for (int i = 0; i < baseOutput.Length; i++)
        {
            result[i] = NumOps.Add(baseOutput[i], nolaOutputTensor[i]);
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
    /// The backward pass:
    /// 1. Propagates gradients through base layer (if not frozen)
    /// 2. Computes coefficient gradients by regenerating basis matrices and computing inner products
    /// 3. Propagates input gradients through NOLA path
    /// 4. Sums input gradients from both paths
    /// </para>
    /// <para><b>For Beginners:</b> During learning, this figures out how to improve the coefficients:
    /// - For each basis matrix, we compute how much changing its coefficient would reduce error
    /// - We regenerate the same random templates (using the fixed seed) to compute gradients
    /// - We combine gradients from both the base layer and NOLA paths
    ///
    /// The magic is that we only need to update a few coefficients, not entire matrices!
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_cachedMatrixA == null || _cachedMatrixB == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass");
        }

        // Backward through base layer
        Tensor<T> baseInputGrad = _baseLayer.Backward(outputGradient);

        // Compute NOLA gradients
        int batchSize = outputGradient.Shape[0];
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        T scaling = NumOps.Divide(NumOps.FromDouble(Alpha), NumOps.FromDouble(Rank));

        // Convert gradient to matrix
        Matrix<T> gradMatrix = new Matrix<T>(batchSize, outputSize);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                gradMatrix[i, j] = outputGradient[i * outputSize + j];
            }
        }

        // Scale gradient
        gradMatrix = gradMatrix.Multiply(scaling);

        // Get input from cache
        if (_lastInput == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass");
        }

        Matrix<T> inputMatrix = new Matrix<T>(batchSize, inputSize);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                inputMatrix[i, j] = _lastInput[i * inputSize + j];
            }
        }

        // Compute intermediate: input * A
        Matrix<T> intermediate = inputMatrix.Multiply(_cachedMatrixA);

        // Compute coefficient gradients for B
        // dL/dc_b = sum over batch of: (input * A)^T * grad * basis_b
        _coefficientsBGradient = new Vector<T>(_numBasis);
        for (int b = 0; b < _numBasis; b++)
        {
            Matrix<T> basisB = GenerateRandomBasis(Rank, outputSize, _numBasis + b);

            // Compute: intermediate^T * grad * basisB
            Matrix<T> temp = intermediate.Transpose().Multiply(gradMatrix);
            T gradSum = NumOps.Zero;
            for (int i = 0; i < temp.Rows; i++)
            {
                for (int j = 0; j < temp.Columns; j++)
                {
                    gradSum = NumOps.Add(gradSum, NumOps.Multiply(temp[i, j], basisB[i, j]));
                }
            }
            _coefficientsBGradient[b] = gradSum;
        }

        // Compute coefficient gradients for A
        // dL/dc_a = sum over batch of: input^T * (grad * B^T) * basis_a
        Matrix<T> gradTimesB = gradMatrix.Multiply(_cachedMatrixB.Transpose());
        _coefficientsAGradient = new Vector<T>(_numBasis);
        for (int b = 0; b < _numBasis; b++)
        {
            Matrix<T> basisA = GenerateRandomBasis(inputSize, Rank, b);

            // Compute: input^T * gradTimesB * basisA
            Matrix<T> temp = inputMatrix.Transpose().Multiply(gradTimesB);
            T gradSum = NumOps.Zero;
            for (int i = 0; i < temp.Rows; i++)
            {
                for (int j = 0; j < temp.Columns; j++)
                {
                    gradSum = NumOps.Add(gradSum, NumOps.Multiply(temp[i, j], basisA[i, j]));
                }
            }
            _coefficientsAGradient[b] = gradSum;
        }

        // Compute input gradients: grad * B^T * A^T * scaling
        Matrix<T> nolaInputGrad = gradMatrix.Multiply(_cachedMatrixB.Transpose()).Multiply(_cachedMatrixA.Transpose());

        // Convert to tensor
        Vector<T> nolaInputGradData = new Vector<T>(batchSize * inputSize);
        int idx = 0;
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                nolaInputGradData[idx++] = nolaInputGrad[i, j];
            }
        }
        Tensor<T> nolaInputGradTensor = new Tensor<T>(new[] { batchSize, inputSize }, nolaInputGradData);

        // Sum input gradients
        Tensor<T> inputGrad = new Tensor<T>(baseInputGrad.Shape);
        for (int i = 0; i < baseInputGrad.Length; i++)
        {
            inputGrad[i] = NumOps.Add(baseInputGrad[i], nolaInputGradTensor[i]);
        }

        // Update parameter gradients vector
        UpdateParameterGradientsFromCoefficients();

        return inputGrad;
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

        // Update coefficients for A
        for (int i = 0; i < _numBasis; i++)
        {
            T update = NumOps.Multiply(_coefficientsAGradient[i], learningRate);
            _coefficientsA[i] = NumOps.Subtract(_coefficientsA[i], update);
        }

        // Update coefficients for B
        for (int i = 0; i < _numBasis; i++)
        {
            T update = NumOps.Multiply(_coefficientsBGradient[i], learningRate);
            _coefficientsB[i] = NumOps.Subtract(_coefficientsB[i], update);
        }

        // Update base layer if not frozen
        if (!_freezeBaseLayer)
        {
            _baseLayer.UpdateParameters(learningRate);
        }

        // Update parameter vector
        UpdateParametersFromCoefficients();
    }

    /// <summary>
    /// Updates the parameter vector from the current coefficient values.
    /// </summary>
    private void UpdateParametersFromCoefficients()
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

        // Pack coefficients A
        for (int i = 0; i < _numBasis; i++)
        {
            Parameters[idx++] = _coefficientsA[i];
        }

        // Pack coefficients B
        for (int i = 0; i < _numBasis; i++)
        {
            Parameters[idx++] = _coefficientsB[i];
        }
    }

    /// <summary>
    /// Updates coefficient values from the parameter vector.
    /// </summary>
    private void UpdateCoefficientsFromParameters()
    {
        int idx = 0;

        // Unpack base layer parameters if not frozen
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

        // Unpack coefficients A
        for (int i = 0; i < _numBasis; i++)
        {
            _coefficientsA[i] = Parameters[idx++];
        }

        // Unpack coefficients B
        for (int i = 0; i < _numBasis; i++)
        {
            _coefficientsB[i] = Parameters[idx++];
        }
    }

    /// <summary>
    /// Updates the parameter gradients vector from coefficient gradients.
    /// </summary>
    private void UpdateParameterGradientsFromCoefficients()
    {
        if (_coefficientsAGradient == null || _coefficientsBGradient == null)
        {
            return;
        }

        ParameterGradients = new Vector<T>(ParameterCount);
        int idx = 0;

        // Pack base layer gradients if not frozen
        if (!_freezeBaseLayer)
        {
            Vector<T> baseGrads = _baseLayer.GetParameterGradients();
            for (int i = 0; i < baseGrads.Length; i++)
            {
                ParameterGradients[idx++] = baseGrads[i];
            }
        }

        // Pack coefficient gradients A
        for (int i = 0; i < _numBasis; i++)
        {
            ParameterGradients[idx++] = _coefficientsAGradient[i];
        }

        // Pack coefficient gradients B
        for (int i = 0; i < _numBasis; i++)
        {
            ParameterGradients[idx++] = _coefficientsBGradient[i];
        }
    }

    /// <summary>
    /// Gets the current parameters as a vector.
    /// </summary>
    public override Vector<T> GetParameters()
    {
        return Parameters.Clone();
    }

    /// <summary>
    /// Sets the layer parameters from a vector.
    /// </summary>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException($"Expected {ParameterCount} parameters, got {parameters.Length}", nameof(parameters));
        }

        Parameters = parameters.Clone();
        UpdateCoefficientsFromParameters();
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
        Matrix<T> matrixA = ReconstructMatrixA();
        Matrix<T> matrixB = ReconstructMatrixB();

        // Compute merged weight matrix: A * B * scaling
        T scaling = NumOps.Divide(NumOps.FromDouble(Alpha), NumOps.FromDouble(Rank));
        Matrix<T> mergedWeight = matrixA.Multiply(matrixB).Multiply(scaling);

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
        _lastInput = null;
        _cachedMatrixA = null;
        _cachedMatrixB = null;
        _coefficientsAGradient = null;
        _coefficientsBGradient = null;
    }

    /// <summary>
    /// Gets the current coefficient values for matrix A (for inspection).
    /// </summary>
    public Vector<T> GetCoefficientsA() => _coefficientsA.Clone();

    /// <summary>
    /// Gets the current coefficient values for matrix B (for inspection).
    /// </summary>
    public Vector<T> GetCoefficientsB() => _coefficientsB.Clone();
}

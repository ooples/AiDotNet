using AiDotNet.Extensions;
using AiDotNet.Interfaces;

namespace AiDotNet.LoRA.Adapters;

/// <summary>
/// LoHa (Low-Rank Hadamard Product Adaptation) adapter for parameter-efficient fine-tuning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// LoHa uses element-wise Hadamard products (⊙) instead of matrix multiplication for adaptation.
/// Instead of computing ΔW = B * A like standard LoRA, LoHa computes:
/// ΔW = sum over rank of (A[i] ⊙ B[i])
///
/// This formulation can capture element-wise patterns that matrix multiplication may miss,
/// making it particularly effective for:
/// - Convolutional layers (local spatial patterns)
/// - Element-wise transformations
/// - Fine-grained weight adjustments
/// </para>
/// <para><b>Mathematical Formulation:</b>
///
/// Standard LoRA: ΔW = B * A where B is rank×output, A is input×rank
/// LoHa: ΔW = Σ(A[i] ⊙ B[i]) where A[i] and B[i] are both input×output
///
/// The Hadamard product (⊙) performs element-wise multiplication, allowing each element
/// of the weight matrix to be adjusted independently across the rank dimensions.
/// </para>
/// <para><b>For Beginners:</b> LoHa is a variant of LoRA that uses element-wise multiplication
/// instead of matrix multiplication. Think of it this way:
///
/// - Standard LoRA: Learns "row and column patterns" that combine via matrix multiply
/// - LoHa: Learns "pixel-by-pixel patterns" that combine via element-wise multiply
///
/// LoHa is especially good when:
/// 1. You need to capture local, element-wise patterns (like in images)
/// 2. The weight matrix has spatial structure (like convolutional filters)
/// 3. You want each weight to be adjusted somewhat independently
///
/// Trade-offs compared to LoRA:
/// - More parameters: Both A and B must be full-sized (input×output) per rank dimension
/// - Different expressiveness: Better for element-wise patterns, different from matrix patterns
/// - Better for CNNs: The element-wise nature matches convolutional structure better
///
/// Example: A 100×100 weight matrix with rank=8
/// - Standard LoRA: 8×100 + 100×8 = 1,600 parameters
/// - LoHa: 2 × 8 × 100 × 100 = 160,000 parameters (each rank has 2 full-sized matrices)
///
/// LoHa uses MORE parameters than LoRA but models element-wise weight interactions via Hadamard products.
/// </para>
/// </remarks>
public class LoHaAdapter<T> : LoRAAdapterBase<T>
{
    /// <summary>
    /// Low-rank matrices A with dimensions (rank, inputSize, outputSize).
    /// Each A[i] is a full-sized matrix for the i-th rank dimension.
    /// </summary>
    private readonly Matrix<T>[] _matricesA;

    /// <summary>
    /// Low-rank matrices B with dimensions (rank, inputSize, outputSize).
    /// Each B[i] is a full-sized matrix for the i-th rank dimension.
    /// </summary>
    private readonly Matrix<T>[] _matricesB;

    /// <summary>
    /// Gradients for matrices A computed during backpropagation.
    /// </summary>
    private Matrix<T>[]? _matricesAGradient;

    /// <summary>
    /// Gradients for matrices B computed during backpropagation.
    /// </summary>
    private Matrix<T>[]? _matricesBGradient;

    /// <summary>
    /// Stored input from the forward pass, needed for gradient computation.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stored base layer output from the forward pass.
    /// </summary>
    private Tensor<T>? _lastBaseOutput;

    /// <summary>
    /// Computed scaling factor (alpha / rank) used during forward pass.
    /// </summary>
    private readonly T _scaling;

    /// <summary>
    /// Initializes a new LoHa adapter wrapping an existing layer.
    /// </summary>
    /// <param name="baseLayer">The layer to adapt with LoHa.</param>
    /// <param name="rank">The rank of the low-rank decomposition.</param>
    /// <param name="alpha">The LoHa scaling factor (defaults to rank if negative).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer's parameters during training.</param>
    /// <exception cref="ArgumentNullException">Thrown when baseLayer is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the base layer doesn't have 1D input/output shapes.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a LoHa adapter for any layer with 1D input/output.
    ///
    /// Parameters:
    /// - baseLayer: The layer you want to make more efficient to fine-tune
    /// - rank: How many element-wise patterns to learn (more = more flexibility, more parameters)
    /// - alpha: How strong the LoHa adaptation is (typically same as rank)
    /// - freezeBaseLayer: Whether to lock the original layer's weights (usually true for efficiency)
    ///
    /// The adapter creates 2×rank full-sized matrices (A and B for each rank dimension),
    /// which are combined using element-wise Hadamard products during forward/backward passes.
    /// </para>
    /// </remarks>
    public LoHaAdapter(ILayer<T> baseLayer, int rank, double alpha = -1, bool freezeBaseLayer = true)
        : base(baseLayer, rank, alpha, freezeBaseLayer)
    {
        // Validate base layer has single-dimensional input/output
        if (baseLayer.GetInputShape().Length != 1 || baseLayer.GetOutputShape().Length != 1)
        {
            throw new ArgumentException("LoHaAdapter only supports layers with 1D input/output shapes", nameof(baseLayer));
        }

        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];

        // Calculate scaling
        _scaling = NumOps.Divide(_loraLayer.Alpha, NumOps.FromDouble(rank));

        // Initialize LoHa matrices (rank sets of full-sized matrices)
        _matricesA = new Matrix<T>[rank];
        _matricesB = new Matrix<T>[rank];

        for (int r = 0; r < rank; r++)
        {
            // Initialize A[r] with random values (Gaussian with std = 1/sqrt(rank))
            _matricesA[r] = new Matrix<T>(inputSize, outputSize);
            T stddev = NumOps.Sqrt(NumOps.Divide(NumOps.One, NumOps.FromDouble(rank)));
            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    _matricesA[r][i, j] = NumOps.Multiply(NumOps.FromDouble(Random.NextGaussian()), stddev);
                }
            }

            // Initialize B[r] to zero (so LoHa has no effect initially)
            _matricesB[r] = new Matrix<T>(inputSize, outputSize);
            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    _matricesB[r][i, j] = NumOps.Zero;
                }
            }
        }

        // Initialize parameter vector
        Parameters = new Vector<T>(ParameterCount);
        UpdateParametersFromMatrices();
    }

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    /// <remarks>
    /// LoHa has 2 * rank * inputSize * outputSize parameters (A and B matrices for each rank).
    /// This is more than standard LoRA but still far less than full fine-tuning.
    /// </remarks>
    public override int ParameterCount
    {
        get
        {
            int inputSize = GetInputShape()[0];
            int outputSize = GetOutputShape()[0];
            int lohaParams = 2 * Rank * inputSize * outputSize;
            return _freezeBaseLayer ? lohaParams : (_baseLayer.ParameterCount + lohaParams);
        }
    }

    /// <summary>
    /// Performs the forward pass through both base layer and LoHa adaptation.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Sum of base layer output and LoHa delta (computed via Hadamard products).</returns>
    /// <remarks>
    /// <para>
    /// The forward pass computes:
    /// 1. base_output = base_layer(input)
    /// 2. loha_delta = sum over rank of (input * A[i] ⊙ B[i]) * scaling
    /// 3. output = base_output + loha_delta
    ///
    /// The Hadamard product (⊙) multiplies corresponding elements, allowing element-wise adaptations.
    /// </para>
    /// <para><b>For Beginners:</b> This runs the input through the original layer and adds a correction.
    ///
    /// The correction is computed by:
    /// 1. Transforming input through each A[i] matrix (one per rank dimension)
    /// 2. Multiplying element-wise with corresponding B[i] matrix (Hadamard product)
    /// 3. Summing all rank contributions together
    /// 4. Scaling by alpha/rank
    ///
    /// This element-wise approach lets LoHa learn fine-grained adjustments to each weight independently.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input.Clone();

        // Forward through base layer
        Tensor<T> baseOutput = _baseLayer.Forward(input);
        _lastBaseOutput = baseOutput.Clone();

        // Compute LoHa delta using Hadamard products
        Tensor<T> lohaDelta = ComputeLoHaDelta(input);

        // Sum the outputs: base + loha_delta
        Tensor<T> result = new Tensor<T>(baseOutput.Shape);
        for (int i = 0; i < baseOutput.Length; i++)
        {
            result[i] = NumOps.Add(baseOutput[i], lohaDelta[i]);
        }

        return result;
    }

    /// <summary>
    /// Computes the LoHa delta using Hadamard products across all rank dimensions.
    /// </summary>
    /// <param name="input">Input tensor of shape [batchSize, inputSize].</param>
    /// <returns>LoHa delta tensor of shape [batchSize, outputSize].</returns>
    /// <remarks>
    /// <para>
    /// Computes: delta = scaling * sum over rank of (input * A[i]) ⊙ B[i]
    ///
    /// For each rank dimension i:
    /// 1. Multiply input by A[i] matrix: intermediate[i] = input * A[i]
    /// 2. Apply Hadamard product with B[i]: result[i] = intermediate[i] ⊙ B[i]
    /// 3. Sum all results and scale: delta = scaling * sum(result[i])
    /// </para>
    /// </remarks>
    private Tensor<T> ComputeLoHaDelta(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int inputSize = input.Shape.Length > 1 ? input.Shape[1] : input.Length;
        int outputSize = GetOutputShape()[0];

        // Convert input to matrix [batchSize, inputSize]
        Matrix<T> inputMatrix = new Matrix<T>(batchSize, inputSize);
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < inputSize; i++)
            {
                inputMatrix[b, i] = input[b * inputSize + i];
            }
        }

        // First compute ΔW = Σ_r (A[r] ⊙ B[r]) in weight space
        Matrix<T> deltaWeights = new Matrix<T>(inputSize, outputSize);
        for (int i = 0; i < inputSize; i++)
        {
            for (int o = 0; o < outputSize; o++)
            {
                deltaWeights[i, o] = NumOps.Zero;
            }
        }

        // Sum over rank: ΔW += A[r] ⊙ B[r] for each r
        for (int r = 0; r < Rank; r++)
        {
            // Hadamard product: A[r] ⊙ B[r] (element-wise multiplication)
            for (int i = 0; i < inputSize; i++)
            {
                for (int o = 0; o < outputSize; o++)
                {
                    T hadamard = NumOps.Multiply(_matricesA[r][i, o], _matricesB[r][i, o]);
                    deltaWeights[i, o] = NumOps.Add(deltaWeights[i, o], hadamard);
                }
            }
        }

        // Now apply ΔW to input: output = input × ΔW
        Matrix<T> deltaMatrix = new Matrix<T>(batchSize, outputSize);
        for (int b = 0; b < batchSize; b++)
        {
            for (int o = 0; o < outputSize; o++)
            {
                T sum = NumOps.Zero;
                for (int i = 0; i < inputSize; i++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(inputMatrix[b, i], deltaWeights[i, o]));
                }
                deltaMatrix[b, o] = sum;
            }
        }

        // Apply scaling
        for (int b = 0; b < batchSize; b++)
        {
            for (int o = 0; o < outputSize; o++)
            {
                deltaMatrix[b, o] = NumOps.Multiply(deltaMatrix[b, o], _scaling);
            }
        }

        // Convert back to tensor
        Vector<T> deltaData = new Vector<T>(batchSize * outputSize);
        int idx = 0;
        for (int b = 0; b < batchSize; b++)
        {
            for (int o = 0; o < outputSize; o++)
            {
                deltaData[idx++] = deltaMatrix[b, o];
            }
        }

        return new Tensor<T>(new[] { batchSize, outputSize }, deltaData);
    }


    /// <summary>
    /// Performs the backward pass through both layers, computing gradients for LoHa matrices.
    /// </summary>
    /// <param name="outputGradient">Gradient flowing back from the next layer.</param>
    /// <returns>Gradient to pass to the previous layer.</returns>
    /// <remarks>
    /// <para>
    /// The backward pass computes gradients using the chain rule for Hadamard products:
    ///
    /// dL/dA[r] = input^T * (dL/doutput ⊙ B[r]) * scaling
    /// dL/dB[r] = (input * A[r]) ⊙ dL/doutput * scaling
    /// dL/dinput = base_gradient + sum over rank of (dL/doutput ⊙ B[r]) * A[r]^T * scaling
    ///
    /// The Hadamard product gradient rule: d/dx (f ⊙ g) = df ⊙ g + f ⊙ dg
    /// </para>
    /// <para><b>For Beginners:</b> This is the learning phase for LoHa. It computes:
    ///
    /// 1. How to adjust each A[i] matrix to reduce error
    /// 2. How to adjust each B[i] matrix to reduce error
    /// 3. What gradient to send to earlier layers
    ///
    /// The math is more complex than standard LoRA because Hadamard products have different
    /// derivative rules than matrix multiplication, but the idea is the same: figure out
    /// how each parameter contributed to the error and adjust accordingly.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastBaseOutput == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass");
        }

        // Backward through base layer
        Tensor<T> baseInputGrad = _baseLayer.Backward(outputGradient);

        // Compute LoHa gradients
        Tensor<T> lohaInputGrad = ComputeLoHaGradients(outputGradient);

        // Sum input gradients
        Tensor<T> inputGrad = new Tensor<T>(lohaInputGrad.Shape);
        for (int i = 0; i < lohaInputGrad.Length; i++)
        {
            inputGrad[i] = NumOps.Add(lohaInputGrad[i], baseInputGrad[i]);
        }

        // Update parameter gradients vector
        UpdateParameterGradientsFromMatrices();

        return inputGrad;
    }

    /// <summary>
    /// Computes gradients for LoHa matrices A and B using Hadamard product gradient rules.
    /// </summary>
    /// <param name="outputGradient">Gradient flowing back from next layer.</param>
    /// <returns>Input gradient from LoHa path.</returns>
    private Tensor<T> ComputeLoHaGradients(Tensor<T> outputGradient)
    {
        int batchSize = _lastInput!.Shape[0];
        int inputSize = _lastInput!.Shape.Length > 1 ? _lastInput.Shape[1] : _lastInput.Length;
        int outputSize = GetOutputShape()[0];

        // Convert to matrices
        Matrix<T> inputMatrix = new Matrix<T>(batchSize, inputSize);
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < inputSize; i++)
            {
                inputMatrix[b, i] = _lastInput[b * inputSize + i];
            }
        }

        Matrix<T> gradMatrix = new Matrix<T>(batchSize, outputSize);
        for (int b = 0; b < batchSize; b++)
        {
            for (int o = 0; o < outputSize; o++)
            {
                gradMatrix[b, o] = outputGradient[b * outputSize + o];
            }
        }

        // Initialize gradients
        _matricesAGradient = new Matrix<T>[Rank];
        _matricesBGradient = new Matrix<T>[Rank];
        for (int r = 0; r < Rank; r++)
        {
            _matricesAGradient[r] = new Matrix<T>(inputSize, outputSize);
            _matricesBGradient[r] = new Matrix<T>(inputSize, outputSize);
        }

        // Accumulate input gradients
        Matrix<T> inputGradMatrix = new Matrix<T>(batchSize, inputSize);

        // For each rank dimension, compute gradients
        for (int r = 0; r < Rank; r++)
        {
            // Compute intermediate = input * A[r]
            Matrix<T> intermediate = new Matrix<T>(batchSize, outputSize);
            for (int b = 0; b < batchSize; b++)
            {
                for (int o = 0; o < outputSize; o++)
                {
                    T sum = NumOps.Zero;
                    for (int i = 0; i < inputSize; i++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(inputMatrix[b, i], _matricesA[r][i, o]));
                    }
                    intermediate[b, o] = sum;
                }
            }

            // Gradient for B[r]: Chain rule for Hadamard product in weight space
            // dL/dB[r][i,o] = sum_batch(dL/dy[b,o] * dy/dB[r][i,o])
            // where dy[b,o]/dB[r][i,o] = input[b,i] * A[r][i,o]
            // Therefore: dL/dB[r][i,o] = sum_batch(gradOutput[b,o] * input[b,i] * A[r][i,o])
            for (int i = 0; i < inputSize; i++)
            {
                for (int o = 0; o < outputSize; o++)
                {
                    T gradSum = NumOps.Zero;
                    for (int b = 0; b < batchSize; b++)
                    {
                        T inputVal = inputMatrix[b, i];
                        T aVal = _matricesA[r][i, o];
                        T outputGrad = gradMatrix[b, o];
                        // dL/dB = gradOutput * input * A (all element-wise for this specific element)
                        T contribution = NumOps.Multiply(NumOps.Multiply(outputGrad, inputVal), aVal);
                        gradSum = NumOps.Add(gradSum, contribution);
                    }
                    _matricesBGradient[r][i, o] = NumOps.Multiply(gradSum, _scaling);
                }
            }

            // Gradient for A[r]: Chain rule for Hadamard product in weight space
            // dL/dA[r][i,o] = sum_batch(dL/dy[b,o] * dy/dA[r][i,o])
            // where dy[b,o]/dA[r][i,o] = input[b,i] * B[r][i,o]
            // Therefore: dL/dA[r][i,o] = sum_batch(gradOutput[b,o] * input[b,i] * B[r][i,o])
            for (int i = 0; i < inputSize; i++)
            {
                for (int o = 0; o < outputSize; o++)
                {
                    T gradSum = NumOps.Zero;
                    for (int b = 0; b < batchSize; b++)
                    {
                        T inputVal = inputMatrix[b, i];
                        T bVal = _matricesB[r][i, o];
                        T outputGrad = gradMatrix[b, o];
                        // dL/dA = gradOutput * input * B (all element-wise for this specific element)
                        T contribution = NumOps.Multiply(NumOps.Multiply(outputGrad, inputVal), bVal);
                        gradSum = NumOps.Add(gradSum, contribution);
                    }
                    _matricesAGradient[r][i, o] = NumOps.Multiply(gradSum, _scaling);
                }
            }

            // Input gradient contribution from this rank
            // dL/dinput[b,i] = sum_o(dL/dy[b,o] * dy/dinput[b,i])
            // where dy[b,o]/dinput[b,i] = sum_r(A[r][i,o] * B[r][i,o]) = ΔW[i,o]
            // Therefore: dL/dinput[b,i] = sum_o(gradOutput[b,o] * ΔW[i,o])
            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < inputSize; i++)
                {
                    T gradSum = NumOps.Zero;
                    for (int o = 0; o < outputSize; o++)
                    {
                        // For this specific rank r, contribution is gradOutput * (A[r] ⊙ B[r])
                        T hadamardProduct = NumOps.Multiply(_matricesA[r][i, o], _matricesB[r][i, o]);
                        T contribution = NumOps.Multiply(gradMatrix[b, o], hadamardProduct);
                        gradSum = NumOps.Add(gradSum, contribution);
                    }
                    T scaled = NumOps.Multiply(gradSum, _scaling);
                    inputGradMatrix[b, i] = NumOps.Add(inputGradMatrix[b, i], scaled);
                }
            }
        }

        // Convert input gradient back to tensor
        Vector<T> inputGradData = new Vector<T>(batchSize * inputSize);
        int idx = 0;
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < inputSize; i++)
            {
                inputGradData[idx++] = inputGradMatrix[b, i];
            }
        }

        return new Tensor<T>(new[] { batchSize, inputSize }, inputGradData);
    }


    /// <summary>
    /// Updates parameters using the specified learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    public override void UpdateParameters(T learningRate)
    {
        if (_matricesAGradient == null || _matricesBGradient == null)
        {
            return;
        }

        // Update all A and B matrices
        for (int r = 0; r < Rank; r++)
        {
            // Update A[r]
            for (int i = 0; i < _matricesA[r].Rows; i++)
            {
                for (int j = 0; j < _matricesA[r].Columns; j++)
                {
                    T update = NumOps.Multiply(_matricesAGradient[r][i, j], learningRate);
                    _matricesA[r][i, j] = NumOps.Subtract(_matricesA[r][i, j], update);
                }
            }

            // Update B[r]
            for (int i = 0; i < _matricesB[r].Rows; i++)
            {
                for (int j = 0; j < _matricesB[r].Columns; j++)
                {
                    T update = NumOps.Multiply(_matricesBGradient[r][i, j], learningRate);
                    _matricesB[r][i, j] = NumOps.Subtract(_matricesB[r][i, j], update);
                }
            }
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
    /// Gets the current parameters as a vector.
    /// </summary>
    /// <returns>Vector containing all LoHa parameters (A and B matrices for all ranks).</returns>
    public override Vector<T> GetParameters()
    {
        return Parameters.Clone();
    }

    /// <summary>
    /// Sets the layer parameters from a vector.
    /// </summary>
    /// <param name="parameters">Vector containing all LoHa parameters.</param>
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
    /// Updates the parameter vector from the current matrix values.
    /// </summary>
    private void UpdateParametersFromMatrices()
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

        // Pack all A matrices
        for (int r = 0; r < Rank; r++)
        {
            for (int i = 0; i < _matricesA[r].Rows; i++)
            {
                for (int j = 0; j < _matricesA[r].Columns; j++)
                {
                    Parameters[idx++] = _matricesA[r][i, j];
                }
            }
        }

        // Pack all B matrices
        for (int r = 0; r < Rank; r++)
        {
            for (int i = 0; i < _matricesB[r].Rows; i++)
            {
                for (int j = 0; j < _matricesB[r].Columns; j++)
                {
                    Parameters[idx++] = _matricesB[r][i, j];
                }
            }
        }
    }

    /// <summary>
    /// Updates the matrices from the parameter vector.
    /// </summary>
    private void UpdateMatricesFromParameters()
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

        // Unpack all A matrices
        for (int r = 0; r < Rank; r++)
        {
            for (int i = 0; i < _matricesA[r].Rows; i++)
            {
                for (int j = 0; j < _matricesA[r].Columns; j++)
                {
                    _matricesA[r][i, j] = Parameters[idx++];
                }
            }
        }

        // Unpack all B matrices
        for (int r = 0; r < Rank; r++)
        {
            for (int i = 0; i < _matricesB[r].Rows; i++)
            {
                for (int j = 0; j < _matricesB[r].Columns; j++)
                {
                    _matricesB[r][i, j] = Parameters[idx++];
                }
            }
        }
    }

    /// <summary>
    /// Updates the parameter gradients vector from the matrix gradients.
    /// </summary>
    private void UpdateParameterGradientsFromMatrices()
    {
        if (_matricesAGradient == null || _matricesBGradient == null)
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

        // Pack all A matrix gradients
        for (int r = 0; r < Rank; r++)
        {
            for (int i = 0; i < _matricesAGradient[r].Rows; i++)
            {
                for (int j = 0; j < _matricesAGradient[r].Columns; j++)
                {
                    ParameterGradients[idx++] = _matricesAGradient[r][i, j];
                }
            }
        }

        // Pack all B matrix gradients
        for (int r = 0; r < Rank; r++)
        {
            for (int i = 0; i < _matricesBGradient[r].Rows; i++)
            {
                for (int j = 0; j < _matricesBGradient[r].Columns; j++)
                {
                    ParameterGradients[idx++] = _matricesBGradient[r][i, j];
                }
            }
        }
    }

    /// <summary>
    /// Merges the LoHa adaptation into the base layer and returns the merged layer.
    /// </summary>
    /// <returns>A new DenseLayer with LoHa weights merged into the base layer's weights.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the base layer type is not DenseLayer or FullyConnectedLayer.</exception>
    /// <remarks>
    /// <para>
    /// This method computes the full LoHa weight delta by summing all Hadamard products:
    /// ΔW = scaling * sum over rank of (A[i] ⊙ B[i])
    ///
    /// The delta is then added to the base layer's weights to create a merged layer.
    /// </para>
    /// <para><b>For Beginners:</b> This "bakes in" your LoHa adaptation to create a regular Dense layer.
    ///
    /// The merging process:
    /// 1. Computes the full weight delta from all A[i] and B[i] matrices using Hadamard products
    /// 2. Adds this delta to the base layer's existing weights
    /// 3. Copies biases unchanged (LoHa doesn't modify biases)
    /// 4. Creates a new DenseLayer with the merged weights
    ///
    /// After merging, you have a single layer that includes all the learned adaptations,
    /// making inference faster and simpler.
    /// </para>
    /// </remarks>
    public override ILayer<T> MergeToOriginalLayer()
    {
        // Support both DenseLayer and FullyConnectedLayer
        DenseLayer<T>? denseBase = _baseLayer as DenseLayer<T>;
        FullyConnectedLayer<T>? fcBase = _baseLayer as FullyConnectedLayer<T>;

        if (denseBase == null && fcBase == null)
        {
            throw new InvalidOperationException("LoHaAdapter only supports DenseLayer or FullyConnectedLayer base layers");
        }

        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];

        // Compute LoHa weight delta: sum over rank of (A[r] ⊙ B[r]) * scaling
        Matrix<T> lohaDelta = new Matrix<T>(inputSize, outputSize);
        for (int i = 0; i < inputSize; i++)
        {
            for (int o = 0; o < outputSize; o++)
            {
                lohaDelta[i, o] = NumOps.Zero;
            }
        }

        for (int r = 0; r < Rank; r++)
        {
            for (int i = 0; i < inputSize; i++)
            {
                for (int o = 0; o < outputSize; o++)
                {
                    // Hadamard product: A[r][i,o] * B[r][i,o]
                    T hadamard = NumOps.Multiply(_matricesA[r][i, o], _matricesB[r][i, o]);
                    lohaDelta[i, o] = NumOps.Add(lohaDelta[i, o], hadamard);
                }
            }
        }

        // Apply scaling
        for (int i = 0; i < inputSize; i++)
        {
            for (int o = 0; o < outputSize; o++)
            {
                lohaDelta[i, o] = NumOps.Multiply(lohaDelta[i, o], _scaling);
            }
        }

        // Get base layer parameters
        Vector<T> baseParams = _baseLayer.GetParameters();
        int weightCount = inputSize * outputSize;

        // Create new parameters with merged weights
        Vector<T> mergedParams = new Vector<T>(baseParams.Length);

        // Merge weights (base layer stores weights in row-major order: [output, input])
        for (int i = 0; i < weightCount; i++)
        {
            int row = i / inputSize;  // output index
            int col = i % inputSize;   // input index
            // lohaDelta is [input, output], so we transpose the indices
            mergedParams[i] = NumOps.Add(baseParams[i], lohaDelta[col, row]);
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
    /// Resets the internal state of both the base layer and LoHa adapter.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This clears the memory of the adapter and base layer.
    /// It's useful when starting to process a completely new, unrelated batch of data.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        _baseLayer.ResetState();
        _lastInput = null;
        _lastBaseOutput = null;
        _matricesAGradient = null;
        _matricesBGradient = null;
    }
}

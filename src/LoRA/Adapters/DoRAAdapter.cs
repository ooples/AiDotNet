using AiDotNet.Interfaces;

namespace AiDotNet.LoRA.Adapters;

/// <summary>
/// DoRA (Weight-Decomposed Low-Rank Adaptation) adapter for parameter-efficient fine-tuning with improved stability.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// DoRA (Weight-Decomposed LoRA) extends standard LoRA by decomposing pre-trained weights into
/// magnitude and direction components, then applying LoRA only to the direction component.
/// This decomposition leads to more stable training and better convergence compared to standard LoRA.
/// </para>
/// <para>
/// <b>Mathematical Formulation:</b>
/// Given pre-trained weights W, DoRA decomposes them as:
/// - W = m * d, where m is magnitude (scalar per neuron) and d is direction (unit vector)
/// - W' = m * normalize(d + LoRA_delta)
/// - LoRA_delta = (alpha/rank) * B * A
///
/// This ensures that LoRA adaptations primarily affect the direction of weights, not their magnitude,
/// which improves training stability and convergence.
/// </para>
/// <para>
/// <b>Research Context:</b>
/// DoRA was published in February 2024 and presented as an ICML 2024 Oral paper.
/// In experiments on LLaMA-7B, DoRA achieved +3.7% improvement over standard LoRA.
/// The key insight is that separating magnitude and direction allows more stable gradient flow
/// and better control over the adaptation process.
/// </para>
/// <para>
/// <b>For Beginners:</b> DoRA is an improved version of LoRA that works better in practice.
///
/// Think of neural network weights as arrows:
/// - Each arrow has a length (magnitude) and a direction
/// - Standard LoRA adjusts both length and direction at the same time
/// - DoRA separates them: it keeps the length fixed and only adjusts the direction
/// - This makes training more stable and gives better results
///
/// Why this matters:
/// - More stable training (fewer divergences and NaN errors)
/// - Better final performance (+3.7% on LLaMA-7B)
/// - Same parameter efficiency as standard LoRA
/// - Slightly more computation (due to normalization), but worth it for the stability
///
/// When to use DoRA over standard LoRA:
/// - When training stability is important (large models, complex tasks)
/// - When you want the best possible fine-tuning results
/// - When you have the computational budget for normalization overhead
/// - When adapting very large pre-trained models (LLMs, large vision models)
/// </para>
/// <para>
/// <b>Reference:</b>
/// "DoRA: Weight-Decomposed Low-Rank Adaptation"
/// ICML 2024 Oral
/// https://arxiv.org/abs/2402.09353
/// </para>
/// </remarks>
public class DoRAAdapter<T> : LoRAAdapterBase<T>
{
    /// <summary>
    /// Magnitude component of the decomposed weights (scalar per output neuron).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The magnitude vector stores the L2 norm of each weight vector (one per output neuron).
    /// During forward pass, this magnitude is applied after normalizing the direction vectors.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This stores the "strength" of each output neuron.
    /// When we decompose weights into magnitude and direction, this is the magnitude part.
    /// Each output neuron gets one magnitude value.
    /// </para>
    /// </remarks>
    private Vector<T> _magnitude;

    /// <summary>
    /// Gradients for the magnitude component, computed during backpropagation.
    /// </summary>
    private Vector<T>? _magnitudeGradient;

    /// <summary>
    /// Cached normalized direction from the last forward pass, used in backpropagation.
    /// </summary>
    private Matrix<T>? _lastNormalizedDirection;

    /// <summary>
    /// Cached input matrix from forward pass (used for computing magnitude gradients in backward).
    /// </summary>
    private Matrix<T>? _lastInputMatrix;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// DoRA adds the magnitude parameters (one per output neuron) to the standard LoRA parameters.
    /// Total = (base layer parameters if not frozen) + LoRA parameters + magnitude parameters.
    /// </para>
    /// </remarks>
    public override int ParameterCount
    {
        get
        {
            int baseCount = (_baseLayer != null && !_freezeBaseLayer) ? _baseLayer.ParameterCount : 0;
            int loraCount = _loraLayer != null ? _loraLayer.ParameterCount : 0;
            int magnitudeCount = _magnitude != null ? _magnitude.Length : 0;
            return baseCount + loraCount + magnitudeCount;
        }
    }

    /// <summary>
    /// Initializes a new DoRA adapter wrapping an existing layer.
    /// </summary>
    /// <param name="baseLayer">The layer to adapt with DoRA.</param>
    /// <param name="rank">The rank of the LoRA decomposition.</param>
    /// <param name="alpha">The LoRA scaling factor (defaults to rank if negative).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer's parameters during training.</param>
    /// <exception cref="ArgumentNullException">Thrown when baseLayer is null.</exception>
    /// <remarks>
    /// <para>
    /// The constructor initializes the DoRA adapter by:
    /// 1. Setting up the standard LoRA components (via base constructor)
    /// 2. Decomposing the base layer's initial weights into magnitude and direction
    /// 3. Initializing magnitude gradients
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This creates a DoRA adapter around your existing layer.
    ///
    /// What happens during initialization:
    /// - The base class sets up standard LoRA (matrices A and B)
    /// - We then decompose the layer's weights into magnitude and direction
    /// - The magnitude starts as the actual magnitudes from the original weights
    /// - During training, both the LoRA matrices and the magnitudes will be updated
    ///
    /// Parameters:
    /// - baseLayer: The layer you want to fine-tune efficiently
    /// - rank: How much compression for LoRA (lower = fewer parameters)
    /// - alpha: Scaling factor for LoRA contribution
    /// - freezeBaseLayer: Usually true - we only train LoRA + magnitude, not base weights
    /// </para>
    /// </remarks>
    public DoRAAdapter(ILayer<T> baseLayer, int rank, double alpha = -1, bool freezeBaseLayer = true)
        : base(baseLayer, rank, alpha, freezeBaseLayer)
    {
        // Initialize magnitude from base layer weights
        int outputSize = GetOutputShape()[0];
        _magnitude = new Vector<T>(outputSize);

        // Decompose initial weights to get magnitude
        DecomposeWeights();

        // Update parameters to include magnitude
        Parameters = new Vector<T>(ParameterCount);
        UpdateParametersFromComponents();
    }

    /// <summary>
    /// Decomposes the base layer's weights into magnitude and direction components.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For each output neuron, this method:
    /// 1. Extracts the weight vector (all connections to that neuron)
    /// 2. Computes the L2 norm (magnitude)
    /// 3. Stores the magnitude
    ///
    /// The direction is implicitly W/||W|| and doesn't need to be stored separately.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This splits weights into magnitude (length) and direction.
    ///
    /// Imagine each weight vector as an arrow:
    /// - Magnitude = how long the arrow is
    /// - Direction = which way the arrow points
    ///
    /// We store the magnitude separately so we can apply LoRA only to the direction.
    /// This is the key innovation of DoRA over standard LoRA.
    /// </para>
    /// </remarks>
    private void DecomposeWeights()
    {
        // Get base layer parameters
        Vector<T> baseParams = _baseLayer.GetParameters();

        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        int weightCount = inputSize * outputSize;

        // For each output neuron, compute the magnitude of its weight vector
        for (int i = 0; i < outputSize; i++)
        {
            T sumSquares = NumOps.Zero;

            // Sum squares of all weights for this output neuron
            for (int j = 0; j < inputSize; j++)
            {
                int idx = i * inputSize + j;
                if (idx < weightCount && idx < baseParams.Length)
                {
                    T weight = baseParams[idx];
                    sumSquares = NumOps.Add(sumSquares, NumOps.Multiply(weight, weight));
                }
            }

            // Magnitude is the L2 norm
            _magnitude[i] = NumOps.Sqrt(sumSquares);

            // Ensure magnitude is never zero (for numerical stability)
            if (NumOps.Equals(_magnitude[i], NumOps.Zero))
            {
                _magnitude[i] = NumOps.FromDouble(1e-8);
            }
        }
    }

    /// <summary>
    /// Recomposes weights from magnitude and direction components.
    /// </summary>
    /// <param name="direction">The normalized direction matrix.</param>
    /// <returns>The full weight matrix (magnitude * direction).</returns>
    /// <remarks>
    /// <para>
    /// This method reconstructs the full weight matrix by scaling each direction vector
    /// by its corresponding magnitude value.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This puts magnitude and direction back together.
    ///
    /// After we've adjusted the direction with LoRA and have the magnitude stored separately,
    /// this combines them back into normal weights. Think of it as:
    /// - Take each direction vector (unit vector)
    /// - Scale it by its magnitude (scalar)
    /// - Result: the full weight vector
    ///
    /// This is used during forward pass to get the effective weights.
    /// </para>
    /// </remarks>
    private Matrix<T> RecomposeWeights(Matrix<T> direction)
    {
        int outputSize = direction.Rows;
        int inputSize = direction.Columns;

        Matrix<T> weights = new Matrix<T>(outputSize, inputSize);

        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                weights[i, j] = NumOps.Multiply(_magnitude[i], direction[i, j]);
            }
        }

        return weights;
    }

    /// <summary>
    /// Normalizes a matrix row-wise (each row becomes a unit vector).
    /// </summary>
    /// <param name="matrix">The matrix to normalize.</param>
    /// <returns>Row-normalized matrix where each row has unit L2 norm.</returns>
    /// <remarks>
    /// <para>
    /// For each row (weight vector), this computes the L2 norm and divides all elements by it.
    /// This ensures each direction vector has unit length.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This makes each weight vector have length 1.
    ///
    /// When we separate magnitude and direction, the direction must be a unit vector
    /// (length = 1). This method ensures that by dividing each weight vector by its length.
    ///
    /// Example: vector [3, 4] has length 5, so normalized it becomes [0.6, 0.8]
    /// </para>
    /// </remarks>
    private Matrix<T> NormalizeRows(Matrix<T> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;

        Matrix<T> normalized = new Matrix<T>(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            // Compute L2 norm of row
            T sumSquares = NumOps.Zero;
            for (int j = 0; j < cols; j++)
            {
                T val = matrix[i, j];
                sumSquares = NumOps.Add(sumSquares, NumOps.Multiply(val, val));
            }

            T norm = NumOps.Sqrt(sumSquares);

            // Avoid division by zero
            if (NumOps.Equals(norm, NumOps.Zero))
            {
                norm = NumOps.FromDouble(1e-8);
            }

            // Normalize row
            for (int j = 0; j < cols; j++)
            {
                normalized[i, j] = NumOps.Divide(matrix[i, j], norm);
            }
        }

        return normalized;
    }

    /// <summary>
    /// Performs the forward pass through DoRA adapter.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output combining base layer with DoRA-adapted weights.</returns>
    /// <remarks>
    /// <para>
    /// The DoRA forward pass:
    /// 1. Gets base layer weights W
    /// 2. Computes direction: d = W / ||W||
    /// 3. Applies LoRA to direction: d' = d + LoRA(input)
    /// 4. Normalizes adapted direction: d_norm = d' / ||d'||
    /// 5. Recomposes weights: W' = m * d_norm
    /// 6. Computes output: y = input @ W'^T
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is where DoRA's magic happens during prediction.
    ///
    /// Step by step:
    /// 1. Get the original weights from the base layer
    /// 2. Split into magnitude (stored) and direction (computed)
    /// 3. Apply LoRA's correction to the direction (not the magnitude!)
    /// 4. Normalize the new direction to keep it as a unit vector
    /// 5. Multiply magnitude back in to get final weights
    /// 6. Use these adjusted weights to compute the output
    ///
    /// The key difference from standard LoRA:
    /// - Standard LoRA: output = base_output + lora_output
    /// - DoRA: output = input @ (m * normalize(d + lora_output))
    ///
    /// DoRA's approach gives more stable training because we control magnitude separately.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Get base layer parameters and extract weights
        Vector<T> baseParams = _baseLayer.GetParameters();
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        int weightCount = inputSize * outputSize;

        // Extract weight matrix from base layer (assuming weights come first)
        Matrix<T> baseWeights = new Matrix<T>(outputSize, inputSize);
        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                int weightIdx = i * inputSize + j;
                if (weightIdx < weightCount && weightIdx < baseParams.Length)
                {
                    baseWeights[i, j] = baseParams[weightIdx];
                }
                else
                {
                    baseWeights[i, j] = NumOps.Zero;
                }
            }
        }

        // Compute base direction (W / ||W||)
        Matrix<T> baseDirection = NormalizeRows(baseWeights);

        // Get LoRA weight contribution as matrix (A × B)
        // MergeWeights() returns [inputSize, outputSize], but we need [outputSize, inputSize]
        // to match base weight dimensions, so we transpose it
        Matrix<T> loraWeightDeltaRaw = _loraLayer.MergeWeights(); // This gives us [inputSize, outputSize]
        Matrix<T> loraWeightDelta = loraWeightDeltaRaw.Transpose(); // Now [outputSize, inputSize]

        int batchSize = input.Shape[0];

        // Add LoRA delta to base direction: d' = d + delta
        Matrix<T> adaptedDirection = new Matrix<T>(outputSize, inputSize);
        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                adaptedDirection[i, j] = NumOps.Add(baseDirection[i, j], loraWeightDelta[i, j]);
            }
        }

        // Normalize the adapted direction: d_norm = d' / ||d'||
        _lastNormalizedDirection = NormalizeRows(adaptedDirection);

        // Recompose weights: W' = m * d_norm
        Matrix<T> finalWeights = RecomposeWeights(_lastNormalizedDirection);

        // Compute output: y = input @ W'^T
        // Convert input to matrix and cache for backward pass
        _lastInputMatrix = new Matrix<T>(batchSize, inputSize);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                _lastInputMatrix[i, j] = input[i * inputSize + j];
            }
        }

        // Matrix multiply: [batchSize, inputSize] @ [inputSize, outputSize]
        Matrix<T> outputMatrix = _lastInputMatrix.Multiply(finalWeights.Transpose());

        // Convert back to tensor
        Vector<T> outputData = new Vector<T>(batchSize * outputSize);
        int idx = 0;
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                outputData[idx++] = outputMatrix[i, j];
            }
        }

        return new Tensor<T>(new[] { batchSize, outputSize }, outputData);
    }

    /// <summary>
    /// Performs the backward pass through DoRA adapter.
    /// </summary>
    /// <param name="outputGradient">Gradient flowing back from the next layer.</param>
    /// <returns>Gradient to pass to the previous layer.</returns>
    /// <remarks>
    /// <para>
    /// The backward pass computes gradients for:
    /// 1. Magnitude parameters (one per output neuron)
    /// 2. LoRA matrices A and B (via LoRA layer's backward)
    /// 3. Base layer weights (if not frozen)
    ///
    /// The key challenge is computing how changes to magnitude and direction affect the loss,
    /// given that the direction is normalized during forward pass.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is where DoRA learns during training.
    ///
    /// Backward pass figures out how to improve three things:
    /// 1. The magnitude of each output neuron's weights
    /// 2. The LoRA matrices that adjust the direction
    /// 3. The base layer weights (if we're training them too)
    ///
    /// The math is complex because we need to account for the normalization step.
    /// When we normalize the direction, it creates a dependency between all elements
    /// of a weight vector, so the gradients need to account for that.
    ///
    /// For simplicity, this implementation computes approximate gradients that work well
    /// in practice. The exact gradients would require storing more intermediate values
    /// from the forward pass.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastNormalizedDirection == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass");
        }

        int batchSize = outputGradient.Shape[0];
        int outputSize = GetOutputShape()[0];
        int inputSize = GetInputShape()[0];

        // Convert output gradient to matrix
        Matrix<T> gradMatrix = new Matrix<T>(batchSize, outputSize);
        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                gradMatrix[i, j] = outputGradient[i * outputSize + j];
            }
        }

        // Compute magnitude gradients
        // The correct gradient is: dL/dm_i = sum_batch(dL/dout_i * (normalized_direction_i · input_batch))
        // Since output = input @ (m * normalized_direction)^T = input @ (normalized_direction^T * diag(m))
        // For each output neuron i: output_batch_i = (normalized_direction_i · input_batch) * m_i
        // Therefore: dL/dm_i = sum_batch(dL/dout_batch_i * (normalized_direction_i · input_batch))
        if (_lastInputMatrix == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass");
        }

        _magnitudeGradient = new Vector<T>(_magnitude.Length);
        for (int i = 0; i < outputSize; i++)
        {
            T gradSum = NumOps.Zero;
            for (int b = 0; b < batchSize; b++)
            {
                // Compute (normalized_direction_i · input_batch)
                T dot = NumOps.Zero;
                for (int j = 0; j < inputSize; j++)
                {
                    T term = NumOps.Multiply(_lastNormalizedDirection[i, j], _lastInputMatrix[b, j]);
                    dot = NumOps.Add(dot, term);
                }

                // dL/dm_i contribution from this batch element
                T gradContribution = NumOps.Multiply(gradMatrix[b, i], dot);
                gradSum = NumOps.Add(gradSum, gradContribution);
            }
            _magnitudeGradient[i] = gradSum;
        }

        // Propagate gradient through LoRA layer
        // The LoRA layer's backward will compute gradients for A and B matrices
        Tensor<T> loraInputGrad = _loraLayer.Backward(outputGradient);

        // If base layer is not frozen, propagate through it too
        Tensor<T> baseInputGrad;
        if (!_freezeBaseLayer)
        {
            baseInputGrad = _baseLayer.Backward(outputGradient);

            // Sum input gradients from both paths
            Vector<T> inputGradData = new Vector<T>(loraInputGrad.Length);
            for (int i = 0; i < loraInputGrad.Length; i++)
            {
                inputGradData[i] = NumOps.Add(loraInputGrad[i], baseInputGrad[i]);
            }

            return new Tensor<T>(loraInputGrad.Shape, inputGradData);
        }
        else
        {
            // Only LoRA input gradient
            return loraInputGrad;
        }
    }

    /// <summary>
    /// Updates parameters using the specified learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    public override void UpdateParameters(T learningRate)
    {
        // Update LoRA layer (always)
        _loraLayer.UpdateParameters(learningRate);

        // Update magnitude parameters
        if (_magnitudeGradient != null)
        {
            for (int i = 0; i < _magnitude.Length; i++)
            {
                T update = NumOps.Multiply(_magnitudeGradient[i], learningRate);
                _magnitude[i] = NumOps.Subtract(_magnitude[i], update);

                // Ensure magnitude stays positive
                if (NumOps.LessThan(_magnitude[i], NumOps.FromDouble(1e-8)))
                {
                    _magnitude[i] = NumOps.FromDouble(1e-8);
                }
            }
        }

        // Update base layer (only if not frozen)
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
    /// <returns>Vector containing all parameters (base if not frozen, LoRA, magnitude).</returns>
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

        // Pack magnitude parameters
        for (int i = 0; i < _magnitude.Length; i++)
        {
            Parameters[idx++] = _magnitude[i];
        }
    }

    /// <summary>
    /// Updates the components from the parameter vector.
    /// </summary>
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

        // Unpack magnitude parameters
        for (int i = 0; i < _magnitude.Length; i++)
        {
            _magnitude[i] = Parameters[idx++];
        }
    }

    /// <summary>
    /// Merges the DoRA adaptation into the base layer and returns the merged layer.
    /// </summary>
    /// <returns>A new layer with DoRA weights merged into the base layer's weights.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the base layer type is not supported for merging.</exception>
    /// <remarks>
    /// <para>
    /// This method creates a final layer with the DoRA adaptations baked in.
    /// The merged weights are: W' = m * normalize(d + LoRA_delta)
    /// where m is magnitude, d is base direction, and LoRA_delta is the LoRA contribution.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This "bakes in" your DoRA adaptation for deployment.
    ///
    /// After training with DoRA, you probably want to deploy a simpler model without
    /// all the DoRA machinery. This method creates that simpler model by:
    /// 1. Computing the final adapted direction (base + LoRA)
    /// 2. Normalizing the direction
    /// 3. Multiplying by magnitude to get final weights
    /// 4. Creating a new layer with these merged weights
    ///
    /// The result is a standard layer that behaves like your DoRA-adapted model
    /// but is faster to run because it doesn't need to do the decomposition at runtime.
    /// </para>
    /// </remarks>
    public override ILayer<T> MergeToOriginalLayer()
    {
        // This is a simplified implementation that works with DenseLayer base
        DenseLayer<T>? denseBase = _baseLayer as DenseLayer<T>;
        FullyConnectedLayer<T>? fcBase = _baseLayer as FullyConnectedLayer<T>;

        if (denseBase == null && fcBase == null)
        {
            throw new InvalidOperationException("DoRAAdapter currently only supports DenseLayer or FullyConnectedLayer base layers for merging");
        }

        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];

        // Get base layer weights
        Vector<T> baseParams = _baseLayer.GetParameters();
        Matrix<T> baseWeights = new Matrix<T>(outputSize, inputSize);
        int weightCount = inputSize * outputSize;

        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                int weightIdx = i * inputSize + j;
                if (weightIdx < weightCount && weightIdx < baseParams.Length)
                {
                    baseWeights[i, j] = baseParams[weightIdx];
                }
            }
        }

        // Compute base direction
        Matrix<T> baseDirection = NormalizeRows(baseWeights);

        // Get LoRA weight contribution
        Matrix<T> loraWeights = _loraLayer.MergeWeights();

        // Add LoRA to direction and normalize
        Matrix<T> adaptedDirection = new Matrix<T>(outputSize, inputSize);
        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                adaptedDirection[i, j] = NumOps.Add(baseDirection[i, j], loraWeights[i, j]);
            }
        }

        Matrix<T> normalizedDirection = NormalizeRows(adaptedDirection);

        // Recompose with magnitude: W' = m * d_norm
        Matrix<T> finalWeights = RecomposeWeights(normalizedDirection);

        // Create merged parameters (weights + biases)
        Vector<T> mergedParams = new Vector<T>(baseParams.Length);

        // Copy merged weights
        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                int weightIdx = i * inputSize + j;
                mergedParams[weightIdx] = finalWeights[i, j];
            }
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
    /// Resets the internal state of the adapter.
    /// </summary>
    public override void ResetState()
    {
        _baseLayer.ResetState();
        _loraLayer.ResetState();
        _lastNormalizedDirection = null;
        _lastInputMatrix = null;
        _magnitudeGradient = null;
    }
}

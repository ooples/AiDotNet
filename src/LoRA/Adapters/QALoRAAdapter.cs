using AiDotNet.Interfaces;

namespace AiDotNet.LoRA.Adapters;

/// <summary>
/// Quantization-Aware LoRA (QA-LoRA) adapter that combines parameter-efficient fine-tuning with group-wise quantization awareness.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// QA-LoRA extends standard LoRA by being aware of quantization during training. This allows the adapter
/// to learn compensations for quantization errors, resulting in better final accuracy compared to
/// post-training quantization approaches. The key innovation is simulating quantization during the
/// forward pass so that gradients account for quantization effects.
/// </para>
/// <para><b>For Beginners:</b> QA-LoRA solves a critical problem when deploying models to resource-constrained devices.
///
/// The Problem:
/// - Modern neural networks use high-precision numbers (32-bit floats)
/// - Mobile/edge devices need lower precision (4-bit or 8-bit integers) for speed and memory
/// - Converting after training (post-training quantization) often loses accuracy
///
/// QA-LoRA's Solution:
/// - Simulates low-precision during training (quantization-aware training)
/// - Learns to compensate for quantization errors
/// - Uses LoRA for parameter efficiency (only trains the adaptation, not full model)
/// - Applies group-wise quantization (groups of weights share scaling factors)
///
/// Key Concepts:
///
/// 1. Quantization: Converting high-precision numbers to low-precision
///    Example: 32-bit float 0.7234 → 4-bit integer 11 (range 0-15)
///
/// 2. Group-wise Quantization: Instead of one scale for all weights, weights are divided into groups,
///    each with its own scale. This preserves more information.
///    Example: 64 weights → 4 groups of 16 weights each, each group has its own scale
///
/// 3. Quantization-Aware Training: During training, simulate quantization in forward pass:
///    - Convert weights to low-precision (quantize)
///    - Immediately convert back to high-precision (dequantize)
///    - Use these "quantized" values for computation
///    - Gradients learn to compensate for the quantization noise
///
/// 4. Straight-Through Estimator (STE): During backward pass, treat quantization as identity
///    - Forward: y = quantize(x)
///    - Backward: ∂y/∂x ≈ 1 (gradient flows through unchanged)
///    - This allows gradients to update the full-precision weights
///
/// Parameters:
/// - QuantizationBits: How many bits to use (4-bit, 8-bit, etc.)
/// - GroupSize: How many weights per quantization group (e.g., 64, 128)
/// - Smaller GroupSize = more scales = better accuracy but more overhead
/// - Larger GroupSize = fewer scales = more efficient but less accurate
///
/// Example Workflow:
/// 1. Training: Forward pass uses simulated 4-bit quantization
/// 2. Gradients: Backward pass learns to work around quantization errors
/// 3. Deployment: Actually quantize the merged weights to 4-bit for inference
/// 4. Result: Much better accuracy than quantizing after training
///
/// Research Context:
/// - QLoRA (May 2023): Introduced efficient 4-bit quantization for LoRA
/// - QA-LoRA: Extends this with quantization-aware training for better results
/// - Typical improvement: 1-3% accuracy gain over post-training quantization
///
/// Use Cases:
/// - Deploying large language models on mobile devices
/// - Edge AI applications with strict memory constraints
/// - Reducing model size while maintaining accuracy
/// - Fine-tuning for deployment on specific hardware (TPUs, specialized accelerators)
/// </para>
/// </remarks>
public class QALoRAAdapter<T> : LoRAAdapterBase<T>
{
    /// <summary>
    /// Number of bits to use for quantization (e.g., 4, 8).
    /// </summary>
    private int _quantizationBits;

    /// <summary>
    /// Number of weights per quantization group.
    /// </summary>
    /// <remarks>
    /// Smaller groups preserve more information but require more scaling factors.
    /// Typical values: 64, 128, 256.
    /// </remarks>
    private int _groupSize;

    /// <summary>
    /// Whether quantization simulation is currently enabled.
    /// </summary>
    /// <remarks>
    /// Can be disabled during initial warmup or final evaluation.
    /// </remarks>
    private bool _quantizationEnabled;

    /// <summary>
    /// Gets or sets the number of bits used for quantization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Common values:
    /// - 4 bits: Extremely memory-efficient, requires careful tuning
    /// - 8 bits: Good balance of efficiency and accuracy
    /// - 16 bits: Close to full precision, minimal savings
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much compression you apply.
    /// - 4-bit: 8x compression (32-bit → 4-bit), more aggressive
    /// - 8-bit: 4x compression (32-bit → 8-bit), safer choice
    /// Lower bits = smaller model but harder to maintain accuracy.
    /// </para>
    /// </remarks>
    public int QuantizationBits
    {
        get => _quantizationBits;
        set
        {
            if (value < 1 || value > 16)
            {
                throw new ArgumentException("Quantization bits must be between 1 and 16", nameof(value));
            }
            _quantizationBits = value;
        }
    }

    /// <summary>
    /// Gets or sets the group size for group-wise quantization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Group-wise quantization divides weights into groups, each with independent scaling factors.
    /// This preserves more dynamic range than using a single scale for all weights.
    /// </para>
    /// <para><b>For Beginners:</b> Imagine you have 1024 weights to quantize:
    /// - GroupSize = 1024: One scale for all weights (simple but loses information)
    /// - GroupSize = 128: Eight scales (1024/128 = 8 groups, better accuracy)
    /// - GroupSize = 64: Sixteen scales (1024/64 = 16 groups, even better but more overhead)
    ///
    /// Smaller groups mean each group's weights are more similar, so a single scale per group
    /// is more accurate. But you need to store more scales.
    /// </para>
    /// </remarks>
    public int GroupSize
    {
        get => _groupSize;
        set
        {
            if (value < 1)
            {
                throw new ArgumentException("Group size must be positive", nameof(value));
            }
            _groupSize = value;
        }
    }

    /// <summary>
    /// Gets or sets whether quantization simulation is enabled during forward/backward passes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Disabling quantization can be useful for:
    /// - Initial warmup phases
    /// - Evaluating full-precision performance
    /// - Debugging training issues
    /// </para>
    /// <para><b>For Beginners:</b> This is like a toggle switch:
    /// - Enabled: Simulate low-precision during training (quantization-aware)
    /// - Disabled: Use full-precision (standard LoRA training)
    /// You might start with it disabled for stability, then enable it partway through training.
    /// </para>
    /// </remarks>
    public bool QuantizationEnabled
    {
        get => _quantizationEnabled;
        set => _quantizationEnabled = value;
    }

    /// <summary>
    /// Initializes a new QA-LoRA adapter with quantization awareness.
    /// </summary>
    /// <param name="baseLayer">The layer to adapt with QA-LoRA.</param>
    /// <param name="rank">The rank of the LoRA decomposition.</param>
    /// <param name="quantizationBits">Number of bits for quantization (e.g., 4, 8).</param>
    /// <param name="groupSize">Number of weights per quantization group.</param>
    /// <param name="alpha">The LoRA scaling factor (defaults to rank if negative).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer's parameters during training.</param>
    /// <exception cref="ArgumentNullException">Thrown when baseLayer is null.</exception>
    /// <exception cref="ArgumentException">Thrown when quantizationBits or groupSize are invalid.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a QA-LoRA adapter that will train with quantization awareness.
    ///
    /// Parameters:
    /// - baseLayer: The layer you want to efficiently fine-tune
    /// - rank: How much compression for LoRA (lower = fewer parameters)
    /// - quantizationBits: Target precision for deployment (4 or 8 typically)
    /// - groupSize: Granularity of quantization (64-128 recommended)
    /// - alpha: How strong the LoRA effect is
    /// - freezeBaseLayer: Whether to lock the original weights (usually true)
    ///
    /// Example: QALoRAAdapter(myLayer, rank=8, quantizationBits=4, groupSize=64)
    /// - Uses 8-rank LoRA for parameter efficiency
    /// - Simulates 4-bit quantization during training
    /// - Groups of 64 weights share scaling factors
    /// </para>
    /// </remarks>
    public QALoRAAdapter(
        ILayer<T> baseLayer,
        int rank,
        int quantizationBits,
        int groupSize,
        double alpha = -1,
        bool freezeBaseLayer = true)
        : base(baseLayer, rank, alpha, freezeBaseLayer)
    {
        if (quantizationBits < 1 || quantizationBits > 16)
        {
            throw new ArgumentException("Quantization bits must be between 1 and 16", nameof(quantizationBits));
        }

        if (groupSize < 1)
        {
            throw new ArgumentException("Group size must be positive", nameof(groupSize));
        }

        _quantizationBits = quantizationBits;
        _groupSize = groupSize;
        _quantizationEnabled = true; // Enabled by default
    }

    /// <summary>
    /// Performs the forward pass through both base and LoRA layers with quantization simulation.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Sum of base layer output and quantized LoRA output.</returns>
    /// <remarks>
    /// <para>
    /// The forward pass with quantization awareness:
    /// 1. Compute base layer output (no quantization)
    /// 2. Get LoRA layer parameters
    /// 3. Simulate quantization: quantize → dequantize (if enabled)
    /// 4. Compute LoRA output with quantized parameters
    /// 5. Sum base + quantized LoRA outputs
    /// </para>
    /// <para><b>For Beginners:</b> This is where quantization-aware training happens!
    ///
    /// Normal LoRA forward pass:
    /// - base_output = base_layer(input)
    /// - lora_output = lora_layer(input)  // Uses full-precision weights
    /// - return base_output + lora_output
    ///
    /// QA-LoRA forward pass:
    /// - base_output = base_layer(input)
    /// - lora_weights_full = get_lora_weights()  // Full precision
    /// - lora_weights_quant = dequantize(quantize(lora_weights_full))  // Simulate quantization
    /// - lora_output = compute_with_quantized_weights(input, lora_weights_quant)
    /// - return base_output + lora_output
    ///
    /// The key difference: We temporarily quantize and dequantize the LoRA weights,
    /// which adds noise. The gradients will learn to work despite this noise!
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Forward through base layer (unchanged)
        Tensor<T> baseOutput = _baseLayer.Forward(input);

        // Forward through LoRA layer with optional quantization simulation
        Tensor<T> loraOutput;

        if (_quantizationEnabled)
        {
            // Simulate quantization on LoRA parameters
            Vector<T> originalParams = _loraLayer.GetParameters();
            Vector<T> quantizedParams = QuantizeAndDequantize(originalParams);

            // Temporarily set quantized parameters
            _loraLayer.SetParameters(quantizedParams);

            // Forward with quantized parameters
            loraOutput = _loraLayer.Forward(input);

            // Restore original parameters (important for gradient computation)
            _loraLayer.SetParameters(originalParams);
        }
        else
        {
            // Standard LoRA forward (no quantization simulation)
            loraOutput = _loraLayer.Forward(input);
        }

        // Sum the outputs
        Tensor<T> result = new Tensor<T>(baseOutput.Shape);
        for (int i = 0; i < baseOutput.Length; i++)
        {
            result[i] = NumOps.Add(baseOutput[i], loraOutput[i]);
        }

        return result;
    }

    /// <summary>
    /// Performs the backward pass through both layers, accounting for quantization in gradients.
    /// </summary>
    /// <param name="outputGradient">Gradient flowing back from the next layer.</param>
    /// <returns>Gradient to pass to the previous layer.</returns>
    /// <remarks>
    /// <para>
    /// The backward pass uses the Straight-Through Estimator (STE) for quantization:
    /// - Forward: y = quantize(x)
    /// - Backward: ∂L/∂x = ∂L/∂y (gradient passes through unchanged)
    ///
    /// This allows gradients to flow to the full-precision weights despite quantization.
    /// </para>
    /// <para><b>For Beginners:</b> This is the tricky part of quantization-aware training!
    ///
    /// The Problem:
    /// - Quantization is a discontinuous operation (rounding)
    /// - Discontinuous operations have zero or undefined gradients
    /// - If gradients can't flow, we can't update weights, so training fails
    ///
    /// The Solution (Straight-Through Estimator):
    /// - Pretend quantization is the identity function during backprop
    /// - Forward: actually quantize (add noise)
    /// - Backward: pretend we didn't quantize (gradient flows through)
    /// - This is mathematically "wrong" but works well in practice!
    ///
    /// Why it works:
    /// - The forward pass sees quantized values (learns to compensate)
    /// - The backward pass updates full-precision weights (maintains precision)
    /// - The network learns weights that work well when quantized
    ///
    /// Example:
    /// Forward: weight = 0.7234 → quantize → 0.7333 (closest 4-bit value)
    /// Backward: gradient flows as if 0.7234 → 0.7234 (identity)
    /// Update: 0.7234 - learning_rate * gradient (updates full-precision weight)
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // The Straight-Through Estimator (STE) means we compute gradients
        // as if quantization was the identity function.
        // The base implementation handles this correctly because:
        // 1. We restored original (full-precision) parameters after Forward
        // 2. Backward computes gradients w.r.t. those full-precision parameters
        // 3. Gradient flow is not blocked by quantization

        // Standard LoRA backward pass
        Tensor<T> loraInputGrad = _loraLayer.Backward(outputGradient);
        Tensor<T> baseInputGrad = _baseLayer.Backward(outputGradient);

        // Sum input gradients
        Tensor<T> inputGrad = new Tensor<T>(loraInputGrad.Shape);
        for (int i = 0; i < loraInputGrad.Length; i++)
        {
            inputGrad[i] = NumOps.Add(loraInputGrad[i], baseInputGrad[i]);
        }

        // Update parameter gradients vector
        UpdateParameterGradientsFromLayers();

        return inputGrad;
    }

    /// <summary>
    /// Simulates quantization and dequantization using group-wise scaling.
    /// </summary>
    /// <param name="parameters">Full-precision parameters to quantize.</param>
    /// <returns>Parameters after quantize→dequantize cycle (simulating quantization noise).</returns>
    /// <remarks>
    /// <para>
    /// Group-wise quantization process:
    /// 1. Divide parameters into groups of size GroupSize
    /// 2. For each group:
    ///    a. Find the maximum absolute value in the group
    ///    b. Compute scale = max_abs / (2^bits - 1)
    ///    c. Quantize: int_value = round(parameter / scale)
    ///    d. Clamp to range [0, 2^bits - 1]
    ///    e. Dequantize: parameter = int_value * scale
    /// 3. Concatenate all groups back together
    /// </para>
    /// <para><b>For Beginners:</b> This is the core of quantization simulation!
    ///
    /// Step-by-step example with 4-bit quantization, group size 4:
    ///
    /// Input: [0.8, 0.6, -0.4, 0.2, 0.9, -0.7, 0.3, -0.5]
    ///
    /// Group 1: [0.8, 0.6, -0.4, 0.2]
    ///   - Max absolute value: 0.8
    ///   - Range for 4-bit: 0 to 15 (2^4 - 1 = 15)
    ///   - Scale: 0.8 / 15 = 0.0533
    ///   - Quantize: [15, 11, -8, 4] (divided by scale, rounded)
    ///   - Clamp to [0, 15]: [15, 11, 0, 4] (negative values clamped)
    ///   - Dequantize: [0.8, 0.5867, 0.0, 0.2133] (multiply by scale)
    ///   - Information lost: -0.4 became 0.0, 0.6 became 0.5867
    ///
    /// Group 2: [0.9, -0.7, 0.3, -0.5]
    ///   - Max absolute value: 0.9
    ///   - Scale: 0.9 / 15 = 0.06
    ///   - Similar process...
    ///
    /// The network learns to work with these quantized values during training,
    /// so when we actually deploy with 4-bit weights, accuracy is maintained!
    /// </para>
    /// </remarks>
    private Vector<T> QuantizeAndDequantize(Vector<T> parameters)
    {
        int numParams = parameters.Length;
        Vector<T> quantized = new Vector<T>(numParams);

        // Calculate number of groups
        int numGroups = (numParams + _groupSize - 1) / _groupSize; // Ceiling division

        // Use signed quantization for asymmetric range (one more negative value)
        // For n-bit signed: range is -2^(n-1) to 2^(n-1)-1
        // e.g., 4-bit signed: -8 to 7, 8-bit signed: -128 to 127
        double maxQuantizedValue = Math.Pow(2.0, _quantizationBits - 1) - 1.0;
        double minQuantizedValue = -Math.Pow(2.0, _quantizationBits - 1);

        // Process each group
        for (int g = 0; g < numGroups; g++)
        {
            int groupStart = g * _groupSize;
            int groupEnd = Math.Min(groupStart + _groupSize, numParams);
            int groupActualSize = groupEnd - groupStart;

            // Find maximum absolute value in this group
            T maxAbs = NumOps.Zero;
            for (int i = groupStart; i < groupEnd; i++)
            {
                T absValue = NumOps.Abs(parameters[i]);
                if (NumOps.GreaterThan(absValue, maxAbs))
                {
                    maxAbs = absValue;
                }
            }

            // Compute scale factor for this group
            // scale = max_abs / (2^bits - 1)
            // If max_abs is zero, use a small epsilon to avoid division by zero
            if (NumOps.Equals(maxAbs, NumOps.Zero))
            {
                maxAbs = NumOps.FromDouble(1e-8);
            }

            T scale = NumOps.Divide(maxAbs, NumOps.FromDouble(maxQuantizedValue));

            // Quantize and dequantize each parameter in the group
            for (int i = groupStart; i < groupEnd; i++)
            {
                // Quantize: int_value = round(param / scale)
                T normalized = NumOps.Divide(parameters[i], scale);
                double normalizedDouble = Convert.ToDouble(normalized);
                double quantizedDouble = Math.Round(normalizedDouble);

                // Clamp to valid signed range [-maxQuantizedValue, maxQuantizedValue]
                quantizedDouble = Math.Max(minQuantizedValue, Math.Min(maxQuantizedValue, quantizedDouble));

                // Dequantize: param = int_value * scale
                T dequantized = NumOps.Multiply(NumOps.FromDouble(quantizedDouble), scale);
                quantized[i] = dequantized;
            }
        }

        return quantized;
    }

    /// <summary>
    /// Updates parameter gradients from both layers.
    /// </summary>
    private void UpdateParameterGradientsFromLayers()
    {
        ParameterGradients = new Vector<T>(ParameterCount);
        int idx = 0;

        // If base layer is not frozen, pack its gradients first
        if (!_freezeBaseLayer)
        {
            Vector<T> baseGrads = _baseLayer.GetParameterGradients();
            for (int i = 0; i < baseGrads.Length; i++)
            {
                ParameterGradients[idx++] = baseGrads[i];
            }
        }

        // Pack LoRA gradients
        Vector<T> loraGrads = _loraLayer.GetParameterGradients();
        for (int i = 0; i < loraGrads.Length; i++)
        {
            ParameterGradients[idx++] = loraGrads[i];
        }
    }

    /// <summary>
    /// Merges the LoRA adaptation into the base layer and returns a quantized merged layer.
    /// </summary>
    /// <returns>A new layer with LoRA weights merged and quantized into the base layer's weights.</returns>
    /// <remarks>
    /// <para>
    /// The merging process for QA-LoRA:
    /// 1. Get the LoRA weight contribution (B * A * scaling)
    /// 2. Add LoRA weights to base layer weights
    /// 3. Apply actual quantization to the merged weights (for deployment)
    /// 4. Return a new layer with quantized merged weights
    /// </para>
    /// <para><b>For Beginners:</b> This is the final step - creating the deployment model!
    ///
    /// Training vs. Deployment:
    /// - During training: Simulate quantization, keep full-precision weights
    /// - After training: Actually quantize and merge for deployment
    ///
    /// What this method does:
    /// 1. Merge: base_weights + lora_weights → full_precision_merged
    /// 2. Quantize: full_precision_merged → quantized_weights (actually reduced to N bits)
    /// 3. Create new layer: DenseLayer with quantized_weights
    ///
    /// Result: A layer that's actually using N-bit precision (smaller, faster)
    /// instead of just simulating it!
    ///
    /// Note: This example quantizes to the parameter vector. For true deployment,
    /// you'd use a specialized quantized layer class that stores integer weights
    /// and performs integer arithmetic. This is a simplified version for demonstration.
    /// </para>
    /// </remarks>
    public override ILayer<T> MergeToOriginalLayer()
    {
        // Currently only supports DenseLayer or FullyConnectedLayer base layers
        DenseLayer<T>? denseBase = _baseLayer as DenseLayer<T>;
        FullyConnectedLayer<T>? fcBase = _baseLayer as FullyConnectedLayer<T>;

        if (denseBase == null && fcBase == null)
        {
            throw new InvalidOperationException("QALoRAAdapter currently only supports DenseLayer or FullyConnectedLayer base layers");
        }

        // Get the LoRA weight contribution
        Matrix<T> loraWeights = _loraLayer.MergeWeights();

        // Get base layer parameters
        Vector<T> baseParams = _baseLayer.GetParameters();

        // Calculate dimensions
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        int weightCount = inputSize * outputSize;

        // Create merged parameters (base + LoRA)
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

        // Apply quantization to the merged weights (actual quantization for deployment)
        Vector<T> mergedWeights = new Vector<T>(weightCount);
        for (int i = 0; i < weightCount; i++)
        {
            mergedWeights[i] = mergedParams[i];
        }

        Vector<T> quantizedWeights = QuantizeAndDequantize(mergedWeights);

        // Put quantized weights back into parameter vector
        for (int i = 0; i < weightCount; i++)
        {
            mergedParams[i] = quantizedWeights[i];
        }

        // Create a new dense layer with quantized merged parameters
        DenseLayer<T> mergedLayer = new DenseLayer<T>(inputSize, outputSize, (IActivationFunction<T>?)null);
        mergedLayer.SetParameters(mergedParams);

        return mergedLayer;
    }

    /// <summary>
    /// Gets statistics about quantization for the current LoRA parameters.
    /// </summary>
    /// <returns>A tuple containing (average error, max error, number of groups).</returns>
    /// <remarks>
    /// <para>
    /// This method helps you understand the quantization quality:
    /// - Average error: Mean absolute difference between full-precision and quantized values
    /// - Max error: Worst-case difference in any parameter
    /// - Number of groups: How many quantization groups are used
    /// </para>
    /// <para><b>For Beginners:</b> Use this to check how much information is lost to quantization.
    ///
    /// Example output:
    /// - Average error: 0.002 (most parameters within 0.002 of original)
    /// - Max error: 0.015 (worst case is 0.015 away from original)
    /// - Number of groups: 16 (using 16 different scales)
    ///
    /// Lower errors mean better quantization. If errors are too high:
    /// - Decrease group size (more scales, more accurate)
    /// - Increase quantization bits (more precision)
    /// - Adjust learning rate (help network adapt better)
    /// </para>
    /// </remarks>
    public (double averageError, double maxError, int numGroups) GetQuantizationStats()
    {
        Vector<T> originalParams = _loraLayer.GetParameters();
        Vector<T> quantizedParams = QuantizeAndDequantize(originalParams);

        double sumError = 0.0;
        double maxError = 0.0;

        for (int i = 0; i < originalParams.Length; i++)
        {
            double error = Math.Abs(Convert.ToDouble(NumOps.Subtract(originalParams[i], quantizedParams[i])));
            sumError += error;
            maxError = Math.Max(maxError, error);
        }

        double avgError = sumError / originalParams.Length;
        int numGroups = (originalParams.Length + _groupSize - 1) / _groupSize;

        return (avgError, maxError, numGroups);
    }
}

using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Implements the Attentive Transformer block used in TabNet architecture for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// The Attentive Transformer learns which features to pay attention to at each decision step.
/// It produces sparse attention masks using Sparsemax, ensuring that only the most relevant
/// features are selected for processing.
/// </para>
/// <para>
/// <b>For Beginners:</b> The Attentive Transformer is the "feature selector" in TabNet.
///
/// At each decision step, it answers the question: "Which features should I focus on next?"
///
/// Key concepts:
/// - Takes processed features and a "prior scales" tensor as input
/// - Prior scales track which features have already been used in previous steps
/// - Outputs a sparse attention mask (many values are exactly 0)
/// - Features with high attention values are selected for processing
/// - Features with 0 attention are completely ignored
///
/// This sparse attention provides interpretability - you can see exactly which
/// features the model considers important for each prediction.
/// </para>
/// <para>
/// The attention mechanism uses:
/// 1. Fully connected layer to compute attention logits
/// 2. Ghost Batch Normalization for regularization
/// 3. Prior scaling to discourage feature reuse
/// 4. Sparsemax to produce sparse probability distribution
/// </para>
/// <para>
/// Reference: "TabNet: Attentive Interpretable Tabular Learning" (Arik &amp; Pfister, AAAI 2021)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class AttentiveTransformer<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _inputDim;
    private readonly int _outputDim;
    private readonly double _relaxationFactor;

    // Attention layers
    private readonly FullyConnectedLayer<T> _fcLayer;
    private readonly GhostBatchNormalization<T> _bnLayer;
    private readonly Sparsemax<T> _sparsemax;

    // Cache for backward pass
    private Tensor<T>? _inputCache;
    private Tensor<T>? _priorScalesCache;
    private Tensor<T>? _attentionMaskCache;
    private Tensor<T>? _sparsemaxInputCache;

    /// <summary>
    /// Gets whether this layer supports training.
    /// </summary>
    public bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the AttentiveTransformer class.
    /// </summary>
    /// <param name="inputDim">The dimension of the processed features input.</param>
    /// <param name="outputDim">The dimension of the attention mask output (number of original features).</param>
    /// <param name="relaxationFactor">Controls how much features can be reused across steps. Default is 1.5.</param>
    /// <param name="virtualBatchSize">Virtual batch size for Ghost Batch Normalization.</param>
    /// <param name="momentum">Momentum for batch normalization.</param>
    /// <param name="epsilon">Epsilon for numerical stability.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When creating an Attentive Transformer:
    /// - inputDim: Size of the features from the Feature Transformer
    /// - outputDim: Number of original features to select from
    /// - relaxationFactor: Higher values allow more feature reuse across steps
    ///   - gamma = 1.0: Each feature used at most once (strictest)
    ///   - gamma = 1.5: Moderate reuse allowed (default)
    ///   - gamma > 2.0: Features can be reused more freely
    ///
    /// The relaxation factor controls the trade-off between feature diversity
    /// (using different features at each step) and allowing important features
    /// to be used multiple times.
    /// </para>
    /// </remarks>
    public AttentiveTransformer(
        int inputDim,
        int outputDim,
        double relaxationFactor = 1.5,
        int virtualBatchSize = 128,
        double momentum = 0.02,
        double epsilon = 1e-5)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _inputDim = inputDim;
        _outputDim = outputDim;
        _relaxationFactor = relaxationFactor;

        // FC layer maps processed features to attention logits
        _fcLayer = new FullyConnectedLayer<T>(inputDim, outputDim, (IActivationFunction<T>?)null);

        // Ghost Batch Normalization for regularization
        _bnLayer = new GhostBatchNormalization<T>(outputDim, virtualBatchSize, momentum, epsilon);

        // Sparsemax for sparse attention
        _sparsemax = new Sparsemax<T>();
    }

    /// <summary>
    /// Performs the forward pass to compute sparse attention mask.
    /// </summary>
    /// <param name="processedFeatures">Processed features from the previous step [batch_size, input_dim].</param>
    /// <param name="priorScales">Prior scales indicating feature usage [batch_size, output_dim].</param>
    /// <returns>Sparse attention mask [batch_size, output_dim].</returns>
    /// <remarks>
    /// <para>
    /// The forward pass computes attention in these steps:
    /// 1. Apply FC layer to get attention logits
    /// 2. Apply batch normalization
    /// 3. Multiply by prior scales (discourages reusing features)
    /// 4. Apply Sparsemax to get sparse probabilities
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This function decides which features to focus on:
    ///
    /// 1. **FC Layer**: Learns which features are important based on current processing state
    /// 2. **Batch Norm**: Helps training stability
    /// 3. **Prior Scaling**: Reduces attention for features already used in previous steps
    ///    (encouraging the model to look at new features)
    /// 4. **Sparsemax**: Produces a sparse attention where many values are exactly 0
    ///
    /// The output is an attention mask where:
    /// - High values (close to 1) mean "pay attention to this feature"
    /// - Zero values mean "completely ignore this feature"
    /// - Values sum to approximately 1 across features
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> processedFeatures, Tensor<T> priorScales)
    {
        _inputCache = processedFeatures;
        _priorScalesCache = priorScales;

        // Step 1: FC layer to compute attention logits
        var logits = _fcLayer.Forward(processedFeatures);

        // Step 2: Batch normalization
        var normalizedLogits = _bnLayer.Forward(logits);

        // Step 3: Apply prior scales (element-wise multiplication)
        // This discourages reusing features from previous steps
        var scaledLogits = MultiplyTensors(normalizedLogits, priorScales);
        _sparsemaxInputCache = scaledLogits;

        // Step 4: Apply Sparsemax to get sparse attention
        var attentionMask = _sparsemax.Forward(scaledLogits, axis: 1);
        _attentionMaskCache = attentionMask;

        return attentionMask;
    }

    /// <summary>
    /// Performs the standard forward pass (without prior scales).
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The attention mask with uniform prior scales.</returns>
    /// <remarks>
    /// This method is for ILayer interface compatibility. For actual TabNet usage,
    /// use the overload that accepts both processedFeatures and priorScales.
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input)
    {
        // Create uniform prior scales (all ones)
        var priorScales = new Tensor<T>([input.Shape[0], _outputDim]);
        for (int i = 0; i < priorScales.Length; i++)
        {
            priorScales[i] = _numOps.One;
        }
        return Forward(input, priorScales);
    }

    /// <summary>
    /// Updates the prior scales based on the current attention mask.
    /// </summary>
    /// <param name="priorScales">Current prior scales.</param>
    /// <param name="attentionMask">Current attention mask.</param>
    /// <returns>Updated prior scales for the next step.</returns>
    /// <remarks>
    /// <para>
    /// Prior scales are updated as: prior_new = prior * (gamma - attention)
    /// where gamma is the relaxation factor.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This function tracks feature usage across decision steps.
    ///
    /// After each step:
    /// - Features with high attention have their prior scales reduced
    /// - This makes them less likely to be selected in future steps
    /// - The relaxation factor controls how quickly scales decrease
    ///
    /// With gamma = 1.5 (default):
    /// - If a feature got attention = 1.0, its scale becomes 0.5 of previous
    /// - If a feature got attention = 0.0, its scale stays the same
    /// - If a feature got attention = 0.5, its scale becomes 1.0 of previous
    ///
    /// This mechanism ensures that the model explores different features
    /// across decision steps rather than repeatedly using the same ones.
    /// </para>
    /// </remarks>
    public Tensor<T> UpdatePriorScales(Tensor<T> priorScales, Tensor<T> attentionMask)
    {
        var newPriorScales = new Tensor<T>(priorScales.Shape);
        var gamma = _numOps.FromDouble(_relaxationFactor);

        for (int i = 0; i < priorScales.Length; i++)
        {
            // prior_new = prior * (gamma - attention)
            var scaleFactor = _numOps.Subtract(gamma, attentionMask[i]);
            newPriorScales[i] = _numOps.Multiply(priorScales[i], scaleFactor);
        }

        return newPriorScales;
    }

    /// <summary>
    /// Computes the sparsity regularization loss.
    /// </summary>
    /// <param name="attentionMask">The attention mask from forward pass.</param>
    /// <returns>The entropy-based sparsity loss.</returns>
    /// <remarks>
    /// <para>
    /// The sparsity loss encourages sparse attention masks by penalizing entropy.
    /// Lower entropy means more focused (sparse) attention.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This loss encourages the model to be "picky" about features.
    ///
    /// Entropy measures how spread out the attention is:
    /// - High entropy: Attention is spread across many features (unfocused)
    /// - Low entropy: Attention is concentrated on few features (focused)
    ///
    /// By minimizing entropy, we encourage the model to focus on fewer,
    /// more important features at each step. This improves:
    /// - Interpretability (clearer feature importance)
    /// - Computational efficiency (fewer features processed)
    /// - Generalization (less overfitting to noise)
    /// </para>
    /// </remarks>
    public T ComputeSparsityLoss(Tensor<T> attentionMask)
    {
        int batchSize = attentionMask.Shape[0];
        int numFeatures = attentionMask.Shape[1];
        var totalEntropy = _numOps.Zero;
        var epsilon = _numOps.FromDouble(1e-15); // For numerical stability

        for (int b = 0; b < batchSize; b++)
        {
            var entropy = _numOps.Zero;
            for (int f = 0; f < numFeatures; f++)
            {
                var p = attentionMask[b * numFeatures + f];
                // Add epsilon to avoid log(0)
                var pSafe = _numOps.Add(p, epsilon);
                // -p * log(p)
                var term = _numOps.Multiply(_numOps.Negate(p), _numOps.Log(pSafe));
                entropy = _numOps.Add(entropy, term);
            }
            totalEntropy = _numOps.Add(totalEntropy, entropy);
        }

        // Average over batch
        return _numOps.Divide(totalEntropy, _numOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Helper method to multiply two tensors element-wise.
    /// </summary>
    private Tensor<T> MultiplyTensors(Tensor<T> a, Tensor<T> b)
    {
        var result = new Tensor<T>(a.Shape);
        for (int i = 0; i < a.Length; i++)
        {
            result[i] = _numOps.Multiply(a[i], b[i]);
        }
        return result;
    }

    /// <summary>
    /// Performs the backward pass through the Attentive Transformer.
    /// </summary>
    /// <param name="outputGradient">The gradient flowing back.</param>
    /// <returns>The gradient with respect to the input.</returns>
    public Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_attentionMaskCache == null || _sparsemaxInputCache == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        // Backward through Sparsemax
        var sparsemaxGrad = _sparsemax.Backward(outputGradient, _attentionMaskCache, axis: 1);

        // Backward through prior scales multiplication (element-wise)
        // If z = x * y, then dL/dx = dL/dz * y
        var scaledGrad = _priorScalesCache != null
            ? MultiplyTensors(sparsemaxGrad, _priorScalesCache)
            : sparsemaxGrad;

        // Backward through batch normalization
        var bnGrad = _bnLayer.Backward(scaledGrad);

        // Backward through FC layer
        var inputGrad = _fcLayer.Backward(bnGrad);

        return inputGrad;
    }

    /// <summary>
    /// Gets all trainable parameters.
    /// </summary>
    public Vector<T> GetParameters()
    {
        var fcParams = _fcLayer.GetParameters();
        var bnParams = _bnLayer.GetParameters();
        var result = new Vector<T>(fcParams.Length + bnParams.Length);

        for (int i = 0; i < fcParams.Length; i++)
        {
            result[i] = fcParams[i];
        }
        for (int i = 0; i < bnParams.Length; i++)
        {
            result[fcParams.Length + i] = bnParams[i];
        }
        return result;
    }

    /// <summary>
    /// Sets the trainable parameters.
    /// </summary>
    public void SetParameters(Vector<T> parameters)
    {
        var fcCount = _fcLayer.ParameterCount;
        var bnCount = _bnLayer.ParameterCount;

        var fcParams = new Vector<T>(fcCount);
        var bnParams = new Vector<T>(bnCount);

        for (int i = 0; i < fcCount; i++)
        {
            fcParams[i] = parameters[i];
        }
        for (int i = 0; i < bnCount; i++)
        {
            bnParams[i] = parameters[fcCount + i];
        }

        _fcLayer.SetParameters(fcParams);
        _bnLayer.SetParameters(bnParams);
    }

    /// <summary>
    /// Gets the parameter gradients.
    /// </summary>
    public Vector<T> GetParameterGradients()
    {
        var fcGrads = _fcLayer.GetParameterGradients();
        var bnGrads = _bnLayer.GetParameterGradients();
        var result = new Vector<T>(fcGrads.Length + bnGrads.Length);

        for (int i = 0; i < fcGrads.Length; i++)
        {
            result[i] = fcGrads[i];
        }
        for (int i = 0; i < bnGrads.Length; i++)
        {
            result[fcGrads.Length + i] = bnGrads[i];
        }
        return result;
    }

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public int ParameterCount => _fcLayer.ParameterCount + _bnLayer.ParameterCount;

    /// <summary>
    /// Gets the input shape.
    /// </summary>
    public int[] GetInputShape() => [_inputDim];

    /// <summary>
    /// Gets the output shape.
    /// </summary>
    public int[] GetOutputShape() => [_outputDim];

    /// <summary>
    /// Gets the weights tensor.
    /// </summary>
    public Tensor<T>? GetWeights() => _fcLayer.GetWeights();

    /// <summary>
    /// Gets the biases tensor.
    /// </summary>
    public Tensor<T>? GetBiases() => _fcLayer.GetBiases();

    /// <summary>
    /// Gets the last computed attention mask.
    /// </summary>
    public Tensor<T>? GetAttentionMask() => _attentionMaskCache;

    /// <summary>
    /// Updates parameters using the specified learning rate.
    /// </summary>
    public void UpdateParameters(T learningRate)
    {
        _fcLayer.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Updates parameters using the specified parameter values.
    /// </summary>
    public void UpdateParameters(Vector<T> parameters)
    {
        SetParameters(parameters);
    }

    /// <summary>
    /// Clears accumulated gradients.
    /// </summary>
    public void ClearGradients()
    {
        _fcLayer.ClearGradients();
        _bnLayer.ResetGradients();
    }

    /// <summary>
    /// Resets the internal state.
    /// </summary>
    public void ResetState()
    {
        _inputCache = null;
        _priorScalesCache = null;
        _attentionMaskCache = null;
        _sparsemaxInputCache = null;
        _fcLayer.ResetState();
    }

    /// <summary>
    /// Sets training mode.
    /// </summary>
    public void SetTrainingMode(bool isTraining)
    {
        _fcLayer.SetTrainingMode(isTraining);
    }

}

using AiDotNet.Interfaces;

namespace AiDotNet.LoRA.Adapters;

/// <summary>
/// X-LoRA (Mixture of LoRA Experts) adapter that uses multiple LoRA experts with learned routing.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// X-LoRA extends standard LoRA by using a mixture of experts approach:
/// - Multiple LoRA adapters ("experts") are applied to the same layer
/// - A gating network learns to weight each expert's contribution based on the input
/// - Different inputs may activate different experts, allowing for more flexible adaptation
/// - This provides greater capacity than a single LoRA adapter with the same total rank
/// </para>
/// <para>
/// The forward pass computes:
/// - base_output = base_layer(input)
/// - For each expert i: expert_output[i] = lora_expert[i](input)
/// - gating_weights = softmax(gating_network(input))
/// - final_lora_output = sum(gating_weights[i] * expert_output[i])
/// - output = base_output + final_lora_output
/// </para>
/// <para><b>For Beginners:</b> X-LoRA is like having multiple specialists instead of one generalist.
///
/// Think of it like this:
/// - Standard LoRA: One adapter tries to handle all tasks
/// - X-LoRA: Multiple expert adapters, each specializing in different patterns
/// - A "gating network" decides which experts to use for each input
///
/// Real-world analogy: Instead of one doctor handling all patients, you have:
/// - Expert 1: Specializes in one type of pattern (e.g., cat images)
/// - Expert 2: Specializes in another pattern (e.g., dog images)
/// - Expert 3: Handles other cases
/// - Gating network: Looks at each input and decides which expert(s) to consult
///
/// Benefits:
/// - More capacity: Multiple experts can learn different aspects
/// - Better specialization: Each expert focuses on what it's good at
/// - Dynamic routing: Different inputs activate different experts
/// - Efficient: Only computes what's needed for each input
///
/// Example: For a 1000x1000 layer with 4 experts at rank=4 each:
/// - Total LoRA parameters: 4 * (4 * 1000 + 4 * 1000) = 32,000 parameters
/// - Gating network: ~1000 parameters
/// - Total: ~33,000 parameters (still 96.7% reduction from 1M!)
/// - But with more capacity than single rank=16 LoRA (32,000 params)
///
/// Trade-offs:
/// + More flexible: Experts specialize in different patterns
/// + Better performance: Often outperforms single LoRA at same parameter count
/// + Dynamic routing: Adapts to different inputs
/// - More complex: Requires training gating network
/// - Slightly slower: Must compute multiple experts and gating weights
///
/// Reference: "Mixture of LoRA Experts" (X-LoRA)
/// https://arxiv.org/abs/2402.07148
/// </para>
/// </remarks>
public class XLoRAAdapter<T> : LoRAAdapterBase<T>
{
    /// <summary>
    /// Array of LoRA expert layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each expert is a separate LoRA layer that can specialize in different patterns.
    /// The number of experts is typically 2-8, balancing capacity and computational cost.
    /// </para>
    /// <para><b>For Beginners:</b> These are the specialist adapters. Each one learns
    /// to handle different types of inputs. The gating network decides which experts
    /// to use for each input.
    /// </para>
    /// </remarks>
    private readonly LoRALayer<T>[] _experts;

    /// <summary>
    /// Gating network that computes expert weights for each input.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The gating network is a small neural network (typically a single dense layer with softmax)
    /// that takes the input and produces a probability distribution over experts.
    /// These probabilities determine how much each expert contributes to the output.
    /// </para>
    /// <para><b>For Beginners:</b> This is the "decision maker" that looks at each input
    /// and decides which experts should handle it. It outputs a weight for each expert
    /// (weights sum to 1.0), indicating how much to trust each expert's advice.
    /// </para>
    /// </remarks>
    private readonly DenseLayer<T> _gatingNetwork;

    /// <summary>
    /// Gets the number of LoRA experts in this adapter.
    /// </summary>
    public int NumberOfExpertss => _experts.Length;

    /// <summary>
    /// Gets the array of LoRA expert layers.
    /// </summary>
    /// <remarks>
    /// Returns a copy of the experts array to prevent external modification.
    /// </remarks>
    public LoRALayer<T>[] Experts => (LoRALayer<T>[])_experts.Clone();

    /// <summary>
    /// Gets the gating network used for routing.
    /// </summary>
    public DenseLayer<T> GatingNetwork => _gatingNetwork;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    /// <remarks>
    /// Includes parameters from:
    /// - Base layer (if not frozen)
    /// - All expert LoRA layers
    /// - Gating network
    /// </remarks>
    public override int ParameterCount
    {
        get
        {
            int expertParams = 0;
            for (int i = 0; i < _experts.Length; i++)
            {
                expertParams += _experts[i].ParameterCount;
            }

            int gatingParams = _gatingNetwork.ParameterCount;
            int baseParams = _freezeBaseLayer ? 0 : _baseLayer.ParameterCount;

            return baseParams + expertParams + gatingParams;
        }
    }

    /// <summary>
    /// Temporary storage for expert outputs during forward pass (needed for backward pass).
    /// </summary>
    private Tensor<T>[]? _lastExpertOutputs;

    /// <summary>
    /// Temporary storage for gating weights during forward pass (needed for backward pass).
    /// </summary>
    private Tensor<T>? _lastGatingWeights;

    /// <summary>
    /// Temporary storage for the last input during forward pass (needed for backward pass).
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Initializes a new X-LoRA adapter with the specified parameters.
    /// </summary>
    /// <param name="baseLayer">The layer to adapt with X-LoRA.</param>
    /// <param name="numberOfExperts">The number of LoRA experts to create.</param>
    /// <param name="expertRank">The rank of each LoRA expert decomposition.</param>
    /// <param name="alpha">The LoRA scaling factor for experts (defaults to expertRank if negative).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer's parameters during training.</param>
    /// <exception cref="ArgumentNullException">Thrown when baseLayer is null.</exception>
    /// <exception cref="ArgumentException">Thrown when numberOfExperts is invalid.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates an X-LoRA adapter with multiple expert adapters.
    ///
    /// Parameters:
    /// - baseLayer: The layer you want to adapt (typically Dense or FullyConnected)
    /// - numberOfExperts: How many specialist adapters to create (typically 2-8)
    /// - expertRank: The rank for each expert (compression level)
    /// - alpha: How strong each expert's adaptation is
    /// - freezeBaseLayer: Whether to lock the original layer's weights (usually true)
    ///
    /// The adapter will:
    /// 1. Create multiple LoRA experts (all with the same rank)
    /// 2. Create a gating network to route inputs to experts
    /// 3. Learn to specialize each expert for different patterns
    ///
    /// Common configurations:
    /// - numberOfExperts=2, expertRank=8: Simple mixture for binary specialization
    /// - numberOfExperts=4, expertRank=4: Balanced approach (4 specialists, 16 total rank)
    /// - numberOfExperts=8, expertRank=2: Many specialists, each handling narrow patterns
    ///
    /// Trade-off: More experts = more specialization but more parameters and computation.
    /// </para>
    /// </remarks>
    public XLoRAAdapter(
        ILayer<T> baseLayer,
        int numberOfExperts,
        int expertRank,
        double alpha = -1,
        bool freezeBaseLayer = true)
        : base(baseLayer, expertRank, alpha, freezeBaseLayer)
    {
        if (numberOfExperts < 2)
        {
            throw new ArgumentException("Number of experts must be at least 2", nameof(numberOfExperts));
        }

        // Create expert LoRA layers
        _experts = new LoRALayer<T>[numberOfExperts];
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];

        for (int i = 0; i < numberOfExperts; i++)
        {
            _experts[i] = new LoRALayer<T>(inputSize, outputSize, expertRank, alpha);
        }

        // Create gating network: input -> numberOfExperts (with softmax activation)
        // The gating network is a simple dense layer that maps input to expert weights
        _gatingNetwork = new DenseLayer<T>(inputSize, numberOfExperts, (IVectorActivationFunction<T>)new SoftmaxActivation<T>());

        // Update parameter vector to include all experts and gating network
        Parameters = new Vector<T>(ParameterCount);
        UpdateParametersFromLayers();
    }

    /// <summary>
    /// Performs the forward pass using mixture of LoRA experts.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output combining base layer and weighted expert outputs.</returns>
    /// <remarks>
    /// <para>
    /// The forward pass:
    /// 1. Computes base layer output
    /// 2. Computes gating weights from gating network (determines expert contributions)
    /// 3. Computes output from each expert
    /// 4. Combines expert outputs using gating weights (weighted sum)
    /// 5. Returns base_output + weighted_expert_output
    /// </para>
    /// <para><b>For Beginners:</b> This is where the magic happens!
    ///
    /// Process:
    /// 1. Run input through base layer (original behavior)
    /// 2. Run input through gating network to get expert weights
    ///    - Example: [0.6, 0.3, 0.1, 0.0] means mostly use expert 1, some expert 2
    /// 3. Run input through all experts to get their opinions
    /// 4. Combine expert outputs using weights (weighted average)
    /// 5. Add combined expert output to base output
    ///
    /// The gating weights ensure that:
    /// - Relevant experts contribute more (high weights)
    /// - Irrelevant experts contribute less (low weights)
    /// - All weights sum to 1.0 (thanks to softmax in gating network)
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input.Clone();

        // Forward through base layer
        Tensor<T> baseOutput = _baseLayer.Forward(input);

        // Compute gating weights for this input
        // The gating network outputs a probability distribution over experts
        Tensor<T> gatingWeights = _gatingNetwork.Forward(input);
        _lastGatingWeights = gatingWeights.Clone();

        // Forward through all experts and store their outputs
        _lastExpertOutputs = new Tensor<T>[_experts.Length];
        Tensor<T> combinedExpertOutput = new Tensor<T>(baseOutput.Shape);

        // Initialize combined output to zero
        for (int i = 0; i < combinedExpertOutput.Length; i++)
        {
            combinedExpertOutput[i] = NumOps.Zero;
        }

        // Get batch size for proper indexing
        int batchSize = input.Shape[0];
        int outputDim = baseOutput.Length / batchSize;

        // Compute weighted sum of expert outputs
        for (int expertIdx = 0; expertIdx < _experts.Length; expertIdx++)
        {
            // Forward through this expert
            Tensor<T> expertOutput = _experts[expertIdx].Forward(input);
            _lastExpertOutputs[expertIdx] = expertOutput.Clone();

            // Weight each sample's expert output by its corresponding gating weight
            for (int batchIdx = 0; batchIdx < batchSize; batchIdx++)
            {
                T gatingWeight = gatingWeights[batchIdx * _experts.Length + expertIdx];

                for (int dimIdx = 0; dimIdx < outputDim; dimIdx++)
                {
                    int outputIdx = batchIdx * outputDim + dimIdx;
                    T weightedValue = NumOps.Multiply(expertOutput[outputIdx], gatingWeight);
                    combinedExpertOutput[outputIdx] = NumOps.Add(combinedExpertOutput[outputIdx], weightedValue);
                }
            }
        }

        // Sum base output and combined expert output
        Tensor<T> result = new Tensor<T>(baseOutput.Shape);
        for (int i = 0; i < baseOutput.Length; i++)
        {
            result[i] = NumOps.Add(baseOutput[i], combinedExpertOutput[i]);
        }

        return result;
    }

    /// <summary>
    /// Performs the backward pass through the mixture of experts.
    /// </summary>
    /// <param name="outputGradient">Gradient flowing back from the next layer.</param>
    /// <returns>Gradient to pass to the previous layer.</returns>
    /// <remarks>
    /// <para>
    /// The backward pass propagates gradients through:
    /// 1. All expert LoRA layers (weighted by their gating weights)
    /// 2. The gating network (to learn better routing)
    /// 3. The base layer (if not frozen)
    /// </para>
    /// <para><b>For Beginners:</b> This is where all components learn to improve!
    ///
    /// During backpropagation:
    /// 1. Each expert receives gradients weighted by how much it was used
    ///    - Expert with weight 0.6 gets 60% of the gradient
    ///    - Expert with weight 0.1 gets 10% of the gradient
    /// 2. The gating network learns to route inputs better
    ///    - If an expert's output helped, increase its weight next time
    ///    - If an expert's output hurt, decrease its weight
    /// 3. The base layer updates if not frozen
    ///
    /// This creates a feedback loop where:
    /// - Experts specialize in patterns they're good at
    /// - Gating network learns which expert to use for which input
    /// - Together, they improve performance beyond single LoRA
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastGatingWeights == null || _lastExpertOutputs == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass");
        }

        int batchSize = _lastInput.Shape[0];
        int outputDim = outputGradient.Length / batchSize;
        int inputDim = _lastInput.Shape.Length > 1 ? _lastInput.Shape[1] : _lastInput.Length / batchSize;

        // Backward through base layer
        Tensor<T> baseInputGrad = _baseLayer.Backward(outputGradient);

        // Backward through experts
        // Each expert receives a gradient weighted by its gating weight
        Tensor<T> expertInputGradSum = new Tensor<T>(new[] { batchSize, inputDim });
        for (int i = 0; i < expertInputGradSum.Length; i++)
        {
            expertInputGradSum[i] = NumOps.Zero;
        }

        // Gradient w.r.t. gating weights (for gating network backprop)
        Tensor<T> gatingGradient = new Tensor<T>(new[] { batchSize, _experts.Length });
        for (int i = 0; i < gatingGradient.Length; i++)
        {
            gatingGradient[i] = NumOps.Zero;
        }

        // Backward through each expert
        for (int expertIdx = 0; expertIdx < _experts.Length; expertIdx++)
        {
            // Weight the output gradient by this expert's gating weight for each sample
            Tensor<T> weightedOutputGrad = new Tensor<T>(outputGradient.Shape);

            for (int batchIdx = 0; batchIdx < batchSize; batchIdx++)
            {
                T gatingWeight = _lastGatingWeights[batchIdx * _experts.Length + expertIdx];

                for (int dimIdx = 0; dimIdx < outputDim; dimIdx++)
                {
                    int outputIdx = batchIdx * outputDim + dimIdx;
                    weightedOutputGrad[outputIdx] = NumOps.Multiply(outputGradient[outputIdx], gatingWeight);
                }

                // Compute gradient w.r.t. gating weight for this expert
                // dL/dg[i] = sum over output dims of (outputGradient * expertOutput)
                T gatingGrad = NumOps.Zero;
                for (int dimIdx = 0; dimIdx < outputDim; dimIdx++)
                {
                    int outputIdx = batchIdx * outputDim + dimIdx;
                    T grad = NumOps.Multiply(outputGradient[outputIdx], _lastExpertOutputs[expertIdx][outputIdx]);
                    gatingGrad = NumOps.Add(gatingGrad, grad);
                }
                gatingGradient[batchIdx * _experts.Length + expertIdx] = gatingGrad;
            }

            // Backward through this expert
            Tensor<T> expertInputGrad = _experts[expertIdx].Backward(weightedOutputGrad);

            // Accumulate input gradients
            for (int i = 0; i < expertInputGrad.Length; i++)
            {
                expertInputGradSum[i] = NumOps.Add(expertInputGradSum[i], expertInputGrad[i]);
            }
        }

        // Backward through gating network
        Tensor<T> gatingInputGrad = _gatingNetwork.Backward(gatingGradient);

        // Sum all input gradients (base + experts + gating)
        Tensor<T> totalInputGrad = new Tensor<T>(baseInputGrad.Shape);
        for (int i = 0; i < baseInputGrad.Length; i++)
        {
            T sum = NumOps.Add(baseInputGrad[i], expertInputGradSum[i]);
            totalInputGrad[i] = NumOps.Add(sum, gatingInputGrad[i]);
        }

        // Update parameter gradients vector
        UpdateParameterGradientsFromLayers();

        return totalInputGrad;
    }

    /// <summary>
    /// Updates parameters using the specified learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    /// <remarks>
    /// Updates all experts, the gating network, and optionally the base layer.
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // Update all experts
        for (int i = 0; i < _experts.Length; i++)
        {
            _experts[i].UpdateParameters(learningRate);
        }

        // Update gating network
        _gatingNetwork.UpdateParameters(learningRate);

        // Only update base layer if not frozen
        if (!_freezeBaseLayer)
        {
            _baseLayer.UpdateParameters(learningRate);
        }

        // Update parameter vector
        UpdateParametersFromLayers();
    }

    /// <summary>
    /// Gets the current parameters as a vector.
    /// </summary>
    /// <returns>Vector containing parameters from all experts, gating network, and optionally base layer.</returns>
    public override Vector<T> GetParameters()
    {
        return Parameters.Clone();
    }

    /// <summary>
    /// Sets the layer parameters from a vector.
    /// </summary>
    /// <param name="parameters">Vector containing parameters for all components.</param>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException($"Expected {ParameterCount} parameters, got {parameters.Length}", nameof(parameters));
        }

        Parameters = parameters.Clone();
        UpdateLayersFromParameters();
    }

    /// <summary>
    /// Updates the parameter vector from the current layer states.
    /// </summary>
    protected override void UpdateParametersFromLayers()
    {
        int idx = 0;

        // If base layer is not frozen, pack its parameters first
        if (!_freezeBaseLayer)
        {
            Vector<T> baseParams = _baseLayer.GetParameters();
            for (int i = 0; i < baseParams.Length; i++)
            {
                Parameters[idx++] = baseParams[i];
            }
        }

        // Pack expert parameters
        for (int expertIdx = 0; expertIdx < _experts.Length; expertIdx++)
        {
            Vector<T> expertParams = _experts[expertIdx].GetParameters();
            for (int i = 0; i < expertParams.Length; i++)
            {
                Parameters[idx++] = expertParams[i];
            }
        }

        // Pack gating network parameters
        Vector<T> gatingParams = _gatingNetwork.GetParameters();
        for (int i = 0; i < gatingParams.Length; i++)
        {
            Parameters[idx++] = gatingParams[i];
        }
    }

    /// <summary>
    /// Updates the layers from the parameter vector.
    /// </summary>
    private void UpdateLayersFromParameters()
    {
        int idx = 0;

        // If base layer is not frozen, unpack its parameters first
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

        // Unpack expert parameters
        for (int expertIdx = 0; expertIdx < _experts.Length; expertIdx++)
        {
            int expertParamCount = _experts[expertIdx].ParameterCount;
            Vector<T> expertParams = new Vector<T>(expertParamCount);
            for (int i = 0; i < expertParamCount; i++)
            {
                expertParams[i] = Parameters[idx++];
            }
            _experts[expertIdx].SetParameters(expertParams);
        }

        // Unpack gating network parameters
        int gatingParamCount = _gatingNetwork.ParameterCount;
        Vector<T> gatingParams = new Vector<T>(gatingParamCount);
        for (int i = 0; i < gatingParamCount; i++)
        {
            gatingParams[i] = Parameters[idx++];
        }
        _gatingNetwork.SetParameters(gatingParams);
    }

    /// <summary>
    /// Updates the parameter gradients vector from the layer gradients.
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

        // Pack expert gradients
        for (int expertIdx = 0; expertIdx < _experts.Length; expertIdx++)
        {
            Vector<T> expertGrads = _experts[expertIdx].GetParameterGradients();
            for (int i = 0; i < expertGrads.Length; i++)
            {
                ParameterGradients[idx++] = expertGrads[i];
            }
        }

        // Pack gating network gradients
        Vector<T> gatingGrads = _gatingNetwork.GetParameterGradients();
        for (int i = 0; i < gatingGrads.Length; i++)
        {
            ParameterGradients[idx++] = gatingGrads[i];
        }
    }

    /// <summary>
    /// Merges all LoRA expert adaptations into the base layer and returns the merged layer.
    /// </summary>
    /// <returns>A new layer with all expert adaptations merged into the base layer's weights.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the base layer type is not DenseLayer or FullyConnectedLayer.</exception>
    /// <remarks>
    /// <para>
    /// Since X-LoRA uses input-dependent gating, the merge averages all expert contributions.
    /// This provides a reasonable approximation but loses the dynamic routing capability.
    /// For deployment, consider keeping the full X-LoRA structure if dynamic routing is important.
    /// </para>
    /// <para><b>For Beginners:</b> This "bakes in" all expert adaptations to create a regular layer.
    ///
    /// Important caveat: X-LoRA's strength is dynamic routing (different experts for different inputs).
    /// When we merge:
    /// 1. We average all expert contributions (equal weighting)
    /// 2. We lose the dynamic routing capability
    /// 3. The result is a static layer that works okay but not as well as the full X-LoRA
    ///
    /// Use this for:
    /// - Simpler deployment when dynamic routing isn't critical
    /// - Compatibility with systems that don't support X-LoRA
    /// - Reducing inference complexity
    ///
    /// DON'T use this if:
    /// - Dynamic routing is important for your task
    /// - Different inputs need very different adaptations
    /// - You want maximum performance
    ///
    /// Better approach for deployment: Keep the full X-LoRA structure and implement efficient inference.
    /// </para>
    /// </remarks>
    public override ILayer<T> MergeToOriginalLayer()
    {
        // Support both DenseLayer and FullyConnectedLayer
        DenseLayer<T>? denseBase = _baseLayer as DenseLayer<T>;
        FullyConnectedLayer<T>? fcBase = _baseLayer as FullyConnectedLayer<T>;

        if (denseBase == null && fcBase == null)
        {
            throw new InvalidOperationException("XLoRAAdapter merging only supports DenseLayer or FullyConnectedLayer base layers");
        }

        // Get base layer parameters
        Vector<T> baseParams = _baseLayer.GetParameters();

        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        int weightCount = inputSize * outputSize;

        // Create new parameters with merged weights
        Vector<T> mergedParams = new Vector<T>(baseParams.Length);

        // Average all expert contributions (since we don't have input-specific gating at merge time)
        T expertScaling = NumOps.Divide(NumOps.One, NumOps.FromDouble(_experts.Length));

        // Start with base weights
        for (int i = 0; i < baseParams.Length; i++)
        {
            mergedParams[i] = baseParams[i];
        }

        // Add averaged expert contributions
        for (int expertIdx = 0; expertIdx < _experts.Length; expertIdx++)
        {
            Matrix<T> expertWeights = _experts[expertIdx].MergeWeights();

            for (int i = 0; i < weightCount; i++)
            {
                int row = i / inputSize;
                int col = i % inputSize;
                T scaledExpertWeight = NumOps.Multiply(expertWeights[row, col], expertScaling);
                mergedParams[i] = NumOps.Add(mergedParams[i], scaledExpertWeight);
            }
        }

        // Use helper method to clone base layer and preserve activation function
        return CreateMergedLayerWithClone(mergedParams);
    }

    /// <summary>
    /// Resets the internal state of the base layer, all experts, and the gating network.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This clears the memory of all components (base layer,
    /// all experts, and gating network). It's useful when starting to process a completely
    /// new, unrelated batch of data.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        _baseLayer.ResetState();
        _loraLayer.ResetState();

        for (int i = 0; i < _experts.Length; i++)
        {
            _experts[i].ResetState();
        }

        _gatingNetwork.ResetState();

        _lastInput = null;
        _lastGatingWeights = null;
        _lastExpertOutputs = null;
    }

    /// <summary>
    /// Gets the gating weights from the last forward pass.
    /// </summary>
    /// <returns>Tensor containing gating weights for each sample and expert.</returns>
    /// <remarks>
    /// <para>
    /// This is useful for analyzing which experts are being used for different inputs.
    /// The weights are per-sample probabilities summing to 1.0 across experts.
    /// </para>
    /// <para><b>For Beginners:</b> This shows you which experts the gating network chose
    /// for the last batch of inputs. High values mean that expert was important, low values
    /// mean it wasn't used much.
    ///
    /// Example interpretation:
    /// - Sample 1: [0.7, 0.2, 0.1, 0.0] -> Mostly expert 1, some expert 2
    /// - Sample 2: [0.0, 0.1, 0.8, 0.1] -> Mostly expert 3
    ///
    /// This helps you understand:
    /// - Which experts specialize in which patterns
    /// - Whether routing is working correctly
    /// - If some experts are underutilized (might reduce number of experts)
    /// </para>
    /// </remarks>
    public Tensor<T>? GetLastGatingWeights()
    {
        return _lastGatingWeights?.Clone();
    }
}

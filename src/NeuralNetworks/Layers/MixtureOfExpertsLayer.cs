using AiDotNet.Autodiff;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements a Mixture-of-Experts (MoE) layer that routes inputs through multiple expert networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// A Mixture-of-Experts layer contains multiple expert networks and a gating/routing network.
/// For each input, the router determines how much weight to give each expert's output,
/// allowing the model to specialize different experts for different types of inputs.
/// This architecture enables models with very high capacity while remaining computationally efficient
/// by activating only a subset of parameters per input.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of a Mixture-of-Experts as a team of specialists working together.
///
/// How it works:
/// - You have multiple "experts" (specialized neural networks)
/// - A "router" (gating network) decides which experts should handle each input
/// - Each expert processes the input independently
/// - The final output is a weighted combination of the experts' outputs
///
/// Why use MoE:
/// - Scalability: Add more experts to increase model capacity without proportionally increasing computation
/// - Specialization: Different experts learn to handle different types of inputs
/// - Efficiency: Only activate the most relevant experts for each input (sparse MoE)
///
/// Real-world analogy:
/// Imagine you're running a hospital with specialists:
/// - A cardiologist (expert 1) handles heart problems
/// - A neurologist (expert 2) handles brain issues
/// - A pediatrician (expert 3) handles children's health
/// - A triage nurse (router) directs patients to the right specialist(s)
///
/// The router learns to send cardiac patients to the cardiologist, neurological cases to the
/// neurologist, etc. This is more efficient than having one doctor handle everything, and allows
/// each specialist to become highly skilled in their area.
/// </para>
/// <para>
/// <b>Key Features:</b>
/// <list type="bullet">
/// <item><description>Support for any number of experts</description></item>
/// <item><description>Learned routing via a dense gating network</description></item>
/// <item><description>Softmax routing: All experts contribute with learned weights</description></item>
/// <item><description>Top-K routing: Only the top K experts are activated per input</description></item>
/// <item><description>Load balancing: Optional auxiliary loss to encourage balanced expert usage</description></item>
/// </list>
/// </para>
/// </remarks>
public class MixtureOfExpertsLayer<T> : LayerBase<T>, IAuxiliaryLossLayer<T>
{
    /// <summary>
    /// The collection of expert networks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each expert is an independent neural network that can process inputs. Experts may have
    /// different architectures or the same architecture with different learned parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This is the list of specialist networks.
    ///
    /// Each expert:
    /// - Has its own weights and biases
    /// - Can learn to specialize in different patterns
    /// - Processes inputs independently from other experts
    ///
    /// You might have 4, 8, 16, or even hundreds of experts depending on your needs.
    /// More experts = more capacity, but also more memory and computation.
    /// </para>
    /// </remarks>
    private readonly List<ILayer<T>> _experts;

    /// <summary>
    /// The router/gating network that determines how to weight each expert's output.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The router is typically a dense layer that takes the input and produces a score for each expert.
    /// These scores are then converted to weights (probabilities) using softmax, determining how much
    /// each expert contributes to the final output.
    /// </para>
    /// <para><b>For Beginners:</b> This is the "manager" that decides which experts should handle each input.
    ///
    /// The router:
    /// - Looks at each input
    /// - Produces a score for each expert
    /// - Scores are converted to weights (probabilities that sum to 1)
    /// - Higher weight = that expert's output matters more for this input
    ///
    /// During training, the router learns patterns like:
    /// - "If the input has property X, send it mostly to expert 3"
    /// - "If the input looks like Y, use experts 1 and 5 equally"
    /// - "For input type Z, expert 2 is the best choice"
    /// </para>
    /// </remarks>
    private readonly ILayer<T> _router;

    /// <summary>
    /// The number of top experts to activate for each input (0 means use all experts).
    /// </summary>
    /// <remarks>
    /// <para>
    /// When TopK is 0, all experts process every input (soft routing).
    /// When TopK > 0, only the K experts with the highest routing scores process each input (sparse routing).
    /// Sparse routing significantly reduces computation while maintaining model quality.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how many experts are used for each input.
    ///
    /// TopK = 0 (all experts):
    /// - Every expert processes every input
    /// - More computation but potentially more accurate
    /// - Good for smaller models or when you need maximum quality
    ///
    /// TopK = 2 (sparse, use top 2):
    /// - Only the 2 best experts for each input are activated
    /// - Much faster and more memory efficient
    /// - Good for large models where you can't afford to run all experts
    ///
    /// For example, with 8 experts and TopK=2:
    /// - The router scores all 8 experts
    /// - Only the top 2 are actually used
    /// - The other 6 are skipped for that input
    /// - This means 75% less computation!
    /// </para>
    /// </remarks>
    private readonly int _topK;

    /// <summary>
    /// Cached input from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This cached value is used during backpropagation to compute gradients for the router.
    /// </para>
    /// <para><b>For Beginners:</b> Stored memory of the last input, needed for learning.
    ///
    /// Why cache this:
    /// - During forward pass, we process the input
    /// - During backward pass, we need to remember what we processed
    /// - This helps calculate how to improve the router and experts
    ///
    /// Think of it like taking notes during a lecture - you can't learn from the lecture
    /// later if you don't remember what was said.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Cached routing weights from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These are the normalized probability weights (after softmax) that determined how much
    /// each expert contributed to the output. Shape: [batch_size, num_experts].
    /// </para>
    /// <para><b>For Beginners:</b> Stored memory of which experts were used and how much.
    ///
    /// For each input in the batch, we remember:
    /// - How much weight expert 1 had
    /// - How much weight expert 2 had
    /// - And so on for all experts
    ///
    /// This helps us learn:
    /// - If routing decisions were good or bad
    /// - How to adjust the router to make better decisions
    /// - Whether experts are being used balanced or if some are overloaded
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastRoutingWeights;

    /// <summary>
    /// Cached expert outputs from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This stores the output from each expert before they are combined. During backpropagation,
    /// these are used to compute gradients for both the router and the experts themselves.
    /// </para>
    /// <para><b>For Beginners:</b> Stored memory of what each expert predicted.
    ///
    /// We save each expert's output so we can:
    /// - Calculate how much each expert contributed to any errors
    /// - Adjust the router if it chose the wrong experts
    /// - Adjust each expert to improve its predictions
    ///
    /// It's like keeping a record of each team member's contribution to a group project,
    /// so you can give appropriate feedback to each person.
    /// </para>
    /// </remarks>
    private List<Tensor<T>>? _lastExpertOutputs;

    /// <summary>
    /// Cached routing logits (before softmax) from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These are the raw scores from the router before applying softmax. They're needed for
    /// computing the gradient of softmax during backpropagation.
    /// </para>
    /// <para><b>For Beginners:</b> The router's raw scores before normalization.
    ///
    /// The router produces raw scores (logits), then we convert them to probabilities (weights).
    /// We save the raw scores because:
    /// - The conversion process (softmax) is nonlinear
    /// - To compute gradients correctly, we need the pre-conversion values
    /// - This is a technical requirement for proper backpropagation through softmax
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastRoutingLogits;

    /// <summary>
    /// Cached top-K indices for sparse routing from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When using Top-K routing, this stores which K experts were selected for each input.
    /// Shape: [batch_size, K]. Null when using all experts (TopK = 0).
    /// </para>
    /// <para><b>For Beginners:</b> A record of which experts were actually used for each input.
    ///
    /// With sparse routing (TopK > 0), we save which experts were chosen:
    /// - Input 1 used experts [2, 5]
    /// - Input 2 used experts [1, 3]
    /// - And so on
    ///
    /// This is important because:
    /// - Only activated experts need to compute gradients
    /// - We need to know which experts to update during learning
    /// - It helps track if certain experts are being used too much or too little
    /// </para>
    /// </remarks>
    private int[,]? _lastTopKIndices;

    /// <summary>
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;

    /// <summary>
    /// Indicates whether to compute and use the auxiliary load balancing loss.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When enabled, the layer computes a load balancing loss that encourages balanced expert usage.
    /// This is typically enabled during training but disabled during inference.
    /// </para>
    /// <para><b>For Beginners:</b> A switch to turn load balancing on or off.
    ///
    /// Load balancing helps ensure all experts are used roughly equally:
    /// - Prevents some experts from being overused
    /// - Prevents other experts from being underused
    /// - Leads to better overall model performance
    ///
    /// Usually you want this ON during training and OFF during inference/testing.
    /// </para>
    /// </remarks>
    private bool _useAuxiliaryLoss;

    /// <summary>
    /// The weight for the auxiliary load balancing loss.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This coefficient controls how much the load balancing objective influences training.
    /// Typical values range from 0.01 to 0.1.
    /// </para>
    /// <para><b>For Beginners:</b> Controls how important load balancing is.
    ///
    /// Typical values:
    /// - 0.01: Gentle encouragement for balance
    /// - 0.05: Moderate encouragement (recommended starting point)
    /// - 0.1: Strong encouragement for balance
    ///
    /// Higher values make experts more balanced but might reduce accuracy slightly.
    /// Lower values prioritize accuracy over balance.
    /// </para>
    /// </remarks>
    private T _auxiliaryLossWeight;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// <c>true</c> if the router or any expert supports training; otherwise, <c>false</c>.
    /// </value>
    /// <remarks>
    /// <para>
    /// The MoE layer supports training if either its router or any of its experts have trainable parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you if the MoE layer can learn from data.
    ///
    /// The layer can learn if:
    /// - The router can learn better routing decisions
    /// - Any expert can improve its predictions
    ///
    /// In almost all practical cases, this will be true since both the router and experts
    /// typically have trainable parameters.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _router.SupportsTraining || _experts.Any(e => e.SupportsTraining);

    /// <summary>
    /// Gets the total number of trainable parameters in the layer.
    /// </summary>
    /// <value>
    /// The sum of the router's parameters and all experts' parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// This includes all parameters from the router and all experts combined. This gives you the
    /// total model capacity and memory requirement for this layer.
    /// </para>
    /// <para><b>For Beginners:</b> The total count of all adjustable numbers in this layer.
    ///
    /// This includes:
    /// - All weights and biases in the router
    /// - All weights and biases in all experts
    ///
    /// For example, with:
    /// - Router: 1000 parameters
    /// - 8 experts with 5000 parameters each: 40,000 parameters
    /// - Total: 41,000 parameters
    ///
    /// More parameters = more capacity to learn, but also more memory needed.
    /// MoE shines because you can have huge capacity (many experts) but still only activate
    /// a fraction of them per input with sparse routing.
    /// </para>
    /// </remarks>
    public override int ParameterCount => _router.ParameterCount + _experts.Sum(e => e.ParameterCount);

    /// <summary>
    /// Gets the number of experts in this MoE layer.
    /// </summary>
    /// <value>
    /// The count of expert networks.
    /// </value>
    /// <remarks>
    /// <para>
    /// This is the total number of expert networks, regardless of how many are activated per input.
    /// </para>
    /// <para><b>For Beginners:</b> How many specialist networks are available.
    ///
    /// Common configurations:
    /// - Small models: 4-8 experts
    /// - Medium models: 8-16 experts
    /// - Large models: 32-128+ experts
    ///
    /// The number of experts affects:
    /// - Model capacity (more experts = more capacity)
    /// - Memory usage (more experts = more memory)
    /// - Specialization potential (more experts = more specialized roles)
    /// </para>
    /// </remarks>
    public int NumExperts => _experts.Count;

    /// <summary>
    /// Initializes a new instance of the <see cref="MixtureOfExpertsLayer{T}"/> class.
    /// </summary>
    /// <param name="experts">The list of expert networks.</param>
    /// <param name="router">The routing/gating network.</param>
    /// <param name="inputShape">The shape of input tensors.</param>
    /// <param name="outputShape">The shape of output tensors.</param>
    /// <param name="topK">Number of experts to activate per input (0 = use all experts). Default is 0.</param>
    /// <param name="activationFunction">Optional activation function to apply after combining expert outputs.</param>
    /// <exception cref="ArgumentException">Thrown when the experts list is empty or when topK is invalid.</exception>
    /// <remarks>
    /// <para>
    /// Creates a Mixture-of-Experts layer with the specified experts and router. All experts should
    /// have compatible input/output shapes. The router should output a tensor with one value per expert.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new MoE layer with your chosen experts and router.
    ///
    /// To create an MoE layer:
    /// 1. Create your expert networks (can be any ILayer&lt;T&gt;, often Expert&lt;T&gt; or DenseLayer&lt;T&gt;)
    /// 2. Create a router (typically a DenseLayer that outputs numExperts values)
    /// 3. Specify input/output shapes
    /// 4. Optionally set topK for sparse routing
    ///
    /// Example - MoE with 4 experts and Top-2 routing:
    /// <code>
    /// // Create 4 expert networks
    /// var experts = new List&lt;ILayer&lt;float&gt;&gt;();
    /// for (int i = 0; i &lt; 4; i++)
    /// {
    ///     var expertLayers = new List&lt;ILayer&lt;float&gt;&gt;
    ///     {
    ///         new DenseLayer&lt;float&gt;(128, 256, new ReLUActivation&lt;float&gt;()),
    ///         new DenseLayer&lt;float&gt;(256, 128, new ReLUActivation&lt;float&gt;())
    ///     };
    ///     experts.Add(new ExpertLayer&lt;float&gt;(expertLayers, new[] { 128 }, new[] { 128 }));
    /// }
    ///
    /// // Create router that outputs 4 scores (one per expert)
    /// var router = new DenseLayer&lt;float&gt;(128, 4);
    ///
    /// // Create MoE layer with Top-2 routing
    /// var moe = new MixtureOfExpertsLayer&lt;float&gt;(
    ///     experts, router,
    ///     new[] { 128 }, new[] { 128 },
    ///     topK: 2
    /// );
    /// </code>
    /// </para>
    /// </remarks>
    public MixtureOfExpertsLayer(
        List<ILayer<T>> experts,
        ILayer<T> router,
        int[] inputShape,
        int[] outputShape,
        int topK = 0,
        IActivationFunction<T>? activationFunction = null,
        bool useLoadBalancing = false,
        T? loadBalancingWeight = default)
        : base(inputShape, outputShape, activationFunction ?? new IdentityActivation<T>())
    {
        if (experts == null || experts.Count == 0)
        {
            throw new ArgumentException("Must have at least one expert.", nameof(experts));
        }

        if (router == null)
        {
            throw new ArgumentNullException(nameof(router), "Router cannot be null.");
        }

        if (topK < 0 || topK > experts.Count)
        {
            throw new ArgumentException(
                $"TopK must be between 0 and {experts.Count} (number of experts). Got {topK}.",
                nameof(topK));
        }

        _experts = experts;
        _router = router;
        _topK = topK;
        _useAuxiliaryLoss = useLoadBalancing;
        _auxiliaryLossWeight = loadBalancingWeight ?? NumOps.FromDouble(0.01); // Default to 0.01
    }

    /// <summary>
    /// Performs the forward pass through the MoE layer.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The output tensor after routing through experts and combining their outputs.</returns>
    /// <remarks>
    /// <para>
    /// The forward pass:
    /// 1. Routes the input through the gating network to get expert scores
    /// 2. Applies softmax to convert scores to routing probabilities
    /// 3. Optionally selects only top-K experts (sparse routing)
    /// 4. Passes input through selected experts
    /// 5. Combines expert outputs using routing weights
    /// 6. Applies the layer's activation function
    /// </para>
    /// <para><b>For Beginners:</b> This is where the MoE layer processes input data.
    ///
    /// Step-by-step process:
    ///
    /// 1. <b>Routing:</b> The router looks at the input and scores each expert
    ///    - Input: data to process
    ///    - Output: a score for each expert (raw numbers)
    ///
    /// 2. <b>Normalization:</b> Convert scores to probabilities using softmax
    ///    - Scores: might be [2.1, -0.5, 1.3, 0.8]
    ///    - Weights: becomes [0.55, 0.04, 0.26, 0.15] (sum = 1.0)
    ///
    /// 3. <b>Selection (if using Top-K):</b> Keep only the best K experts
    ///    - With Top-2, keep experts with weights 0.55 and 0.26
    ///    - Set others to 0 and renormalize: [0.68, 0, 0.32, 0]
    ///
    /// 4. <b>Expert Processing:</b> Run input through selected experts
    ///    - Expert 1 produces output A
    ///    - Expert 3 produces output B
    ///    - Others are skipped (if using Top-K)
    ///
    /// 5. <b>Combination:</b> Mix expert outputs using weights
    ///    - Output = 0.68 * A + 0.32 * B
    ///    - This is the weighted average of expert predictions
    ///
    /// 6. <b>Activation:</b> Apply final transformation
    ///    - Usually identity (no change) or ReLU
    ///
    /// The result is a smart combination of expert predictions, where each expert
    /// contributes based on its relevance to the specific input.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Store original shape for any-rank tensor support
        _originalInputShape = input.Shape;
        int rank = input.Shape.Length;

        // Handle any-rank tensor: collapse to 2D for processing
        Tensor<T> input2D;
        int batchSize;

        if (rank == 1)
        {
            // 1D: [features] -> add batch dim
            batchSize = 1;
            int featureSize = input.Shape[0];
            input2D = input.Reshape(new[] { 1, featureSize });
        }
        else if (rank == 2)
        {
            // Standard 2D: [batch, features]
            batchSize = input.Shape[0];
            input2D = input;
        }
        else
        {
            // Higher-rank: collapse all leading dims into batch
            int flatBatch = 1;
            for (int d = 0; d < rank - 1; d++)
                flatBatch *= input.Shape[d];
            batchSize = flatBatch;
            int featureSize = input.Shape[rank - 1];
            input2D = input.Reshape(new[] { flatBatch, featureSize });
        }

        // Cache input for backward pass
        _lastInput = input2D;

        // Step 1: Compute routing scores
        var routingLogits = _router.Forward(input2D);
        _lastRoutingLogits = routingLogits;

        // Step 2: Apply softmax to get routing probabilities
        var routingWeights = ApplySoftmax(routingLogits);

        // Step 3: Apply Top-K selection if enabled
        if (_topK > 0)
        {
            (routingWeights, _lastTopKIndices) = ApplyTopK(routingWeights, _topK);
        }
        else
        {
            _lastTopKIndices = null;
        }

        _lastRoutingWeights = routingWeights;

        // Step 4: Get outputs from all experts (or only top-K if sparse)
        _lastExpertOutputs = new List<Tensor<T>>();

        if (_topK > 0)
        {
            // Sparse: Only compute outputs for top-K experts
            for (int i = 0; i < _experts.Count; i++)
            {
                // Check if this expert is in top-K for any batch item
                bool isActive = false;
                for (int b = 0; b < batchSize; b++)
                {
                    if (IsExpertActive(b, i))
                    {
                        isActive = true;
                        break;
                    }
                }

                if (isActive)
                {
                    _lastExpertOutputs.Add(_experts[i].Forward(input2D));
                }
                else
                {
                    // Create a zero tensor for inactive experts
                    _lastExpertOutputs.Add(new Tensor<T>(new int[] { batchSize }.Concat(OutputShape.Skip(1)).ToArray()));
                }
            }
        }
        else
        {
            // Dense: All experts process all inputs
            for (int i = 0; i < _experts.Count; i++)
            {
                _lastExpertOutputs.Add(_experts[i].Forward(input2D));
            }
        }

        // Step 5: Combine expert outputs using routing weights
        var combinedOutput = CombineExpertOutputs(_lastExpertOutputs, routingWeights);

        // Step 6: Apply activation function
        var output = ApplyActivation(combinedOutput);

        // Restore original batch dimensions for any-rank support
        if (_originalInputShape != null && _originalInputShape.Length > 2)
        {
            // Output shape: [...leadingDims, outputFeatures]
            int outputFeatures = output.Shape[1];
            int[] newShape = new int[_originalInputShape.Length];
            for (int d = 0; d < _originalInputShape.Length - 1; d++)
                newShape[d] = _originalInputShape[d];
            newShape[_originalInputShape.Length - 1] = outputFeatures;
            output = output.Reshape(newShape);
        }
        else if (_originalInputShape != null && _originalInputShape.Length == 1)
        {
            // 1D input -> 1D output (remove batch dim)
            output = output.Reshape(new[] { output.Shape[1] });
        }

        return output;
    }

    /// <summary>
    /// Performs the backward pass through the MoE layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to this layer's output.</param>
    /// <returns>The gradient of the loss with respect to this layer's input.</returns>
    /// <exception cref="InvalidOperationException">Thrown when backward is called before forward.</exception>
    /// <remarks>
    /// <para>
    /// The backward pass:
    /// 1. Applies the derivative of the activation function
    /// 2. Computes gradients for each expert's contribution
    /// 3. Backpropagates through active experts
    /// 4. Computes gradients for the router
    /// 5. Backpropagates through the router
    /// 6. Returns the combined input gradient
    /// </para>
    /// <para><b>For Beginners:</b> This is where the MoE layer learns from its mistakes.
    ///
    /// The backward pass works in reverse:
    ///
    /// 1. <b>Receive Error Signal:</b> Get information about how wrong the output was
    ///    - This comes from layers after this one (or from the loss function)
    ///
    /// 2. <b>Activation Gradient:</b> Account for the activation function
    ///    - If we applied ReLU, apply its derivative
    ///    - This adjusts the error signal appropriately
    ///
    /// 3. <b>Expert Gradients:</b> Calculate how each expert should improve
    ///    - Weight the error by how much each expert contributed
    ///    - Expert with weight 0.7 gets more of the blame/credit than one with 0.1
    ///    - Send these weighted errors back through each expert
    ///
    /// 4. <b>Router Gradients:</b> Calculate how routing should improve
    ///    - If expert 1 was useful, increase its future routing weight for similar inputs
    ///    - If expert 3 was harmful, decrease its future routing weight
    ///    - This helps the router make better decisions next time
    ///
    /// 5. <b>Combine Input Gradients:</b> Sum up gradients from router and experts
    ///    - This tells earlier layers how they should adjust
    ///
    /// After backward pass completes, all components know how to improve, but haven't
    /// changed yet. The actual changes happen in UpdateParameters().
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        return UseAutodiff
            ? BackwardViaAutodiff(outputGradient)
            : BackwardManual(outputGradient);
    }

    /// <summary>
    /// Manual backward pass implementation using optimized gradient calculations.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    private Tensor<T> BackwardManual(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastRoutingWeights == null || _lastExpertOutputs == null || _lastRoutingLogits == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        // Step 1: Apply activation derivative
        var activationGradient = ApplyActivationDerivative(_lastInput, outputGradient);

        // Step 2: Backpropagate through experts
        var inputGradientFromExperts = new Tensor<T>(_lastInput.Shape);

        for (int i = 0; i < _experts.Count; i++)
        {
            // Weight the gradient by the routing weight for this expert
            var weightedGradient = WeightGradientByRouting(activationGradient, _lastRoutingWeights, i);

            // Only backprop through expert if it was active
            bool wasActive = _topK == 0 || IsExpertUsedInBatch(i);

            if (wasActive)
            {
                var expertInputGradient = _experts[i].Backward(weightedGradient);
                inputGradientFromExperts = inputGradientFromExperts.Add(expertInputGradient);
            }
        }

        // Step 3: Compute router gradient
        var routerGradient = ComputeRouterGradient(activationGradient, _lastExpertOutputs, _lastRoutingWeights, _lastRoutingLogits);

        // Step 4: Backpropagate through router
        var inputGradientFromRouter = _router.Backward(routerGradient);

        // Step 5: Combine gradients from both paths
        var totalInputGradient = inputGradientFromExperts.Add(inputGradientFromRouter);

        return totalInputGradient;
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation by delegating to the autodiff implementations
    /// of the expert layers and router network. Each sublayer will use its own autodiff if available.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastRoutingWeights == null || _lastExpertOutputs == null || _lastRoutingLogits == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Step 1: Apply activation derivative
        var activationGradient = ApplyActivationDerivative(_lastInput, outputGradient);

        // Step 2: Backpropagate through experts (composite - each expert handles its own autodiff)
        var inputGradientFromExperts = new Tensor<T>(_lastInput.Shape);

        for (int i = 0; i < _experts.Count; i++)
        {
            // Weight the gradient by the routing weight for this expert
            var weightedGradient = WeightGradientByRouting(activationGradient, _lastRoutingWeights, i);

            // Only backprop through expert if it was active
            bool wasActive = _topK == 0 || IsExpertUsedInBatch(i);

            if (wasActive)
            {
                var expertInputGradient = _experts[i].Backward(weightedGradient);
                inputGradientFromExperts = inputGradientFromExperts.Add(expertInputGradient);
            }
        }

        // Step 3: Compute router gradient
        var routerGradient = ComputeRouterGradient(activationGradient, _lastExpertOutputs, _lastRoutingWeights, _lastRoutingLogits);

        // Step 4: Backpropagate through router (will use its own autodiff)
        var inputGradientFromRouter = _router.Backward(routerGradient);

        // Step 5: Combine gradients from both paths
        var totalInputGradient = inputGradientFromExperts.Add(inputGradientFromRouter);

        return totalInputGradient;
    }


    /// <summary>
    /// Updates all trainable parameters using the specified learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method updates parameters for both the router and all expert networks that support training.
    /// </para>
    /// <para><b>For Beginners:</b> This applies all the learned improvements to the router and experts.
    ///
    /// After the backward pass calculated how everything should change:
    /// - The router updates its weights to make better routing decisions
    /// - Each expert updates its weights to make better predictions
    /// - The learning rate controls how big these updates are
    ///
    /// Learning rate guidelines:
    /// - Too small: Learning is very slow but stable
    /// - Too large: Learning is fast but might be unstable
    /// - Just right: Balances speed and stability (often 0.001 to 0.01)
    ///
    /// After calling this method, the MoE layer should perform slightly better than before.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // Update router parameters
        if (_router.SupportsTraining)
        {
            _router.UpdateParameters(learningRate);
        }

        // Update all expert parameters
        foreach (var expert in _experts.Where(e => e.SupportsTraining))
        {
            expert.UpdateParameters(learningRate);
        }
    }

    /// <summary>
    /// Gets all trainable parameters as a single vector.
    /// </summary>
    /// <returns>A vector containing all parameters from the router and all experts.</returns>
    /// <remarks>
    /// <para>
    /// Parameters are ordered as: [router parameters] [expert1 parameters] [expert2 parameters] ...
    /// </para>
    /// <para><b>For Beginners:</b> Collects all learned values into one list.
    ///
    /// The returned vector contains:
    /// - First, all parameters from the router
    /// - Then, all parameters from expert 1
    /// - Then, all parameters from expert 2
    /// - And so on
    ///
    /// This is useful for:
    /// - Saving the entire MoE model to disk
    /// - Implementing advanced optimization algorithms
    /// - Analyzing the model's learned parameters
    /// - Transferring knowledge to another model
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Use Vector<T>.Concatenate for production-grade parameter collection
        var paramVectors = new List<Vector<T>>();

        // Add router parameters
        if (_router.ParameterCount > 0)
        {
            paramVectors.Add(_router.GetParameters());
        }

        // Add expert parameters
        foreach (var expert in _experts.Where(e => e.ParameterCount > 0))
        {
            paramVectors.Add(expert.GetParameters());
        }

        return paramVectors.Count > 0 ? Vector<T>.Concatenate(paramVectors.ToArray()) : new Vector<T>(0);
    }

    /// <summary>
    /// Sets all trainable parameters from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing parameters for the router and all experts.</param>
    /// <exception cref="ArgumentException">Thrown when the parameter count doesn't match.</exception>
    /// <remarks>
    /// <para>
    /// Parameters should be in the same order as returned by GetParameters():
    /// [router parameters] [expert1 parameters] [expert2 parameters] ...
    /// </para>
    /// <para><b>For Beginners:</b> Loads previously saved parameters back into the model.
    ///
    /// This is the opposite of GetParameters():
    /// - Takes a vector of all parameters
    /// - Distributes them to the router and experts
    /// - Must match the exact format returned by GetParameters()
    ///
    /// Use this to:
    /// - Load a saved model from disk
    /// - Initialize with pre-trained parameters
    /// - Implement custom optimization algorithms
    ///
    /// If the parameter count doesn't match exactly, an error is thrown to prevent
    /// accidentally corrupting the model.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException(
                $"Expected {ParameterCount} parameters, but got {parameters.Length}.",
                nameof(parameters));
        }

        // Use Vector.Slice for production-grade parameter distribution
        int offset = 0;

        // Set router parameters
        if (_router.ParameterCount > 0)
        {
            _router.SetParameters(parameters.Slice(offset, _router.ParameterCount));
            offset += _router.ParameterCount;
        }

        // Set expert parameters
        foreach (var expert in _experts.Where(e => e.ParameterCount > 0))
        {
            expert.SetParameters(parameters.Slice(offset, expert.ParameterCount));
            offset += expert.ParameterCount;
        }
    }

    /// <summary>
    /// Resets the internal state of the layer, clearing all cached values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This clears cached values from forward/backward passes and resets the state of the router
    /// and all experts. Call this between training batches or when switching between training and inference.
    /// </para>
    /// <para><b>For Beginners:</b> Clears the layer's "short-term memory".
    ///
    /// This resets:
    /// - Cached inputs and outputs
    /// - Routing weights and decisions
    /// - Expert activations
    /// - All temporary values used for learning
    ///
    /// When to call this:
    /// - Between different batches of training data
    /// - When switching from training to testing mode
    /// - Before processing a new, unrelated input
    ///
    /// This ensures that information from one batch doesn't leak into the next batch,
    /// which could cause incorrect gradient calculations or predictions.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        _lastInput = null;
        _lastRoutingWeights = null;
        _lastRoutingLogits = null;
        _lastExpertOutputs = null;
        _lastTopKIndices = null;

        // Reset router state
        _router.ResetState();

        // Reset all expert states
        foreach (var expert in _experts)
        {
            expert.ResetState();
        }
    }

    /// <summary>
    /// Creates a deep copy of this MoE layer.
    /// </summary>
    /// <returns>A new MixtureOfExpertsLayer with the same configuration and parameters.</returns>
    /// <remarks>
    /// <para>
    /// Creates an independent copy of this layer, including the router and all experts.
    /// Changes to the clone won't affect the original.
    /// </para>
    /// <para><b>For Beginners:</b> Makes an identical copy of the entire MoE layer.
    ///
    /// The clone includes:
    /// - A copy of the router
    /// - Copies of all experts
    /// - Same configuration (TopK, shapes, etc.)
    /// - Same learned parameters
    ///
    /// Useful for:
    /// - Creating an ensemble of similar models
    /// - Experimenting with different training approaches
    /// - Saving checkpoints during training
    /// - Implementing certain meta-learning algorithms
    ///
    /// The clone is completely independent - training one won't affect the other.
    /// </para>
    /// </remarks>
    public override LayerBase<T> Clone()
    {
        // Clone router
        ILayer<T> clonedRouter = _router;
        if (_router is LayerBase<T> routerBase)
        {
            clonedRouter = (ILayer<T>)routerBase.Clone();
        }

        // Clone experts
        var clonedExperts = _experts.Select(e =>
        {
            if (e is LayerBase<T> expertBase)
            {
                return (ILayer<T>)expertBase.Clone();
            }
            return e;
        }).ToList();

        return new MixtureOfExpertsLayer<T>(
            clonedExperts,
            clonedRouter,
            InputShape,
            OutputShape,
            _topK,
            ScalarActivation,
            _useAuxiliaryLoss,
            _auxiliaryLossWeight);
    }

    #region IAuxiliaryLossLayer Implementation

    /// <summary>
    /// Gets or sets a value indicating whether to use the auxiliary load balancing loss.
    /// </summary>
    /// <value>
    /// <c>true</c> to compute and apply load balancing loss during training; otherwise, <c>false</c>.
    /// </value>
    /// <remarks>
    /// <para>
    /// When enabled, the layer computes a load balancing loss that encourages balanced expert usage.
    /// This loss is added to the primary task loss during training to prevent expert imbalance.
    /// </para>
    /// <para><b>For Beginners:</b> Turn load balancing on or off.
    ///
    /// Enable this during training to ensure all experts are used roughly equally.
    /// Disable during inference/testing since load balancing is only needed during training.
    ///
    /// Benefits of load balancing:
    /// - Prevents expert collapse (all inputs routed to the same expert)
    /// - Encourages specialization across different experts
    /// - Improves overall model quality and generalization
    /// </para>
    /// </remarks>
    public bool UseAuxiliaryLoss
    {
        get => _useAuxiliaryLoss;
        set => _useAuxiliaryLoss = value;
    }

    /// <summary>
    /// Gets or sets the weight for the auxiliary load balancing loss.
    /// </summary>
    /// <value>
    /// The coefficient that determines how much the load balancing loss influences training.
    /// </value>
    /// <remarks>
    /// <para>
    /// This weight is multiplied by the load balancing loss before adding it to the primary loss.
    /// Typical values range from 0.01 to 0.1. Higher values enforce stronger load balancing.
    /// </para>
    /// <para><b>For Beginners:</b> Controls the importance of load balancing.
    ///
    /// Recommended starting value: 0.01
    ///
    /// Tuning guidelines:
    /// - If experts are very imbalanced: increase (e.g., 0.05 or 0.1)
    /// - If primary task accuracy suffers: decrease (e.g., 0.005)
    /// - Monitor both primary loss and expert usage statistics to find the right balance
    /// </para>
    /// </remarks>
    public T AuxiliaryLossWeight
    {
        get => _auxiliaryLossWeight;
        set => _auxiliaryLossWeight = value;
    }

    /// <summary>
    /// Computes the load balancing auxiliary loss based on expert usage from the last forward pass.
    /// </summary>
    /// <returns>The load balancing loss value.</returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown when called before a forward pass or when auxiliary loss is disabled.
    /// </exception>
    /// <remarks>
    /// <para>
    /// The load balancing loss encourages balanced expert usage by penalizing imbalanced routing.
    /// It is computed as the dot product of two fractions for each expert:
    /// - Token fraction: Proportion of tokens (inputs) routed to this expert
    /// - Probability mass fraction: Average routing probability for this expert
    ///
    /// Loss = NumExperts * sum(token_fraction_i * prob_mass_fraction_i) for all experts i
    ///
    /// This loss is minimized when all experts receive equal numbers of tokens and equal
    /// total probability mass, encouraging balanced utilization.
    /// </para>
    /// <para><b>For Beginners:</b> Calculates a penalty for imbalanced expert usage.
    ///
    /// How it works:
    ///
    /// 1. <b>Count Token Assignments:</b>
    ///    - For each expert, count how many inputs chose it (with Top-K) or had non-zero weight
    ///    - Example with 8 inputs and 4 experts: [3, 2, 2, 1] tokens per expert
    ///
    /// 2. <b>Calculate Probability Mass:</b>
    ///    - For each expert, sum up its routing weights across all inputs
    ///    - Example: [0.4, 0.3, 0.2, 0.1] total probability per expert
    ///
    /// 3. <b>Compute Load Balancing Loss:</b>
    ///    - Convert counts to fractions: [3/8, 2/8, 2/8, 1/8] = [0.375, 0.25, 0.25, 0.125]
    ///    - Convert probabilities to fractions: [0.4, 0.3, 0.2, 0.1]
    ///    - Dot product: 0.375*0.4 + 0.25*0.3 + 0.25*0.2 + 0.125*0.1
    ///    - Multiply by numExperts (4): gives load balancing loss
    ///
    /// Why this works:
    /// - If all experts are used equally, both fractions are [0.25, 0.25, 0.25, 0.25]
    /// - Dot product: 0.25*0.25 * 4 = 0.25 (minimum possible)
    /// - If imbalanced like [0.5, 0.3, 0.15, 0.05] × [0.6, 0.25, 0.1, 0.05]
    /// - Dot product: 0.5*0.6 + ... = higher value (penalty for imbalance)
    ///
    /// The loss is minimized when usage is perfectly balanced!
    /// </para>
    /// </remarks>
    public T ComputeAuxiliaryLoss()
    {
        if (!_useAuxiliaryLoss)
        {
            return NumOps.Zero;
        }

        if (_lastRoutingWeights == null || _lastInput == null)
        {
            throw new InvalidOperationException(
                "Forward pass must be called before computing auxiliary loss.");
        }

        int batchSize = _lastInput.Shape[0];
        int numExperts = _experts.Count;

        // VECTORIZED: Use Engine operations for probability mass computation
        // Sum routing weights along batch dimension to get probability mass per expert
        var probMassTensor = Engine.ReduceSum(_lastRoutingWeights, new[] { 0 }, keepDims: false);
        var probMassVec = probMassTensor.ToVector();

        // VECTORIZED: Compute token counts using tensor operations
        T threshold = NumOps.FromDouble(0.01);

        Tensor<T> tokenCountsTensor;
        if (_topK > 0 && _lastTopKIndices != null)
        {
            // For Top-K, sparse weights are already zero for inactive experts
            // Count non-zero entries per expert (column-wise)
            var isActive = Engine.TensorGreaterThan(_lastRoutingWeights, NumOps.Zero);
            // Convert boolean-like comparison result to counts
            tokenCountsTensor = Engine.ReduceSum(isActive, new[] { 0 }, keepDims: false);
        }
        else
        {
            // For dense routing, count where weight > threshold
            var isActive = Engine.TensorGreaterThan(_lastRoutingWeights, threshold);
            tokenCountsTensor = Engine.ReduceSum(isActive, new[] { 0 }, keepDims: false);
        }
        var tokenCountVec = tokenCountsTensor.ToVector();

        // VECTORIZED: Normalize to get fractions using Vector operations

        T totalTokens = tokenCountVec.Sum();
        T totalProbMass = probMassVec.Sum();

        // VECTORIZED: Compute load balancing loss using vector operations
        T safeTokenTotal = NumOps.GreaterThan(totalTokens, NumOps.Zero) ? totalTokens : NumOps.One;
        T safeProbTotal = NumOps.GreaterThan(totalProbMass, NumOps.Zero) ? totalProbMass : NumOps.One;

        var tokenFractions = (Vector<T>)Engine.Divide(tokenCountVec, safeTokenTotal);
        var probFractions = (Vector<T>)Engine.Divide(probMassVec, safeProbTotal);

        // Element-wise multiply and sum
        var products = (Vector<T>)Engine.Multiply(tokenFractions, probFractions);
        T loss = products.Sum();

        loss = NumOps.Multiply(NumOps.FromDouble(numExperts), loss);

        return loss;
    }

    /// <summary>
    /// Gets diagnostic information about expert usage and load balancing.
    /// </summary>
    /// <returns>
    /// A dictionary containing diagnostic metrics including per-expert usage statistics,
    /// load balancing metrics, and routing weight distributions.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method provides detailed statistics about expert usage that can be used for
    /// monitoring training progress, debugging routing issues, and tuning load balancing parameters.
    /// </para>
    /// <para><b>For Beginners:</b> Gets a detailed report about how experts are being used.
    ///
    /// The returned dictionary includes:
    /// - <b>expert_i_tokens:</b> How many inputs were routed to expert i
    /// - <b>expert_i_prob_mass:</b> Total routing weight for expert i across all inputs
    /// - <b>expert_i_avg_weight:</b> Average routing weight when expert i is selected
    /// - <b>load_balance_loss:</b> Current load balancing loss value
    /// - <b>usage_variance:</b> Variance in expert usage (lower is better balanced)
    /// - <b>max_min_ratio:</b> Ratio of most-used to least-used expert (1.0 is perfect)
    ///
    /// Use this information to:
    /// - Monitor if experts are being used balanced or if some are overused
    /// - Decide if you need to adjust the load balancing weight
    /// - Detect expert collapse (all inputs routed to one expert)
    /// - Track training health over time
    ///
    /// Example output:
    /// {
    ///   "expert_0_tokens": "245",
    ///   "expert_1_tokens": "198",
    ///   "expert_2_tokens": "223",
    ///   "expert_3_tokens": "234",
    ///   "expert_0_prob_mass": "0.28",
    ///   "expert_1_prob_mass": "0.22",
    ///   ...
    ///   "load_balance_loss": "0.253",
    ///   "usage_variance": "0.0012",
    ///   "max_min_ratio": "1.24"
    /// }
    /// </para>
    /// </remarks>
    public Dictionary<string, string> GetAuxiliaryLossDiagnostics()
    {
        var diagnostics = new Dictionary<string, string>();

        if (_lastRoutingWeights == null || _lastInput == null)
        {
            diagnostics["status"] = "No forward pass performed yet";
            return diagnostics;
        }

        int batchSize = _lastInput.Shape[0];
        int numExperts = _experts.Count;

        // Compute per-expert statistics
        var tokenCounts = new T[numExperts];
        var probabilityMass = new T[numExperts];
        var avgWeights = new T[numExperts];

        for (int i = 0; i < numExperts; i++)
        {
            T tokenCount = NumOps.Zero;
            T probMass = NumOps.Zero;
            T weightSum = NumOps.Zero;
            T activeCount = NumOps.Zero;

            for (int b = 0; b < batchSize; b++)
            {
                var weight = _lastRoutingWeights[b, i];

                // Count token assignment
                bool isActive = _topK > 0
                    ? IsExpertActive(b, i)
                    : NumOps.GreaterThan(weight, NumOps.FromDouble(0.01));

                if (isActive)
                {
                    tokenCount = NumOps.Add(tokenCount, NumOps.One);
                    activeCount = NumOps.Add(activeCount, NumOps.One);
                    weightSum = NumOps.Add(weightSum, weight);
                }

                // Accumulate probability mass
                probMass = NumOps.Add(probMass, weight);
            }

            tokenCounts[i] = tokenCount;
            probabilityMass[i] = probMass;
            avgWeights[i] = NumOps.GreaterThan(activeCount, NumOps.Zero)
                ? NumOps.Divide(weightSum, activeCount)
                : NumOps.Zero;

            // Add to diagnostics
            diagnostics[$"expert_{i}_tokens"] = Convert.ToDouble(tokenCount).ToString("F0");
            diagnostics[$"expert_{i}_prob_mass"] = Convert.ToDouble(probMass).ToString("F4");
            diagnostics[$"expert_{i}_avg_weight"] = Convert.ToDouble(avgWeights[i]).ToString("F4");
        }

        // Compute load balancing loss
        if (_useAuxiliaryLoss)
        {
            var loadBalanceLoss = ComputeAuxiliaryLoss();
            diagnostics["load_balance_loss"] = Convert.ToDouble(loadBalanceLoss).ToString("F6");
        }

        // VECTORIZED: Compute usage variance using Vector operations
        var tokenCountVec = new Vector<T>(tokenCounts);
        T meanTokens = NumOps.Divide(tokenCountVec.Sum(), NumOps.FromDouble(numExperts));

        // Compute variance: E[(X - mean)^2]
        var meanVec = Engine.Fill(numExperts, meanTokens);
        var diffs = Engine.Subtract(tokenCountVec, meanVec);
        var diffsSquared = Engine.Multiply(diffs, diffs);
        T variance = NumOps.Divide(Engine.Sum(diffsSquared), NumOps.FromDouble(numExperts));
        diagnostics["usage_variance"] = Convert.ToDouble(variance).ToString("F6");

        // Compute max/min ratio (diagnostics - use indexed access)
        T maxTokens = tokenCounts[0];
        T minTokens = tokenCounts[0];
        for (int i = 1; i < numExperts; i++)
        {
            if (NumOps.GreaterThan(tokenCounts[i], maxTokens)) maxTokens = tokenCounts[i];
            if (NumOps.LessThan(tokenCounts[i], minTokens)) minTokens = tokenCounts[i];
        }

        T maxMinRatio = NumOps.GreaterThan(minTokens, NumOps.Zero)
            ? NumOps.Divide(maxTokens, minTokens)
            : NumOps.FromDouble(double.PositiveInfinity);
        diagnostics["max_min_ratio"] = Convert.ToDouble(maxMinRatio).ToString("F4");

        diagnostics["num_experts"] = numExperts.ToString();
        diagnostics["batch_size"] = batchSize.ToString();
        diagnostics["top_k"] = _topK.ToString();

        return diagnostics;
    }

    /// <summary>
    /// Gets diagnostic information about this component's state and behavior.
    /// Overrides <see cref="LayerBase{T}.GetDiagnostics"/> to include auxiliary loss diagnostics.
    /// </summary>
    /// <returns>
    /// A dictionary containing diagnostic metrics including both base layer diagnostics and
    /// auxiliary loss diagnostics from <see cref="GetAuxiliaryLossDiagnostics"/>.
    /// </returns>
    public override Dictionary<string, string> GetDiagnostics()
    {
        var diagnostics = base.GetDiagnostics();

        // Merge auxiliary loss diagnostics
        var auxDiagnostics = GetAuxiliaryLossDiagnostics();
        foreach (var kvp in auxDiagnostics)
        {
            diagnostics[kvp.Key] = kvp.Value;
        }

        return diagnostics;
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Applies softmax to routing logits to produce normalized probability weights.
    /// </summary>
    /// <param name="logits">The raw routing scores from the router.</param>
    /// <returns>Normalized probability weights that sum to 1 for each batch item.</returns>
    /// <remarks>
    /// <para>
    /// Softmax converts raw scores (logits) into probabilities using the formula:
    /// softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
    ///
    /// For numerical stability, we subtract the maximum value before exponentiation:
    /// softmax(x) = softmax(x - max(x))
    /// </para>
    /// <para><b>For Beginners:</b> Converts raw scores into probabilities.
    ///
    /// Why use softmax:
    /// - Converts any numbers into probabilities (0 to 1)
    /// - All probabilities sum to 1
    /// - Larger scores become larger probabilities
    /// - Smaller scores become smaller probabilities
    ///
    /// Example:
    /// - Input scores: [2.0, 1.0, 0.1, -1.0]
    /// - After softmax: [0.58, 0.21, 0.09, 0.03]
    /// - Notice they sum to 1.0
    ///
    /// The "subtract max" trick prevents numerical overflow (very large exponentials)
    /// without changing the final result.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplySoftmax(Tensor<T> logits)
    {
        // Fully vectorized softmax using tensor operations
        // logits shape: [batchSize, numExperts]

        // Step 1: Find max per row for numerical stability (axis=1)
        var maxPerRow = Engine.ReduceMax(logits, new[] { 1 }, keepDims: true, out _); // [batchSize, 1]

        // Step 2: Subtract max for numerical stability - use broadcasting
        var shiftedLogits = Engine.TensorSubtract<T>(logits, maxPerRow); // [batchSize, numExperts]

        // Step 3: Apply exp element-wise
        var expValues = Engine.TensorExp(shiftedLogits); // [batchSize, numExperts]

        // Step 4: Sum exp values per row
        var expSum = Engine.ReduceSum(expValues, new[] { 1 }, keepDims: true); // [batchSize, 1]

        // Step 5: Normalize - divide each row by its sum (with broadcasting)
        var softmax = Engine.TensorDivide<T>(expValues, expSum); // [batchSize, numExperts]

        return softmax;
    }

    /// <summary>
    /// Applies Top-K selection to routing weights, keeping only the K highest weights per batch item.
    /// </summary>
    /// <param name="weights">The routing probability weights.</param>
    /// <param name="k">The number of top experts to keep.</param>
    /// <returns>A tuple of (sparse weights, top-K indices).</returns>
    /// <remarks>
    /// <para>
    /// Top-K selection:
    /// 1. Finds the K experts with the highest weights for each input
    /// 2. Sets all other experts' weights to zero
    /// 3. Renormalizes the remaining K weights to sum to 1
    /// 4. Returns both the sparse weights and the indices of selected experts
    /// </para>
    /// <para><b>For Beginners:</b> Keeps only the best K experts for each input.
    ///
    /// Why use Top-K:
    /// - Dramatically reduces computation (only K experts run instead of all N)
    /// - Maintains model quality (the best experts are still used)
    /// - Enables scaling to huge models (hundreds of experts, but only use 2-4)
    ///
    /// Example with 6 experts and K=2:
    /// - Original weights: [0.30, 0.05, 0.25, 0.10, 0.20, 0.10]
    /// - Top-2 are indices 0 and 2 (weights 0.30 and 0.25)
    /// - After Top-K: [0.545, 0, 0.455, 0, 0, 0]
    /// - Notice top-2 are renormalized to sum to 1.0
    /// - Experts 1, 3, 4, 5 are completely skipped
    ///
    /// With this example, we use only 33% of experts but keep the most relevant ones!
    /// </para>
    /// </remarks>
    private (Tensor<T> sparseWeights, int[,] topKIndices) ApplyTopK(Tensor<T> weights, int k)
    {
        int batchSize = weights.Shape[0];
        int numExperts = weights.Shape[1];

        // VECTORIZED: Use TensorTopK to get top-k values and indices
        var topKValues = Engine.TensorTopK(weights, k, axis: 1, out Tensor<int> topKIndicesTensor);
        // topKValues shape: [batchSize, k]
        // topKIndicesTensor shape: [batchSize, k]

        // VECTORIZED: Compute row sums for normalization
        var sumPerRow = Engine.ReduceSum(topKValues, new[] { 1 }, keepDims: true); // [batchSize, 1]

        // VECTORIZED: Normalize top-k values
        var normalizedTopK = Engine.TensorDivide(topKValues, sumPerRow); // [batchSize, k]

        // VECTORIZED: Scatter normalized values back to full expert dimension
        // Create zero tensor for sparse weights
        var sparseWeights = new Tensor<T>(weights.Shape);
        sparseWeights.Fill(NumOps.Zero);

        // Use TensorScatter to place normalized values at correct positions
        sparseWeights = Engine.TensorScatter(sparseWeights, topKIndicesTensor, normalizedTopK, axis: 1);

        // Convert topKIndicesTensor to int[,] for backward compatibility
        var topKIndices = new int[batchSize, k];
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < k; i++)
            {
                topKIndices[b, i] = topKIndicesTensor[b, i];
            }
        }

        return (sparseWeights, topKIndices);
    }

    /// <summary>
    /// Combines expert outputs using routing weights to produce the final output.
    /// </summary>
    /// <param name="expertOutputs">The outputs from all experts.</param>
    /// <param name="routingWeights">The routing weights for each expert.</param>
    /// <returns>The weighted combination of expert outputs.</returns>
    /// <remarks>
    /// <para>
    /// Combines outputs as: output = sum(weight_i * expertOutput_i) for all i.
    /// This is a weighted average where experts with higher routing weights contribute more.
    /// </para>
    /// <para><b>For Beginners:</b> Mixes expert predictions based on their weights.
    ///
    /// The combination works like voting:
    /// - Each expert makes a prediction
    /// - Each expert has a weight (importance)
    /// - The final output is a weighted average
    ///
    /// Example with 3 experts:
    /// - Expert 1 predicts [1.0, 2.0, 3.0] with weight 0.5
    /// - Expert 2 predicts [2.0, 1.0, 4.0] with weight 0.3
    /// - Expert 3 predicts [0.0, 3.0, 2.0] with weight 0.2
    ///
    /// Final output:
    /// - [1.0*0.5 + 2.0*0.3 + 0.0*0.2, 2.0*0.5 + 1.0*0.3 + 3.0*0.2, 3.0*0.5 + 4.0*0.3 + 2.0*0.2]
    /// - = [1.1, 1.9, 3.1]
    ///
    /// Experts with higher weights have more influence on the final prediction.
    /// </para>
    /// </remarks>
    private Tensor<T> CombineExpertOutputs(List<Tensor<T>> expertOutputs, Tensor<T> routingWeights)
    {
        if (expertOutputs.Count == 0)
        {
            throw new ArgumentException("Must have at least one expert output.", nameof(expertOutputs));
        }

        // Initialize combined output with zeros
        var combined = new Tensor<T>(expertOutputs[0].Shape);
        combined.Fill(NumOps.Zero);

        // Vectorized: For each expert, multiply its output by routing weights and accumulate
        // expertOutputs[i] shape: [batchSize, outputDim]
        // routingWeights shape: [batchSize, numExperts]
        // We need weight for expert i: routingWeights[:, i] with shape [batchSize, 1] for broadcasting

        for (int i = 0; i < expertOutputs.Count; i++)
        {
            var expertOutput = expertOutputs[i];

            // Extract routing weight for expert i as a column tensor [batchSize, 1]
            var weightColumn = Engine.TensorSlice(routingWeights, new[] { 0, i }, new[] { routingWeights.Shape[0], 1 });

            // Multiply expert output by weight (broadcasts across output dimensions)
            var weightedOutput = Engine.TensorMultiply(expertOutput, weightColumn);

            // Accumulate
            combined = Engine.TensorAdd(combined, weightedOutput);
        }

        return combined;
    }

    /// <summary>
    /// Computes the gradient for the router during backpropagation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer output.</param>
    /// <param name="expertOutputs">The cached expert outputs from the forward pass.</param>
    /// <param name="routingWeights">The routing weights (after softmax) from the forward pass.</param>
    /// <param name="routingLogits">The routing logits (before softmax) from the forward pass.</param>
    /// <returns>The gradient of the loss with respect to the router's output (logits).</returns>
    /// <remarks>
    /// <para>
    /// The router gradient computation involves:
    /// 1. Computing how the loss changes with respect to routing weights
    /// 2. Applying the Jacobian of softmax to get gradients for logits
    ///
    /// For each expert i: d(loss)/d(weight_i) = sum_over_output(d(loss)/d(output) * expertOutput_i)
    /// Then apply softmax derivative to get d(loss)/d(logit_i)
    /// </para>
    /// <para><b>For Beginners:</b> Calculates how the router should change its decisions.
    ///
    /// This is the most complex part of MoE learning. Here's what happens:
    ///
    /// 1. <b>Assess Expert Contributions:</b>
    ///    - For each expert, calculate: did increasing its weight help or hurt?
    ///    - If expert's output agreed with what we wanted: increase its weight
    ///    - If expert's output disagreed: decrease its weight
    ///
    /// 2. <b>Account for Softmax:</b>
    ///    - Routing weights go through softmax, which creates dependencies
    ///    - Increasing one weight means others must decrease (they sum to 1)
    ///    - We need to account for this coupling in the gradients
    ///
    /// 3. <b>Produce Router Gradients:</b>
    ///    - Convert weight gradients to logit gradients (pre-softmax)
    ///    - These gradients tell the router how to adjust its parameters
    ///
    /// The goal: Make the router better at sending each input to the experts that will
    /// handle it well.
    ///
    /// Example:
    /// - If expert 3 made a great prediction for certain inputs
    /// - The router should learn to route similar inputs to expert 3 in the future
    /// - If expert 1 made poor predictions
    /// - The router should learn to avoid expert 1 for those types of inputs
    /// </para>
    /// </remarks>
    private Tensor<T> ComputeRouterGradient(
        Tensor<T> outputGradient,
        List<Tensor<T>> expertOutputs,
        Tensor<T> routingWeights,
        Tensor<T> routingLogits)
    {
        int batchSize = outputGradient.Shape[0];
        int numExperts = expertOutputs.Count;

        // VECTORIZED: Compute weight gradients using tensor operations
        // d(loss)/d(weight_i) = sum_j(d(loss)/d(output_j) * expertOutput_i_j)
        // This is the element-wise product followed by sum over output dimension
        var weightGradients = new Tensor<T>(new int[] { batchSize, numExperts });

        for (int i = 0; i < numExperts; i++)
        {
            var expertOutput = expertOutputs[i];
            if (expertOutput.Shape.Length == 2)
            {
                // Element-wise multiply outputGradient with expertOutput, then sum over output dim
                var product = Engine.TensorMultiply(outputGradient, expertOutput);
                var summed = Engine.ReduceSum(product, new[] { 1 }, keepDims: false); // [batchSize]

                // Store in weightGradients column
                Engine.TensorSetSliceAxis(weightGradients, summed.Reshape([batchSize, 1]), 1, i);
            }
        }

        // VECTORIZED: Apply softmax backward using Engine operation
        // The softmax backward formula is: grad_logits = softmax * (grad_weights - sum(grad_weights * softmax))
        // This is equivalent to the full Jacobian application
        var logitGradients = Engine.TensorSoftmaxBackward(routingWeights, weightGradients, axis: 1);

        return logitGradients;
    }

    /// <summary>
    /// Weights the output gradient by the routing weight for a specific expert.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer output.</param>
    /// <param name="routingWeights">The routing weights from the forward pass.</param>
    /// <param name="expertIndex">The index of the expert to weight for.</param>
    /// <returns>The weighted gradient to pass to the expert.</returns>
    /// <remarks>
    /// <para>
    /// Each expert's gradient is scaled by its routing weight, since it only partially
    /// contributed to the output.
    /// </para>
    /// <para><b>For Beginners:</b> Adjusts error signal based on how much the expert contributed.
    ///
    /// If an expert contributed 70% to the output (weight = 0.7):
    /// - It receives 70% of the error signal
    /// - This is fair because it was 70% responsible for the output
    ///
    /// If an expert contributed only 5% (weight = 0.05):
    /// - It receives only 5% of the error signal
    /// - It had little influence, so it gets little blame/credit
    ///
    /// This ensures experts are updated proportionally to their contribution.
    /// </para>
    /// </remarks>
    private Tensor<T> WeightGradientByRouting(Tensor<T> outputGradient, Tensor<T> routingWeights, int expertIndex)
    {
        // Vectorized: Extract routing weights for this expert as a column [batchSize, 1]
        var weightColumn = Engine.TensorSlice(routingWeights, new[] { 0, expertIndex }, new[] { routingWeights.Shape[0], 1 });

        // Multiply gradient by weight (broadcasts across output dimensions)
        // outputGradient shape: [batchSize, outputDim]
        // weightColumn shape: [batchSize, 1] -> broadcasts to [batchSize, outputDim]
        var weightedGradient = Engine.TensorMultiply(outputGradient, weightColumn);

        return weightedGradient;
    }

    /// <summary>
    /// Checks if a specific expert is active for a specific batch item in Top-K routing.
    /// </summary>
    /// <param name="batchIndex">The batch item index.</param>
    /// <param name="expertIndex">The expert index.</param>
    /// <returns>True if the expert is in the top-K for this batch item, false otherwise.</returns>
    /// <remarks>
    /// <para>
    /// When using Top-K routing, this helper checks if a specific expert was selected
    /// for a specific input.
    /// </para>
    /// <para><b>For Beginners:</b> Checks if an expert was chosen for a particular input.
    ///
    /// With Top-K routing:
    /// - Each input selects K experts
    /// - This function checks if a given expert was one of them
    ///
    /// Used to:
    /// - Skip computation for inactive experts
    /// - Skip gradient updates for experts that weren't used
    /// - Track which experts are being utilized
    /// </para>
    /// </remarks>
    private bool IsExpertActive(int batchIndex, int expertIndex)
    {
        if (_lastTopKIndices == null || _topK == 0)
        {
            return true; // All experts active if not using Top-K
        }

        for (int k = 0; k < _topK; k++)
        {
            if (_lastTopKIndices[batchIndex, k] == expertIndex)
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Checks if a specific expert was used for any batch item.
    /// </summary>
    /// <param name="expertIndex">The expert index.</param>
    /// <returns>True if the expert was used in at least one batch item, false otherwise.</returns>
    /// <remarks>
    /// <para>
    /// This determines if an expert needs to perform backpropagation - only experts that
    /// were used in the forward pass need to compute gradients.
    /// </para>
    /// <para><b>For Beginners:</b> Checks if an expert was used at all in this batch.
    ///
    /// With sparse routing:
    /// - Some experts might not be selected for any inputs in a batch
    /// - Those experts can skip backpropagation entirely
    /// - This saves computation time
    ///
    /// This is one of the key efficiency benefits of sparse MoE!
    /// </para>
    /// </remarks>
    private bool IsExpertUsedInBatch(int expertIndex)
    {
        if (_lastTopKIndices == null || _topK == 0)
        {
            return true; // All experts used if not using Top-K
        }

        int batchSize = _lastTopKIndices.GetLength(0);
        for (int b = 0; b < batchSize; b++)
        {
            if (IsExpertActive(b, expertIndex))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Comparer for sorting numeric values in descending order.
    /// </summary>
    private class NumericComparer : IComparer<T>
    {
        private readonly INumericOperations<T> _ops = MathHelper.GetNumericOperations<T>();

        public int Compare(T? x, T? y)
        {
            if (x == null && y == null) return 0;
            if (x == null) return 1;
            if (y == null) return -1;

            if (_ops.GreaterThan(x, y)) return -1;  // Descending order
            if (_ops.LessThan(x, y)) return 1;
            return 0;
        }
    }

    #endregion

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        if (inputNodes.Count == 0)
            throw new ArgumentException("At least one input node is required.", nameof(inputNodes));

        // Check that all components support JIT
        if (!_router.SupportsJitCompilation)
            throw new NotSupportedException("MoE router does not support JIT compilation.");

        foreach (var expert in _experts)
        {
            if (!expert.SupportsJitCompilation)
                throw new NotSupportedException($"Expert does not support JIT compilation.");
        }

        // MixtureOfExpertsLayer JIT uses soft routing with TopK selection:
        // 1. Router computes routing logits for each expert
        // 2. TopKSoftmax selects top-K experts with differentiable routing weights
        // 3. Each expert processes the input
        // 4. Outputs are weighted by routing weights and summed

        var input = inputNodes[0];

        // Get routing logits from router
        var routingLogits = _router.ExportComputationGraph(inputNodes);

        // Apply TopKSoftmax for differentiable expert selection
        var routingWeights = TensorOperations<T>.TopKSoftmax(routingLogits, _topK);

        // Process through each expert and compute weighted sum
        ComputationNode<T>? output = null;
        int numExperts = _experts.Count;

        for (int i = 0; i < numExperts; i++)
        {
            // Get expert output
            var expertOutput = _experts[i].ExportComputationGraph(inputNodes);

            // Get routing weight for this expert (slice from routing weights)
            var expertWeight = TensorOperations<T>.Slice(routingWeights, i, 1, axis: -1);

            // Weight the expert output
            var weightedOutput = TensorOperations<T>.ElementwiseMultiply(expertOutput, expertWeight);

            // Accumulate outputs
            if (output == null)
            {
                output = weightedOutput;
            }
            else
            {
                output = TensorOperations<T>.Add(output, weightedOutput);
            }
        }

        // Apply layer activation
        output = ApplyActivationToGraph(output!);

        return output;
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    /// <value>
    /// <c>true</c> if both the router and all experts support JIT compilation; otherwise, <c>false</c>.
    /// </value>
    /// <remarks>
    /// <para>
    /// JIT compilation for MoE uses TopKSoftmax for differentiable expert selection.
    /// The routing is performed by the router network, and the selected experts'
    /// outputs are weighted by the softmax-normalized routing scores.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation =>
        _router.SupportsJitCompilation && _experts.All(e => e.SupportsJitCompilation);

}

using AiDotNet.Autodiff;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a synaptic plasticity layer that models biological learning mechanisms through spike-timing-dependent plasticity.
/// </summary>
/// <remarks>
/// <para>
/// A synaptic plasticity layer implements biologically-inspired learning rules that modify connection strengths based on 
/// the relative timing of pre- and post-synaptic neuron activations. This implements spike-timing-dependent plasticity (STDP),
/// a form of Hebbian learning observed in biological neural systems. The layer maintains traces of neuronal activity and
/// applies long-term potentiation (LTP) and long-term depression (LTD) based on the temporal relationship between spikes.
/// </para>
/// <para><b>For Beginners:</b> This layer mimics how brain cells (neurons) learn by strengthening or weakening their connections.
/// 
/// Think of it like forming memories:
/// - When two connected neurons activate in sequence (one fires, then the other), their connection gets stronger
/// - When they activate in the opposite order, their connection gets weaker
/// - Over time, pathways that represent useful patterns become stronger
/// 
/// For example, imagine learning to associate a bell sound with food (like Pavlov's dog experiment):
/// - Initially, there's a weak connection between "hear bell" neurons and "expect food" neurons
/// - When the bell regularly comes before food, the connection strengthens
/// - Eventually, just the bell alone strongly activates the "expect food" response
/// 
/// This mimics how real brains learn patterns and form associations between related events.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class SynapticPlasticityLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The input tensor from the last forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the input tensor from the most recent forward pass, which is used during the update process
    /// to determine which presynaptic neurons were active.
    /// </para>
    /// <para><b>For Beginners:</b> This stores which input neurons were recently active.
    ///
    /// Think of it as recording:
    /// - Which sensors or input neurons sent signals
    /// - How strong those signals were
    /// - This information is used to determine which connections should be modified
    ///
    /// For example, in visual learning, this might represent which specific visual features
    /// were detected in an image.
    /// </para>
    /// </remarks>
    private Tensor<T> _lastInput;

    /// <summary>
    /// The output tensor from the last forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the output tensor from the most recent forward pass, which is used during the update process
    /// to determine which postsynaptic neurons were active.
    /// </para>
    /// <para><b>For Beginners:</b> This stores which output neurons were recently active.
    ///
    /// This records:
    /// - Which output neurons responded to the input
    /// - How strongly they responded
    /// - This information helps determine which connections to strengthen or weaken
    ///
    /// For example, if learning to recognize faces, this might represent which "face detector"
    /// neurons activated in response to an image.
    /// </para>
    /// </remarks>
    private Tensor<T> _lastOutput;

    /// <summary>
    /// The weight matrix representing connection strengths between neurons.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the connection strengths between all pairs of neurons in the layer. Each weight represents
    /// the strength of the synaptic connection from one neuron to another, with higher values indicating stronger connections.
    /// </para>
    /// <para><b>For Beginners:</b> This represents how strongly connected each pair of neurons is.
    /// 
    /// The weights matrix works like this:
    /// - Each entry represents the strength of one neuron's influence on another
    /// - Higher values mean stronger connections
    /// - These connections are what the brain changes during learning
    /// - Initially random, these values organize through experience
    /// 
    /// Think of it as a map showing how strongly each neuron can activate each other neuron,
    /// which gets updated as the network learns patterns.
    /// </para>
    /// </remarks>
    private Tensor<T> _weights;

    /// <summary>
    /// The rate at which long-term potentiation (strengthening) occurs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the learning rate for long-term potentiation (LTP), which controls how quickly synaptic
    /// connections strengthen when pre-synaptic activity precedes post-synaptic activity.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how quickly connections strengthen during learning.
    /// 
    /// Long-term potentiation is:
    /// - The process of making connections stronger
    /// - Happens when one neuron helps trigger another neuron
    /// - A key mechanism for forming memories in real brains
    /// 
    /// A higher rate means:
    /// - Faster learning
    /// - But potentially less stability and precision
    /// 
    /// This is similar to how quickly you form associations between related events.
    /// </para>
    /// </remarks>
    private readonly double _stdpLtpRate;

    /// <summary>
    /// The rate at which long-term depression (weakening) occurs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the learning rate for long-term depression (LTD), which controls how quickly synaptic
    /// connections weaken when post-synaptic activity precedes pre-synaptic activity.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how quickly connections weaken during learning.
    /// 
    /// Long-term depression is:
    /// - The process of making connections weaker
    /// - Happens when the timing of neural activity is reversed
    /// - Just as important as strengthening for learning correctly
    /// 
    /// This is like how your brain weakens incorrect associations when events don't 
    /// actually predict each other.
    /// </para>
    /// </remarks>
    private readonly double _stdpLtdRate;

    /// <summary>
    /// The rate at which homeostatic mechanisms regulate synaptic strength.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the rate of homeostatic plasticity, which regulates synaptic strengths to maintain stability
    /// in the network. It prevents runaway potentiation by making it harder to strengthen already-strong synapses.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how the system maintains balance and stability.
    /// 
    /// Homeostasis works like this:
    /// - Very strong connections become harder to strengthen further
    /// - Very weak connections become harder to weaken further
    /// - This prevents connections from all becoming maximum or minimum strength
    /// 
    /// This is similar to how your brain maintains overall balance - not all connections
    /// can be strong, or the brain would become hyperactive and unstable.
    /// </para>
    /// </remarks>
    private readonly double _homeostasisRate;

    /// <summary>
    /// The maximum allowed value for a synaptic weight.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field defines the upper bound for synaptic weights, ensuring that no connection becomes arbitrarily strong.
    /// </para>
    /// <para><b>For Beginners:</b> This is the maximum strength a connection can reach.
    /// 
    /// Setting a maximum weight:
    /// - Prevents connections from becoming infinitely strong
    /// - Reflects the biological reality that synapses have maximum possible strengths
    /// - Helps maintain stability in the network
    /// 
    /// Think of it as a limit to how strong any single connection can become,
    /// just like there are physical limits to real neural connections.
    /// </para>
    /// </remarks>
    private readonly double _maxWeight = 1.0;

    /// <summary>
    /// The minimum allowed value for a synaptic weight.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field defines the lower bound for synaptic weights, ensuring that no connection becomes arbitrarily weak
    /// or strongly inhibitory. Typically set to 0 to prevent negative weights.
    /// </para>
    /// <para><b>For Beginners:</b> This is the minimum strength a connection can have.
    /// 
    /// Setting a minimum weight:
    /// - Prevents connections from becoming infinitely weak or negative
    /// - Often set to zero (meaning no connection) in simple models
    /// - In more complex models, might be negative to allow inhibitory connections
    /// 
    /// This represents the weakest possible state of a connection between neurons.
    /// </para>
    /// </remarks>
    private readonly double _minWeight;

    /// <summary>
    /// The activity traces of presynaptic neurons.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the decaying traces of recent presynaptic activity, which are used to implement
    /// spike-timing-dependent plasticity. Each trace decays exponentially over time.
    /// </para>
    /// <para><b>For Beginners:</b> This tracks the recent history of input neuron activity.
    ///
    /// Presynaptic traces work like this:
    /// - When an input neuron spikes (activates strongly), its trace jumps to 1.0
    /// - This trace then gradually fades over time
    /// - The trace represents "this neuron was recently active"
    ///
    /// Think of it like a gradually fading footprint showing which input neurons
    /// were active in the recent past.
    /// </para>
    /// </remarks>
    private Tensor<T> _presynapticTraces;

    /// <summary>
    /// The activity traces of postsynaptic neurons.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the decaying traces of recent postsynaptic activity, which are used to implement
    /// spike-timing-dependent plasticity. Each trace decays exponentially over time.
    /// </para>
    /// <para><b>For Beginners:</b> This tracks the recent history of output neuron activity.
    ///
    /// Postsynaptic traces work the same way as presynaptic traces, but for output neurons:
    /// - They jump to 1.0 when the neuron activates
    /// - Then gradually decay over time
    /// - They help determine which connections should be strengthened or weakened
    ///
    /// These traces allow the network to consider the relative timing between
    /// input and output activity, which is crucial for spike-timing-dependent plasticity.
    /// </para>
    /// </remarks>
    private Tensor<T> _postsynapticTraces;

    /// <summary>
    /// The current spike state of presynaptic neurons (binary).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the current spike state of each presynaptic neuron, with 1 indicating a spike and 0 indicating
    /// no spike. A spike occurs when the neuron's activation exceeds a threshold (typically 0.5).
    /// </para>
    /// <para><b>For Beginners:</b> This records which input neurons are currently firing.
    ///
    /// Spikes are:
    /// - Binary events (either a neuron spikes or it doesn't)
    /// - Determined by whether the neuron's activation exceeds a threshold
    /// - How real neurons communicate in the brain
    ///
    /// In this model, an input value above 0.5 is considered a spike, which is a simplified
    /// version of how biological neurons generate electrical impulses when sufficiently activated.
    /// </para>
    /// </remarks>
    private Tensor<T> _presynapticSpikes;

    /// <summary>
    /// The current spike state of postsynaptic neurons (binary).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the current spike state of each postsynaptic neuron, with 1 indicating a spike and 0 indicating
    /// no spike. A spike occurs when the neuron's activation exceeds a threshold (typically 0.5).
    /// </para>
    /// <para><b>For Beginners:</b> This records which output neurons are currently firing.
    ///
    /// Just like with input neurons:
    /// - Output neurons either spike (1) or don't spike (0)
    /// - A spike happens when activation exceeds the threshold
    /// - This binary state is used in determining how connections should change
    ///
    /// The combination of current spikes and spike traces allows the network to implement
    /// timing-dependent learning rules.
    /// </para>
    /// </remarks>
    private Tensor<T> _postsynapticSpikes;

    /// <summary>
    /// The decay rate of activity traces.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field controls how quickly the activity traces decay over time. A value closer to 1 means traces persist
    /// longer, while a value closer to 0 means they decay rapidly.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how quickly the memory of recent activity fades.
    /// 
    /// The trace decay:
    /// - Determines how long a neuron's activity is remembered
    /// - Affects the time window in which connections can be modified
    /// - Influences the timing sensitivity of the learning
    /// 
    /// A higher value (closer to 1) means:
    /// - Longer-lasting traces
    /// - Larger time windows for plasticity
    /// - Less precise timing sensitivity
    /// 
    /// For example, a decay rate of 0.95 means each trace retains 95% of its value
    /// with each time step, creating a gradually fading memory of recent activity.
    /// </para>
    /// </remarks>
    private readonly double _traceDecay;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// <c>true</c> for this layer, as it implements synaptic plasticity rules for learning.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates whether the synaptic plasticity layer can be trained. Since this layer implements
    /// biologically-inspired learning rules, it supports training, although the mechanism differs from the
    /// standard backpropagation approach.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer has internal values (synaptic weights) that can be adjusted during training
    /// - It will improve its performance as it sees more data
    /// - It participates in the learning process
    /// 
    /// For this layer, the value is always true because its whole purpose is to implement
    /// biologically-inspired learning rules that modify connection strengths based on experience.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => false;

    /// <summary>
    /// Initializes a new instance of the <see cref="SynapticPlasticityLayer{T}"/> class.
    /// </summary>
    /// <param name="size">The number of neurons in the layer.</param>
    /// <param name="stdpLtpRate">The rate of long-term potentiation (strengthening). Default is 0.005.</param>
    /// <param name="stdpLtdRate">The rate of long-term depression (weakening). Default is 0.0025.</param>
    /// <param name="homeostasisRate">The rate of homeostatic regulation. Default is 0.0001.</param>
    /// <param name="minWeight">The minimum allowed weight value. Default is 0.</param>
    /// <param name="maxWeight">The maximum allowed weight value. Default is 1.</param>
    /// <param name="traceDecay">The decay rate for activity traces. Default is 0.95.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a synaptic plasticity layer with the specified number of neurons and plasticity parameters.
    /// The layer maintains a full connectivity matrix between all neurons, with weights initialized to small random values.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new synaptic plasticity layer.
    /// 
    /// The parameters you provide determine:
    /// - size: How many neurons are in the layer
    /// - stdpLtpRate: How quickly connections strengthen (higher = faster learning)
    /// - stdpLtdRate: How quickly connections weaken (higher = faster forgetting)
    /// - homeostasisRate: How strongly the system maintains balance (higher = more aggressive balancing)
    /// - minWeight/maxWeight: The range of possible connection strengths
    /// - traceDecay: How quickly the memory of recent activity fades
    /// 
    /// These settings control the learning dynamics and how the layer will adapt to patterns over time.
    /// </para>
    /// </remarks>
    public SynapticPlasticityLayer(int size, double stdpLtpRate = 0.005,
        double stdpLtdRate = 0.0025, double homeostasisRate = 0.0001, double minWeight = 0, double maxWeight = 1, double traceDecay = 0.95) : base([size], [size])
    {
        // Initialize cached state tensors
        _lastInput = new Tensor<T>([size]);
        _lastInput.Fill(NumOps.Zero);
        _lastOutput = new Tensor<T>([size]);
        _lastOutput.Fill(NumOps.Zero);

        _stdpLtpRate = stdpLtpRate;
        _stdpLtdRate = stdpLtdRate;
        _homeostasisRate = homeostasisRate;
        _minWeight = minWeight;
        _maxWeight = maxWeight;
        _traceDecay = traceDecay;

        // Initialize weights with small random values
        _weights = Tensor<T>.CreateRandom([size, size]);

        // Initialize trace and spike tensors
        _presynapticTraces = new Tensor<T>([size]);
        _presynapticTraces.Fill(NumOps.Zero);
        _postsynapticTraces = new Tensor<T>([size]);
        _postsynapticTraces.Fill(NumOps.Zero);
        _presynapticSpikes = new Tensor<T>([size]);
        _presynapticSpikes.Fill(NumOps.Zero);
        _postsynapticSpikes = new Tensor<T>([size]);
        _postsynapticSpikes.Fill(NumOps.Zero);
    }

    /// <summary>
    /// Performs the forward pass of the synaptic plasticity layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor (same as input for this pass-through layer).</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the synaptic plasticity layer. As a pass-through layer, it simply
    /// records the input and returns it unchanged. The actual learning occurs during the update step.
    /// </para>
    /// <para><b>For Beginners:</b> This method passes the input data through the layer unchanged.
    /// 
    /// During the forward pass:
    /// - The layer receives input activations
    /// - It stores these activations for later use in learning
    /// - It returns the same activations without modification
    /// 
    /// This layer doesn't change the data during the forward pass because:
    /// - It functions as a "pass-through" layer
    /// - The learning happens during the update step, not the forward pass
    /// - This matches how biological plasticity works (neurons transmit signals unchanged,
    ///   but their connections change strength afterward)
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Flatten to 1D tensor if needed
        var inputFlat = input.Shape.Length == 1
            ? input
            : input.Reshape([input.Length]);

        // Store for STDP learning
        _lastInput = inputFlat;
        _lastOutput = inputFlat; // Pass-through layer

        return input;
    }

    /// <summary>
    /// Performs the backward pass of the synaptic plasticity layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input (same as output gradient for this pass-through layer).</returns>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the synaptic plasticity layer. As a pass-through layer, it simply
    /// passes the gradient back without modification. The actual weight updates are handled in the UpdateParameters method.
    /// </para>
    /// <para><b>For Beginners:</b> This method passes the gradient unchanged back to the previous layer.
    /// 
    /// During the backward pass:
    /// - The layer receives error gradients from the next layer
    /// - It passes these gradients back without changing them
    /// - No learning happens in this step for this particular layer
    /// 
    /// This layer uses a different learning mechanism than backpropagation:
    /// - Instead of using gradients directly, it uses spike timing relationships
    /// - The actual learning happens in the UpdateParameters method
    /// - This backward pass is only needed to maintain compatibility with the neural network framework
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
        // This is a pass-through layer, so we simply pass the gradient back
        // No weight updates are performed here as they're handled in UpdateParameters
        return outputGradient;
    }

    /// <summary>
    /// Backward pass implementation using automatic differentiation.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method uses automatic differentiation to compute gradients. Specialized operations
    /// are not yet available in TensorOperations, so this falls back to the manual implementation.
    /// </para>
    /// </remarks>
    private Tensor<T> BackwardViaAutodiff(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Identity passthrough graph for gradients
        var inputNode = Autodiff.TensorOperations<T>.Variable(_lastInput, "stdp_input", requiresGradient: true);
        var outputNode = inputNode;
        outputNode.Gradient = outputGradient;

        // Inline topological sort
        var visited = new HashSet<Autodiff.ComputationNode<T>>();
        var topoOrder = new List<Autodiff.ComputationNode<T>>();
        var stack = new Stack<(Autodiff.ComputationNode<T> node, bool processed)>();
        stack.Push((outputNode, false));

        while (stack.Count > 0)
        {
            var (node, processed) = stack.Pop();
            if (visited.Contains(node)) continue;

            if (processed)
            {
                visited.Add(node);
                topoOrder.Add(node);
            }
            else
            {
                stack.Push((node, true));
                if (node.Parents != null)
                {
                    foreach (var parent in node.Parents)
                    {
                        if (!visited.Contains(parent))
                            stack.Push((parent, false));
                    }
                }
            }
        }

        for (int i = topoOrder.Count - 1; i >= 0; i--)
        {
            var node = topoOrder[i];
            if (node.RequiresGradient && node.BackwardFunction != null && node.Gradient != null)
            {
                node.BackwardFunction(node.Gradient);
            }
        }

        return inputNode.Gradient ?? throw new InvalidOperationException("Gradient computation failed in synaptic plasticity autodiff.");
    }


    /// <summary>
    /// Updates the synaptic weights based on spike-timing-dependent plasticity rules.
    /// </summary>
    /// <param name="learningRate">A global modulation factor for the learning process.</param>
    /// <remarks>
    /// <para>
    /// This method applies spike-timing-dependent plasticity (STDP) rules to update the synaptic weights based on
    /// the relative timing of pre- and post-synaptic activity. It implements long-term potentiation (LTP) when presynaptic
    /// activity precedes postsynaptic activity, and long-term depression (LTD) in the reverse case. It also applies
    /// homeostatic mechanisms to maintain network stability.
    /// </para>
    /// <para><b>For Beginners:</b> This method is where the actual learning happens, mimicking how real neurons modify their connections.
    /// 
    /// The learning process involves several steps:
    /// 
    /// 1. Update trace records:
    ///    - Existing traces decay slightly (like memories fading)
    ///    - New spikes are recorded, setting traces to 1.0 for active neurons
    /// 
    /// 2. For each connection between neurons (i ? j):
    ///    
    ///    a) If neuron i fired and neuron j was recently active (pre before post):
    ///       - Strengthen the connection (long-term potentiation)
    ///       - This reinforces connections where one neuron might have caused the other to fire
    ///    
    ///    b) If neuron j fired and neuron i was recently active (post before pre):
    ///       - Weaken the connection (long-term depression)
    ///       - This weakens connections with incorrect timing relationships
    ///    
    ///    c) Apply homeostasis to maintain balance:
    ///       - Very strong connections become slightly weaker
    ///       - Very weak connections become slightly stronger
    ///       - This prevents all weights from becoming maximum or minimum
    /// 
    /// 3. Ensure all weights stay within allowed limits
    /// 
    /// This biologically-inspired learning process helps the network discover patterns
    /// and temporal relationships in the data without using traditional backpropagation.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        int size = GetInputShape()[0];

        // Update spike traces (exponential decay)
        for (int i = 0; i < size; i++)
        {
            // Decay traces over time
            _presynapticTraces[i] = NumOps.Multiply(_presynapticTraces[i], NumOps.FromDouble(_traceDecay));
            _postsynapticTraces[i] = NumOps.Multiply(_postsynapticTraces[i], NumOps.FromDouble(_traceDecay));

            // Record new spikes (assuming binary activation where 1.0 = spike)
            if (NumOps.GreaterThan(_lastInput[i], NumOps.FromDouble(0.5)))
            {
                _presynapticSpikes[i] = NumOps.One;
                _presynapticTraces[i] = NumOps.One; // Set trace to 1.0 when spike occurs
            }
            else
            {
                _presynapticSpikes[i] = NumOps.Zero;
            }

            if (NumOps.GreaterThan(_lastOutput[i], NumOps.FromDouble(0.5)))
            {
                _postsynapticSpikes[i] = NumOps.One;
                _postsynapticTraces[i] = NumOps.One; // Set trace to 1.0 when spike occurs
            }
            else
            {
                _postsynapticSpikes[i] = NumOps.Zero;
            }
        }

        // Apply STDP rule to update weights
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                // Skip self-connections
                if (i == j) continue;

                T currentWeight = _weights[i, j];
                T weightChange = NumOps.Zero;

                // LTP: If presynaptic neuron fired before postsynaptic neuron
                if (NumOps.Equals(_presynapticSpikes[i], NumOps.One) &&
                    NumOps.GreaterThan(_postsynapticTraces[j], NumOps.Zero))
                {
                    // The strength of potentiation depends on the postsynaptic trace
                    T potentiation = NumOps.Multiply(
                        NumOps.FromDouble(_stdpLtpRate),
                        NumOps.Multiply(_postsynapticTraces[j],
                            NumOps.Subtract(NumOps.FromDouble(_maxWeight), currentWeight))
                    );
                    weightChange = NumOps.Add(weightChange, potentiation);
                }

                // LTD: If postsynaptic neuron fired before presynaptic neuron
                if (NumOps.Equals(_postsynapticSpikes[j], NumOps.One) &&
                    NumOps.GreaterThan(_presynapticTraces[i], NumOps.Zero))
                {
                    // The strength of depression depends on the presynaptic trace
                    T depression = NumOps.Multiply(
                        NumOps.FromDouble(_stdpLtdRate),
                        NumOps.Multiply(_presynapticTraces[i],
                            NumOps.Subtract(currentWeight, NumOps.FromDouble(_minWeight)))
                    );
                    weightChange = NumOps.Subtract(weightChange, depression);
                }

                // Apply calcium-based metaplasticity (homeostasis)
                // If a synapse is very strong, make it harder to strengthen further
                T homeostasisFactor = NumOps.Multiply(
                    NumOps.FromDouble(_homeostasisRate),
                    NumOps.Subtract(currentWeight, NumOps.FromDouble(0.5))
                );
                weightChange = NumOps.Subtract(weightChange, homeostasisFactor);

                // Apply neuromodulation (using the provided learning rate as a global modulator)
                weightChange = NumOps.Multiply(weightChange, learningRate);

                // Update weight
                _weights[i, j] = NumOps.Add(currentWeight, weightChange);

                // Ensure weight stays within bounds
                if (NumOps.LessThan(_weights[i, j], NumOps.FromDouble(_minWeight)))
                    _weights[i, j] = NumOps.FromDouble(_minWeight);
                if (NumOps.GreaterThan(_weights[i, j], NumOps.FromDouble(_maxWeight)))
                    _weights[i, j] = NumOps.FromDouble(_maxWeight);
            }
        }
    }

    /// <summary>
    /// Gets all trainable parameters of the layer as a single vector.
    /// </summary>
    /// <returns>A vector containing the weight matrix parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the weight matrix as a flattened vector. Although this layer primarily
    /// uses STDP learning rules, exposing parameters allows for saving/loading state.
    /// </para>
    /// <para><b>For Beginners:</b> This method returns the layer's weights for saving or inspection.
    ///
    /// While the layer uses spike-timing-dependent plasticity rules for learning,
    /// it still has parameters (the weight matrix) that can be:
    /// - Saved to disk
    /// - Loaded from a previously trained model
    /// - Inspected for analysis
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Return the weight tensor as a flattened vector
        return new Vector<T>(_weights.ToArray());
    }

    /// <summary>
    /// Sets the trainable parameters of the layer from a single vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets the weight matrix from a flattened vector. This is useful for loading
    /// saved model weights or for implementing optimization algorithms.
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        int size = GetInputShape()[0];
        int expectedParams = size * size;

        if (parameters.Length != expectedParams)
        {
            throw new ArgumentException($"Expected {expectedParams} parameters, but got {parameters.Length}");
        }

        // Restore weights without hot-path conversions
        _weights = new Tensor<T>(new[] { size, size }, parameters);
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the synaptic plasticity layer by clearing the last input and output
    /// vectors. This can be useful when processing new, unrelated sequences or when restarting training.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory of recent activity.
    /// 
    /// When resetting the state:
    /// - The layer forgets what inputs and outputs it recently saw
    /// - This is useful when starting to process a new, unrelated example
    /// - It prevents information from one sequence affecting another
    /// 
    /// Note that this doesn't reset the learned weights, only the temporary state variables.
    /// Think of it like clearing short-term memory while keeping long-term memories intact.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Reset the internal state of the layer using Tensor<T>
        int size = GetInputShape()[0];

        _lastInput = new Tensor<T>([size]);
        _lastInput.Fill(NumOps.Zero);
        _lastOutput = new Tensor<T>([size]);
        _lastOutput.Fill(NumOps.Zero);

        // Also reset traces and spikes
        _presynapticTraces = new Tensor<T>([size]);
        _presynapticTraces.Fill(NumOps.Zero);
        _postsynapticTraces = new Tensor<T>([size]);
        _postsynapticTraces.Fill(NumOps.Zero);
        _presynapticSpikes = new Tensor<T>([size]);
        _presynapticSpikes.Fill(NumOps.Zero);
        _postsynapticSpikes = new Tensor<T>([size]);
        _postsynapticSpikes.Fill(NumOps.Zero);
    }

    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape == null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        if (inputNodes.Count == 0)
            throw new ArgumentException("At least one input node is required.", nameof(inputNodes));

        // SynapticPlasticityLayer JIT provides a differentiable approximation of STDP:
        // The forward pass is a simple weighted transformation: output = W @ input
        // The STDP learning rule is approximated through standard gradient descent
        // during backpropagation.

        var input = inputNodes[0];

        // Get dimensions
        int inputSize = _weights.Shape[1];
        int outputSize = _weights.Shape[0];

        // Create weights constant
        var weightsNode = TensorOperations<T>.Constant(_weights, "stdp_weights");

        // Reshape input for matrix multiplication
        var inputReshaped = TensorOperations<T>.Reshape(input, inputSize, 1);

        // Forward: W @ input
        var weighted = TensorOperations<T>.MatrixMultiply(weightsNode, inputReshaped);
        var output = TensorOperations<T>.Reshape(weighted, outputSize);

        // Apply activation
        output = ApplyActivationToGraph(output);

        return output;
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    /// <value>
    /// Always <c>true</c>. SynapticPlasticityLayer uses a differentiable forward pass.
    /// </value>
    /// <remarks>
    /// <para>
    /// JIT compilation for SynapticPlasticity exports the forward pass as a simple
    /// matrix multiplication. The STDP learning dynamics are approximated through
    /// standard gradient-based optimization during training. The temporal spike
    /// timing information is not used in the JIT-compiled forward pass.
    /// </para>
    /// </remarks>
    public override bool SupportsJitCompilation => true;

}

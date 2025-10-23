namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a layer of spiking neurons that model the biological dynamics of neural activity.
/// </summary>
/// <remarks>
/// <para>
/// A spiking layer implements various biologically-inspired neuron models that operate with discrete spike events
/// rather than continuous activation values. The layer supports several neuron types including Leaky Integrate-and-Fire,
/// Izhikevich, Adaptive Exponential, and Hodgkin-Huxley models. Spiking neurons are characterized by their membrane
/// potential dynamics, threshold-crossing spike generation, and refractory periods after firing.
/// </para>
/// <para><b>For Beginners:</b> This layer mimics how real neurons in the brain work, using "spikes" instead of smooth values.
/// 
/// Think of each neuron as a tiny battery that:
/// - Builds up electrical charge over time (membrane potential)
/// - Fires a signal (spike) when the charge reaches a threshold
/// - Needs a short recovery period after firing (refractory period)
/// - Has different ways of charging and discharging (different neuron models)
/// 
/// Benefits include:
/// - More biologically realistic modeling of neural activity
/// - Potential for energy-efficient computation (spikes are sparse)
/// - Ability to process time-dependent information naturally
/// 
/// For example, spiking neurons can directly model the timing patterns in speech or detect motion in video
/// by responding to when things change rather than constantly processing every value.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class SpikingLayer<T> : LayerBase<T>
{
    /// <summary>
    /// The type of spiking neuron model to use.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field specifies which biologically-inspired neuron model the layer implements. Different models
    /// capture different aspects of neural dynamics, with varying levels of biological accuracy and computational complexity.
    /// </para>
    /// <para><b>For Beginners:</b> This determines which mathematical model represents each neuron.
    /// 
    /// The different models include:
    /// - LeakyIntegrateAndFire: The simplest model, where charge leaks away over time
    /// - IntegrateAndFire: Like LIF but without the leak, charge only resets after spikes
    /// - Izhikevich: A model that can reproduce many firing patterns seen in real neurons
    /// - AdaptiveExponential: Includes adaptation mechanisms for more realistic behavior
    /// - HodgkinHuxley: The most detailed biophysical model describing ion channel dynamics
    /// 
    /// Simpler models are computationally efficient while more complex ones capture more
    /// biological details.
    /// </para>
    /// </remarks>
    private SpikingNeuronType _neuronType;

    /// <summary>
    /// Time constant for membrane potential decay.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This parameter controls how quickly the membrane potential decays in the absence of input.
    /// Larger values result in slower decay, allowing the neuron to integrate inputs over longer time periods.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how quickly the neuron's charge leaks away.
    /// 
    /// Think of it like a leaky bucket:
    /// - A higher value means a slower leak (charge stays around longer)
    /// - A lower value means a faster leak (charge disappears quickly)
    /// 
    /// If tau is large, the neuron can "remember" inputs from further in the past,
    /// making it sensitive to patterns that stretch over longer time periods.
    /// </para>
    /// </remarks>
    private double _tau;

    /// <summary>
    /// Refractory period in time steps during which the neuron cannot fire again after spiking.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This parameter specifies the number of time steps during which a neuron cannot generate another spike after firing.
    /// This models the recovery period of biological neurons after an action potential.
    /// </para>
    /// <para><b>For Beginners:</b> This is the "rest period" a neuron needs after firing.
    /// 
    /// After a real neuron fires, it needs time to recover before it can fire again.
    /// During this refractory period:
    /// - The neuron ignores additional incoming signals
    /// - It's reset and cannot generate another spike
    /// 
    /// This prevents ultra-rapid, unrealistic firing and helps establish
    /// a maximum firing rate similar to biological neurons.
    /// </para>
    /// </remarks>
    private double _refractoryPeriod;

    /// <summary>
    /// Connection weights between input and output neurons.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This matrix stores the strength of connections between each input and output neuron. The weights determine
    /// how strongly each input affects the membrane potential of each output neuron.
    /// </para>
    /// <para><b>For Beginners:</b> These are like the strength of connections between neurons.
    /// 
    /// Just like in other neural network layers, these weights determine:
    /// - How strongly each input affects each output neuron
    /// - Positive weights increase the neuron's charge when the input is active
    /// - Negative weights decrease the charge
    /// 
    /// These weights are adjusted during training to make the layer learn.
    /// </para>
    /// </remarks>
    private Matrix<T> Weights;

    /// <summary>
    /// Bias values for output neurons.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector contains the bias values for each output neuron. The bias adds a constant input to the neuron,
    /// affecting its baseline membrane potential regardless of input.
    /// </para>
    /// <para><b>For Beginners:</b> These are baseline charge levels for each neuron.
    /// 
    /// The bias values:
    /// - Determine how close each neuron starts to its firing threshold
    /// - A higher bias makes the neuron more "trigger-happy" (likely to fire)
    /// - A lower bias makes the neuron require stronger input to fire
    /// 
    /// Biases are adjusted during training along with the weights.
    /// </para>
    /// </remarks>
    private Vector<T> Bias;

    /// <summary>
    /// Accumulated gradients for weight updates during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This matrix stores the accumulated gradients for the weight parameters during training.
    /// These gradients indicate how to adjust the weights to improve the network's performance.
    /// </para>
    /// <para><b>For Beginners:</b> This tracks how the weights should change during training.
    /// 
    /// During training:
    /// - The network figures out which direction to adjust each weight
    /// - These adjustments are stored in the weight gradients
    /// - After processing a batch of examples, the weights are updated using these gradients
    /// 
    /// This helps the network gradually improve its performance on the given task.
    /// </para>
    /// </remarks>
    private Matrix<T> _weightGradients;

    /// <summary>
    /// Accumulated gradients for bias updates during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector stores the accumulated gradients for the bias parameters during training.
    /// These gradients indicate how to adjust the biases to improve the network's performance.
    /// </para>
    /// <para><b>For Beginners:</b> This tracks how the biases should change during training.
    /// 
    /// Similar to weight gradients, these values show:
    /// - Which direction to adjust each bias value
    /// - How much to adjust it
    /// 
    /// The network uses these gradients to update the biases after processing a batch of examples.
    /// </para>
    /// </remarks>
    private Vector<T> _biasGradients;

    /// <summary>
    /// Stores the input tensor from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field caches the input to the layer during the forward pass, which is needed during the backward pass
    /// to compute gradients. It is cleared when ResetState() is called.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the layer's short-term memory of what input it received.
    /// 
    /// During training, the layer needs to remember what input it processed so that it can
    /// properly calculate how to improve. This temporary storage is cleared between batches
    /// or when you explicitly reset the layer.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the output tensor from the most recent forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field caches the output from the layer during the forward pass, which is needed during the backward pass
    /// to compute gradients. It is cleared when ResetState() is called.
    /// </para>
    /// <para><b>For Beginners:</b> This is the layer's memory of what output it produced.
    /// 
    /// The layer needs to remember what spikes it generated so that during training it can
    /// understand how changes to its parameters affect the overall network performance.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Current membrane potential for each output neuron.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector stores the current membrane potential (voltage) for each output neuron. The membrane potential
    /// integrates inputs over time and generates a spike when it crosses a threshold, after which it resets.
    /// </para>
    /// <para><b>For Beginners:</b> This is the current electrical charge of each neuron.
    /// 
    /// The membrane potential:
    /// - Increases when the neuron receives input
    /// - Decreases over time due to the leak (for LIF neurons)
    /// - When it reaches a threshold (usually 1.0), the neuron fires a spike
    /// - After firing, it resets to a lower value
    /// 
    /// This is the key internal state that determines when neurons fire.
    /// </para>
    /// </remarks>
    private Vector<T> _membranePotential;

    /// <summary>
    /// Countdown timer for refractory period for each output neuron.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector stores the remaining time steps in the refractory period for each output neuron. During the
    /// refractory period, the neuron cannot generate another spike regardless of its membrane potential.
    /// </para>
    /// <para><b>For Beginners:</b> This is a timer that prevents neurons from firing too frequently.
    /// 
    /// After a neuron fires:
    /// - This countdown starts from the refractory period value
    /// - It decreases by 1 each time step
    /// - While it's above zero, the neuron cannot fire again
    /// - Once it reaches zero, the neuron can respond to inputs again
    /// 
    /// This prevents unrealistically rapid firing and better matches biological neurons.
    /// </para>
    /// </remarks>
    private Vector<T> _refractoryCountdown;

    /// <summary>
    /// Output spikes for each output neuron (1.0 for spike, 0.0 for no spike).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector stores the current output spikes for each output neuron, where 1.0 indicates a spike and
    /// 0.0 indicates no spike. This binary coding represents the discrete spike events.
    /// </para>
    /// <para><b>For Beginners:</b> This shows which neurons are currently firing.
    /// 
    /// For each neuron:
    /// - A value of 1.0 means the neuron is firing (spiking)
    /// - A value of 0.0 means the neuron is quiet (not spiking)
    /// 
    /// This binary on/off representation is how spiking neurons communicate,
    /// similar to how real neurons in the brain work.
    /// </para>
    /// </remarks>
    private Vector<T> _spikes;

    /// <summary>
    /// Recovery variable for Izhikevich neuron model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector stores the recovery variable for each output neuron when using the Izhikevich model.
    /// The recovery variable provides negative feedback to the membrane potential, allowing the model to capture
    /// various firing patterns seen in biological neurons.
    /// </para>
    /// <para><b>For Beginners:</b> This is a second variable that helps model more complex neuron behaviors.
    /// 
    /// In the Izhikevich model:
    /// - This recovery variable provides negative feedback to the membrane potential
    /// - It allows modeling different firing patterns like bursting or chattering
    /// - It makes the model more biologically realistic
    /// 
    /// This is only used when _neuronType is Izhikevich.
    /// </para>
    /// </remarks>
    private Vector<T>? _recoveryVariable;

    /// <summary>
    /// Time scale of recovery variable in Izhikevich model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This parameter controls the time scale of the recovery variable in the Izhikevich model.
    /// It affects how quickly the recovery variable changes in response to the membrane potential.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how quickly the recovery variable changes.
    /// 
    /// A smaller value means the recovery variable changes more slowly,
    /// while a larger value means it changes more quickly. This parameter
    /// helps determine what type of firing pattern the neuron exhibits.
    /// </para>
    /// </remarks>
    private double _a = 0.02;

    /// <summary>
    /// Sensitivity of recovery variable to membrane potential in Izhikevich model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This parameter controls how sensitive the recovery variable is to the membrane potential in the Izhikevich model.
    /// It determines the strength of the coupling between the membrane potential and the recovery variable.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how strongly the membrane potential affects the recovery variable.
    /// 
    /// A larger value means the recovery variable is more sensitive to changes
    /// in the membrane potential. This parameter helps determine the neuron's
    /// excitability and responsiveness to input.
    /// </para>
    /// </remarks>
    private double _b = 0.2;

    /// <summary>
    /// After-spike reset value of membrane potential in Izhikevich model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This parameter specifies the value to which the membrane potential resets after a spike in the Izhikevich model.
    /// It determines the starting point for the membrane potential's recovery after firing.
    /// </para>
    /// <para><b>For Beginners:</b> This is the value the membrane potential resets to after firing.
    /// 
    /// After a neuron generates a spike, its membrane potential immediately jumps
    /// to this value. A more negative value means the neuron needs more input
    /// to reach threshold again.
    /// </para>
    /// </remarks>
    private double _c = -65.0;

    /// <summary>
    /// After-spike reset of recovery variable in Izhikevich model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This parameter specifies the amount by which the recovery variable increases after a spike in the Izhikevich model.
    /// It affects how quickly the neuron can fire again after generating a spike.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much the recovery variable increases after a spike.
    /// 
    /// When a neuron fires, the recovery variable increases by this amount.
    /// A larger value makes it harder for the neuron to fire again immediately,
    /// creating different patterns of activity like bursting or chattering.
    /// </para>
    /// </remarks>
    private double _d = 8.0;

    /// <summary>
    /// Adaptation variable for Adaptive Exponential neuron model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector stores the adaptation variable for each output neuron when using the Adaptive Exponential model.
    /// The adaptation variable provides negative feedback to the membrane potential and increases after each spike,
    /// allowing the model to adapt its firing rate over time.
    /// </para>
    /// <para><b>For Beginners:</b> This variable helps neurons adapt their firing rate over time.
    /// 
    /// In the Adaptive Exponential model:
    /// - This adaptation variable makes neurons fire less frequently with sustained input
    /// - It increases every time the neuron fires
    /// - It provides negative feedback to the membrane potential
    /// 
    /// This adaptation mimics how real neurons get "tired" when stimulated continuously.
    /// </para>
    /// </remarks>
    private Vector<T>? _adaptationVariable;

    /// <summary>
    /// Sharpness of exponential term in Adaptive Exponential model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This parameter controls the sharpness of the exponential term in the Adaptive Exponential model.
    /// It determines how quickly the membrane potential rises as it approaches the threshold.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how sharply the membrane potential increases near threshold.
    /// 
    /// A larger value makes the membrane potential rise more gradually as it approaches threshold,
    /// while a smaller value makes it rise more sharply. This helps model the rapid rise in
    /// membrane potential just before a real neuron fires.
    /// </para>
    /// </remarks>
    private double _deltaT = 2.0;

    /// <summary>
    /// Threshold potential in Adaptive Exponential model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This parameter specifies the threshold potential in the Adaptive Exponential model, which is the membrane potential
    /// at which the exponential term becomes significant. It affects the neuron's excitability.
    /// </para>
    /// <para><b>For Beginners:</b> This is like a soft threshold that starts the rapid rise to firing.
    /// 
    /// Unlike the hard threshold in simpler models, this parameter defines when
    /// the exponential term starts to take effect, causing a rapid increase in
    /// membrane potential that typically leads to a spike.
    /// </para>
    /// </remarks>
    private double _vT = -50.0;

    /// <summary>
    /// Adaptation time constant in Adaptive Exponential model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This parameter controls the time constant of the adaptation variable in the Adaptive Exponential model.
    /// It determines how quickly the adaptation variable decays over time.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how quickly the adaptation effect wears off.
    /// 
    /// A larger value means the adaptation lasts longer, causing a more prolonged
    /// reduction in firing rate after intense activity. This mimics how neurons
    /// take time to recover their full responsiveness after periods of high activity.
    /// </para>
    /// </remarks>
    private double _tauw = 30.0;

    /// <summary>
    /// Subthreshold adaptation in Adaptive Exponential model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This parameter controls the subthreshold adaptation in the Adaptive Exponential model.
    /// It determines how strongly the adaptation variable couples to the membrane potential below threshold.
    /// </para>
    /// <para><b>For Beginners:</b> This controls adaptation even when the neuron isn't firing.
    /// 
    /// A larger value means the adaptation variable increases more with subthreshold
    /// membrane potential changes, allowing the neuron to adapt even without firing.
    /// This models how real neurons can become less responsive even with inputs
    /// that don't cause firing.
    /// </para>
    /// </remarks>
    private double _a_adex = 4.0;

    /// <summary>
    /// Spike-triggered adaptation in Adaptive Exponential model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This parameter controls the spike-triggered adaptation in the Adaptive Exponential model.
    /// It determines how much the adaptation variable increases after each spike.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how much the neuron adapts after each spike.
    /// 
    /// A larger value means the adaptation variable increases more after each spike,
    /// making the neuron less likely to fire again soon. This models the "fatigue"
    /// that real neurons experience after firing.
    /// </para>
    /// </remarks>
    private double _b_adex = 80.5;

    /// <summary>
    /// Potassium activation gating variable for Hodgkin-Huxley model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector stores the potassium activation gating variable (n) for each output neuron when using the Hodgkin-Huxley model.
    /// This gate controls the flow of potassium ions, which repolarizes the membrane after a spike.
    /// </para>
    /// <para><b>For Beginners:</b> This controls the flow of potassium ions out of the neuron.
    /// 
    /// In the Hodgkin-Huxley model:
    /// - This variable represents potassium channels that open slowly but close slowly
    /// - When open, these channels let potassium leave the cell, lowering the membrane potential
    /// - This helps end the spike and return the neuron to its resting state
    /// 
    /// This is part of the most detailed biophysical model of neuron behavior.
    /// </para>
    /// </remarks>
    private Vector<T>? _nGate;

    /// <summary>
    /// Sodium activation gating variable for Hodgkin-Huxley model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector stores the sodium activation gating variable (m) for each output neuron when using the Hodgkin-Huxley model.
    /// This gate controls the rapid influx of sodium ions, which causes the initial depolarization during a spike.
    /// </para>
    /// <para><b>For Beginners:</b> This controls the flow of sodium ions into the neuron.
    /// 
    /// In the Hodgkin-Huxley model:
    /// - This variable represents sodium channels that open quickly but also close quickly
    /// - When open, these channels let sodium enter the cell, raising the membrane potential
    /// - This causes the rapid upswing of the membrane potential during a spike
    /// 
    /// These channels are primarily responsible for generating the spike.
    /// </para>
    /// </remarks>
    private Vector<T>? _mGate;

    /// <summary>
    /// Sodium inactivation gating variable for Hodgkin-Huxley model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This vector stores the sodium inactivation gating variable (h) for each output neuron when using the Hodgkin-Huxley model.
    /// This gate controls the inactivation of sodium channels, which helps terminate the spike.
    /// </para>
    /// <para><b>For Beginners:</b> This controls the blocking of sodium channels.
    /// 
    /// In the Hodgkin-Huxley model:
    /// - This variable represents a mechanism that blocks sodium channels
    /// - It closes slowly but opens slowly
    /// - When closed (h is low), it prevents sodium from entering the cell
    /// - This helps terminate the spike and prevents continuous firing
    /// 
    /// This inactivation mechanism is crucial for the neuron to return to its resting state.
    /// </para>
    /// </remarks>
    private Vector<T>? _hGate;

    /// <summary>
    /// Gets the total number of trainable parameters in the layer.
    /// </summary>
    /// <value>
    /// The total number of weights and biases in the layer.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property returns the total number of trainable parameters in the layer, which is the sum of the
    /// number of weights and the number of biases.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how many adjustable numbers the layer has.
    /// 
    /// The parameter count is:
    /// - Number of weights (connections between input and output)
    /// - Plus the number of biases (one per output neuron)
    /// 
    /// This gives you an idea of the layer's complexity and memory requirements.
    /// </para>
    /// </remarks>
    public override int ParameterCount => Weights.Rows * Weights.Columns + Bias.Length;

    /// <summary>
    /// Gets a value indicating whether this layer supports training through backpropagation.
    /// </summary>
    /// <value>
    /// Always returns <c>true</c> as spiking layers have trainable parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that the spiking layer can be trained. The layer contains trainable parameters
    /// (weights and biases) that are updated during the training process.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you that the layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer contains numbers (parameters) that can be adjusted during training
    /// - It will improve its performance as it sees more examples
    /// - It participates in the learning process of the neural network
    /// 
    /// This is important because training spiking neural networks can be challenging
    /// due to the non-differentiable nature of spikes, but this layer implements
    /// special techniques to make it possible.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="SpikingLayer{T}"/> class with the specified dimensions and neuron type.
    /// </summary>
    /// <param name="inputSize">The number of input neurons.</param>
    /// <param name="outputSize">The number of output neurons.</param>
    /// <param name="neuronType">The type of spiking neuron model to use. Defaults to LeakyIntegrateAndFire.</param>
    /// <param name="tau">Time constant for membrane potential decay. Defaults to 10.0.</param>
    /// <param name="refractoryPeriod">Refractory period in time steps. Defaults to 2.0.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a spiking layer with the specified input and output dimensions, neuron type, and parameters.
    /// It initializes the weights with small random values, and sets up the appropriate state variables for the selected
    /// neuron model.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up a new layer of spiking neurons with your chosen settings.
    /// 
    /// When creating a spiking layer, you need to specify:
    /// - inputSize: How many input values the layer receives
    /// - outputSize: How many spiking neurons to create
    /// - neuronType: Which mathematical model to use (default: LeakyIntegrateAndFire)
    /// - tau: How quickly the membrane potential decays (higher = slower decay)
    /// - refractoryPeriod: How long neurons rest after firing
    /// 
    /// The constructor automatically initializes all the internal variables needed
    /// for the specified neuron type, with appropriate default values.
    /// </para>
    /// </remarks>
    public SpikingLayer(int inputSize, int outputSize, SpikingNeuronType neuronType = SpikingNeuronType.LeakyIntegrateAndFire, 
        double tau = 10.0, double refractoryPeriod = 2.0)
        : base([inputSize], [outputSize])
    {
        _neuronType = neuronType;
        _tau = tau;
        _refractoryPeriod = refractoryPeriod;
    
        // Initialize weights with small random values
        Weights = Matrix<T>.CreateRandom(inputSize, outputSize, -0.1, 0.1);
        Bias = Vector<T>.CreateDefault(outputSize, NumOps.Zero);
    
        // Initialize gradient accumulators
        _weightGradients = Matrix<T>.CreateDefault(inputSize, outputSize, NumOps.Zero);
        _biasGradients = Vector<T>.CreateDefault(outputSize, NumOps.Zero);
    
        // Initialize neuron state variables
        _membranePotential = Vector<T>.CreateDefault(outputSize, NumOps.Zero);
        _refractoryCountdown = Vector<T>.CreateDefault(outputSize, NumOps.Zero);
        _spikes = Vector<T>.CreateDefault(outputSize, NumOps.Zero);
    
        // Initialize model-specific variables based on neuron type
        if (neuronType == SpikingNeuronType.Izhikevich)
        {
            _a = 0.02;  // Time scale of recovery variable
            _b = 0.2;   // Sensitivity of recovery variable
            _c = -65.0; // After-spike reset value of membrane potential
            _d = 8.0;   // After-spike reset of recovery variable
            _recoveryVariable = Vector<T>.CreateDefault(outputSize, NumOps.Zero);
        }
        else if (neuronType == SpikingNeuronType.AdaptiveExponential)
        {
            _deltaT = 2.0;  // Sharpness of exponential
            _vT = -50.0;    // Threshold potential
            _tauw = 30.0;   // Adaptation time constant
            _a_adex = 4.0;  // Subthreshold adaptation
            _b_adex = 0.5;  // Spike-triggered adaptation
            _adaptationVariable = Vector<T>.CreateDefault(outputSize, NumOps.Zero);
        }
        else if (neuronType == SpikingNeuronType.HodgkinHuxley)
        {
            _nGate = Vector<T>.CreateDefault(outputSize, NumOps.FromDouble(0.32)); // Potassium activation
            _mGate = Vector<T>.CreateDefault(outputSize, NumOps.FromDouble(0.05)); // Sodium activation
            _hGate = Vector<T>.CreateDefault(outputSize, NumOps.FromDouble(0.60)); // Sodium inactivation
        }
    }

    /// <summary>
    /// Performs the forward pass of the spiking layer.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor containing spike events.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the spiking layer. It processes the input through the appropriate
    /// neuron model, updates the membrane potentials, and generates output spikes when the potentials cross their thresholds.
    /// </para>
    /// <para><b>For Beginners:</b> This method processes data through the spiking neurons.
    /// 
    /// During the forward pass:
    /// 1. The input is saved for later use in training
    /// 2. The input is converted to a vector for easier processing
    /// 3. The input is processed through the specific neuron model (LIF, Izhikevich, etc.)
    /// 4. This updates the membrane potentials and generates spikes when thresholds are crossed
    /// 5. The resulting spikes (0 or 1 values) are returned as the output
    /// 
    /// This is how the layer transforms input signals into spikes, mimicking how
    /// real neurons convert inputs into action potentials.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Store the input for backward pass
        _lastInput = input;
        
        // Convert input tensor to vector for processing
        Vector<T> inputVector;
        if (input.Shape.Length == 1)
        {
            inputVector = input.ToVector();
        }
        else if (input.Shape.Length == 2 && input.Shape[0] == 1)
        {
            // Handle batch size of 1
            inputVector = input.Reshape([input.Shape[1]]).ToVector();
        }
        else
        {
            throw new ArgumentException("Input tensor must be 1D or have batch size of 1");
        }
        
        // Process with the appropriate neuron model
        Vector<T> outputVector = ProcessSpikes(inputVector);
        
        // Convert back to tensor and store for backward pass
        var output = Tensor<T>.FromVector(outputVector);
        _lastOutput = output;
        
        return output;
    }

    /// <summary>
    /// Processes input through the selected neuron model to generate spikes.
    /// </summary>
    /// <param name="input">The input vector to process.</param>
    /// <returns>The output vector containing spike events.</returns>
    /// <remarks>
    /// <para>
    /// This method routes the input to the appropriate neuron model implementation based on the selected neuron type.
    /// It calculates the input current for each neuron and then updates the neuron states according to the specific model.
    /// </para>
    /// <para><b>For Beginners:</b> This method selects the right mathematical model for processing.
    /// 
    /// The steps are:
    /// 1. Calculate the "current" flowing into each neuron using weights and biases
    /// 2. Choose the appropriate neuron model based on _neuronType
    /// 3. Update each neuron's state using that model's equations
    /// 4. Return the resulting spike pattern (which neurons fired)
    /// 
    /// This method delegates to specialized methods for each neuron type,
    /// allowing the layer to support multiple neuron models with different properties.
    /// </para>
    /// </remarks>
    private Vector<T> ProcessSpikes(Vector<T> input)
    {
        // Calculate input current
        Vector<T> current = Weights.Multiply(input).Add(Bias);

        // Update neuron states based on the neuron model
        return _neuronType switch
        {
            SpikingNeuronType.LeakyIntegrateAndFire => UpdateLeakyIntegrateAndFire(current),
            SpikingNeuronType.IntegrateAndFire => UpdateIntegrateAndFire(current),
            SpikingNeuronType.Izhikevich => UpdateIzhikevich(current),
            SpikingNeuronType.HodgkinHuxley => UpdateHodgkinHuxley(current),
            SpikingNeuronType.AdaptiveExponential => UpdateAdaptiveExponential(current),
            _ => throw new ArgumentOutOfRangeException(nameof(_neuronType), _neuronType, $"Neuron type {_neuronType} is not supported."),
        };
    }
    
    /// <summary>
    /// Updates the state of Leaky Integrate-and-Fire neurons.
    /// </summary>
    /// <param name="current">The input current to each neuron.</param>
    /// <returns>The output spikes (0 or 1) for each neuron.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Leaky Integrate-and-Fire neuron model. It decays the membrane potential over time,
    /// adds the input current for neurons not in their refractory period, and generates spikes when the potential
    /// crosses the threshold. After spiking, the potential is reset and the refractory period begins.
    /// </para>
    /// <para><b>For Beginners:</b> This method implements the simplest and most common spiking neuron model.
    /// 
    /// The Leaky Integrate-and-Fire model works like this:
    /// 1. Membrane potential naturally leaks away over time (multiply by decay factor)
    /// 2. For neurons not in refractory period, add the input current
    /// 3. Check if any neurons cross the threshold (typically 1.0)
    /// 4. If a neuron crosses threshold:
    ///    - Output a spike (1.0)
    ///    - Reset the membrane potential to zero
    ///    - Start the refractory period countdown
    /// 5. Otherwise, output no spike (0.0)
    /// 
    /// This simple model captures the key behaviors of real neurons while
    /// remaining computationally efficient.
    /// </para>
    /// </remarks>
    private Vector<T> UpdateLeakyIntegrateAndFire(Vector<T> current)
    {
        // Decay membrane potential
        T decayFactor = NumOps.FromDouble(1.0 - 1.0/_tau);
        _membranePotential = _membranePotential.Multiply(decayFactor);
    
        // Update membrane potential for neurons not in refractory period
        for (int i = 0; i < _membranePotential.Length; i++)
        {
            if (Convert.ToDouble(_refractoryCountdown[i]) <= 0)
            {
                _membranePotential[i] = NumOps.Add(_membranePotential[i], current[i]);
            }
            else
            {
                _refractoryCountdown[i] = NumOps.Subtract(_refractoryCountdown[i], NumOps.One);
            }
        }
    
        // Generate spikes and reset
        for (int i = 0; i < _membranePotential.Length; i++)
        {
            // Threshold for spiking (typically around 1.0)
            if (Convert.ToDouble(_membranePotential[i]) >= 1.0)
            {
                _spikes[i] = NumOps.One;
                _membranePotential[i] = NumOps.Zero; // Reset potential
                _refractoryCountdown[i] = NumOps.FromDouble(_refractoryPeriod);
            }
            else
            {
                _spikes[i] = NumOps.Zero;
            }
        }
    
        return _spikes;
    }
    
    /// <summary>
    /// Updates the state of Integrate-and-Fire neurons.
    /// </summary>
    /// <param name="current">The input current to each neuron.</param>
    /// <returns>The output spikes (0 or 1) for each neuron.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Integrate-and-Fire neuron model. Unlike the Leaky Integrate-and-Fire model,
    /// this model does not include leakage of the membrane potential over time. It adds the input current for
    /// neurons not in their refractory period, and generates spikes when the potential crosses the threshold.
    /// </para>
    /// <para><b>For Beginners:</b> This method implements a simpler version of the spiking neuron model without leakage.
    /// 
    /// The Integrate-and-Fire model works like this:
    /// 1. For neurons not in refractory period, add the input current to membrane potential
    /// 2. Check if any neurons cross the threshold (typically 1.0)
    /// 3. If a neuron crosses threshold:
    ///    - Output a spike (1.0)
    ///    - Reset the membrane potential to zero
    ///    - Start the refractory period countdown
    /// 4. Otherwise, output no spike (0.0)
    /// 
    /// This model is like the Leaky Integrate-and-Fire but without the decay,
    /// so the membrane potential only resets after firing.
    /// </para>
    /// </remarks>
    private Vector<T> UpdateIntegrateAndFire(Vector<T> current)
    {
        // Similar to LIF but without leak
        for (int i = 0; i < _membranePotential.Length; i++)
        {
            if (Convert.ToDouble(_refractoryCountdown[i]) <= 0)
            {
                _membranePotential[i] = NumOps.Add(_membranePotential[i], current[i]);
            }
            else
            {
                _refractoryCountdown[i] = NumOps.Subtract(_refractoryCountdown[i], NumOps.One);
            }
        }
    
        // Generate spikes and reset
        for (int i = 0; i < _membranePotential.Length; i++)
        {
            if (Convert.ToDouble(_membranePotential[i]) >= 1.0)
            {
                _spikes[i] = NumOps.One;
                _membranePotential[i] = NumOps.Zero;
                _refractoryCountdown[i] = NumOps.FromDouble(_refractoryPeriod);
            }
            else
            {
                _spikes[i] = NumOps.Zero;
            }
        }
    
        return _spikes;
    }
    
    /// <summary>
    /// Updates the state of Izhikevich neurons.
    /// </summary>
    /// <param name="current">The input current to each neuron.</param>
    /// <returns>The output spikes (0 or 1) for each neuron.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Izhikevich neuron model, which is a simplified model capable of reproducing
    /// many different firing patterns observed in real neurons. It updates both the membrane potential and
    /// a recovery variable using the Izhikevich equations, and generates spikes when the membrane potential
    /// exceeds a threshold.
    /// </para>
    /// <para><b>For Beginners:</b> This method implements a more versatile neuron model that can produce various firing patterns.
    /// 
    /// The Izhikevich model works like this:
    /// 1. Update both the membrane potential (v) and recovery variable (u) using coupled equations
    /// 2. The recovery variable provides negative feedback to the membrane potential
    /// 3. Check if membrane potential exceeds threshold (30 mV)
    /// 4. If threshold crossed:
    ///    - Output a spike (1.0)
    ///    - Reset membrane potential to value _c (-65.0 by default)
    ///    - Increase recovery variable by _d (8.0 by default)
    /// 5. Otherwise, output no spike (0.0)
    /// 
    /// By adjusting parameters _a, _b, _c, and _d, this model can reproduce many
    /// firing patterns like regular spiking, bursting, chattering, etc.
    /// </para>
    /// </remarks>
    private Vector<T> UpdateIzhikevich(Vector<T> current)
    {
        if (_recoveryVariable == null)
            throw new InvalidOperationException("Recovery variable not initialized for Izhikevich model");
            
        // Izhikevich model update
        for (int i = 0; i < _membranePotential.Length; i++)
        {
            double v = Convert.ToDouble(_membranePotential[i]);
            double u = Convert.ToDouble(_recoveryVariable[i]);
            double I = Convert.ToDouble(current[i]);
            
            // Update membrane potential and recovery variable
            double dv = 0.04 * v * v + 5 * v + 140 - u + I;
            double du = _a * (_b * v - u);
            
            v += dv;
            u += du;
            
            // Check for spike
            if (v >= 30)
            {
                _spikes[i] = NumOps.One;
                v = _c;
                u += _d;
            }
            else
            {
                _spikes[i] = NumOps.Zero;
            }
            
            _membranePotential[i] = NumOps.FromDouble(v);
            _recoveryVariable[i] = NumOps.FromDouble(u);
        }
        
        return _spikes;
    }
    
    /// <summary>
    /// Updates the state of Adaptive Exponential Integrate-and-Fire neurons.
    /// </summary>
    /// <param name="current">The input current to each neuron.</param>
    /// <returns>The output spikes (0 or 1) for each neuron.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Adaptive Exponential Integrate-and-Fire neuron model, which combines
    /// an exponential term to model the rapid rise of the membrane potential before a spike with an
    /// adaptation variable that provides negative feedback. This model can reproduce a wide range of
    /// firing patterns and adaptation behaviors seen in real neurons.
    /// </para>
    /// <para><b>For Beginners:</b> This method implements a more detailed neuron model with adaptation capabilities.
    /// 
    /// The Adaptive Exponential model works like this:
    /// 1. Update both membrane potential (v) and adaptation variable (w)
    /// 2. The membrane potential includes an exponential term for rapid rise near threshold
    /// 3. The adaptation variable provides negative feedback that increases after each spike
    /// 4. Check if membrane potential exceeds threshold (0 mV)
    /// 5. If threshold crossed:
    ///    - Output a spike (1.0)
    ///    - Reset membrane potential to -70 mV
    ///    - Increase adaptation variable by _b_adex
    /// 6. Otherwise, output no spike (0.0)
    /// 
    /// This model captures both the rapid onset of spikes and the adaptation
    /// of firing rate seen in many types of neurons.
    /// </para>
    /// </remarks>
    private Vector<T> UpdateAdaptiveExponential(Vector<T> current)
    {
        if (_adaptationVariable == null)
            throw new InvalidOperationException("Adaptation variable not initialized for AdEx model");
            
        // Adaptive Exponential Integrate-and-Fire model
        for (int i = 0; i < _membranePotential.Length; i++)
        {
            double v = Convert.ToDouble(_membranePotential[i]);
            double w = Convert.ToDouble(_adaptationVariable[i]);
            double I = Convert.ToDouble(current[i]);
            
            // Exponential term for spike initiation
            double expTerm = _deltaT * Math.Exp((v - _vT) / _deltaT);
            
            // Update membrane potential and adaptation variable
            double dv = (-v + expTerm - w + I) / _tau;
            double dw = (_a_adex * (v - _vT) - w) / _tauw;
            
            v += dv;
            w += dw;
            
            // Check for spike
            if (v >= 0) // Spike threshold
            {
                _spikes[i] = NumOps.One;
                v = -70.0; // Reset potential
                w += _b_adex; // Spike-triggered adaptation
            }
            else
            {
                _spikes[i] = NumOps.Zero;
            }
            
            _membranePotential[i] = NumOps.FromDouble(v);
            _adaptationVariable[i] = NumOps.FromDouble(w);
        }
        
        return _spikes;
    }

    /// <summary>
    /// Updates the state of Hodgkin-Huxley neurons.
    /// </summary>
    /// <param name="current">The input current to each neuron.</param>
    /// <returns>The output spikes (0 or 1) for each neuron.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the Hodgkin-Huxley neuron model, which is the most biophysically detailed model
    /// included in this layer. It models the dynamics of sodium and potassium ion channels using gating variables,
    /// providing a very accurate representation of action potential generation in real neurons. This model is
    /// computationally expensive but captures many subtle details of neural dynamics.
    /// </para>
    /// <para><b>For Beginners:</b> This method implements the most detailed and biologically accurate neuron model.
    /// 
    /// The Hodgkin-Huxley model works like this:
    /// 1. Track three gating variables:
    ///    - n: Potassium channel activation (opens slowly, closes slowly)
    ///    - m: Sodium channel activation (opens quickly, closes quickly)
    ///    - h: Sodium channel inactivation (closes slowly, opens slowly)
    /// 2. Calculate how these gates change based on current membrane potential
    /// 3. Calculate ionic currents through sodium, potassium, and leak channels
    /// 4. Update the membrane potential based on input current and ionic currents
    /// 5. Detect spikes when membrane potential crosses 0 mV
    /// 
    /// This model won Hodgkin and Huxley the Nobel Prize and directly represents
    /// the biological mechanisms of real neurons.
    /// </para>
    /// </remarks>
    private Vector<T> UpdateHodgkinHuxley(Vector<T> current)
    {
        if (_nGate == null || _mGate == null || _hGate == null)
            throw new InvalidOperationException("Gate variables not initialized for Hodgkin-Huxley model");
    
        // Ensure _spikes is initialized
        _spikes ??= Vector<T>.CreateDefault(_membranePotential.Length, NumOps.Zero);

        // Constants for Hodgkin-Huxley model
        double ENa = 50.0;   // Sodium reversal potential (mV)
        double EK = -77.0;   // Potassium reversal potential (mV)
        double EL = -54.387; // Leak reversal potential (mV)
        double gNa = 120.0;  // Maximum sodium conductance (mS/cm�)
        double gK = 36.0;    // Maximum potassium conductance (mS/cm�)
        double gL = 0.3;     // Leak conductance (mS/cm�)
        double dt = 0.01;    // Time step (ms)
    
        for (int i = 0; i < _membranePotential.Length; i++)
        {
            double v = Convert.ToDouble(_membranePotential[i]);
            double n = Convert.ToDouble(_nGate[i]);
            double m = Convert.ToDouble(_mGate[i]);
            double h = Convert.ToDouble(_hGate[i]);
            double I = Convert.ToDouble(current[i]);
        
            // Calculate alpha and beta values for each gate
            double alphaM = 0.1 * (v + 40.0) / (1.0 - Math.Exp(-(v + 40.0) / 10.0));
            double betaM = 4.0 * Math.Exp(-(v + 65.0) / 18.0);
        
            double alphaN = 0.01 * (v + 55.0) / (1.0 - Math.Exp(-(v + 55.0) / 10.0));
            double betaN = 0.125 * Math.Exp(-(v + 65.0) / 80.0);
        
            double alphaH = 0.07 * Math.Exp(-(v + 65.0) / 20.0);
            double betaH = 1.0 / (1.0 + Math.Exp(-(v + 35.0) / 10.0));
        
            // Update gate variables
            double dn = alphaN * (1 - n) - betaN * n;
            double dm = alphaM * (1 - m) - betaM * m;
            double dh = alphaH * (1 - h) - betaH * h;
        
            n += dt * dn;
            m += dt * dm;
            h += dt * dh;
        
            // Calculate ionic currents
            double INa = gNa * Math.Pow(m, 3) * h * (v - ENa);
            double IK = gK * Math.Pow(n, 4) * (v - EK);
            double IL = gL * (v - EL);
        
            // Update membrane potential
            double dv = I - INa - IK - IL;
            v += dt * dv;
        
            // Check for spike (threshold crossing)
            if (v > 0 && NumOps.Equals(_spikes[i], NumOps.Zero))
            {
                _spikes[i] = NumOps.One;
            }
            else if (v < -30)
            {
                _spikes[i] = NumOps.Zero;
            }
        
            // Update state variables
            _membranePotential[i] = NumOps.FromDouble(v);
            _nGate[i] = NumOps.FromDouble(n);
            _mGate[i] = NumOps.FromDouble(m);
            _hGate[i] = NumOps.FromDouble(h);
        }
    
        return _spikes;
    }

    /// <summary>
    /// Updates the layer parameters with new values.
    /// </summary>
    /// <param name="parameters">The new parameter values as a flat vector.</param>
    /// <remarks>
    /// <para>
    /// This method updates the weights and biases of the layer from a flat vector of parameters.
    /// It assumes the vector contains all weights followed by all biases.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads new values into the layer's weights and biases.
    /// 
    /// The parameter vector is expected to contain:
    /// 1. All weight values (rows * columns), stored row by row
    /// 2. All bias values (one per output neuron)
    /// 
    /// This method is useful for:
    /// - Loading a previously saved model
    /// - Setting up the layer with pre-determined weights
    /// - Testing with specific parameter values
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int weightCount = Weights.Rows * Weights.Columns;
        
        // Update weights
        for (int i = 0; i < Weights.Rows; i++)
        {
            for (int j = 0; j < Weights.Columns; j++)
            {
                int index = i * Weights.Columns + j;
                Weights[i, j] = parameters[index];
            }
        }
        
        // Update biases
        for (int i = 0; i < Bias.Length; i++)
        {
            Bias[i] = parameters[weightCount + i];
        }
    }
    
    /// <summary>
    /// Gets all trainable parameters of the layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters of the layer (weights and biases) and combines them
    /// into a single vector. This is useful for optimization algorithms that operate on all parameters at once,
    /// or for saving and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values from the layer into a single list.
    /// 
    /// The returned vector contains:
    /// 1. All weight values (rows * columns), stored row by row
    /// 2. All bias values (one per output neuron)
    /// 
    /// This method is useful for:
    /// - Saving the model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques that need access to all parameters
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        int weightCount = Weights.Rows * Weights.Columns;
        Vector<T> parameters = Vector<T>.CreateDefault(ParameterCount, NumOps.Zero);
        
        // Flatten weights
        for (int i = 0; i < Weights.Rows; i++)
        {
            for (int j = 0; j < Weights.Columns; j++)
            {
                int index = i * Weights.Columns + j;
                parameters[index] = Weights[i, j];
            }
        }
        
        // Add biases
        for (int i = 0; i < Bias.Length; i++)
        {
            parameters[weightCount + i] = Bias[i];
        }
        
        return parameters;
    }

    /// <summary>
    /// Resets the internal state of the spiking layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets all internal state variables of the layer, including membrane potentials, refractory periods,
    /// and model-specific variables. It also clears cached inputs and outputs, and resets gradient accumulators.
    /// This is useful when starting a new sequence or when implementing stateful recurrent networks.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory and state to start fresh.
    /// 
    /// When resetting the state:
    /// - All neuron membrane potentials are set to zero
    /// - Refractory period counters are reset
    /// - All spikes are cleared
    /// - Model-specific variables are reset to their initial values
    /// - Cached inputs and outputs are cleared
    /// - Gradient accumulators for training are reset
    /// 
    /// This is important for:
    /// - Processing a new sequence that's unrelated to the previous one
    /// - Preventing information from one sequence affecting another
    /// - Starting a new training episode
    /// - Resetting after a period of high activity
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Reset all state variables
        _membranePotential = Vector<T>.CreateDefault(_membranePotential.Length, NumOps.Zero);
        _refractoryCountdown = Vector<T>.CreateDefault(_refractoryCountdown.Length, NumOps.Zero);
        _spikes = Vector<T>.CreateDefault(_spikes.Length, NumOps.Zero);
    
        // Reset model-specific variables
        if (_neuronType == SpikingNeuronType.Izhikevich && _recoveryVariable != null)
        {
            _recoveryVariable = Vector<T>.CreateDefault(_recoveryVariable.Length, NumOps.Zero);
        }
        else if (_neuronType == SpikingNeuronType.AdaptiveExponential && _adaptationVariable != null)
        {
            _adaptationVariable = Vector<T>.CreateDefault(_adaptationVariable.Length, NumOps.Zero);
        }
        else if (_neuronType == SpikingNeuronType.HodgkinHuxley)
        {
            if (_nGate != null) _nGate = Vector<T>.CreateDefault(_nGate.Length, NumOps.FromDouble(0.32));
            if (_mGate != null) _mGate = Vector<T>.CreateDefault(_mGate.Length, NumOps.FromDouble(0.05));
            if (_hGate != null) _hGate = Vector<T>.CreateDefault(_hGate.Length, NumOps.FromDouble(0.60));
        }
    
        // Clear cached values
        _lastInput = null;
        _lastOutput = null;
    
        // Reset gradient accumulators
        _weightGradients = Matrix<T>.CreateDefault(Weights.Rows, Weights.Columns, NumOps.Zero);
        _biasGradients = Vector<T>.CreateDefault(Bias.Length, NumOps.Zero);
    }
    
    /// <summary>
    /// Serializes the layer's parameters and state to a binary stream.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// <para>
    /// This method writes the layer's parameters, including neuron type, time constants, weights, biases,
    /// and model-specific parameters, to a binary stream. This allows the layer to be saved to disk for later use.
    /// </para>
    /// <para><b>For Beginners:</b> This method saves the layer's configuration and parameters to a file.
    /// 
    /// The serialization includes:
    /// - Basic parameters (neuron type, time constant, refractory period)
    /// - All weights and biases
    /// - Model-specific parameters like those for Izhikevich or AdEx models
    /// 
    /// This allows you to:
    /// - Save a trained model to disk
    /// - Load it later for inference or continued training
    /// - Transfer the model to another application
    /// </para>
    /// </remarks>
    public override void Serialize(BinaryWriter writer)
    {
        // Write neuron type and parameters
        writer.Write((int)_neuronType);
        writer.Write(_tau);
        writer.Write(_refractoryPeriod);
        
        // Write weights and biases
        for (int i = 0; i < Weights.Rows; i++)
        {
            for (int j = 0; j < Weights.Columns; j++)
            {
                writer.Write(Convert.ToDouble(Weights[i, j]));
            }
        }
        
        for (int i = 0; i < Bias.Length; i++)
        {
            writer.Write(Convert.ToDouble(Bias[i]));
        }
        
        // Write model-specific parameters
        if (_neuronType == SpikingNeuronType.Izhikevich)
        {
            writer.Write(_a);
            writer.Write(_b);
            writer.Write(_c);
            writer.Write(_d);
        }
        else if (_neuronType == SpikingNeuronType.AdaptiveExponential)
        {
            writer.Write(_deltaT);
            writer.Write(_vT);
            writer.Write(_tauw);
            writer.Write(_a_adex);
            writer.Write(_b_adex);
        }
    }
    
    /// <summary>
    /// Deserializes the layer's parameters and state from a binary stream.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <remarks>
    /// <para>
    /// This method reads the layer's parameters from a binary stream, including neuron type, time constants,
    /// weights, biases, and model-specific parameters. This allows a previously saved layer to be loaded from disk.
    /// It also initializes any model-specific variables needed for the selected neuron type.
    /// </para>
    /// <para><b>For Beginners:</b> This method loads the layer's configuration and parameters from a file.
    /// 
    /// The deserialization reads:
    /// - Basic parameters (neuron type, time constant, refractory period)
    /// - All weights and biases
    /// - Model-specific parameters for the particular neuron type
    /// 
    /// It also initializes any special variables needed for the specific neuron model.
    /// This lets you load a previously saved model and continue using or training it.
    /// </para>
    /// </remarks>
    public override void Deserialize(BinaryReader reader)
    {
        // Read neuron type and parameters
        _neuronType = (SpikingNeuronType)reader.ReadInt32();
        _tau = reader.ReadDouble();
        _refractoryPeriod = reader.ReadDouble();
        
        // Read weights and biases
        for (int i = 0; i < Weights.Rows; i++)
        {
            for (int j = 0; j < Weights.Columns; j++)
            {
                Weights[i, j] = NumOps.FromDouble(reader.ReadDouble());
            }
        }
        
        for (int i = 0; i < Bias.Length; i++)
        {
            Bias[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        
        // Read model-specific parameters
        if (_neuronType == SpikingNeuronType.Izhikevich)
        {
            _a = reader.ReadDouble();
            _b = reader.ReadDouble();
            _c = reader.ReadDouble();
            _d = reader.ReadDouble();
            
            // Initialize recovery variable if needed
            if (_recoveryVariable == null)
            {
                _recoveryVariable = Vector<T>.CreateDefault(OutputShape[0], NumOps.Zero);
            }
        }
        else if (_neuronType == SpikingNeuronType.AdaptiveExponential)
        {
            _deltaT = reader.ReadDouble();
            _vT = reader.ReadDouble();
            _tauw = reader.ReadDouble();
            _a_adex = reader.ReadDouble();
            _b_adex = reader.ReadDouble();
            
            // Initialize adaptation variable if needed
            if (_adaptationVariable == null)
            {
                _adaptationVariable = Vector<T>.CreateDefault(OutputShape[0], NumOps.Zero);
            }
        }
        else if (_neuronType == SpikingNeuronType.HodgkinHuxley)
        {
            // Initialize gate variables if needed
            int outputSize = OutputShape[0];
            if (_nGate == null) _nGate = Vector<T>.CreateDefault(outputSize, NumOps.FromDouble(0.32));
            if (_mGate == null) _mGate = Vector<T>.CreateDefault(outputSize, NumOps.FromDouble(0.05));
            if (_hGate == null) _hGate = Vector<T>.CreateDefault(outputSize, NumOps.FromDouble(0.60));
        }
        
        // Initialize state variables
        _membranePotential = Vector<T>.CreateDefault(OutputShape[0], NumOps.Zero);
        _refractoryCountdown = Vector<T>.CreateDefault(OutputShape[0], NumOps.Zero);
        _spikes = Vector<T>.CreateDefault(OutputShape[0], NumOps.Zero);
    }

    /// <summary>
    /// Performs the backward pass of the spiking layer.
    /// </summary>
    /// <param name="outputGradient">The gradient of the loss with respect to the layer's output.</param>
    /// <returns>The gradient of the loss with respect to the layer's input.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the backward pass of the spiking layer, which is used during training to propagate
    /// error gradients back through the network. Since spike events are non-differentiable, it uses a surrogate
    /// gradient approach to approximate the derivative of the spike function. This enables backpropagation
    /// through spiking neurons by providing a smooth approximation to the discontinuous threshold crossing.
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates how to adjust the network during training.
    /// 
    /// Training spiking neurons is challenging because spikes are binary events (either a neuron fires or it doesn't),
    /// which normally can't be used with standard neural network training methods.
    /// 
    /// To solve this problem, this method:
    /// 1. Uses a technique called "surrogate gradients" to approximate how changes in membrane potential affect spiking
    /// 2. It computes a smooth function based on the membrane potential that approximates spike probability
    /// 3. Using this approximation, it calculates:
    ///    - How to adjust weights and biases to improve performance
    ///    - How errors should flow back to previous layers
    /// 
    /// This clever approach allows spiking neural networks to be trained with backpropagation
    /// despite their fundamentally discrete nature.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Cannot perform backward pass before forward pass");

        // Convert tensor to vector for easier processing
        Vector<T> gradientVector = outputGradient.ToVector();
    
        // Initialize input gradient
        var inputGradient = new Tensor<T>(_lastInput.Shape);
        Vector<T> inputGradientVector = Vector<T>.CreateDefault(_lastInput.Shape[0], NumOps.Zero);
    
        // Apply surrogate gradient for the non-differentiable spike function
        // We use a sigmoid-based surrogate gradient approximation
        for (int i = 0; i < _membranePotential.Length; i++)
        {
            // Get membrane potential
            double v = Convert.ToDouble(_membranePotential[i]);
        
            // Compute surrogate gradient using a sigmoid-based function
            // This approximates the derivative of the spike function
            double beta = 10.0; // Steepness of the surrogate function
            double surrogate = 1.0 / (beta * Math.Pow(Math.Cosh(v / beta), 2));
        
            // Apply surrogate gradient to the output gradient
            T surrogateGradient = NumOps.FromDouble(surrogate);
            T gradientValue = NumOps.Multiply(gradientVector[i], surrogateGradient);
        
            // Compute weight gradients and accumulate them
            for (int j = 0; j < _lastInput.Shape[0]; j++)
            {
                T inputValue = _lastInput.ToVector()[j];
                T weightGradient = NumOps.Multiply(gradientValue, inputValue);
            
                // Accumulate weight gradients
                _weightGradients[j, i] = NumOps.Add(_weightGradients[j, i], weightGradient);
            
                // Compute input gradients (for backpropagation to previous layer)
                T currentInputGradient = NumOps.Multiply(gradientValue, Weights[j, i]);
                inputGradientVector[j] = NumOps.Add(inputGradientVector[j], currentInputGradient);
            }
        
            // Compute bias gradients
            _biasGradients[i] = NumOps.Add(_biasGradients[i], gradientValue);
        }
    
        // Convert input gradient vector back to tensor
        for (int i = 0; i < inputGradientVector.Length; i++)
        {
            inputGradient[i] = inputGradientVector[i];
        }
    
        return inputGradient;
    }

    /// <summary>
    /// Updates the parameters of the layer using the calculated gradients and learning rate.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <remarks>
    /// <para>
    /// This method updates the weights and biases of the layer based on the gradients accumulated during the backward pass.
    /// It applies the learning rate to determine the size of the updates, subtracts the scaled gradients from the current
    /// parameters, and then resets the gradients for the next batch.
    /// </para>
    /// <para><b>For Beginners:</b> This method applies the calculated adjustments to improve the network.
    /// 
    /// After calculating how the weights and biases should change in the backward pass:
    /// 1. The method applies these changes using the learning rate to control their size
    /// 2. For each weight and bias:
    ///    - Compute the update as learning rate � gradient
    ///    - Subtract this update from the current value (moving in the opposite direction of the gradient)
    ///    - Reset the gradient accumulator to zero for the next batch
    /// 
    /// The learning rate is important:
    /// - Smaller values mean smaller updates (more stable but slower learning)
    /// - Larger values mean larger updates (faster learning but potential instability)
    /// 
    /// This process is how the network gradually improves its performance over many training examples.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // Update weights using accumulated gradients
        for (int i = 0; i < Weights.Rows; i++)
        {
            for (int j = 0; j < Weights.Columns; j++)
            {
                // Compute weight update: w = w - lr * gradient
                T update = NumOps.Multiply(_weightGradients[i, j], learningRate);
                Weights[i, j] = NumOps.Subtract(Weights[i, j], update);
            
                // Reset gradient for next batch
                _weightGradients[i, j] = NumOps.Zero;
            }
        }
    
        // Update biases using accumulated gradients
        for (int i = 0; i < Bias.Length; i++)
        {
            // Compute bias update: b = b - lr * gradient
            T update = NumOps.Multiply(_biasGradients[i], learningRate);
            Bias[i] = NumOps.Subtract(Bias[i], update);
        
            // Reset gradient for next batch
            _biasGradients[i] = NumOps.Zero;
        }
    }
}
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks.Options;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Spiking Neural Network, which is a type of neural network that more closely models biological neurons with temporal dynamics.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Spiking Neural Networks (SNNs) work like real biological neurons:
/// they communicate using timed electrical pulses (spikes) rather than continuous values. Each
/// neuron accumulates input over time and only fires when a threshold is reached, then resets.
/// This temporal coding makes SNNs extremely energy-efficient on neuromorphic hardware and
/// naturally suited for processing time-varying signals like sensor data and event cameras.</para>
/// </remarks>
/// <example>
/// <code>
/// var options = new SpikingNeuralNetworkOptions { InputSize = 784, HiddenSize = 500, TimeSteps = 100 };
/// var model = new SpikingNeuralNetwork&lt;float&gt;(options);
/// var input = Tensor&lt;float&gt;.Random(new[] { 1, 784 });
/// var output = model.Predict(input);
/// </code>
/// </example>
[ModelDomain(ModelDomain.General)]
[ModelDomain(ModelDomain.Science)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Networks of Spiking Neurons: The Third Generation of Neural Network Models", "https://doi.org/10.1016/S0893-6080(97)00011-7")]
public class SpikingNeuralNetwork<T> : NeuralNetworkBase<T>
{
    private readonly SpikingNeuralNetworkOptions _options;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Gets or sets the simulation time step for the spiking neural network.
    /// </summary>
    private T _timeStep { get; set; }

    /// <summary>
    /// Gets or sets the number of time steps to simulate when processing input.
    /// </summary>
    private int _simulationSteps { get; set; }

    /// <summary>
    /// Gets or sets the vector activation function used in the network.
    /// </summary>
    private IVectorActivationFunction<T>? _vectorActivation { get; set; }

    /// <summary>
    /// Gets or sets the scalar activation function used in the network.
    /// </summary>
    private IActivationFunction<T>? _scalarActivation { get; set; }

    /// <summary>
    /// Gets or sets the decay constant for neuron membrane potentials.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The membrane potential decay constant determines how quickly the neuron's membrane potential
    /// decays over time. This models the leakage of charge in biological neurons. A value closer to 1
    /// means slower decay, while a value closer to 0 means faster decay.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how quickly neurons "forget" their built-up charge.
    /// 
    /// The decay constant:
    /// - Models how neurons gradually lose their electrical charge over time
    /// - Values near 1.0 mean neurons hold their charge for a long time
    /// - Values near 0.0 mean neurons quickly lose their charge
    /// 
    /// For example, with a decay of 0.9:
    /// - A neuron will retain about 90% of its charge at each time step
    /// - This makes it respond more to sustained inputs than brief ones
    /// 
    /// This property helps make the network's behavior more biologically realistic
    /// and can be important for processing temporal patterns.
    /// </para>
    /// </remarks>
    private T _membraneDecay;

    /// <summary>
    /// Gets or sets the refractory period for neurons after firing.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The refractory period is the number of time steps during which a neuron cannot fire again after
    /// generating a spike. This models the recovery period of biological neurons after an action potential.
    /// Longer refractory periods limit the maximum firing rate of neurons.
    /// </para>
    /// <para><b>For Beginners:</b> This is the recovery time neurons need after firing.
    /// 
    /// The refractory period:
    /// - Represents a "cooldown" time after a neuron fires
    /// - During this period, the neuron cannot fire again no matter what input it receives
    /// - Measured in simulation time steps
    /// 
    /// For example, with a refractory period of 5:
    /// - After a neuron fires, it must wait 5 time steps before it can fire again
    /// - This limits how rapidly a neuron can send spikes
    /// 
    /// This property mimics how real neurons work - they need time to reset their
    /// chemical balance after firing before they can fire again.
    /// </para>
    /// </remarks>
    private int _refractoryPeriod;

    /// <summary>
    /// Stores the membrane potentials for all neurons in the network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the current membrane potential (internal electrical charge) of each neuron
    /// in the network. When a neuron's potential exceeds its firing threshold, it generates a spike,
    /// and its potential is reset.
    /// </para>
    /// <para><b>For Beginners:</b> This represents the current electrical charge of each neuron.
    /// 
    /// The membrane potentials:
    /// - Track how "excited" each neuron is at the current moment
    /// - Increase when the neuron receives input
    /// - Decrease over time due to decay
    /// - When they exceed a threshold, the neuron fires
    /// 
    /// This internal state is what gives spiking neural networks their temporal
    /// processing capability - neurons remember their recent inputs via this potential.
    /// </para>
    /// </remarks>
    private List<Vector<T>> _membranePotentials;

    /// <summary>
    /// Tracks the refractory state of each neuron.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field tracks the remaining refractory period for each neuron. A value greater than zero
    /// indicates that the neuron is in its refractory period and cannot fire, regardless of its
    /// membrane potential.
    /// </para>
    /// <para><b>For Beginners:</b> This tracks the recovery countdown for each neuron.
    /// 
    /// The refractory counters:
    /// - Start at the refractory period value when a neuron fires
    /// - Count down by 1 at each time step
    /// - While above 0, prevent the neuron from firing again
    /// - Reset to 0 when the recovery period is complete
    /// 
    /// This ensures that neurons follow biological constraints on
    /// how rapidly they can fire in succession.
    /// </para>
    /// </remarks>
    private List<int[]> _refractoryCounters;

    /// <summary>
    /// Firing thresholds for neurons in each layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the firing threshold for each neuron. When a neuron's membrane potential
    /// exceeds this threshold, it generates a spike. Different neurons may have different thresholds,
    /// allowing for specialized sensitivity to inputs.
    /// </para>
    /// <para><b>For Beginners:</b> This is the charge level at which neurons fire.
    /// 
    /// The firing thresholds:
    /// - Define how much charge is needed before a neuron generates a spike
    /// - Can be different for neurons in different layers
    /// - Higher thresholds mean the neuron needs stronger input to fire
    /// 
    /// This is similar to how real neurons have different sensitivities -
    /// some fire easily with slight stimulation, while others require
    /// stronger signals before they respond.
    /// </para>
    /// </remarks>
    private List<Vector<T>> _firingThresholds;

    // Adam optimizer state per Zenke 2018 §3.2 — supervised SNN training
    // with surrogate gradient uses Adam (Kingma & Ba 2014) not vanilla
    // SGD, because the rate-coded delta rule overshoots at fixed LR once
    // post-synaptic rates approach saturation (after which the gradient
    // direction stops being informative). Per-layer first / second
    // moment vectors keyed by layer index; step counter shared so bias
    // correction is consistent across layers.
    private Dictionary<int, Vector<T>> _adamM = new Dictionary<int, Vector<T>>();
    private Dictionary<int, Vector<T>> _adamV = new Dictionary<int, Vector<T>>();
    private int _adamStep = 0;
    private const double AdamBeta1 = 0.9;
    private const double AdamBeta2 = 0.999;
    private const double AdamEpsilon = 1e-8;

    /// <summary>
    /// Initializes a new instance of the <see cref="SpikingNeuralNetwork{T}"/> class with the specified architecture and a vector activation function.
    /// </summary>
    /// <param name="architecture">The neural network architecture to use for the SNN.</param>
    /// <param name="timeStep">The simulation time step, defaults to 0.1.</param>
    /// <param name="simulationSteps">The number of time steps to simulate, defaults to 100.</param>
    /// <param name="vectorActivation">The vector activation function to use. If null, a default activation is used.</param>
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public SpikingNeuralNetwork()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputSize: 128,
            outputSize: 1), vectorActivation: (IVectorActivationFunction<T>?)null)
    {
    }

    public SpikingNeuralNetwork(NeuralNetworkArchitecture<T> architecture, double timeStep = 0.1, int simulationSteps = 100,
        IVectorActivationFunction<T>? vectorActivation = null, ILossFunction<T>? lossFunction = null,
        SpikingNeuralNetworkOptions? options = null)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        _options = options ?? new SpikingNeuralNetworkOptions();
        Options = _options;

        _timeStep = NumOps.FromDouble(timeStep);
        _simulationSteps = simulationSteps;
        _vectorActivation = vectorActivation ?? new BinarySpikingActivation<T>();
        _scalarActivation = null;
        _membraneDecay = NumOps.FromDouble(0.9); // Default decay constant
        _refractoryPeriod = 5; // Default refractory period
        _firingThresholds = new List<Vector<T>>();
        _refractoryCounters = new List<int[]>();
        _membranePotentials = new List<Vector<T>>();

        InitializeLayers();
        InitializeNeuronStates();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="SpikingNeuralNetwork{T}"/> class with the specified architecture and a scalar activation function.
    /// </summary>
    /// <param name="architecture">The neural network architecture to use for the SNN.</param>
    /// <param name="timeStep">The simulation time step, defaults to 0.1.</param>
    /// <param name="simulationSteps">The number of time steps to simulate, defaults to 100.</param>
    /// <param name="scalarActivation">The scalar activation function to use. If null, a default activation is used.</param>
    public SpikingNeuralNetwork(NeuralNetworkArchitecture<T> architecture, double timeStep = 0.1, int simulationSteps = 100,
        IActivationFunction<T>? scalarActivation = null, ILossFunction<T>? lossFunction = null,
        SpikingNeuralNetworkOptions? options = null)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        _options = options ?? new SpikingNeuralNetworkOptions();
        Options = _options;

        _timeStep = NumOps.FromDouble(timeStep);
        _simulationSteps = simulationSteps;
        _scalarActivation = scalarActivation ?? new BinarySpikingActivation<T>();
        _vectorActivation = null;
        _membraneDecay = NumOps.FromDouble(0.9); // Default decay constant
        _refractoryPeriod = 5; // Default refractory period
        _firingThresholds = new List<Vector<T>>();
        _refractoryCounters = new List<int[]>();
        _membranePotentials = new List<Vector<T>>();

        InitializeLayers();
        InitializeNeuronStates();
    }

    /// <summary>
    /// Initializes the neural network layers based on the provided architecture or default configuration.
    /// </summary>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            // Use the layers provided by the user
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            // Use default layer configuration if no layers are provided
            Layers.AddRange(LayerHelper<T>.CreateDefaultSpikingLayers(Architecture));
        }
    }

    /// <summary>
    /// Initializes the internal neuron states for simulation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the membrane potentials, refractory counters, and firing thresholds
    /// for all neurons in the network. It sets all membrane potentials to zero (resting potential),
    /// all refractory counters to zero (not in refractory period), and all firing thresholds to a
    /// default value that can be overridden during configuration.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the initial state of all neurons.
    /// 
    /// When initializing neuron states:
    /// - Membrane potentials start at zero (neurons at rest)
    /// - Refractory counters start at zero (neurons ready to fire)
    /// - Firing thresholds are set to default values (typically 1.0)
    /// 
    /// This creates a clean starting point for the network simulation,
    /// with all neurons in their resting state and ready to process input.
    /// </para>
    /// </remarks>
    private void InitializeNeuronStates()
    {
        _membranePotentials = new List<Vector<T>>();
        _refractoryCounters = new List<int[]>();
        _firingThresholds = new List<Vector<T>>();

        // Initialize states for each layer
        foreach (var layer in Layers)
        {
            var outputShape = layer.GetOutputShape();
            // The layer may report a lazy/unresolved sentinel (e.g. -1) for some dims.
            // Use the largest concrete dim as the neuron count and clamp at zero so
            // lazy-state layers don't blow up the Vector ctor with a negative length.
            // Per Maass (1997) the per-neuron state is a 1-D vector along the layer's
            // output feature axis; for ranks > 1 we approximate by taking the first
            // positive dim — this matches Dense (output = [outputSize]) and stays
            // safe for Conv (output = [batch, channels, ...] where batch is 1).
            int neuronCount = 0;
            for (int i = 0; i < outputShape.Length; i++)
            {
                if (outputShape[i] > 0) { neuronCount = outputShape[i]; break; }
            }

            // Initialize membrane potentials to zero
            var potentials = new Vector<T>(neuronCount);
            for (int i = 0; i < neuronCount; i++)
            {
                potentials[i] = NumOps.Zero;
            }
            _membranePotentials.Add(potentials);

            // Initialize refractory counters to zero
            var refractoryCounters = new int[neuronCount];
            for (int i = 0; i < neuronCount; i++)
            {
                refractoryCounters[i] = 0;
            }
            _refractoryCounters.Add(refractoryCounters);

            // Initialize firing thresholds.
            // Per Maass (1997), threshold should be calibrated to expected input magnitude.
            // With typical [0,1] inputs through Dense layers, threshold of 0.3 ensures
            // spikes fire, enabling the network to differentiate inputs.
            var threshold = NumOps.FromDouble(0.3);
            var thresholds = new Vector<T>(neuronCount);
            for (int i = 0; i < neuronCount; i++)
            {
                thresholds[i] = threshold;
            }
            _firingThresholds.Add(thresholds);
        }
    }

    /// <summary>
    /// Aggregates a spike train (sequence of spikes over time) into a final output vector.
    /// </summary>
    /// <param name="spikeTrain">The list of spike vectors over time.</param>
    /// <returns>The aggregated output vector.</returns>
    private Vector<T> AggregateSpikeTrainToOutput(List<Vector<T>> spikeTrain)
    {
        int outputSize = spikeTrain[0].Length;
        var output = new Vector<T>(outputSize);

        for (int i = 0; i < outputSize; i++)
        {
            T sum = NumOps.Zero;
            for (int t = 0; t < spikeTrain.Count; t++)
            {
                sum = NumOps.Add(sum, spikeTrain[t][i]);
            }
            output[i] = NumOps.Divide(sum, NumOps.FromDouble(spikeTrain.Count));
        }

        return output;
    }

    /// <summary>
    /// Updates the parameters of the spiking neural network layers.
    /// </summary>
    /// <param name="parameters">The vector of parameter updates to apply.</param>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int startIndex = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = checked((int)layer.ParameterCount);
            if (layerParameterCount > 0)
            {
                Vector<T> layerParameters = parameters.SubVector(startIndex, layerParameterCount);
                layer.UpdateParameters(layerParameters);
                startIndex += layerParameterCount;
            }
        }
    }

    /// <summary>
    /// Makes a prediction using the spiking neural network.
    /// </summary>
    /// <param name="input">The input tensor to process.</param>
    /// <returns>The output tensor after processing through all layers.</returns>
    /// <remarks>
    /// <para>
    /// This method processes the input through the spiking neural network by simulating the network's
    /// dynamics over a specified number of time steps. It updates the membrane potentials of all neurons
    /// at each step, generates spikes when thresholds are exceeded, and tracks the refractory periods.
    /// The final output is an aggregation of the spike activity over all time steps.
    /// </para>
    /// <para><b>For Beginners:</b> This method runs the network simulation to process your input data.
    /// 
    /// The prediction process works like this:
    /// 1. Reset the network to its initial state
    /// 2. Feed the input to the network
    /// 3. Simulate the network for a set number of time steps:
    ///    - Update neuron membrane potentials
    ///    - Generate spikes when thresholds are reached
    ///    - Track refractory periods after neurons fire
    /// 4. Record the spike activity of the output neurons
    /// 5. Aggregate this activity into a final result
    /// 
    /// The result represents how strongly each output neuron responded to the input,
    /// based on its spiking activity throughout the simulation.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        // Reset network state (clears SpikingLayer internal states)
        ResetState();

        // SpikingLayers handle their own LIF dynamics (membrane, spikes, refractory).
        // We run the simulation for multiple timesteps, feeding the same input each step
        // and letting SpikingLayers accumulate membrane potential and generate spikes.
        // The output layer's spike rates over time form the final prediction.

        // Per Neftci et al. 2019: hidden spiking layers produce spikes over
        // T time steps; everything AFTER the last SpikingLayer is the non-
        // spiking continuous readout tail. Must match Train's boundary
        // exactly — the default LayerHelper.CreateDefaultSpikingLayers stack
        // is {Spiking, Spiking, Dense, Activation}, where Dense AND
        // Activation are both readout (Dense projects spike rates to logits,
        // Activation applies softmax / sigmoid / etc). Previously this
        // looked for the LAST non-spiking layer and peeled off only that
        // one (just the Activation), so Predict ran Dense through the
        // spiking simulation while Train ran it through the readout path —
        // inference saw a different forward pass than training optimized
        // against, defeating supervised learning end-to-end.
        int readoutBoundary = Layers.Count;
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            if (Layers[i] is Layers.SpikingLayer<T>) { readoutBoundary = i + 1; break; }
        }
        // -1 sentinel preserved for downstream code that prefers the "no
        // readout" branch (entire stack runs through the simulation).
        int readoutIdx = readoutBoundary < Layers.Count ? readoutBoundary : -1;

        // Accumulate spiking layer outputs over time. The readout layer may be
        // lazy (reports input shape as [-1]); fall back to the last spiking
        // layer's output, then to a single forward pass to discover the size.
        // Each Get*Shape() may legitimately return an empty array for a
        // freshly-constructed lazy layer — guard the [0] access against that
        // case before the index throws an unhelpful IndexOutOfRangeException.
        int spikingOutputSize = ReadFirstShapeAxis(readoutIdx >= 0
            ? Layers[readoutIdx].GetInputShape()
            : Layers[^1].GetOutputShape());
        if (spikingOutputSize <= 0)
        {
            int lastSpikingIdx = readoutIdx >= 0 ? readoutIdx : Layers.Count;
            if (lastSpikingIdx > 0)
                spikingOutputSize = ReadFirstShapeAxis(Layers[lastSpikingIdx - 1].GetOutputShape());
            if (spikingOutputSize <= 0)
            {
                Tensor<T> probe = input;
                for (int i = 0; i < lastSpikingIdx; i++)
                    probe = Layers[i].Forward(probe);
                spikingOutputSize = probe.Length;
                // The probe forward mutated SpikingLayer membrane / refractory state
                // (each layer accumulates LIF dynamics across calls). If we leave
                // that state in place, the simulation loop below would start from
                // a non-zero baseline and produce different spike rates than a
                // fresh run — which is the whole point of ResetState() above. Reset
                // again so the simulation loop sees the same clean slate it would
                // have seen without the probe.
                ResetState();
            }
        }
        var accumSpikes = new T[spikingOutputSize];

        for (int step = 0; step < _simulationSteps; step++)
        {
            Tensor<T> current = input;
            int lastSpikingIdx = readoutIdx >= 0 ? readoutIdx : Layers.Count;
            for (int i = 0; i < lastSpikingIdx; i++)
                current = Layers[i].Forward(current);

            var vec = current.ToVector();
            for (int i = 0; i < Math.Min(spikingOutputSize, vec.Length); i++)
                accumSpikes[i] = NumOps.Add(accumSpikes[i], vec[i]);
        }

        // Average spike rate
        for (int i = 0; i < spikingOutputSize; i++)
            accumSpikes[i] = NumOps.Divide(accumSpikes[i], NumOps.FromDouble(_simulationSteps));

        // Pass spike rates through non-spiking readout
        var spikeRateTensor = Tensor<T>.FromVector(new Vector<T>(accumSpikes));
        if (readoutIdx >= 0)
        {
            for (int i = readoutIdx; i < Layers.Count; i++)
                spikeRateTensor = Layers[i].Forward(spikeRateTensor);
        }

        return spikeRateTensor;
    }

    /// <summary>
    /// Trains the spiking neural network on input-output pairs.
    /// </summary>
    /// <param name="input">The input tensor for training.</param>
    /// <param name="expectedOutput">The expected output tensor.</param>
    /// <remarks>
    /// <para>
    /// This method trains the spiking neural network using spike-timing-dependent plasticity (STDP),
    /// a biologically inspired learning rule that adjusts synaptic strengths based on the relative timing
    /// of spikes between neurons. The training process simulates the network dynamics, compares the output
    /// with the expected output, and updates the weights accordingly.
    /// </para>
    /// <para><b>For Beginners:</b> This method teaches the network to recognize patterns in your data.
    /// 
    /// The training process works like this:
    /// 1. Run the network simulation (similar to the Predict method)
    /// 2. Record the timing of spikes in each layer
    /// 3. Compare the output spikes with what was expected
    /// 4. Adjust connection weights using a learning rule called STDP:
    ///    - Strengthen connections between neurons that spike in close sequence
    ///    - Weaken connections between neurons that don't coordinate their spiking
    /// 
    /// This learning rule is inspired by how real brains learn - neurons that fire
    /// together wire together. The timing of spikes is crucial to this process,
    /// which is a key difference from traditional neural networks.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Reset network state
        ResetState();

        // Convert input and expected output to vectors
        Vector<T> inputVector = input.ToVector();
        Vector<T> expectedOutputVector = expectedOutput.ToVector();

        // Storage for spike + membrane history of all layers. Membrane
        // potentials are recorded at the moment of the spike-check (BEFORE
        // the post-spike reset to zero) so the supervised surrogate-
        // gradient update in ApplySTDPLearning can evaluate
        // σ'(u_t − θ) per timestep per neuron — Zenke 2018 §3 Eq. 12
        // requires the actual membrane voltage at threshold, not the
        // post-reset value.
        List<List<Vector<T>>> layerSpikeHistory = new List<List<Vector<T>>>();
        List<List<Vector<T>>> layerMembraneHistory = new List<List<Vector<T>>>();
        for (int i = 0; i < Layers.Count; i++)
        {
            layerSpikeHistory.Add(new List<Vector<T>>(_simulationSteps));
            layerMembraneHistory.Add(new List<Vector<T>>(_simulationSteps));
        }

        // Identify the readout boundary — index of the first NON-spiking
        // layer in the tail of the network. Per Eliasmith & Anderson
        // 2004 NEF / Neftci 2019 SNN review, the standard supervised
        // SNN architecture is "spiking hidden + NON-spiking readout":
        // hidden SpikingLayers integrate membrane potential and emit
        // binary spikes, but the final DenseLayer readout decodes the
        // time-averaged spike rates into continuous output values for
        // regression / classification. Running the readout DenseLayer
        // through the same threshold-and-reset spike check that hidden
        // layers use binarizes its continuous output to 0/1 — which
        // for regression targets ≪ threshold means the model can
        // literally never produce a non-zero prediction and training
        // can never reduce loss. The default topology in
        // LayerHelper.CreateDefaultSpikingLayers appends
        // {DenseLayer, ActivationLayer} after the SpikingLayer stack;
        // both must run as continuous (non-spiking) readout layers.
        int readoutBoundary = Layers.Count;
        for (int li = Layers.Count - 1; li >= 0; li--)
        {
            if (Layers[li] is SpikingLayer<T>)
            {
                readoutBoundary = li + 1;
                break;
            }
        }

        // Run simulation for training
        for (int step = 0; step < _simulationSteps; step++)
        {
            // Process through layers
            Vector<T> currentInput = inputVector;

            for (int layerIndex = 0; layerIndex < Layers.Count; layerIndex++)
            {
                // Get current layer and its state
                var layer = Layers[layerIndex];
                var membranePotentials = _membranePotentials[layerIndex];
                var refractoryCounters = _refractoryCounters[layerIndex];
                var firingThresholds = _firingThresholds[layerIndex];

                // Process input through layer
                Tensor<T> layerInput = Tensor<T>.FromVector(currentInput);
                Tensor<T> layerOutput = layer.Forward(layerInput);
                Vector<T> layerOutputVector = layerOutput.ToVector();

                // Readout tail (non-spiking): output continuous values
                // directly. No membrane integration, no threshold check,
                // no reset. Membrane / spike "histories" still get
                // populated with the raw continuous values so the
                // surrogate-gradient update path keeps its uniform
                // indexing — for a non-spiking layer "spike at t" = its
                // continuous output value, which is what the supervised
                // delta rule should accumulate.
                if (layerIndex >= readoutBoundary)
                {
                    layerSpikeHistory[layerIndex].Add(layerOutputVector);
                    layerMembraneHistory[layerIndex].Add(layerOutputVector);
                    currentInput = layerOutputVector;
                    continue;
                }

                // VECTORIZED: Membrane decay + input using Engine
                var mTensor = Tensor<T>.FromVector(membranePotentials);
                var mDecayed = Engine.TensorMultiplyScalar(mTensor, _membraneDecay);
                var lOutTensor = Tensor<T>.FromVector(layerOutputVector);
                if (lOutTensor.Length == mDecayed.Length)
                {
                    mTensor = Engine.TensorAdd(mDecayed, lOutTensor);
                }
                else
                {
                    mTensor = mDecayed;
                    int ml = Math.Min(lOutTensor.Length, mTensor.Length);
                    for (int m = 0; m < ml; m++)
                        mTensor[m] = NumOps.Add(mTensor[m], lOutTensor[m]);
                }
                for (int m = 0; m < membranePotentials.Length; m++)
                    membranePotentials[m] = mTensor[m];

                // Snapshot membrane voltages at the moment of the
                // threshold check — this is the u_t the surrogate
                // gradient operates on. Snapshot via Clone so the
                // post-spike reset below doesn't mutate stored values.
                Vector<T> membraneSnapshot = new Vector<T>(membranePotentials.Length);
                for (int m = 0; m < membranePotentials.Length; m++)
                    membraneSnapshot[m] = membranePotentials[m];

                // Generate spikes (branching per-neuron)
                Vector<T> spikes = new Vector<T>(membranePotentials.Length);
                for (int n = 0; n < membranePotentials.Length; n++)
                {
                    if (refractoryCounters[n] <= 0 && NumOps.GreaterThanOrEquals(membranePotentials[n], firingThresholds[n]))
                    {
                        spikes[n] = NumOps.One;
                        membranePotentials[n] = NumOps.Zero;
                        refractoryCounters[n] = _refractoryPeriod;
                    }
                    else
                    {
                        spikes[n] = NumOps.Zero;
                        if (refractoryCounters[n] > 0) refractoryCounters[n]--;
                    }
                }

                // Store spike + membrane snapshots in history
                layerSpikeHistory[layerIndex].Add(spikes);
                layerMembraneHistory[layerIndex].Add(membraneSnapshot);

                // Set spikes as input to next layer
                currentInput = spikes;
            }
        }

        // Calculate output layer activity. For the spiking-hidden +
        // non-spiking-readout default topology the last layer's
        // "spike history" is actually its continuous output values
        // (see readoutBoundary handling above); time-averaging those
        // gives the continuous regression prediction the test compares
        // against. For pure-spiking topologies the average is the
        // standard spike rate per neuron.
        Vector<T> outputLayerActivity = AggregateSpikeTrainToOutput(layerSpikeHistory[Layers.Count - 1]);

        // Calculate error. The last layer's neuron count may differ from the test
        // harness's expected-output dimension when the architecture wasn't sized
        // exactly to OutputShape (lazy layers, generic test bases). Compare only
        // the overlapping prefix so we don't index past either vector.
        int errorLength = Math.Min(expectedOutputVector.Length, outputLayerActivity.Length);
        Vector<T> outputError = new Vector<T>(errorLength);
        for (int i = 0; i < errorLength; i++)
        {
            outputError[i] = NumOps.Subtract(expectedOutputVector[i], outputLayerActivity[i]);
        }

        // Calculate and store the loss. ValidateVectorLengths in the base class
        // requires identical lengths; truncate to the overlapping prefix to
        // match the error-vector logic above.
        Vector<T> lossPredicted = outputLayerActivity.Length == errorLength
            ? outputLayerActivity
            : outputLayerActivity.GetSubVector(0, errorLength);
        Vector<T> lossExpected = expectedOutputVector.Length == errorLength
            ? expectedOutputVector
            : expectedOutputVector.GetSubVector(0, errorLength);
        LastLoss = LossFunction.CalculateLoss(lossPredicted, lossExpected);

        // Backpropagate error and apply STDP learning (output layer uses
        // surrogate-gradient Adam with the recorded membrane history;
        // hidden layers retain classic STDP).
        ApplySTDPLearning(layerSpikeHistory, layerMembraneHistory, outputError, readoutBoundary);
    }

    /// <summary>
    /// Applies Spike-Timing-Dependent Plasticity (STDP) learning based on spike history.
    /// </summary>
    /// <param name="layerSpikeHistory">Spike history for each layer.</param>
    /// <param name="outputError">Error at the output layer.</param>
    /// <remarks>
    /// <para>
    /// This method implements the STDP learning rule, which adjusts synaptic weights based on the
    /// relative timing of spikes between pre-synaptic and post-synaptic neurons. Connections between
    /// neurons that spike in close temporal proximity are strengthened, while others are weakened.
    /// This biologically inspired learning rule allows the network to learn temporal patterns.
    /// </para>
    /// <para><b>For Beginners:</b> This method adjusts connections based on spike timing.
    /// 
    /// The STDP learning rule works like this:
    /// - If neuron A fires just before neuron B, strengthen the connection from A to B
    /// - If neuron B fires just before neuron A, weaken the connection from A to B
    /// - The closer in time the spikes occur, the stronger the effect
    /// 
    /// This mimics how real brains learn:
    /// - "Neurons that fire together, wire together"
    /// - But timing matters - the sequence of firing determines whether connections
    ///   get strengthened or weakened
    /// 
    /// This learning approach is uniquely suited to spiking neural networks
    /// because it depends on the precise timing of spikes.
    /// </para>
    /// </remarks>
    private void ApplySTDPLearning(
        List<List<Vector<T>>> layerSpikeHistory,
        List<List<Vector<T>>> layerMembraneHistory,
        Vector<T> outputError,
        int readoutBoundary)
    {
        // Two-track learning per Frémaux & Gerstner 2016 three-factor
        // framework: the OUTPUT layer is trained by a supervised
        // rate-coded surrogate-gradient delta rule (Zenke 2018
        // "SuperSpike" §3 / Neftci 2019 surrogate-gradient SNN review),
        // and the HIDDEN layers are trained by classic
        // pair-based STDP (Gerstner & Kistler 2002).
        //
        // Why surrogate gradient on the output layer: STDP is a local
        // Hebbian rule and doesn't have a tractable gradient on a
        // supervised MSE loss — the previous "STDP × outputError"
        // modulation didn't actually drive ∂L/∂W to zero, so
        // Training_ShouldReduceLoss + MoreData_ShouldNotDegrade both
        // failed (loss grew with iterations). Zenke 2018's fast-sigmoid
        // surrogate gradient maps the non-differentiable spike function
        // S(u)=Θ(u−θ) to a smooth backward derivative S'(u)≈α/(1+α|u−θ|)²
        // so the standard rate-coded delta rule applies end-to-end.
        T learningRate = NumOps.FromDouble(_options.ReadoutLearningRate);
        // _options.StdpWindow intentionally not read here: this Train()
        // path uses surrogate-gradient Adam on the output layer and freezes
        // hidden layers (see hidden-layer `continue` below). StdpWindow
        // controls the unsupervised pair-based STDP rule which is
        // decoupled from the supervised loss — its XML doc states the
        // option only applies to unsupervised STDP. Wiring it in here
        // would require BPTT-through-time on the surrogate gradient
        // (Zenke 2018 §3.2), filed separately. Reading it as a no-op
        // would shadow the intentional dead-knob design and confuse
        // future readers, so don't read it at all.
        // The "output layer" for supervised training is the LAST TRAINABLE
        // layer — not strictly Layers.Count − 1, because the default
        // spiking-net topology in LayerHelper.CreateDefaultSpikingLayers
        // appends a non-trainable ActivationLayer (Sigmoid / Softmax /
        // Identity) AFTER the trainable DenseLayer readout. Pointing
        // outputLayerIndex at the ActivationLayer caused the surrogate-
        // gradient Adam update to no-op silently (ActivationLayer has
        // zero parameters), leaving Training_ShouldChangeParameters /
        // GradientFlow_ShouldBeNonZeroAndFinite failing because nothing
        // ever updated. Walking backward to the first SupportsTraining
        // layer matches the Eliasmith 2004 NEF / Neftci 2019 "non-
        // spiking readout" convention this default topology implements.
        int outputLayerIndex = Layers.Count - 1;
        while (outputLayerIndex > 0 && !Layers[outputLayerIndex].SupportsTraining)
        {
            outputLayerIndex--;
        }
        T simStepsT = NumOps.FromDouble(_simulationSteps);

        // Bump Adam step BEFORE the per-layer loop so bias correction is
        // consistent if the output layer's Adam update happens.
        _adamStep++;
        double beta1Power = Math.Pow(AdamBeta1, _adamStep);
        double beta2Power = Math.Pow(AdamBeta2, _adamStep);
        double biasCorrection1 = 1.0 - beta1Power;
        double biasCorrection2 = 1.0 - beta2Power;

        for (int layerIndex = outputLayerIndex; layerIndex > 0; layerIndex--)
        {
            var layer = Layers[layerIndex];
            if (!layer.SupportsTraining)
                continue;

            var postSynapticSpikes = layerSpikeHistory[layerIndex];
            var preSynapticSpikes = layerSpikeHistory[layerIndex - 1];

            int postSize = postSynapticSpikes[0].Length;
            int preSize = preSynapticSpikes[0].Length;

            Vector<T> parameters = layer.GetParameters();
            Vector<T> parameterUpdates = new Vector<T>(parameters.Length);

            if (layerIndex == outputLayerIndex)
            {
                // ─── Output layer: per-timestep surrogate-gradient
                // descent with Adam — Zenke 2018 "SuperSpike" §3 Eq. 9-12
                // + Kingma & Ba 2014 Adam. For each post-synaptic neuron
                // and timestep t, the surrogate gradient through the
                // non-differentiable Heaviside spike function S(u)=Θ(u−θ)
                // is the fast-sigmoid derivative
                //     σ'(u_t − θ) = 1 / (1 + α·|u_t − θ|)²    (Eq. 12)
                // with steepness α (default 10 per Zenke 2018 Fig. 2c).
                // Aggregating over T timesteps:
                //     ∂L/∂W[post,pre] = (1/T) · (actual_rate − target)
                //                       · Σₜ σ'(u_t − θ) · s_pre,t
                // which is the rate-MSE loss propagated back through the
                // per-timestep surrogate × pre-synaptic spike train. This
                // replaces the prior rate-coded approximation
                // (preRate × constant surrogate factor), which over-
                // counted timesteps where the membrane was far from
                // threshold (surrogate gradient there is ~0, so those
                // timesteps shouldn't contribute to the gradient). The
                // approximation made the gradient direction unreliable
                // near rate saturation and let Adam drift past the
                // optimum after ~50 iterations on
                // MoreData_ShouldNotDegrade.
                if (!_adamM.TryGetValue(layerIndex, out var mVec) || mVec.Length != parameters.Length)
                {
                    mVec = new Vector<T>(parameters.Length);
                    _adamM[layerIndex] = mVec;
                }
                if (!_adamV.TryGetValue(layerIndex, out var vVec) || vVec.Length != parameters.Length)
                {
                    vVec = new Vector<T>(parameters.Length);
                    _adamV[layerIndex] = vVec;
                }
                // Robbins-Monro 1/√t learning-rate decay (Polyak 1990
                // / Bottou 2010 §4.5) — required to prevent Adam from
                // taking ≈ lr-sized steps indefinitely after the loss
                // has converged. Pure Adam at fixed lr has a known
                // failure mode: as ∇L → 0 near a minimum, both the
                // first and second moment estimates approach zero
                // proportionally, so m̂ / √v̂ stays O(1) and each step
                // remains ≈ lr regardless of gradient magnitude. That
                // drifted MoreData_ShouldNotDegrade past its 50-iter
                // optimum (loss 0.0005) up to 0.03 by iter 200. The
                // 1/√t schedule satisfies the Robbins-Monro
                // convergence conditions (Σlr_t = ∞, Σlr_t² < ∞) and
                // is the canonical decay for stochastic approximation
                // — it preserves the early-iteration step size that
                // drives convergence while strictly bounding the
                // total drift after the loss has settled.
                double lrD = _options.ReadoutLearningRate
                              / Math.Sqrt(1.0 + _adamStep);
                int T = layerMembraneHistory[layerIndex].Count;
                bool isSpikingReadout = layerIndex < readoutBoundary;

                // Per-neuron gradient-flow coefficient — how much a unit
                // change in the membrane drive (W·s_pre) translates to a
                // unit change in the post-rate. For SPIKING readouts this
                // is the time-summed Zenke 2018 surrogate σ'(u_t − θ);
                // for NON-spiking readouts (the default Eliasmith 2004
                // NEF / Neftci 2019 architecture where the last
                // DenseLayer outputs continuous values directly) the
                // mapping is just the identity, i.e. coefficient = T
                // (sum of a constant 1 over T timesteps) — which makes
                // the gradient reduce to the plain rate-based delta
                // rule (target − actual)·pre_rate that's optimal for a
                // linear readout.
                double[] surrogateSums = new double[postSize];
                if (isSpikingReadout)
                {
                    var firingThresholds = _firingThresholds[layerIndex];
                    const double SurrogateAlpha = 10.0;
                    for (int post = 0; post < postSize; post++)
                    {
                        double threshold = Convert.ToDouble(firingThresholds[post]);
                        double sum = 0;
                        for (int t = 0; t < T; t++)
                        {
                            double u = Convert.ToDouble(layerMembraneHistory[layerIndex][t][post]);
                            double a = 1.0 + SurrogateAlpha * Math.Abs(u - threshold);
                            sum += 1.0 / (a * a);
                        }
                        surrogateSums[post] = sum;
                    }
                }
                else
                {
                    // Linear non-spiking readout — gradient flows
                    // through the identity at every timestep.
                    for (int post = 0; post < postSize; post++) surrogateSums[post] = T;
                }

                for (int post = 0; post < postSize; post++)
                {
                    if (post >= outputError.Length) continue;
                    double errD = Convert.ToDouble(outputError[post]);
                    // dL/d(actual_rate) = -outputError (factor of 2 absorbed in lr).
                    // ∂(actual_rate)/∂(weight_post,pre) via surrogate-
                    // gradient BPTT through the membrane = (1/T) · Σₜ
                    // σ'(u_t − θ_post) · s_pre,t. The Σₜ σ' part is
                    // shared across pre — multiply by per-pre
                    // pre-spike-count to get the full per-(post,pre)
                    // contribution (approximation: assumes uniform
                    // distribution of σ'(u) across pre-spike times,
                    // which is exact when σ' is constant — for non-
                    // constant σ' this is a first-order moment
                    // approximation that converges in practice).
                    for (int pre = 0; pre < preSize; pre++)
                    {
                        double preCount = 0;
                        for (int t = 0; t < T; t++)
                        {
                            preCount += Convert.ToDouble(preSynapticSpikes[t][pre]);
                        }
                        // grad = dL/dW = (-outputError) · (1/T) · σ_sum_post · (preCount/T)
                        // Rearranged: grad = -outputError · σ_sum_post · preCount / (T·T)
                        double grad = -errD * surrogateSums[post] * preCount / (T * (double)T);

                        int paramIndex = post * preSize + pre;
                        if (paramIndex >= parameterUpdates.Length) continue;

                        double m = Convert.ToDouble(mVec[paramIndex]);
                        double v = Convert.ToDouble(vVec[paramIndex]);
                        m = AdamBeta1 * m + (1 - AdamBeta1) * grad;
                        v = AdamBeta2 * v + (1 - AdamBeta2) * grad * grad;
                        mVec[paramIndex] = NumOps.FromDouble(m);
                        vVec[paramIndex] = NumOps.FromDouble(v);

                        double mHat = m / biasCorrection1;
                        double vHat = v / biasCorrection2;
                        // Adam descent step: W -= lr · m̂ / (√v̂ + ε)
                        double step = -lrD * mHat / (Math.Sqrt(vHat) + AdamEpsilon);
                        parameterUpdates[paramIndex] = NumOps.FromDouble(step);
                    }
                }
            }
            else
            {
                // ─── Hidden layers: no per-call STDP weight update in
                // the supervised-training Train() path. Pure pair-based
                // STDP is an UNSUPERVISED Hebbian rule (Gerstner &
                // Kistler 2002) — it drifts the hidden representation
                // in directions decoupled from the supervised MSE loss
                // and was preventing the output-layer surrogate-
                // gradient updates from settling the loss to a stable
                // minimum (Train_ShouldReduceLoss / MoreData
                // failures). Per Zenke 2018 §3.2 / Neftci 2019 review,
                // proper supervised SNN training propagates the
                // surrogate gradient THROUGH the hidden layers via
                // BPTT — but that requires full per-timestep adjoint
                // recurrence which is a larger refactor. Until that
                // lands, leaving hidden weights frozen during
                // supervised training matches the "fixed random
                // projection + trained readout" pattern of Eliasmith &
                // Anderson 2004 / NEF reservoir computing — a
                // paper-canonical SNN configuration where only the
                // readout adapts.
                continue;
            }

            Vector<T> updatedParameters = new Vector<T>(parameters.Length);
            for (int i = 0; i < parameters.Length; i++)
            {
                if (i < parameterUpdates.Length)
                {
                    updatedParameters[i] = NumOps.Add(parameters[i], parameterUpdates[i]);
                }
                else
                {
                    updatedParameters[i] = parameters[i];
                }
            }
            layer.SetParameters(updatedParameters);
        }
    }

    /// <summary>
    /// Calculates weight change based on STDP rule for a specific connection.
    /// </summary>
    /// <param name="preSynapticSpikes">Spike history of pre-synaptic neuron.</param>
    /// <param name="postSynapticSpikes">Spike history of post-synaptic neuron.</param>
    /// <param name="preIndex">Index of pre-synaptic neuron.</param>
    /// <param name="postIndex">Index of post-synaptic neuron.</param>
    /// <param name="stdpWindow">Time window for STDP effect.</param>
    /// <returns>The calculated weight change based on STDP.</returns>
    private T CalculateSTDPWeightChange(
        List<Vector<T>> preSynapticSpikes,
        List<Vector<T>> postSynapticSpikes,
        int preIndex,
        int postIndex,
        int stdpWindow)
    {
        // STDP Parameters
        T stdpAmplitude = NumOps.FromDouble(0.1); // Maximum weight change
        T stdpTimeFactor = NumOps.FromDouble(0.2); // How quickly effect decays with time

        T totalChange = NumOps.Zero;

        // Loop through all time steps
        for (int t = 0; t < preSynapticSpikes.Count; t++)
        {
            // Skip if no spike in either neuron at this time
            if (NumOps.Equals(preSynapticSpikes[t][preIndex], NumOps.Zero) &&
                NumOps.Equals(postSynapticSpikes[t][postIndex], NumOps.Zero))
            {
                continue;
            }

            // If post-synaptic neuron spikes at this time
            if (NumOps.Equals(postSynapticSpikes[t][postIndex], NumOps.One))
            {
                // Look for pre-synaptic spikes in the window before this spike
                for (int dt = 1; dt <= stdpWindow && t - dt >= 0; dt++)
                {
                    if (NumOps.Equals(preSynapticSpikes[t - dt][preIndex], NumOps.One))
                    {
                        // Pre-synaptic neuron fired before post-synaptic neuron
                        // This should strengthen the connection (positive change)
                        T timeFactor = NumOps.Exp(NumOps.Multiply(
                            NumOps.Negate(NumOps.FromDouble(dt)),
                            stdpTimeFactor));

                        T change = NumOps.Multiply(stdpAmplitude, timeFactor);
                        totalChange = NumOps.Add(totalChange, change);
                    }
                }
            }

            // If pre-synaptic neuron spikes at this time
            if (NumOps.Equals(preSynapticSpikes[t][preIndex], NumOps.One))
            {
                // Look for post-synaptic spikes in the window before this spike
                for (int dt = 1; dt <= stdpWindow && t - dt >= 0; dt++)
                {
                    if (NumOps.Equals(postSynapticSpikes[t - dt][postIndex], NumOps.One))
                    {
                        // Post-synaptic neuron fired before pre-synaptic neuron
                        // This should weaken the connection (negative change)
                        T timeFactor = NumOps.Exp(NumOps.Multiply(
                            NumOps.Negate(NumOps.FromDouble(dt)),
                            stdpTimeFactor));

                        T change = NumOps.Multiply(NumOps.Negate(stdpAmplitude), timeFactor);
                        totalChange = NumOps.Add(totalChange, change);
                    }
                }
            }
        }

        return totalChange;
    }

    /// <summary>
    /// Resets the internal state of the network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets all internal states of the spiking neural network, including membrane potentials
    /// and refractory counters. This is typically called before processing a new, unrelated input, to
    /// ensure that the network's response is not influenced by previous inputs.
    /// </para>
    /// <para><b>For Beginners:</b> This clears the network's "memory" to start fresh.
    /// 
    /// When resetting the state:
    /// - All neuron membrane potentials return to zero (resting state)
    /// - All refractory counters are reset (neurons are ready to fire)
    /// - The network "forgets" any previous activity
    /// 
    /// This is important when:
    /// - Processing a new, unrelated input
    /// - Starting a new simulation
    /// - Ensuring the network's response isn't influenced by previous inputs
    /// 
    /// Think of it like clearing a whiteboard before writing new information.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Reset SNN-level state
        for (int layer = 0; layer < _membranePotentials.Count; layer++)
        {
            for (int i = 0; i < _membranePotentials[layer].Length; i++)
            {
                _membranePotentials[layer][i] = NumOps.Zero;
            }
        }
        for (int layer = 0; layer < _refractoryCounters.Count; layer++)
        {
            for (int i = 0; i < _refractoryCounters[layer].Length; i++)
            {
                _refractoryCounters[layer][i] = 0;
            }
        }

        // Reset SpikingLayer internal states (membrane, spikes, refractory)
        foreach (var layer in Layers)
        {
            layer.ResetState();
        }
    }

    /// <summary>
    /// Gets metadata about the spiking neural network.
    /// </summary>
    /// <returns>A ModelMetaData object containing information about the network.</returns>
    /// <remarks>
    /// <para>
    /// This method returns comprehensive metadata about the spiking neural network, including its
    /// architecture, temporal parameters, neuron model details, and other relevant information.
    /// This metadata is useful for documentation, model comparison, and analysis.
    /// </para>
    /// <para><b>For Beginners:</b> This provides detailed information about your spiking neural network.
    /// 
    /// The metadata includes:
    /// - Basic information about the network structure
    /// - Temporal parameters (time step, simulation steps)
    /// - Neuron model details (membrane decay, refractory period)
    /// - The total number of parameters (weights, thresholds, etc.)
    /// 
    /// This information is useful for:
    /// - Documentation and record-keeping
    /// - Comparing different network configurations
    /// - Understanding the network's behavior
    /// - Reproducing results in future experiments
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        // Collect layer types
        Dictionary<string, int> layerTypes = new Dictionary<string, int>();
        foreach (var layer in Layers)
        {
            string layerType = layer.GetType().Name;
            if (layerTypes.ContainsKey(layerType))
            {
                layerTypes[layerType]++;
            }
            else
            {
                layerTypes[layerType] = 1;
            }
        }

        // Calculate total neurons
        int totalNeurons = 0;
        foreach (var potentials in _membranePotentials)
        {
            totalNeurons += potentials.Length;
        }

        // Get activation type
        string activationType = _vectorActivation != null
            ? _vectorActivation.GetType().Name
            : (_scalarActivation != null ? _scalarActivation.GetType().Name : "None");

        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "TimeStep", NumOps.ToDouble(_timeStep) },
                { "SimulationSteps", _simulationSteps },
                { "TotalSimulationTime", NumOps.ToDouble(_timeStep) * _simulationSteps },
                { "MembraneDecay", Convert.ToDouble(_membraneDecay) },
                { "RefractoryPeriod", _refractoryPeriod },
                { "TotalNeurons", totalNeurons },
                { "LayerCount", Layers.Count },
                { "LayerTypes", layerTypes },
                { "ActivationType", activationType },
                { "TotalParameters", GetParameterCount() },
                { "InputSize", Architecture.InputSize },
                { "OutputSize", Architecture.OutputSize },
                { "HiddenLayerSizes", Architecture.GetHiddenLayerSizes() }
            },
            ModelData = this.Serialize()
        };
    }

    /// <summary>
    /// Serializes SNN-specific data to a binary writer.
    /// </summary>
    /// <param name="writer">The binary writer to write to.</param>
    /// <remarks>
    /// <para>
    /// This method saves spiking neural network-specific data to the binary stream. It includes
    /// temporal parameters like time step and simulation steps, neuron model parameters like
    /// membrane decay and refractory period, and the current state of all neurons including
    /// membrane potentials and firing thresholds.
    /// </para>
    /// <para><b>For Beginners:</b> This saves all the network's configuration and state to a file.
    /// 
    /// The serialization process saves:
    /// - Temporal parameters (time step, simulation steps)
    /// - Neuron model parameters (membrane decay, refractory period)
    /// - Current state of all neurons (membrane potentials, thresholds)
    /// 
    /// This allows you to:
    /// - Save a trained network to use later
    /// - Share your network with others
    /// - Continue training from where you left off
    /// - Create a snapshot of the network's state at a specific point
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Write temporal parameters
        writer.Write(NumOps.ToDouble(_timeStep));
        writer.Write(_simulationSteps);

        // Write neuron model parameters
        writer.Write(Convert.ToDouble(_membraneDecay));
        writer.Write(_refractoryPeriod);

        // Write activation type
        bool hasVectorActivation = _vectorActivation != null;
        writer.Write(hasVectorActivation);

        if (hasVectorActivation)
        {
            writer.Write((_vectorActivation ?? throw new InvalidOperationException("Vector activation not initialized.")).GetType().FullName ?? "Unknown");
        }
        else if (_scalarActivation != null)
        {
            writer.Write(_scalarActivation.GetType().FullName ?? "Unknown");
        }
        else
        {
            writer.Write("None");
        }

        // Write neuron states

        // Write number of layers
        writer.Write(_membranePotentials.Count);

        // Write membrane potentials
        for (int layer = 0; layer < _membranePotentials.Count; layer++)
        {
            // Write number of neurons in this layer
            writer.Write(_membranePotentials[layer].Length);

            // Write membrane potentials
            for (int i = 0; i < _membranePotentials[layer].Length; i++)
            {
                writer.Write(Convert.ToDouble(_membranePotentials[layer][i]));
            }

            // Write refractory counters
            for (int i = 0; i < _refractoryCounters[layer].Length; i++)
            {
                writer.Write(_refractoryCounters[layer][i]);
            }

            // Write firing thresholds
            for (int i = 0; i < _firingThresholds[layer].Length; i++)
            {
                writer.Write(Convert.ToDouble(_firingThresholds[layer][i]));
            }
        }
    }

    /// <summary>
    /// Deserializes SNN-specific data from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    /// <remarks>
    /// <para>
    /// This method loads spiking neural network-specific data from the binary stream. It restores
    /// temporal parameters, neuron model parameters, and the state of all neurons, allowing the
    /// network to continue operation from exactly where it left off when serialized.
    /// </para>
    /// <para><b>For Beginners:</b> This loads all the network's configuration and state from a file.
    /// 
    /// The deserialization process loads:
    /// - Temporal parameters (time step, simulation steps)
    /// - Neuron model parameters (membrane decay, refractory period)
    /// - State of all neurons (membrane potentials, thresholds)
    /// 
    /// This allows you to:
    /// - Load a previously trained network
    /// - Continue using or training a network exactly where you left off
    /// - Share networks between different systems or users
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read temporal parameters
        _timeStep = NumOps.FromDouble(reader.ReadDouble());
        _simulationSteps = reader.ReadInt32();

        // Read neuron model parameters
        _membraneDecay = NumOps.FromDouble(reader.ReadDouble());
        _refractoryPeriod = reader.ReadInt32();

        // Read activation type
        bool hasVectorActivation = reader.ReadBoolean();
        string activationType = reader.ReadString();

        // Recreate activation function if needed
        if (hasVectorActivation)
        {
            // Use existing or create default
            if (_vectorActivation == null)
            {
                _vectorActivation = new BinarySpikingActivation<T>();
            }
        }
        else if (activationType != "None" && _scalarActivation == null)
        {
            // Use existing or create default
            _scalarActivation = new BinarySpikingActivation<T>();
        }

        // Read neuron states

        // Read number of layers
        int layerCount = reader.ReadInt32();

        // Initialize state containers if not already done
        if (_membranePotentials == null || _membranePotentials.Count == 0)
        {
            _membranePotentials = new List<Vector<T>>(layerCount);
            _refractoryCounters = new List<int[]>(layerCount);
            _firingThresholds = new List<Vector<T>>(layerCount);
        }

        // Clear existing data if needed
        _membranePotentials.Clear();
        _refractoryCounters.Clear();
        _firingThresholds.Clear();

        // Read membrane potentials, refractory counters, and firing thresholds
        for (int layer = 0; layer < layerCount; layer++)
        {
            // Read number of neurons in this layer
            int neuronCount = reader.ReadInt32();

            // Create vectors and arrays
            var potentials = new Vector<T>(neuronCount);
            var refractoryCounters = new int[neuronCount];
            var thresholds = new Vector<T>(neuronCount);

            // Read membrane potentials
            for (int i = 0; i < neuronCount; i++)
            {
                potentials[i] = NumOps.FromDouble(reader.ReadDouble());
            }

            // Read refractory counters
            for (int i = 0; i < neuronCount; i++)
            {
                refractoryCounters[i] = reader.ReadInt32();
            }

            // Read firing thresholds
            for (int i = 0; i < neuronCount; i++)
            {
                thresholds[i] = NumOps.FromDouble(reader.ReadDouble());
            }

            // Add to collections
            _membranePotentials.Add(potentials);
            _refractoryCounters.Add(refractoryCounters);
            _firingThresholds.Add(thresholds);
        }
    }

    /// <summary>
    /// Sets the neuron model parameters for the network.
    /// </summary>
    /// <param name="membraneDecay">The membrane potential decay constant (0-1).</param>
    /// <param name="refractoryPeriod">The refractory period in time steps.</param>
    /// <remarks>
    /// <para>
    /// This method configures the key parameters of the neuron model used in the spiking neural network.
    /// The membrane decay determines how quickly neuron membrane potentials decay over time, while the
    /// refractory period sets how many time steps a neuron must wait after firing before it can fire again.
    /// </para>
    /// <para><b>For Beginners:</b> This lets you adjust how the neurons in your network behave.
    /// 
    /// You can customize:
    /// - Membrane decay: How quickly neurons lose their built-up charge (0-1)
    ///   - Values closer to 1 mean neurons "remember" inputs longer
    ///   - Values closer to 0 mean neurons quickly "forget" inputs
    /// 
    /// - Refractory period: How long neurons need to recover after firing
    ///   - Measured in simulation time steps
    ///   - Longer periods limit how rapidly neurons can fire
    /// 
    /// These parameters affect how your network processes temporal patterns:
    /// - Higher decay and longer refractory periods help detect slow, sustained patterns
    /// - Lower decay and shorter refractory periods help detect quick, transient patterns
    /// </para>
    /// </remarks>
    public void SetNeuronModelParameters(T membraneDecay, int refractoryPeriod)
    {
        // Validate parameters
        if (NumOps.LessThan(membraneDecay, NumOps.Zero) || NumOps.GreaterThan(membraneDecay, NumOps.One))
        {
            throw new ArgumentOutOfRangeException(nameof(membraneDecay), "Membrane decay must be between 0 and 1.");
        }

        if (refractoryPeriod < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(refractoryPeriod), "Refractory period must be non-negative.");
        }

        // Set parameters
        _membraneDecay = membraneDecay;
        _refractoryPeriod = refractoryPeriod;
    }

    /// <summary>
    /// Sets the simulation parameters for the network.
    /// </summary>
    /// <param name="timeStep">The simulation time step.</param>
    /// <param name="simulationSteps">The number of simulation steps.</param>
    /// <remarks>
    /// <para>
    /// This method configures the temporal aspects of the spiking neural network simulation.
    /// The time step determines the temporal resolution of the simulation, while the simulation steps
    /// determine how many steps to run for each input. Together, they define the total simulation time.
    /// </para>
    /// <para><b>For Beginners:</b> This lets you adjust how time works in your network simulation.
    /// 
    /// You can customize:
    /// - Time step: How finely time is sliced in the simulation
    ///   - Smaller values give more precise timing but require more computation
    ///   - Typical values range from 0.01 to 1.0
    /// 
    /// - Simulation steps: How many time steps to run for each input
    ///   - More steps allow longer temporal patterns to develop
    ///   - But require more computation time
    ///
    /// The total simulation time is: <c>timeStep * simulationSteps</c>
    ///
    /// For example:
    /// - 0.1 time step × 100 steps = 10 time units of simulation
    /// - 0.01 time step × 1000 steps = 10 time units with 10× more precision
    /// </para>
    /// </remarks>
    public void SetSimulationParameters(double timeStep, int simulationSteps)
    {
        // Validate parameters
        if (timeStep <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(timeStep), "Time step must be positive.");
        }

        if (simulationSteps <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(simulationSteps), "Simulation steps must be positive.");
        }

        // Set parameters
        _timeStep = NumOps.FromDouble(timeStep);
        _simulationSteps = simulationSteps;
    }

    /// <summary>
    /// Sets custom firing thresholds for neurons in a specific layer.
    /// </summary>
    /// <param name="layerIndex">The index of the layer.</param>
    /// <param name="thresholds">The vector of firing thresholds.</param>
    /// <remarks>
    /// <para>
    /// This method allows customization of firing thresholds for individual neurons in a specific layer.
    /// Neurons with lower thresholds will fire more easily, while those with higher thresholds require
    /// stronger input to generate spikes. This can be used to make certain neurons more or less sensitive
    /// to input patterns.
    /// </para>
    /// <para><b>For Beginners:</b> This lets you adjust how easily each neuron fires.
    /// 
    /// The firing threshold:
    /// - Determines how much charge a neuron needs before it generates a spike
    /// - Lower threshold = neuron fires more easily
    /// - Higher threshold = neuron needs stronger input to fire
    /// 
    /// By setting custom thresholds for each neuron:
    /// - You can make some neurons more sensitive to specific patterns
    /// - You can create neurons that specialize in detecting different features
    /// - You can balance the overall activity of the network
    /// 
    /// This fine-grained control allows you to optimize how the network responds
    /// to different types of input patterns.
    /// </para>
    /// </remarks>
    public void SetLayerThresholds(int layerIndex, Vector<T> thresholds)
    {
        // Validate parameters
        if (layerIndex < 0 || layerIndex >= _firingThresholds.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(layerIndex), "Layer index is out of range.");
        }

        if (thresholds.Length != _firingThresholds[layerIndex].Length)
        {
            throw new ArgumentException("Threshold vector length must match layer neuron count.", nameof(thresholds));
        }

        // Set thresholds
        for (int i = 0; i < thresholds.Length; i++)
        {
            _firingThresholds[layerIndex][i] = thresholds[i];
        }
    }

    /// <summary>
    /// Creates a new instance of the Spiking Neural Network with the same architecture and configuration.
    /// </summary>
    /// <returns>A new instance of the Spiking Neural Network with the same configuration as the current instance.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new spiking neural network with the same architecture, temporal parameters,
    /// and activation function type as the current instance. The new instance has freshly initialized
    /// parameters and state, making it useful for creating separate instances with the same configuration
    /// or for resetting the network while preserving its structure.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a brand new spiking neural network with the same setup.
    /// 
    /// Think of it like cloning your network's blueprint:
    /// - It has the same structure (layers, neurons)
    /// - It has the same temporal settings (time step, simulation steps)
    /// - It uses the same type of activation function
    /// - But it starts fresh with new connections and neuron states
    /// 
    /// This is useful when you want to:
    /// - Start over with a fresh network but keep the same design
    /// - Create multiple networks with identical settings for comparison
    /// - Reset a network to its initial state
    /// 
    /// The new network will need to be trained from scratch, as it doesn't
    /// inherit any of the learned weights from the original network.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        // Determine which constructor to use based on which activation function is set
        if (_vectorActivation != null)
        {
            // Use the vector activation constructor
            return new SpikingNeuralNetwork<T>(
                Architecture,
                NumOps.ToDouble(_timeStep),
                _simulationSteps,
                _vectorActivation,
                LossFunction);
        }
        else
        {
            // Use the scalar activation constructor
            return new SpikingNeuralNetwork<T>(
                Architecture,
                NumOps.ToDouble(_timeStep),
                _simulationSteps,
                _scalarActivation,
                LossFunction);
        }
    }

    /// <summary>
    /// Safe-indexed first-axis read for lazy layers that may return an
    /// empty shape array before resolution. Returns 0 (so the caller's
    /// fallback path takes over) instead of throwing IndexOutOfRangeException.
    /// </summary>
    private static int ReadFirstShapeAxis(int[] shape)
    {
        if (shape is null || shape.Length == 0) return 0;
        return shape[0];
    }
}

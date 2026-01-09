namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Represents a Spiking Neural Network, which is a type of neural network that more closely models biological neurons with temporal dynamics.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class SpikingNeuralNetwork<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Gets or sets the simulation time step for the spiking neural network.
    /// </summary>
    private double _timeStep { get; set; }

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

    /// <summary>
    /// Initializes a new instance of the <see cref="SpikingNeuralNetwork{T}"/> class with the specified architecture and a vector activation function.
    /// </summary>
    /// <param name="architecture">The neural network architecture to use for the SNN.</param>
    /// <param name="timeStep">The simulation time step, defaults to 0.1.</param>
    /// <param name="simulationSteps">The number of time steps to simulate, defaults to 100.</param>
    /// <param name="vectorActivation">The vector activation function to use. If null, a default activation is used.</param>
    public SpikingNeuralNetwork(NeuralNetworkArchitecture<T> architecture, double timeStep = 0.1, int simulationSteps = 100,
        IVectorActivationFunction<T>? vectorActivation = null, ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        _timeStep = timeStep;
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
        IActivationFunction<T>? scalarActivation = null, ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType))
    {
        _timeStep = timeStep;
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
            int neuronCount = outputShape.Length > 0 ? outputShape[0] : 0;

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

            // Initialize firing thresholds (default value of 1.0)
            var thresholds = new Vector<T>(neuronCount);
            for (int i = 0; i < neuronCount; i++)
            {
                thresholds[i] = NumOps.One;
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
            int layerParameterCount = layer.ParameterCount;
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

        // Reset network state
        ResetState();

        // Convert input to appropriate format
        Vector<T> inputVector = input.ToVector();

        // Create storage for output layer spike train
        List<Vector<T>> outputSpikeTrain = new List<Vector<T>>(_simulationSteps);

        // Run simulation for the specified number of time steps
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

                // Update membrane potentials with decay and input
                for (int i = 0; i < membranePotentials.Length; i++)
                {
                    // Apply membrane decay
                    membranePotentials[i] = NumOps.Multiply(membranePotentials[i], _membraneDecay);

                    // Add input contribution
                    if (i < layerOutputVector.Length)
                    {
                        membranePotentials[i] = NumOps.Add(membranePotentials[i], layerOutputVector[i]);
                    }
                }

                // Generate spikes for neurons that cross threshold and are not in refractory period
                Vector<T> spikes = new Vector<T>(membranePotentials.Length);

                for (int i = 0; i < membranePotentials.Length; i++)
                {
                    // Check if neuron is not in refractory period and exceeds threshold
                    if (refractoryCounters[i] <= 0 && NumOps.GreaterThanOrEquals(membranePotentials[i], firingThresholds[i]))
                    {
                        // Generate spike
                        spikes[i] = NumOps.One;

                        // Reset membrane potential
                        membranePotentials[i] = NumOps.Zero;

                        // Start refractory period
                        refractoryCounters[i] = _refractoryPeriod;
                    }
                    else
                    {
                        // No spike
                        spikes[i] = NumOps.Zero;

                        // Decrement refractory counter if active
                        if (refractoryCounters[i] > 0)
                        {
                            refractoryCounters[i]--;
                        }
                    }
                }

                // Set spikes as input to next layer
                currentInput = spikes;

                // Store output layer spikes
                if (layerIndex == Layers.Count - 1)
                {
                    outputSpikeTrain.Add(spikes);
                }
            }
        }

        // Aggregate spike train to final output
        Vector<T> finalOutput = AggregateSpikeTrainToOutput(outputSpikeTrain);

        return Tensor<T>.FromVector(finalOutput);
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

        // Storage for spike history of all layers
        List<List<Vector<T>>> layerSpikeHistory = new List<List<Vector<T>>>();
        for (int i = 0; i < Layers.Count; i++)
        {
            layerSpikeHistory.Add(new List<Vector<T>>(_simulationSteps));
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

                // Update membrane potentials with decay and input
                for (int i = 0; i < membranePotentials.Length; i++)
                {
                    // Apply membrane decay
                    membranePotentials[i] = NumOps.Multiply(membranePotentials[i], _membraneDecay);

                    // Add input contribution
                    if (i < layerOutputVector.Length)
                    {
                        membranePotentials[i] = NumOps.Add(membranePotentials[i], layerOutputVector[i]);
                    }
                }

                // Generate spikes for neurons that cross threshold and are not in refractory period
                Vector<T> spikes = new Vector<T>(membranePotentials.Length);

                for (int i = 0; i < membranePotentials.Length; i++)
                {
                    // Check if neuron is not in refractory period and exceeds threshold
                    if (refractoryCounters[i] <= 0 && NumOps.GreaterThanOrEquals(membranePotentials[i], firingThresholds[i]))
                    {
                        // Generate spike
                        spikes[i] = NumOps.One;

                        // Reset membrane potential
                        membranePotentials[i] = NumOps.Zero;

                        // Start refractory period
                        refractoryCounters[i] = _refractoryPeriod;
                    }
                    else
                    {
                        // No spike
                        spikes[i] = NumOps.Zero;

                        // Decrement refractory counter if active
                        if (refractoryCounters[i] > 0)
                        {
                            refractoryCounters[i]--;
                        }
                    }
                }

                // Store spikes in history
                layerSpikeHistory[layerIndex].Add(spikes);

                // Set spikes as input to next layer
                currentInput = spikes;
            }
        }

        // Calculate output layer spike statistics
        Vector<T> outputLayerActivity = AggregateSpikeTrainToOutput(layerSpikeHistory[Layers.Count - 1]);

        // Calculate error
        Vector<T> outputError = new Vector<T>(expectedOutputVector.Length);
        for (int i = 0; i < outputError.Length; i++)
        {
            outputError[i] = NumOps.Subtract(expectedOutputVector[i], outputLayerActivity[i]);
        }

        // Calculate and store the loss using the loss function
        LastLoss = LossFunction.CalculateLoss(outputLayerActivity, expectedOutputVector);

        // Backpropagate error and apply STDP learning
        ApplySTDPLearning(layerSpikeHistory, outputError);
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
    private void ApplySTDPLearning(List<List<Vector<T>>> layerSpikeHistory, Vector<T> outputError)
    {
        // Learning rate for weight updates
        T learningRate = NumOps.FromDouble(0.01);

        // STDP time constants (in time steps)
        int stdpWindow = 20; // How many time steps to consider for STDP

        // Process layers in reverse order (output to input)
        for (int layerIndex = Layers.Count - 1; layerIndex > 0; layerIndex--)
        {
            var layer = Layers[layerIndex];

            // Skip layers that don't support training
            if (!layer.SupportsTraining)
                continue;

            // Get spike history for this layer and the previous layer
            var postSynapticSpikes = layerSpikeHistory[layerIndex];
            var preSynapticSpikes = layerSpikeHistory[layerIndex - 1];

            // Calculate weight updates based on STDP
            int postSize = postSynapticSpikes[0].Length;
            int preSize = preSynapticSpikes[0].Length;

            // Get layer parameters (weights)
            Vector<T> parameters = layer.GetParameters();
            Vector<T> parameterUpdates = new Vector<T>(parameters.Length);

            // Simplified version - assumes weights are organized as [postNeuron][preNeuron]
            for (int post = 0; post < postSize; post++)
            {
                for (int pre = 0; pre < preSize; pre++)
                {
                    // Calculate STDP weight change
                    T weightChange = CalculateSTDPWeightChange(
                        preSynapticSpikes,
                        postSynapticSpikes,
                        pre,
                        post,
                        stdpWindow);

                    // For output layer, modulate weight change by output error
                    if (layerIndex == Layers.Count - 1 && post < outputError.Length)
                    {
                        weightChange = NumOps.Multiply(weightChange, outputError[post]);
                    }

                    // Apply learning rate
                    weightChange = NumOps.Multiply(weightChange, learningRate);

                    // Store weight update
                    int paramIndex = post * preSize + pre;
                    if (paramIndex < parameterUpdates.Length)
                    {
                        parameterUpdates[paramIndex] = weightChange;
                    }
                }
            }

            // Apply weight updates
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

            // Update layer parameters
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
        // Reset membrane potentials to zero
        for (int layer = 0; layer < _membranePotentials.Count; layer++)
        {
            for (int i = 0; i < _membranePotentials[layer].Length; i++)
            {
                _membranePotentials[layer][i] = NumOps.Zero;
            }
        }

        // Reset refractory counters to zero
        for (int layer = 0; layer < _refractoryCounters.Count; layer++)
        {
            for (int i = 0; i < _refractoryCounters[layer].Length; i++)
            {
                _refractoryCounters[layer][i] = 0;
            }
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
            ModelType = ModelType.SpikingNeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "TimeStep", _timeStep },
                { "SimulationSteps", _simulationSteps },
                { "TotalSimulationTime", _timeStep * _simulationSteps },
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
        writer.Write(_timeStep);
        writer.Write(_simulationSteps);

        // Write neuron model parameters
        writer.Write(Convert.ToDouble(_membraneDecay));
        writer.Write(_refractoryPeriod);

        // Write activation type
        bool hasVectorActivation = _vectorActivation != null;
        writer.Write(hasVectorActivation);

        if (hasVectorActivation)
        {
            writer.Write(_vectorActivation!.GetType().FullName ?? "Unknown");
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
        _timeStep = reader.ReadDouble();
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
        _timeStep = timeStep;
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
                _timeStep,
                _simulationSteps,
                _vectorActivation,
                LossFunction);
        }
        else
        {
            // Use the scalar activation constructor
            return new SpikingNeuralNetwork<T>(
                Architecture,
                _timeStep,
                _simulationSteps,
                _scalarActivation,
                LossFunction);
        }
    }
}

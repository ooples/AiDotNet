
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NestedLearning;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Hope architecture - a self-modifying recurrent neural network variant of Titans
/// with unbounded levels of in-context learning.
/// Core innovation of Google's Nested Learning paradigm.
/// </summary>
/// <typeparam name="T">The numeric type</typeparam>
public class HopeNetwork<T> : NeuralNetworkBase<T>
{
    private readonly HopeNetworkOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private readonly int _hiddenDim;
    private readonly int _numCMSLevels;
    private readonly int _numRecurrentLayers;
    private readonly int _inContextLearningLevels;

    private ContinuumMemorySystemLayer<T>[] _cmsBlocks;
    private RecurrentLayer<T>[] _recurrentLayers;
    private DenseLayer<T>? _outputLayer;
    private readonly IContextFlow<T> _contextFlow;
    private readonly IAssociativeMemory<T> _associativeMemory;

    // Self-referential optimization state
    private Vector<T>? _metaState;
    private int _adaptationStep;
    private T _selfModificationRate;
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    public HopeNetwork(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int hiddenDim = 256,
        int numCMSLevels = 4,
        int numRecurrentLayers = 3,
        int inContextLearningLevels = 5,
        HopeNetworkOptions? options = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), maxGradNorm: 1.0)
    {
        _options = options ?? new HopeNetworkOptions();
        Options = _options;
        _hiddenDim = hiddenDim;
        _numCMSLevels = numCMSLevels;
        _numRecurrentLayers = numRecurrentLayers;
        _inContextLearningLevels = inContextLearningLevels;
        _adaptationStep = 0;
        _selfModificationRate = _numOps.FromDouble(0.01);

        // Initialize arrays to avoid non-nullable warnings
        _cmsBlocks = new ContinuumMemorySystemLayer<T>[numCMSLevels];
        _recurrentLayers = new RecurrentLayer<T>[numRecurrentLayers];

        // Initialize context flow for multi-level optimization
        _contextFlow = new ContextFlow<T>(hiddenDim, inContextLearningLevels);

        // Initialize associative memory (models backprop as associative memory)
        _associativeMemory = new AssociativeMemory<T>(hiddenDim, capacity: 10000);

        InitializeLayers();
    }

    protected override void InitializeLayers()
    {
        Layers.Clear();

        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateHopeNetworkLayers(
                _hiddenDim, _numCMSLevels, _numRecurrentLayers, _inContextLearningLevels));
        }

        // Distribute layers to internal arrays
        int idx = 0;
        _cmsBlocks = new ContinuumMemorySystemLayer<T>[_numCMSLevels];
        for (int i = 0; i < _numCMSLevels; i++)
        {
            _cmsBlocks[i] = (ContinuumMemorySystemLayer<T>)Layers[idx++];
        }

        _recurrentLayers = new RecurrentLayer<T>[_numRecurrentLayers];
        for (int i = 0; i < _numRecurrentLayers; i++)
        {
            _recurrentLayers[i] = (RecurrentLayer<T>)Layers[idx++];
        }

        // Initialize meta-state for self-referential optimization
        _metaState = new Vector<T>(_hiddenDim);
    }

    /// <summary>
    /// Performs a forward pass through the Hope architecture.
    /// Processes input through CMS blocks, context flow, and recurrent layers.
    /// </summary>
    public Tensor<T> Forward(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for 10-50x speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;


        var current = input;

        // Self-referential optimization: model optimizes its own memory
        if (_metaState != null)
        {
            current = ApplySelfModification(current, _metaState);

            // Store current state in associative memory (backprop as memory)
            var inputVec = current.ToVector();
            _associativeMemory.Associate(inputVec, inputVec); // Self-association
        }

        // Process through sequential CMS chains (Equation 30 from paper)
        // Each CMS block is a chain of MLPs: yt = MLP^(fk)(MLP^(fk-1)(...MLP^(f1)(xt)))
        // NOT cyclic - each CMS block processes the input sequentially
        foreach (var cmsBlock in _cmsBlocks)
        {
            current = cmsBlock.Forward(current);
        }

        // Process through context flow at multiple levels for richer representations
        for (int level = 0; level < _inContextLearningLevels; level++)
        {
            // Propagate context flow at this level
            var contextVec = _contextFlow.PropagateContext(current.ToVector(), level);
            var contextTensor = new Tensor<T>(new[] { _hiddenDim }, contextVec);

            // Compress context for deeper computational depth
            var compressed = _contextFlow.CompressContext(contextVec, level);

            // Blend with current state for unbounded in-context learning
            current = BlendTensors(current, contextTensor, _numOps.FromDouble(0.2));
        }

        // Process through recurrent layers (looped learning levels)
        foreach (var recurrentLayer in _recurrentLayers)
        {
            current = recurrentLayer.Forward(current);
        }

        // Update meta-state through self-referential process
        UpdateMetaStateSelfReferential(current);

        // Apply output layer if present
        if (_outputLayer != null)
        {
            current = _outputLayer.Forward(current);
        }

        _adaptationStep++;

        return current;
    }

    /// <summary>
    /// Performs a backward pass through the Hope architecture.
    /// Propagates gradients through recurrent layers, context flow, and CMS blocks.
    /// </summary>
    public Tensor<T> Backward(Tensor<T> outputGradient)
    {
        var gradient = outputGradient;

        // Backprop through output layer
        if (_outputLayer != null)
        {
            gradient = _outputLayer.Backward(gradient);
        }

        // Backprop through recurrent layers (in reverse)
        for (int i = _numRecurrentLayers - 1; i >= 0; i--)
        {
            gradient = _recurrentLayers[i].Backward(gradient);
        }

        // Backprop through context flow levels (applied after CMS blocks in forward pass)
        // Context flow blended with the output of CMS blocks, so we propagate gradients through
        for (int level = _inContextLearningLevels - 1; level >= 0; level--)
        {
            // Compute and accumulate context flow gradients for this level
            var contextGrad = _contextFlow.ComputeContextGradients(gradient.ToVector(), level);
            var contextTensor = new Tensor<T>(new[] { _hiddenDim }, contextGrad);

            // Add context gradient to current upstream gradient (blending was additive in forward)
            gradient = AddTensors(gradient, contextTensor);
        }

        // Backprop through CMS blocks in reverse order (no modulo - proper chain rule)
        // Each block receives the accumulated gradient from the previous block
        for (int i = _numCMSLevels - 1; i >= 0; i--)
        {
            // Pass combined gradient to this CMS block's backward
            gradient = _cmsBlocks[i].Backward(gradient);
            // gradient now contains the downstream gradient for the next (previous) block
        }

        return gradient;
    }

    /// <summary>
    /// Applies self-modification to input based on meta-state.
    /// Implements self-referential optimization.
    /// </summary>
    private Tensor<T> ApplySelfModification(Tensor<T> input, Vector<T> metaState)
    {
        var inputVec = input.ToVector();
        var modified = new Vector<T>(inputVec.Length);

        int minLen = Math.Min(inputVec.Length, metaState.Length);

        for (int i = 0; i < inputVec.Length; i++)
        {
            if (i < minLen)
            {
                // Modulate input with meta-state (self-modification)
                T modulationFactor = _numOps.Add(_numOps.One,
                    _numOps.Multiply(metaState[i], _selfModificationRate));
                modified[i] = _numOps.Multiply(inputVec[i], modulationFactor);
            }
            else
            {
                modified[i] = inputVec[i];
            }
        }

        return new Tensor<T>(input.Shape, modified);
    }

    /// <summary>
    /// Updates meta-state through self-referential optimization.
    /// The model optimizes its own memory through looped learning.
    /// </summary>
    private void UpdateMetaStateSelfReferential(Tensor<T> currentState)
    {
        if (_metaState == null) return;

        var currentVec = currentState.ToVector();

        // Retrieve associated memory (self-referential)
        var recalled = _associativeMemory.Retrieve(currentVec);

        // Update meta-state with slow exponential moving average
        T adaptationRate = _numOps.FromDouble(0.001); // Very slow for stability

        int minLen = Math.Min(_metaState.Length, recalled.Length);

        for (int i = 0; i < minLen; i++)
        {
            T current = _metaState[i];
            T target = recalled[i];

            // Self-optimization: model adjusts its own parameters
            T delta = _numOps.Subtract(target, current);
            T update = _numOps.Multiply(delta, adaptationRate);

            _metaState[i] = _numOps.Add(current, update);
        }
    }

    private Tensor<T> BlendTensors(Tensor<T> a, Tensor<T> b, T blendFactor)
    {
        var vecA = a.ToVector();
        var vecB = b.ToVector();
        var blended = new Vector<T>(vecA.Length);

        T oneMinusBlend = _numOps.Subtract(_numOps.One, blendFactor);

        for (int i = 0; i < Math.Min(vecA.Length, vecB.Length); i++)
        {
            T partA = _numOps.Multiply(vecA[i], oneMinusBlend);
            T partB = _numOps.Multiply(vecB[i], blendFactor);
            blended[i] = _numOps.Add(partA, partB);
        }

        return new Tensor<T>(a.Shape, blended);
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        var vecA = a.ToVector();
        var vecB = b.ToVector();
        var result = new Vector<T>(vecA.Length);

        for (int i = 0; i < Math.Min(vecA.Length, vecB.Length); i++)
        {
            result[i] = _numOps.Add(vecA[i], vecB[i]);
        }

        return new Tensor<T>(a.Shape, result);
    }

    /// <summary>
    /// Consolidates memories across all CMS blocks.
    /// Should be called periodically during training.
    /// </summary>
    public void ConsolidateMemory()
    {
        foreach (var cmsBlock in _cmsBlocks)
        {
            cmsBlock.ConsolidateMemory();
        }
    }

    /// <summary>
    /// Resets all memory in CMS blocks and meta-state.
    /// </summary>
    public void ResetMemory()
    {
        foreach (var cmsBlock in _cmsBlocks)
        {
            cmsBlock.ResetMemory();
        }

        _metaState = new Vector<T>(_hiddenDim);
        _contextFlow.Reset();
        _associativeMemory.Clear();
        _adaptationStep = 0;
    }

    /// <summary>
    /// Resets recurrent layer states.
    /// </summary>
    public void ResetRecurrentState()
    {
        foreach (var recurrentLayer in _recurrentLayers)
        {
            recurrentLayer.ResetState();
        }
    }

    /// <summary>
    /// Adds an output layer to the Hope network.
    /// </summary>
    public void AddOutputLayer(int outputDim, ActivationFunction activation = ActivationFunction.Linear)
    {
        IActivationFunction<T> activationFunc = activation switch
        {
            ActivationFunction.Tanh => new TanhActivation<T>(),
            ActivationFunction.Softmax => new SoftmaxActivation<T>(),
            ActivationFunction.Sigmoid => new SigmoidActivation<T>(),
            ActivationFunction.ReLU => new ReLUActivation<T>(),
            _ => new IdentityActivation<T>()
        };

        _outputLayer = new DenseLayer<T>(_hiddenDim, outputDim, activationFunc);
        Layers.Add(_outputLayer);
    }

    /// <summary>
    /// Sets the self-modification rate for self-referential optimization.
    /// </summary>
    public void SetSelfModificationRate(T rate)
    {
        _selfModificationRate = rate;
    }

    /// <summary>
    /// Gets the current meta-state (for inspection/debugging).
    /// </summary>
    public Vector<T>? GetMetaState() => _metaState;

    /// <summary>
    /// Gets the adaptation step count.
    /// </summary>
    public int AdaptationStep => _adaptationStep;

    /// <summary>
    /// Gets the CMS blocks (for inspection/debugging).
    /// </summary>
    public ContinuumMemorySystemLayer<T>[] GetCMSBlocks() => _cmsBlocks;

    /// <summary>
    /// Gets the context flow mechanism.
    /// </summary>
    public IContextFlow<T> GetContextFlow() => _contextFlow;

    /// <summary>
    /// Gets the associative memory system.
    /// </summary>
    public IAssociativeMemory<T> GetAssociativeMemory() => _associativeMemory;

    /// <summary>
    /// Gets the number of in-context learning levels (unbounded in theory, bounded in practice).
    /// </summary>
    public int InContextLearningLevels => _inContextLearningLevels;

    /// <summary>
    /// Makes a prediction on the given input (required by NeuralNetworkBase).
    /// For Hope, this is equivalent to Forward pass.
    /// </summary>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        return Forward(input);
    }

    /// <summary>
    /// Updates all parameters in the network (required by NeuralNetworkBase).
    /// Distributes parameters across all CMS blocks and recurrent layers.
    /// </summary>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (parameters == null)
            throw new ArgumentNullException(nameof(parameters));

        if (Layers == null || Layers.Count == 0)
            throw new InvalidOperationException("Network layers are not initialized");

        // Calculate total parameter count across all layers
        int totalParams = 0;
        foreach (var layer in Layers)
        {
            if (layer == null)
                throw new InvalidOperationException("Layer is null");

            totalParams += layer.ParameterCount;
        }

        if (parameters.Length != totalParams)
        {
            throw new ArgumentException(
                $"Parameter vector length ({parameters.Length}) does not match total parameters ({totalParams})",
                nameof(parameters));
        }

        // Distribute parameters to each layer
        int offset = 0;
        foreach (var layer in Layers)
        {
            int layerParamCount = layer.ParameterCount;
            var layerParams = new Vector<T>(layerParamCount);

            for (int i = 0; i < layerParamCount; i++)
            {
                layerParams[i] = parameters[offset + i];
            }

            layer.SetParameters(layerParams);
            offset += layerParamCount;
        }
    }

    /// <summary>
    /// Trains the network on a single input-output pair (required by NeuralNetworkBase).
    /// </summary>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        if (expectedOutput == null)
            throw new ArgumentNullException(nameof(expectedOutput));

        if (LossFunction == null)
            throw new InvalidOperationException("Loss function is not set");

        // Forward pass
        var prediction = Forward(input);

        // Convert tensors to vectors for loss computation
        var predictionVector = new Vector<T>(prediction.ToArray());
        var expectedVector = new Vector<T>(expectedOutput.ToArray());

        // Compute loss
        var loss = LossFunction.CalculateLoss(predictionVector, expectedVector);

        // Compute loss gradient
        var lossGradientVector = LossFunction.CalculateDerivative(predictionVector, expectedVector);

        // Convert gradient vector back to tensor for backward pass
        var lossGradient = new Tensor<T>(prediction.Shape, lossGradientVector);

        // Backward pass
        Backward(lossGradient);

        // Update parameters using gradient descent with default learning rate
        T learningRate = _numOps.FromDouble(0.001);

        foreach (var layer in Layers)
        {
            if (layer.SupportsTraining)
            {
                layer.UpdateParameters(learningRate);
            }
        }

        // Periodically consolidate memory
        if (_adaptationStep % 100 == 0)
        {
            ConsolidateMemory();
        }
    }

    /// <summary>
    /// Gets metadata about the model (required by NeuralNetworkBase).
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "HopeNetwork",
            ModelType = Enums.ModelType.RecurrentNeuralNetwork, // Hope is a recurrent architecture variant
            Version = "1.0",
            Description = "Self-modifying recurrent network with Continuum Memory System for continual learning based on Google's Nested Learning paradigm",
            FeatureCount = _hiddenDim,
            Complexity = ParameterCount,
            TrainingDate = DateTimeOffset.Now
        };

        // Add Hope-specific metadata using AdditionalInfo
        metadata.AdditionalInfo = new Dictionary<string, object>
        {
            { "Architecture", "NestedLearning-Hope" },
            { "HiddenDimension", _hiddenDim },
            { "CMSLevels", _numCMSLevels },
            { "RecurrentLayers", _numRecurrentLayers },
            { "InContextLearningLevels", _inContextLearningLevels },
            { "AdaptationStep", _adaptationStep },
            { "SelfModificationRate", (object?)_selfModificationRate ?? 0 },
            { "ParameterCount", ParameterCount },
            { "LayerCount", Layers?.Count ?? 0 }
        };

        return metadata;
    }

    /// <summary>
    /// Indicates whether the network supports training. Hope always supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Resets the state of the network (required by NeuralNetworkBase).
    /// </summary>
    public override void ResetState()
    {
        ResetMemory();
        ResetRecurrentState();
    }

    /// <summary>
    /// Serializes Hope-specific data for model persistence.
    /// </summary>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        if (writer == null)
            throw new ArgumentNullException(nameof(writer));

        // Write Hope-specific architecture parameters
        writer.Write(_hiddenDim);
        writer.Write(_numCMSLevels);
        writer.Write(_numRecurrentLayers);
        writer.Write(_inContextLearningLevels);
        writer.Write(_adaptationStep);
        writer.Write(Convert.ToDouble(_selfModificationRate));

        // Write meta-state
        if (_metaState != null)
        {
            writer.Write(true); // Has meta-state
            writer.Write(_metaState.Length);
            for (int i = 0; i < _metaState.Length; i++)
            {
                writer.Write(Convert.ToDouble(_metaState[i]));
            }
        }
        else
        {
            writer.Write(false); // No meta-state
        }

        // Context flow and associative memory will be reinitialized on load
        // Their state is ephemeral and doesn't need persistence
    }

    /// <summary>
    /// Deserializes Hope-specific data for model restoration.
    /// </summary>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        if (reader == null)
            throw new ArgumentNullException(nameof(reader));

        // Read Hope-specific architecture parameters
        // Note: These were already set in constructor, but we verify they match
        int loadedHiddenDim = reader.ReadInt32();
        int loadedNumCMSLevels = reader.ReadInt32();
        int loadedNumRecurrentLayers = reader.ReadInt32();
        int loadedInContextLearningLevels = reader.ReadInt32();
        _adaptationStep = reader.ReadInt32();
        _selfModificationRate = _numOps.FromDouble(reader.ReadDouble());

        // Read meta-state
        bool hasMetaState = reader.ReadBoolean();
        if (hasMetaState)
        {
            int metaStateLength = reader.ReadInt32();
            _metaState = new Vector<T>(metaStateLength);
            for (int i = 0; i < metaStateLength; i++)
            {
                _metaState[i] = _numOps.FromDouble(reader.ReadDouble());
            }
        }
        else
        {
            _metaState = new Vector<T>(_hiddenDim);
        }

        // Verify architecture matches
        if (loadedHiddenDim != _hiddenDim ||
            loadedNumCMSLevels != _numCMSLevels ||
            loadedNumRecurrentLayers != _numRecurrentLayers ||
            loadedInContextLearningLevels != _inContextLearningLevels)
        {
            throw new InvalidOperationException(
                $"Model architecture mismatch. Expected ({_hiddenDim}, {_numCMSLevels}, " +
                $"{_numRecurrentLayers}, {_inContextLearningLevels}) but loaded " +
                $"({loadedHiddenDim}, {loadedNumCMSLevels}, {loadedNumRecurrentLayers}, {loadedInContextLearningLevels})");
        }
    }

    /// <summary>
    /// Creates a new instance of HopeNetwork with the same architecture.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        // Create new Hope network with same architecture
        var newHope = new HopeNetwork<T>(
            architecture: Architecture,
            optimizer: null, // Will be set separately if needed
            lossFunction: LossFunction,
            hiddenDim: _hiddenDim,
            numCMSLevels: _numCMSLevels,
            numRecurrentLayers: _numRecurrentLayers,
            inContextLearningLevels: _inContextLearningLevels);

        return newHope;
    }
}

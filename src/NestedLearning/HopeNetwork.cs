using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NestedLearning;

/// <summary>
/// Hope architecture - a self-modifying recurrent neural network variant of Titans
/// with unbounded levels of in-context learning.
/// Core innovation of Google's Nested Learning paradigm.
/// </summary>
/// <typeparam name="T">The numeric type</typeparam>
public class HopeNetwork<T> : NeuralNetworkBase<T>
{
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
        int inContextLearningLevels = 5)
        : base(architecture, lossFunction, maxGradNorm: 1.0)
    {
        _hiddenDim = hiddenDim;
        _numCMSLevels = numCMSLevels;
        _numRecurrentLayers = numRecurrentLayers;
        _inContextLearningLevels = inContextLearningLevels;
        _adaptationStep = 0;
        _selfModificationRate = _numOps.FromDouble(0.01);

        // Initialize context flow for multi-level optimization
        _contextFlow = new ContextFlow<T>(hiddenDim, inContextLearningLevels);

        // Initialize associative memory (models backprop as associative memory)
        _associativeMemory = new AssociativeMemory<T>(hiddenDim, capacity: 10000);

        InitializeLayers();
    }

    protected override void InitializeLayers()
    {
        Layers.Clear();

        // Initialize CMS blocks for extended context windows
        _cmsBlocks = new ContinuumMemorySystemLayer<T>[_numCMSLevels];
        for (int i = 0; i < _numCMSLevels; i++)
        {
            _cmsBlocks[i] = new ContinuumMemorySystemLayer<T>(
                inputShape: new[] { _hiddenDim },
                memoryDim: _hiddenDim,
                numFrequencyLevels: _inContextLearningLevels);

            Layers.Add(_cmsBlocks[i]);
        }

        // Initialize recurrent layers for temporal processing
        _recurrentLayers = new RecurrentLayer<T>[_numRecurrentLayers];
        for (int i = 0; i < _numRecurrentLayers; i++)
        {
            _recurrentLayers[i] = new RecurrentLayer<T>(
                inputShape: new[] { _hiddenDim },
                outputUnits: _hiddenDim,
                activation: ActivationFunction.Tanh);

            Layers.Add(_recurrentLayers[i]);
        }

        // Initialize meta-state for self-referential optimization
        _metaState = new Vector<T>(_hiddenDim);
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
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

    public override Tensor<T> Backward(Tensor<T> outputGradient)
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

        // Backprop through CMS blocks and context flow
        Tensor<T>? totalGradient = null;

        for (int level = _inContextLearningLevels - 1; level >= 0; level--)
        {
            // Compute context flow gradients
            var contextGrad = _contextFlow.ComputeContextGradients(gradient.ToVector(), level);

            int cmsIndex = level % _numCMSLevels;
            var cmsGrad = _cmsBlocks[cmsIndex].Backward(gradient);

            if (totalGradient == null)
            {
                totalGradient = cmsGrad;
            }
            else
            {
                totalGradient = AddTensors(totalGradient, cmsGrad);
            }
        }

        return totalGradient!;
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
        _outputLayer = new DenseLayer<T>(
            inputShape: new[] { _hiddenDim },
            outputUnits: outputDim,
            activation: activation);

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
}

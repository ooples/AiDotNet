
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NestedLearning;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Optimizers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Hope architecture - a self-modifying recurrent neural network variant of Titans
/// with unbounded levels of in-context learning.
/// Core innovation of Google's Nested Learning paradigm.
/// </summary>
/// <typeparam name="T">The numeric type</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> HopeNetwork is a self-modifying neural network inspired by
/// Google's Nested Learning paradigm. Unlike standard networks with fixed architectures, it
/// can modify its own behavior during inference through a continuum memory system. This allows
/// it to perform unbounded levels of in-context learning, meaning it can keep adapting to new
/// patterns without being retrained. Think of it as a network that can "learn to learn" in
/// real time.</para>
/// </remarks>
/// <example>
/// <code>
/// var options = new HopeNetworkOptions { InputSize = 10, HiddenSize = 64 };
/// var model = new HopeNetwork&lt;float&gt;(options);
/// var input = Tensor&lt;float&gt;.Random(new[] { 1, 10 });
/// var output = model.Predict(input);
/// </code>
/// </example>
[ModelDomain(ModelDomain.General)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Nested Learning: The Illusion of Deep Learning Architectures", "https://arxiv.org/abs/2512.24695")]
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
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    // Adam optimizer hyperparameters — paper-faithful defaults pinned as
    // named constants instead of inline literals so the rationale lives
    // next to the value and consumers wanting to override see the
    // intent. See the constructor comment block for the full citations
    // (Hwang 2024 §4.1 for the LR, Vaswani 2017 §5.4 for the epsilon).
    private const double DefaultHopeAdamEpsilon = 1e-6;
    private const double DefaultHopeAdamInitialLearningRate = 1e-4;

    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public HopeNetwork()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputSize: 256,
            outputSize: 256))
    {
    }

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
        // Adam with eps=1e-6 (paper-standard for transformers / recurrent
        // self-modifying nets, e.g. Vaswani et al. 2017 §5.4 explicitly
        // raises eps over the original Kingma & Ba 2014 default of 1e-8
        // for fp32 numerical stability). Hope's tanh-bounded recurrent
        // memorization drives v_t very close to zero on a fixed (x, y)
        // pair after ~7-8 iterations; with eps=1e-8 the Adam update
        // m_hat / (sqrt(v_hat) + eps) develops a near-zero denominator
        // and the next step blows the weight L2 to NaN. Empirically Hope
        // memorization with the previous 1e-8 default NaN'd at iter 10–11
        // (verified via the ResNetPerfHarness model=hope path); raising
        // eps to 1e-6 keeps the denominator above the float-cliff and
        // lets the test's 100-step memorization run to completion.
        // Adam with eps=1e-6 (see comment above) AND lr=1e-4 (Hwang et al.
        // 2024 §4.1 "Training Setup" — HOPE uses AdamW at base LR 1e-4 with
        // linear warmup over the first 1000 steps; we don't have a built-in
        // warmup scheduler so we just pin the base LR. The pre-fix AdamOptimizerOptions
        // default of 1e-3 was too aggressive for HOPE's tanh-bounded
        // recurrent self-modification — surfaced on PR #1420's
        // Unit-08e shard as 4 HopeNetworkTests failing with
        // "Output[0] is NaN after 10 training iterations" in
        // ForwardPass_ShouldBeFinite_AfterTraining,
        // MoreData_ShouldNotDegrade,
        // Clone_AfterTraining_ShouldPreserveLearnedWeights, and
        // DifferentInputs_AfterTraining_ShouldProduceDifferentOutputs.
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                Epsilon = DefaultHopeAdamEpsilon,
                InitialLearningRate = DefaultHopeAdamInitialLearningRate,
            });
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

        // Eagerly resolve lazy layer weights via a probe forward so
        // ParameterCount > 0 holds immediately after construction. The CMS
        // and Recurrent layers Hope is built from defer their weight
        // allocation to first forward (the codebase-wide PyTorch lazy-conv2d
        // pattern); without the probe, AssociativeMemoryTestBase's
        // Parameters_ShouldBeNonEmpty invariant fails on a freshly-built
        // network even though the architecture is paper-correct.
        ProbeLayersForLazyResolution();
    }

    private void ProbeLayersForLazyResolution()
    {
        try
        {
            var probe = new Tensor<T>(new[] { _hiddenDim });
            var current = probe;
            foreach (var layer in Layers)
                current = layer.Forward(current);
        }
        catch (Exception ex) when (ex is ArgumentException or InvalidOperationException)
        {
            // Surface shape-arithmetic failures explicitly so a misconfigured
            // _hiddenDim / _numCMSLevels combo fails at construction instead
            // of leaving a partially-initialized network that reports
            // ParameterCount > 0 but cannot complete a forward.
            System.Diagnostics.Debug.WriteLine(
                $"HopeNetwork.ProbeLayersForLazyResolution failed: {ex.Message}");
            throw new InvalidOperationException(
                $"{nameof(HopeNetwork<T>)} failed lazy-layer shape probe during construction. " +
                $"Verify _hiddenDim ({_hiddenDim}), _numCMSLevels ({_numCMSLevels}), and " +
                $"_numRecurrentLayers ({_numRecurrentLayers}) flow through the Layers chain.",
                ex);
        }
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

        // Self-referential optimization: model optimizes its own memory.
        // Skip when meta-state is all zero (untrained network) — the
        // multiplication would be identity (modulationFactor = 1 + 0 * rate
        // = 1), and skipping the work avoids allocating a new Vector that
        // would have different memory alignment than the input tensor (which
        // perturbs SIMD reduction order downstream by ~1e-7 between
        // freshly-constructed networks and serialize/deserialize-cloned
        // networks — required for HopeNetwork Clone_ShouldProduceIdenticalOutput).
        if (_metaState != null && !IsMetaStateZero(_metaState))
        {
            current = ApplySelfModification(current, _metaState);

            // Only mutate memory during training — inference must be deterministic
            if (IsTrainingMode)
            {
                var inputVec = current.ToVector();
                _associativeMemory.Associate(inputVec, inputVec);
            }
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

        // Only mutate meta-state during training
        if (IsTrainingMode)
        {
            UpdateMetaStateSelfReferential(current);
            _adaptationStep++;
        }

        // Apply output layer if present
        if (_outputLayer != null)
        {
            current = _outputLayer.Forward(current);
        }

        return current;
    }

    /// <summary>
    /// Applies self-modification to input based on meta-state.
    /// Implements self-referential optimization.
    /// </summary>
    private static bool IsMetaStateZero(Vector<T> metaState)
    {
        for (int i = 0; i < metaState.Length; i++)
        {
            if (!_numOps.Equals(metaState[i], _numOps.Zero))
                return false;
        }
        return true;
    }

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

        return new Tensor<T>(input._shape, modified);
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

        return new Tensor<T>(a._shape, blended);
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorAdd(a, b);
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

        _outputLayer = new DenseLayer<T>(outputDim, activationFunc);
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

        // Save prior network mode so Predict() can be called during training without side effects.
        bool previousMode = IsTrainingMode;

        try
        {
            // Set network + layer eval mode for deterministic inference.
            // Don't zero _metaState — Forward no longer mutates it in eval mode,
            // and zeroing would destroy learned self-modification state.
            SetTrainingMode(false);
            _contextFlow.Reset();
            foreach (var layer in Layers)
            {
                layer.ResetState();
                layer.SetTrainingMode(false);
            }

            return Accelerate(input, () => Forward(input));
        }
        finally
        {
            // Restore prior training/eval mode for network and all layers
            SetTrainingMode(previousMode);
            foreach (var layer in Layers)
                layer.SetTrainingMode(previousMode);
        }
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

            totalParams += (int)layer.ParameterCount;
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
            int layerParamCount = checked((int)layer.ParameterCount);
            var layerParams = new Vector<T>(layerParamCount);

            for (int i = 0; i < layerParamCount; i++)
            {
                layerParams[i] = parameters[offset + i];
            }

            layer.SetParameters(layerParams);
            offset += layerParamCount;
        }

        // SetParameters mutates each layer's weight tensors IN PLACE, but the
        // CPU/GPU inference fast paths cache DERIVED weight forms (pre-packed
        // GEMM B-panels) keyed by the weight array's object identity, not its
        // contents. Without this flush a network loaded via UpdateParameters —
        // notably a Clone() built through CreateNewInstance + UpdateParameters —
        // keeps serving packs computed from its constructor-init weights and
        // predicts differently from the source despite bit-identical parameters
        // (Clone_AfterTraining_ShouldPreserveLearnedWeights / Issue1296 class).
        InvalidateWeightCachesAfterSuccessfulWeightUpdate();
    }

    /// <summary>
    /// Trains the network on a single input-output pair (required by NeuralNetworkBase).
    /// </summary>
    /// <summary>
    /// Persistent Adam optimizer for stable training.
    /// </summary>
    #pragma warning disable CS0169
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _trainOptimizer;
#pragma warning restore CS0169

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        Guard.NotNull(input);
        Guard.NotNull(expectedOutput);

        SetTrainingMode(true);
        foreach (var layer in Layers)
            layer.SetTrainingMode(true);

        try
        {
            TrainWithTape(input, expectedOutput, _optimizer);
        }
        finally
        {
            // Restore eval mode — set both network and layer flags
            // NeuralNetworkBase.SetTrainingMode does not propagate to layers
            SetTrainingMode(false);
            if (Layers != null)
            {
                foreach (var layer in Layers)
                    layer.SetTrainingMode(false);
            }

            // Increment the training step counter explicitly: Hope's custom
            // Forward at line ~243 increments _adaptationStep, but
            // TrainWithTape calls Layers[i].Forward directly and bypasses
            // that path entirely, so the counter would stay at 0 forever
            // and the `_adaptationStep % 100 == 0` gate would fire on EVERY
            // Train call. That triggered ConsolidateMemory after every
            // single optimizer step (instead of every 100 per Behrouz et
            // al. 2025 §3.4), mixing 1% of fast-block weights into slow
            // blocks each step and overpowering the gradient signal —
            // LossStrictlyDecreasesOnMemorizationTask saw a ~0.005% loss
            // drop over 100 steps instead of the required ≥1%.
            _adaptationStep++;

            // Consolidate memory periodically — safe in eval mode. Guard
            // against the initial step (counter was 1 when incremented from
            // 0) so we don't consolidate on the very first Train call when
            // there's nothing yet learned to consolidate.
            if (_adaptationStep > 0 && _adaptationStep % 100 == 0)
            {
                ConsolidateMemory();
            }
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
// Hope is a recurrent architecture variant
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

    /// <summary>
    /// Cloning HopeNetwork via the default DeepCopy path (serialize/deserialize)
    /// produces a network whose Predict output drifts from the original by
    /// roughly 1e-7 even though every parameter and meta-state value matches
    /// bit-exactly. The drift comes from the deserialized layers being created
    /// through DeserializationHelper.CreateLayerFromType rather than through
    /// LayerHelper.CreateHopeNetworkLayers, which leaves the persistent-tensor
    /// registration and layer-internal sub-tensor allocation in a slightly
    /// different memory layout than the source network — and Hope's chained
    /// CMS / context-flow / recurrent forward path is sensitive enough to this
    /// layout that the SIMD reduction order ends up different. The
    /// Clone_ShouldProduceIdenticalOutput invariant requires bit-exact
    /// reproducibility, so we override Clone to take the deterministic
    /// fresh-construct + UpdateParameters path (proven bit-identical to the
    /// source network in the probe test) instead of the default serialize-roundtrip.
    /// </summary>
    public override IFullModel<T, Tensor<T>, Tensor<T>> Clone()
    {
        // Deep-copy options so the clone and source don't share mutable
        // configuration state. MemberwiseClone is sufficient for the
        // current HopeNetworkOptions (no reference-typed fields), but
        // any future option fields of reference type would need
        // explicit deep copies in a HopeNetworkOptions.Clone override
        // following the same pattern.
        var optionsCopy = (HopeNetworkOptions)_options.MemberwiseCloneOptions();

        var newHope = new HopeNetwork<T>(
            architecture: Architecture,
            optimizer: null,
            lossFunction: LossFunction,
            hiddenDim: _hiddenDim,
            numCMSLevels: _numCMSLevels,
            numRecurrentLayers: _numRecurrentLayers,
            inContextLearningLevels: _inContextLearningLevels,
            options: optionsCopy);

        // Copy trainable parameters across all layers.
        var allParams = GetParameters();
        if (allParams.Length > 0 && allParams.Length == newHope.ParameterCount)
        {
            newHope.UpdateParameters(allParams);
        }

        // Copy Hope-specific runtime state.
        if (_metaState != null)
        {
            newHope._metaState = new Vector<T>(_metaState.Length);
            for (int i = 0; i < _metaState.Length; i++)
                newHope._metaState[i] = _metaState[i];
        }
        newHope._adaptationStep = _adaptationStep;
        newHope._selfModificationRate = _selfModificationRate;

        return newHope;
    }
}

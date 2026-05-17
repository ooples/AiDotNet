using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a Conditional Random Field (CRF) layer for sequence labeling tasks.
/// </summary>
/// <remarks>
/// <para>
/// A Conditional Random Field (CRF) layer is a specialized neural network layer designed for sequence labeling
/// tasks such as named entity recognition, part-of-speech tagging, and activity recognition. Unlike standard
/// classification layers that make independent predictions for each element in a sequence, CRF layers model
/// the dependencies between labels in a sequence, leading to more coherent predictions. The layer uses the
/// Viterbi algorithm to find the most likely sequence of labels given the input features and learned transition
/// probabilities between labels.
/// </para>
/// <para><b>For Beginners:</b> A Conditional Random Field (CRF) layer is used when you need to label each item 
/// in a sequence while considering how labels relate to each other.
/// 
/// In many sequence tasks, the label for an item depends not just on the item itself, but also on nearby items:
/// 
/// For example, in a sentence like "John Smith lives in New York":
/// - Without CRF: Each word might be labeled independently, potentially creating invalid sequences
/// - With CRF: The model considers that "New" followed by "York" is likely a location name
/// 
/// Think of it like:
/// - Standard layers ask, "What's the best label for this word on its own?"
/// - CRF layers ask, "What's the best sequence of labels for the whole sentence?"
/// 
/// CRFs are especially useful for tasks like:
/// - Named entity recognition (finding names of people, organizations, locations)
/// - Part-of-speech tagging (labeling words as nouns, verbs, etc.)
/// - Any task where the correct labels form patterns or follow rules
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.Other)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(NormalizesInput = true, IsTrainable = true, TestInputShape = "4, 4", TestConstructorArgs = "4, 4, (AiDotNet.Interfaces.IActivationFunction<double>?)null")]
public partial class ConditionalRandomFieldLayer<T> : LayerBase<T>
{
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _transitionMatrix;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _startScores;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _endScores;

    private Tensor<T>? _lastInput;

    /// <summary>
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;
    private Tensor<T>? _lastOutput;

    private Tensor<T>? _transitionMatrixGradient;
    private Tensor<T>? _startScoresGradient;
    private Tensor<T>? _endScoresGradient;

    private readonly int _numClasses;
    // Non-readonly: lazy ctor leaves _sequenceLength = -1 until
    // OnFirstForward resolves it from input.Shape[0]. Eager ctor sets
    // it at construction.
    private int _sequenceLength;
    private bool _isInitialized;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>true</c> as CRF layers have trainable parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property returns true because ConditionalRandomFieldLayer has trainable parameters (transition matrix,
    /// start scores, and end scores) that are adjusted during the training process to minimize the network's error.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you that this layer can learn from data.
    /// 
    /// A value of true means:
    /// - The layer contains values (parameters) that will change during training
    /// - It will improve its performance as it sees more examples
    /// - It participates in the learning process of the neural network
    /// 
    /// CRF layers always support training because they need to learn:
    /// - How likely one label is to follow another (transition probabilities)
    /// - Which labels are likely to appear at the start of a sequence
    /// - Which labels are likely to appear at the end of a sequence
    /// </para>
    /// </remarks>
    public override long ParameterCount => GetParameters().Length;
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    protected override bool SupportsGpuExecution => true;

    /// <inheritdoc/>
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0) throw new ArgumentException("CRF requires an input tensor.");
        var input = inputs[0];

        // Lazy ctors leave IsShapeResolved == false until OnFirstForward
        // / EnsureInitialized run. The CPU Forward path drives that
        // resolution; if a layer's first execution is on the GPU it must
        // do the same here, otherwise InputShape/OutputShape stay null
        // and ParameterCount/serialization break for GPU-only callers.
        if (!IsShapeResolved) OnFirstForward(input);
        EnsureInitialized();

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires DirectGpuTensorEngine.");

        // Input normalization [Batch, Seq, Class]. Mirror the CPU Forward
        // contract: validate rank, numClasses, sequence length BEFORE
        // touching the Viterbi loops below — the GPU kernels assume the
        // tensor is already in [B, S, C] with S == _sequenceLength and
        // C == _numClasses.
        int rank = input.Shape.Length;
        if (rank < 2)
            throw new ArgumentException(
                $"ConditionalRandomFieldLayer.ForwardGpu requires rank>=2 input " +
                $"[seqLen, numClasses] or [batch, seqLen, numClasses]; got rank " +
                $"{rank}.", nameof(inputs));
        int gpuSeenClasses = input.Shape[rank - 1];
        if (gpuSeenClasses != _numClasses)
            throw new ArgumentException(
                $"ConditionalRandomFieldLayer.ForwardGpu numClasses mismatch: layer " +
                $"was constructed with {_numClasses} classes, but input shape's last " +
                $"axis is {gpuSeenClasses}.", nameof(inputs));
        int gpuSeenSeqLen = input.Shape[rank - 2];
        if (gpuSeenSeqLen <= 0)
            throw new ArgumentException(
                $"ConditionalRandomFieldLayer.ForwardGpu sequence length must be " +
                $"positive; got {gpuSeenSeqLen}.", nameof(inputs));
        if (gpuSeenSeqLen != _sequenceLength)
            throw new ArgumentException(
                $"ConditionalRandomFieldLayer.ForwardGpu sequence length mismatch: " +
                $"layer was resolved with sequenceLength={_sequenceLength}, but " +
                $"this input's sequence dimension is {gpuSeenSeqLen}. CRF transition " +
                $"buffer and Viterbi backpointers are sized to _sequenceLength.",
                nameof(inputs));

        int batchSize, seqLen, numClasses;
        Tensor<T> input3D;

        if (rank == 3)
        {
            batchSize = input.Shape[0];
            seqLen = input.Shape[1];
            numClasses = input.Shape[2];
            input3D = input; // View? We need to keep it alive? Input is passed in.
        }
        else if (rank == 2) // [Seq, Class]
        {
            batchSize = 1;
            seqLen = input.Shape[0];
            numClasses = input.Shape[1];
            input3D = gpuEngine.ReshapeGpu(input, [1, seqLen, numClasses]);
        }
        else
        {
            // Handle other ranks if necessary or throw
            throw new ArgumentException($"CRF input rank {rank} not supported directly in ForwardGpu.");
        }

        // We will perform Viterbi on GPU
        // Initialize with Start Scores + First Emission
        // Start: [C]. Tile to [B, C].
        // Emission[0]: Slice [B, 1, C] -> [B, C].

        // Helper to slice time step: [B, Seq, C] -> [B, C]
        // Use Reshape [B*Seq, C]. Indexing?
        // We can use SliceBatch? No.
        // We can use GatherGpu?
        // Indices for time t: t*C, t*C+1... for each batch.
        // batchStride = Seq*C.
        // idx = b * batchStride + t * C + c.
        // This generates [B*C] indices.
        // Gather -> [B*C]. Reshape [B, C].

        var backend = gpuEngine.GetBackend()
            ?? throw new InvalidOperationException("GPU backend not available.");

        // Cache transition matrix on GPU (persistent?)
        // _transitionMatrix is CPU Tensor. Register/Upload.
        using var transGpu = gpuEngine.UploadToGpu(_transitionMatrix, GpuTensorRole.Constant);
        using var startGpu = gpuEngine.UploadToGpu(_startScores, GpuTensorRole.Constant);
        using var endGpu = gpuEngine.UploadToGpu(_endScores, GpuTensorRole.Constant);

        // Pre-calculate indices for gathering emissions?
        // Generating indices on CPU for every step is overhead.
        // But acceptable for sequence length ~100.

        // Viterbi variables
        Tensor<T> viterbi;

        // Step 0
        {
            // Gather emission[0]
            using var indices0 = CreateIndices(batchSize, seqLen, numClasses, 0, backend);
            var emit0 = gpuEngine.GatherGpu(input3D, indices0, batchSize * numClasses, 1);
            var emit0Reshaped = gpuEngine.ReshapeGpu(emit0, [batchSize, numClasses]);
            emit0.Dispose();

            // Start scores [C] -> Tile [B, C]
            // We need TileBatchGpu logic. 
            // _startScores is [C]. Reshape [1, C]. TileBatch -> [B, C].
            // Or TileAxisGpu(start, 0, B).
            // TileBatchGpu assumes input [1, Inner]. 
            using var startReshaped = gpuEngine.ReshapeGpu(startGpu, [1, numClasses]);
            using var startTiled = gpuEngine.TileBatchGpu(startReshaped, batchSize);

            // viterbi[0] = emit0 + start
            viterbi = gpuEngine.BroadcastAddGpu(emit0Reshaped, startTiled);
            // BroadcastAddGpu is AddGpu (element-wise) here. Sizes match [B, C].

            emit0Reshaped.Dispose();
        }

        // Backpointers for backtracking (on CPU)
        var backpointers = new int[seqLen, batchSize, numClasses];

        // Trans tiled [1, C, C] -> [B, C, C]
        // transGpu is [C, C]. Reshape [1, C, C]. TileBatch [B, C, C].
        using var transReshaped = gpuEngine.ReshapeGpu(transGpu, [1, numClasses, numClasses]);
        using var transTiled = gpuEngine.TileBatchGpu(transReshaped, batchSize);

        for (int t = 1; t < seqLen; t++)
        {
            // Gather emission[t]
            using var indicesT = CreateIndices(batchSize, seqLen, numClasses, t, backend);
            var emitT = gpuEngine.GatherGpu(input3D, indicesT, batchSize * numClasses, 1);
            var emitTReshaped = gpuEngine.ReshapeGpu(emitT, [batchSize, numClasses]); // [B, C]
            emitT.Dispose();

            // Expand viterbi [B, C] -> [B, C, 1] -> Tile [B, C, C]
            // We want score[b, prev, curr] = viterbi[b, prev] + trans[prev, curr]
            using var viterbiExpanded = gpuEngine.ReshapeGpu(viterbi, [batchSize, numClasses, 1]);
            using var viterbiTiled = gpuEngine.TileAxisGpu(viterbiExpanded, 2, numClasses); // [B, C, C]

            // Add transitions
            using var scores = gpuEngine.BroadcastAddGpu(viterbiTiled, transTiled); // [B, C, C]

            // Max over prev (axis 1) -> [B, 1, C]
            using var maxScores = gpuEngine.MaxAxisGpu(scores, 1);

            // ArgMax over prev -> [B, 1, C] indices (float)
            using var argMaxScores = gpuEngine.ArgMaxAxisGpu(scores, 1);

            // Update viterbi: maxScores + emission
            // maxScores is [B, 1, C]. Reshape [B, C].
            using var maxScoresFlat = gpuEngine.ReshapeGpu(maxScores, [batchSize, numClasses]);

            var nextViterbi = gpuEngine.BroadcastAddGpu(maxScoresFlat, emitTReshaped);

            // Download indices for backtracking
            // argMaxScores is [B, 1, C].
            var indicesFloat = new float[batchSize * numClasses];
            backend.DownloadBuffer(argMaxScores.Buffer, indicesFloat);

            // Store as int
            // Parallel loop over batch? Small size.
            for (int b = 0; b < batchSize; b++)
            {
                for (int c = 0; c < numClasses; c++)
                {
                    backpointers[t, b, c] = (int)indicesFloat[b * numClasses + c];
                }
            }

            viterbi.Dispose();
            viterbi = nextViterbi;
            emitTReshaped.Dispose();
        }

        // Final step: Add end scores
        // End [C] -> [1, C] -> Tile [B, C]
        using var endReshaped = gpuEngine.ReshapeGpu(endGpu, [1, numClasses]);
        using var endTiled = gpuEngine.TileBatchGpu(endReshaped, batchSize);

        using var finalScores = gpuEngine.BroadcastAddGpu(viterbi, endTiled);
        viterbi.Dispose();

        // Find best final path
        using var bestEnd = gpuEngine.ArgMaxAxisGpu(finalScores, 1); // [B, 1]
        var bestEndFloat = new float[batchSize];
        backend.DownloadBuffer(bestEnd.Buffer, bestEndFloat);

        // Backtrack on CPU
        var paths = new int[batchSize, seqLen];
        for (int b = 0; b < batchSize; b++)
        {
            paths[b, seqLen - 1] = (int)bestEndFloat[b];
            for (int t = seqLen - 2; t >= 0; t--)
            {
                paths[b, t] = backpointers[t + 1, b, paths[b, t + 1]];
            }
        }

        // Convert paths to One-Hot Tensor on CPU and upload
        var outputData = new float[batchSize * seqLen * numClasses];
        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                int cls = paths[b, t];
                outputData[b * seqLen * numClasses + t * numClasses + cls] = 1.0f;
            }
        }

        if (IsTrainingMode)
        {
            // Cache input for backward (if needed)
            // But Backward implementation uses CPU logic usually? 
            // The Backward logic in this layer is Manual/Autodiff on CPU tensors.
            // If we run Forward on GPU, we should cache tensors if we want GPU backward.
            // But Backward logic is complex (Forward-Backward algo).
            // For now, cache CPU tensor to support existing Backward.
            _lastInput = input;
            _lastOutput = new Tensor<T>(new Vector<T>(outputData.Select(x => NumOps.FromFloat(x)).ToArray()), [batchSize, seqLen, numClasses]);
        }

        return gpuEngine.UploadToGpu<T>(outputData, [batchSize, seqLen, numClasses], GpuTensorRole.Activation);
    }

    private IGpuBuffer CreateIndices(int batch, int seqLen, int numClasses, int t, IDirectGpuBackend backend)
    {
        var indices = new int[batch * numClasses];
        int batchStride = seqLen * numClasses;
        int timeOffset = t * numClasses;

        System.Threading.Tasks.Parallel.For(0, batch, b =>
        {
            int baseIdx = b * batchStride + timeOffset;
            int outBase = b * numClasses;
            for (int c = 0; c < numClasses; c++)
            {
                indices[outBase + c] = baseIdx + c;
            }
        });

        return backend.AllocateIntBuffer(indices);
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ConditionalRandomFieldLayer{T}"/> class with a scalar activation function.
    /// </summary>
    /// <param name="numClasses">The number of possible label classes.</param>
    /// <param name="sequenceLength">The length of the input sequences.</param>
    /// <param name="scalarActivation">The scalar activation function to apply to inputs. Defaults to identity if not specified.</param>
    /// <param name="engine">The computation engine for vectorized operations. Defaults to CPU if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new CRF layer with the specified number of classes and sequence length.
    /// It initializes the transition matrix, start scores, and end scores with appropriate random values.
    /// The scalar activation function is applied to the input features before the CRF processing.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new CRF layer with a standard activation function.
    /// 
    /// When creating a CRF layer, you need to specify:
    /// - How many different labels (classes) there are (e.g., 9 parts of speech)
    /// - How long each input sequence is (e.g., maximum sentence length)
    /// - Optionally, an activation function to transform the input features
    /// 
    /// The layer creates and initializes:
    /// - A transition matrix that learns how likely one label is to follow another
    /// - Start scores that learn which labels commonly appear at the beginning
    /// - End scores that learn which labels commonly appear at the end
    /// 
    /// These values start as small random numbers and are refined during training.
    /// </para>
    /// </remarks>
    public ConditionalRandomFieldLayer(int numClasses, int sequenceLength, IActivationFunction<T>? scalarActivation = null)
        : base([sequenceLength, numClasses], [sequenceLength, numClasses], scalarActivation ?? new IdentityActivation<T>())
    {
        if (sequenceLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(sequenceLength), "sequenceLength must be greater than 0.");

        _numClasses = numClasses;
        _sequenceLength = sequenceLength;
        _transitionMatrix = new Tensor<T>([_numClasses, _numClasses]);
        _startScores = new Tensor<T>([_numClasses]);
        _endScores = new Tensor<T>([_numClasses]);

        InitializeParameters();
        _isInitialized = true;
    }

    /// <summary>
    /// Lazy constructor: resolves <c>sequenceLength</c> from
    /// <c>input.Shape[0]</c> on first <see cref="Forward"/>.
    /// <paramref name="numClasses"/> (label vocabulary size) is
    /// architectural and stays required; only sequence length is
    /// shape-dependent. Note: parameter tensors (transition matrix,
    /// start/end scores) only depend on numClasses, not sequenceLength,
    /// so they're allocated eagerly here; only the base layer's
    /// InputShape / OutputShape are deferred.
    /// </summary>
    /// <param name="numClasses">Number of label classes (vocabulary size).</param>
    /// <param name="scalarActivation">Optional scalar activation (defaults to identity).</param>
    public ConditionalRandomFieldLayer(int numClasses, IActivationFunction<T>? scalarActivation = null)
        : base([-1, numClasses], [-1, numClasses], scalarActivation ?? new IdentityActivation<T>())
    {
        if (numClasses <= 0)
            throw new ArgumentOutOfRangeException(nameof(numClasses), "numClasses must be greater than 0.");

        _numClasses = numClasses;
        _sequenceLength = -1;
        _transitionMatrix = new Tensor<T>([_numClasses, _numClasses]);
        _startScores = new Tensor<T>([_numClasses]);
        _endScores = new Tensor<T>([_numClasses]);

        // Parameters do not depend on sequenceLength, so initialize
        // eagerly. _isInitialized stays false so EnsureInitialized's
        // sequenceLength validation runs once on first forward.
        InitializeParameters();
        _isInitialized = false;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="ConditionalRandomFieldLayer{T}"/> class with a vector activation function.
    /// </summary>
    /// <param name="numClasses">The number of possible label classes.</param>
    /// <param name="sequenceLength">The length of the input sequences.</param>
    /// <param name="vectorActivation">The vector activation function to apply to inputs. Defaults to identity if not specified.</param>
    /// <param name="engine">The computation engine for vectorized operations. Defaults to CPU if not specified.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new CRF layer with the specified number of classes and sequence length.
    /// It initializes the transition matrix, start scores, and end scores with appropriate random values.
    /// This overload accepts a vector activation function, which operates on entire vectors rather than
    /// individual elements when transforming the input features.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor creates a new CRF layer with a vector-based activation function.
    /// 
    /// A vector activation function:
    /// - Operates on entire groups of numbers at once, rather than one at a time
    /// - Can capture relationships between different elements in the input
    /// - Defaults to the Identity function, which doesn't change the values
    /// 
    /// This constructor works the same way as the other one, but it's useful when you need more
    /// complex activation patterns that consider the relationships between different inputs.
    /// </para>
    /// </remarks>
    public ConditionalRandomFieldLayer(int numClasses, int sequenceLength, IVectorActivationFunction<T>? vectorActivation = null)
        : base([sequenceLength, numClasses], [sequenceLength, numClasses], vectorActivation ?? new IdentityActivation<T>())
    {
        if (sequenceLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(sequenceLength), "sequenceLength must be greater than 0.");

        _numClasses = numClasses;
        _sequenceLength = sequenceLength;
        _transitionMatrix = new Tensor<T>([_numClasses, _numClasses]);
        _startScores = new Tensor<T>([_numClasses]);
        _endScores = new Tensor<T>([_numClasses]);

        InitializeParameters();
        _isInitialized = true;
    }

    /// <summary>
    /// Lazy constructor with vector activation — resolves
    /// <c>sequenceLength</c> from <c>input.Shape[0]</c> on first
    /// <see cref="Forward"/>. See the scalar-activation lazy ctor for
    /// design notes on why parameters initialize eagerly.
    /// </summary>
    /// <param name="numClasses">Number of label classes (vocabulary size).</param>
    /// <param name="vectorActivation">Optional vector activation (defaults to identity).</param>
    public ConditionalRandomFieldLayer(int numClasses, IVectorActivationFunction<T>? vectorActivation)
        : base([-1, numClasses], [-1, numClasses], vectorActivation ?? new IdentityActivation<T>())
    {
        if (numClasses <= 0)
            throw new ArgumentOutOfRangeException(nameof(numClasses), "numClasses must be greater than 0.");

        _numClasses = numClasses;
        _sequenceLength = -1;
        _transitionMatrix = new Tensor<T>([_numClasses, _numClasses]);
        _startScores = new Tensor<T>([_numClasses]);
        _endScores = new Tensor<T>([_numClasses]);

        InitializeParameters();
        _isInitialized = false;
    }

    /// <inheritdoc />
    /// <remarks>
    /// Reads the sequence length from <c>input.Shape[0]</c>. Per-batch
    /// inputs come in as <c>[sequenceLength, numClasses]</c> in this
    /// layer's contract.
    /// </remarks>
    protected override void OnFirstForward(Tensor<T> input)
    {
        int rank = input.Shape.Length;
        if (rank < 2)
            throw new ArgumentException(
                $"ConditionalRandomFieldLayer requires rank>=2 input [seqLen, numClasses]; got rank {rank}.", nameof(input));

        int sequenceLength = input.Shape[rank - 2];
        if (sequenceLength <= 0)
            throw new ArgumentException(
                $"ConditionalRandomFieldLayer's sequence length must be positive; got {sequenceLength} from input shape.",
                nameof(input));

        int seenClasses = input.Shape[rank - 1];
        if (seenClasses != _numClasses)
            throw new ArgumentException(
                $"ConditionalRandomFieldLayer's numClasses mismatch: layer was constructed with {_numClasses} classes, " +
                $"but input shape's last axis is {seenClasses}.", nameof(input));

        _sequenceLength = sequenceLength;
        ResolveShapes(new[] { sequenceLength, _numClasses }, new[] { sequenceLength, _numClasses });
    }

    /// <inheritdoc />
    /// <remarks>
    /// CRF parameters (transition matrix, start/end scores) only depend
    /// on <c>numClasses</c>, not <c>sequenceLength</c>, so they were
    /// allocated eagerly in the lazy ctor. EnsureInitialized just flips
    /// the initialized flag once shape is resolved.
    /// </remarks>
    protected override void EnsureInitialized()
    {
        if (_isInitialized) return;
        if (_sequenceLength <= 0)
            throw new InvalidOperationException(
                "ConditionalRandomFieldLayer cannot initialize until OnFirstForward has resolved the sequence length from input shape.");

        _isInitialized = true;
    }

    /// <summary>
    /// Initializes the layer's parameters with scaled random values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method initializes the transition matrix, start scores, and end scores with small random values
    /// scaled based on the number of classes. This initialization approach helps with training convergence
    /// by keeping the initial values in an appropriate range.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up the initial values for the layer's parameters.
    /// 
    /// Before training begins, we need to:
    /// - Fill the transition matrix with small random values
    /// - Set start and end scores to small random values
    /// 
    /// This initialization is important because:
    /// - Starting with the right range of random values helps the network learn faster
    /// - If values are too large or too small, the network might have trouble learning
    /// 
    /// The scaling factor ensures the random values are an appropriate size based on
    /// the number of classes in the problem.
    /// </para>
    /// </remarks>
    private void InitializeParameters()
    {
        // VECTORIZED: Initialize parameters with scaled random values
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_numClasses + _numClasses)));
        T half = NumOps.FromDouble(0.5);

        // Initialize transition matrix: (random - 0.5) * scale
        var transRandom = Tensor<T>.CreateRandom(_transitionMatrix.Length, 1).Reshape(_transitionMatrix._shape);
        var transHalf = new Tensor<T>(_transitionMatrix._shape);
        transHalf.Fill(half);
        var transCentered = Engine.TensorSubtract(transRandom, transHalf);
        _transitionMatrix = Engine.TensorMultiplyScalar(transCentered, scale);

        // Initialize start scores: (random - 0.5) * scale
        var startRandom = Tensor<T>.CreateRandom(_startScores.Length, 1).Reshape(_startScores._shape);
        var startHalf = new Tensor<T>(_startScores._shape);
        startHalf.Fill(half);
        var startCentered = Engine.TensorSubtract(startRandom, startHalf);
        _startScores = Engine.TensorMultiplyScalar(startCentered, scale);

        // Initialize end scores: (random - 0.5) * scale
        var endRandom = Tensor<T>.CreateRandom(_endScores.Length, 1).Reshape(_endScores._shape);
        var endHalf = new Tensor<T>(_endScores._shape);
        endHalf.Fill(half);
        var endCentered = Engine.TensorSubtract(endRandom, endHalf);
        _endScores = Engine.TensorMultiplyScalar(endCentered, scale);

        // Register after all reassignments so references are to final tensors
        RegisterTrainableParameter(_transitionMatrix, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_startScores, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_endScores, PersistentTensorRole.Weights);
    }

    /// <summary>
    /// Performs the forward pass of the CRF layer.
    /// </summary>
    /// <param name="input">The input tensor containing sequence features.</param>
    /// <returns>The output tensor containing the most likely sequence labels.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the CRF layer using the Viterbi algorithm to find
    /// the most likely sequence of labels. It first applies any activation function to transform the
    /// input features, then uses dynamic programming to find the optimal label sequence considering
    /// the transition scores between labels, start scores, and end scores. The output is a one-hot
    /// encoded tensor representing the best label at each position in the sequence.
    /// </para>
    /// <para><b>For Beginners:</b> This method finds the best sequence of labels for the input data.
    /// 
    /// The forward pass has several steps:
    /// 
    /// 1. Transform the input features using the activation function (if specified)
    /// 2. For each sequence in the batch, run the Viterbi algorithm:
    ///    - Start with the initial scores and input features
    ///    - For each position in the sequence, calculate the best previous label
    ///    - Keep track of the best path using "backpointers"
    ///    - Find the best final label considering the end scores
    ///    - Trace backwards to find the optimal sequence of labels
    /// 3. Convert the best label sequence to a one-hot encoded output
    /// 
    /// The Viterbi algorithm is like finding the shortest path through a grid,
    /// where each step considers both the current position's score and the
    /// transition cost from the previous position.
    /// 
    /// This approach ensures that the entire sequence of labels makes sense together,
    /// rather than just picking the best label at each position independently.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Lazy-ctor instances start with _sequenceLength = -1; resolve
        // from input.Shape on first call. Eager-ctor instances are
        // already initialized so both calls are no-ops.
        if (!IsShapeResolved) OnFirstForward(input);
        EnsureInitialized();

        // Validate input shape on every call, not just the first.
        // OnFirstForward only fires once; later inputs with different
        // sequence-length or class-count would otherwise reach the
        // Viterbi loops below and silently truncate or index out of
        // range. Numbers-of-classes is architectural (fixed at
        // construction), sequence-length is shape-dependent (varies
        // between inputs), but both must match what the layer was
        // constructed for.
        int seenRank = input.Shape.Length;
        if (seenRank < 2)
            throw new ArgumentException(
                $"ConditionalRandomFieldLayer requires rank>=2 input [seqLen, numClasses] " +
                $"or [batch, seqLen, numClasses]; got rank {seenRank}.", nameof(input));
        int seenClasses = input.Shape[seenRank - 1];
        if (seenClasses != _numClasses)
            throw new ArgumentException(
                $"ConditionalRandomFieldLayer's numClasses mismatch: layer was constructed " +
                $"with {_numClasses} classes, but input shape's last axis is {seenClasses}. " +
                $"numClasses is architectural and cannot vary across inputs.", nameof(input));
        int seenSeqLen = input.Shape[seenRank - 2];
        if (seenSeqLen <= 0)
            throw new ArgumentException(
                $"ConditionalRandomFieldLayer's sequence length must be positive; got " +
                $"{seenSeqLen} from input shape.", nameof(input));
        if (seenSeqLen != _sequenceLength)
            throw new ArgumentException(
                $"ConditionalRandomFieldLayer's sequence length mismatch: layer was " +
                $"resolved with sequenceLength={_sequenceLength} (from constructor or " +
                $"first forward pass), but this input's sequence dimension is " +
                $"{seenSeqLen}. Viterbi decoding, transitions buffer, and gradient " +
                $"reshape are all sized to _sequenceLength; a different value would " +
                $"silently truncate or run out of bounds. If you need variable-length " +
                $"sequences, pad to a fixed length and mask, or construct one CRF per " +
                $"length bucket.", nameof(input));

        // Store original shape for any-rank tensor support
        _originalInputShape = input._shape;
        int rank = input.Shape.Length;

        // CRF expects 3D input: [batchSize, sequenceLength, numClasses]
        // Handle any-rank tensor: normalize to 3D for processing
        Tensor<T> input3D;
        int batchSize;

        if (rank == 2)
        {
            // 2D [sequenceLength, numClasses]: add batch dim
            batchSize = 1;
            input3D = Engine.Reshape(input, [1, input.Shape[0], input.Shape[1]]);
        }
        else if (rank == 3)
        {
            // Standard 3D [batchSize, sequenceLength, numClasses]
            batchSize = input.Shape[0];
            input3D = input;
        }
        else if (rank > 3)
        {
            // Higher-rank: collapse leading dims into batch
            // Input shape: [...batch dims..., sequenceLength, numClasses]
            int flatBatch = 1;
            for (int d = 0; d < rank - 2; d++)
                flatBatch *= input.Shape[d];
            batchSize = flatBatch;
            input3D = Engine.Reshape(input, [flatBatch, input.Shape[rank - 2], input.Shape[rank - 1]]);
        }
        else
        {
            // 1D: treat as [1, 1, features] - single batch, single timestep
            batchSize = 1;
            input3D = Engine.Reshape(input, [1, 1, input.Shape[0]]);
        }

        _lastInput = input3D;

        // Apply the configured activation BEFORE the training-mode short-
        // circuit so training and inference both decode the SAME score
        // surface. Returning raw emissions in training mode while inference
        // decoded the activated emissions would leave the model learning
        // one objective and serving another — equivalent to silently
        // training on a different layer than the one in production.
        // For the common IdentityActivation case this is a no-op via the
        // reference-equal early return inside ApplyActivation.
        Tensor<T> sequenceScores;
        if (UsingVectorActivation)
        {
            sequenceScores = ApplyActivation(input3D);
        }
        else if (ScalarActivation != null && !(ScalarActivation is IdentityActivation<T>))
        {
            sequenceScores = ApplyActivation(input3D);
        }
        else
        {
            sequenceScores = input3D;
        }

        // Training-mode short-circuit: return the activated emissions so the
        // tape sees an unbroken differentiable chain from the upstream
        // projection back to the loss. The proper CRF training loss is the
        // negative log-likelihood (log-partition − gold-path), which a
        // CRF-aware caller (e.g. BiLSTMCRF.Train via the dedicated
        // CrfNegativeLogLikelihood helpers below) computes against this
        // emissions tensor and the gold labels separately. Returning
        // Viterbi-decoded labels here (the prior behavior) silently broke
        // backprop: the per-timestep argmax + one-hot encoding is not
        // differentiable, and the per-class log-sum-exp loops with raw
        // Math.Exp/Math.Log scalar arithmetic bypass the tape entirely
        // — every test that asserts "params changed after Train" failed
        // because the gradient chain ended at the CRF.
        //
        // Inference still runs the full Viterbi decode below; that path
        // doesn't need to be tape-tracked.
        if (IsTrainingMode)
        {
            _lastOutput = sequenceScores;
            return _originalInputShape != null && _originalInputShape.Length != 3
                ? Engine.Reshape(sequenceScores, _originalInputShape)
                : sequenceScores;
        }

        var output = TensorAllocator.Rent<T>([batchSize, _sequenceLength, _numClasses]);

        // Process each batch item (Viterbi requires sequential time processing)
        for (int b = 0; b < batchSize; b++)
        {
            // === VECTORIZED: Extract sequence for this batch item ===
            var batchSeq = sequenceScores.GetSliceAlongDimension(b, 0); // [sequenceLength, numClasses]

            // === VECTORIZED Viterbi Algorithm ===
            var viterbi = new Tensor<T>([_sequenceLength, _numClasses]);
            var backpointers = new Matrix<int>(_sequenceLength, _numClasses);

            // VECTORIZED: Initialize first timestep - startScores + emissions[0]
            var firstEmissions = batchSeq.GetSliceAlongDimension(0, 0); // [numClasses]
            var firstViterbi = Engine.TensorAdd(firstEmissions, _startScores);
            viterbi.SetSlice(0, 0, firstViterbi);

            // Recursion over time (inherently sequential)
            for (int t = 1; t < _sequenceLength; t++)
            {
                var currentEmissions = batchSeq.GetSliceAlongDimension(t, 0); // [numClasses]
                var prevViterbi = viterbi.GetSliceAlongDimension(t - 1, 0); // [numClasses]

                // For each current class, compute max over prev classes
                // score[c] = max_prevC(viterbi[t-1, prevC] + transition[prevC, c]) + emissions[t, c]
                // This can be done by broadcasting:
                // prevViterbi: [numClasses, 1] + transition: [numClasses, numClasses] -> [numClasses, numClasses]
                // Then max over axis 0

                var prevExpanded = Engine.Reshape(prevViterbi, [_numClasses, 1]); // [numClasses, 1]
                var scoresWithTrans = Engine.TensorBroadcastAdd(prevExpanded, _transitionMatrix); // [numClasses, numClasses]

                // This branch is INFERENCE-ONLY: the training-mode short-circuit
                // at the top of Forward (line ~730) returns raw emissions before
                // the Viterbi loop runs, and CRF training uses the dedicated
                // tape-tracked log-sum-exp implementation in
                // ComputeNegativeLogLikelihood. So per-class Viterbi-max +
                // backpointer recording is the only path we need here.
                var maxScores = new Tensor<T>([_numClasses]);
                for (int c = 0; c < _numClasses; c++)
                {
                    T maxVal = NumOps.MinValue;
                    int maxIdx = 0;
                    for (int prevC = 0; prevC < _numClasses; prevC++)
                    {
                        T val = scoresWithTrans[prevC, c];
                        if (NumOps.GreaterThan(val, maxVal))
                        {
                            maxVal = val;
                            maxIdx = prevC;
                        }
                    }
                    maxScores[c] = maxVal;
                    backpointers[t, c] = maxIdx;
                }

                // Add emissions: maxScores + currentEmissions
                var currentViterbi = Engine.TensorAdd(maxScores, currentEmissions);
                viterbi.SetSlice(0, t, currentViterbi);
            }

            // === VECTORIZED Termination ===
            var lastViterbi = viterbi.GetSliceAlongDimension(_sequenceLength - 1, 0);
            var finalScores = Engine.TensorAdd(lastViterbi, _endScores);

            // Find argmax
            T maxFinalScore = NumOps.MinValue;
            int maxFinalClass = 0;
            for (int c = 0; c < _numClasses; c++)
            {
                if (NumOps.GreaterThan(finalScores[c], maxFinalScore))
                {
                    maxFinalScore = finalScores[c];
                    maxFinalClass = c;
                }
            }

            // Backtracking (inherently sequential)
            var bestPath = new int[_sequenceLength];
            bestPath[_sequenceLength - 1] = maxFinalClass;
            for (int t = _sequenceLength - 2; t >= 0; t--)
            {
                bestPath[t] = backpointers[t + 1, bestPath[t + 1]];
            }

            // Inference-only path (training-mode short-circuits at the top
            // of Forward and never reaches the Viterbi loop): write the
            // one-hot encoded best path into the output tensor.
            for (int t = 0; t < _sequenceLength; t++)
            {
                output[b, t, bestPath[t]] = NumOps.One;
            }
        }

        _lastOutput = output;

        // Restore original rank if needed for any-rank tensor support
        if (_originalInputShape != null && _originalInputShape.Length != 3)
        {
            return Engine.Reshape(output, _originalInputShape);
        }

        return output;
    }

    private Tensor<T> NormalizeOutputGradient(Tensor<T> outputGradient)
    {
        if (_originalInputShape == null || _originalInputShape.Length == 3)
        {
            return outputGradient;
        }

        if (_originalInputShape.Length == 2)
        {
            return outputGradient.Reshape([1, _originalInputShape[0], _originalInputShape[1]]);
        }

        if (_originalInputShape.Length == 1)
        {
            int expected = _sequenceLength * _numClasses;
            if (outputGradient.Length == expected)
            {
                return outputGradient.Reshape([1, _sequenceLength, _numClasses]);
            }

            return outputGradient.Reshape([1, 1, outputGradient.Shape[0]]);
        }

        int flatBatch = 1;
        for (int d = 0; d < _originalInputShape.Length - 2; d++)
        {
            flatBatch *= _originalInputShape[d];
        }

        return outputGradient.Reshape([flatBatch, _sequenceLength, _numClasses]);
    }

    /// <summary>
    /// Updates the layer's parameters using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate to use for the parameter updates.</param>
    /// <exception cref="InvalidOperationException">Thrown when update is called before backward.</exception>
    /// <remarks>
    /// <para>
    /// This method updates the layer's parameters (transition matrix, start scores, and end scores) based on
    /// the gradients calculated during the backward pass. The learning rate controls the size of the parameter
    /// updates. The update is performed by subtracting the scaled gradients from the current parameters.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates the layer's internal values during training.
    /// 
    /// After calculating the gradients in the backward pass:
    /// - This method applies those changes to the transition matrix, start scores, and end scores
    /// - The learning rate controls how big each update step is
    /// - Smaller learning rates mean slower but more stable learning
    /// - Larger learning rates mean faster but potentially unstable learning
    /// 
    /// The formula is simple: new_value = old_value - (gradient * learning_rate)
    /// 
    /// This is how the layer "learns" from data over time, gradually improving its ability
    /// to predict the correct sequence of labels.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        if (_transitionMatrixGradient == null || _startScoresGradient == null || _endScoresGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        // Update using Engine tensor operations: param = param - lr * gradient
        var scaledTransGrad = Engine.TensorMultiplyScalar(_transitionMatrixGradient, learningRate);
        _transitionMatrix = Engine.TensorSubtract(_transitionMatrix, scaledTransGrad);

        var scaledStartGrad = Engine.TensorMultiplyScalar(_startScoresGradient, learningRate);
        _startScores = Engine.TensorSubtract(_startScores, scaledStartGrad);

        var scaledEndGrad = Engine.TensorMultiplyScalar(_endScoresGradient, learningRate);
        _endScores = Engine.TensorSubtract(_endScores, scaledEndGrad);
    }

    /// <summary>
    /// Gets all trainable parameters from the layer as a single vector.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves all trainable parameters from the layer (transition matrix, start scores, and
    /// end scores) and combines them into a single vector. This is useful for optimization algorithms that
    /// operate on all parameters at once, or for saving and loading model weights.
    /// </para>
    /// <para><b>For Beginners:</b> This method collects all the learnable values from the layer into a single list.
    /// 
    /// The parameters include:
    /// - The transition matrix (shows how likely one label is to follow another)
    /// - The start scores (shows which labels are likely at sequence beginnings)
    /// - The end scores (shows which labels are likely at sequence endings)
    /// 
    /// All these values are flattened into a single long list (vector).
    /// 
    /// This is useful for:
    /// - Saving the model to disk
    /// - Loading parameters from a previously trained model
    /// - Advanced optimization techniques that need access to all parameters
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // Use Vector<T>.Concatenate for efficient parameter collection
        var flatTrans = new Vector<T>(_transitionMatrix.ToArray());
        var flatStart = new Vector<T>(_startScores.ToArray());
        var flatEnd = new Vector<T>(_endScores.ToArray());

        return Vector<T>.Concatenate(Vector<T>.Concatenate(flatTrans, flatStart), flatEnd);
    }

    /// <summary>
    /// Computes the linear-chain CRF negative log-likelihood for a batched
    /// emissions tensor and the corresponding gold-label sequence:
    /// <c>NLL(emissions, labels) = logZ(emissions) − goldScore(emissions, labels)</c>
    /// where <c>logZ</c> is the partition function (computed via the standard
    /// forward algorithm) and <c>goldScore</c> sums the emission + transition
    /// + start/end scores along the gold path. Per Lafferty et al. 2001 §3 +
    /// the BiLSTM-CRF formulation in Lample et al. 2016 §3.2.
    /// </summary>
    /// <param name="emissions">Emission scores from the upstream projection.
    /// Shape: <c>[batch, sequenceLength, numClasses]</c>. Must be tape-tracked
    /// for backprop to flow into the upstream layers.</param>
    /// <param name="labels">Gold label indices. Shape:
    /// <c>[batch, sequenceLength]</c> with integer class indices in
    /// <c>[0, numClasses)</c>, or 1D <c>[sequenceLength]</c> for a single
    /// example (auto-promoted to a singleton batch).</param>
    /// <returns>Per-batch-element NLL averaged. Single scalar tensor that
    /// the caller can use directly as the training loss.</returns>
    /// <remarks>
    /// <para>
    /// All operations route through tape-tracked <see cref="Engine"/> ops
    /// (TensorAdd, TensorMultiplyScalar, ReduceLogSumExp, etc.) so backprop
    /// reaches BOTH the upstream emission-producing layers AND this layer's
    /// own transition matrix / start scores / end scores. The forward
    /// algorithm's per-timestep log-sum-exp is implemented via the engine's
    /// tape-aware <see cref="IEngine.LogSumExp{T}"/> when present, falling
    /// back to a numerically stable <c>maxVal + log(sum(exp(x − maxVal)))</c>
    /// chain built from engine ops.
    /// </para>
    /// <para>
    /// Returning a scalar (not the decoded labels) is the contract a CRF
    /// loss expects — the BiLSTMCRF / CNNBiLSTMCRF training step uses this
    /// as the model's loss in place of cross-entropy on Viterbi-decoded
    /// labels (which would be non-differentiable).
    /// </para>
    /// </remarks>
    internal Tensor<T> ComputeNegativeLogLikelihood(Tensor<T> emissions, Tensor<T> labels)
    {
        if (emissions == null) throw new ArgumentNullException(nameof(emissions));
        if (labels == null) throw new ArgumentNullException(nameof(labels));

        // Promote rank-2 emissions [seqLen, numClasses] to a singleton batch.
        Tensor<T> emissions3D;
        if (emissions.Rank == 2)
            emissions3D = Engine.Reshape(emissions, [1, emissions.Shape[0], emissions.Shape[1]]);
        else if (emissions.Rank == 3)
            emissions3D = emissions;
        else
            throw new ArgumentException(
                $"ComputeNegativeLogLikelihood requires rank-2 [seqLen, numClasses] or rank-3 [batch, seqLen, numClasses] emissions; got rank {emissions.Rank}.",
                nameof(emissions));

        int batchSize = emissions3D.Shape[0];
        int seqLen = emissions3D.Shape[1];
        int numClasses = emissions3D.Shape[2];
        if (numClasses != _numClasses)
            throw new ArgumentException(
                $"emissions last axis ({numClasses}) doesn't match CRF numClasses ({_numClasses}).",
                nameof(emissions));

        // Promote rank-1 labels [seqLen] to a singleton batch [1, seqLen].
        Tensor<T> labels2D;
        if (labels.Rank == 1)
            labels2D = Engine.Reshape(labels, [1, labels.Shape[0]]);
        else if (labels.Rank == 2)
            labels2D = labels;
        else
            throw new ArgumentException(
                $"ComputeNegativeLogLikelihood requires rank-1 [seqLen] or rank-2 [batch, seqLen] labels; got rank {labels.Rank}.",
                nameof(labels));

        if (labels2D.Shape[0] != batchSize || labels2D.Shape[1] != seqLen)
            throw new ArgumentException(
                $"labels shape [{string.Join(",", labels2D.Shape.ToArray())}] doesn't match emissions [{batchSize}, {seqLen}, ...].",
                nameof(labels));

        // Empty-batch guard: with batchSize == 0 the accumulator loop
        // below never executes and totalNll stays null, which the prior
        // implementation papered over with `totalNll!` (forbidden by
        // project rules + still crashes at runtime). An empty-batch loss
        // is meaningless (no examples to score), so reject it explicitly
        // with a clear diagnostic rather than NRE downstream.
        if (batchSize == 0)
            throw new ArgumentException(
                "emissions has batch size 0; cannot compute NLL on an empty batch.",
                nameof(emissions));

        // Per-batch tape-tracked NLL accumulator. All ops below route through
        // Engine (TensorAdd, ReduceSum, TensorExp, etc.) so gradients flow into
        // emissions (upstream layers) AND _transitionMatrix / _startScores /
        // _endScores (this layer's parameters).
        Tensor<T>? totalNll = null;
        for (int b = 0; b < batchSize; b++)
        {
            // Slice this batch element: [seqLen, numClasses]
            var emissionsB = Engine.TensorSliceAxis(emissions3D, axis: 0, index: b);

            // Build constant one-hot encoding of labels for batch b — needed
            // for tape-tracked gathers on emissions / transitions / start / end.
            // Labels are integer indices stored as T; we read them as ints here
            // and emit a constant one-hot tensor (no gradient w.r.t. labels).
            var labelsOH = BuildLabelOneHotForBatch(labels2D, b, seqLen, numClasses);

            // === logZ via tape-tracked forward algorithm ===
            // alpha_0 = emissions[0] + startScores
            var emit0 = Engine.TensorSliceAxis(emissionsB, axis: 0, index: 0); // [C]
            var alpha = Engine.TensorAdd(emit0, _startScores);                 // [C]

            for (int t = 1; t < seqLen; t++)
            {
                var emitT = Engine.TensorSliceAxis(emissionsB, axis: 0, index: t); // [C]

                // scores[i, j] = alpha[i] + transitions[i, j] + emitT[j]
                var alphaCol = Engine.Reshape(alpha, [_numClasses, 1]);            // [C, 1]
                var emitRow = Engine.Reshape(emitT, [1, _numClasses]);             // [1, C]
                var alphaPlusTrans = Engine.TensorBroadcastAdd(alphaCol, _transitionMatrix); // [C, C]
                var scores = Engine.TensorBroadcastAdd(alphaPlusTrans, emitRow);   // [C, C]

                // alpha_new = LogSumExp(scores, axis=0) → [C]
                alpha = TapeLogSumExpAxis(scores, axis: 0); // [C] (axis-0 reduced)
            }

            // logZ = LogSumExp(alpha + endScores)
            var alphaEnd = Engine.TensorAdd(alpha, _endScores);                // [C]
            var alphaEnd2D = Engine.Reshape(alphaEnd, [1, _numClasses]);       // [1, C]
            var logZ = TapeLogSumExpAxis(alphaEnd2D, axis: 1);                 // [1]

            // === goldScore via one-hot encoding of labels ===
            // emission contribution: sum_t emissions[t, labels[t]] = sum(emissionsB * labelsOH)
            var emissionMasked = Engine.TensorMultiply(emissionsB, labelsOH);  // [seqLen, C]
            var emissionTotal = Engine.ReduceSum(emissionMasked, [0, 1], keepDims: false); // scalar
            var emissionTotal1D = Engine.Reshape(emissionTotal, [1]);

            // start contribution: startScores[labels[0]] = sum(ohFirst * startScores)
            var ohFirst = Engine.TensorSliceAxis(labelsOH, axis: 0, index: 0); // [C]
            var startMasked = Engine.TensorMultiply(ohFirst, _startScores);    // [C]
            var startTotal = Engine.ReduceSum(startMasked, [0], keepDims: true); // [1]

            // end contribution: endScores[labels[seqLen-1]]
            var ohLast = Engine.TensorSliceAxis(labelsOH, axis: 0, index: seqLen - 1); // [C]
            var endMasked = Engine.TensorMultiply(ohLast, _endScores);         // [C]
            var endTotal = Engine.ReduceSum(endMasked, [0], keepDims: true);   // [1]

            // transition contribution (if seqLen > 1):
            //   sum_{t=1..T-1} transitions[labels[t-1], labels[t]]
            // = sum_{t, i, j} ohPrev[t, i] * transitions[i, j] * ohCurr[t, j]
            // Build ohPrev [seqLen-1, C] and ohCurr [seqLen-1, C], take outer
            // product per timestep [seqLen-1, C, C], elementwise-multiply by
            // transitions broadcast to [1, C, C], reduce.
            Tensor<T> goldScore;
            if (seqLen > 1)
            {
                var ohPrev = SliceLabelOneHotSubrange(labelsOH, start: 0, count: seqLen - 1);     // [seqLen-1, C]
                var ohCurr = SliceLabelOneHotSubrange(labelsOH, start: 1, count: seqLen - 1);     // [seqLen-1, C]

                var ohPrevExpanded = Engine.Reshape(ohPrev, [seqLen - 1, _numClasses, 1]);
                var ohCurrExpanded = Engine.Reshape(ohCurr, [seqLen - 1, 1, _numClasses]);
                var ohOuter = Engine.TensorBroadcastMultiply(ohPrevExpanded, ohCurrExpanded);    // [seqLen-1, C, C]

                var transExpanded = Engine.Reshape(_transitionMatrix, [1, _numClasses, _numClasses]);
                var transMasked = Engine.TensorBroadcastMultiply(ohOuter, transExpanded);        // [seqLen-1, C, C]
                var transTotal = Engine.ReduceSum(transMasked, [0, 1, 2], keepDims: false);      // scalar
                var transTotal1D = Engine.Reshape(transTotal, [1]);

                // goldScore = emission + transition + start + end
                goldScore = Engine.TensorAdd(emissionTotal1D, transTotal1D);
                goldScore = Engine.TensorAdd(goldScore, startTotal);
                goldScore = Engine.TensorAdd(goldScore, endTotal);
            }
            else
            {
                // No transitions for seqLen == 1.
                goldScore = Engine.TensorAdd(emissionTotal1D, startTotal);
                goldScore = Engine.TensorAdd(goldScore, endTotal);
            }

            // Per-example NLL = logZ - goldScore  (both shape [1])
            var perExampleNll = Engine.TensorSubtract(logZ, goldScore);

            totalNll = totalNll == null
                ? perExampleNll
                : Engine.TensorAdd(totalNll, perExampleNll);
        }

        // Mean over batch. batchSize == 0 was rejected above, so the
        // accumulator loop executed at least once and totalNll is
        // guaranteed non-null here — promote to a local with a
        // pattern-matched non-null guard so we don't lean on the
        // null-forgiving operator (forbidden by project rules).
        if (totalNll is not { } accumulated)
            throw new InvalidOperationException(
                "CRF NLL accumulator was unexpectedly null after a non-empty batch loop.");
        var invBatch = NumOps.FromDouble(1.0 / batchSize);
        var meanNll = Engine.TensorMultiplyScalar(accumulated, invBatch); // shape [1]
        return meanNll;
    }

    /// <summary>
    /// Tape-tracked log-sum-exp along a single axis, returning the reduced
    /// tensor with that axis removed. Implemented as
    /// <c>max + log(sum(exp(x − max)))</c> for numerical stability — all
    /// sub-operations route through <see cref="Engine"/> so the autograd tape
    /// can backprop through this composite reduction.
    /// </summary>
    private Tensor<T> TapeLogSumExpAxis(Tensor<T> x, int axis)
    {
        // max along axis, keepDims=true so we can broadcast-subtract
        var max = Engine.ReduceMax(x, [axis], keepDims: true, out _);
        var negMax = Engine.TensorNegate(max);
        var shifted = Engine.TensorBroadcastAdd(x, negMax);
        var expShifted = Engine.TensorExp(shifted);
        var sumExp = Engine.ReduceSum(expShifted, [axis], keepDims: true);
        var logSum = Engine.TensorLog(sumExp);
        var lseKeepDim = Engine.TensorAdd(logSum, max);

        // Now squeeze the reduced axis to match the shape contract.
        var inShape = x.Shape.ToArray();
        var outShape = new int[inShape.Length - 1];
        int oi = 0;
        for (int i = 0; i < inShape.Length; i++)
        {
            if (i == axis) continue;
            outShape[oi++] = inShape[i];
        }
        return Engine.Reshape(lseKeepDim, outShape);
    }

    /// <summary>
    /// Builds a [seqLen, numClasses] one-hot encoding of the gold labels for
    /// a single batch element. Returned tensor is a constant — the tape will
    /// treat it as a leaf with no gradient.
    /// </summary>
    private Tensor<T> BuildLabelOneHotForBatch(Tensor<T> labels2D, int b, int seqLen, int numClasses)
    {
        // ComputeNegativeLogLikelihood documents that labels carry integer
        // class indices. A fractional double like 0.51 would have been
        // silently rounded to 1 by the prior implementation — that loses
        // information (the caller almost certainly didn't mean "round
        // 0.51 toward 1") and trains the model against fabricated targets.
        // Fail fast instead: require values that round-trip exactly
        // through int (within FP-representation tolerance) and reject
        // anything else with a clear diagnostic.
        const double IntegerTolerance = 1e-6;
        var oh = new Tensor<T>([seqLen, numClasses]);
        for (int t = 0; t < seqLen; t++)
        {
            double rawLabel = NumOps.ToDouble(labels2D[b, t]);
            double rounded = Math.Round(rawLabel);
            if (Math.Abs(rawLabel - rounded) > IntegerTolerance)
                throw new ArgumentException(
                    $"Label at [b={b}, t={t}] is {rawLabel}, which is not an integer class index. " +
                    "CRF training requires integer label indices in [0, NumClasses); fractional values are not allowed.",
                    nameof(labels2D));
            int label = (int)rounded;
            if (label < 0 || label >= numClasses)
                throw new ArgumentOutOfRangeException(nameof(labels2D),
                    $"Label at [b={b}, t={t}] is {label}, must be in [0, {numClasses}).");
            oh[t, label] = NumOps.One;
        }
        return oh;
    }

    /// <summary>
    /// Returns a tape-tracked slice of a [seqLen, numClasses] one-hot tensor
    /// covering rows <c>[start, start+count)</c>. Used to construct the
    /// previous/current label one-hots for transition-score gathering.
    /// </summary>
    private Tensor<T> SliceLabelOneHotSubrange(Tensor<T> labelsOH, int start, int count)
    {
        // Pull rows one-by-one via TensorSliceAxis and stack via repeated
        // broadcast adds into a fresh tensor. Building the subrange directly
        // from the dense one-hot keeps it constant (no gradient needed).
        int numClasses = labelsOH.Shape[1];
        var sub = new Tensor<T>([count, numClasses]);
        for (int i = 0; i < count; i++)
        {
            int row = start + i;
            for (int c = 0; c < numClasses; c++)
                sub[i, c] = labelsOH[row, c];
        }
        return sub;
    }

    /// <summary>
    /// Sets the trainable parameters for the layer.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters to set.</param>
    /// <exception cref="ArgumentException">Thrown when the parameters vector has incorrect length.</exception>
    /// <remarks>
    /// <para>
    /// This method sets the trainable parameters for the layer (transition matrix, start scores, and end scores)
    /// from a single vector. This is useful for loading saved model weights or for implementing optimization
    /// algorithms that operate on all parameters at once.
    /// </para>
    /// <para><b>For Beginners:</b> This method updates all the learnable values in the layer.
    /// 
    /// When setting parameters:
    /// - The input must be a vector with the correct length
    /// - The first part is used for the transition matrix
    /// - The next part is used for the start scores
    /// - The final part is used for the end scores
    /// 
    /// This is useful for:
    /// - Loading a previously saved model
    /// - Transferring parameters from another model
    /// - Testing different parameter values
    /// 
    /// An error is thrown if the input vector doesn't have the expected number of parameters.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameterGradients()
    {
        var flatTrans = _transitionMatrixGradient != null
            ? new Vector<T>(_transitionMatrixGradient.ToArray())
            : new Vector<T>(_numClasses * _numClasses);
        var flatStart = _startScoresGradient != null
            ? new Vector<T>(_startScoresGradient.ToArray())
            : new Vector<T>(_numClasses);
        var flatEnd = _endScoresGradient != null
            ? new Vector<T>(_endScoresGradient.ToArray())
            : new Vector<T>(_numClasses);

        return Vector<T>.Concatenate(Vector<T>.Concatenate(flatTrans, flatStart), flatEnd);
    }

    public override void ClearGradients()
    {
        _transitionMatrixGradient = null;
        _startScoresGradient = null;
        _endScoresGradient = null;
    }

    /// <summary>
    /// Persists CRF-specific constructor parameters so deserialization can
    /// reconstruct the layer with the same <c>numClasses</c> and
    /// <c>sequenceLength</c>. Without this override, the deserialization
    /// helper's metadata-matcher derives <c>numClasses</c> from the
    /// serialized <c>outputShape</c>'s last axis — which is unreliable when
    /// the layer was constructed eagerly (the base ctor stores
    /// <c>[-1, numClasses]</c> as the placeholder output shape, and any
    /// rank-3 forward sets it to <c>[batch, seqLen, numClasses]</c>, so the
    /// last-axis derivation only works coincidentally). Round-tripping
    /// these values explicitly closes the Clone()/DeepCopy() failure with
    /// "Expected N parameters, but got M" that broke every CRF-using model
    /// when N == new instance's (numClasses² + 2·numClasses) and M == the
    /// original instance's saved param count.
    /// </summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["NumClasses"] = _numClasses.ToString(System.Globalization.CultureInfo.InvariantCulture);
        // _sequenceLength may be -1 in the lazy-ctor case (resolved on
        // first Forward). Persist whatever the current value is — the
        // deserialize-matcher will derive from inputShape[0] when the
        // value is -1, matching the lazy-ctor contract.
        metadata["SequenceLength"] = _sequenceLength.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int transSize = _numClasses * _numClasses;
        int totalParams = transSize + _numClasses * 2;

        if (parameters.Length != totalParams)
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");

        // VECTORIZED: Use Vector.Slice and Tensor.FromVector
        var transVec = parameters.Slice(0, transSize);
        var startVec = parameters.Slice(transSize, _numClasses);
        var endVec = parameters.Slice(transSize + _numClasses, _numClasses);

        _transitionMatrix = Tensor<T>.FromVector(transVec).Reshape(_transitionMatrix._shape);
        _startScores = Tensor<T>.FromVector(startVec).Reshape(_startScores._shape);
        _endScores = Tensor<T>.FromVector(endVec).Reshape(_endScores._shape);
    }

    /// <summary>
    /// Resets the internal state of the CRF layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the CRF layer, including the cached inputs, outputs,
    /// and parameter gradients. This is useful when starting to process a new sequence or batch of data.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's temporary memory to start fresh.
    /// 
    /// When resetting the state:
    /// - Stored inputs and outputs are cleared
    /// - Calculated gradients are cleared
    /// - The layer forgets any information from previous batches
    /// 
    /// This is important for:
    /// - Processing a new, unrelated batch of data
    /// - Preventing information from one batch affecting another
    /// - Freeing up memory that's no longer needed
    /// 
    /// Note that this doesn't reset the learned parameters (transition matrix, start scores, end scores),
    /// just the temporary information used during a single forward/backward pass.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _transitionMatrixGradient = null;
        _startScoresGradient = null;
        _endScoresGradient = null;
    }

}

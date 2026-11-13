using AiDotNet.Helpers;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;
using AiDotNet.Data.Structures;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Neural Turing Machine (NTM) for meta-learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// Neural Turing Machines augment neural networks with an external memory matrix
/// and differentiable attention mechanisms for reading and writing. This enables
/// algorithms to be learned and executed within the neural network itself.
/// </para>
/// <para><b>For Beginners:</b> NTM is like a neural computer with RAM:
///
/// **How it works:**
/// 1. Controller network processes inputs like a CPU
/// 2. Generates read/write keys for memory access
/// 3. Attention mechanism determines where to read/write
/// 4. External memory stores information persistently
/// 5. Differentiable operations allow end-to-end learning
///
/// **Key difference from standard NN:**
/// - Standard NN: Fixed computation graph
/// - NTM: Can learn to store and retrieve information dynamically
/// - Like giving a neural network a scratchpad to work with
/// </para>
/// <para><b>Algorithm - Neural Turing Machine:</b>
/// <code>
/// # Components
/// controller = LSTM() or MLP()      # Processes inputs and outputs
/// memory = MemoryMatrix(N x M)       # N locations, M dimensions each
/// read_heads = [ReadHead() x R]     # R parallel read heads
/// write_head = WriteHead()          # Single write head
///
/// # Forward pass
/// for each timestep t:
///     # Controller receives input and previous reads
///     controller_input = concatenate(x_t, read_contents_t-1)
///     controller_output = controller(controller_input)
///
///     # Generate read/write addressing
///     read_keys = controller.generate_read_keys(controller_output)
///     write_key = controller.generate_write_key(controller_output)
///     write_erase = controller.generate_erase_vector(controller_output)
///     write_add = controller.generate_add_vector(controller_output)
///
///     # Read from memory: content addressing (cosine similarity x key strength, softmax), interpolation
///     # with the previous weighting, circular shift and sharpening (Graves et al. 2014, eq. 5-9)
///     read_contents = []
///     for each read_head in read_heads:
///         weights = attention(read_head.key, memory)
///         content = weighted_sum(weights, memory)
///         read_contents.append(content)
///
///     # Write to memory
///     write_weights = attention(write_head.key, memory)
///     memory = memory * (1 - write_weights * write_erase)
///     memory = memory + write_weights * write_add
///
///     # Generate output
///     output = controller.generate_output(controller_output, read_contents)
/// </code>
/// </para>
/// <para><b>Key Insights:</b>
///
/// 1. **Differentiable Memory**: Both reading and writing use differentiable
///    attention, allowing the entire system to be trained with backpropagation.
///
/// 2. **Algorithmic Learning**: NTM can learn to implement algorithms like
///    sorting, copying, and associative recall directly from examples.
///
/// 3. **Variable Computation**: The computation graph can change based on
///    what's stored in memory, enabling dynamic reasoning.
///
/// 4. **Persistent State**: Information can be stored across timesteps,
///    enabling long-term memory and reasoning.
/// </para>
/// <para>
/// <b>Production Features:</b>
/// - LSTM or MLP controllers
/// - Multiple read/write heads
/// - Content-based and location-based addressing
/// - Memory initialization strategies
/// - Memory usage monitoring
/// - Differentiable memory operations
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
// Graves et al. supplies the ARCHITECTURE - controller, addressable memory, read and write heads - and nothing
// else: its tasks are copy, repeat-copy, associative recall and sorting, and it states no classification
// objective. The few-shot classification setting this class is exercised under, including the softmax output
// layer and the cross-entropy episode loss, is Santoro et al. 2016, which builds exactly this memory-augmented
// controller into a meta-learner. Both are recorded so the citation covers what the class actually does.
[ResearchPaper("Neural Turing Machines",
    "https://arxiv.org/abs/1410.5401",
    Year = 2014,
    Authors = "Alex Graves, Greg Wayne, Ivo Danihelka")]
[ResearchPaper("Meta-Learning with Memory-Augmented Neural Networks",
    "https://arxiv.org/abs/1605.06065",
    Year = 2016,
    Authors = "Adam Santoro, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, Timothy Lillicrap")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class NTMAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly NTMOptions<T, TInput, TOutput> _ntmOptions;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _ntmOptions;
    private readonly INTMController<T> _controller;
    private readonly NTMMemory<T> _memory;
    private readonly List<NTMReadHead<T>> _readHeads;
    private readonly NTMWriteHead<T> _writeHead;
    [Scratch]
    private List<Tensor<T>> _previousReadContents;

    // The one-hot label presented with the NEXT input (Santoro et al. 2016, section 3.1: x_t arrives with
    // y_{t-1}). Null presents a zero label.
    [Scratch]
    private Tensor<T>? _pendingLabel;

    /// <summary>
    /// Initializes a new instance of the NTMAlgorithm class.
    /// </summary>
    /// <param name="options">The configuration options for NTM.</param>
    /// <exception cref="ArgumentNullException">Thrown when options or required components are null.</exception>
    /// <exception cref="ArgumentException">Thrown when configuration validation fails.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a Neural Turing Machine ready for meta-learning:
    ///
    /// <b>What NTM needs:</b>
    /// - <b>controller:</b> LSTM or MLP that controls the system
    /// - <b>memorySize:</b> Size of external memory matrix
    /// - <b>memoryWidth:</b> Dimension of each memory location
    /// - <b>numReadHeads:</b> How many parallel read operations
    /// - <b>controllerType:</b> LSTM for sequences, MLP for fixed-size inputs
    ///
    /// <b>What makes it special:</b>
    /// - Can learn algorithms (sorting, copying) from data
    /// - Has external memory like RAM
    /// - Memory operations are differentiable
    /// - Can reason and plan using stored information
    /// </para>
    /// </remarks>
    public NTMAlgorithm(NTMOptions<T, TInput, TOutput> options)
        : base(
            options?.MetaModel ?? throw new ArgumentNullException(nameof(options), "MetaModel must be set in options."),
            // Graves et al. 2014 specifies no classification objective - its tasks are copy and recall - so the
            // loss comes from the few-shot setting this class is used in, which is Santoro et al. 2016: softmax
            // the controller output and minimise cross-entropy against the label (eq. 24-25). It used to
            // inherit the inner model's DefaultLossFunction, which is squared error. The caller's own
            // LossFunction still wins when supplied.
            options.LossFunction ?? new AiDotNet.LossFunctions.CrossEntropyWithLogitsLoss<T>(),
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer,
            optimizerUse: MetaLearnerOptimizerUse.Outer)
    {
        _ntmOptions = options;

        // Validate before building anything sized by the options: a negative shift range or class count would
        // otherwise fail inside a tensor constructor with a less useful exception.
        if (!_ntmOptions.IsValid())
        {
            throw new ArgumentException("NTM configuration is invalid. Check all parameters.", nameof(options));
        }

        // Initialize controller
        _controller = options.ControllerType switch
        {
            NTMControllerType.LSTM => new LSTMNTMController<T, TInput, TOutput>(options),
            NTMControllerType.MLP => new MLPNTMController<T, TInput, TOutput>(options),
            _ => throw new ArgumentException("Invalid controller type")
        };

        // Initialize memory
        _memory = new NTMMemory<T>(options.MemorySize, options.MemoryWidth, options.RandomSeed);

        // Initialize read heads
        _readHeads = new List<NTMReadHead<T>>();
        for (int i = 0; i < options.NumReadHeads; i++)
        {
            _readHeads.Add(new NTMReadHead<T>(options.MemoryWidth, options.MemorySize, i));
        }

        // Initialize write head
        _writeHead = new NTMWriteHead<T>(options.MemoryWidth, options.MemorySize);

        // Initialize previous read contents
        _previousReadContents = new List<Tensor<T>>();
        for (int i = 0; i < options.NumReadHeads; i++)
        {
            _previousReadContents.Add(new Tensor<T>(new int[] { options.MemoryWidth }));
        }

        // Initialize memory if configured
        if (_ntmOptions.InitializeMemory)
        {
            InitializeMemory();
        }
    }

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.NTM;

    /// <summary>The controller the algorithm trains, shared with every adapted model.</summary>
    internal INTMController<T> Controller => _controller;

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));
        }

        var episodeLosses = new List<Tensor<T>>();
        foreach (var task in taskBatch.Tasks)
        {
            // Reset memory state for new episode
            ResetMemoryState();

            // Train on this episode
            episodeLosses.Add(TrainEpisode(task));
        }

        if (episodeLosses.Count == 0)
        {
            return NumOps.Zero;
        }

        // Return average loss
        T totalLoss = Engine.TensorSum(Engine.TensorConcatenate(episodeLosses.ToArray(), 0));
        return NumOps.Divide(totalLoss, NumOps.FromDouble(taskBatch.BatchSize));
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The adapted model returns one score per class for each example, so this is the configured loss of their
    /// logarithm against the class indices - cross-entropy by default. A Vector output carries the predicted
    /// class of each example instead, and its loss is the classification error rate. The base's default compared
    /// the whole score block against a vector of labels, which cannot be lined up at all.
    /// </remarks>
    protected override T ComputeLossFromOutput(TOutput predictions, TOutput expectedOutput)
        => ClassifierOutputs<T>.ProbabilityLoss(LossFunction, predictions, expectedOutput);

    /// <inheritdoc />
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null)
        {
            throw new ArgumentNullException(nameof(task));
        }

        // Every episode starts from the initial memory, fresh head weightings and a reset controller (Santoro et
        // al. 2016, section 3: memory is wiped between episodes), exactly as MetaTrain's episodes do. Without
        // this the adapted model inherited the last training episode's memory and head weightings and whatever
        // controller state the previous Adapt or Predict left behind, so adapting to the same task twice gave
        // different predictions.
        ResetMemoryState();

        // For NTM, adaptation involves using the controller to write to memory
        // and returning a model that can continue using this memory state
        var adaptedModel = new NTMModel<T, TInput, TOutput>(
            _controller,
            _memory.Clone(),
            _readHeads.Select(rh => rh.Clone()).ToList(),
            _writeHead.Clone(),
            _ntmOptions);

        // Process support set to initialize memory
        ProcessSupportSet(task.SupportInput, task.SupportOutput, adaptedModel);

        return adaptedModel;
    }

    /// <summary>
    /// Trains the NTM on a single episode: one forward pass over the support then the query sequence on a
    /// gradient tape, and one outer step on the controller with the tape's gradient of the query loss.
    /// </summary>
    /// <returns>The episode loss before the update, as a one-element tensor.</returns>
    /// <remarks>
    /// Graves et al. 2014 train every NTM component end to end by backpropagation through time (section 4).
    /// This used to approximate the gradient by forward finite differences, re-running the whole episode once
    /// per controller parameter with a step of 1e-5, which is both O(parameters) episodes per update and too
    /// coarse to resolve in single precision.
    /// </remarks>
    private Tensor<T> TrainEpisode(IMetaLearningTask<T, TInput, TOutput> task)
    {
        Tensor<T> episodeLoss;
        IReadOnlyList<Tensor<T>> parameters;
        Dictionary<Tensor<T>, Tensor<T>> tapeGradients;
        using (var tape = new GradientTape<T>())
        {
            // Process support set to initialize memory
            ProcessSupportSet(task.SupportInput, task.SupportOutput, null);

            // Process query set
            var queryScores = ProcessSequence(task.QueryInput);

            // Compute loss, plus memory regularization
            episodeLoss = AddMemoryRegularizationTerms(ComputeLoss(queryScores, task.QueryOutput));

            // Collected after the forward pass: the LSTM controller resolves its input-gate weights to the real
            // input width at its first step, so the tensors the episode used are only known now.
            parameters = _controller.GetParameterTensors();
            tapeGradients = tape.ComputeGradients(episodeLoss, parameters);
        }

        // Backpropagate through time and update all components
        UpdateComponents(FlattenGradients(parameters, tapeGradients));

        return episodeLoss;
    }

    /// <summary>
    /// Processes support set to initialize memory state.
    /// </summary>
    /// <param name="supportInputs">The support set inputs.</param>
    /// <param name="supportOutputs">The support set outputs.</param>
    /// <param name="model">Optional model to use for processing. If null, uses the algorithm's own components.</param>
    private void ProcessSupportSet(TInput supportInputs, TOutput supportOutputs, NTMModel<T, TInput, TOutput>? model)
    {
        // Convert inputs to sequence format
        var inputSequence = ConvertToSequence(supportInputs);
        var labelSequence = _ntmOptions.PresentPreviousLabel
            ? ConvertOutputToSequence(supportOutputs, inputSequence.Length)
            : null;

        // Process each time step - use the model's components if provided
        for (int t = 0; t < inputSequence.Length; t++)
        {
            var label = labelSequence?[t];
            if (model != null)
            {
                // Process through the model to prime its memory
                model.ProcessTimestepInternal(inputSequence[t], label);
            }
            else
            {
                ProcessTimestep(inputSequence[t], label);
            }
        }
    }

    /// <summary>
    /// Processes a sequence of query inputs, whose labels are not shown to the controller.
    /// </summary>
    /// <returns>The controller's class scores, one row per step: <c>[steps, NumClasses]</c>.</returns>
    /// <remarks>
    /// This used to return the last step's scores only, which the loss then lined up against the whole label
    /// tensor, so every query example but the last went unscored.
    /// </remarks>
    private Tensor<T> ProcessSequence(TInput inputs)
    {
        // Convert to sequence format
        var inputSequence = ConvertToSequence(inputs);
        int classes = _ntmOptions.NumClasses;
        if (inputSequence.Length == 0)
        {
            return new Tensor<T>(new int[] { 0, classes });
        }

        // Process each time step
        var rows = new Tensor<T>[inputSequence.Length];
        for (int t = 0; t < inputSequence.Length; t++)
        {
            var output = ProcessTimestep(inputSequence[t], null);
            rows[t] = Engine.Reshape(NTMSequences<T>.FitLength(output, classes), new[] { 1, classes });
        }

        return Engine.TensorConcatenate(rows, 0);
    }

    /// <summary>
    /// Processes a single timestep.
    /// </summary>
    /// <param name="input">The step's input features.</param>
    /// <param name="label">The step's one-hot label, presented with the next input; null when it is unknown.</param>
    private Tensor<T> ProcessTimestep(Tensor<T> input, Tensor<T>? label)
    {
        // Combine input (and the previous label) with previous read contents
        var controllerInput = CombineInputWithReadContents(PresentLabel(input, label));

        // Forward pass through controller
        var controllerOutput = _controller.Forward(controllerInput, _previousReadContents);

        // Generate addressing parameters (Graves et al. 2014, section 3.3)
        var readAddressing = _controller.GenerateReadAddressing(controllerOutput);
        var writeAddressing = _controller.GenerateWriteAddressing(controllerOutput);
        var eraseVector = _controller.GenerateEraseVector(controllerOutput);
        var addVector = _controller.GenerateAddVector(controllerOutput);

        // Read from memory using all read heads
        var currentReadContents = new List<Tensor<T>>();
        for (int i = 0; i < _readHeads.Count; i++)
        {
            var readWeights = _readHeads[i].ComputeReadWeights(readAddressing[i], _memory);
            currentReadContents.Add(_memory.Read(readWeights));
        }

        // Write to memory
        var writeWeights = _writeHead.ComputeWriteWeights(writeAddressing, _memory);
        _memory.Write(writeWeights, eraseVector, addVector);

        // Generate output
        var output = _controller.GenerateOutput(controllerOutput, currentReadContents);

        // Update previous read contents for next timestep
        _previousReadContents = currentReadContents;

        return output;
    }

    /// <summary>
    /// Appends the pending label to the input and queues this step's label for the next one, so the controller
    /// sees <c>(x_t, y_{t-1})</c> (Santoro et al. 2016, section 3.1). A no-op when label presentation is off.
    /// </summary>
    private Tensor<T> PresentLabel(Tensor<T> input, Tensor<T>? label)
    {
        if (!_ntmOptions.PresentPreviousLabel)
        {
            return input;
        }

        var presented = NTMSequences<T>.WithLabel(input, _pendingLabel, _ntmOptions.NumClasses);
        _pendingLabel = label;
        return presented;
    }

    /// <summary>
    /// Combines input with previous read contents.
    /// </summary>
    private Tensor<T> CombineInputWithReadContents(Tensor<T> input)
    {
        // Concatenate input with flattened read contents
        var pieces = new[] { FlattenTensor(input) }
            .Concat(_previousReadContents.Select(FlattenTensor))
            .ToArray();
        return Engine.TensorConcatenate(pieces, 0);
    }

    /// <summary>
    /// Resets the memory state for a new episode.
    /// </summary>
    private void ResetMemoryState()
    {
        // Reset memory to initial state
        if (_ntmOptions.InitializeMemory)
        {
            InitializeMemory();
        }
        else
        {
            _memory.Reset();
        }

        // Reset read contents
        for (int i = 0; i < _previousReadContents.Count; i++)
        {
            _previousReadContents[i] = new Tensor<T>(new int[] { _ntmOptions.MemoryWidth });
        }

        // Reset each head's previous weighting w_{t-1} and the pending label
        foreach (var readHead in _readHeads)
        {
            readHead.Reset();
        }

        _writeHead.Reset();
        _pendingLabel = null;

        // Reset controller hidden state
        _controller.Reset();
    }

    /// <summary>
    /// Initializes memory with default values.
    /// </summary>
    private void InitializeMemory()
    {
        switch (_ntmOptions.MemoryInitialization)
        {
            case NTMMemoryInitialization.Zeros:
                _memory.InitializeZeros();
                break;
            case NTMMemoryInitialization.Random:
                _memory.InitializeRandom();
                break;
            case NTMMemoryInitialization.Learned:
                _memory.InitializeLearned();
                break;
            default:
                _memory.InitializeZeros();
                break;
        }
    }

    /// <summary>
    /// Adds the memory regularization terms to a loss tensor, on the tape so they shape the controller update.
    /// </summary>
    private Tensor<T> AddMemoryRegularizationTerms(Tensor<T> baseLoss)
    {
        var totalLoss = Engine.Reshape(baseLoss, new[] { 1 });

        // Add memory usage regularization
        if (_ntmOptions.MemoryUsageRegularization > 0.0)
        {
            var usage = _memory.UsagePenaltyTensor();
            totalLoss = Engine.TensorAdd(totalLoss,
                Engine.TensorMultiplyScalar(usage, NumOps.FromDouble(_ntmOptions.MemoryUsageRegularization)));
        }

        // Add memory sharpness regularization
        if (_ntmOptions.MemorySharpnessRegularization > 0.0)
        {
            var sharpness = _memory.SharpnessPenaltyTensor();
            totalLoss = Engine.TensorAdd(totalLoss,
                Engine.TensorMultiplyScalar(sharpness, NumOps.FromDouble(_ntmOptions.MemorySharpnessRegularization)));
        }

        return totalLoss;
    }

    /// <summary>
    /// Updates the controller with one outer step along the episode's tape gradient.
    /// </summary>
    private void UpdateComponents(Vector<T> gradients)
    {
        // Get current controller parameters
        var controllerParams = _controller.GetParameters();
        if (controllerParams.Length == 0)
        {
            return;
        }

        // Apply gradient clipping if configured
        if (_options.GradientClipThreshold.HasValue && _options.GradientClipThreshold.Value > 0)
        {
            gradients = ClipGradients(gradients, _options.GradientClipThreshold.Value);
        }

        // Update parameters using learning rate
        var updatedParams = ApplyOuterGradients(controllerParams, gradients, _options.OuterLearningRate);

        // Apply updated parameters to controller
        SetControllerParameters(updatedParams);
    }

    /// <summary>
    /// The tape's gradient of every controller parameter tensor, flattened in <see cref="INTMController{T}.GetParameters"/>
    /// order. A tensor the loss does not reach has a zero gradient.
    /// </summary>
    private Vector<T> FlattenGradients(IReadOnlyList<Tensor<T>> parameters, Dictionary<Tensor<T>, Tensor<T>> gradients)
    {
        if (parameters.Count == 0)
        {
            return new Vector<T>(0);
        }

        var pieces = new Tensor<T>[parameters.Count];
        for (int i = 0; i < parameters.Count; i++)
        {
            var parameter = parameters[i];
            pieces[i] = gradients.TryGetValue(parameter, out var gradient)
                ? Engine.Reshape(gradient, new[] { parameter.Length })
                : new Tensor<T>(new[] { parameter.Length });
        }

        return Engine.TensorConcatenate(pieces, 0).ToVector();
    }

    /// <summary>
    /// Sets controller parameters (updates internal weights).
    /// </summary>
    private void SetControllerParameters(Vector<T> parameters)
    {
        // Update the controller's internal weights directly
        if (_controller != null && parameters.Length > 0)
        {
            _controller.SetParameters(parameters);
        }

        // Also update the meta model with the new parameters for consistency
        if (MetaModel != null && parameters.Length > 0)
        {
            var currentParams = InterfaceGuard.Parameterizable(MetaModel).GetParameters();
            int copyLength = Math.Min(parameters.Length, currentParams.Length);
            parameters.AsSpan().Slice(0, copyLength).CopyTo(currentParams.AsWritableSpan());
            InterfaceGuard.Parameterizable(MetaModel).SetParameters(currentParams);
        }
    }

    // Helper methods

    private Tensor<T>[] ConvertToSequence(TInput inputs) => NTMSequences<T>.Steps(inputs);

    /// <summary>
    /// The labels as one one-hot row per timestep, matching <see cref="ConvertToSequence"/>'s one row per step.
    /// </summary>
    private Tensor<T>[] ConvertOutputToSequence(TOutput outputs, int steps)
        => NTMSequences<T>.SplitRows(NTMSequences<T>.OneHotRows(outputs, steps, _ntmOptions.NumClasses));

    private Tensor<T> FlattenTensor(Tensor<T> tensor) => NTMSequences<T>.Flatten(tensor);

    /// <summary>
    /// The episode loss: the configured loss (cross-entropy with logits by default) of the query scores against
    /// the one-hot query labels, Santoro et al. 2016 eq. 24-25. It used to be the squared error between the last
    /// step's scores and the first few label entries, which ignored the configured loss.
    /// </summary>
    private Tensor<T> ComputeLoss(Tensor<T> predictions, TOutput targets)
    {
        int rows = predictions.Shape.Length > 0 ? predictions.Shape[0] : 0;
        var oneHot = NTMSequences<T>.OneHotRows(targets, rows, _ntmOptions.NumClasses);
        return LossFunction.ComputeTapeLoss(predictions, oneHot);
    }
}

/// <summary>
/// Sequence and label layout shared by the NTM algorithm and its adapted model.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
internal static class NTMSequences<T>
{
    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>A tensor as a rank-1 tensor of all its elements.</summary>
    internal static Tensor<T> Flatten(Tensor<T> tensor) => Engine.Reshape(tensor, new[] { tensor.Length });

    /// <summary>
    /// A tensor's elements cut or zero-padded to <paramref name="length"/>: the first <paramref name="length"/>
    /// elements, then zeros.
    /// </summary>
    internal static Tensor<T> FitLength(Tensor<T> tensor, int length)
    {
        var flat = Flatten(tensor);
        if (flat.Length == length)
        {
            return flat;
        }

        if (flat.Length > length)
        {
            return Engine.TensorNarrow(flat, 0, 0, length);
        }

        return Engine.TensorConcatenate(new[] { flat, new Tensor<T>(new[] { length - flat.Length }) }, 0);
    }

    /// <summary>The input followed by the presented one-hot label (zeros when none is pending).</summary>
    internal static Tensor<T> WithLabel(Tensor<T> input, Tensor<T>? label, int numClasses)
    {
        var presented = label is null ? new Tensor<T>(new[] { numClasses }) : FitLength(label, numClasses);
        return Engine.TensorConcatenate(new[] { Flatten(input), presented }, 0);
    }

    /// <summary>The rows of a <c>[rows, features]</c> tensor as rank-1 tensors.</summary>
    internal static Tensor<T>[] SplitRows(Tensor<T> rows)
    {
        int count = rows.Shape[0];
        int width = count == 0 ? 0 : rows.Length / count;
        var result = new Tensor<T>[count];
        for (int r = 0; r < count; r++)
        {
            result[r] = Engine.Reshape(Engine.TensorNarrow(rows, 0, r, 1), new[] { width });
        }

        return result;
    }

    /// <summary>
    /// An input as one feature vector per timestep. A Matrix&lt;T&gt; [rows, cols] is the common meta-learning
    /// support/query shape: each row is one timestep and its <c>cols</c> entries are that step's feature vector.
    /// Handling it here (and identically in the model's input conversion) is what keeps the controller's
    /// assembled input width stable across the train-to-predict flow.
    /// </summary>
    internal static Tensor<T>[] Steps(object? inputs)
    {
        if (inputs is Matrix<T> matrix)
        {
            // Fail fast on empty matrices: a zero-row/column input would produce a
            // degenerate (empty or width-1) sequence and silently corrupt the
            // controller's learned input width later, rather than surfacing the
            // misconfiguration here.
            if (matrix.Rows <= 0 || matrix.Columns <= 0)
                throw new ArgumentException(
                    $"Matrix input must have positive dimensions for NTM sequence conversion; " +
                    $"got [{matrix.Rows} x {matrix.Columns}].",
                    nameof(inputs));

            return SplitRows(Tensor<T>.FromMatrix(matrix));
        }

        if (inputs is Tensor<T> tensor)
        {
            if (tensor.Shape.Length == 1)
            {
                return new Tensor<T>[] { tensor };
            }

            if (tensor.Shape.Length >= 2)
            {
                int sequenceLength = tensor.Shape[0];
                if (sequenceLength == 0)
                {
                    return new Tensor<T>[0];
                }

                return SplitRows(Engine.Reshape(tensor, new[] { sequenceLength, tensor.Length / sequenceLength }));
            }
        }

        return new Tensor<T>[] { new Tensor<T>(new int[] { 1 }) };
    }

    /// <summary>
    /// Labels as <c>[rows, numClasses]</c> one-hot rows. Accepts class indices (one per row), one-hot or
    /// probability rows <c>[rows, numClasses]</c>, or those rows flattened.
    /// </summary>
    /// <exception cref="NotSupportedException">The labels are not a Tensor, Vector or Matrix.</exception>
    /// <exception cref="ArgumentException">The labels cannot be lined up with <paramref name="rows"/> steps.</exception>
    internal static Tensor<T> OneHotRows(object? labels, int rows, int numClasses)
    {
        Tensor<T> values = labels switch
        {
            Tensor<T> tensor => tensor,
            Vector<T> vector => Tensor<T>.FromVector(vector),
            Matrix<T> matrix => Tensor<T>.FromMatrix(matrix),
            _ => throw new NotSupportedException(
                $"NTM labels must be a Tensor<T>, Vector<T> or Matrix<T>, not '{labels?.GetType().Name ?? "null"}'."),
        };

        if (values.Shape.Length == 2 && values.Shape[0] == rows && values.Shape[1] == numClasses)
        {
            return values;
        }

        if (values.Length == rows)
        {
            // One class index per step.
            var indices = Engine.Reshape(ClassifierOutputs<T>.Labels(values, numClasses), new[] { rows, 1 });
            var classes = Engine.Reshape(Engine.TensorArange<T>(0, numClasses, 1), new[] { 1, numClasses });
            return Engine.TensorEquals(
                Engine.TensorBroadcastTo(indices, new[] { rows, numClasses }),
                Engine.TensorBroadcastTo(classes, new[] { rows, numClasses }));
        }

        if (values.Length == rows * numClasses)
        {
            return Engine.Reshape(values, new[] { rows, numClasses });
        }

        throw new ArgumentException(
            $"{values.Length} label values match neither {rows} class indices nor {rows} one-hot rows of "
            + $"{numClasses} classes.", nameof(labels));
    }
}

/// <summary>
/// The addressing parameters one NTM head emits at one step (Graves et al. 2014, section 3.3).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// A head finds its weighting over memory locations in four stages: content addressing by cosine similarity to
/// <see cref="Key"/>, sharpened by <see cref="KeyStrength"/> (eq. 5-6); interpolation with the previous
/// weighting by <see cref="Gate"/> (eq. 7); a circular convolution with <see cref="Shift"/> (eq. 8); and
/// sharpening by <see cref="Sharpening"/> (eq. 9).
/// </para>
/// </remarks>
public sealed class NTMAddressingParameters<T>
{
    /// <summary>
    /// Initializes the addressing parameters of one head.
    /// </summary>
    /// <param name="key">The key vector k_t, compared with every memory row.</param>
    /// <param name="keyStrength">The key strength beta_t &gt;= 0, one element.</param>
    /// <param name="gate">The interpolation gate g_t in (0, 1), one element.</param>
    /// <param name="shift">The shift weighting s_t over the integer offsets -R..R, an odd number of elements summing to one.</param>
    /// <param name="sharpening">The sharpening exponent gamma_t &gt;= 1, one element.</param>
    /// <exception cref="ArgumentNullException">A parameter is null.</exception>
    /// <exception cref="ArgumentException">A scalar parameter does not hold one element, or the shift weighting covers an even number of offsets.</exception>
    public NTMAddressingParameters(Tensor<T> key, Tensor<T> keyStrength, Tensor<T> gate, Tensor<T> shift, Tensor<T> sharpening)
    {
        Guard.NotNull(key);
        Guard.NotNull(keyStrength);
        Guard.NotNull(gate);
        Guard.NotNull(shift);
        Guard.NotNull(sharpening);
        if (keyStrength.Length != 1) throw new ArgumentException("The key strength is one value.", nameof(keyStrength));
        if (gate.Length != 1) throw new ArgumentException("The interpolation gate is one value.", nameof(gate));
        if (sharpening.Length != 1) throw new ArgumentException("The sharpening exponent is one value.", nameof(sharpening));
        if (shift.Length % 2 == 0)
            throw new ArgumentException("The shift weighting covers the offsets -R..R, an odd number of them.", nameof(shift));

        Key = key;
        KeyStrength = keyStrength;
        Gate = gate;
        Shift = shift;
        Sharpening = sharpening;
    }

    /// <summary>Gets the key vector k_t.</summary>
    public Tensor<T> Key { get; }

    /// <summary>Gets the key strength beta_t.</summary>
    public Tensor<T> KeyStrength { get; }

    /// <summary>Gets the interpolation gate g_t.</summary>
    public Tensor<T> Gate { get; }

    /// <summary>Gets the shift weighting s_t over the offsets -R..R.</summary>
    public Tensor<T> Shift { get; }

    /// <summary>Gets the sharpening exponent gamma_t.</summary>
    public Tensor<T> Sharpening { get; }

    /// <summary>
    /// Pure content addressing: beta = 1, g = 1, no shift and gamma = 1, so the weighting is the softmax of the
    /// cosine similarities to <paramref name="key"/>.
    /// </summary>
    /// <param name="key">The key vector.</param>
    /// <param name="shiftRange">The shift range R; the shift weighting puts all its mass on offset 0.</param>
    public static NTMAddressingParameters<T> ContentOnly(Tensor<T> key, int shiftRange = 1)
    {
        if (shiftRange < 0) throw new ArgumentOutOfRangeException(nameof(shiftRange), "The shift range cannot be negative.");
        var ops = MathHelper.GetNumericOperations<T>();
        var shift = new Tensor<T>(new[] { 2 * shiftRange + 1 });
        shift[shiftRange] = ops.One;
        return new NTMAddressingParameters<T>(
            key,
            Tensor<T>.CreateDefault(new[] { 1 }, ops.One),
            Tensor<T>.CreateDefault(new[] { 1 }, ops.One),
            shift,
            Tensor<T>.CreateDefault(new[] { 1 }, ops.One));
    }
}

/// <summary>
/// NTM model for inference with persistent memory.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
public class NTMModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private static IEngine Engine => AiDotNetEngine.Current;

    private readonly INTMController<T> _controller;
    private readonly NTMMemory<T> _memory;
    private readonly List<NTMReadHead<T>> _readHeads;
    private readonly NTMWriteHead<T> _writeHead;
    private readonly NTMOptions<T, TInput, TOutput> _options;
    private readonly List<Tensor<T>> _readContents;

    // The one-hot label presented with the next input (Santoro et al. 2016: x_t arrives with y_{t-1}).
    private Tensor<T>? _pendingLabel;

    // This model's own controller recurrent state. The controller's weights are shared with the algorithm and
    // every other adapted model, so its hidden state is swapped in before each step and saved after it;
    // otherwise adapting a second task overwrote the state the first adapted model would continue from.
    private object? _controllerState;

    /// <summary>
    /// Initializes a new instance of the NTMModel class.
    /// </summary>
    public NTMModel(
        INTMController<T> controller,
        NTMMemory<T> memory,
        List<NTMReadHead<T>> readHeads,
        NTMWriteHead<T> writeHead,
        NTMOptions<T, TInput, TOutput> options)
    {
        Guard.NotNull(controller);
        _controller = controller;
        Guard.NotNull(memory);
        _memory = memory;
        Guard.NotNull(readHeads);
        _readHeads = readHeads;
        Guard.NotNull(writeHead);
        _writeHead = writeHead;
        Guard.NotNull(options);
        _options = options;

        // Initialize read contents
        _readContents = new List<Tensor<T>>();
        for (int i = 0; i < readHeads.Count; i++)
        {
            _readContents.Add(new Tensor<T>(new int[] { options.MemoryWidth }));
        }

        _controllerState = (controller as INTMControllerState)?.CaptureState();
    }

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <inheritdoc/>
    /// <remarks>
    /// Returns one row of class probabilities per example, <c>[rows, NumClasses]</c>: the softmax of the
    /// controller output, which is how Santoro et al. 2016 read a class distribution off the memory-augmented
    /// controller (eq. 24). It used to return the raw scores, which the meta-learner's loss then took the
    /// logarithm of as if they were probabilities.
    /// </remarks>
    public TOutput Predict(TInput input)
    {
        // One score row per example, stepping per example: the memory the NTM reads and writes evolves across
        // the timesteps.
        int count = MbPAConversions<T>.GetBatchSize(input);
        int classes = _options.NumClasses;
        if (count <= 0)
        {
            return ClassifierOutputs<T>.ToOutput<TOutput>(new Tensor<T>(new int[] { 0, classes }));
        }

        using var noGrad = new NoGradScope<T>();
        var rows = new Tensor<T>[count];
        for (int r = 0; r < count; r++)
        {
            var single = MbPAConversions<T>.SliceExample(input, r);
            var stepOutput = PredictTimestep(ConvertInputToTensor(single));
            rows[r] = Engine.Reshape(NTMSequences<T>.FitLength(stepOutput, classes), new[] { 1, classes });
        }

        var probabilities = Engine.Softmax(Engine.TensorConcatenate(rows, 0), -1);
        return ClassifierOutputs<T>.ToOutput<TOutput>(probabilities);
    }

    /// <summary>One timestep: address memory, read, write, emit, and carry the read contents forward.</summary>
    private Tensor<T> PredictTimestep(Tensor<T> inputTensor)
    {
        var (controllerOutput, currentReadContents) = Advance(inputTensor, null);

        // Generate output
        var output = _controller.GenerateOutput(controllerOutput, currentReadContents);

        // Update read contents for next call
        KeepReadContents(currentReadContents);

        return output;
    }

    /// <summary>
    /// Processes a single timestep using the model's internal components.
    /// This is used during adaptation to prime the model's memory.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="label">The step's one-hot label, presented with the next input; null when it is unknown.</param>
    internal void ProcessTimestepInternal(Tensor<T> input, Tensor<T>? label)
    {
        var (_, currentReadContents) = Advance(input, label);

        // Update read contents for next timestep
        KeepReadContents(currentReadContents);
    }

    /// <summary>
    /// Runs the controller on one input, then reads and writes memory with the heads' addressing.
    /// </summary>
    private (Tensor<T> ControllerOutput, List<Tensor<T>> ReadContents) Advance(Tensor<T> input, Tensor<T>? label)
    {
        // Combine input (and the previous label) with previous read contents
        var controllerInput = CombineInputWithReadContents(PresentLabel(input, label));

        // Forward pass through controller, from this model's own recurrent state
        var stateful = _controller as INTMControllerState;
        if (stateful is not null && _controllerState is not null)
        {
            stateful.RestoreState(_controllerState);
        }

        var controllerOutput = _controller.Forward(controllerInput, _readContents);
        _controllerState = stateful?.CaptureState();

        // Generate addressing parameters
        var readAddressing = _controller.GenerateReadAddressing(controllerOutput);
        var writeAddressing = _controller.GenerateWriteAddressing(controllerOutput);
        var eraseVector = _controller.GenerateEraseVector(controllerOutput);
        var addVector = _controller.GenerateAddVector(controllerOutput);

        // Read from memory using all read heads
        var currentReadContents = new List<Tensor<T>>();
        for (int i = 0; i < _readHeads.Count; i++)
        {
            var readWeights = _readHeads[i].ComputeReadWeights(readAddressing[i], _memory);
            currentReadContents.Add(_memory.Read(readWeights));
        }

        // Write to memory
        var writeWeights = _writeHead.ComputeWriteWeights(writeAddressing, _memory);
        _memory.Write(writeWeights, eraseVector, addVector);

        return (controllerOutput, currentReadContents);
    }

    private void KeepReadContents(List<Tensor<T>> currentReadContents)
    {
        for (int i = 0; i < _readContents.Count && i < currentReadContents.Count; i++)
        {
            _readContents[i] = currentReadContents[i];
        }
    }

    /// <summary>Appends the pending label and queues this step's (Santoro et al. 2016, section 3.1).</summary>
    private Tensor<T> PresentLabel(Tensor<T> input, Tensor<T>? label)
    {
        if (!_options.PresentPreviousLabel)
        {
            return input;
        }

        var presented = NTMSequences<T>.WithLabel(input, _pendingLabel, _options.NumClasses);
        _pendingLabel = label;
        return presented;
    }

    /// <summary>
    /// Converts input to tensor format.
    /// </summary>
    private Tensor<T> ConvertInputToTensor(TInput input)
    {
        if (input is Tensor<T> tensor)
        {
            return tensor;
        }

        if (input is Vector<T> vector)
        {
            return Tensor<T>.FromVector(vector);
        }

        // A Matrix<T> [rows, cols] feature vector: take the first row's `cols`
        // features. This MUST match the training sequence's per-row featureSize (cols)
        // so the controller's assembled input width is identical on the train
        // and predict paths — otherwise the input-gate weights silently resize
        // between training and inference.
        if (input is Matrix<T> matrix && matrix.Rows > 0)
        {
            return Engine.Reshape(Engine.TensorNarrow(Tensor<T>.FromMatrix(matrix), 0, 0, 1), new[] { matrix.Columns });
        }

        return new Tensor<T>(new int[] { _options.MemoryWidth });
    }

    /// <summary>
    /// Combines input with previous read contents.
    /// </summary>
    private Tensor<T> CombineInputWithReadContents(Tensor<T> input)
    {
        var pieces = new[] { NTMSequences<T>.Flatten(input) }
            .Concat(_readContents.Select(NTMSequences<T>.Flatten))
            .ToArray();
        return Engine.TensorConcatenate(pieces, 0);
    }

    /// <inheritdoc/>
    public void Train(TInput inputs, TOutput targets)
    {
        throw new NotSupportedException("Use the training algorithm to train NTM.");
    }

    /// <inheritdoc/>
    public void UpdateParameters(Vector<T> parameters)
    {
        throw new NotSupportedException("NTM parameters are updated during training.");
    }

    /// <inheritdoc/>
    public Vector<T> GetParameters()
    {
        return _controller.GetParameters();
    }

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata()
    {
        return Metadata;
    }
}

/// <summary>
/// External memory matrix for Neural Turing Machine.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// Every write replaces the contents with a new tensor rather than mutating them, so a gradient tape keeps each
/// step's memory and backpropagates through the whole episode.
/// </remarks>
public class NTMMemory<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private static IEngine Engine => AiDotNetEngine.Current;

    // Content-addressing denominator floor on the norms (eq. 6 has none; it guards zero rows and keys). It is the
    // square root of the 1e-8 floor the squared norms have always had, and a normal Half value.
    private const double NormFloor = 1e-4;

    // Floor applied before the sharpening power (eq. 9), so log(0) never reaches the tape.
    private const double SharpenFloor = 1e-16;

    // Floors inside the row-norm computation: the row scale and the scaled sum of squares of an all-zero row.
    private const double ScaleFloor = 1e-12;
    private const double SquareFloor = 1e-24;

    // The smallest positive normal Half. A floor below it flushes to zero in Half arithmetic, which turns the
    // guards above into 0/0 and log(0).
    private const double HalfSmallestNormal = 6.103515625e-5;

    /// <summary>A guard constant in T that stays positive in Half (Half flushes 1e-8 and smaller to zero).</summary>
    private static T Floor(double epsilon)
        => NumOps.FromDouble(typeof(T) == typeof(Half) ? Math.Max(epsilon, HalfSmallestNormal) : epsilon);

    /// <summary>
    /// The L2 norm of every row of a <c>[rows, width]</c> tensor, <c>[rows]</c>. Each row is divided by its
    /// largest magnitude before squaring and the norm scaled back, so squares of large entries never overflow
    /// (Half saturates at 65504, so a row entry of 256 already did).
    /// </summary>
    private static Tensor<T> RowNorms(Tensor<T> rows)
    {
        int count = rows.Shape[0];
        var scale = Engine.TensorClampMin(Engine.ReduceMax(Engine.TensorAbs(rows), new[] { 1 }, true), Floor(ScaleFloor));
        var scaled = Engine.TensorMultiply(rows, Engine.TensorReciprocal(scale));
        var sumSquares = Engine.ReduceSum(Engine.TensorMultiply(scaled, scaled), new[] { 1 }, false);
        return Engine.TensorMultiply(
            Engine.Reshape(scale, new[] { count }),
            Engine.TensorSqrt(Engine.TensorClampMin(sumSquares, Floor(SquareFloor))));
    }

    /// <summary>The rows of a <c>[rows, width]</c> tensor divided by their norms, floored at <see cref="NormFloor"/>.</summary>
    private static Tensor<T> UnitRows(Tensor<T> rows)
    {
        var norms = Engine.TensorClampMin(RowNorms(rows), Floor(NormFloor));
        return Engine.TensorMultiply(rows, Engine.Reshape(Engine.TensorReciprocal(norms), new[] { rows.Shape[0], 1 }));
    }

    private Tensor<T> _contents;
    private readonly int _size;
    private readonly int _width;
    private readonly int? _randomSeed;
    private readonly Random _random;
    [Scratch]
    private Tensor<T>? _lastWriteWeights;  // Track last write weights for sharpness computation

    /// <summary>
    /// Initializes a new instance of NTMMemory.
    /// </summary>
    /// <param name="size">Number of memory slots.</param>
    /// <param name="width">Dimension of each memory slot.</param>
    /// <param name="randomSeed">Optional random seed for reproducibility.</param>
    public NTMMemory(int size, int width, int? randomSeed = null)
    {
        _size = size;
        _width = width;
        _randomSeed = randomSeed;
        _random = randomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(randomSeed.Value)
            : RandomHelper.CreateSecureRandom();
        _contents = new Tensor<T>(new int[] { size, width });
    }

    /// <summary>
    /// Gets the size (number of slots) of the memory.
    /// </summary>
    public int Size => _size;

    /// <summary>
    /// Gets the width (dimension) of each memory slot.
    /// </summary>
    public int Width => _width;

    /// <summary>
    /// Gets the memory matrix M_t, <c>[Size, Width]</c>.
    /// </summary>
    public Tensor<T> Contents => _contents;

    /// <summary>
    /// Computes a head's weighting over the memory locations (Graves et al. 2014, eq. 5-9).
    /// </summary>
    /// <param name="parameters">The head's addressing parameters for this step.</param>
    /// <param name="previousWeights">The head's previous weighting w_{t-1}, <c>[Size]</c>.</param>
    /// <returns>The weighting w_t, <c>[Size]</c>, non-negative and summing to one.</returns>
    /// <remarks>
    /// <para>
    /// Content addressing (eq. 5-6): w^c_t(i) = softmax_i(beta_t K[k_t, M_t(i)]) with K the cosine similarity.
    /// Interpolation (eq. 7): w^g_t = g_t w^c_t + (1 - g_t) w_{t-1}. Convolutional shift (eq. 8):
    /// w~_t(i) = sum_j w^g_t(j) s_t(i - j), indices modulo Size. Sharpening (eq. 9):
    /// w_t(i) = w~_t(i)^gamma_t / sum_j w~_t(j)^gamma_t.
    /// </para>
    /// <para>
    /// The key is compared with the first <c>min(Width, key length)</c> columns, as the content similarity has
    /// always been.
    /// </para>
    /// </remarks>
    public Tensor<T> Address(NTMAddressingParameters<T> parameters, Tensor<T> previousWeights)
    {
        Guard.NotNull(parameters);
        Guard.NotNull(previousWeights);

        var key = NTMSequences<T>.Flatten(parameters.Key);
        int width = Math.Min(_width, key.Length);
        if (key.Length != width)
        {
            key = Engine.TensorNarrow(key, 0, 0, width);
        }

        var memory = width == _width ? _contents : Engine.TensorNarrow(_contents, 1, 0, width);

        // K[k, M(i)] = k . M(i) / (||k|| ||M(i)||)  (eq. 6), as the dot product of the unit vectors so neither
        // the squares nor the dot product of large entries can overflow.
        var unitKey = Engine.Reshape(UnitRows(Engine.Reshape(key, new[] { 1, width })), new[] { width, 1 });
        var similarity = Engine.Reshape(Engine.TensorMatMul(UnitRows(memory), unitKey), new[] { _size });

        // w^c = softmax(beta K)  (eq. 5)
        var content = Engine.Softmax(
            Engine.TensorMultiply(similarity, NTMSequences<T>.Flatten(parameters.KeyStrength)), -1);

        // w^g = g w^c + (1 - g) w_{t-1}  (eq. 7)
        var gate = NTMSequences<T>.Flatten(parameters.Gate);
        var oneMinusGate = Engine.TensorAddScalar(Engine.TensorNegate(gate), NumOps.One);
        var gated = Engine.TensorAdd(
            Engine.TensorMultiply(content, gate),
            Engine.TensorMultiply(NTMSequences<T>.Flatten(previousWeights), oneMinusGate));

        // w~(i) = sum_j w^g(j) s(i - j)  (eq. 8)
        var shifted = CircularConvolution(gated, NTMSequences<T>.Flatten(parameters.Shift));

        // w(i) = w~(i)^gamma / sum_j w~(j)^gamma  (eq. 9). The ratio is unchanged by dividing w~ by its largest
        // entry first, which keeps that entry's power at 1: otherwise a large gamma underflows every power to
        // zero and the normalization is 0/0.
        var sharpenFloor = Floor(SharpenFloor);
        var peak = Engine.TensorClampMin(Engine.ReduceMax(shifted, new[] { 0 }, true), sharpenFloor);
        var relative = Engine.TensorMultiply(shifted, Engine.TensorReciprocal(peak));
        var powered = Engine.TensorExp(Engine.TensorMultiply(
            Engine.TensorLog(Engine.TensorClampMin(relative, sharpenFloor)),
            NTMSequences<T>.Flatten(parameters.Sharpening)));
        return Engine.TensorMultiply(powered, Engine.TensorReciprocal(Engine.ReduceSum(powered, new[] { 0 }, true)));
    }

    /// <summary>
    /// Circular convolution of a weighting with a shift weighting over the offsets -R..R (eq. 8): the sum of the
    /// weighting rolled by each offset, weighted by that offset's shift probability.
    /// </summary>
    private Tensor<T> CircularConvolution(Tensor<T> weighting, Tensor<T> shift)
    {
        int offsets = shift.Length;
        int radius = (offsets - 1) / 2;
        var rolled = new Tensor<T>[offsets];
        for (int offset = -radius; offset <= radius; offset++)
        {
            // torch.roll semantics: rolled(i) = weighting(i - offset).
            rolled[offset + radius] = Engine.Reshape(
                Engine.TensorRoll(weighting, new[] { offset }, new[] { 0 }), new[] { 1, _size });
        }

        var stacked = Engine.TensorConcatenate(rolled, 0);
        return Engine.Reshape(Engine.TensorMatMul(Engine.Reshape(shift, new[] { 1, offsets }), stacked), new[] { _size });
    }

    /// <summary>
    /// Reads from memory using attention weights.
    /// </summary>
    /// <param name="readWeights">The attention weights for reading.</param>
    /// <returns>The weighted sum of memory contents.</returns>
    public Tensor<T> Read(Vector<T> readWeights)
    {
        Guard.NotNull(readWeights);
        return Read(Tensor<T>.FromVector(readWeights));
    }

    /// <summary>
    /// Reads from memory: r_t = sum_i w_t(i) M_t(i) (Graves et al. 2014, eq. 2).
    /// </summary>
    /// <param name="readWeights">The read weighting, <c>[Size]</c>.</param>
    /// <returns>The read vector, <c>[Width]</c>.</returns>
    public Tensor<T> Read(Tensor<T> readWeights)
    {
        Guard.NotNull(readWeights);
        var row = Engine.Reshape(readWeights, new[] { 1, _size });
        return Engine.Reshape(Engine.TensorMatMul(row, _contents), new[] { _width });
    }

    /// <summary>
    /// Writes to memory using interpolation.
    /// </summary>
    /// <param name="writeWeights">The attention weights for writing.</param>
    /// <param name="eraseVector">The erase vector.</param>
    /// <param name="addVector">The add vector.</param>
    public void Write(Vector<T> writeWeights, Tensor<T> eraseVector, Tensor<T> addVector)
    {
        Guard.NotNull(writeWeights);
        Write(Tensor<T>.FromVector(writeWeights.Clone()), eraseVector, addVector);
    }

    /// <summary>
    /// Writes to memory: erase then add, M_t = M_{t-1} * (1 - w_t e_t^T) + w_t a_t^T (Graves et al. 2014,
    /// eq. 3-4). An erase or add vector shorter than <see cref="Width"/> is zero-padded.
    /// </summary>
    /// <param name="writeWeights">The write weighting, <c>[Size]</c>.</param>
    /// <param name="eraseVector">The erase vector e_t, elements in [0, 1].</param>
    /// <param name="addVector">The add vector a_t.</param>
    public void Write(Tensor<T> writeWeights, Tensor<T> eraseVector, Tensor<T> addVector)
    {
        Guard.NotNull(writeWeights);
        Guard.NotNull(eraseVector);
        Guard.NotNull(addVector);

        // Store write weights for sharpness computation
        _lastWriteWeights = NTMSequences<T>.Flatten(writeWeights);

        var column = Engine.Reshape(writeWeights, new[] { _size, 1 });
        var erase = Engine.Reshape(NTMSequences<T>.FitLength(eraseVector, _width), new[] { 1, _width });
        var add = Engine.Reshape(NTMSequences<T>.FitLength(addVector, _width), new[] { 1, _width });
        var retained = Engine.TensorAddScalar(Engine.TensorNegate(Engine.TensorMatMul(column, erase)), NumOps.One);
        _contents = Engine.TensorAdd(Engine.TensorMultiply(_contents, retained), Engine.TensorMatMul(column, add));
    }

    /// <summary>
    /// Resets memory to initial state.
    /// </summary>
    public void Reset()
    {
        _contents = new Tensor<T>(new int[] { _size, _width });
        _lastWriteWeights = null;
    }

    /// <summary>
    /// Initializes memory with zeros.
    /// </summary>
    public void InitializeZeros()
    {
        Reset();
    }

    /// <summary>
    /// Initializes memory with random values.
    /// </summary>
    public void InitializeRandom()
    {
        var values = new Tensor<T>(new int[] { _size, _width });
        for (int i = 0; i < values.Length; i++)
        {
            // (u - 0.5) * 0.1, uniform in [-0.05, 0.05): one draw per element in row-major order, scaled in double
            // and rounded to T once, so a seeded memory is unchanged in every T.
            values[i] = NumOps.FromDouble((_random.NextDouble() - 0.5) * 0.1);
        }

        _contents = values;
    }

    /// <summary>
    /// Initializes memory with learned initialization.
    /// </summary>
    public void InitializeLearned()
    {
        // Use learned initialization matrix - for now, use zeros
        Reset();
    }

    /// <summary>
    /// Clones the memory matrix.
    /// </summary>
    /// <returns>A deep copy of this memory.</returns>
    public NTMMemory<T> Clone()
    {
        var cloned = new NTMMemory<T>(_size, _width, _randomSeed);
        cloned._contents = _contents.Clone();
        return cloned;
    }

    /// <summary>
    /// Gets the memory value at the specified location.
    /// </summary>
    public T GetValue(int row, int col)
    {
        return _contents[new[] { row, col }];
    }

    /// <summary>
    /// Computes memory usage penalty for regularization.
    /// </summary>
    /// <returns>The usage penalty value.</returns>
    public T ComputeUsagePenalty()
    {
        using var noGrad = new NoGradScope<T>();
        return UsagePenaltyTensor()[0];
    }

    /// <summary>
    /// The usage penalty as a one-element tensor: the summed row norms over <c>Size * Width</c>.
    /// </summary>
    internal Tensor<T> UsagePenaltyTensor()
    {
        // Floored below the square root so an all-zero row has a finite gradient, in Half as well.
        var usage = Engine.ReduceSum(RowNorms(_contents), new[] { 0 }, true);
        return Engine.TensorDivideScalar(usage, NumOps.FromDouble((double)_size * _width));
    }

    /// <summary>
    /// Computes memory sharpness penalty for regularization.
    /// Measures how focused the attention distribution is and penalizes very sharp attention.
    /// </summary>
    /// <returns>The sharpness penalty value (higher = more focused attention).</returns>
    public T ComputeSharpnessPenalty()
    {
        using var noGrad = new NoGradScope<T>();
        return SharpnessPenaltyTensor()[0];
    }

    /// <summary>
    /// The sharpness penalty as a one-element tensor: max(0, sum_i w(i)^2 - 1/N) of the last write weighting.
    /// </summary>
    internal Tensor<T> SharpnessPenaltyTensor()
    {
        // If no write has occurred yet, return zero
        var weights = _lastWriteWeights;
        if (weights is null || weights.Length == 0)
        {
            return new Tensor<T>(new int[] { 1 });
        }

        // Concentration: sum of squared weights, 1/N for a uniform weighting and 1 for a one-hot one.
        var sumSquared = Engine.ReduceSum(Engine.TensorMultiply(weights, weights), new[] { 0 }, true);

        // Only penalize concentration above uniform.
        var excess = Engine.TensorAddScalar(sumSquared, NumOps.FromDouble(-1.0 / weights.Length));
        return Engine.ReLU(excess);
    }
}

/// <summary>
/// Interface for NTM controller.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("NTMController")]
public interface INTMController<T>
{
    /// <summary>
    /// Forward pass through the controller.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="readContents">The previous read contents.</param>
    /// <returns>The controller output.</returns>
    Tensor<T> Forward(Tensor<T> input, List<Tensor<T>> readContents);

    /// <summary>
    /// Generates read keys for all read heads.
    /// </summary>
    /// <param name="output">The controller output.</param>
    /// <returns>List of read keys.</returns>
    List<Tensor<T>> GenerateReadKeys(Tensor<T> output);

    /// <summary>
    /// Generates write key.
    /// </summary>
    /// <param name="output">The controller output.</param>
    /// <returns>The write key.</returns>
    Tensor<T> GenerateWriteKey(Tensor<T> output);

    /// <summary>
    /// Generates the full addressing parameters (key, key strength, gate, shift, sharpening) of every read
    /// head (Graves et al. 2014, section 3.3).
    /// </summary>
    /// <param name="output">The controller output.</param>
    /// <returns>One set of addressing parameters per read head.</returns>
    List<NTMAddressingParameters<T>> GenerateReadAddressing(Tensor<T> output);

    /// <summary>
    /// Generates the full addressing parameters of the write head (Graves et al. 2014, section 3.3).
    /// </summary>
    /// <param name="output">The controller output.</param>
    /// <returns>The write head's addressing parameters.</returns>
    NTMAddressingParameters<T> GenerateWriteAddressing(Tensor<T> output);

    /// <summary>
    /// Generates erase vector for writing.
    /// </summary>
    /// <param name="output">The controller output.</param>
    /// <returns>The erase vector.</returns>
    Tensor<T> GenerateEraseVector(Tensor<T> output);

    /// <summary>
    /// Generates add vector for writing.
    /// </summary>
    /// <param name="output">The controller output.</param>
    /// <returns>The add vector.</returns>
    Tensor<T> GenerateAddVector(Tensor<T> output);

    /// <summary>
    /// Generates final output.
    /// </summary>
    /// <param name="output">The controller output.</param>
    /// <param name="readContents">The current read contents.</param>
    /// <returns>The final output.</returns>
    Tensor<T> GenerateOutput(Tensor<T> output, List<Tensor<T>> readContents);

    /// <summary>
    /// Gets controller parameters.
    /// </summary>
    /// <returns>The parameter vector.</returns>
    Vector<T> GetParameters();

    /// <summary>
    /// Gets the controller's trainable tensors, in <see cref="GetParameters"/> order. They are the sources a
    /// gradient tape differentiates the episode loss against.
    /// </summary>
    /// <returns>The parameter tensors.</returns>
    IReadOnlyList<Tensor<T>> GetParameterTensors();

    /// <summary>
    /// Sets controller parameters (updates internal weights).
    /// </summary>
    /// <param name="parameters">The parameter vector to set.</param>
    void SetParameters(Vector<T> parameters);

    /// <summary>
    /// Resets controller state.
    /// </summary>
    void Reset();
}

/// <summary>
/// A controller whose recurrent state can be saved and restored, so several adapted NTM models can share its
/// weights while each continues from its own state.
/// </summary>
internal interface INTMControllerState
{
    /// <summary>The controller's current recurrent state. The returned object is never mutated afterwards.</summary>
    object CaptureState();

    /// <summary>Makes a state returned by <see cref="CaptureState"/> the controller's current state.</summary>
    void RestoreState(object state);
}

/// <summary>
/// Projections and parameter layout shared by the NTM controllers.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
internal static class NTMControllerMath<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// The projected values one head's location addressing needs: beta, g, gamma and the 2R+1 shift logits.
    /// </summary>
    internal static int AddressingBlockSize(int shiftRange) => 3 + 2 * shiftRange + 1;

    /// <summary>weights @ input + biases, for <c>weights [out, in]</c>.</summary>
    internal static Tensor<T> Project(Tensor<T> weights, Tensor<T> biases, Tensor<T> input)
    {
        int outputSize = weights.Shape[0];
        var column = Engine.Reshape(input, new[] { weights.Shape[1], 1 });
        return Engine.TensorAdd(Engine.Reshape(Engine.TensorMatMul(weights, column), new[] { outputSize }), biases);
    }

    /// <summary>A <c>[heads * width]</c> projection as one <c>[width]</c> tensor per head.</summary>
    internal static List<Tensor<T>> SplitHeads(Tensor<T> projected, int heads, int width)
    {
        var result = new List<Tensor<T>>();
        for (int head = 0; head < heads; head++)
        {
            result.Add(Engine.TensorNarrow(projected, 0, head * width, width));
        }

        return result;
    }

    /// <summary>
    /// The hidden state followed by one <c>[width]</c> slot per read head: each read vector cut or zero-padded to
    /// the width, and zeros for a head with no read vector.
    /// </summary>
    internal static Tensor<T> CombineHiddenAndReads(Tensor<T> hidden, List<Tensor<T>> readContents, int heads, int width)
    {
        var pieces = new Tensor<T>[heads + 1];
        pieces[0] = NTMSequences<T>.Flatten(hidden);
        for (int head = 0; head < heads; head++)
        {
            pieces[head + 1] = head < readContents.Count
                ? NTMSequences<T>.FitLength(readContents[head], width)
                : new Tensor<T>(new[] { width });
        }

        return Engine.TensorConcatenate(pieces, 0);
    }

    /// <summary>
    /// One head's addressing parameters from its block of the addressing projection, with the paper's
    /// constraints: beta = softplus (&gt;= 0), g = sigmoid (in (0, 1)), gamma = 1 + softplus (&gt;= 1) and s = softmax
    /// over the 2R+1 shifts (Graves et al. 2014, section 3.3).
    /// </summary>
    internal static NTMAddressingParameters<T> HeadAddressing(Tensor<T> key, Tensor<T> projected, int head, int shiftRange)
    {
        int offset = head * AddressingBlockSize(shiftRange);
        var keyStrength = Engine.Softplus(Engine.TensorNarrow(projected, 0, offset, 1));
        var gate = Engine.Sigmoid(Engine.TensorNarrow(projected, 0, offset + 1, 1));
        var sharpening = Engine.TensorAddScalar(Engine.Softplus(Engine.TensorNarrow(projected, 0, offset + 2, 1)), NumOps.One);
        var shift = Engine.Softmax(Engine.TensorNarrow(projected, 0, offset + 3, 2 * shiftRange + 1), -1);
        return new NTMAddressingParameters<T>(key, keyStrength, gate, shift, sharpening);
    }

    /// <summary>The read heads' addressing parameters: head i uses block i of the projection.</summary>
    internal static List<NTMAddressingParameters<T>> ReadAddressing(List<Tensor<T>> keys, Tensor<T> projected, int shiftRange)
    {
        var result = new List<NTMAddressingParameters<T>>();
        for (int head = 0; head < keys.Count; head++)
        {
            result.Add(HeadAddressing(keys[head], projected, head, shiftRange));
        }

        return result;
    }

    /// <summary>The tensors' elements, concatenated in order.</summary>
    internal static Vector<T> FlattenParameters(IReadOnlyList<Tensor<T>> tensors)
    {
        if (tensors.Count == 0)
        {
            return new Vector<T>(0);
        }

        return Engine.TensorConcatenate(tensors.Select(NTMSequences<T>.Flatten).ToArray(), 0).ToVector();
    }

    /// <summary>Copies consecutive runs of the vector into the tensors, in order.</summary>
    internal static void AssignParameters(Vector<T> parameters, IReadOnlyList<Tensor<T>> tensors)
    {
        Guard.NotNull(parameters);
        var source = parameters.AsSpan();
        int offset = 0;
        foreach (var tensor in tensors)
        {
            source.Slice(offset, tensor.Length).CopyTo(tensor.AsWritableSpan());
            offset += tensor.Length;
        }
    }
}

/// <summary>
/// LSTM-based NTM controller implementation with learnable parameters.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// This controller uses an LSTM cell to process inputs and generate addressing
/// parameters for the NTM memory. The LSTM maintains hidden state across timesteps,
/// enabling sequential reasoning and temporal dependencies.
/// </para>
/// <para><b>Architecture:</b>
/// <code>
/// Input (inputSize + numReadHeads * memoryWidth)
///   ↓
/// LSTM Cell (hiddenSize)
///   ↓
/// Linear projections → ReadKeys, WriteKey, Erase, Add, Addressing (beta, g, s, gamma per head), Output
/// </code>
/// </para>
/// </remarks>
public class LSTMNTMController<T, TInput, TOutput> : INTMController<T>, INTMControllerState
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private static IEngine Engine => AiDotNetEngine.Current;

    private readonly NTMOptions<T, TInput, TOutput> _options;
    // Not readonly: the true controller input width (external input + read
    // vectors) is only known at the first forward, so _inputSize and the
    // input-gate weights are resolved there (see Forward).
    private int _inputSize;
    // True once the first forward has resolved the real input width. After that
    // a width change is a caller bug (inconsistent input assembly), not a re-init
    // trigger — re-init would silently wipe learned/loaded _weightsInput.
    private bool _inputSizeResolved;
    private readonly int _hiddenSize;
    private readonly int _memoryWidth;
    private readonly int _numReadHeads;
    private readonly int _outputSize;
    private readonly int _shiftRange;
    private readonly Random _random;

    // LSTM gate weights: input, forget, cell, output gates
    // Input weights: [4 * hiddenSize, inputSize]
    [AiDotNet.Attributes.Scratch]
    private Tensor<T> _weightsInput;
    // Hidden weights: [4 * hiddenSize, hiddenSize]
    private readonly Tensor<T> _weightsHidden;
    // Biases: [4 * hiddenSize]
    private readonly Tensor<T> _biases;

    // Output projection weights
    private readonly Tensor<T> _readKeyWeights;    // [numReadHeads * memoryWidth, hiddenSize]
    private readonly Tensor<T> _readKeyBiases;     // [numReadHeads * memoryWidth]
    private readonly Tensor<T> _writeKeyWeights;   // [memoryWidth, hiddenSize]
    private readonly Tensor<T> _writeKeyBiases;    // [memoryWidth]
    private readonly Tensor<T> _eraseWeights;      // [memoryWidth, hiddenSize]
    private readonly Tensor<T> _eraseBiases;       // [memoryWidth]
    private readonly Tensor<T> _addWeights;        // [memoryWidth, hiddenSize]
    private readonly Tensor<T> _addBiases;         // [memoryWidth]
    private readonly Tensor<T> _outputWeights;     // [outputSize, hiddenSize + numReadHeads * memoryWidth]
    private readonly Tensor<T> _outputBiases;      // [outputSize]
    private readonly Tensor<T> _addressingWeights; // [(numReadHeads + 1) * (2 * shiftRange + 4), hiddenSize]
    private readonly Tensor<T> _addressingBiases;  // [(numReadHeads + 1) * (2 * shiftRange + 4)]

    // Hidden and cell state
    private Tensor<T> _hiddenState;
    private Tensor<T> _cellState;

    /// <summary>
    /// Initializes a new instance of LSTMNTMController with learnable weights.
    /// </summary>
    /// <param name="options">The NTM options.</param>
    public LSTMNTMController(NTMOptions<T, TInput, TOutput> options)
    {
        Guard.NotNull(options);
        _options = options;

        _hiddenSize = options.ControllerHiddenSize;
        _memoryWidth = options.MemoryWidth;
        _numReadHeads = options.NumReadHeads;
        _outputSize = options.NumClasses;
        _shiftRange = options.ShiftRange;

        // Input size is the first dimension from the meta-model's expected input
        // For NTM, controller input = input features + read contents
        // Estimate input features from model or use a reasonable default
        _inputSize = _memoryWidth + _numReadHeads * _memoryWidth; // Will be adjusted on first forward pass

        _random = options.RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(options.RandomSeed.Value)
            : RandomHelper.CreateSecureRandom();

        // Initialize LSTM weights with Xavier/Glorot initialization
        double inputScale = Math.Sqrt(2.0 / (_inputSize + _hiddenSize));
        double hiddenScale = Math.Sqrt(2.0 / (2.0 * _hiddenSize));

        // LSTM gates: i, f, c, o (input, forget, cell, output)
        _weightsInput = InitializeTensor(new int[] { 4 * _hiddenSize, _inputSize }, inputScale);
        _weightsHidden = InitializeTensor(new int[] { 4 * _hiddenSize, _hiddenSize }, hiddenScale);

        // Forget gate bias (the second quarter) starts at 1.0 for better gradient flow, the rest at zero.
        _biases = Engine.TensorConcatenate(new[]
        {
            new Tensor<T>(new int[] { _hiddenSize }),
            Tensor<T>.CreateDefault(new int[] { _hiddenSize }, NumOps.One),
            new Tensor<T>(new int[] { 2 * _hiddenSize }),
        }, 0);

        // Initialize projection weights
        double projScale = Math.Sqrt(2.0 / _hiddenSize);
        _readKeyWeights = InitializeTensor(new int[] { _numReadHeads * _memoryWidth, _hiddenSize }, projScale);
        _readKeyBiases = new Tensor<T>(new int[] { _numReadHeads * _memoryWidth });
        _writeKeyWeights = InitializeTensor(new int[] { _memoryWidth, _hiddenSize }, projScale);
        _writeKeyBiases = new Tensor<T>(new int[] { _memoryWidth });
        _eraseWeights = InitializeTensor(new int[] { _memoryWidth, _hiddenSize }, projScale);
        _eraseBiases = new Tensor<T>(new int[] { _memoryWidth });
        _addWeights = InitializeTensor(new int[] { _memoryWidth, _hiddenSize }, projScale);
        _addBiases = new Tensor<T>(new int[] { _memoryWidth });

        // Output projection takes hidden state + read contents
        int outputInputSize = _hiddenSize + _numReadHeads * _memoryWidth;
        double outputScale = Math.Sqrt(2.0 / outputInputSize);
        _outputWeights = InitializeTensor(new int[] { _outputSize, outputInputSize }, outputScale);
        _outputBiases = new Tensor<T>(new int[] { _outputSize });

        // Location-addressing projection (beta, g, gamma, shift logits per head; the write head's block last),
        // drawn after every other weight so the existing weights keep their seeded values.
        int addressingSize = (_numReadHeads + 1) * NTMControllerMath<T>.AddressingBlockSize(_shiftRange);
        _addressingWeights = InitializeTensor(new int[] { addressingSize, _hiddenSize }, projScale);
        _addressingBiases = new Tensor<T>(new int[] { addressingSize });

        // Initialize hidden and cell states
        _hiddenState = new Tensor<T>(new int[] { _hiddenSize });
        _cellState = new Tensor<T>(new int[] { _hiddenSize });
    }

    private Tensor<T> InitializeTensor(int[] shape, double scale)
    {
        var weights = new Tensor<T>(shape);
        for (int i = 0; i < weights.Length; i++)
        {
            // Xavier initialization, uniform in [-scale, scale]: one draw per weight in the same order as
            // always, scaled in double and rounded to T once, so a seeded controller is unchanged in every T.
            weights[i] = NumOps.FromDouble((_random.NextDouble() * 2.0 - 1.0) * scale);
        }

        return weights;
    }

    /// <inheritdoc/>
    public Tensor<T> Forward(Tensor<T> input, List<Tensor<T>> readContents)
    {
        // Concatenate input with read contents if not already done
        var fullInput = input;
        int inputLength = GetTensorLength(fullInput);

        // Lazy input resolution: the constructor can only ESTIMATE the controller
        // input width; the real width (external input + read vectors) is known
        // only at the first forward. Resolve the input-gate weights to that actual
        // width exactly ONCE. After that the width must be stable — the NTM input
        // is [external features + numReadHeads × memoryWidth], all fixed — so a
        // later change means the caller assembled the input inconsistently (the
        // train and predict paths now share one Matrix→features convention, see
        // ConvertToSequence / ConvertInputToTensor). Reinitializing on drift would
        // silently discard learned/loaded input-gate weights, so we fail fast
        // instead (CodeRabbit #1455).
        if (!_inputSizeResolved)
        {
            if (_weightsInput.Shape[1] != inputLength)
            {
                double resolvedScale = Math.Sqrt(2.0 / (inputLength + _hiddenSize));
                _weightsInput = InitializeTensor(new int[] { 4 * _hiddenSize, inputLength }, resolvedScale);
            }
            _inputSize = inputLength;
            _inputSizeResolved = true;
        }
        else if (inputLength != _inputSize)
        {
            throw new ArgumentException(
                $"NTM controller input width changed from {_inputSize} to {inputLength} after the " +
                "first forward. The controller input width (external features + read vectors) must be " +
                "stable once resolved; reinitializing the input-gate weights would discard learned " +
                "parameters. Check that the input is assembled consistently across train and predict.",
                nameof(input));
        }

        (_hiddenState, _cellState) = Engine.TensorLstmCell(
            Engine.Reshape(fullInput, new[] { inputLength }), _hiddenState, _cellState, _weightsInput, _weightsHidden, _biases);

        return _hiddenState;
    }

    private int GetTensorLength(Tensor<T> tensor)
    {
        int length = 1;
        foreach (int dim in tensor._shape)
        {
            length *= dim;
        }
        return length;
    }

    /// <inheritdoc/>
    public List<Tensor<T>> GenerateReadKeys(Tensor<T> output)
    {
        // keys = W_readKey @ hiddenState + bias, one memoryWidth slice per head
        var projected = NTMControllerMath<T>.Project(_readKeyWeights, _readKeyBiases, _hiddenState);
        return NTMControllerMath<T>.SplitHeads(projected, _numReadHeads, _memoryWidth);
    }

    /// <inheritdoc/>
    public Tensor<T> GenerateWriteKey(Tensor<T> output)
    {
        // key = W_writeKey @ hiddenState + bias
        return NTMControllerMath<T>.Project(_writeKeyWeights, _writeKeyBiases, _hiddenState);
    }

    /// <inheritdoc/>
    public List<NTMAddressingParameters<T>> GenerateReadAddressing(Tensor<T> output)
    {
        var projected = NTMControllerMath<T>.Project(_addressingWeights, _addressingBiases, _hiddenState);
        return NTMControllerMath<T>.ReadAddressing(GenerateReadKeys(output), projected, _shiftRange);
    }

    /// <inheritdoc/>
    public NTMAddressingParameters<T> GenerateWriteAddressing(Tensor<T> output)
    {
        var projected = NTMControllerMath<T>.Project(_addressingWeights, _addressingBiases, _hiddenState);
        return NTMControllerMath<T>.HeadAddressing(GenerateWriteKey(output), projected, _numReadHeads, _shiftRange);
    }

    /// <inheritdoc/>
    public Tensor<T> GenerateEraseVector(Tensor<T> output)
    {
        // erase = sigmoid(W_erase @ hiddenState + bias)
        return Engine.Sigmoid(NTMControllerMath<T>.Project(_eraseWeights, _eraseBiases, _hiddenState));
    }

    /// <inheritdoc/>
    public Tensor<T> GenerateAddVector(Tensor<T> output)
    {
        // add = tanh(W_add @ hiddenState + bias)
        return Engine.Tanh(NTMControllerMath<T>.Project(_addWeights, _addBiases, _hiddenState));
    }

    /// <inheritdoc/>
    public Tensor<T> GenerateOutput(Tensor<T> output, List<Tensor<T>> readContents)
    {
        // Concatenate hidden state with read contents, then output = W_output @ combinedInput + bias
        var combinedInput = NTMControllerMath<T>.CombineHiddenAndReads(_hiddenState, readContents, _numReadHeads, _memoryWidth);
        return NTMControllerMath<T>.Project(_outputWeights, _outputBiases, combinedInput);
    }

    /// <inheritdoc/>
    public Vector<T> GetParameters() => NTMControllerMath<T>.FlattenParameters(GetParameterTensors());

    /// <inheritdoc/>
    public IReadOnlyList<Tensor<T>> GetParameterTensors()
    {
        return new List<Tensor<T>>
        {
            _weightsInput, _weightsHidden, _biases,
            _readKeyWeights, _readKeyBiases,
            _writeKeyWeights, _writeKeyBiases,
            _eraseWeights, _eraseBiases,
            _addWeights, _addBiases,
            _outputWeights, _outputBiases,
            _addressingWeights, _addressingBiases,
        };
    }

    /// <inheritdoc/>
    public void SetParameters(Vector<T> parameters)
    {
        // Copy all parameters back to weight tensors in same order as GetParameters
        NTMControllerMath<T>.AssignParameters(parameters, GetParameterTensors());
    }

    /// <inheritdoc/>
    public void Reset()
    {
        // Reset LSTM hidden and cell states to zeros (fresh tensors, so a finished episode's tape keeps its own)
        _hiddenState = new Tensor<T>(new int[] { _hiddenSize });
        _cellState = new Tensor<T>(new int[] { _hiddenSize });
    }

    // The hidden and cell states are replaced on every step and reset, never written in place, so the pair
    // itself is the snapshot.
    object INTMControllerState.CaptureState() => new LstmState(_hiddenState, _cellState);

    void INTMControllerState.RestoreState(object state)
    {
        Guard.NotNull(state);
        if (state is not LstmState lstm)
        {
            throw new ArgumentException("The state was not captured from an LSTM NTM controller.", nameof(state));
        }

        _hiddenState = lstm.Hidden;
        _cellState = lstm.Cell;
    }

    private sealed class LstmState
    {
        public LstmState(Tensor<T> hidden, Tensor<T> cell)
        {
            Hidden = hidden;
            Cell = cell;
        }

        public Tensor<T> Hidden { get; }

        public Tensor<T> Cell { get; }
    }
}

/// <summary>
/// MLP-based NTM controller implementation with learnable parameters.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// This controller uses a multi-layer perceptron to process inputs and generate
/// addressing parameters for the NTM memory. Unlike LSTM, MLP is stateless and
/// processes each timestep independently.
/// </para>
/// <para><b>Architecture:</b>
/// <code>
/// Input (inputSize + numReadHeads * memoryWidth)
///   ↓
/// Hidden Layer 1 (hiddenSize) + ReLU
///   ↓
/// Hidden Layer 2 (hiddenSize) + ReLU
///   ↓
/// Linear projections → ReadKeys, WriteKey, Erase, Add, Addressing (beta, g, s, gamma per head), Output
/// </code>
/// </para>
/// </remarks>
public class MLPNTMController<T, TInput, TOutput> : INTMController<T>, INTMControllerState
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private static IEngine Engine => AiDotNetEngine.Current;

    private readonly NTMOptions<T, TInput, TOutput> _options;
    // Not readonly: like the LSTM controller, the real input width (features, presented label and read vectors)
    // is only known at the first forward, which resolves the first layer to it once.
    private int _inputSize;
    private bool _inputSizeResolved;
    private readonly int _hiddenSize;
    private readonly int _memoryWidth;
    private readonly int _numReadHeads;
    private readonly int _outputSize;
    private readonly int _shiftRange;
    private readonly Random _random;

    // MLP layer weights (2 hidden layers)
    private Tensor<T> _layer1Weights;           // [hiddenSize, inputSize]
    private readonly Tensor<T> _layer1Biases;   // [hiddenSize]
    private readonly Tensor<T> _layer2Weights;  // [hiddenSize, hiddenSize]
    private readonly Tensor<T> _layer2Biases;   // [hiddenSize]

    // Output projection weights
    private readonly Tensor<T> _readKeyWeights;    // [numReadHeads * memoryWidth, hiddenSize]
    private readonly Tensor<T> _readKeyBiases;     // [numReadHeads * memoryWidth]
    private readonly Tensor<T> _writeKeyWeights;   // [memoryWidth, hiddenSize]
    private readonly Tensor<T> _writeKeyBiases;    // [memoryWidth]
    private readonly Tensor<T> _eraseWeights;      // [memoryWidth, hiddenSize]
    private readonly Tensor<T> _eraseBiases;       // [memoryWidth]
    private readonly Tensor<T> _addWeights;        // [memoryWidth, hiddenSize]
    private readonly Tensor<T> _addBiases;         // [memoryWidth]
    private readonly Tensor<T> _outputWeights;     // [outputSize, hiddenSize + numReadHeads * memoryWidth]
    private readonly Tensor<T> _outputBiases;      // [outputSize]
    private readonly Tensor<T> _addressingWeights; // [(numReadHeads + 1) * (2 * shiftRange + 4), hiddenSize]
    private readonly Tensor<T> _addressingBiases;  // [(numReadHeads + 1) * (2 * shiftRange + 4)]

    // Cached hidden state for projection operations
    [Scratch]
    private Tensor<T> _lastHiddenState;

    /// <summary>
    /// Initializes a new instance of MLPNTMController with learnable weights.
    /// </summary>
    /// <param name="options">The NTM options.</param>
    public MLPNTMController(NTMOptions<T, TInput, TOutput> options)
    {
        Guard.NotNull(options);
        _options = options;

        _hiddenSize = options.ControllerHiddenSize;
        _memoryWidth = options.MemoryWidth;
        _numReadHeads = options.NumReadHeads;
        _outputSize = options.NumClasses;
        _shiftRange = options.ShiftRange;

        // Input size = input features + read contents (an estimate, resolved at the first forward)
        _inputSize = _memoryWidth + _numReadHeads * _memoryWidth;

        _random = options.RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(options.RandomSeed.Value)
            : RandomHelper.CreateSecureRandom();

        // Initialize MLP weights with He initialization (for ReLU)
        double layer1Scale = Math.Sqrt(2.0 / _inputSize);
        double layer2Scale = Math.Sqrt(2.0 / _hiddenSize);

        _layer1Weights = InitializeTensor(new int[] { _hiddenSize, _inputSize }, layer1Scale);
        _layer1Biases = new Tensor<T>(new int[] { _hiddenSize });
        _layer2Weights = InitializeTensor(new int[] { _hiddenSize, _hiddenSize }, layer2Scale);
        _layer2Biases = new Tensor<T>(new int[] { _hiddenSize });

        // Initialize projection weights
        double projScale = Math.Sqrt(2.0 / _hiddenSize);
        _readKeyWeights = InitializeTensor(new int[] { _numReadHeads * _memoryWidth, _hiddenSize }, projScale);
        _readKeyBiases = new Tensor<T>(new int[] { _numReadHeads * _memoryWidth });
        _writeKeyWeights = InitializeTensor(new int[] { _memoryWidth, _hiddenSize }, projScale);
        _writeKeyBiases = new Tensor<T>(new int[] { _memoryWidth });
        _eraseWeights = InitializeTensor(new int[] { _memoryWidth, _hiddenSize }, projScale);
        _eraseBiases = new Tensor<T>(new int[] { _memoryWidth });
        _addWeights = InitializeTensor(new int[] { _memoryWidth, _hiddenSize }, projScale);
        _addBiases = new Tensor<T>(new int[] { _memoryWidth });

        // Output projection takes hidden state + read contents
        int outputInputSize = _hiddenSize + _numReadHeads * _memoryWidth;
        double outputScale = Math.Sqrt(2.0 / outputInputSize);
        _outputWeights = InitializeTensor(new int[] { _outputSize, outputInputSize }, outputScale);
        _outputBiases = new Tensor<T>(new int[] { _outputSize });

        // Location-addressing projection, drawn after every other weight so those keep their seeded values.
        int addressingSize = (_numReadHeads + 1) * NTMControllerMath<T>.AddressingBlockSize(_shiftRange);
        _addressingWeights = InitializeTensor(new int[] { addressingSize, _hiddenSize }, projScale);
        _addressingBiases = new Tensor<T>(new int[] { addressingSize });

        // Initialize cached hidden state
        _lastHiddenState = new Tensor<T>(new int[] { _hiddenSize });
    }

    private Tensor<T> InitializeTensor(int[] shape, double scale)
    {
        var weights = new Tensor<T>(shape);
        for (int i = 0; i < weights.Length; i++)
        {
            // He initialization, uniform in [-scale, scale]: one draw per weight in the same order as
            // always, scaled in double and rounded to T once, so a seeded controller is unchanged in every T.
            weights[i] = NumOps.FromDouble((_random.NextDouble() * 2.0 - 1.0) * scale);
        }

        return weights;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The first layer is resolved to the real input width at the first forward, exactly once, as the LSTM
    /// controller's input gate is. It used to keep the constructor's estimate and silently drop the trailing
    /// inputs (or treat missing ones as zero) whenever the real width differed - which with a presented label
    /// always cut off the last read vector.
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input, List<Tensor<T>> readContents)
    {
        int inputLength = GetTensorLength(input);
        if (!_inputSizeResolved)
        {
            if (_layer1Weights.Shape[1] != inputLength)
            {
                _layer1Weights = InitializeTensor(new int[] { _hiddenSize, inputLength }, Math.Sqrt(2.0 / inputLength));
            }
            _inputSize = inputLength;
            _inputSizeResolved = true;
        }
        else if (inputLength != _inputSize)
        {
            throw new ArgumentException(
                $"NTM controller input width changed from {_inputSize} to {inputLength} after the first forward. " +
                "Check that the input is assembled consistently across train and predict.",
                nameof(input));
        }

        // Layer 1: hidden1 = ReLU(W1 * input + b1)
        var hidden1 = Engine.ReLU(NTMControllerMath<T>.Project(_layer1Weights, _layer1Biases, input));

        // Layer 2: hidden2 = ReLU(W2 * hidden1 + b2)
        var hidden2 = Engine.ReLU(NTMControllerMath<T>.Project(_layer2Weights, _layer2Biases, hidden1));

        // Cache the hidden state for projection operations
        _lastHiddenState = hidden2;

        return hidden2;
    }

    private int GetTensorLength(Tensor<T> tensor)
    {
        int length = 1;
        foreach (int dim in tensor._shape)
        {
            length *= dim;
        }
        return length;
    }

    /// <inheritdoc/>
    public List<Tensor<T>> GenerateReadKeys(Tensor<T> output)
    {
        var projected = NTMControllerMath<T>.Project(_readKeyWeights, _readKeyBiases, _lastHiddenState);
        return NTMControllerMath<T>.SplitHeads(projected, _numReadHeads, _memoryWidth);
    }

    /// <inheritdoc/>
    public Tensor<T> GenerateWriteKey(Tensor<T> output)
    {
        return NTMControllerMath<T>.Project(_writeKeyWeights, _writeKeyBiases, _lastHiddenState);
    }

    /// <inheritdoc/>
    public List<NTMAddressingParameters<T>> GenerateReadAddressing(Tensor<T> output)
    {
        var projected = NTMControllerMath<T>.Project(_addressingWeights, _addressingBiases, _lastHiddenState);
        return NTMControllerMath<T>.ReadAddressing(GenerateReadKeys(output), projected, _shiftRange);
    }

    /// <inheritdoc/>
    public NTMAddressingParameters<T> GenerateWriteAddressing(Tensor<T> output)
    {
        var projected = NTMControllerMath<T>.Project(_addressingWeights, _addressingBiases, _lastHiddenState);
        return NTMControllerMath<T>.HeadAddressing(GenerateWriteKey(output), projected, _numReadHeads, _shiftRange);
    }

    /// <inheritdoc/>
    public Tensor<T> GenerateEraseVector(Tensor<T> output)
    {
        // Apply sigmoid to constrain to [0, 1]
        return Engine.Sigmoid(NTMControllerMath<T>.Project(_eraseWeights, _eraseBiases, _lastHiddenState));
    }

    /// <inheritdoc/>
    public Tensor<T> GenerateAddVector(Tensor<T> output)
    {
        // Apply tanh for bounded output
        return Engine.Tanh(NTMControllerMath<T>.Project(_addWeights, _addBiases, _lastHiddenState));
    }

    /// <inheritdoc/>
    public Tensor<T> GenerateOutput(Tensor<T> output, List<Tensor<T>> readContents)
    {
        // Concatenate hidden state with read contents, then output = W_output @ combinedInput + bias
        var combinedInput = NTMControllerMath<T>.CombineHiddenAndReads(_lastHiddenState, readContents, _numReadHeads, _memoryWidth);
        return NTMControllerMath<T>.Project(_outputWeights, _outputBiases, combinedInput);
    }

    /// <inheritdoc/>
    public Vector<T> GetParameters() => NTMControllerMath<T>.FlattenParameters(GetParameterTensors());

    /// <inheritdoc/>
    public IReadOnlyList<Tensor<T>> GetParameterTensors()
    {
        return new List<Tensor<T>>
        {
            _layer1Weights, _layer1Biases,
            _layer2Weights, _layer2Biases,
            _readKeyWeights, _readKeyBiases,
            _writeKeyWeights, _writeKeyBiases,
            _eraseWeights, _eraseBiases,
            _addWeights, _addBiases,
            _outputWeights, _outputBiases,
            _addressingWeights, _addressingBiases,
        };
    }

    /// <inheritdoc/>
    public void SetParameters(Vector<T> parameters)
    {
        // Copy all parameters back to weight tensors in same order as GetParameters
        NTMControllerMath<T>.AssignParameters(parameters, GetParameterTensors());
    }

    /// <inheritdoc/>
    public void Reset()
    {
        // MLP is stateless, but reset cached hidden state
        _lastHiddenState = new Tensor<T>(new int[] { _hiddenSize });
    }

    // The cached hidden state is replaced by every forward and reset, never written in place.
    object INTMControllerState.CaptureState() => _lastHiddenState;

    void INTMControllerState.RestoreState(object state)
    {
        Guard.NotNull(state);
        if (state is not Tensor<T> hidden)
        {
            throw new ArgumentException("The state was not captured from an MLP NTM controller.", nameof(state));
        }

        _lastHiddenState = hidden;
    }
}

/// <summary>
/// NTM read head (Graves et al. 2014, section 3.1 and 3.3).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class NTMReadHead<T>
{
    private readonly int _memoryWidth;
    private readonly int _memorySize;
    private readonly int _headIndex;

    /// <summary>
    /// Initializes a new instance of NTMReadHead.
    /// </summary>
    /// <param name="memoryWidth">The width of each memory slot.</param>
    /// <param name="memorySize">The number of memory slots.</param>
    /// <param name="headIndex">The index of this read head.</param>
    public NTMReadHead(int memoryWidth, int memorySize, int headIndex)
    {
        _memoryWidth = memoryWidth;
        _memorySize = memorySize;
        _headIndex = headIndex;
    }

    /// <summary>
    /// Gets the head's previous weighting w_{t-1}; null before its first step, which interpolates with zeros.
    /// </summary>
    public Tensor<T>? PreviousWeights { get; private set; }

    /// <summary>
    /// Computes the read weighting with content and location addressing (Graves et al. 2014, eq. 5-9), and
    /// keeps it as the next step's previous weighting.
    /// </summary>
    /// <param name="addressing">The head's addressing parameters for this step.</param>
    /// <param name="memory">The memory matrix.</param>
    /// <returns>The read weights, <c>[memory.Size]</c>.</returns>
    /// <remarks>
    /// This used to stop at content addressing with an implicit key strength of 1: no interpolation gate, no
    /// convolutional shift and no sharpening, so a head could never iterate over locations.
    /// </remarks>
    public Tensor<T> ComputeReadWeights(NTMAddressingParameters<T> addressing, NTMMemory<T> memory)
    {
        Guard.NotNull(addressing);
        Guard.NotNull(memory);
        var weights = memory.Address(addressing, PreviousWeights ?? new Tensor<T>(new int[] { memory.Size }));
        PreviousWeights = weights;
        return weights;
    }

    /// <summary>
    /// Forgets the previous weighting, for a new episode.
    /// </summary>
    public void Reset()
    {
        PreviousWeights = null;
    }

    /// <summary>
    /// Clones the read head.
    /// </summary>
    /// <returns>A new read head with the same configuration and previous weighting.</returns>
    public NTMReadHead<T> Clone()
    {
        return new NTMReadHead<T>(_memoryWidth, _memorySize, _headIndex) { PreviousWeights = PreviousWeights?.Clone() };
    }
}

/// <summary>
/// NTM write head (Graves et al. 2014, section 3.2 and 3.3).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class NTMWriteHead<T>
{
    private readonly int _memoryWidth;
    private readonly int _memorySize;

    /// <summary>
    /// Initializes a new instance of NTMWriteHead.
    /// </summary>
    /// <param name="memoryWidth">The width of each memory slot.</param>
    /// <param name="memorySize">The number of memory slots.</param>
    public NTMWriteHead(int memoryWidth, int memorySize)
    {
        _memoryWidth = memoryWidth;
        _memorySize = memorySize;
    }

    /// <summary>
    /// Gets the head's previous weighting w_{t-1}; null before its first step, which interpolates with zeros.
    /// </summary>
    public Tensor<T>? PreviousWeights { get; private set; }

    /// <summary>
    /// Computes the write weighting with content and location addressing (Graves et al. 2014, eq. 5-9), and
    /// keeps it as the next step's previous weighting.
    /// </summary>
    /// <param name="addressing">The head's addressing parameters for this step.</param>
    /// <param name="memory">The memory matrix.</param>
    /// <returns>The write weights, <c>[memory.Size]</c>.</returns>
    public Tensor<T> ComputeWriteWeights(NTMAddressingParameters<T> addressing, NTMMemory<T> memory)
    {
        Guard.NotNull(addressing);
        Guard.NotNull(memory);
        var weights = memory.Address(addressing, PreviousWeights ?? new Tensor<T>(new int[] { memory.Size }));
        PreviousWeights = weights;
        return weights;
    }

    /// <summary>
    /// Forgets the previous weighting, for a new episode.
    /// </summary>
    public void Reset()
    {
        PreviousWeights = null;
    }

    /// <summary>
    /// Clones the write head.
    /// </summary>
    /// <returns>A new write head with the same configuration and previous weighting.</returns>
    public NTMWriteHead<T> Clone()
    {
        return new NTMWriteHead<T>(_memoryWidth, _memorySize) { PreviousWeights = PreviousWeights?.Clone() };
    }
}

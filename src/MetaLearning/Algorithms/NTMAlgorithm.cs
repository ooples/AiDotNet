using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;

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
///     # Read from memory using attention
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
public class NTMAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly NTMOptions<T, TInput, TOutput> _ntmOptions;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _ntmOptions;
    private readonly INTMController<T> _controller;
    private readonly NTMMemory<T> _memory;
    private readonly List<NTMReadHead<T>> _readHeads;
    private readonly NTMWriteHead<T> _writeHead;
    private List<Tensor<T>> _previousReadContents;

    // Cached inputs/outputs for gradient computation
    private TInput? _cachedQueryInput;
    private TOutput? _cachedQueryOutput;
    private TInput? _cachedSupportInput;
    private TOutput? _cachedSupportOutput;

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
            options.LossFunction ?? options.MetaModel.DefaultLossFunction,
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer)
    {
        _ntmOptions = options;

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

        // Validate configuration
        if (!_ntmOptions.IsValid())
        {
            throw new ArgumentException("NTM configuration is invalid. Check all parameters.", nameof(options));
        }

        // Initialize memory if configured
        if (_ntmOptions.InitializeMemory)
        {
            InitializeMemory();
        }
    }

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.NTM;

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));
        }

        T totalLoss = NumOps.Zero;

        foreach (var task in taskBatch.Tasks)
        {
            // Reset memory state for new episode
            ResetMemoryState();

            // Train on this episode
            T episodeLoss = TrainEpisode(task);
            totalLoss = NumOps.Add(totalLoss, episodeLoss);
        }

        // Return average loss
        return NumOps.Divide(totalLoss, NumOps.FromDouble(taskBatch.BatchSize));
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null)
        {
            throw new ArgumentNullException(nameof(task));
        }

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
    /// Trains the NTM on a single episode.
    /// </summary>
    private T TrainEpisode(IMetaLearningTask<T, TInput, TOutput> task)
    {
        // Cache both support and query data for gradient computation
        _cachedSupportInput = task.SupportInput;
        _cachedSupportOutput = task.SupportOutput;
        _cachedQueryInput = task.QueryInput;
        _cachedQueryOutput = task.QueryOutput;

        // Process support set to initialize memory
        ProcessSupportSet(task.SupportInput, task.SupportOutput, null);

        // Process query set
        var queryPredictions = ProcessSequence(task.QueryInput, task.QueryOutput);

        // Compute loss
        T episodeLoss = ComputeLoss(queryPredictions, task.QueryOutput);

        // Add memory regularization
        episodeLoss = AddMemoryRegularization(episodeLoss);

        // Backpropagate through time and update all components
        UpdateComponents(episodeLoss);

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
        var targetSequence = ConvertOutputToSequence(supportOutputs);

        // Process each time step - use the model's components if provided
        if (model != null)
        {
            // Process through the model to prime its memory
            for (int t = 0; t < inputSequence.Length; t++)
            {
                // Use model.Predict which processes through the model's own memory/controller
                model.ProcessTimestepInternal(inputSequence[t]);
            }
        }
        else
        {
            // Use the algorithm's own components
            for (int t = 0; t < inputSequence.Length; t++)
            {
                ProcessTimestep(inputSequence[t], targetSequence[t]);
            }
        }
    }

    /// <summary>
    /// Processes a sequence of inputs and targets.
    /// </summary>
    private Tensor<T> ProcessSequence(TInput inputs, TOutput targets)
    {
        // Convert to sequence format
        var inputSequence = ConvertToSequence(inputs);
        var targetSequence = ConvertOutputToSequence(targets);
        var outputs = new List<Tensor<T>>();

        // Process each time step
        for (int t = 0; t < inputSequence.Length; t++)
        {
            var output = ProcessTimestep(inputSequence[t], targetSequence[t]);
            outputs.Add(output);
        }

        // Return final output or sequence of outputs
        return outputs.Count > 0 ? outputs[outputs.Count - 1] : new Tensor<T>(new int[] { _ntmOptions.NumClasses });
    }

    /// <summary>
    /// Processes a single timestep.
    /// </summary>
    private Tensor<T> ProcessTimestep(Tensor<T> input, Tensor<T> target)
    {
        // Combine input with previous read contents
        var controllerInput = CombineInputWithReadContents(input);

        // Forward pass through controller
        var controllerOutput = _controller.Forward(controllerInput, _previousReadContents);

        // Generate addressing parameters
        var readKeys = _controller.GenerateReadKeys(controllerOutput);
        var writeKey = _controller.GenerateWriteKey(controllerOutput);
        var eraseVector = _controller.GenerateEraseVector(controllerOutput);
        var addVector = _controller.GenerateAddVector(controllerOutput);

        // Read from memory using all read heads
        var currentReadContents = new List<Tensor<T>>();
        for (int i = 0; i < _readHeads.Count; i++)
        {
            var readWeights = _readHeads[i].ComputeReadWeights(readKeys[i], _memory);
            var readContent = _memory.Read(readWeights);
            currentReadContents.Add(readContent);
        }

        // Write to memory
        var writeWeights = _writeHead.ComputeWriteWeights(writeKey, _memory);
        _memory.Write(writeWeights, eraseVector, addVector);

        // Generate output
        var output = _controller.GenerateOutput(controllerOutput, currentReadContents);

        // Update previous read contents for next timestep
        _previousReadContents = currentReadContents;

        return output;
    }

    /// <summary>
    /// Combines input with previous read contents.
    /// </summary>
    private Tensor<T> CombineInputWithReadContents(Tensor<T> input)
    {
        // Concatenate input with flattened read contents
        var combined = input;

        foreach (var readContent in _previousReadContents)
        {
            var flattened = FlattenTensor(readContent);
            combined = ConcatenateTensors(combined, flattened);
        }

        return combined;
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
    /// Adds memory regularization to the loss.
    /// </summary>
    private T AddMemoryRegularization(T baseLoss)
    {
        T totalLoss = baseLoss;

        // Add memory usage regularization
        if (_ntmOptions.MemoryUsageRegularization > 0.0)
        {
            T memoryUsage = _memory.ComputeUsagePenalty();
            T regWeight = NumOps.FromDouble(_ntmOptions.MemoryUsageRegularization);
            totalLoss = NumOps.Add(totalLoss, NumOps.Multiply(memoryUsage, regWeight));
        }

        // Add memory sharpness regularization
        if (_ntmOptions.MemorySharpnessRegularization > 0.0)
        {
            T sharpness = _memory.ComputeSharpnessPenalty();
            T regWeight = NumOps.FromDouble(_ntmOptions.MemorySharpnessRegularization);
            totalLoss = NumOps.Add(totalLoss, NumOps.Multiply(sharpness, regWeight));
        }

        return totalLoss;
    }

    /// <summary>
    /// Updates all NTM components through backpropagation.
    /// </summary>
    /// <remarks>
    /// Uses finite difference gradient approximation for the controller parameters.
    /// Memory state is updated implicitly through the write operations during forward pass.
    /// </remarks>
    private void UpdateComponents(T loss)
    {
        // Get current controller parameters
        var controllerParams = _controller.GetParameters();
        if (controllerParams.Length == 0)
        {
            return;
        }

        // Compute gradients using finite differences
        var gradients = ComputeControllerGradients(controllerParams, loss);

        // Apply gradient clipping if configured
        if (_options.GradientClipThreshold.HasValue && _options.GradientClipThreshold.Value > 0)
        {
            gradients = ClipGradients(gradients, _options.GradientClipThreshold.Value);
        }

        // Update parameters using learning rate
        var updatedParams = ApplyGradients(controllerParams, gradients, _options.OuterLearningRate);

        // Apply updated parameters to controller
        SetControllerParameters(updatedParams);
    }

    /// <summary>
    /// Computes gradients for controller parameters using finite differences.
    /// </summary>
    private Vector<T> ComputeControllerGradients(Vector<T> parameters, T baseLoss)
    {
        var gradients = new Vector<T>(parameters.Length);
        double epsilon = 1e-5;

        for (int i = 0; i < parameters.Length; i++)
        {
            // Perturb parameter positively
            T original = parameters[i];
            parameters[i] = NumOps.Add(original, NumOps.FromDouble(epsilon));
            SetControllerParameters(parameters);

            // Compute perturbed loss (run a forward pass with perturbed parameters)
            T perturbedLoss = ComputeCurrentLoss();

            // Compute gradient: (f(x+h) - f(x)) / h
            gradients[i] = NumOps.Divide(
                NumOps.Subtract(perturbedLoss, baseLoss),
                NumOps.FromDouble(epsilon));

            // Restore original parameter
            parameters[i] = original;
        }

        // Restore original parameters
        SetControllerParameters(parameters);

        return gradients;
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
            var currentParams = MetaModel.GetParameters();
            int copyLength = Math.Min(parameters.Length, currentParams.Length);
            for (int i = 0; i < copyLength; i++)
            {
                currentParams[i] = parameters[i];
            }
            MetaModel.SetParameters(currentParams);
        }
    }

    /// <summary>
    /// Computes current loss by running forward pass on cached inputs.
    /// Resets memory state and re-runs support set processing to ensure
    /// consistent gradient computation for finite differences.
    /// </summary>
    private T ComputeCurrentLoss()
    {
        // If no cached inputs, return zero loss
        if (_cachedQueryInput == null || _cachedQueryOutput == null
            || _cachedSupportInput == null || _cachedSupportOutput == null)
        {
            return NumOps.Zero;
        }

        // Reset to a clean episode state to ensure consistent gradient computation
        ResetMemoryState();

        // Re-run support set processing to rebuild memory state
        ProcessSupportSet(_cachedSupportInput, _cachedSupportOutput, null);

        // Run forward pass with current controller parameters on cached query data
        var predictions = ProcessSequence(_cachedQueryInput, _cachedQueryOutput);

        // Compute loss including memory regularization (same objective as TrainEpisode)
        T loss = ComputeLoss(predictions, _cachedQueryOutput);
        return AddMemoryRegularization(loss);
    }

    // Helper methods

    private Tensor<T>[] ConvertToSequence(TInput inputs)
    {
        // Convert input to sequence of tensors
        if (inputs is Tensor<T> tensor)
        {
            if (tensor.Shape.Length == 1)
            {
                return new Tensor<T>[] { tensor };
            }
            else if (tensor.Shape.Length >= 2)
            {
                int sequenceLength = tensor.Shape[0];
                var sequence = new Tensor<T>[sequenceLength];
                int featureSize = 1;
                for (int i = 1; i < tensor.Shape.Length; i++)
                {
                    featureSize *= tensor.Shape[i];
                }

                for (int t = 0; t < sequenceLength; t++)
                {
                    sequence[t] = new Tensor<T>(new int[] { featureSize });
                    for (int f = 0; f < featureSize; f++)
                    {
                        sequence[t][f] = tensor.GetFlat(t * featureSize + f);
                    }
                }
                return sequence;
            }
        }

        return new Tensor<T>[] { new Tensor<T>(new int[] { 1 }) };
    }

    private Tensor<T>[] ConvertOutputToSequence(TOutput outputs)
    {
        // Convert output to sequence of tensors
        if (outputs is Tensor<T> tensor)
        {
            if (tensor.Shape.Length == 1)
            {
                return new Tensor<T>[] { tensor };
            }
            else if (tensor.Shape.Length >= 2)
            {
                int sequenceLength = tensor.Shape[0];
                var sequence = new Tensor<T>[sequenceLength];
                int featureSize = 1;
                for (int i = 1; i < tensor.Shape.Length; i++)
                {
                    featureSize *= tensor.Shape[i];
                }

                for (int t = 0; t < sequenceLength; t++)
                {
                    sequence[t] = new Tensor<T>(new int[] { featureSize });
                    for (int f = 0; f < featureSize; f++)
                    {
                        sequence[t][f] = tensor.GetFlat(t * featureSize + f);
                    }
                }
                return sequence;
            }
        }

        return new Tensor<T>[] { new Tensor<T>(new int[] { 1 }) };
    }

    private Tensor<T> FlattenTensor(Tensor<T> tensor)
    {
        // Flatten tensor to 1D
        int totalSize = 1;
        for (int i = 0; i < tensor.Shape.Length; i++)
        {
            totalSize *= tensor.Shape[i];
        }

        var flattened = new Tensor<T>(new int[] { totalSize });
        for (int i = 0; i < totalSize; i++)
        {
            flattened[i] = tensor.GetFlat(i);
        }
        return flattened;
    }

    private Tensor<T> ConcatenateTensors(Tensor<T> a, Tensor<T> b)
    {
        // Concatenate along first dimension (both are 1D)
        int sizeA = 1;
        for (int i = 0; i < a.Shape.Length; i++) sizeA *= a.Shape[i];

        int sizeB = 1;
        for (int i = 0; i < b.Shape.Length; i++) sizeB *= b.Shape[i];

        var result = new Tensor<T>(new int[] { sizeA + sizeB });

        for (int i = 0; i < sizeA; i++)
        {
            result[i] = a.GetFlat(i);
        }
        for (int i = 0; i < sizeB; i++)
        {
            result[sizeA + i] = b.GetFlat(i);
        }

        return result;
    }

    private T ComputeLoss(Tensor<T> predictions, TOutput targets)
    {
        // Compute appropriate loss (cross-entropy, MSE, etc.)
        if (targets is Tensor<T> targetTensor)
        {
            T loss = NumOps.Zero;
            int size = Math.Min(
                predictions.Shape.Length > 0 ? predictions.Shape[0] : 0,
                targetTensor.Shape.Length > 0 ? targetTensor.Shape[0] : 0);

            for (int i = 0; i < size; i++)
            {
                T diff = NumOps.Subtract(predictions[i], targetTensor[i]);
                loss = NumOps.Add(loss, NumOps.Multiply(diff, diff));
            }

            if (size > 0)
            {
                loss = NumOps.Divide(loss, NumOps.FromDouble(size));
            }

            return loss;
        }

        return NumOps.FromDouble(1.0);
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
    private readonly INTMController<T> _controller;
    private readonly NTMMemory<T> _memory;
    private readonly List<NTMReadHead<T>> _readHeads;
    private readonly NTMWriteHead<T> _writeHead;
    private readonly NTMOptions<T, TInput, TOutput> _options;
    private readonly List<Tensor<T>> _readContents;

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
        _controller = controller ?? throw new ArgumentNullException(nameof(controller));
        _memory = memory ?? throw new ArgumentNullException(nameof(memory));
        _readHeads = readHeads ?? throw new ArgumentNullException(nameof(readHeads));
        _writeHead = writeHead ?? throw new ArgumentNullException(nameof(writeHead));
        _options = options ?? throw new ArgumentNullException(nameof(options));

        // Initialize read contents
        _readContents = new List<Tensor<T>>();
        for (int i = 0; i < readHeads.Count; i++)
        {
            _readContents.Add(new Tensor<T>(new int[] { options.MemoryWidth }));
        }
    }

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <inheritdoc/>
    public TOutput Predict(TInput input)
    {
        // Convert input to tensor format
        var inputTensor = ConvertInputToTensor(input);

        // Combine input with previous read contents
        var controllerInput = CombineInputWithReadContents(inputTensor);

        // Forward pass through controller
        var controllerOutput = _controller.Forward(controllerInput, _readContents);

        // Generate addressing parameters
        var readKeys = _controller.GenerateReadKeys(controllerOutput);
        var writeKey = _controller.GenerateWriteKey(controllerOutput);
        var eraseVector = _controller.GenerateEraseVector(controllerOutput);
        var addVector = _controller.GenerateAddVector(controllerOutput);

        // Read from memory using all read heads
        var currentReadContents = new List<Tensor<T>>();
        for (int i = 0; i < _readHeads.Count; i++)
        {
            var readWeights = _readHeads[i].ComputeReadWeights(readKeys[i], _memory);
            var readContent = _memory.Read(readWeights);
            currentReadContents.Add(readContent);
        }

        // Write to memory
        var writeWeights = _writeHead.ComputeWriteWeights(writeKey, _memory);
        _memory.Write(writeWeights, eraseVector, addVector);

        // Generate output
        var output = _controller.GenerateOutput(controllerOutput, currentReadContents);

        // Update read contents for next call
        for (int i = 0; i < _readContents.Count && i < currentReadContents.Count; i++)
        {
            _readContents[i] = currentReadContents[i];
        }

        return ConvertTensorToOutput(output);
    }

    /// <summary>
    /// Processes a single timestep using the model's internal components.
    /// This is used during adaptation to prime the model's memory.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    internal void ProcessTimestepInternal(Tensor<T> input)
    {
        // Combine input with previous read contents
        var controllerInput = CombineInputWithReadContents(input);

        // Forward pass through controller
        var controllerOutput = _controller.Forward(controllerInput, _readContents);

        // Generate addressing parameters
        var readKeys = _controller.GenerateReadKeys(controllerOutput);
        var writeKey = _controller.GenerateWriteKey(controllerOutput);
        var eraseVector = _controller.GenerateEraseVector(controllerOutput);
        var addVector = _controller.GenerateAddVector(controllerOutput);

        // Read from memory using all read heads
        var currentReadContents = new List<Tensor<T>>();
        for (int i = 0; i < _readHeads.Count; i++)
        {
            var readWeights = _readHeads[i].ComputeReadWeights(readKeys[i], _memory);
            var readContent = _memory.Read(readWeights);
            currentReadContents.Add(readContent);
        }

        // Write to memory
        var writeWeights = _writeHead.ComputeWriteWeights(writeKey, _memory);
        _memory.Write(writeWeights, eraseVector, addVector);

        // Update read contents for next timestep
        for (int i = 0; i < _readContents.Count && i < currentReadContents.Count; i++)
        {
            _readContents[i] = currentReadContents[i];
        }
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
            var result = new Tensor<T>(new int[] { vector.Length });
            for (int i = 0; i < vector.Length; i++)
            {
                result[i] = vector[i];
            }
            return result;
        }

        return new Tensor<T>(new int[] { _options.MemoryWidth });
    }

    /// <summary>
    /// Combines input with previous read contents.
    /// </summary>
    private Tensor<T> CombineInputWithReadContents(Tensor<T> input)
    {
        int inputSize = input.Shape.Length > 0 ? input.Shape[0] : 0;
        int readSize = _readContents.Sum(r => r.Shape.Length > 0 ? r.Shape[0] : 0);

        var combined = new Tensor<T>(new int[] { inputSize + readSize });
        int idx = 0;

        for (int i = 0; i < inputSize; i++)
        {
            combined[idx++] = input[i];
        }

        foreach (var readContent in _readContents)
        {
            int size = readContent.Shape.Length > 0 ? readContent.Shape[0] : 0;
            for (int i = 0; i < size; i++)
            {
                combined[idx++] = readContent[i];
            }
        }

        return combined;
    }

    /// <summary>
    /// Converts tensor output to the expected output type.
    /// </summary>
    private TOutput ConvertTensorToOutput(Tensor<T> tensor)
    {
        if (typeof(TOutput) == typeof(Tensor<T>))
        {
            return (TOutput)(object)tensor;
        }

        if (typeof(TOutput) == typeof(Vector<T>))
        {
            return (TOutput)(object)tensor.ToVector();
        }

        // Default: return the tensor cast to TOutput
        return (TOutput)(object)tensor;
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
public class NTMMemory<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Matrix<T> _memoryMatrix;
    private readonly int _size;
    private readonly int _width;
    private readonly int? _randomSeed;
    private readonly Random _random;
    private Vector<T>? _lastWriteWeights;  // Track last write weights for sharpness computation

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
        _memoryMatrix = new Matrix<T>(size, width);
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
    /// Reads from memory using attention weights.
    /// </summary>
    /// <param name="readWeights">The attention weights for reading.</param>
    /// <returns>The weighted sum of memory contents.</returns>
    public Tensor<T> Read(Vector<T> readWeights)
    {
        // Weighted sum of memory rows
        var readContent = new Tensor<T>(new int[] { _width });

        for (int i = 0; i < _width; i++)
        {
            T weightedSum = NumOps.Zero;
            for (int j = 0; j < _size; j++)
            {
                T weightedValue = NumOps.Multiply(_memoryMatrix[j, i], readWeights[j]);
                weightedSum = NumOps.Add(weightedSum, weightedValue);
            }
            readContent[i] = weightedSum;
        }

        return readContent;
    }

    /// <summary>
    /// Writes to memory using interpolation.
    /// </summary>
    /// <param name="writeWeights">The attention weights for writing.</param>
    /// <param name="eraseVector">The erase vector.</param>
    /// <param name="addVector">The add vector.</param>
    public void Write(Vector<T> writeWeights, Tensor<T> eraseVector, Tensor<T> addVector)
    {
        // Store write weights for sharpness computation
        _lastWriteWeights = writeWeights.Clone();

        for (int i = 0; i < _size; i++)
        {
            for (int j = 0; j < _width; j++)
            {
                // Erase then add
                T eraseVal = j < eraseVector.Shape[0] ? eraseVector[j] : NumOps.Zero;
                T addVal = j < addVector.Shape[0] ? addVector[j] : NumOps.Zero;
                T current = _memoryMatrix[i, j];

                T erased = NumOps.Multiply(current, NumOps.Subtract(NumOps.One, NumOps.Multiply(writeWeights[i], eraseVal)));
                T updated = NumOps.Add(erased, NumOps.Multiply(writeWeights[i], addVal));

                _memoryMatrix[i, j] = updated;
            }
        }
    }

    /// <summary>
    /// Resets memory to initial state.
    /// </summary>
    public void Reset()
    {
        for (int i = 0; i < _size; i++)
        {
            for (int j = 0; j < _width; j++)
            {
                _memoryMatrix[i, j] = NumOps.Zero;
            }
        }
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
        for (int i = 0; i < _size; i++)
        {
            for (int j = 0; j < _width; j++)
            {
                _memoryMatrix[i, j] = NumOps.FromDouble((_random.NextDouble() - 0.5) * 0.1);
            }
        }
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
        for (int i = 0; i < _size; i++)
        {
            for (int j = 0; j < _width; j++)
            {
                cloned._memoryMatrix[i, j] = _memoryMatrix[i, j];
            }
        }
        return cloned;
    }

    /// <summary>
    /// Gets the memory value at the specified location.
    /// </summary>
    public T GetValue(int row, int col)
    {
        return _memoryMatrix[row, col];
    }

    /// <summary>
    /// Computes memory usage penalty for regularization.
    /// </summary>
    /// <returns>The usage penalty value.</returns>
    public T ComputeUsagePenalty()
    {
        // Measure how much of memory is being used
        T usage = NumOps.Zero;
        for (int i = 0; i < _size; i++)
        {
            T rowNorm = NumOps.Zero;
            for (int j = 0; j < _width; j++)
            {
                T squared = NumOps.Multiply(_memoryMatrix[i, j], _memoryMatrix[i, j]);
                rowNorm = NumOps.Add(rowNorm, squared);
            }
            double normValue = NumOps.ToDouble(rowNorm);
            usage = NumOps.Add(usage, NumOps.FromDouble(Math.Sqrt(Math.Max(0, normValue))));
        }
        return NumOps.Divide(usage, NumOps.FromDouble(_size * _width));
    }

    /// <summary>
    /// Computes memory sharpness penalty for regularization.
    /// Measures how focused the attention distribution is and penalizes very sharp attention.
    /// </summary>
    /// <returns>The sharpness penalty value (higher = more focused attention).</returns>
    public T ComputeSharpnessPenalty()
    {
        // If no write has occurred yet, return zero
        if (_lastWriteWeights == null || _lastWriteWeights.Length == 0)
        {
            return NumOps.Zero;
        }

        // Compute concentration measure: sum of squared weights
        // Higher values mean more focused/sharp attention (approaching one-hot)
        // Lower values mean more spread out attention (approaching uniform)
        T sumSquared = NumOps.Zero;
        for (int i = 0; i < _lastWriteWeights.Length; i++)
        {
            T w = _lastWriteWeights[i];
            sumSquared = NumOps.Add(sumSquared, NumOps.Multiply(w, w));
        }

        // For a uniform distribution over N items: sum(w^2) = 1/N
        // For a one-hot distribution: sum(w^2) = 1
        // Return the excess concentration above uniform as the penalty
        T uniformConcentration = NumOps.FromDouble(1.0 / _lastWriteWeights.Length);
        T excessConcentration = NumOps.Subtract(sumSquared, uniformConcentration);

        // Only penalize if more concentrated than uniform (positive excess)
        double excessValue = NumOps.ToDouble(excessConcentration);
        return excessValue > 0 ? excessConcentration : NumOps.Zero;
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
/// Linear projections → ReadKeys, WriteKey, Erase, Add, Output
/// </code>
/// </para>
/// </remarks>
public class LSTMNTMController<T, TInput, TOutput> : INTMController<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly NTMOptions<T, TInput, TOutput> _options;
    private readonly int _inputSize;
    private readonly int _hiddenSize;
    private readonly int _memoryWidth;
    private readonly int _numReadHeads;
    private readonly int _outputSize;
    private readonly Random _random;

    // LSTM gate weights: input, forget, cell, output gates
    // Input weights: [4 * hiddenSize, inputSize]
    private readonly Tensor<T> _weightsInput;
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

    // Hidden and cell state
    private Tensor<T> _hiddenState;
    private Tensor<T> _cellState;

    /// <summary>
    /// Initializes a new instance of LSTMNTMController with learnable weights.
    /// </summary>
    /// <param name="options">The NTM options.</param>
    public LSTMNTMController(NTMOptions<T, TInput, TOutput> options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));

        _hiddenSize = options.ControllerHiddenSize;
        _memoryWidth = options.MemoryWidth;
        _numReadHeads = options.NumReadHeads;
        _outputSize = options.NumClasses;

        // Input size is the first dimension from the meta-model's expected input
        // For NTM, controller input = input features + read contents
        // Estimate input features from model or use a reasonable default
        _inputSize = _memoryWidth + _numReadHeads * _memoryWidth; // Will be adjusted on first forward pass

        _random = options.RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(options.RandomSeed.Value)
            : RandomHelper.CreateSecureRandom();

        // Initialize LSTM weights with Xavier/Glorot initialization
        double inputScale = Math.Sqrt(2.0 / (_inputSize + _hiddenSize));
        double hiddenScale = Math.Sqrt(2.0 / (_hiddenSize * 2));

        // LSTM gates: i, f, c, o (input, forget, cell, output)
        _weightsInput = InitializeTensor(new int[] { 4 * _hiddenSize, _inputSize }, inputScale);
        _weightsHidden = InitializeTensor(new int[] { 4 * _hiddenSize, _hiddenSize }, hiddenScale);
        _biases = new Tensor<T>(new int[] { 4 * _hiddenSize });

        // Initialize forget gate bias to 1.0 for better gradient flow
        for (int i = _hiddenSize; i < 2 * _hiddenSize; i++)
        {
            _biases[i] = NumOps.FromDouble(1.0);
        }

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

        // Initialize hidden and cell states
        _hiddenState = new Tensor<T>(new int[] { _hiddenSize });
        _cellState = new Tensor<T>(new int[] { _hiddenSize });
    }

    private Tensor<T> InitializeTensor(int[] shape, double scale)
    {
        var tensor = new Tensor<T>(shape);
        int totalSize = 1;
        foreach (int dim in shape)
        {
            totalSize *= dim;
        }

        for (int i = 0; i < totalSize; i++)
        {
            // Xavier initialization: uniform in [-scale, scale]
            double value = (_random.NextDouble() * 2.0 - 1.0) * scale;
            tensor[i] = NumOps.FromDouble(value);
        }
        return tensor;
    }

    /// <inheritdoc/>
    public Tensor<T> Forward(Tensor<T> input, List<Tensor<T>> readContents)
    {
        // Concatenate input with read contents if not already done
        var fullInput = input;
        int inputLength = GetTensorLength(fullInput);

        // LSTM forward pass: compute all four gates
        // gates = W_input * x + W_hidden * h + b
        var gates = new Tensor<T>(new int[] { 4 * _hiddenSize });

        // Input contribution
        for (int g = 0; g < 4 * _hiddenSize; g++)
        {
            T sum = _biases[g];

            // W_input * x
            int validInputSize = Math.Min(inputLength, _weightsInput.Shape[1]);
            for (int j = 0; j < validInputSize; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(_weightsInput[new[] { g, j }], fullInput[j]));
            }

            // W_hidden * h
            for (int j = 0; j < _hiddenSize; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(_weightsHidden[new[] { g, j }], _hiddenState[j]));
            }

            gates[g] = sum;
        }

        // Apply activations: i, f gates use sigmoid; c uses tanh; o uses sigmoid
        var inputGate = new Tensor<T>(new int[] { _hiddenSize });
        var forgetGate = new Tensor<T>(new int[] { _hiddenSize });
        var cellGate = new Tensor<T>(new int[] { _hiddenSize });
        var outputGate = new Tensor<T>(new int[] { _hiddenSize });

        for (int i = 0; i < _hiddenSize; i++)
        {
            inputGate[i] = Sigmoid(gates[i]);
            forgetGate[i] = Sigmoid(gates[_hiddenSize + i]);
            cellGate[i] = Tanh(gates[2 * _hiddenSize + i]);
            outputGate[i] = Sigmoid(gates[3 * _hiddenSize + i]);
        }

        // Update cell state: c = f * c_prev + i * g
        for (int i = 0; i < _hiddenSize; i++)
        {
            _cellState[i] = NumOps.Add(
                NumOps.Multiply(forgetGate[i], _cellState[i]),
                NumOps.Multiply(inputGate[i], cellGate[i]));
        }

        // Update hidden state: h = o * tanh(c)
        for (int i = 0; i < _hiddenSize; i++)
        {
            _hiddenState[i] = NumOps.Multiply(outputGate[i], Tanh(_cellState[i]));
        }

        return _hiddenState;
    }

    private int GetTensorLength(Tensor<T> tensor)
    {
        int length = 1;
        foreach (int dim in tensor.Shape)
        {
            length *= dim;
        }
        return length;
    }

    private T Sigmoid(T x)
    {
        double val = NumOps.ToDouble(x);
        return NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-val)));
    }

    private T Tanh(T x)
    {
        double val = NumOps.ToDouble(x);
        return NumOps.FromDouble(Math.Tanh(val));
    }

    /// <inheritdoc/>
    public List<Tensor<T>> GenerateReadKeys(Tensor<T> output)
    {
        var keys = new List<Tensor<T>>();

        for (int head = 0; head < _numReadHeads; head++)
        {
            var key = new Tensor<T>(new int[] { _memoryWidth });
            int keyOffset = head * _memoryWidth;

            for (int i = 0; i < _memoryWidth; i++)
            {
                T sum = _readKeyBiases[keyOffset + i];
                for (int j = 0; j < _hiddenSize; j++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(
                        _readKeyWeights[new[] { keyOffset + i, j }],
                        _hiddenState[j]));
                }
                key[i] = sum;
            }
            keys.Add(key);
        }

        return keys;
    }

    /// <inheritdoc/>
    public Tensor<T> GenerateWriteKey(Tensor<T> output)
    {
        var key = new Tensor<T>(new int[] { _memoryWidth });

        for (int i = 0; i < _memoryWidth; i++)
        {
            T sum = _writeKeyBiases[i];
            for (int j = 0; j < _hiddenSize; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(_writeKeyWeights[new[] { i, j }], _hiddenState[j]));
            }
            key[i] = sum;
        }

        return key;
    }

    /// <inheritdoc/>
    public Tensor<T> GenerateEraseVector(Tensor<T> output)
    {
        var erase = new Tensor<T>(new int[] { _memoryWidth });

        for (int i = 0; i < _memoryWidth; i++)
        {
            T sum = _eraseBiases[i];
            for (int j = 0; j < _hiddenSize; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(_eraseWeights[new[] { i, j }], _hiddenState[j]));
            }
            // Apply sigmoid to constrain to [0, 1]
            erase[i] = Sigmoid(sum);
        }

        return erase;
    }

    /// <inheritdoc/>
    public Tensor<T> GenerateAddVector(Tensor<T> output)
    {
        var add = new Tensor<T>(new int[] { _memoryWidth });

        for (int i = 0; i < _memoryWidth; i++)
        {
            T sum = _addBiases[i];
            for (int j = 0; j < _hiddenSize; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(_addWeights[new[] { i, j }], _hiddenState[j]));
            }
            // Apply tanh for bounded output
            add[i] = Tanh(sum);
        }

        return add;
    }

    /// <inheritdoc/>
    public Tensor<T> GenerateOutput(Tensor<T> output, List<Tensor<T>> readContents)
    {
        // Concatenate hidden state with read contents
        int totalInputSize = _hiddenSize + _numReadHeads * _memoryWidth;
        var combinedInput = new Tensor<T>(new int[] { totalInputSize });

        // Copy hidden state
        for (int i = 0; i < _hiddenSize; i++)
        {
            combinedInput[i] = _hiddenState[i];
        }

        // Copy read contents
        int offset = _hiddenSize;
        for (int head = 0; head < readContents.Count && head < _numReadHeads; head++)
        {
            for (int i = 0; i < _memoryWidth && i < readContents[head].Shape[0]; i++)
            {
                combinedInput[offset + head * _memoryWidth + i] = readContents[head][i];
            }
        }

        // Linear projection to output
        var result = new Tensor<T>(new int[] { _outputSize });
        for (int i = 0; i < _outputSize; i++)
        {
            T sum = _outputBiases[i];
            for (int j = 0; j < totalInputSize; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(_outputWeights[new[] { i, j }], combinedInput[j]));
            }
            result[i] = sum;
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> GetParameters()
    {
        // Count total parameters
        int totalParams = 0;
        totalParams += 4 * _hiddenSize * _inputSize;      // _weightsInput
        totalParams += 4 * _hiddenSize * _hiddenSize;     // _weightsHidden
        totalParams += 4 * _hiddenSize;                    // _biases
        totalParams += _numReadHeads * _memoryWidth * _hiddenSize; // _readKeyWeights
        totalParams += _numReadHeads * _memoryWidth;       // _readKeyBiases
        totalParams += _memoryWidth * _hiddenSize;         // _writeKeyWeights
        totalParams += _memoryWidth;                        // _writeKeyBiases
        totalParams += _memoryWidth * _hiddenSize;         // _eraseWeights
        totalParams += _memoryWidth;                        // _eraseBiases
        totalParams += _memoryWidth * _hiddenSize;         // _addWeights
        totalParams += _memoryWidth;                        // _addBiases
        int outputInputSize = _hiddenSize + _numReadHeads * _memoryWidth;
        totalParams += _outputSize * outputInputSize;      // _outputWeights
        totalParams += _outputSize;                         // _outputBiases

        var parameters = new Vector<T>(totalParams);
        int idx = 0;

        // Copy all weights to parameter vector
        idx = CopyTensorToVector(_weightsInput, parameters, idx);
        idx = CopyTensorToVector(_weightsHidden, parameters, idx);
        idx = CopyTensorToVector(_biases, parameters, idx);
        idx = CopyTensorToVector(_readKeyWeights, parameters, idx);
        idx = CopyTensorToVector(_readKeyBiases, parameters, idx);
        idx = CopyTensorToVector(_writeKeyWeights, parameters, idx);
        idx = CopyTensorToVector(_writeKeyBiases, parameters, idx);
        idx = CopyTensorToVector(_eraseWeights, parameters, idx);
        idx = CopyTensorToVector(_eraseBiases, parameters, idx);
        idx = CopyTensorToVector(_addWeights, parameters, idx);
        idx = CopyTensorToVector(_addBiases, parameters, idx);
        idx = CopyTensorToVector(_outputWeights, parameters, idx);
        CopyTensorToVector(_outputBiases, parameters, idx);

        return parameters;
    }

    private int CopyTensorToVector(Tensor<T> tensor, Vector<T> vector, int startIdx)
    {
        int length = GetTensorLength(tensor);
        for (int i = 0; i < length; i++)
        {
            vector[startIdx + i] = tensor[i];
        }
        return startIdx + length;
    }

    private int CopyVectorToTensor(Vector<T> vector, Tensor<T> tensor, int startIdx)
    {
        int length = GetTensorLength(tensor);
        for (int i = 0; i < length; i++)
        {
            tensor[i] = vector[startIdx + i];
        }
        return startIdx + length;
    }

    /// <inheritdoc/>
    public void SetParameters(Vector<T> parameters)
    {
        int idx = 0;

        // Copy all parameters back to weight tensors in same order as GetParameters
        idx = CopyVectorToTensor(parameters, _weightsInput, idx);
        idx = CopyVectorToTensor(parameters, _weightsHidden, idx);
        idx = CopyVectorToTensor(parameters, _biases, idx);
        idx = CopyVectorToTensor(parameters, _readKeyWeights, idx);
        idx = CopyVectorToTensor(parameters, _readKeyBiases, idx);
        idx = CopyVectorToTensor(parameters, _writeKeyWeights, idx);
        idx = CopyVectorToTensor(parameters, _writeKeyBiases, idx);
        idx = CopyVectorToTensor(parameters, _eraseWeights, idx);
        idx = CopyVectorToTensor(parameters, _eraseBiases, idx);
        idx = CopyVectorToTensor(parameters, _addWeights, idx);
        idx = CopyVectorToTensor(parameters, _addBiases, idx);
        idx = CopyVectorToTensor(parameters, _outputWeights, idx);
        CopyVectorToTensor(parameters, _outputBiases, idx);
    }

    /// <inheritdoc/>
    public void Reset()
    {
        // Reset LSTM hidden and cell states to zeros
        for (int i = 0; i < _hiddenSize; i++)
        {
            _hiddenState[i] = NumOps.Zero;
            _cellState[i] = NumOps.Zero;
        }
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
/// Linear projections → ReadKeys, WriteKey, Erase, Add, Output
/// </code>
/// </para>
/// </remarks>
public class MLPNTMController<T, TInput, TOutput> : INTMController<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly NTMOptions<T, TInput, TOutput> _options;
    private readonly int _inputSize;
    private readonly int _hiddenSize;
    private readonly int _memoryWidth;
    private readonly int _numReadHeads;
    private readonly int _outputSize;
    private readonly Random _random;

    // MLP layer weights (2 hidden layers)
    private readonly Tensor<T> _layer1Weights;  // [hiddenSize, inputSize]
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

    // Cached hidden state for projection operations
    private Tensor<T> _lastHiddenState;

    /// <summary>
    /// Initializes a new instance of MLPNTMController with learnable weights.
    /// </summary>
    /// <param name="options">The NTM options.</param>
    public MLPNTMController(NTMOptions<T, TInput, TOutput> options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));

        _hiddenSize = options.ControllerHiddenSize;
        _memoryWidth = options.MemoryWidth;
        _numReadHeads = options.NumReadHeads;
        _outputSize = options.NumClasses;

        // Input size = input features + read contents
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

        // Initialize cached hidden state
        _lastHiddenState = new Tensor<T>(new int[] { _hiddenSize });
    }

    private Tensor<T> InitializeTensor(int[] shape, double scale)
    {
        var tensor = new Tensor<T>(shape);
        int totalSize = 1;
        foreach (int dim in shape)
        {
            totalSize *= dim;
        }

        for (int i = 0; i < totalSize; i++)
        {
            // He initialization: uniform in [-scale, scale]
            double value = (_random.NextDouble() * 2.0 - 1.0) * scale;
            tensor[i] = NumOps.FromDouble(value);
        }
        return tensor;
    }

    /// <inheritdoc/>
    public Tensor<T> Forward(Tensor<T> input, List<Tensor<T>> readContents)
    {
        var fullInput = input;
        int inputLength = GetTensorLength(fullInput);

        // Layer 1: hidden1 = ReLU(W1 * input + b1)
        var hidden1 = new Tensor<T>(new int[] { _hiddenSize });
        for (int i = 0; i < _hiddenSize; i++)
        {
            T sum = _layer1Biases[i];
            int validInputSize = Math.Min(inputLength, _layer1Weights.Shape[1]);
            for (int j = 0; j < validInputSize; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(_layer1Weights[new[] { i, j }], fullInput[j]));
            }
            hidden1[i] = ReLU(sum);
        }

        // Layer 2: hidden2 = ReLU(W2 * hidden1 + b2)
        var hidden2 = new Tensor<T>(new int[] { _hiddenSize });
        for (int i = 0; i < _hiddenSize; i++)
        {
            T sum = _layer2Biases[i];
            for (int j = 0; j < _hiddenSize; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(_layer2Weights[new[] { i, j }], hidden1[j]));
            }
            hidden2[i] = ReLU(sum);
        }

        // Cache the hidden state for projection operations
        _lastHiddenState = hidden2;

        return hidden2;
    }

    private int GetTensorLength(Tensor<T> tensor)
    {
        int length = 1;
        foreach (int dim in tensor.Shape)
        {
            length *= dim;
        }
        return length;
    }

    private T ReLU(T x)
    {
        double val = NumOps.ToDouble(x);
        return NumOps.FromDouble(Math.Max(0.0, val));
    }

    private T Sigmoid(T x)
    {
        double val = NumOps.ToDouble(x);
        return NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-val)));
    }

    private T Tanh(T x)
    {
        double val = NumOps.ToDouble(x);
        return NumOps.FromDouble(Math.Tanh(val));
    }

    /// <inheritdoc/>
    public List<Tensor<T>> GenerateReadKeys(Tensor<T> output)
    {
        var keys = new List<Tensor<T>>();

        for (int head = 0; head < _numReadHeads; head++)
        {
            var key = new Tensor<T>(new int[] { _memoryWidth });
            int keyOffset = head * _memoryWidth;

            for (int i = 0; i < _memoryWidth; i++)
            {
                T sum = _readKeyBiases[keyOffset + i];
                for (int j = 0; j < _hiddenSize; j++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(
                        _readKeyWeights[new[] { keyOffset + i, j }],
                        _lastHiddenState[j]));
                }
                key[i] = sum;
            }
            keys.Add(key);
        }

        return keys;
    }

    /// <inheritdoc/>
    public Tensor<T> GenerateWriteKey(Tensor<T> output)
    {
        var key = new Tensor<T>(new int[] { _memoryWidth });

        for (int i = 0; i < _memoryWidth; i++)
        {
            T sum = _writeKeyBiases[i];
            for (int j = 0; j < _hiddenSize; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(_writeKeyWeights[new[] { i, j }], _lastHiddenState[j]));
            }
            key[i] = sum;
        }

        return key;
    }

    /// <inheritdoc/>
    public Tensor<T> GenerateEraseVector(Tensor<T> output)
    {
        var erase = new Tensor<T>(new int[] { _memoryWidth });

        for (int i = 0; i < _memoryWidth; i++)
        {
            T sum = _eraseBiases[i];
            for (int j = 0; j < _hiddenSize; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(_eraseWeights[new[] { i, j }], _lastHiddenState[j]));
            }
            // Apply sigmoid to constrain to [0, 1]
            erase[i] = Sigmoid(sum);
        }

        return erase;
    }

    /// <inheritdoc/>
    public Tensor<T> GenerateAddVector(Tensor<T> output)
    {
        var add = new Tensor<T>(new int[] { _memoryWidth });

        for (int i = 0; i < _memoryWidth; i++)
        {
            T sum = _addBiases[i];
            for (int j = 0; j < _hiddenSize; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(_addWeights[new[] { i, j }], _lastHiddenState[j]));
            }
            // Apply tanh for bounded output
            add[i] = Tanh(sum);
        }

        return add;
    }

    /// <inheritdoc/>
    public Tensor<T> GenerateOutput(Tensor<T> output, List<Tensor<T>> readContents)
    {
        // Concatenate hidden state with read contents
        int totalInputSize = _hiddenSize + _numReadHeads * _memoryWidth;
        var combinedInput = new Tensor<T>(new int[] { totalInputSize });

        // Copy hidden state
        for (int i = 0; i < _hiddenSize; i++)
        {
            combinedInput[i] = _lastHiddenState[i];
        }

        // Copy read contents
        int offset = _hiddenSize;
        for (int head = 0; head < readContents.Count && head < _numReadHeads; head++)
        {
            for (int i = 0; i < _memoryWidth && i < readContents[head].Shape[0]; i++)
            {
                combinedInput[offset + head * _memoryWidth + i] = readContents[head][i];
            }
        }

        // Linear projection to output
        var result = new Tensor<T>(new int[] { _outputSize });
        for (int i = 0; i < _outputSize; i++)
        {
            T sum = _outputBiases[i];
            for (int j = 0; j < totalInputSize; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(_outputWeights[new[] { i, j }], combinedInput[j]));
            }
            result[i] = sum;
        }

        return result;
    }

    /// <inheritdoc/>
    public Vector<T> GetParameters()
    {
        // Count total parameters
        int totalParams = 0;
        totalParams += _hiddenSize * _inputSize;           // _layer1Weights
        totalParams += _hiddenSize;                         // _layer1Biases
        totalParams += _hiddenSize * _hiddenSize;          // _layer2Weights
        totalParams += _hiddenSize;                         // _layer2Biases
        totalParams += _numReadHeads * _memoryWidth * _hiddenSize; // _readKeyWeights
        totalParams += _numReadHeads * _memoryWidth;       // _readKeyBiases
        totalParams += _memoryWidth * _hiddenSize;         // _writeKeyWeights
        totalParams += _memoryWidth;                        // _writeKeyBiases
        totalParams += _memoryWidth * _hiddenSize;         // _eraseWeights
        totalParams += _memoryWidth;                        // _eraseBiases
        totalParams += _memoryWidth * _hiddenSize;         // _addWeights
        totalParams += _memoryWidth;                        // _addBiases
        int outputInputSize = _hiddenSize + _numReadHeads * _memoryWidth;
        totalParams += _outputSize * outputInputSize;      // _outputWeights
        totalParams += _outputSize;                         // _outputBiases

        var parameters = new Vector<T>(totalParams);
        int idx = 0;

        // Copy all weights to parameter vector
        idx = CopyTensorToVector(_layer1Weights, parameters, idx);
        idx = CopyTensorToVector(_layer1Biases, parameters, idx);
        idx = CopyTensorToVector(_layer2Weights, parameters, idx);
        idx = CopyTensorToVector(_layer2Biases, parameters, idx);
        idx = CopyTensorToVector(_readKeyWeights, parameters, idx);
        idx = CopyTensorToVector(_readKeyBiases, parameters, idx);
        idx = CopyTensorToVector(_writeKeyWeights, parameters, idx);
        idx = CopyTensorToVector(_writeKeyBiases, parameters, idx);
        idx = CopyTensorToVector(_eraseWeights, parameters, idx);
        idx = CopyTensorToVector(_eraseBiases, parameters, idx);
        idx = CopyTensorToVector(_addWeights, parameters, idx);
        idx = CopyTensorToVector(_addBiases, parameters, idx);
        idx = CopyTensorToVector(_outputWeights, parameters, idx);
        CopyTensorToVector(_outputBiases, parameters, idx);

        return parameters;
    }

    private int CopyTensorToVector(Tensor<T> tensor, Vector<T> vector, int startIdx)
    {
        int length = GetTensorLength(tensor);
        for (int i = 0; i < length; i++)
        {
            vector[startIdx + i] = tensor[i];
        }
        return startIdx + length;
    }

    private int CopyVectorToTensor(Vector<T> vector, Tensor<T> tensor, int startIdx)
    {
        int length = GetTensorLength(tensor);
        for (int i = 0; i < length; i++)
        {
            tensor[i] = vector[startIdx + i];
        }
        return startIdx + length;
    }

    /// <inheritdoc/>
    public void SetParameters(Vector<T> parameters)
    {
        int idx = 0;

        // Copy all parameters back to weight tensors in same order as GetParameters
        idx = CopyVectorToTensor(parameters, _layer1Weights, idx);
        idx = CopyVectorToTensor(parameters, _layer1Biases, idx);
        idx = CopyVectorToTensor(parameters, _layer2Weights, idx);
        idx = CopyVectorToTensor(parameters, _layer2Biases, idx);
        idx = CopyVectorToTensor(parameters, _readKeyWeights, idx);
        idx = CopyVectorToTensor(parameters, _readKeyBiases, idx);
        idx = CopyVectorToTensor(parameters, _writeKeyWeights, idx);
        idx = CopyVectorToTensor(parameters, _writeKeyBiases, idx);
        idx = CopyVectorToTensor(parameters, _eraseWeights, idx);
        idx = CopyVectorToTensor(parameters, _eraseBiases, idx);
        idx = CopyVectorToTensor(parameters, _addWeights, idx);
        idx = CopyVectorToTensor(parameters, _addBiases, idx);
        idx = CopyVectorToTensor(parameters, _outputWeights, idx);
        CopyVectorToTensor(parameters, _outputBiases, idx);
    }

    /// <inheritdoc/>
    public void Reset()
    {
        // MLP is stateless, but reset cached hidden state
        for (int i = 0; i < _hiddenSize; i++)
        {
            _lastHiddenState[i] = NumOps.Zero;
        }
    }
}

/// <summary>
/// NTM read head for content-based addressing.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class NTMReadHead<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

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
    /// Computes read weights using cosine similarity.
    /// </summary>
    /// <param name="key">The key tensor.</param>
    /// <param name="memory">The memory matrix.</param>
    /// <returns>The read weights vector.</returns>
    public Vector<T> ComputeReadWeights(Tensor<T> key, NTMMemory<T> memory)
    {
        // Compute similarity between key and all memory locations
        var weights = new Vector<T>(memory.Size);
        T sumWeights = NumOps.Zero;

        for (int i = 0; i < memory.Size; i++)
        {
            // Cosine similarity
            T dotProduct = NumOps.Zero;
            T keyNorm = NumOps.Zero;
            T memNorm = NumOps.Zero;

            for (int j = 0; j < memory.Width && j < key.Shape[0]; j++)
            {
                T keyVal = key[j];
                T memVal = memory.GetValue(i, j);

                dotProduct = NumOps.Add(dotProduct, NumOps.Multiply(keyVal, memVal));
                keyNorm = NumOps.Add(keyNorm, NumOps.Multiply(keyVal, keyVal));
                memNorm = NumOps.Add(memNorm, NumOps.Multiply(memVal, memVal));
            }

            double keyNormVal = Math.Sqrt(Math.Max(NumOps.ToDouble(keyNorm), 1e-8));
            double memNormVal = Math.Sqrt(Math.Max(NumOps.ToDouble(memNorm), 1e-8));
            double similarity = NumOps.ToDouble(dotProduct) / (keyNormVal * memNormVal);

            // Apply softmax-like transformation
            T expSim = NumOps.FromDouble(Math.Exp(similarity));
            weights[i] = expSim;
            sumWeights = NumOps.Add(sumWeights, expSim);
        }

        // Normalize weights
        double sumVal = NumOps.ToDouble(sumWeights);
        if (sumVal > 1e-8)
        {
            for (int i = 0; i < memory.Size; i++)
            {
                weights[i] = NumOps.Divide(weights[i], sumWeights);
            }
        }

        return weights;
    }

    /// <summary>
    /// Clones the read head.
    /// </summary>
    /// <returns>A new read head with the same configuration.</returns>
    public NTMReadHead<T> Clone()
    {
        return new NTMReadHead<T>(_memoryWidth, _memorySize, _headIndex);
    }
}

/// <summary>
/// NTM write head for content-based addressing.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class NTMWriteHead<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

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
    /// Computes write weights using cosine similarity.
    /// </summary>
    /// <param name="key">The key tensor.</param>
    /// <param name="memory">The memory matrix.</param>
    /// <returns>The write weights vector.</returns>
    public Vector<T> ComputeWriteWeights(Tensor<T> key, NTMMemory<T> memory)
    {
        // Compute similarity between key and all memory locations
        var weights = new Vector<T>(memory.Size);
        T sumWeights = NumOps.Zero;

        for (int i = 0; i < memory.Size; i++)
        {
            // Cosine similarity
            T dotProduct = NumOps.Zero;
            T keyNorm = NumOps.Zero;
            T memNorm = NumOps.Zero;

            for (int j = 0; j < memory.Width && j < key.Shape[0]; j++)
            {
                T keyVal = key[j];
                T memVal = memory.GetValue(i, j);

                dotProduct = NumOps.Add(dotProduct, NumOps.Multiply(keyVal, memVal));
                keyNorm = NumOps.Add(keyNorm, NumOps.Multiply(keyVal, keyVal));
                memNorm = NumOps.Add(memNorm, NumOps.Multiply(memVal, memVal));
            }

            double keyNormVal = Math.Sqrt(Math.Max(NumOps.ToDouble(keyNorm), 1e-8));
            double memNormVal = Math.Sqrt(Math.Max(NumOps.ToDouble(memNorm), 1e-8));
            double similarity = NumOps.ToDouble(dotProduct) / (keyNormVal * memNormVal);

            // Apply softmax-like transformation
            T expSim = NumOps.FromDouble(Math.Exp(similarity));
            weights[i] = expSim;
            sumWeights = NumOps.Add(sumWeights, expSim);
        }

        // Normalize weights
        double sumVal = NumOps.ToDouble(sumWeights);
        if (sumVal > 1e-8)
        {
            for (int i = 0; i < memory.Size; i++)
            {
                weights[i] = NumOps.Divide(weights[i], sumWeights);
            }
        }

        return weights;
    }

    /// <summary>
    /// Clones the write head.
    /// </summary>
    /// <returns>A new write head with the same configuration.</returns>
    public NTMWriteHead<T> Clone()
    {
        return new NTMWriteHead<T>(_memoryWidth, _memorySize);
    }
}

using System.Globalization;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.LoRA.Adapters;

/// <summary>
/// Multi-task LoRA adapter that manages multiple task-specific LoRA layers for complex multi-task learning scenarios.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// MultiLoRA extends the basic LoRA concept to handle multiple tasks simultaneously within a single layer.
/// Instead of having one LoRA adaptation, it maintains a dictionary of task-specific LoRA layers,
/// with a routing mechanism to select the appropriate adapter for each task.
/// </para>
/// <para>
/// Key features:
/// - Multiple task-specific LoRA adapters sharing the same base layer
/// - Dynamic task switching during inference and training
/// - Per-task rank configuration for optimal parameter efficiency
/// - Shared base layer weights across all tasks
/// - Task-specific merging for deployment
/// </para>
/// <para><b>For Beginners:</b> Think of MultiLoRA as having one teacher (the base layer) and multiple
/// students (task-specific LoRA adapters), each specializing in different subjects.
///
/// In regular LoRA:
/// - You have one base layer (the teacher)
/// - One LoRA adapter (one student learning one subject)
/// - Output = base + lora_adaptation
///
/// In MultiLoRA:
/// - You have one base layer (the teacher)
/// - Multiple LoRA adapters (multiple students, each specializing in different tasks)
/// - Output = base + task_specific_lora_adaptation
///
/// This is powerful for:
/// 1. Multi-domain learning: Train on medical, legal, and technical documents simultaneously
/// 2. Multi-lingual models: One adapter per language
/// 3. Multi-task learning: Sentiment analysis, named entity recognition, question answering, etc.
/// 4. Continual learning: Add new tasks without forgetting old ones
///
/// Example use case:
/// - Base: Pre-trained language model
/// - Task 1: Sentiment analysis (rank=4)
/// - Task 2: Named entity recognition (rank=8)
/// - Task 3: Question answering (rank=16)
///
/// You can switch between tasks at runtime, and each task only trains its specific LoRA weights!
/// </para>
/// </remarks>
public class MultiLoRAAdapter<T> : LoRAAdapterBase<T>, ILayerSerializationExtras<T>
{
    /// <summary>
    /// Dictionary mapping task names to their specific LoRA layers.
    /// </summary>
    private readonly Dictionary<string, LoRALayer<T>> _taskAdapters;

    /// <summary>
    /// The name of the currently active task.
    /// </summary>
    private string _currentTask;

    /// <summary>
    /// Gets the dictionary of task-specific LoRA adapters.
    /// </summary>
    /// <remarks>
    /// Each task has its own dedicated LoRA layer with potentially different ranks.
    /// This allows for task-specific parameter efficiency optimization.
    /// </remarks>
    public IReadOnlyDictionary<string, LoRALayer<T>> TaskAdapters => _taskAdapters;

    /// <summary>
    /// Gets or sets the name of the currently active task.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Changing this property switches which task-specific adapter is used during forward/backward passes.
    /// This allows dynamic task switching during inference or training.
    /// </para>
    /// <para><b>For Beginners:</b> This is like switching between different "modes" of your model.
    /// Set it to "sentiment" for sentiment analysis, "ner" for named entity recognition, etc.
    /// The base layer stays the same, but the adaptation changes based on the task.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when trying to set a task that hasn't been added.</exception>
    public string CurrentTask
    {
        get => _currentTask;
        set
        {
            if (!_taskAdapters.ContainsKey(value))
            {
                throw new ArgumentException($"Task '{value}' has not been added. Available tasks: {string.Join(", ", _taskAdapters.Keys)}", nameof(value));
            }
            _currentTask = value;
        }
    }

    /// <summary>
    /// Gets the number of tasks configured in this adapter.
    /// </summary>
    public int NumberOfTasks => _taskAdapters.Count;

    /// <summary>
    /// Gets the total parameter count across all task adapters.
    /// </summary>
    /// <remarks>
    /// This includes parameters from the base layer (if not frozen) plus all task-specific LoRA layers.
    /// </remarks>
    public override int ParameterCount
    {
        get
        {
            int totalParams = (_baseLayer != null && !_freezeBaseLayer) ? _baseLayer.ParameterCount : 0;
            if (_taskAdapters != null)
            {
                foreach (var adapter in _taskAdapters.Values)
                {
                    if (adapter != null)
                    {
                        totalParams += adapter.ParameterCount;
                    }
                }
            }
            return totalParams;
        }
    }

    /// <summary>
    /// Initializes a new Multi-LoRA adapter with an initial default task.
    /// </summary>
    /// <param name="baseLayer">The layer to adapt with multiple LoRA adapters.</param>
    /// <param name="defaultTaskName">The name of the default task.</param>
    /// <param name="defaultRank">The rank for the default task's LoRA layer.</param>
    /// <param name="alpha">The LoRA scaling factor (defaults to rank if negative).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer's parameters during training.</param>
    /// <exception cref="ArgumentNullException">Thrown when baseLayer or defaultTaskName is null.</exception>
    /// <exception cref="ArgumentException">Thrown when defaultTaskName is empty or whitespace.</exception>
    /// <remarks>
    /// <para>
    /// The adapter is initialized with one default task. Additional tasks can be added using AddTask().
    /// </para>
    /// <para><b>For Beginners:</b> This creates a MultiLoRA adapter starting with one task.
    /// Think of it like creating a multi-tool that starts with one blade, and you can add more tools later.
    ///
    /// Parameters:
    /// - baseLayer: The shared foundation layer (like the handle of a multi-tool)
    /// - defaultTaskName: A name for your first task (e.g., "sentiment", "translation")
    /// - defaultRank: How complex this task's adaptation is (higher = more parameters)
    /// - alpha: Strength of the adaptation
    /// - freezeBaseLayer: Whether to lock the base layer (usually true to save memory)
    ///
    /// After creation, you can add more tasks with different ranks optimized for each task's complexity.
    /// </para>
    /// </remarks>
    public MultiLoRAAdapter(
        ILayer<T> baseLayer,
        string defaultTaskName,
        int defaultRank,
        double alpha = -1,
        bool freezeBaseLayer = true)
        : base(baseLayer, defaultRank, alpha, freezeBaseLayer)
    {
        if (string.IsNullOrWhiteSpace(defaultTaskName))
        {
            throw new ArgumentException("Default task name cannot be null or whitespace", nameof(defaultTaskName));
        }

        _taskAdapters = new Dictionary<string, LoRALayer<T>>();
        _currentTask = defaultTaskName;

        // Add the default task using the base class's LoRA layer
        _taskAdapters[defaultTaskName] = _loraLayer;
    }

    /// <summary>
    /// Adds a new task with its own LoRA adapter.
    /// </summary>
    /// <param name="taskName">The name of the task (must be unique).</param>
    /// <param name="rank">The rank for this task's LoRA layer.</param>
    /// <param name="alpha">The LoRA scaling factor for this task (defaults to rank if negative).</param>
    /// <exception cref="ArgumentException">Thrown when taskName is null, empty, whitespace, or already exists.</exception>
    /// <remarks>
    /// <para>
    /// Each task can have a different rank, allowing you to optimize parameter usage based on task complexity.
    /// More complex tasks can use higher ranks, while simpler tasks can use lower ranks.
    /// </para>
    /// <para><b>For Beginners:</b> This adds a new "mode" to your model.
    ///
    /// Example:
    /// - Task "sentiment" with rank=4: Simple classification (positive/negative/neutral)
    /// - Task "ner" with rank=8: More complex named entity recognition
    /// - Task "qa" with rank=16: Even more complex question answering
    ///
    /// Each task gets its own small set of parameters (determined by rank) that learn task-specific
    /// adaptations, while all tasks share the same base layer knowledge.
    ///
    /// Benefits:
    /// - Different ranks for different task complexities
    /// - No interference between tasks (each has separate parameters)
    /// - Can train tasks independently or simultaneously
    /// - Add new tasks without retraining existing ones
    /// </para>
    /// </remarks>
    public void AddTask(string taskName, int rank, double alpha = -1)
    {
        if (string.IsNullOrWhiteSpace(taskName))
        {
            throw new ArgumentException("Task name cannot be null or whitespace", nameof(taskName));
        }

        if (_taskAdapters.ContainsKey(taskName))
        {
            throw new ArgumentException($"Task '{taskName}' already exists", nameof(taskName));
        }

        // Create a new LoRA layer for this task
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        LoRALayer<T> taskAdapter = new LoRALayer<T>(inputSize, outputSize, rank, alpha);

        _taskAdapters[taskName] = taskAdapter;
    }

    /// <summary>
    /// Removes a task and its associated LoRA adapter.
    /// </summary>
    /// <param name="taskName">The name of the task to remove.</param>
    /// <returns>True if the task was removed, false if it didn't exist.</returns>
    /// <exception cref="InvalidOperationException">Thrown when trying to remove the last remaining task.</exception>
    /// <remarks>
    /// <para>
    /// You cannot remove the last task. At least one task must always be present.
    /// If removing the current task, the CurrentTask property will be set to the first remaining task.
    /// </para>
    /// <para><b>For Beginners:</b> This removes a task you no longer need.
    /// Like removing a tool from your multi-tool, but you must always keep at least one.
    /// If you remove the currently active task, the adapter automatically switches to another available task.
    /// </para>
    /// </remarks>
    public bool RemoveTask(string taskName)
    {
        if (_taskAdapters.Count <= 1)
        {
            throw new InvalidOperationException("Cannot remove the last task. At least one task must remain.");
        }

        bool removed = _taskAdapters.Remove(taskName);

        // If we removed the current task, switch to the first available task
        if (removed && _currentTask == taskName)
        {
            _currentTask = _taskAdapters.Keys.First();
        }

        return removed;
    }

    /// <summary>
    /// Sets the current task for subsequent forward/backward operations.
    /// </summary>
    /// <param name="taskName">The name of the task to activate.</param>
    /// <exception cref="ArgumentException">Thrown when the task doesn't exist.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This switches which task the model is currently working on.
    /// Call this before forward() to tell the model what kind of task it should perform.
    ///
    /// Example usage:
    /// ```csharp
    /// adapter.SetCurrentTask("sentiment");
    /// var sentimentOutput = adapter.Forward(input);
    ///
    /// adapter.SetCurrentTask("ner");
    /// var nerOutput = adapter.Forward(sameInput);
    /// ```
    ///
    /// Same input, different outputs based on which task is active!
    /// </para>
    /// </remarks>
    public void SetCurrentTask(string taskName)
    {
        CurrentTask = taskName; // Uses property setter for validation
    }

    /// <summary>
    /// Gets the LoRA layer for a specific task.
    /// </summary>
    /// <param name="taskName">The name of the task.</param>
    /// <returns>The LoRA layer for the specified task.</returns>
    /// <exception cref="ArgumentException">Thrown when the task doesn't exist.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This lets you access a specific task's LoRA layer directly.
    /// Useful for inspecting parameters, getting statistics, or manual manipulation.
    /// </para>
    /// </remarks>
    public LoRALayer<T> GetTaskAdapter(string taskName)
    {
        if (!_taskAdapters.TryGetValue(taskName, out var adapter))
        {
            throw new ArgumentException($"Task '{taskName}' not found. Available tasks: {string.Join(", ", _taskAdapters.Keys)}", nameof(taskName));
        }
        return adapter;
    }

    /// <summary>
    /// Gets the rank of a specific task's LoRA adapter.
    /// </summary>
    /// <param name="taskName">The name of the task.</param>
    /// <returns>The rank of the task's LoRA layer.</returns>
    /// <exception cref="ArgumentException">Thrown when the task doesn't exist.</exception>
    public int GetTaskRank(string taskName)
    {
        return GetTaskAdapter(taskName).Rank;
    }

    /// <summary>
    /// Performs the forward pass using the currently active task's adapter.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Sum of base layer output and current task's LoRA output.</returns>
    /// <remarks>
    /// <para>
    /// The forward pass computes: output = base_layer(input) + current_task_lora(input)
    /// </para>
    /// <para><b>For Beginners:</b> This processes data through the model using the current task.
    /// 1. Input goes through the base layer (shared knowledge)
    /// 2. Input goes through the current task's LoRA layer (task-specific adaptation)
    /// 3. Results are added together
    ///
    /// The magic: Different tasks produce different outputs even though they share the same base layer!
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Forward through base layer
        Tensor<T> baseOutput = _baseLayer.Forward(input);

        // Forward through current task's LoRA layer
        LoRALayer<T> currentAdapter = _taskAdapters[_currentTask];
        Tensor<T> loraOutput = currentAdapter.Forward(input);

        // Sum the outputs
        Tensor<T> result = new Tensor<T>(baseOutput.Shape);
        for (int i = 0; i < baseOutput.Length; i++)
        {
            result[i] = NumOps.Add(baseOutput[i], loraOutput[i]);
        }

        return result;
    }

    /// <summary>
    /// Performs the backward pass through the current task's adapter.
    /// </summary>
    /// <param name="outputGradient">Gradient flowing back from the next layer.</param>
    /// <returns>Gradient to pass to the previous layer.</returns>
    /// <remarks>
    /// <para>
    /// The backward pass only updates the current task's LoRA parameters. Other tasks are unaffected.
    /// This allows task-specific learning without interference.
    /// </para>
    /// <para><b>For Beginners:</b> During training, this updates only the current task's parameters.
    ///
    /// Benefits:
    /// - Training task A doesn't mess up task B's learning
    /// - Can train tasks one at a time or in batches
    /// - No "catastrophic forgetting" between tasks
    ///
    /// The gradients flow through:
    /// 1. Current task's LoRA layer (gets updated)
    /// 2. Base layer (only updated if not frozen)
    /// 3. Combined gradients flow back to previous layers
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // Backward through current task's LoRA layer
        LoRALayer<T> currentAdapter = _taskAdapters[_currentTask];
        Tensor<T> loraInputGrad = currentAdapter.Backward(outputGradient);

        // Backward through base layer
        Tensor<T> baseInputGrad = _baseLayer.Backward(outputGradient);

        // Sum input gradients
        Tensor<T> inputGrad = new Tensor<T>(loraInputGrad.Shape);
        for (int i = 0; i < loraInputGrad.Length; i++)
        {
            inputGrad[i] = NumOps.Add(loraInputGrad[i], baseInputGrad[i]);
        }

        // Update parameter gradients vector
        UpdateParameterGradientsFromLayers();

        return inputGrad;
    }

    /// <summary>
    /// Updates parameters for the current task only.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    /// <remarks>
    /// <para>
    /// Only the current task's LoRA parameters are updated. Other tasks remain unchanged.
    /// The base layer is updated only if not frozen.
    /// </para>
    /// <para><b>For Beginners:</b> This is where learning happens for the current task.
    /// Only the active task's parameters get updated, leaving other tasks untouched.
    /// This is key to multi-task learning without interference!
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // Update current task's LoRA layer
        LoRALayer<T> currentAdapter = _taskAdapters[_currentTask];
        currentAdapter.UpdateParameters(learningRate);

        // Update base layer if not frozen
        if (!_freezeBaseLayer)
        {
            _baseLayer.UpdateParameters(learningRate);
        }

        // Update parameter vector
        UpdateParametersFromLayers();
    }

    /// <summary>
    /// Gets the current parameters as a vector.
    /// </summary>
    /// <returns>Vector containing base parameters (if not frozen) and all task adapters' parameters.</returns>
    public override Vector<T> GetParameters()
    {
        Vector<T> parameters = new Vector<T>(ParameterCount);
        int idx = 0;

        // Base layer parameters (if not frozen)
        if (!_freezeBaseLayer)
        {
            Vector<T> baseParams = _baseLayer.GetParameters();
            for (int i = 0; i < baseParams.Length; i++)
            {
                parameters[idx++] = baseParams[i];
            }
        }

        // All task adapters' parameters (stable ordering for deterministic serialization)
        // Guard against null _taskAdapters during base constructor calls
        if (_taskAdapters != null)
        {
            foreach (var taskName in _taskAdapters.Keys.OrderBy(k => k, StringComparer.Ordinal))
            {
                var adapter = _taskAdapters[taskName];
                Vector<T> taskParams = adapter.GetParameters();
                for (int i = 0; i < taskParams.Length; i++)
                {
                    parameters[idx++] = taskParams[i];
                }
            }
        }

        return parameters;
    }

    /// <summary>
    /// Sets the layer parameters from a vector.
    /// </summary>
    /// <param name="parameters">Vector containing all parameters.</param>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException($"Expected {ParameterCount} parameters, got {parameters.Length}", nameof(parameters));
        }

        int idx = 0;

        // Base layer parameters (if not frozen)
        if (!_freezeBaseLayer)
        {
            int baseParamCount = _baseLayer.ParameterCount;
            Vector<T> baseParams = new Vector<T>(baseParamCount);
            for (int i = 0; i < baseParamCount; i++)
            {
                baseParams[i] = parameters[idx++];
            }
            _baseLayer.SetParameters(baseParams);
        }

        // All task adapters' parameters (stable ordering for deterministic serialization)
        // Guard against null _taskAdapters during construction or early calls
        if (_taskAdapters != null)
        {
            foreach (var taskName in _taskAdapters.Keys.OrderBy(k => k, StringComparer.Ordinal))
            {
                var adapter = _taskAdapters[taskName];
                int taskParamCount = adapter.ParameterCount;
                Vector<T> taskParams = new Vector<T>(taskParamCount);
                for (int i = 0; i < taskParamCount; i++)
                {
                    taskParams[i] = parameters[idx++];
                }
                adapter.SetParameters(taskParams);
            }
        }

        Parameters = parameters.Clone();
    }

    /// <summary>
    /// Merges a specific task's LoRA weights into the base layer.
    /// </summary>
    /// <param name="taskName">The name of the task to merge.</param>
    /// <returns>A new layer with the specified task's LoRA weights merged into the base layer.</returns>
    /// <exception cref="ArgumentException">Thrown when the task doesn't exist.</exception>
    /// <exception cref="NotSupportedException">Thrown when the base layer type doesn't support merging.</exception>
    /// <remarks>
    /// <para>
    /// This creates a deployment-ready layer for a specific task by merging its LoRA weights
    /// into the base layer. This is useful when you want to deploy a single-task model.
    /// </para>
    /// <para><b>For Beginners:</b> This "bakes in" one task's adaptations for deployment.
    ///
    /// Use case:
    /// - You trained a MultiLoRA model with 5 tasks
    /// - For production, you only need the "sentiment" task
    /// - Call MergeTaskToLayer("sentiment") to create a standalone layer
    /// - Deploy just that layer (smaller, faster, simpler)
    ///
    /// The merged layer has the base weights + that task's LoRA weights combined into one.
    /// </para>
    /// </remarks>
    public ILayer<T> MergeTaskToLayer(string taskName)
    {
        if (!_taskAdapters.TryGetValue(taskName, out var taskAdapter))
        {
            throw new ArgumentException($"Task '{taskName}' not found. Available tasks: {string.Join(", ", _taskAdapters.Keys)}", nameof(taskName));
        }

        // This implementation assumes the base layer is a DenseLayer or FullyConnectedLayer
        // More sophisticated implementations could support other layer types
        DenseLayer<T>? denseBase = _baseLayer as DenseLayer<T>;
        FullyConnectedLayer<T>? fcBase = _baseLayer as FullyConnectedLayer<T>;

        if (denseBase == null && fcBase == null)
        {
            throw new NotSupportedException($"Merging is currently only supported for DenseLayer and FullyConnectedLayer base layers. Base layer type: {_baseLayer.GetType().Name}");
        }

        // Get the LoRA weight contribution for this task
        Matrix<T> loraWeights = taskAdapter.MergeWeights();

        // Get base layer parameters
        Vector<T> baseParams = _baseLayer.GetParameters();

        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        int weightCount = inputSize * outputSize;

        // Create new parameters with merged weights
        Vector<T> mergedParams = new Vector<T>(baseParams.Length);

        // Merge weights
        for (int i = 0; i < weightCount; i++)
        {
            int row = i / inputSize;
            int col = i % inputSize;
            mergedParams[i] = NumOps.Add(baseParams[i], loraWeights[row, col]);
        }

        // Copy biases unchanged
        for (int i = weightCount; i < baseParams.Length; i++)
        {
            mergedParams[i] = baseParams[i];
        }

        // Use helper method to clone base layer and preserve activation function
        return CreateMergedLayerWithClone(mergedParams);
    }

    /// <summary>
    /// Merges the currently active task's LoRA weights into the base layer.
    /// </summary>
    /// <returns>A new layer with current task's LoRA weights merged into the base layer.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is a shortcut to merge the current task without specifying its name.
    /// Equivalent to calling MergeTaskToLayer(CurrentTask).
    /// </para>
    /// </remarks>
    public override ILayer<T> MergeToOriginalLayer()
    {
        return MergeTaskToLayer(_currentTask);
    }

    /// <summary>
    /// Updates the parameter vector from the current layer states.
    /// </summary>
    protected override void UpdateParametersFromLayers()
    {
        Parameters = GetParameters();
    }

    /// <summary>
    /// Updates the parameter gradients vector from the layer gradients.
    /// </summary>
    private void UpdateParameterGradientsFromLayers()
    {
        // Guard against incomplete initialization
        if (_taskAdapters == null)
        {
            ParameterGradients = new Vector<T>(ParameterCount);
            return;
        }

        ParameterGradients = new Vector<T>(ParameterCount);
        int idx = 0;

        // Base layer gradients (if not frozen)
        if (!_freezeBaseLayer)
        {
            Vector<T> baseGrads = _baseLayer.GetParameterGradients();
            for (int i = 0; i < baseGrads.Length; i++)
            {
                ParameterGradients[idx++] = baseGrads[i];
            }
        }

        // All task adapters' gradients (in same order as GetParameters/SetParameters)
        // Guard against invalid current task
        LoRALayer<T>? currentAdapter = null;
        if (_currentTask != null && _taskAdapters.ContainsKey(_currentTask))
        {
            currentAdapter = _taskAdapters[_currentTask];
        }

        foreach (var taskName in _taskAdapters.Keys.OrderBy(k => k, StringComparer.Ordinal))
        {
            var adapter = _taskAdapters[taskName];
            Vector<T>? grads = (adapter == currentAdapter && currentAdapter != null)
                ? adapter.GetParameterGradients()
                : null;

            for (int i = 0; i < adapter.ParameterCount; i++)
            {
                ParameterGradients[idx++] = grads != null ? grads[i] : NumOps.Zero;
            }
        }
    }

    int ILayerSerializationExtras<T>.ExtraParameterCount => _freezeBaseLayer && _baseLayer != null ? _baseLayer.ParameterCount : 0;

    Vector<T> ILayerSerializationExtras<T>.GetExtraParameters()
    {
        if (!_freezeBaseLayer || _baseLayer == null)
        {
            return new Vector<T>(0);
        }

        return _baseLayer.GetParameters();
    }

    void ILayerSerializationExtras<T>.SetExtraParameters(Vector<T> extraParameters)
    {
        if (!_freezeBaseLayer || _baseLayer == null)
        {
            return;
        }

        if (extraParameters.Length != _baseLayer.ParameterCount)
        {
            throw new ArgumentException(
                $"Expected {_baseLayer.ParameterCount} extra parameters for frozen base layer, got {extraParameters.Length}",
                nameof(extraParameters));
        }

        _baseLayer.SetParameters(extraParameters);
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var meta = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["FreezeBaseLayer"] = _freezeBaseLayer.ToString(CultureInfo.InvariantCulture),
            ["BaseLayerTypeId"] = Uri.EscapeDataString(BuildLayerTypeIdentifier(_baseLayer))
        };

        if (_taskAdapters != null)
        {
            var ordered = _taskAdapters.Keys.OrderBy(k => k, StringComparer.Ordinal).ToArray();
            meta["Tasks"] = string.Join("|", ordered.Select(Uri.EscapeDataString));
            meta["TaskRanks"] = string.Join("|", ordered.Select(t => _taskAdapters[t].Rank.ToString(CultureInfo.InvariantCulture)));
            meta["TaskAlphas"] = string.Join("|", ordered.Select(t => Convert.ToDouble(_taskAdapters[t].Alpha).ToString(CultureInfo.InvariantCulture)));
        }

        if (!string.IsNullOrWhiteSpace(_currentTask))
        {
            meta["CurrentTask"] = Uri.EscapeDataString(_currentTask);
        }

        return meta;
    }

    private static string BuildLayerTypeIdentifier(ILayer<T> layer)
    {
        string typeName = layer.GetType().Name;
        var metadata = new Dictionary<string, string>(StringComparer.Ordinal);

        if (layer is LayerBase<T> layerBase)
        {
            foreach (var kvp in layerBase.GetMetadata())
            {
                metadata[kvp.Key] = kvp.Value;
            }

            if (layerBase.VectorActivation != null)
            {
                metadata["VectorActivationType"] = layerBase.VectorActivation.GetType().AssemblyQualifiedName ?? layerBase.VectorActivation.GetType().FullName ?? string.Empty;
            }
            else if (layerBase.ScalarActivation != null)
            {
                metadata["ScalarActivationType"] = layerBase.ScalarActivation.GetType().AssemblyQualifiedName ?? layerBase.ScalarActivation.GetType().FullName ?? string.Empty;
            }
        }

        if (metadata.Count == 0)
        {
            return typeName;
        }

        foreach (var kvp in metadata.OrderBy(k => k.Key, StringComparer.Ordinal))
        {
            typeName += $";{kvp.Key}={kvp.Value}";
        }

        return typeName;
    }

    /// <summary>
    /// Resets the internal state of all layers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This clears the memory of the base layer and all task adapters.
    /// Use this when starting to process a completely new, unrelated batch of data.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        _baseLayer.ResetState();

        // Defensive null guard for task adapters
        if (_taskAdapters != null)
        {
            foreach (var adapter in _taskAdapters.Values)
            {
                adapter.ResetState();
            }
        }
    }
}

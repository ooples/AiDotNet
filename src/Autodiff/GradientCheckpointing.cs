namespace AiDotNet.Autodiff;

/// <summary>
/// Provides gradient checkpointing functionality for memory-efficient training.
/// </summary>
/// <remarks>
/// <para>
/// Gradient checkpointing (also known as activation checkpointing or memory checkpointing)
/// is a technique that trades computation time for memory by not storing all intermediate
/// activations during the forward pass. Instead, it recomputes them during the backward pass.
/// </para>
/// <para><b>For Beginners:</b> When training large neural networks, storing all intermediate
/// results (activations) can use a lot of memory. Gradient checkpointing saves memory by:
///
/// 1. Only storing activations at certain "checkpoints"
/// 2. During backpropagation, recomputing the activations between checkpoints
///
/// This uses less memory but takes more time (roughly 30% more computation).
/// It's essential for training very large models that wouldn't otherwise fit in GPU memory.
/// </para>
/// <para>
/// This implementation follows patterns from PyTorch's torch.utils.checkpoint and
/// TensorFlow's tf.recompute_grad.
/// </para>
/// </remarks>
public static class GradientCheckpointing<T>
{
    /// <summary>
    /// Thread-local stack to track checkpoint boundaries during forward/backward passes.
    /// </summary>
    [ThreadStatic]
    private static Stack<CheckpointContext<T>>? _checkpointStack;

    /// <summary>
    /// Executes a function with gradient checkpointing.
    /// </summary>
    /// <param name="function">The function to execute with checkpointing.</param>
    /// <param name="inputs">The input nodes to the function.</param>
    /// <returns>The output node from the function.</returns>
    /// <remarks>
    /// <para>
    /// The function will be executed during the forward pass, but its intermediate
    /// activations will not be saved. During the backward pass, the function will
    /// be re-executed to recompute the needed activations.
    /// </para>
    /// <para><b>For Beginners:</b> Wrap parts of your model in this function to save memory:
    ///
    /// <code>
    /// // Without checkpointing (uses more memory):
    /// var output = layer1.Forward(input);
    /// output = layer2.Forward(output);
    ///
    /// // With checkpointing (uses less memory):
    /// var output = GradientCheckpointing&lt;float&gt;.Checkpoint(
    ///     () => {
    ///         var x = layer1.Forward(input);
    ///         return layer2.Forward(x);
    ///     },
    ///     new[] { input }
    /// );
    /// </code>
    /// </para>
    /// </remarks>
    public static ComputationNode<T> Checkpoint(
        Func<ComputationNode<T>> function,
        IEnumerable<ComputationNode<T>> inputs)
    {
        var inputList = inputs.ToList();

        // Create checkpoint context
        var context = new CheckpointContext<T>
        {
            Function = function,
            Inputs = inputList,
            SavedTensors = new Dictionary<ComputationNode<T>, Tensor<T>>()
        };

        // Push context onto stack
        if (_checkpointStack == null)
        {
            _checkpointStack = new Stack<CheckpointContext<T>>();
        }
        _checkpointStack.Push(context);

        // Stop recording during checkpoint forward pass
        var tape = GradientTape<T>.Current;
        bool wasRecording = tape?.IsRecording ?? false;
        tape?.StopRecording();

        try
        {
            // Execute forward pass without recording
            var output = function();

            // Save only the inputs and output for recomputation
            foreach (var input in inputList)
            {
                if (input.Value != null)
                {
                    context.SavedTensors[input] = input.Value.Clone();
                }
            }
            context.Output = output;
            context.OutputValue = output.Value?.Clone();

            // Create a wrapper node that will trigger recomputation during backward
            var checkpointNode = CreateCheckpointNode(context, output);

            return checkpointNode;
        }
        finally
        {
            // Restore recording state
            if (wasRecording)
            {
                tape?.ResumeRecording();
            }

            // Pop context
            _checkpointStack.Pop();
        }
    }

    /// <summary>
    /// Executes a function with gradient checkpointing, supporting multiple outputs.
    /// </summary>
    /// <param name="function">The function to execute with checkpointing.</param>
    /// <param name="inputs">The input nodes to the function.</param>
    /// <returns>The output nodes from the function.</returns>
    public static IReadOnlyList<ComputationNode<T>> CheckpointMultiOutput(
        Func<IReadOnlyList<ComputationNode<T>>> function,
        IEnumerable<ComputationNode<T>> inputs)
    {
        var inputList = inputs.ToList();

        // Create checkpoint context
        var context = new CheckpointContext<T>
        {
            Inputs = inputList,
            SavedTensors = new Dictionary<ComputationNode<T>, Tensor<T>>()
        };

        if (_checkpointStack == null)
        {
            _checkpointStack = new Stack<CheckpointContext<T>>();
        }
        _checkpointStack.Push(context);

        var tape = GradientTape<T>.Current;
        bool wasRecording = tape?.IsRecording ?? false;
        tape?.StopRecording();

        try
        {
            var outputs = function();

            foreach (var input in inputList)
            {
                if (input.Value != null)
                {
                    context.SavedTensors[input] = input.Value.Clone();
                }
            }

            context.MultiOutputs = outputs.ToList();

            var checkpointNodes = outputs.Select((output, index) =>
                CreateCheckpointNode(context, output, index)).ToList();

            return checkpointNodes;
        }
        finally
        {
            if (wasRecording)
            {
                tape?.ResumeRecording();
            }
            _checkpointStack.Pop();
        }
    }

    /// <summary>
    /// Creates a checkpoint node that wraps the output and handles recomputation.
    /// </summary>
    private static ComputationNode<T> CreateCheckpointNode(
        CheckpointContext<T> context,
        ComputationNode<T> output,
        int outputIndex = 0)
    {
        // Create a pass-through node that triggers recomputation on backward
        var checkpointNode = new ComputationNode<T>(output.Value)
        {
            Parents = new List<ComputationNode<T>> { output },
            OperationType = OperationType.Custom,
            RequiresGradient = output.RequiresGradient,
            BackwardFunction = (grad) => RecomputeAndBackward(context, grad, outputIndex)
        };

        // Record to tape if active
        GradientTape<T>.Current?.RecordOperation(checkpointNode);

        return checkpointNode;
    }

    /// <summary>
    /// Recomputes the forward pass and executes backward during gradient computation.
    /// </summary>
    private static void RecomputeAndBackward(
        CheckpointContext<T> context,
        Tensor<T> outputGrad,
        int outputIndex)
    {
        // Restore input values from saved tensors
        foreach (var kvp in context.SavedTensors)
        {
            var inputNode = kvp.Key;
            var savedValue = kvp.Value;
            inputNode.Value = savedValue.Clone();
        }

        // Create a temporary tape for recomputation
        using (var recomputeTape = new GradientTape<T>(persistent: false))
        {
            // Watch all inputs
            foreach (var input in context.Inputs)
            {
                recomputeTape.Watch(input);
            }

            // Recompute forward pass
            ComputationNode<T> recomputedOutput;
            if (context.Function != null)
            {
                recomputedOutput = context.Function();
            }
            else if (context.MultiOutputs != null && outputIndex < context.MultiOutputs.Count)
            {
                recomputedOutput = context.MultiOutputs[outputIndex];
            }
            else
            {
                return;
            }

            // Set the gradient on the recomputed output
            recomputedOutput.Gradient = outputGrad;

            // Perform backward pass on the recomputed graph
            recomputedOutput.Backward();

            // Propagate gradients back to original inputs
            foreach (var input in context.Inputs)
            {
                if (input.Gradient == null && context.SavedTensors.ContainsKey(input))
                {
                    // Find the corresponding recomputed input and copy its gradient
                    var recomputedInput = context.Inputs.FirstOrDefault(i =>
                        ReferenceEquals(i, input));
                    if (recomputedInput?.Gradient != null)
                    {
                        input.Gradient = recomputedInput.Gradient.Clone();
                    }
                }
            }
        }
    }

    /// <summary>
    /// Creates a sequential checkpoint that divides a sequence of layers into segments.
    /// </summary>
    /// <param name="layers">The sequence of layer functions to checkpoint.</param>
    /// <param name="input">The input to the first layer.</param>
    /// <param name="segmentSize">Number of layers per checkpoint segment. Default: 2</param>
    /// <returns>The output from the final layer.</returns>
    /// <remarks>
    /// <para>
    /// This is a convenience method for checkpointing sequential models. It automatically
    /// divides the layers into segments and applies checkpointing to each segment.
    /// </para>
    /// <para><b>For Beginners:</b> For models with many sequential layers (like ResNet or Transformers),
    /// this automatically applies checkpointing efficiently:
    ///
    /// <code>
    /// var layers = new List&lt;Func&lt;ComputationNode&lt;float&gt;, ComputationNode&lt;float&gt;&gt;&gt;
    /// {
    ///     x => layer1.Forward(x),
    ///     x => layer2.Forward(x),
    ///     x => layer3.Forward(x),
    ///     x => layer4.Forward(x)
    /// };
    ///
    /// // Checkpoint every 2 layers
    /// var output = GradientCheckpointing&lt;float&gt;.SequentialCheckpoint(layers, input, segmentSize: 2);
    /// </code>
    /// </para>
    /// </remarks>
    public static ComputationNode<T> SequentialCheckpoint(
        IReadOnlyList<Func<ComputationNode<T>, ComputationNode<T>>> layers,
        ComputationNode<T> input,
        int segmentSize = 2)
    {
        if (layers == null || layers.Count == 0)
        {
            return input;
        }

        if (segmentSize <= 0)
        {
            segmentSize = 1;
        }

        var current = input;
        int numSegments = (layers.Count + segmentSize - 1) / segmentSize;

        for (int seg = 0; seg < numSegments; seg++)
        {
            int startIdx = seg * segmentSize;
            int endIdx = Math.Min(startIdx + segmentSize, layers.Count);

            var segmentLayers = layers.Skip(startIdx).Take(endIdx - startIdx).ToList();
            var segmentInput = current;

            current = Checkpoint(
                () =>
                {
                    var x = segmentInput;
                    foreach (var layer in segmentLayers)
                    {
                        x = layer(x);
                    }
                    return x;
                },
                new[] { segmentInput }
            );
        }

        return current;
    }

    /// <summary>
    /// Estimates memory savings from using gradient checkpointing.
    /// </summary>
    /// <param name="numLayers">Number of layers in the model.</param>
    /// <param name="activationSize">Size of activations per layer in bytes.</param>
    /// <param name="segmentSize">Number of layers per checkpoint segment.</param>
    /// <returns>A tuple of (memory without checkpointing, memory with checkpointing, savings percentage).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This helps you estimate how much memory you'll save:
    ///
    /// <code>
    /// var (without, with, savings) = GradientCheckpointing&lt;float&gt;.EstimateMemorySavings(
    ///     numLayers: 24,
    ///     activationSize: 100_000_000,  // 100MB per layer
    ///     segmentSize: 4
    /// );
    /// Console.WriteLine($"Saves {savings:P1} memory");
    /// </code>
    /// </para>
    /// </remarks>
    public static (long WithoutCheckpoint, long WithCheckpoint, double SavingsPercent) EstimateMemorySavings(
        int numLayers,
        long activationSize,
        int segmentSize = 2)
    {
        // Without checkpointing: store all activations
        long withoutCheckpoint = numLayers * activationSize;

        // With checkpointing: store only sqrt(n) activations plus segment activations
        int numSegments = (numLayers + segmentSize - 1) / segmentSize;
        // Peak memory is: segment activations + checkpoint storage
        long withCheckpoint = (segmentSize * activationSize) + (numSegments * activationSize);

        double savings = 1.0 - (double)withCheckpoint / withoutCheckpoint;

        return (withoutCheckpoint, withCheckpoint, savings * 100);
    }
}

/// <summary>
/// Context information for a checkpoint operation.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
internal class CheckpointContext<T>
{
    /// <summary>
    /// The function to recompute during backward pass.
    /// </summary>
    public Func<ComputationNode<T>>? Function { get; set; }

    /// <summary>
    /// The input nodes to the checkpointed function.
    /// </summary>
    public List<ComputationNode<T>> Inputs { get; set; } = new();

    /// <summary>
    /// Saved tensor values for recomputation.
    /// </summary>
    public Dictionary<ComputationNode<T>, Tensor<T>> SavedTensors { get; set; } = new();

    /// <summary>
    /// The single output node (for single-output checkpoints).
    /// </summary>
    public ComputationNode<T>? Output { get; set; }

    /// <summary>
    /// The saved output value.
    /// </summary>
    public Tensor<T>? OutputValue { get; set; }

    /// <summary>
    /// Multiple output nodes (for multi-output checkpoints).
    /// </summary>
    public List<ComputationNode<T>>? MultiOutputs { get; set; }
}

/// <summary>
/// Provides extension methods for gradient checkpointing on computation nodes.
/// </summary>
public static class CheckpointingExtensions
{
    /// <summary>
    /// Wraps a computation with gradient checkpointing.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="input">The input node.</param>
    /// <param name="function">The function to checkpoint.</param>
    /// <returns>The checkpointed output.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> A convenient way to checkpoint computations:
    ///
    /// <code>
    /// // Instead of:
    /// var output = GradientCheckpointing&lt;float&gt;.Checkpoint(() => layer(input), new[] { input });
    ///
    /// // You can write:
    /// var output = input.WithCheckpoint(x => layer(x));
    /// </code>
    /// </para>
    /// </remarks>
    public static ComputationNode<T> WithCheckpoint<T>(
        this ComputationNode<T> input,
        Func<ComputationNode<T>, ComputationNode<T>> function)
    {
        return GradientCheckpointing<T>.Checkpoint(
            () => function(input),
            new[] { input }
        );
    }

    /// <summary>
    /// Applies a sequence of functions with gradient checkpointing.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="input">The input node.</param>
    /// <param name="functions">The sequence of functions to apply.</param>
    /// <param name="segmentSize">Number of functions per checkpoint segment.</param>
    /// <returns>The final output.</returns>
    public static ComputationNode<T> WithSequentialCheckpoint<T>(
        this ComputationNode<T> input,
        IReadOnlyList<Func<ComputationNode<T>, ComputationNode<T>>> functions,
        int segmentSize = 2)
    {
        return GradientCheckpointing<T>.SequentialCheckpoint(functions, input, segmentSize);
    }
}

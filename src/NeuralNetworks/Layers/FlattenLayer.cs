using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a flatten layer that reshapes multi-dimensional input data into a 1D vector.
/// </summary>
/// <remarks>
/// <para>
/// A flatten layer transforms multi-dimensional input data (such as images or feature maps) into a one-dimensional
/// vector. This is often necessary when transitioning from convolutional layers to fully connected layers
/// in a neural network. The flatten operation preserves all values and their order, just changing the way
/// they are arranged from a multi-dimensional tensor to a single vector.
/// </para>
/// <para><b>For Beginners:</b> A flatten layer converts multi-dimensional data into a simple list of numbers.
/// 
/// Imagine you have a 2D grid of numbers (like a small image):
/// ```
/// [
///   [1, 2, 3],
///   [4, 5, 6]
/// ]
/// ```
/// 
/// The flatten layer turns this into a single row:
/// ```
/// [1, 2, 3, 4, 5, 6]
/// ```
/// 
/// This transformation is needed because:
/// - Convolutional layers work with 2D or 3D data (like images)
/// - Fully connected layers expect a simple list of numbers
/// - Flatten layers bridge these two types of layers
/// 
/// Think of it like taking a book (a 3D object with pages) and reading all the text 
/// in order from beginning to end (a 1D sequence). All the information is preserved,
/// but it's rearranged into a different shape.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.Structural)]
[LayerTask(LayerTask.Projection)]
[LayerProperty(IsTrainable = false, ChangesShape = true, TestInputShape = "1, 2, 2", TestConstructorArgs = "")]
// Roles and ranks come from ForwardTraced, which is explicit about the two cases it handles:
// rank 1 returns the input untouched ("Already 1D: nothing to flatten"), and every rank >= 2
// keeps Shape[0] as the batch and multiplies the remainder into one axis
// (`Engine.Reshape(input, [batchSize, actualOutputSize])`). So the OUTPUT is always rank 1 or
// rank 2 - one declaration with BatchOptional covers both - while the INPUT is declared once per
// accepted rank because the axes being multiplied need distinct roles for Product to name them.
[TensorLayout(TensorAxis.Batch, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Input)]
// ONE BatchOptional declaration covers rank 4 (batched [B,C,H,W]) AND rank 3 (per-sample [C,H,W]),
// which is what AxesForRank already does and what chain resolution already passes. Declaring rank 3
// separately as [Batch, Height, Width] claimed the leading axis was a batch when in a per-sample chain
// it is Channels - so the contract flattened two axes where the layer flattens three.
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Output,
    Note = "Batch is preserved and everything after it is collapsed into a single feature axis.")]
[AutoParameters]
public partial class FlattenLayer<T> : LayerBase<T>, IBatchAwareShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// HAND-WRITTEN because the relation is a Product, which no generator can infer: the size of the
    /// output feature axis is the product of every input axis after the batch. Read straight off
    /// ForwardTraced's `for (int i = 1; i &lt; input.Shape.Length; i++) actualOutputSize *= input.Shape[i];`.
    /// </para>
    /// <para>
    /// Rank 1 is a genuine identity - the method returns the input object itself - and rank 2 degenerates
    /// to Same(Features), because a product over a single axis IS that axis. Both are stated explicitly
    /// rather than folded into the general case so the contract says what the code does at each rank.
    /// </para>
    /// <para>
    /// Ranks above four decline rather than guess: a fifth leading axis would need a fifth DISTINCT role
    /// to be nameable, and two anonymous placeholders in one layout cannot be told apart.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
        => OutputAxesFor(inputRank, isBatched: true);

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// THIS LAYER IS WHY THE isBatched FLAG EXISTS. It collapses everything AFTER the batch axis, so
    /// the answer depends entirely on whether the leading axis is one. A rank-3 tensor <c>[3,8,9]</c>
    /// handed to <c>Forward</c> is one batch and two feature axes, giving <c>[3,72]</c>. A rank-3
    /// PER-SAMPLE shape <c>[32,7,7]</c> from chain resolution is <c>[Channels, Height, Width]</c> with
    /// no batch at all, giving <c>[1568]</c>. Both are correct; rank alone cannot distinguish them, and
    /// a contract forced to pick one was wrong for whichever caller it did not pick.
    /// </para>
    /// <para>
    /// Un-batched: every axis collapses into one. Batched: the batch is carried and the rest collapse.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank, bool isBatched)
    {
        OutputAxisContract Batch() => new(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch));

        if (!isBatched)
        {
            // No batch axis: the whole shape becomes one feature axis.
            return inputRank switch
            {
                1 => new[]
                {
                    new OutputAxisContract(TensorAxis.Features, AxisRelation.Same(TensorAxis.Features)),
                },
                2 => new[]
                {
                    new OutputAxisContract(
                        TensorAxis.Features,
                        AxisRelation.Product(TensorAxis.Batch, TensorAxis.Features)),
                },
                3 => new[]
                {
                    new OutputAxisContract(
                        TensorAxis.Features,
                        AxisRelation.Product(
                            TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width)),
                },
                _ => null,
            };
        }

        return inputRank switch
        {
            1 => new[]
            {
                new OutputAxisContract(TensorAxis.Features, AxisRelation.Same(TensorAxis.Features)),
            },
            2 => new[]
            {
                Batch(),
                new OutputAxisContract(TensorAxis.Features, AxisRelation.Same(TensorAxis.Features)),
            },
            // MATCHES ForwardTraced, which is what actually runs: it preserves axis 0 and multiplies
            // the rest (`for (int i = 1; i < input.Shape.Length; i++)`), so a rank-3 tensor [3,8,9]
            // comes back [3,72].
            //
            // THE LAYER DISAGREES WITH ITSELF AND THIS CONTRACT CANNOT FIX THAT. OnFirstForward
            // multiplies from index 0 (`for (int i = 0; ...) _outputSize *= shape[i]`), so it declares
            // [216] for the same rank-3 input. The two are reached by different paths - OnFirstForward
            // via ResolveShapesOnly during chain resolution (per-sample shapes) AND via the first real
            // forward (a batched tensor) - so it is applying one convention to both. The contract
            // follows the FORWARD because that is the shape callers actually receive; the shadow
            // comparison reports the residual disagreement rather than hiding it.
            3 => new[]
            {
                Batch(),
                new OutputAxisContract(
                    TensorAxis.Features, AxisRelation.Product(TensorAxis.Height, TensorAxis.Width)),
            },
            4 => new[]
            {
                Batch(),
                new OutputAxisContract(
                    TensorAxis.Features,
                    AxisRelation.Product(TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width)),
            },
            _ => null,
        };
    }

    /// <summary>
    /// The shape of the input tensor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This array stores the dimensions of the input tensor (excluding the batch dimension).
    /// It is used during the forward and backward passes to correctly flatten and unflatten the tensors.
    /// </para>
    /// <para><b>For Beginners:</b> This remembers the original shape of the input data.
    /// 
    /// For example:
    /// - For a 28×28 grayscale image: [28, 28, 1]
    /// - For RGB color channels: [height, width, 3]
    /// - For a feature map with multiple channels: [height, width, channels]
    /// 
    /// The layer needs to store this original shape:
    /// - To correctly convert multi-dimensional data to a flat vector
    /// - To convert gradients back to the original shape during training
    /// 
    /// It's like keeping a map of how the data was originally organized so you
    /// can "unfold" it in exactly the same way later.
    /// </para>
    /// </remarks>
    private int[] _inputShape;

    /// <summary>
    /// The size of the output vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the total size of the flattened output vector, which is the product of all
    /// dimensions in the input shape. It represents the number of elements in the input tensor
    /// for a single example.
    /// </para>
    /// <para><b>For Beginners:</b> This is the total number of values after flattening.
    /// 
    /// The output size is calculated by multiplying all the dimensions of the input:
    /// - For a 28×28 image: 28 × 28 = 784 values
    /// - For a 16×16×32 feature map: 16 × 16 × 32 = 8,192 values
    /// 
    /// This number tells us:
    /// - How long the flattened vector will be
    /// - How many neurons the next layer (usually a fully connected layer) will receive
    /// 
    /// Pre-calculating this size makes processing more efficient.
    /// </para>
    /// </remarks>
    private int _outputSize;

    /// <summary>
    /// The input tensor from the last forward pass, saved for backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This tensor stores the input received during the last forward pass. It is necessary
    /// for computing gradients during the backward pass, as it provides information about
    /// the original shape of the data.
    /// </para>
    /// <para><b>For Beginners:</b> This remembers what input data was processed most recently.
    /// 
    /// During training:
    /// - The layer needs to remember the shape and organization of its input
    /// - This helps when calculating how to send gradients back to previous layers
    /// - Without this information, the layer couldn't "unflatten" the gradients correctly
    /// 
    /// This is automatically cleared between training batches to save memory.
    /// </para>
    /// </remarks>
    private Tensor<T>? _lastInput;

    // GPU-resident cached tensors for GPU training pipeline
    private int[]? _lastInputGpuShape;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>false</c> because flatten layers have no trainable parameters.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that the flatten layer does not have any trainable parameters.
    /// The layer simply performs a reshape operation and does not learn during training.
    /// However, it still participates in backpropagation by passing gradients back to previous
    /// layers in the correct shape.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you that this layer doesn't learn or change during training.
    /// 
    /// A value of false means:
    /// - The layer has no weights or biases to adjust
    /// - It performs the same operation regardless of training
    /// - It's a fixed transformation layer, not a learning layer
    /// 
    /// Unlike convolutional or fully connected layers (which learn patterns from data),
    /// the flatten layer just reorganizes data without changing its content.
    /// 
    /// It's like rearranging furniture in a room - you're not adding or removing
    /// anything, just changing how it's organized.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => false;

    /// <summary>
    /// Initializes a new instance of the <see cref="FlattenLayer{T}"/> class.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor (excluding the batch dimension).</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new flatten layer that will reshape input data with the specified shape
    /// into a one-dimensional vector. The output size is calculated as the product of all dimensions in
    /// the input shape. The layer expects input tensors with shape [batchSize, ...inputShape].
    /// </para>
    /// <para><b>For Beginners:</b> This sets up the flatten layer by specifying what shape of data it will receive.
    /// 
    /// When creating a flatten layer, you need to specify:
    /// - The dimensions of your input data (not counting the batch size)
    /// 
    /// For example:
    /// ```csharp
    /// // Create a flatten layer for 28×28 grayscale images
    /// var flattenLayer = new FlattenLayer<float>();
    /// 
    /// // Create a flatten layer for output from a convolutional layer with 64 feature maps of size 7×7
    /// var flattenConvOutput = new FlattenLayer<float>();
    /// ```
    /// 
    /// The constructor automatically calculates how large the output vector will be
    /// by multiplying all the dimensions together.
    /// </para>
    /// </remarks>
    public FlattenLayer()
        : base(new[] { -1 }, new[] { -1 })
    {
        _inputShape = Array.Empty<int>();
        _outputSize = -1;
    }

    /// <summary>
    /// Resolves shape on first forward by reading input.Shape and computing the flattened output size.
    /// </summary>
    protected override void OnFirstForward(Tensor<T> input)
    {
        BindToActualInput(input);
    }

    /// <inheritdoc />
    protected override void ReconcileShapeOnlyResolution(Tensor<T> input)
    {
        // ResolveLazyLayerShapes receives per-sample shapes, while the real forward receives a
        // batched tensor and Flatten(start_dim=1) preserves its leading axis. Recompute from the
        // tensor that will actually be flattened instead of retaining a sequential-walk estimate
        // from a custom/branched model forward.
        BindToActualInput(input);
    }

    private void BindToActualInput(Tensor<T> input)
    {
        var shape = input.Shape.ToArray();
        if (shape.Length == 0)
            throw new ArgumentException("FlattenLayer cannot flatten a rank-0 tensor.", nameof(input));

        int firstFeatureAxis = shape.Length == 1 ? 0 : 1;
        _inputShape = shape.Skip(firstFeatureAxis).ToArray();
        _outputSize = 1;
        for (int i = firstFeatureAxis; i < shape.Length; i++)
            _outputSize = checked(_outputSize * shape[i]);

        ResolveShapes(_inputShape, new[] { _outputSize });
    }

    /// <summary>
    /// Performs the forward pass of the flatten layer, reshaping multi-dimensional data into a vector.
    /// </summary>
    /// <param name="input">The input tensor to flatten. Shape: [batchSize, ...inputShape].</param>
    /// <returns>The flattened output tensor. Shape: [batchSize, outputSize].</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the flatten layer. It takes a multi-dimensional tensor
    /// and reshapes it into a 2D tensor where each row corresponds to a flattened example from the batch.
    /// For unbatched inputs (rank <= 3), it returns a 1D vector of length input.Length. The values are
    /// preserved and their order is maintained according to a row-major traversal of the input tensor.
    /// The input tensor is cached for use during the backward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This method converts multi-dimensional data into simple vectors.
    /// 
    /// The forward pass works like this:
    /// 1. Take multi-dimensional input (like a 3D image)
    /// 2. For each example in the batch:
    ///    - Go through all positions in the multi-dimensional input
    ///    - Place each value into the corresponding position in a flat vector
    /// 3. Return a tensor with shape [batchSize, flattenedSize]
    /// 
    /// For example, with a batch of 3D data like [batchSize, height, width, channels]:
    /// - Input shape: [32, 7, 7, 64] (32 examples, each 7×7 with 64 channels)
    /// - Output shape: [32, 3136] (32 examples, each with 7×7—64=3136 values)
    /// - For an unbatched input like [7, 7, 64], the output is a 1D vector of length 3136
    /// 
    /// The method carefully preserves the order of values so they can be
    /// "unflattened" back to the original shape during backpropagation.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        EnsureInitializedFromInput(input);

        // Flatten has no weights and is intentionally shape-polymorphic. A model can legitimately
        // use the same instance for different public-input layouts (for example, a fused inference
        // path followed by the canonical training path). Keep the reported per-sample contract in
        // lockstep with every real tensor, not only the first one; otherwise the reshape succeeds
        // while the layer continues to report a stale feature width to downstream validation.
        int firstFeatureAxis = input.Rank == 1 ? 0 : 1;
        int actualOutputSize = 1;
        for (int i = firstFeatureAxis; i < input.Shape.Length; i++)
            actualOutputSize = checked(actualOutputSize * input.Shape[i]);
        bool inputExtentsMatch = _inputShape.Length == input.Rank - firstFeatureAxis;
        for (int i = 0; inputExtentsMatch && i < _inputShape.Length; i++)
            inputExtentsMatch = _inputShape[i] == input.Shape[i + firstFeatureAxis];
        if (_outputSize != actualOutputSize
            || !inputExtentsMatch)
        {
            BindToActualInput(input);
        }

        _lastInput = ShouldCacheForBackward ? input : null; // #1668: skip in inference (arena safety)

        // Handle truly unbatched 1D input
        if (input.Rank == 1)
        {
            // Already 1D: nothing to flatten
            _outputSize = input.Length;
            return input;
        }

        // For rank >= 2: preserve first dimension (batch) and flatten the rest
        // This matches PyTorch nn.Flatten(start_dim=1) behavior
        int batchSize = input.Shape[0];
        _outputSize = actualOutputSize;
        return Engine.Reshape(input, [batchSize, actualOutputSize]);
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the layer by clearing the cached input
    /// from the previous forward pass. This is useful when starting to process a new batch of
    /// data or when switching between training and inference modes.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    ///
    /// When resetting the state:
    /// - The saved input is cleared
    /// - The layer forgets the previous data it processed
    /// - This frees up memory and prepares for new data
    ///
    /// This is typically called:
    /// - Between training batches
    /// - When switching from training to evaluation mode
    /// - When starting to process completely new data
    ///
    /// It's like wiping a whiteboard clean before starting a new calculation.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward pass
        _lastInput = null;
        _lastInputGpuShape = null;
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    /// <value>
    /// Always <c>true</c> because flatten is a zero-copy reshape that can be done via GPU tensor view.
    /// </value>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Performs the forward pass on GPU using a zero-copy view reshape.
    /// </summary>
    /// <param name="input">The GPU-resident input tensor.</param>
    /// <returns>A GPU tensor view with the flattened shape.</returns>
    /// <remarks>
    /// <para>
    /// This method implements GPU-resident flatten by creating a view into the input tensor
    /// with the flattened shape. No data is copied - only the shape interpretation changes.
    /// </para>
    /// <para><b>For Beginners:</b> The GPU version of flatten is very efficient because:
    /// - It doesn't move any data
    /// - It just tells the GPU "interpret this same data with a different shape"
    /// - This is called a "view" operation
    ///
    /// For example, if input has shape [32, 7, 7, 64], the view will have shape [32, 3136]
    /// but still points to the exact same memory on the GPU.
    /// </para>
    /// </remarks>
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        var input = inputs[0];

        // Cache input shape for BackwardGpu
        if (IsTrainingMode)
        {
            _lastInputGpuShape = input._shape.ToArray();
        }

        // Handle unbatched input (3D: [C, H, W] or 2D: [H, W] or 1D)
        if (input.Shape.Length <= 3)
        {
            // Unbatched input: flatten to 1D vector
            return input.Reshape([input.Length]);
        }

        // Batched input: flatten spatial dimensions keeping batch dimension
        int batchSize = input.Shape[0];
        int flattenedSize = input.Length / batchSize;
        return input.Reshape([batchSize, flattenedSize]);
    }
}

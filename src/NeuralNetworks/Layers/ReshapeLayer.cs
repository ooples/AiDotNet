using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a reshape layer that transforms the dimensions of input data without changing its content.
/// </summary>
/// <remarks>
/// <para>
/// The ReshapeLayer rearranges the elements of the input tensor into a new shape without changing the data itself.
/// This operation is useful for connecting layers with different shape requirements or for preparing data for
/// specific layer types. The total number of elements must remain the same between the input and output shapes.
/// </para>
/// <para><b>For Beginners:</b> This layer changes how your data is organized without changing the data itself.
/// 
/// Think of the ReshapeLayer like reorganizing a deck of playing cards:
/// - If you have cards arranged in 4 rows of 13 cards (representing the 4 suits)
/// - You could reorganize them into 13 rows of 4 cards (representing the 13 ranks)
/// - The cards themselves haven't changed, just how they're arranged
/// 
/// For example, in image processing:
/// - You might have an image of shape [height, width, channels]
/// - But a particular layer might need the data as a flat vector
/// - A reshape layer can convert between these formats without losing information
/// 
/// Common use cases include:
/// - Flattening data (e.g., converting a 2D image to a 1D vector for a dense layer)
/// - Reshaping for convolutional operations (e.g., turning a vector into a 3D tensor)
/// - Batch dimension manipulation (e.g., splitting or combining batch items)
/// 
/// The key requirement is that the total number of elements stays the same - you're just
/// reorganizing them into a different dimensional structure.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.Structural)]
[LayerTask(LayerTask.Projection)]
[LayerProperty(IsTrainable = false, ChangesShape = true, TestInputShape = "1, 4", TestConstructorArgs = "new[] { 2, 2 }")]
// POSITIONAL roles on both sides. A general reshape has no intrinsic axis meanings - that is the whole
// point of it - and an earlier pass declined to annotate it for exactly that reason, on the grounds
// that several role-less axes would all have to be TensorAxis.Other, which ADNSHAPE002 forbids
// repeating. But they do not have to be Other: distinct positional names work, and they are SAFE here
// for a stronger reason than usual. Every non-batch output relation below is Fixed, read from the
// constructor's target shape, so NO relation reads an input axis at all - the names cannot influence a
// resolved size even in principle. This is the same reasoning [ElementWiseShape] and ConcatenateLayer
// already rely on, applied where it is least ambiguous.
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height,
    Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class ReshapeLayer<T> : LayerBase<T>, IShapeContract
{
    /// <summary>
    /// The shape of the input tensor, excluding the batch dimension.
    /// </summary>
    /// <remarks>
    /// This array stores the dimensions of the input tensor not including the batch dimension (which is 
    /// always the first dimension). It is used to validate input shapes and to perform the reshaping operation.
    /// </remarks>
    private int[] _inputShape;

    /// <summary>
    /// The shape of the output tensor, excluding the batch dimension.
    /// </summary>
    /// <remarks>
    /// This array stores the dimensions of the output tensor not including the batch dimension (which is 
    /// always the first dimension). It defines the target shape for the reshaping operation.
    /// </remarks>
    /// <summary>Positional roles, matching the layouts declared on the class.</summary>
    private static readonly TensorAxis[] ReshapeRoles =
    {
        TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    };

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// The output is <c>[batch, .._outputShape]</c> - see <c>ForwardTraced</c>, which builds exactly
    /// that target. So the batch axis carries through and every other axis is
    /// <see cref="AxisRelation.Fixed"/> at the size the constructor was given. That is a legitimate
    /// <c>Fixed</c> source under the annotation rules: it is a constructor argument, not an observed
    /// constant.
    /// </para>
    /// <para>
    /// THE OUTPUT RANK DOES NOT DEPEND ON THE INPUT RANK - it is <c>_outputShape.Length + 1</c>,
    /// whatever came in, which is why the same list is returned for every accepted input rank. The
    /// layer constrains its input only by per-sample element count (<c>OnFirstForward</c> multiplies
    /// the non-batch dimensions and compares), and no relation in the vocabulary states "the product of
    /// these axes equals the product of those" - but nothing needs to, because no output axis is
    /// derived from an input axis.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank < 2 || inputRank > ReshapeRoles.Length) return null;

        var target = _outputShape;
        if (target is null || target.Length == 0) return null;
        if (target.Length + 1 > ReshapeRoles.Length) return null;

        var axes = new List<OutputAxisContract>(target.Length + 1)
        {
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
        };

        for (int i = 0; i < target.Length; i++)
        {
            if (target[i] <= 0) return null;
            axes.Add(new OutputAxisContract(ReshapeRoles[i + 1], AxisRelation.Fixed(target[i])));
        }

        return axes;
    }

    private int[] _outputShape;

    /// <summary>
    /// Stores the input tensor from the most recent forward pass for use in backpropagation.
    /// </summary>
    /// <remarks>
    /// This cached input is needed during the backward pass to compute the appropriate gradients.
    /// The tensor is null before the first forward pass or after a reset.
    /// </remarks>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Cached GPU input shape for backward pass.
    /// </summary>
    private int[]? _gpuCachedInputShape;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    /// <value>
    /// Always <c>true</c> for ReshapeLayer, indicating that the layer can participate in backpropagation.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property indicates that the ReshapeLayer supports the training process through backpropagation.
    /// While the layer itself has no trainable parameters, it needs to properly propagate gradients during
    /// the backward pass, reshaping them to match the input shape.
    /// </para>
    /// <para><b>For Beginners:</b> This property tells you if the layer can participate in the learning process.
    /// 
    /// A value of true means:
    /// - The layer can pass learning signals (gradients) backward through it
    /// - It contributes to the training of the entire network
    /// 
    /// While this layer doesn't have any internal values that it learns directly,
    /// it's designed to let learning signals flow through it to previous layers.
    /// It just needs to reshape these signals to match the original input shape.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="ReshapeLayer{T}"/> class.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor, excluding the batch dimension.</param>
    /// <param name="outputShape">The shape of the output tensor, excluding the batch dimension.</param>
    /// <exception cref="ArgumentException">Thrown when the total number of elements in the input and output shapes are not equal.</exception>
    /// <remarks>
    /// <para>
    /// This constructor creates a new ReshapeLayer with the specified input and output shapes. It validates that
    /// the total number of elements remains the same between the input and output shapes, as the layer can only
    /// rearrange elements, not create or remove them.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new reshape layer for your neural network.
    /// 
    /// When you create this layer, you specify:
    /// - inputShape: The current organization of your data (not including the batch dimension)
    /// - outputShape: The desired organization of your data (not including the batch dimension)
    /// 
    /// For example:
    /// - If inputShape is [28, 28] (like a 28×28 image)
    /// - You could set outputShape to [784] to flatten it into a single vector
    /// 
    /// The constructor checks that the total number of elements stays the same:
    /// - For the example above, 28×28 = 784, so the shapes are compatible
    /// - If the total elements don't match, you'll get an error
    /// 
    /// The batch dimension (first dimension) is handled automatically and not included in these shapes.
    /// </para>
    /// </remarks>
    public ReshapeLayer(int[] outputShape)
        : base(MakeUnknown(outputShape.Length), outputShape)
    {
        _inputShape = Array.Empty<int>();
        _outputShape = outputShape;
    }

    private static int[] MakeUnknown(int rank)
    {
        var s = new int[rank];
        for (int i = 0; i < rank; i++) s[i] = -1;
        return s;
    }

    /// <summary>
    /// Resolves input shape on first forward; validates element-count compatibility with target output.
    /// </summary>
    protected override void OnFirstForward(Tensor<T> input)
    {
        var shape = input.Shape.ToArray();
        // shape[0] is the batch dimension; the reshape applies PER-SAMPLE and the layer's
        // shapes exclude batch (see the class remarks). The previous code multiplied the
        // whole input shape — including batch — and compared it to the per-sample output,
        // so a batched input like [2, 8, 4] (per-sample 32) was rejected against output
        // [32] because 2*8*4 = 64 != 32. Count per-sample elements (dims 1..) instead.
        int inElems = 1;
        for (int i = 1; i < shape.Length; i++) inElems *= shape[i];
        int outElems = 1;
        for (int i = 0; i < _outputShape.Length; i++) outElems *= _outputShape[i];
        if (inElems != outElems)
            throw new ArgumentException(
                $"ReshapeLayer per-sample input element count ({inElems}) does not match output element count ({outElems}).",
                nameof(input));

        var perSample = new int[shape.Length - 1];
        Array.Copy(shape, 1, perSample, 0, perSample.Length);
        _inputShape = perSample;
        ResolveShapes(perSample, _outputShape);
    }

    /// <summary>
    /// Gets the target shape for the reshape operation.
    /// </summary>
    /// <returns>The target shape array (excluding batch dimension).</returns>
    public int[] GetTargetShape()
    {
        return _outputShape;
    }

    /// <summary>
    /// Performs the forward pass of the reshape layer.
    /// </summary>
    /// <param name="input">The input tensor to reshape.</param>
    /// <returns>The reshaped output tensor.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the forward pass of the reshape layer. It creates a new tensor with the specified
    /// output shape and copies the elements from the input tensor into the output tensor while preserving their
    /// order. The input tensor is cached for use during the backward pass.
    /// </para>
    /// <para><b>For Beginners:</b> This method reorganizes your data into the new shape.
    /// 
    /// During the forward pass:
    /// 1. The layer saves the original input for later use in the backward pass
    /// 2. It creates a new, empty tensor with the target shape
    /// 3. It copies all values from the input to the output tensor
    /// 4. The data values themselves stay exactly the same, just arranged differently
    /// 
    /// The layer handles each item in your batch separately, maintaining the batch structure.
    /// 
    /// Think of it like pouring water from one differently-shaped container to another - 
    /// the amount of water stays the same, but it takes the shape of the new container.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        EnsureInitializedFromInput(input);
        _lastInput = ShouldCacheForBackward ? input : null; // #1668: skip in inference (arena safety)
        int batchSize = input.Shape[0];
        int[] targetShape = new int[_outputShape.Length + 1];
        targetShape[0] = batchSize;
        Array.Copy(_outputShape, 0, targetShape, 1, _outputShape.Length);

        return Engine.Reshape(input, targetShape);
    }

    /// <summary>
    /// Resets the internal state of the reshape layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method resets the internal state of the reshape layer, clearing the cached input tensor from the
    /// forward pass. This is useful when starting to process a new batch of data.
    /// </para>
    /// <para><b>For Beginners:</b> This method clears the layer's memory to start fresh.
    /// 
    /// When resetting the state:
    /// - The saved input from the previous forward pass is cleared
    /// - The layer forgets any information from previous batches
    /// 
    /// This is important for:
    /// - Processing a new, unrelated batch of data
    /// - Preventing information from one batch affecting another
    /// - Managing memory usage efficiently
    /// 
    /// Since this layer has no learned parameters, resetting just clears the temporarily
    /// stored input that was used for the backward pass.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        // Clear cached values from forward pass
        _lastInput = null;
        _gpuCachedInputShape = null;
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    /// <value>
    /// Always <c>true</c> because reshape is a zero-copy operation that can be done via GPU tensor view.
    /// </value>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Performs the forward pass on GPU using a zero-copy view reshape.
    /// </summary>
    /// <param name="input">The GPU-resident input tensor.</param>
    /// <returns>A GPU tensor view with the reshaped dimensions.</returns>
    /// <remarks>
    /// <para>
    /// This method implements GPU-resident reshape by creating a view into the input tensor
    /// with the target shape. No data is copied - only the shape interpretation changes.
    /// </para>
    /// <para><b>For Beginners:</b> The GPU version of reshape is very efficient because:
    /// - It doesn't move any data
    /// - It just tells the GPU "interpret this same data with a different shape"
    /// - This is called a "view" operation
    ///
    /// For example, if input has shape [32, 28, 28, 1] and target is [784],
    /// the view will have shape [32, 784] but still points to the same GPU memory.
    /// </para>
    /// </remarks>
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        var input = inputs[0];

        // Cache input shape for backward pass
        if (IsTrainingMode)
        {
            _gpuCachedInputShape = (int[])input._shape.Clone();
        }

        // Calculate full target shape including batch dimension
        int batchSize = input.Shape[0];
        int[] targetShape = new int[_outputShape.Length + 1];
        targetShape[0] = batchSize;
        Array.Copy(_outputShape, 0, targetShape, 1, _outputShape.Length);

        return input.Reshape(targetShape);
    }
}

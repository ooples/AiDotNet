using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Splits input along the feature axis into two equal halves, processes each half through its own
/// independent sub-network (stream), and concatenates the two stream outputs.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Many real-world neural networks need to process two different types of
/// information at the same time. For example:
/// - An audio-visual model processes audio features and video features separately
/// - A multi-modal model processes text embeddings and image embeddings in parallel
/// - A siamese network processes two inputs through shared or separate encoders
///
/// This layer solves the problem of expressing parallel processing within a sequential layer stack.
/// Instead of needing complex custom forward logic, you can simply place this layer in your network
/// and it will:
/// 1. <b>Split</b> the input features into two equal halves (e.g., [256 audio | 256 visual])
/// 2. <b>Process</b> each half through its own set of layers (Stream A and Stream B)
/// 3. <b>Concatenate</b> the two outputs back together
///
/// All operations are tracked on the gradient tape so backpropagation works correctly through
/// both streams simultaneously.
///
/// <b>Example:</b> If you have 512 input features representing concatenated audio+visual features:
/// <code>
/// var audioLayers = new[] { new DenseLayer&lt;double&gt;(256, 128, relu) };
/// var videoLayers = new[] { new DenseLayer&lt;double&gt;(256, 128, relu) };
/// var parallel = new ParallelStreamsLayer&lt;double&gt;(512, 128, 128, audioLayers, videoLayers);
/// // Input: [batch, 512] -> Output: [batch, 256] (128 from each stream)
/// </code>
/// </para>
/// <para>
/// <b>How it works internally:</b>
/// <list type="number">
/// <item><description>The input tensor is sliced along the last axis into two equal halves using
/// <c>Engine.TensorSlice</c> (tape-tracked)</description></item>
/// <item><description>Each half is passed through its respective stream's layers sequentially</description></item>
/// <item><description>The two stream outputs are concatenated along the last axis using
/// <c>Engine.TensorConcatenate</c> (tape-tracked)</description></item>
/// </list>
/// </para>
/// <para>
/// <b>Gradient flow:</b> Because the split and concatenation use Engine operations that record on
/// the autodiff tape, gradients flow correctly through both streams during backpropagation.
/// Each stream receives only the gradients relevant to its portion of the input, ensuring proper
/// learning for both sub-networks.
/// </para>
/// </remarks>
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, Cost = ComputeCost.High)]
// A TWO-STREAM decorator. ForwardTraced slices the last axis in half, runs each half through its own
// sub-layer list, and returns "Engine.TensorConcatenate([outputA, outputB], axis: rank - 1)" - so the
// leading axes are the input's untouched and the trailing axis is the SUM of the two streams' widths.
//
// THE WIDTHS ARE READ FROM THE STREAMS, NEVER FROM THE CONSTRUCTOR. streamAOutputSize and
// streamBOutputSize only reach base(...) as the declared output shape; ForwardTraced validates the
// INPUT feature size against _splitSize and never checks either output size, so a stream whose last
// layer disagrees with the number it was constructed with produces the layer's real width and the
// constructor argument is simply stale. Declaring Fixed off those arguments would be right for one
// configuration and wrong for the layer - the same defect SequenceLastLayer was carrying.
//
// Rank 1 and rank 2 are declared because those are the forms the forward documents and handles
// ("[batch, features] or [features]"). Higher ranks run too, but the leading roles would have to be
// invented, so they decline.
[TensorLayout(TensorAxis.Batch, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Input,
    Note = "The trailing axis must equal _splitSize * 2; the forward rejects anything else.")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Output,
    Note = "Trailing width is stream A's output width plus stream B's.")]
[AutoParameters]
public partial class ParallelStreamsLayer<T> : LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Composed from BOTH streams, because the concatenation's joined axis is the sum of what each
    /// produced - <c>AxisRelation.SumOf</c> is the form for exactly that. Each term is the trailing
    /// relation of the LAST layer of its stream, asked at this layer's rank, because the slice
    /// preserves rank and each stream is run to completion before the join.
    /// </para>
    /// <para>
    /// A term is accepted ONLY when it is <c>Fixed</c>. A stream ending in a shape-preserving layer
    /// reports its trailing axis as <c>Same(Features)</c>, and that would be resolved against the WHOLE
    /// input's feature count - twice what the stream actually saw, since the input was halved before
    /// the stream ever ran. Half of a contract that is right and half that is off by a factor of two is
    /// not a contract, so this declines instead.
    /// </para>
    /// <para>
    /// Only expressible because <c>OutputAxesFor</c> is an INSTANCE method: the widths belong to the
    /// sub-networks this layer was handed, and no type-level attribute could see them.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        // Only the two forms the forward documents; the leading roles above cover no other rank.
        if (inputRank is not (1 or 2)) return null;

        var widthA = TrailingWidthOf(_streamA, inputRank);
        var widthB = TrailingWidthOf(_streamB, inputRank);
        if (widthA is null || widthB is null) return null;

        var features = new OutputAxisContract(
            TensorAxis.Features, AxisRelation.SumOf(widthA, widthB));

        return inputRank == 1
            ? new[] { features }
            : new[]
            {
                // The slice and the concatenation both act on the LAST axis only, so the batch axis
                // arrives and leaves untouched.
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                features,
            };
    }

    /// <summary>
    /// The width one stream contributes to the concatenation, or <c>null</c> when it cannot be stated.
    /// </summary>
    /// <remarks>
    /// Read off the stream's LAST layer, which is what <c>RunStream</c> returns. Anything other than a
    /// <c>Fixed</c> trailing relation is refused: a relation that reads an input axis would be resolved
    /// against this layer's input rather than the halved slice the stream was given.
    /// </remarks>
    private static AxisRelation? TrailingWidthOf(List<ILayer<T>> stream, int inputRank)
    {
        if (stream.Count == 0) return null;

        var axes = (stream[stream.Count - 1] as IShapeContract)?.OutputAxesFor(inputRank);

        // A stream that changes rank invalidates the question: the trailing relation would then be
        // answering about a differently shaped tensor than the one being concatenated at rank - 1.
        if (axes is null || axes.Count != inputRank) return null;

        var trailing = axes[axes.Count - 1].Relation;

        return trailing.Kind == AxisRelation.Form.Fixed ? trailing : null;
    }

    /// <summary>
    /// The layers that process the first half of the input features.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Stream A is the first "branch" of the parallel processor. It receives
    /// the first half of the input features and transforms them through its own sequence of layers.
    /// For example, in an audio-visual model, this might be the audio encoder.
    /// </para>
    /// </remarks>
    private readonly List<ILayer<T>> _streamA;

    /// <summary>
    /// The layers that process the second half of the input features.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Stream B is the second "branch" of the parallel processor. It receives
    /// the second half of the input features and transforms them independently from Stream A.
    /// For example, in an audio-visual model, this might be the visual encoder.
    /// </para>
    /// </remarks>
    private readonly List<ILayer<T>> _streamB;

    /// <summary>
    /// Half the input feature dimension — the number of features each stream receives.
    /// </summary>
    private readonly int _splitSize;

    /// <summary>Construction state: the 'inputSize' the layer was built with.</summary>
    private readonly int _inputSize;

    /// <summary>Construction state: the 'streamAOutputSize' the layer was built with.</summary>
    private readonly int _streamAOutputSize;

    /// <summary>Construction state: the 'streamBOutputSize' the layer was built with.</summary>
    private readonly int _streamBOutputSize;

    /// <summary>
    /// Creates a parallel streams layer that splits input features and processes each half independently.
    /// </summary>
    /// <param name="inputSize">Total input feature dimension. Must be even so the input can be split
    /// into two equal halves. For example, 512 means each stream receives 256 features.</param>
    /// <param name="streamAOutputSize">Output dimension of stream A. This is determined by the last
    /// layer in the stream A sub-network.</param>
    /// <param name="streamBOutputSize">Output dimension of stream B. This is determined by the last
    /// layer in the stream B sub-network.</param>
    /// <param name="streamALayers">The sequence of layers for stream A. These layers are executed in
    /// order, with each layer's output feeding into the next. The first layer must accept
    /// <c>inputSize / 2</c> features as input.</param>
    /// <param name="streamBLayers">The sequence of layers for stream B. Same rules as stream A — the
    /// first layer must accept <c>inputSize / 2</c> features.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Think of this like a Y-shaped pipe:
    /// - The input flows in at the top and gets split into two pipes
    /// - Each pipe has its own set of filters (layers) that transform the data
    /// - At the bottom, the two pipes merge back into one
    ///
    /// The total output size is <c>streamAOutputSize + streamBOutputSize</c>.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when <paramref name="inputSize"/> is odd.</exception>
    public ParallelStreamsLayer(
        int inputSize,
        int streamAOutputSize,
        int streamBOutputSize,
        IEnumerable<ILayer<T>> streamALayers,
        IEnumerable<ILayer<T>> streamBLayers)
        : base([inputSize], [streamAOutputSize + streamBOutputSize])
    {
        _streamBOutputSize = streamBOutputSize;
        _streamAOutputSize = streamAOutputSize;
        _inputSize = inputSize;
        // Fail fast at the boundary on bad inputs so misuse surfaces here, not deep in
        // the forward pass where the error message would be cryptic shape mismatches.
        if (inputSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputSize), "Input size must be positive.");
        if (streamAOutputSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(streamAOutputSize), "Stream A output size must be positive.");
        if (streamBOutputSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(streamBOutputSize), "Stream B output size must be positive.");
        if (streamALayers is null)
            throw new ArgumentNullException(nameof(streamALayers));
        if (streamBLayers is null)
            throw new ArgumentNullException(nameof(streamBLayers));
        if (inputSize % 2 != 0)
            throw new ArgumentException("Input size must be even for equal split.", nameof(inputSize));

        _splitSize = inputSize / 2;
        _streamA = new List<ILayer<T>>(streamALayers);
        _streamB = new List<ILayer<T>>(streamBLayers);

        if (_streamA.Any(l => l is null) || _streamB.Any(l => l is null))
            throw new ArgumentException("Stream layer collections must not contain null entries.");

        foreach (var layer in _streamA) RegisterSubLayer(layer);
        foreach (var layer in _streamB) RegisterSubLayer(layer);
    }

    /// <summary>
    /// Gets whether this layer supports training through backpropagation.
    /// </summary>
    /// <value><c>true</c> if any sub-layer in either stream supports training.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This tells the training system whether this layer has any learnable
    /// parameters. If at least one layer in either stream can be trained, the whole parallel
    /// streams layer is considered trainable.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _streamA.Any(l => l.SupportsTraining) || _streamB.Any(l => l.SupportsTraining);

    /// <summary>
    /// Performs the forward pass: splits input, runs both streams, and concatenates outputs.
    /// </summary>
    /// <param name="input">The input tensor with shape <c>[batch, features]</c> or <c>[features]</c>.
    /// The feature dimension must equal the <c>inputSize</c> specified in the constructor.</param>
    /// <returns>A tensor with shape <c>[batch, streamAOutput + streamBOutput]</c> containing the
    /// concatenated outputs from both streams.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is where the actual parallel processing happens:
    /// 1. The input is split in half along the feature dimension
    /// 2. The first half goes through Stream A's layers
    /// 3. The second half goes through Stream B's layers (independently)
    /// 4. Both outputs are joined back together
    ///
    /// All operations use Engine methods that record on the gradient tape, so
    /// backpropagation will correctly compute gradients for both streams.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input is null)
            throw new ArgumentNullException(nameof(input));

        int rank = input.Shape.Length;
        if (rank == 0)
            throw new ArgumentException("Input tensor must have at least one dimension.", nameof(input));

        int featureSize = input.Shape[^1];
        // Enforce the constructor-time split contract: an odd or wrong-sized last dimension
        // would silently drop one or more features via integer division and slicing,
        // producing wrong outputs without any warning. Fail loudly instead.
        int expectedFeatureSize = _splitSize * 2;
        if (featureSize != expectedFeatureSize)
        {
            throw new ArgumentException(
                $"Expected input feature dimension {expectedFeatureSize} (configured at construction), " +
                $"but got {featureSize}.",
                nameof(input));
        }
        int halfSize = _splitSize;

        // Split input along last axis using Engine ops (tape-tracked)
        int[] startA = new int[rank];
        int[] startB = new int[rank];
        int[] sliceLen = new int[rank];
        for (int d = 0; d < rank; d++)
        {
            startA[d] = 0;
            startB[d] = 0;
            sliceLen[d] = input.Shape[d];
        }
        startB[rank - 1] = halfSize;
        sliceLen[rank - 1] = halfSize;

        var inputA = Engine.TensorSlice(input, startA, sliceLen);
        var inputB = Engine.TensorSlice(input, startB, sliceLen);

        // Run each stream
        var outputA = RunStream(_streamA, inputA);
        var outputB = RunStream(_streamB, inputB);

        // Concatenate along last axis (tape-tracked)
        return Engine.TensorConcatenate([outputA, outputB], axis: rank - 1);
    }

    /// <summary>
    /// Collects parameter gradients from all layers in both streams.
    /// </summary>
    /// <returns>A vector of gradients ordered the same as <see cref="GetParameters"/>:
    /// Stream A gradients first, then Stream B gradients.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> After the backward pass computes how much each parameter contributed
    /// to the loss, this method collects those gradient values. The optimizer uses these to
    /// decide how to adjust each parameter.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameterGradients()
    {
        var parts = new List<Vector<T>>();
        foreach (var layer in _streamA) parts.Add(layer.GetParameterGradients());
        foreach (var layer in _streamB) parts.Add(layer.GetParameterGradients());
        return Vector<T>.Concatenate(parts.ToArray());
    }

    /// <summary>
    /// Updates parameters in both streams using the given learning rate.
    /// </summary>
    /// <param name="learningRate">The step size for parameter updates. Smaller values mean
    /// more cautious learning.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This applies the computed gradients to update all weights and biases
    /// in both streams. Each layer handles its own update using the standard formula:
    /// <c>parameter = parameter - learningRate * gradient</c>
    /// </para>
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        foreach (var layer in _streamA) layer.UpdateParameters(learningRate);
        foreach (var layer in _streamB) layer.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Resets the internal state of all layers in both streams.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Some layers (like recurrent layers) maintain internal memory between
    /// forward passes. This method clears that memory for all layers in both streams, which is
    /// useful when starting to process a new batch of data or a new sequence.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        foreach (var layer in _streamA) layer.ResetState();
        foreach (var layer in _streamB) layer.ResetState();
    }

    /// <summary>
    /// Sets training or inference mode for all layers in both streams.
    /// </summary>
    /// <param name="isTraining"><c>true</c> for training mode (enables dropout, batch norm updates);
    /// <c>false</c> for inference mode (deterministic behavior).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Neural networks behave differently during training vs. making predictions:
    /// - <b>Training mode:</b> Dropout randomly disables neurons, batch normalization updates statistics
    /// - <b>Inference mode:</b> All neurons active, batch normalization uses learned statistics
    /// This method propagates the mode setting to every layer in both streams.
    /// </para>
    /// </remarks>
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        foreach (var layer in _streamA) layer.SetTrainingMode(isTraining);
        foreach (var layer in _streamB) layer.SetTrainingMode(isTraining);
    }

    /// <summary>
    /// Runs input through a sequence of layers, returning the final output.
    /// </summary>
    private static Tensor<T> RunStream(List<ILayer<T>> layers, Tensor<T> input)
    {
        var current = input;
        foreach (var layer in layers)
            current = layer.Forward(current);
        return current;
    }
}

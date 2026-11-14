using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Builds the three-channel GraFPrint peak-extractor input from a single-channel spectrogram.
/// </summary>
/// <remarks>
/// The reference implementation min-max normalizes each spectrogram and concatenates normalized
/// time and frequency coordinate planes before its learned peak-extraction convolution. Keeping
/// this operation as an explicit layer prevents the native model from silently degrading into a
/// conventional image CNN that never receives the graph node coordinates described by the paper.
/// </remarks>
[LayerCategory(LayerCategory.Input)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = false, ChangesShape = true, ExpectedInputRank = 3,
    TestInputShape = "1, 4, 4", TestConstructorArgs = "")]
[TensorLayout(TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    BatchOptional = true, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class GraFPrintCoordinateAugmentationLayer<T> : LayerBase<T>, IShapeContract
{
    /// <summary>Creates a parameter-free coordinate augmentation layer.</summary>
    public GraFPrintCoordinateAugmentationLayer()
        : base([-1, -1, -1], [3, -1, -1])
    {
    }

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank == 3)
        {
            return
            [
                new(TensorAxis.Channels, AxisRelation.Fixed(3)),
                new(TensorAxis.Height, AxisRelation.Same(TensorAxis.Height)),
                new(TensorAxis.Width, AxisRelation.Same(TensorAxis.Width))
            ];
        }
        if (inputRank == 4)
        {
            return
            [
                new(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                new(TensorAxis.Channels, AxisRelation.Fixed(3)),
                new(TensorAxis.Height, AxisRelation.Same(TensorAxis.Height)),
                new(TensorAxis.Width, AxisRelation.Same(TensorAxis.Width))
            ];
        }
        return null;
    }

    /// <inheritdoc />
    protected override void OnFirstForward(Tensor<T> input)
    {
        GetDimensions(input, out _, out int channels, out int height, out int width);
        if (channels != 1)
            throw new ArgumentException(
                $"GraFPrint coordinate augmentation requires one spectrogram channel, but received {channels}.",
                nameof(input));

        ResolveShapes([channels, height, width], [3, height, width]);
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        GetDimensions(input, out int batch, out int channels, out int height, out int width);
        if (channels != 1)
            throw new ArgumentException(
                $"GraFPrint coordinate augmentation requires one spectrogram channel, but received {channels}.",
                nameof(input));

        bool batched = input.Rank == 4;
        int[] outputShape = batched ? [batch, 3, height, width] : [3, height, width];
        if (batch == 0 || height == 0 || width == 0)
            return new Tensor<T>(outputShape);

        T epsilon = NumOps.FromDouble(1e-12);
        int pixels = height * width;
        Tensor<T> output;
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            var source = Engine.Reshape(input, [batch, pixels]);

            // Per-sample min and max as the scalar scan found them: starting from MaxValue / MinValue
            // and moving only to strictly smaller / larger values, so NaN and values beyond the
            // starting bound are never taken.
            var isNan = Engine.TensorIsNan(source);
            var maxFill = new Tensor<T>([batch, pixels]);
            maxFill.Fill(NumOps.MaxValue);
            var minFill = new Tensor<T>([batch, pixels]);
            minFill.Fill(NumOps.MinValue);
            var min = Engine.TensorMin(
                Engine.TensorNegate(Engine.ReduceMax(
                    Engine.TensorNegate(Engine.TensorWhere(isNan, maxFill, source)), new[] { 1 }, keepDims: true)),
                NumOps.MaxValue);
            var max = Engine.TensorMax(
                Engine.ReduceMax(Engine.TensorWhere(isNan, minFill, source), new[] { 1 }, keepDims: true),
                NumOps.MinValue);

            // The reference formula is undefined for a constant spectrogram. Preserve its
            // finite absolute level in that degenerate case so distinct valid tensors do
            // not collapse to the same graph, while leaving the published min-max path
            // unchanged for every non-constant spectrogram.
            var range = Engine.TensorSubtract(max, min);                                     // [batch, 1]
            var hasDynamicRange = Engine.TensorBroadcastTo(
                Engine.TensorGreaterThan(range, epsilon), [batch, pixels]);
            var ones = new Tensor<T>([batch, 1]);
            ones.Fill(NumOps.One);
            var safeRange = Engine.TensorWhere(Engine.TensorGreaterThan(range, epsilon), range, ones);
            var normalized = Engine.TensorWhere(
                hasDynamicRange, Engine.TensorDivide(Engine.TensorSubtract(source, min), safeRange), source);

            // time = w / (width - 1) along the columns, frequency = h / (height - 1) along the rows.
            // Each coordinate is one double division rounded once to T. The engine's scalar divide
            // multiplies by a reciprocal in T, which is an ulp off the quotient for about one float
            // coordinate in five (5/6 came out as 0.8333334 instead of 0.8333333).
            var time = width > 1
                ? new Tensor<double>(
                    Enumerable.Range(0, width).Select(w => w / (double)(width - 1)).ToArray(), [width]).Cast<T>()
                : new Tensor<T>([width]);
            var frequency = height > 1
                ? new Tensor<double>(
                    Enumerable.Range(0, height).Select(h => h / (double)(height - 1)).ToArray(), [height]).Cast<T>()
                : new Tensor<T>([height]);
            var timeGrid = Engine.TensorBroadcastTo(Engine.Reshape(time, [1, 1, 1, width]), [batch, 1, height, width]);
            var frequencyGrid = Engine.TensorBroadcastTo(Engine.Reshape(frequency, [1, 1, height, 1]), [batch, 1, height, width]);

            output = Engine.Reshape(
                Engine.TensorConcatenate(
                    new[] { timeGrid, frequencyGrid, Engine.Reshape(normalized, [batch, 1, height, width]) }, 1),
                outputShape).Clone();
        }

        // This is intentionally a detached data transformation. It is the first operation in
        // GraFPrint, so there are no trainable parameters upstream; the following convolution is
        // still fully tape-tracked and receives exact weight gradients from these node features.
        return output;
    }

    private static void GetDimensions(
        Tensor<T> input, out int batch, out int channels, out int height, out int width)
    {
        if (input.Rank == 3)
        {
            batch = 1;
            channels = input.Shape[0];
            height = input.Shape[1];
            width = input.Shape[2];
            return;
        }

        if (input.Rank == 4)
        {
            batch = input.Shape[0];
            channels = input.Shape[1];
            height = input.Shape[2];
            width = input.Shape[3];
            return;
        }

        throw new ArgumentException(
            $"GraFPrint coordinate augmentation requires rank-3 [C,H,W] or rank-4 [B,C,H,W] input; received rank {input.Rank}.",
            nameof(input));
    }

    /// <inheritdoc />
    public override void ResetState()
    {
    }
}

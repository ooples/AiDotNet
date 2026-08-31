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
        var output = new Tensor<T>(batched
            ? [batch, 3, height, width]
            : [3, height, width]);

        T epsilon = NumOps.FromDouble(1e-12);
        for (int b = 0; b < batch; b++)
        {
            T min = NumOps.MaxValue;
            T max = NumOps.MinValue;
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    T value = batched ? input[b, 0, h, w] : input[0, h, w];
                    if (NumOps.LessThan(value, min)) min = value;
                    if (NumOps.GreaterThan(value, max)) max = value;
                }
            }

            T range = NumOps.Subtract(max, min);
            bool hasDynamicRange = NumOps.GreaterThan(range, epsilon);

            for (int h = 0; h < height; h++)
            {
                T frequency = NumOps.FromDouble(height > 1 ? h / (double)(height - 1) : 0.0);
                for (int w = 0; w < width; w++)
                {
                    T time = NumOps.FromDouble(width > 1 ? w / (double)(width - 1) : 0.0);
                    T source = batched ? input[b, 0, h, w] : input[0, h, w];
                    // The reference formula is undefined for a constant spectrogram. Preserve its
                    // finite absolute level in that degenerate case so distinct valid tensors do
                    // not collapse to the same graph, while leaving the published min-max path
                    // unchanged for every non-constant spectrogram.
                    T normalized = hasDynamicRange
                        ? NumOps.Divide(NumOps.Subtract(source, min), range)
                        : source;

                    if (batched)
                    {
                        output[b, 0, h, w] = time;
                        output[b, 1, h, w] = frequency;
                        output[b, 2, h, w] = normalized;
                    }
                    else
                    {
                        output[0, h, w] = time;
                        output[1, h, w] = frequency;
                        output[2, h, w] = normalized;
                    }
                }
            }
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

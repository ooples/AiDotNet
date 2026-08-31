using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Applies GraFPrint's frequency-only peak-extractor stride while preserving every time frame.
/// </summary>
/// <remarks>
/// AiDotNet's general 2-D convolution currently has one stride value for both spatial axes. The
/// reference peak extractor uses stride <c>(2, 1)</c>. Running the learned convolution at stride
/// one and gathering every second frequency row is mathematically the same operation and keeps the
/// gather on the gradient tape, unlike copying selected values into a detached tensor.
/// </remarks>
[LayerCategory(LayerCategory.Structural)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = false, ChangesShape = true, ExpectedInputRank = 3,
    TestInputShape = "2, 4, 4", TestConstructorArgs = "2")]
[TensorLayout(TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    BatchOptional = true, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class GraFPrintFrequencyStrideLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _frequencyStride;

    /// <summary>Creates a frequency-only striding layer.</summary>
    public GraFPrintFrequencyStrideLayer([LayerState] int frequencyStride = 2)
        : base([-1, -1, -1], [-1, -1, -1])
    {
        if (frequencyStride <= 0)
            throw new ArgumentOutOfRangeException(nameof(frequencyStride));
        _frequencyStride = frequencyStride;
    }

    /// <summary>Gets the frequency-axis stride.</summary>
    public int FrequencyStride => _frequencyStride;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        var spatial = new OutputAxisContract(
            TensorAxis.Height,
            AxisRelation.Window(TensorAxis.Height, kernel: 1, stride: _frequencyStride, padding: 0));
        if (inputRank == 3)
        {
            return
            [
                new(TensorAxis.Channels, AxisRelation.Same(TensorAxis.Channels)),
                spatial,
                new(TensorAxis.Width, AxisRelation.Same(TensorAxis.Width))
            ];
        }
        if (inputRank == 4)
        {
            return
            [
                new(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                new(TensorAxis.Channels, AxisRelation.Same(TensorAxis.Channels)),
                spatial,
                new(TensorAxis.Width, AxisRelation.Same(TensorAxis.Width))
            ];
        }
        return null;
    }

    /// <inheritdoc />
    protected override void OnFirstForward(Tensor<T> input)
    {
        GetDimensions(input, out _, out int channels, out int height, out int width);
        ResolveShapes(
            [channels, height, width],
            [channels, DivideRoundUp(height, _frequencyStride), width]);
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        GetDimensions(input, out int batch, out int channels, out int height, out int width);
        bool batched = input.Rank == 4;
        int outHeight = DivideRoundUp(height, _frequencyStride);
        var indices = new int[checked(batch * channels * outHeight * width)];

        int destination = 0;
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int oh = 0; oh < outHeight; oh++)
                {
                    int sourceH = oh * _frequencyStride;
                    for (int w = 0; w < width; w++)
                    {
                        indices[destination++] = batched
                            ? ((b * channels + c) * height + sourceH) * width + w
                            : (c * height + sourceH) * width + w;
                    }
                }
            }
        }

        var flat = Engine.Reshape(input, [input.Length]);
        var gathered = Engine.TensorGather(
            flat, new Tensor<int>(indices, [indices.Length]), axis: 0);
        return Engine.Reshape(gathered, batched
            ? [batch, channels, outHeight, width]
            : [channels, outHeight, width]);
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["FrequencyStride"] = _frequencyStride.ToString(
            System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    private static int DivideRoundUp(int value, int divisor) => (value + divisor - 1) / divisor;

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
            $"GraFPrint frequency stride requires rank-3 [C,H,W] or rank-4 [B,C,H,W] input; received rank {input.Rank}.",
            nameof(input));
    }

    /// <inheritdoc />
    public override void ResetState()
    {
    }
}

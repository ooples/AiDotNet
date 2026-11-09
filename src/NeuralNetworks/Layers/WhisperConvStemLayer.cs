using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Whisper's audio-encoder stem (Radford et al. 2022, §2.2): two width-3 convolutions over time, each followed by
/// GELU, the second with stride 2. It maps [frames, mels] log-mel features to [frames / 2, width] encoder inputs.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Matches the reference <c>AudioEncoder</c>: <c>Conv1d(n_mels, width, 3, padding=1)</c> then
/// <c>Conv1d(width, width, 3, stride=2, padding=1)</c>. A 30-second clip's 3,000 frames become the encoder's 1,500
/// positions.
/// </para>
/// <para>
/// <b>For Beginners:</b> Before attention looks at the audio, two small filters slide along time to pick out local
/// patterns and halve the number of steps, which makes the attention four times cheaper.
/// </para>
/// </remarks>
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class WhisperConvStemLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _numMels;
    private readonly int _width;

    private readonly Conv1DLayer<T> _conv1;
    private readonly Conv1DLayer<T> _conv2;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Creates the stem.
    /// </summary>
    /// <param name="numMels">Mel bins per input frame (80 for Whisper, 128 for large-v3).</param>
    /// <param name="width">The encoder width.</param>
    public WhisperConvStemLayer([LayerState] int numMels, [LayerState] int width)
        : base(new[] { -1, numMels }, new[] { -1, width })
    {
        if (numMels <= 0) throw new ArgumentOutOfRangeException(nameof(numMels), "numMels must be positive.");
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width), "width must be positive.");

        _numMels = numMels;
        _width = width;
        var gelu = (Interfaces.IActivationFunction<T>)new GELUActivation<T>();
        _conv1 = new Conv1DLayer<T>(inputChannels: numMels, outputChannels: width, kernelSize: 3, dilation: 1,
            stride: 1, padding: 1, activation: gelu);
        _conv2 = new Conv1DLayer<T>(inputChannels: width, outputChannels: width, kernelSize: 3, dilation: 1,
            stride: 2, padding: 1, activation: gelu);

        RegisterSubLayer(_conv1);
        RegisterSubLayer(_conv2);
    }

    /// <summary>The number of encoder positions a clip of <paramref name="frames"/> frames gives.</summary>
    public static int OutputFrames(int frames) => (frames - 1) / 2 + 1;

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank < 2)
            throw new ArgumentException(
                $"Whisper's stem takes [..., frames, mels]; got rank {input.Rank}.", nameof(input));

        int rank = input.Rank;
        int batch = 1;
        for (int i = 0; i < rank - 2; i++) batch *= input.Shape[i];
        int frames = input.Shape[rank - 2];
        if (input.Shape[rank - 1] != _numMels)
            throw new ArgumentException(
                $"Whisper's stem was built for {_numMels} mel bins; got {input.Shape[rank - 1]}.", nameof(input));

        // [B, T, M] -> [B, M, T] -> conv -> conv -> [B, W, T'] -> [B, T', W].
        var channelsFirst = Engine.TensorPermute(Engine.Reshape(input, new[] { batch, frames, _numMels }), new[] { 0, 2, 1 });
        var convolved = _conv2.Forward(_conv1.Forward(channelsFirst));
        var timeMajor = Engine.TensorPermute(convolved, new[] { 0, 2, 1 });
        if (rank == 3) return timeMajor;

        var outputShape = input.Shape.ToArray();
        outputShape[rank - 2] = convolved.Shape[2];
        outputShape[rank - 1] = _width;
        return Engine.Reshape(timeMajor, outputShape);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        _conv1.UpdateParameters(learningRate);
        _conv2.UpdateParameters(learningRate);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _conv1.ResetState();
        _conv2.ResetState();
    }
}

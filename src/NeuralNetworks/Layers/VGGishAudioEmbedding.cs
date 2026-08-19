using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// VGGish audio embedding: the AudioSet convolutional network that turns a log-mel patch into a
/// compact per-segment embedding.
/// </summary>
/// <typeparam name="T">The numeric type of the network.</typeparam>
/// <remarks>
/// <para>
/// <b>What this is.</b> Hershey et al., <i>CNN Architectures for Large-Scale Audio Classification</i>
/// (ICASSP 2017), released as <c>vggish</c> in the TensorFlow AudioSet models. It is VGG with the
/// last convolutional group dropped and the classifier replaced by a 128-wide embedding: four
/// convolutional groups (64, 128, 256x2, 512x2) of 3x3 SAME-padded filters, each followed by a 2x2
/// stride-2 max pool, then flatten and three fully-connected layers. The final layer is
/// <b>linear</b> — the embedding is taken pre-activation, exactly as the reference implementation
/// does.
/// </para>
/// <para>
/// <b>What it expects.</b> A stabilised log-mel patch shaped <c>[frames, mels]</c> — the paper uses
/// 96 frames x 64 mel bins, which is 0.96 s at a 25 ms window and 10 ms hop. Produce it with
/// <see cref="AiDotNet.Diffusion.Audio.MelSpectrogram{T}"/> configured through
/// <see cref="VGGishMelSpectrogramDefaults"/>; do not hand-roll it. A leading singleton channel is
/// added here, so the convolutional stack sees <c>[1, frames, mels]</c>.
/// </para>
/// <para>
/// <b>Why a composite rather than loose layers.</b> The children are registered with
/// <see cref="LayerBase{T}.RegisterSubLayer"/>, so the base class declares each one as a parameter
/// slot under a stable identifier. Parameter count, flat get/set, chunked restore and checkpoint
/// identity all follow from that single registration — there is no hand-written positional walk to
/// drift out of step, which is the failure mode that produces "Expected N parameters, got M" on a
/// clone. This is the same composition model as <c>torch.nn.Module</c>, with the addition that the
/// slot identifiers are stable rather than positional, so reordering or renaming does not
/// invalidate a checkpoint the way a PyTorch <c>state_dict</c> key would.
/// </para>
/// <para><b>For Beginners:</b> Sound arrives as a long list of samples, which is not a useful shape
/// for a network. The usual pipeline turns it into a picture — time along one axis, pitch along the
/// other — and then reads that picture with an image network. This layer is the image network. Give
/// it one such picture and it returns a short list of numbers summarising what was heard.</para>
/// </remarks>
// Declared, not inferred: the convolutional stack pools its two spatial axes away entirely and the
// dense head fixes the width, so the output size comes from EmbeddingSize rather than from any input
// axis. A probe cannot derive that, which is why OutputAxesFor below is hand-written, as
// MaxPoolingLayer's is.
[TensorLayout(TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input,
    Note = "A [frames, mels] log-mel patch. The singleton channel is added internally.")]
[TensorLayout(TensorAxis.Channels, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
public partial class VGGishAudioEmbedding<T> : LayerBase<T>, IShapeContract
{
    /// <summary>Filter count of the first convolutional group in the published network.</summary>
    public const int PaperConv1Filters = 64;

    /// <summary>Filter count of the second convolutional group in the published network.</summary>
    public const int PaperConv2Filters = 128;

    /// <summary>Filter count of the third convolutional group (two layers) in the published network.</summary>
    public const int PaperConv3Filters = 256;

    /// <summary>Filter count of the fourth convolutional group (two layers) in the published network.</summary>
    public const int PaperConv4Filters = 512;

    /// <summary>Width of the two hidden fully-connected layers in the published network.</summary>
    public const int PaperFullyConnectedWidth = 4096;

    /// <summary>Size of the published embedding.</summary>
    public const int PaperEmbeddingSize = 128;

    /// <summary>Frames per patch in the published front-end (0.96 s at a 10 ms hop).</summary>
    public const int PaperPatchFrames = 96;

    private readonly ConvolutionalLayer<T> _conv1;
    private readonly MaxPoolingLayer<T> _pool1;
    private readonly ConvolutionalLayer<T> _conv2;
    private readonly MaxPoolingLayer<T> _pool2;
    private readonly ConvolutionalLayer<T> _conv3a;
    private readonly ConvolutionalLayer<T> _conv3b;
    private readonly MaxPoolingLayer<T> _pool3;
    private readonly ConvolutionalLayer<T> _conv4a;
    private readonly ConvolutionalLayer<T> _conv4b;
    private readonly MaxPoolingLayer<T> _pool4;
    private readonly FlattenLayer<T> _flatten;
    private readonly DenseLayer<T> _fc1;
    private readonly DenseLayer<T> _fc2;
    private readonly DenseLayer<T> _embedding;

    /// <summary>Size of the embedding this layer produces.</summary>
    public int EmbeddingSize { get; }

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Builds the embedding network. Every width defaults to the published value and can be
    /// overridden.
    /// </summary>
    /// <param name="conv1Filters">Filters in group 1. Defaults to <see cref="PaperConv1Filters"/>.</param>
    /// <param name="conv2Filters">Filters in group 2. Defaults to <see cref="PaperConv2Filters"/>.</param>
    /// <param name="conv3Filters">Filters in each of group 3's two layers. Defaults to <see cref="PaperConv3Filters"/>.</param>
    /// <param name="conv4Filters">Filters in each of group 4's two layers. Defaults to <see cref="PaperConv4Filters"/>.</param>
    /// <param name="fullyConnectedWidth">Width of the two hidden dense layers. Defaults to <see cref="PaperFullyConnectedWidth"/>.</param>
    /// <param name="embeddingSize">Size of the produced embedding. Defaults to <see cref="PaperEmbeddingSize"/>.</param>
    /// <remarks>
    /// <para>
    /// The defaults reproduce the published network, which is roughly 67 M parameters — dominated by
    /// the two 4096-wide dense layers. That is the right default for fidelity and the wrong size for
    /// a unit-test fixture, which is precisely why the widths are constructor parameters rather than
    /// constants: a fixture can build a small variant without a second, divergent implementation
    /// existing anywhere.
    /// </para>
    /// </remarks>
    public VGGishAudioEmbedding(
        int conv1Filters = PaperConv1Filters,
        int conv2Filters = PaperConv2Filters,
        int conv3Filters = PaperConv3Filters,
        int conv4Filters = PaperConv4Filters,
        int fullyConnectedWidth = PaperFullyConnectedWidth,
        int embeddingSize = PaperEmbeddingSize)
        : base(new[] { -1, -1 }, new[] { embeddingSize })
    {
        Positive(conv1Filters, nameof(conv1Filters));
        Positive(conv2Filters, nameof(conv2Filters));
        Positive(conv3Filters, nameof(conv3Filters));
        Positive(conv4Filters, nameof(conv4Filters));
        Positive(fullyConnectedWidth, nameof(fullyConnectedWidth));
        Positive(embeddingSize, nameof(embeddingSize));

        EmbeddingSize = embeddingSize;

        // 3x3 kernels with padding 1 reproduce TensorFlow's SAME padding at stride 1, so each group
        // preserves its spatial extent and only the pools reduce it. ReLU on every convolution.
        // ReLUActivation implements both the scalar and vector activation interfaces, so the
        // layer constructors are ambiguous without an explicit interface. Scalar is correct here:
        // ReLU is elementwise, with no cross-channel coupling for a vector form to exploit.
        IActivationFunction<T> relu = new ReLUActivation<T>();

        _conv1 = new ConvolutionalLayer<T>(conv1Filters, kernelSize: 3, stride: 1, padding: 1, activationFunction: relu);
        _pool1 = new MaxPoolingLayer<T>(poolSize: 2, stride: 2);
        _conv2 = new ConvolutionalLayer<T>(conv2Filters, kernelSize: 3, stride: 1, padding: 1, activationFunction: relu);
        _pool2 = new MaxPoolingLayer<T>(poolSize: 2, stride: 2);
        _conv3a = new ConvolutionalLayer<T>(conv3Filters, kernelSize: 3, stride: 1, padding: 1, activationFunction: relu);
        _conv3b = new ConvolutionalLayer<T>(conv3Filters, kernelSize: 3, stride: 1, padding: 1, activationFunction: relu);
        _pool3 = new MaxPoolingLayer<T>(poolSize: 2, stride: 2);
        _conv4a = new ConvolutionalLayer<T>(conv4Filters, kernelSize: 3, stride: 1, padding: 1, activationFunction: relu);
        _conv4b = new ConvolutionalLayer<T>(conv4Filters, kernelSize: 3, stride: 1, padding: 1, activationFunction: relu);
        _pool4 = new MaxPoolingLayer<T>(poolSize: 2, stride: 2);

        _flatten = new FlattenLayer<T>();
        _fc1 = new DenseLayer<T>(fullyConnectedWidth, activationFunction: relu);
        _fc2 = new DenseLayer<T>(fullyConnectedWidth, activationFunction: relu);

        // Linear. The reference implementation names this tensor the embedding and reads it BEFORE
        // any activation, so applying ReLU here would clamp away every negative coordinate and
        // change what downstream models consume.
        _embedding = new DenseLayer<T>(embeddingSize, activationFunction: new IdentityActivation<T>());

        RegisterSubLayer(_conv1);
        RegisterSubLayer(_pool1);
        RegisterSubLayer(_conv2);
        RegisterSubLayer(_pool2);
        RegisterSubLayer(_conv3a);
        RegisterSubLayer(_conv3b);
        RegisterSubLayer(_pool3);
        RegisterSubLayer(_conv4a);
        RegisterSubLayer(_conv4b);
        RegisterSubLayer(_pool4);
        RegisterSubLayer(_flatten);
        RegisterSubLayer(_fc1);
        RegisterSubLayer(_fc2);
        RegisterSubLayer(_embedding);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The whole patch collapses to one embedding: both spatial axes are pooled and flattened away,
    /// and the final dense layer fixes the width. So the single output axis is
    /// <see cref="AxisRelation.Fixed"/> at <see cref="EmbeddingSize"/> — it does not derive from the
    /// input's frame count or mel count at all.
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (EmbeddingSize <= 0 || inputRank < 2 || inputRank > 3) return null;

        return new[]
        {
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(EmbeddingSize)),
        };
    }

    private static void Positive(int value, string name)
    {
        if (value <= 0) throw new ArgumentOutOfRangeException(name, value, "Width must be positive.");
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Accepts a single <c>[frames, mels]</c> patch or an already-channelled
    /// <c>[channels, frames, mels]</c> tensor. A rank-2 input is given the singleton channel the
    /// convolutional stack expects. Rank 1 is rejected rather than reshaped: a rank-1 "spectrogram"
    /// has no time axis, and silently promoting it would hide exactly the defect this layer exists
    /// to remove.
    /// </remarks>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Shape.Length < 2)
        {
            throw new ArgumentException(
                $"{nameof(VGGishAudioEmbedding<T>)} requires a [frames, mels] log-mel patch; got rank " +
                $"{input.Shape.Length}. A rank-1 input has no time axis, so there is no sequence to " +
                "embed — compute the patch with MelSpectrogram rather than reducing the waveform to a " +
                "single vector.",
                nameof(input));
        }

        var current = input.Shape.Length == 2
            ? input.Reshape(new[] { 1, input.Shape[0], input.Shape[1] })
            : input;

        current = _conv1.Forward(current);
        current = _pool1.Forward(current);
        current = _conv2.Forward(current);
        current = _pool2.Forward(current);
        current = _conv3a.Forward(current);
        current = _conv3b.Forward(current);
        current = _pool3.Forward(current);
        current = _conv4a.Forward(current);
        current = _conv4b.Forward(current);
        current = _pool4.Forward(current);
        current = _flatten.Forward(current);
        current = _fc1.Forward(current);
        current = _fc2.Forward(current);
        return _embedding.Forward(current);
    }

    /// <inheritdoc/>
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        foreach (var child in GetSubLayers()) child?.SetTrainingMode(isTraining);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        foreach (var child in GetSubLayers()) child?.UpdateParameters(learningRate);
    }

    /// <inheritdoc/>
    public override void ClearGradients()
    {
        base.ClearGradients();
        foreach (var child in GetSubLayers()) child?.ClearGradients();
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        foreach (var child in GetSubLayers()) child?.ResetState();
    }
}

/// <summary>
/// The log-mel front-end settings the VGGish embedding was trained against.
/// </summary>
/// <remarks>
/// Pass these to <see cref="AiDotNet.Diffusion.Audio.MelSpectrogram{T}"/> so the patch fed to
/// <see cref="VGGishAudioEmbedding{T}"/> matches the published pipeline. Values are from the
/// reference <c>vggish_params.py</c>: 16 kHz audio, a 25 ms window and 10 ms hop (400 and 160
/// samples at that rate), 64 mel bins spanning 125–7500 Hz, and <c>log(mel + 0.01)</c> compression.
/// The offset form matters — the layer's alternative dB compression is a different scale, and
/// feeding it changes the input distribution the convolutional stack expects.
/// </remarks>
public static class VGGishMelSpectrogramDefaults
{
    /// <summary>Sample rate the published front-end assumes, in Hz.</summary>
    public const int SampleRate = 16000;

    /// <summary>Number of mel bins.</summary>
    public const int MelBins = 64;

    /// <summary>STFT window length in samples — 25 ms at <see cref="SampleRate"/>.</summary>
    public const int WindowLengthSamples = 400;

    /// <summary>STFT hop length in samples — 10 ms at <see cref="SampleRate"/>.</summary>
    public const int HopLengthSamples = 160;

    /// <summary>Lowest mel band edge, in Hz.</summary>
    public const double MinFrequencyHz = 125.0;

    /// <summary>Highest mel band edge, in Hz.</summary>
    public const double MaxFrequencyHz = 7500.0;

    /// <summary>Additive floor for the stabilised log compression, <c>log(mel + offset)</c>.</summary>
    public const double LogOffset = 0.01;
}

using System.Collections.Generic;
// AiDotNet.Attributes is REQUIRED for [TensorLayout] to bind to the right type: two other Tensors
// namespaces declare a TensorLayout, and without this using the attribute silently resolves to one
// of those and the contract is never seen.
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech;

/// <summary>
/// Base class for neural vocoder models that convert mel-spectrograms to audio waveforms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Vocoders form the second stage of a two-stage TTS pipeline:
/// Text -> [Acoustic Model] -> Mel-Spectrogram -> [Vocoder] -> Waveform.
/// </para>
/// <para>
/// Subclasses include HiFi-GAN, WaveNet, WaveRNN, BigVGAN, DiffWave, Vocos,
/// and other models that synthesize raw audio from mel-spectrogram representations.
/// </para>
/// </remarks>
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Frames,
    Direction = TensorLayoutDirection.Input,
    Note = "A mel-spectrogram: MelChannels bands by however many frames the acoustic model produced.")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Length,
    Direction = TensorLayoutDirection.Output,
    Note = "A mono waveform whose length is the frame count times UpsampleFactor - a SCALED relation, "
         + "not a constant, because the frame count is free.")]
public abstract class VocoderBase<T> : TtsModelBase<T>, IVocoder<T>, IShapeContract
{
    /// <summary>
    /// The vocoder family's output law: <c>[Batch, Frames * UpsampleFactor]</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the case the symbolic vocabulary exists for. A vocoder's output length is not a
    /// constant and not a copy of an input axis - it is the frame count MULTIPLIED by the hop size,
    /// and the hop size is a constructor-time value. <c>Scaled(Frames, UpsampleFactor)</c> states
    /// exactly that, once, for every vocoder in the family; a shape system that could only record
    /// observed integers would have to re-record HiFi-GAN, WaveNet, BigVGAN, Vocos and the rest
    /// separately, and would still be wrong for any frame count it had not seen.
    /// </para>
    /// <para>
    /// <see cref="UpsampleFactor"/> defaults to <see cref="TtsModelBase{T}.HopSize"/> and is virtual,
    /// so a vocoder that upsamples by something other than the hop size states it there and inherits
    /// a correct contract without writing one.
    /// </para>
    /// <para>
    /// This base had ZERO SUBCLASSES until the vocoders were re-parented onto it: all 17 declared
    /// <c>: TtsModelBase&lt;T&gt;, IVocoder&lt;T&gt;</c> and skipped it, so nothing inherited what it
    /// defined - the same defect the segmentation family had. Re-parenting them took the reachable
    /// count from 0 to 17 and made the law measurable for the first time.
    /// </para>
    /// <para>
    /// IT IS NOT THE DEFAULT, because measuring all 17 gave <b>3 agreed and 14 DISAGREED</b>, and the
    /// reason is not a wrong ratio - it is that <c>Predict</c> does not mean the same thing across
    /// this family. Only the GAN vocoders synthesise a whole waveform in one call:
    /// </para>
    /// <list type="bullet">
    /// <item><description><b>Agreed</b> - HiFiGAN, MelGAN, UnivNet: <c>[1,80,8] -&gt; [1,1,2048]</c>,
    /// exactly 8 frames x 256.</description></item>
    /// <item><description><c>[1,80,1]</c> - BigVGAN, DiffWave, FreGrad, PriorGrad, WaveGrad. These are
    /// DIFFUSION vocoders and Predict is one denoising step, not a synthesis.</description></item>
    /// <item><description><c>[1,513,2048]</c> - APNet, ISTFTNet. 513 is the FFT bin count: they emit a
    /// SPECTRUM and run the iSTFT separately, so their output axis is frequency.</description></item>
    /// <item><description><c>[1,1]</c> / <c>[1,1,8]</c> - WaveNet, WaveRNN, WaveGlow, ParallelWaveGAN:
    /// autoregressive or flow models producing one step per call.</description></item>
    /// <item><description><c>[1,2048]</c> - Vocos gets the LENGTH right and omits the channel axis.
    /// </description></item>
    /// <item><description><c>[1,1,2048]</c> against a contract of <c>[1,1,2400]</c> - MultiBandMelGAN
    /// upsamples by 256 while its HopSize says 300, so those two disagree in the model itself.
    /// </description></item>
    /// </list>
    /// <para>
    /// So the interface is shared and the OPERATION is not, which is a finding about IVocoder rather
    /// than about this contract. The three that were verified opt in by overriding this with
    /// <see cref="WaveformUpsampleContract"/>; the rest decline until each one's Predict is given a
    /// meaning worth stating.
    /// </para>
    /// </remarks>
    // OVERRIDE, not a new declaration: TtsModelBase now states the family's mel law, and a vocoder
    // emits a waveform rather than a mel frame - so it must REPLACE that answer, not sit beside it.
    // Hiding it with `new` would have left callers holding a TtsModelBase reference reading the mel
    // law for a vocoder, which is the silent wrong answer this whole system exists to prevent.
    public override IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => null;

    /// <summary>The family law, exposed so a vocoder with an extra axis can still reuse it.</summary>
    /// <remarks>
    /// The mono CHANNEL axis is measured, not decorative. The first version of this law returned
    /// <c>[Batch, Length]</c> and the forward pass returned <c>[1,1,2048]</c> for a <c>[1,80,8]</c>
    /// mel: the ratio was exactly right - 8 frames x an UpsampleFactor of 256 - and only the rank was
    /// wrong, because the generator ends at one output channel rather than squeezing it away.
    /// </remarks>
    protected IReadOnlyList<OutputAxisContract>? WaveformUpsampleContract(int inputRank)
    {
        int factor = UpsampleFactor;
        if (inputRank != 3 || factor <= 0) return null;
        return
        [
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Channels, AxisRelation.Fixed(1)),
            new OutputAxisContract(TensorAxis.Length, AxisRelation.Scaled(TensorAxis.Frames, factor)),
        ];
    }

    /// <inheritdoc />
    public abstract Tensor<T> MelToWaveform(Tensor<T> melSpectrogram);

    /// <summary>Gets the sample rate.</summary>
    int IVocoder<T>.SampleRate => SampleRate;

    /// <summary>Gets the mel channels.</summary>
    int IVocoder<T>.MelChannels => MelChannels;

    /// <summary>Gets the upsampling factor (hop size).</summary>
    public virtual int UpsampleFactor => HopSize;

    /// <summary>
    /// Initializes a new instance of the VocoderBase class.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="lossFunction">Optional loss function override.</param>
    /// <param name="maxGradNorm">
    /// Gradient-clipping norm, forwarded to <see cref="TtsModelBase{T}"/>. Present because this base
    /// previously omitted it while its own parent accepted it, so a vocoder that clipped gradients
    /// (DiffWave) could not be re-parented onto this class without losing the setting.
    /// </param>
    protected VocoderBase(
        NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0
    )
        : base(architecture, lossFunction, maxGradNorm) { }
}

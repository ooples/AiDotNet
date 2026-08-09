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
[TensorLayout(TensorAxis.Batch, TensorAxis.Length,
    Direction = TensorLayoutDirection.Output,
    Note = "A waveform whose length is the frame count times UpsampleFactor - a SCALED relation, not "
         + "a constant, because the frame count is free.")]
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
    /// IT IS NOT THE DEFAULT, because this law has never been exercised against a real Predict, and
    /// the reason is itself a defect worth recording: <b>this class has ZERO subclasses.</b> All 17
    /// vocoders - HiFiGAN, WaveNet, WaveRNN, BigVGAN, DiffWave, Vocos, MelGAN, ParallelWaveGAN,
    /// UnivNet, ISTFTNet, WaveGlow, WaveGrad, PriorGrad, FreGrad, APNet, APNet2, MultiBandMelGAN -
    /// declare <c>: TtsModelBase&lt;T&gt;, IVocoder&lt;T&gt;</c> and skip this base entirely, so
    /// nothing inherits what it defines. That is the same defect the segmentation family had, where
    /// eight family bases had no users until the models were re-parented onto them.
    /// </para>
    /// <para>
    /// Until those 17 are re-parented, a law declared here would be a claim about nothing. Declining
    /// is the honest default; <see cref="WaveformUpsampleContract"/> is ready for the moment a
    /// subclass exists to test it against.
    /// </para>
    /// </remarks>
    public virtual IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => null;

    /// <summary>The family law, exposed so a vocoder with an extra axis can still reuse it.</summary>
    protected IReadOnlyList<OutputAxisContract>? WaveformUpsampleContract(int inputRank)
    {
        int factor = UpsampleFactor;
        if (inputRank != 3 || factor <= 0) return null;
        return
        [
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
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
    protected VocoderBase(
        NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T>? lossFunction = null
    )
        : base(architecture, lossFunction) { }
}

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
public abstract class VocoderBase<T> : TtsModelBase<T>, IVocoder<T>
{
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
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction)
    {
    }
}

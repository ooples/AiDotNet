using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech;

/// <summary>
/// Base class for acoustic TTS models that generate mel-spectrograms from text.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Acoustic models form the first stage of a two-stage TTS pipeline:
/// Text -> [Acoustic Model] -> Mel-Spectrogram -> [Vocoder] -> Waveform.
/// </para>
/// <para>
/// Subclasses include Tacotron, Tacotron2, FastSpeech, FastSpeech2, GlowTTS, GradTTS,
/// and other models that produce mel-spectrograms requiring a separate vocoder for audio output.
/// </para>
/// </remarks>
public abstract class AcousticModelBase<T> : TtsModelBase<T>, IAcousticModel<T>
{
    /// <inheritdoc />
    public abstract Tensor<T> TextToMel(string text);

    /// <inheritdoc />
    public abstract Tensor<T> Synthesize(string text);

    /// <summary>Gets the number of mel frequency channels.</summary>
    int IAcousticModel<T>.MelChannels => MelChannels;

    /// <summary>Gets the mel hop size.</summary>
    int IAcousticModel<T>.HopSize => HopSize;

    /// <summary>Gets the FFT window size.</summary>
    public virtual int FftSize => 1024;

    /// <summary>Gets the sample rate.</summary>
    int ITtsModel<T>.SampleRate => SampleRate;

    /// <summary>Gets the maximum text length.</summary>
    public virtual int MaxTextLength => 512;

    /// <summary>
    /// Initializes a new instance of the AcousticModelBase class.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="lossFunction">Optional loss function override.</param>
    protected AcousticModelBase(
        NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction)
    {
    }
}

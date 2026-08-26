using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Effects;

/// <summary>
/// Configuration options for the FiNS (Filtered Noise Shaping) room impulse response estimator.
/// </summary>
/// <remarks>
/// <para>
/// FiNS estimates a time-domain room impulse response directly from reverberant speech. A strided
/// 1-D convolutional encoder compresses the waveform to a single latent embedding, and a decoder
/// then builds the response as two parts: a short early component predicted sample-by-sample, and a
/// late field synthesised by shaping filtered noise with predicted time-domain masks.
/// </para>
/// <para>
/// <b>For Beginners:</b> When you clap in a room, you hear a sharp initial sound followed by a
/// smeared tail of echoes. Those two parts behave very differently: the beginning has clear
/// structure worth predicting exactly, while the tail is essentially noise that fades out at a
/// different speed in each frequency band. FiNS mirrors that split — it predicts the sharp start
/// directly, and it makes the tail by taking noise, splitting it into frequency bands, and learning
/// how loud each band should be over time. Modelling the tail as shaped noise rather than
/// predicting 48,000 individual samples is what makes the problem tractable.
/// </para>
/// <para>
/// <b>Reference:</b> Steinmetz, Ithapu and Calamia, "Filtered Noise Shaping for Time Domain Room
/// Impulse Response Estimation from Reverberant Speech", WASPAA 2021 (arXiv:2107.07503). Defaults
/// below reproduce the paper's configuration; each is annotated with its source.
/// </para>
/// </remarks>
public class RoomImpulseResponseOptions : ModelOptions
{
    #region Audio

    /// <summary>
    /// Gets or sets the audio sample rate in Hz. Default 48000, the rate FiNS operates at.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> How many audio samples make up one second. The paper works at 48 kHz,
    /// so a one-second impulse response is 48,000 numbers long.
    /// </remarks>
    public int SampleRate { get; set; } = 48000;

    #endregion

    #region Encoder (paper §2.1)

    /// <summary>
    /// Gets or sets the number of strided convolutional encoder blocks. Default 14 (paper:
    /// "fourteen blocks", giving a receptive field over 100,000 timesteps = 2.4 s at 48 kHz).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Each block halves the length of the signal. Stacking fourteen of them
    /// lets the last block "see" more than two seconds of audio at once, which is what the model
    /// needs in order to judge how long the room takes to go quiet.
    /// </remarks>
    public int NumEncoderBlocks { get; set; } = 14;

    /// <summary>Gets or sets the encoder convolution kernel size. Default 15 (paper).</summary>
    /// <remarks>
    /// <b>For Beginners:</b> How many neighbouring samples each filter looks at in one step.
    /// </remarks>
    public int EncoderKernelSize { get; set; } = 15;

    /// <summary>
    /// Gets or sets the encoder convolution stride. Default 2 (paper), so each block halves the
    /// time resolution.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A stride of 2 means the filter hops two samples at a time, so the
    /// signal coming out of each block is half as long as the one going in.
    /// </remarks>
    public int EncoderStride { get; set; } = 2;

    /// <summary>
    /// Gets or sets the channel count reached at the final encoder block. Default 512 (paper:
    /// "512 channels at the final layer"). Earlier blocks ramp up to this geometrically.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> As the signal gets shorter, it gets "wider" — more parallel feature
    /// detectors — so information is preserved rather than thrown away.
    /// </remarks>
    public int EncoderMaxChannels { get; set; } = 512;

    /// <summary>
    /// Gets or sets the dimensionality of the latent embedding z. Default 128 (paper: adaptive
    /// average pooling to 512, then a three-layer MLP producing z of 128 dimensions).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The whole recording is squeezed down to 128 numbers — the room's
    /// "fingerprint". Everything the decoder builds comes from just those numbers.
    /// </remarks>
    public int LatentDim { get; set; } = 128;

    #endregion

    #region Decoder and filtered noise shaping (paper §2.2)

    /// <summary>
    /// Gets or sets the number of noise bands M. Default 10 (paper: "M = 10 filtered noise
    /// signals").
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The noise is split into 10 frequency bands — think of a 10-band
    /// graphic equaliser. Each band gets its own fade-out curve, because in a real room high
    /// frequencies die away faster than low ones.
    /// </remarks>
    public int NumNoiseBands { get; set; } = 10;

    /// <summary>
    /// Gets or sets the FIR order P of each learned band-pass filter. Default 1023 (paper:
    /// "filters of order P = 1023 to provide sufficient low frequency response").
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Longer filters can separate low frequencies more precisely. 1023 taps
    /// is long enough to carve out clean bass bands.
    /// </remarks>
    public int NoiseFilterOrder { get; set; } = 1023;

    /// <summary>
    /// Gets or sets the estimated RIR length L in samples. Default 48000 (paper: "48,000 samples
    /// (1 second) in length" at 48 kHz).
    /// </summary>
    public int RIRLength { get; set; } = 48000;

    /// <summary>
    /// Gets or sets the early-component length E in samples. Default 2400 (paper: "E = 2400,
    /// corresponding to the first 50 ms"). Samples beyond E in the early branch are zeroed.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The first 50 milliseconds carry the direct sound and the first few
    /// distinct reflections. Those have real structure, so the model predicts them outright instead
    /// of approximating them with noise.
    /// </remarks>
    public int EarlyResponseLength { get; set; } = 2400;

    /// <summary>
    /// Gets or sets the number of decoder blocks. Each block upsamples with transposed convolutions
    /// and then refines with dilated convolutions, both FiLM-conditioned on z (paper, following the
    /// GAN-TTS generator).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The paper specifies the block STRUCTURE but not a block count, since the count is whatever
    /// is needed to upsample to <see cref="RIRLength"/>. Default 5 is an implementation choice, not
    /// a quoted value.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The decoder starts from a very short sequence and repeatedly stretches
    /// it, roughly doubling the length each time, until it is as long as the impulse response.
    /// </para>
    /// </remarks>
    public int NumDecoderBlocks { get; set; } = 5;

    #endregion

    #region Loss (paper §2.3)

    /// <summary>
    /// Gets or sets the STFT frame sizes for the multi-resolution STFT loss. Default
    /// {64, 512, 2048, 8192} (paper: "R = 4 resolutions with frame sizes of 64, 512, 2048, 8192").
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The model is scored by comparing spectrograms at four different
    /// zoom levels at once. Short frames judge the sharp early part; long frames judge the slow
    /// decay of the tail. The paper found this loss alone worked best.
    /// </remarks>
    public int[] StftFrameSizes { get; set; } = [64, 512, 2048, 8192];

    #endregion

    #region Enhancement

    /// <summary>Gets or sets the dereverberation strength (0-1).</summary>
    /// <remarks>
    /// <b>For Beginners:</b> How aggressively to remove the room sound when using the estimated
    /// response to clean up a recording. 0 leaves the audio untouched, 1 removes as much as it can.
    /// </remarks>
    public double DereverberationStrength { get; set; } = 0.8;

    /// <summary>Gets or sets the RT60 estimation window in seconds.</summary>
    /// <remarks>
    /// <b>For Beginners:</b> RT60 is how long the room takes to get 60 dB quieter. This is the span
    /// of the estimated response used to measure it.
    /// </remarks>
    public double RT60WindowSeconds { get; set; } = 1.0;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 1e-4;

    #endregion

    /// <summary>
    /// Validates that the configuration describes a buildable FiNS model.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Catches impossible settings up front — such as an early component
    /// longer than the whole impulse response — with a message naming the offending option, rather
    /// than letting them surface later as a confusing shape error deep inside the network.
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when an option is outside its valid range.</exception>
    public void Validate()
    {
        if (SampleRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(SampleRate), SampleRate, "Sample rate must be positive.");
        if (NumEncoderBlocks <= 0)
            throw new ArgumentOutOfRangeException(nameof(NumEncoderBlocks), NumEncoderBlocks, "Encoder block count must be positive.");
        if (EncoderKernelSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(EncoderKernelSize), EncoderKernelSize, "Encoder kernel size must be positive.");
        if (EncoderStride <= 0)
            throw new ArgumentOutOfRangeException(nameof(EncoderStride), EncoderStride, "Encoder stride must be positive.");
        if (EncoderMaxChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(EncoderMaxChannels), EncoderMaxChannels, "Encoder channel count must be positive.");
        if (LatentDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(LatentDim), LatentDim, "Latent dimension must be positive.");
        if (NumNoiseBands <= 0)
            throw new ArgumentOutOfRangeException(nameof(NumNoiseBands), NumNoiseBands, "Noise band count M must be positive.");
        if (NoiseFilterOrder <= 0)
            throw new ArgumentOutOfRangeException(nameof(NoiseFilterOrder), NoiseFilterOrder, "Noise filter order P must be positive.");
        if (NumDecoderBlocks <= 0)
            throw new ArgumentOutOfRangeException(nameof(NumDecoderBlocks), NumDecoderBlocks, "Decoder block count must be positive.");
        if (RIRLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(RIRLength), RIRLength, "RIR length must be positive.");

        // The early branch is written into the first E samples of an L-sample response, so E > L
        // would index past the end of the response the decoder produces.
        if (EarlyResponseLength <= 0 || EarlyResponseLength > RIRLength)
            throw new ArgumentOutOfRangeException(nameof(EarlyResponseLength), EarlyResponseLength,
                $"Early response length must be in (0, {nameof(RIRLength)}={RIRLength}].");

        if (StftFrameSizes is null || StftFrameSizes.Length == 0)
            throw new ArgumentOutOfRangeException(nameof(StftFrameSizes), "At least one STFT frame size is required for the multi-resolution loss.");
        foreach (int frameSize in StftFrameSizes)
        {
            if (frameSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(StftFrameSizes), frameSize, "STFT frame sizes must be positive.");
        }

        if (DereverberationStrength < 0.0 || DereverberationStrength > 1.0)
            throw new ArgumentOutOfRangeException(nameof(DereverberationStrength), DereverberationStrength, "Dereverberation strength must be in [0, 1].");
        if (RT60WindowSeconds <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(RT60WindowSeconds), RT60WindowSeconds, "RT60 window must be positive.");
        if (double.IsNaN(LearningRate) || double.IsInfinity(LearningRate) || LearningRate <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(LearningRate), LearningRate, "Learning rate must be finite and positive.");
    }
}

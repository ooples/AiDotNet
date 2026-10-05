namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Options for APNet (Ai and Ling 2023): an all-frame-level vocoder that predicts the log amplitude and the phase
/// spectra directly and reconstructs the waveform by an inverse STFT.</summary>
/// <remarks>
/// <para>
/// Defaults are the paper's (§IV): 16 kHz audio; 80-band mel spectrograms and spectra from a 1024-point FFT with a
/// 320-sample (20 ms) window and an 80-sample (5 ms) shift; ASP and PSP residual networks of 3 parallel blocks with
/// kernels 3, 7, 11 and 3 subblocks with dilations 1, 3, 5, 512 channels; the shared loss weights; HiFi-GAN's MPD and
/// MSD with the least-squares GAN loss; batch 16 and 8,000-sample segments.
/// </para>
/// <para>What the paper leaves open follows yangai520/APNet: input and output convolutions with kernel 7; mel spectrograms
/// from a centred STFT, Slaney bands 0–8 kHz, <c>ln(max(x, 1e-5))</c>; the feature-matching weight 2; 655 updates per
/// epoch (10,480 training utterances at batch 16).</para>
/// <para><b>For Beginners:</b> These options configure the APNet model. Default values follow the original paper settings.</para>
/// </remarks>
public class APNetOptions : AmplitudePhaseVocoderOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public APNetOptions(APNetOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        Channels = other.Channels;
        ResblockDilationSizes = other.ResblockDilationSizes.Select(d => (int[])d.Clone()).ToArray();
        FeatureMatchingWeight = other.FeatureMatchingWeight;
    }

    /// <summary>Creates the paper's APNet configuration.</summary>
    public APNetOptions()
    {
        SampleRate = 16000;
        HopSize = 80;
        WindowSize = 320;
        SegmentSize = 8000;
        UpdatesPerEpoch = 655;
        // The parallel residual blocks' kernels are VocoderOptions.ResblockKernelSizes (3, 7, 11).
        ResblockKernelSizes = [3, 7, 11];
    }

    /// <summary>Gets or sets the channels of the ASP and PSP networks (512).</summary>
    public int Channels { get; set; } = 512;

    /// <summary>Gets or sets the dilations of each block's subblocks (1, 3, 5).</summary>
    public int[][] ResblockDilationSizes { get; set; } = [[1, 3, 5], [1, 3, 5], [1, 3, 5]];

    /// <summary>Gets or sets the feature-matching weight (2).</summary>
    public double FeatureMatchingWeight { get; set; } = 2;
}

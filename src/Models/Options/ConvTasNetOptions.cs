using AiDotNet.Audio.Enhancement;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for ConvTasNet audio source separation models.
/// </summary>
public class ConvTasNetOptions : AudioNeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ConvTasNetOptions"/> class carrying
    /// this model's shipped defaults.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> You do not need to set any of these. They are the values this
    /// model has always used, moved here from its constructor so they can be seen and
    /// changed in one place.
    /// </para>
    /// <para>
    /// Carried over unchanged. Whether each matches the published paper is verified, and
    /// corrected where it does not, in a later phase of issue #2090.
    /// </para>
    /// </remarks>
    public ConvTasNetOptions()
    {
        SampleRate = 8000;
        EncoderDim = 512;
        KernelSize = 16;
        NumSources = 2;
        BottleneckDim = 128;
        HiddenDim = 512;
        NumBlocks = 8;
        NumRepeats = 3;
        TcnKernelSize = 3;
    }


    /// <summary>
    /// Gets or sets the sample rate.
    /// </summary>
    public int SampleRate { get; set; }

    /// <summary>
    /// Gets or sets the encoder dim.
    /// </summary>
    public int EncoderDim { get; set; }

    /// <summary>
    /// Gets or sets the kernel size.
    /// </summary>
    public int KernelSize { get; set; }

    /// <summary>
    /// Gets or sets the num sources.
    /// </summary>
    public int NumSources { get; set; }

    /// <summary>
    /// Gets or sets the bottleneck dim.
    /// </summary>
    public int BottleneckDim { get; set; }

    /// <summary>
    /// Gets or sets the hidden dim.
    /// </summary>
    public int HiddenDim { get; set; }

    /// <summary>
    /// Gets or sets the num blocks.
    /// </summary>
    public int NumBlocks { get; set; }

    /// <summary>
    /// Gets or sets the num repeats.
    /// </summary>
    public int NumRepeats { get; set; }

    /// <summary>
    /// Gets or sets the tcn kernel size.
    /// </summary>
    public int TcnKernelSize { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        ValidateCore();
    }
}

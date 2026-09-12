using AiDotNet.Audio.Enhancement;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for DCCRN (Deep Complex Convolution Recurrent Network) audio enhancement models.
/// </summary>
public class DCCRNOptions : AudioNeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="DCCRNOptions"/> class carrying
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
    public DCCRNOptions()
    {
        SampleRate = 16000;
        FftSize = 512;
        HopSize = 256;
        NumStages = 6;
        BaseChannels = 32;
        LstmHiddenDim = 256;
        NumLstmLayers = 2;
        UseComplexMask = true;
        KernelSize = 5;
        Stride = 2;
    }


    /// <summary>
    /// Gets or sets the sample rate.
    /// </summary>
    public int SampleRate { get; set; }

    /// <summary>
    /// Gets or sets the fft size.
    /// </summary>
    public int FftSize { get; set; }

    /// <summary>
    /// Gets or sets the hop size.
    /// </summary>
    public int HopSize { get; set; }

    /// <summary>
    /// Gets or sets the num stages.
    /// </summary>
    public int NumStages { get; set; }

    /// <summary>
    /// Gets or sets the base channels.
    /// </summary>
    public int BaseChannels { get; set; }

    /// <summary>
    /// Gets or sets the lstm hidden dim.
    /// </summary>
    public int LstmHiddenDim { get; set; }

    /// <summary>
    /// Gets or sets the num lstm layers.
    /// </summary>
    public int NumLstmLayers { get; set; }

    /// <summary>
    /// Gets or sets the use complex mask.
    /// </summary>
    public bool UseComplexMask { get; set; }

    /// <summary>
    /// Gets or sets the kernel size.
    /// </summary>
    public int KernelSize { get; set; }

    /// <summary>
    /// Gets or sets the stride.
    /// </summary>
    public int Stride { get; set; }

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

using AiDotNet.Audio.Enhancement;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for DeepFilterNet audio enhancement models.
/// </summary>
public class DeepFilterNetOptions : AudioNeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="DeepFilterNetOptions"/> class carrying
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
    public DeepFilterNetOptions()
    {
        SampleRate = 48000;
        FftSize = 960;
        HopSize = 480;
        NumErbBands = 32;
        HiddenDim = 96;
        DfOrder = 5;
        DfBins = 96;
        NumGruLayers = 2;
        Lookahead = 2;
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
    /// Gets or sets the num erb bands.
    /// </summary>
    public int NumErbBands { get; set; }

    /// <summary>
    /// Gets or sets the hidden dim.
    /// </summary>
    public int HiddenDim { get; set; }

    /// <summary>
    /// Gets or sets the df order.
    /// </summary>
    public int DfOrder { get; set; }

    /// <summary>
    /// Gets or sets the df bins.
    /// </summary>
    public int DfBins { get; set; }

    /// <summary>
    /// Gets or sets the num gru layers.
    /// </summary>
    public int NumGruLayers { get; set; }

    /// <summary>
    /// Gets or sets the lookahead.
    /// </summary>
    public int Lookahead { get; set; }

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

using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Speaker;

/// <summary>
/// Configuration options for the SpeakerVerifier.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The verifier answers two questions about a recording: "is this the same
/// person as this other recording?" (verification) and "which of the enrolled people is this?"
/// (identification). Each is a comparison score checked against a threshold, and the two
/// thresholds are set separately here.
/// </para>
/// <para>
/// <b>#2090 note.</b> This class existed but no constructor took it, so neither threshold had any
/// effect: the model applied a single 0.6 to both decisions. Both now default to 0.6 — the value
/// that was actually in force — and both are read. <c>VerificationThreshold</c> previously
/// defaulted to 0.7 here, a number nothing ever applied.
/// </para>
/// </remarks>
public class SpeakerVerifierOptions : SpeakerRecognitionOptions
{
    /// <summary>Initializes a new instance carrying the model's published defaults.</summary>
    public SpeakerVerifierOptions()
    {
        SampleRate = 16000;
        EmbeddingDimension = 256;
        HiddenDim = 256;
        NumEncoderLayers = 3;
        VerificationThreshold = 0.6;
        IdentificationThreshold = 0.6;
    }

    /// <summary>Initializes a new instance by copying another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public SpeakerVerifierOptions(SpeakerVerifierOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        SampleRate = other.SampleRate;
        EmbeddingDimension = other.EmbeddingDimension;
        HiddenDim = other.HiddenDim;
        NumEncoderLayers = other.NumEncoderLayers;
        VerificationThreshold = other.VerificationThreshold;
        IdentificationThreshold = other.IdentificationThreshold;
        EmbeddingModelPath = other.EmbeddingModelPath;
        OnnxOptions = other.OnnxOptions;
        MaxGradNorm = other.MaxGradNorm;
    }

    /// <summary>
    /// Gets or sets the similarity a pair must reach to be accepted as the same speaker.
    /// Default: 0.6.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Raise it to make false acceptances rarer at the cost of
    /// rejecting more genuine matches; lower it for the opposite trade.</para>
    /// </remarks>
    public double VerificationThreshold { get; set; }

    /// <summary>
    /// Gets or sets the similarity the best-matching enrolled speaker must reach before
    /// identification returns them rather than "unknown". Default: 0.6.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Kept separate from <see cref="VerificationThreshold"/> because identification compares
    /// against every enrolled speaker rather than one, so the chance of a spurious high score
    /// grows with the number enrolled and the threshold often needs to be stricter.
    /// </para>
    /// </remarks>
    public double IdentificationThreshold { get; set; }

    /// <summary>
    /// Gets or sets the path to an ONNX embedding model. Null selects the native trainable model.
    /// </summary>
    public string? EmbeddingModelPath { get; set; }

    /// <summary>
    /// Gets or sets the ONNX runtime options used when <see cref="EmbeddingModelPath"/> is supplied.
    /// </summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    /// <summary>Throws if a value this model requires has been left unset or is not positive.</summary>
    /// <exception cref="ArgumentException">Thrown when a required value is zero or negative.</exception>
    public void Validate()
    {
        ValidateCore();
        Require(VerificationThreshold, nameof(VerificationThreshold));
        Require(IdentificationThreshold, nameof(IdentificationThreshold));
    }
}

namespace AiDotNet.Models.Options;

/// <summary>
/// Base configuration options for audio neural network models.
/// </summary>
/// <remarks>
/// <para>
/// Every options class under <c>src/Audio</c> already derives from this, so it is where the
/// audio models' validation helpers belong — the same relationship
/// <see cref="DocumentNeuralNetworkOptions"/> has with the document models.
/// </para>
/// <para>
/// Unlike the other families, this base declares NO signal properties. The audio options
/// classes already declared their own <c>SampleRate</c>, <c>NumMels</c>, <c>FftSize</c>,
/// <c>HopLength</c> and the rest before any shared base existed — around thirty of them — so
/// declaring those names here would hide each leaf's property rather than share one.
/// </para>
/// </remarks>
public class AudioNeuralNetworkOptions : ModelHyperparameterOptions
{
    /// <summary>
    /// Throws if a value this model requires has been left unset.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Deliberately empty. This family spans synthesis, recognition, enhancement, classification
    /// and generation and shares no knob universally — a vocoder has no vocabulary, a classifier
    /// no speaking rate. Requiring a value the family does not share compiles clean and then
    /// throws at construction; that cost 30 failing tests in the GAN family, 11 in Document and
    /// 6 in Video before the rule was made explicit. A model that needs a value calls
    /// <c>Require</c> in its own <c>Validate()</c>.
    /// </para>
    /// </remarks>
    protected void ValidateCore()
    {
    }

    /// <summary>
    /// Gets or sets whether transcription returns per-segment timestamps by default.
    /// </summary>
    /// <value>Defaults to <c>false</c>, matching the previous method-parameter default.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Timestamps tell you WHEN each piece of text was spoken, not just
    /// what was said. They cost a little extra work, so they are off unless you ask for them.</para>
    /// <para>
    /// Declared here rather than per model because every <c>ISpeechRecognizer</c> implementation --
    /// 104 of them -- already branches on this flag; before this it could only be set per call, so
    /// there was no way to configure a model to return timestamps by default. WhisperOptions
    /// declared its own copy that nothing read.
    /// </para>
    /// </remarks>
    public bool ReturnTimestamps { get; set; } = false;
}

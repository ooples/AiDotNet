namespace AiDotNet.TextToSpeech.ProprietaryAPI;

/// <summary>Options for the Amazon Polly client (<c>SynthesizeSpeech</c>, <c>POST /v1/speech</c>).</summary>
/// <remarks>
/// <para>Defaults: region us-east-1, the neural engine, the voice Joanna, plain-text input and 16 kHz PCM (Polly returns
/// PCM at 8 or 16 kHz only). Requests are signed with AWS Signature Version 4 from the access key, the secret key and,
/// for temporary credentials, the session token. Polly accepts at most 3,000 billed characters per request.</para>
/// <para><b>For Beginners:</b> Set <see cref="AccessKeyId"/> and <see cref="SecretAccessKey"/> to an IAM user's keys with
/// the polly:SynthesizeSpeech permission.</para>
/// </remarks>
public class AmazonPollyOptions : CloudTtsOptions
{
    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public AmazonPollyOptions(AmazonPollyOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        Region = other.Region;
        AccessKeyId = other.AccessKeyId;
        SecretAccessKey = other.SecretAccessKey;
        SessionToken = other.SessionToken;
        Engine = other.Engine;
        LanguageCode = other.LanguageCode;
        UseSsml = other.UseSsml;
    }

    /// <summary>Creates the default options.</summary>
    public AmazonPollyOptions()
    {
        VoiceId = "Joanna";
        SampleRate = 16000;
        MaxTextLength = 3000;
    }

    /// <summary>Gets or sets the AWS region (us-east-1).</summary>
    public string Region { get; set; } = "us-east-1";

    /// <summary>Gets or sets the AWS access key id.</summary>
    public string AccessKeyId { get; set; } = string.Empty;

    /// <summary>Gets or sets the AWS secret access key.</summary>
    public string SecretAccessKey { get; set; } = string.Empty;

    /// <summary>Gets or sets the session token of temporary credentials, or null.</summary>
    public string? SessionToken { get; set; }

    /// <summary>Gets or sets the engine: standard, neural, long-form or generative (neural).</summary>
    public string Engine { get; set; } = "neural";

    /// <summary>Gets or sets the language of a bilingual voice, or null for the voice's default.</summary>
    public string? LanguageCode { get; set; }

    /// <summary>Gets or sets whether the text is SSML rather than plain text.</summary>
    public bool UseSsml { get; set; }
}

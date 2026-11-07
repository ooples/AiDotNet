using System.Net.Http;
using System.Text;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.ProprietaryAPI;

/// <summary>A client of Amazon Polly's <c>SynthesizeSpeech</c> API.</summary>
/// <typeparam name="T">The numeric type of the returned waveform.</typeparam>
/// <remarks>
/// <para>Sends <c>POST https://polly.{region}.amazonaws.com/v1/speech</c> with the JSON body
/// <c>{ Engine, OutputFormat: "pcm", SampleRate, Text, TextType, VoiceId[, LanguageCode] }</c>, signed with AWS Signature
/// Version 4 for the service "polly"; the response is signed 16-bit little-endian mono PCM.</para>
/// <para><b>For Beginners:</b> <c>new AmazonPolly&lt;float&gt;(new AmazonPollyOptions { AccessKeyId = "…", SecretAccessKey = "…" })
/// .Synthesize("Hello")</c> returns the spoken audio.</para>
/// </remarks>
public class AmazonPolly<T> : CloudTtsClientBase<T>
{
    /// <summary>Creates the client; a supplied <see cref="HttpClient"/> is used as is and not disposed.</summary>
    public AmazonPolly(AmazonPollyOptions? options = null, HttpClient? httpClient = null)
        : base(options ?? new AmazonPollyOptions(), httpClient)
    {
    }

    private AmazonPollyOptions Options => (AmazonPollyOptions)Settings;

    /// <inheritdoc />
    public override string ProviderName => "Amazon Polly";

    /// <summary>The clock the signature is dated by (the system clock; replaceable for tests).</summary>
    internal Func<DateTime> UtcNow { get; set; } = () => DateTime.UtcNow;

    /// <inheritdoc />
    protected override void ValidateConfiguration()
    {
        var o = Options;
        if (string.IsNullOrWhiteSpace(o.AccessKeyId) || string.IsNullOrWhiteSpace(o.SecretAccessKey))
            throw new InvalidOperationException("Amazon Polly needs AWS credentials: set AccessKeyId and SecretAccessKey.");
        if (string.IsNullOrWhiteSpace(o.Region)) throw new InvalidOperationException("Amazon Polly needs a region.");
        RequireSampleRate(8000, 16000);
    }

    /// <inheritdoc />
    protected override HttpRequestMessage CreateRequest(string text)
    {
        var o = Options;
        var body = new JObject
        {
            ["Engine"] = o.Engine,
            ["OutputFormat"] = "pcm",
            ["SampleRate"] = o.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture),
            ["Text"] = text,
            ["TextType"] = o.UseSsml ? "ssml" : "text",
            ["VoiceId"] = o.VoiceId,
        };
        if (!string.IsNullOrWhiteSpace(o.LanguageCode)) body["LanguageCode"] = o.LanguageCode;
        var bytes = Encoding.UTF8.GetBytes(body.ToString(Newtonsoft.Json.Formatting.None));
        var request = new HttpRequestMessage(HttpMethod.Post, $"{BaseUrl($"https://polly.{o.Region}.amazonaws.com")}/v1/speech")
        {
            Content = new ByteArrayContent(bytes),
        };
        request.Content.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("application/json");
        AwsSignatureV4.Sign(request, bytes, "polly", o.Region, o.AccessKeyId, o.SecretAccessKey, o.SessionToken, UtcNow());
        return request;
    }

    /// <inheritdoc />
    protected override Tensor<T> DecodeAudio(byte[] body, string? mediaType) => FromPcm16(body);
}

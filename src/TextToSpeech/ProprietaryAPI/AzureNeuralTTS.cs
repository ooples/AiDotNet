using System.Net.Http;
using System.Text;

namespace AiDotNet.TextToSpeech.ProprietaryAPI;

/// <summary>A client of the Azure AI Speech text-to-speech REST API.</summary>
/// <typeparam name="T">The numeric type of the returned waveform.</typeparam>
/// <remarks>
/// <para>Sends <c>POST https://{region}.tts.speech.microsoft.com/cognitiveservices/v1</c> with the SSML
/// <c>&lt;speak version='1.0' xml:lang='…'&gt;&lt;voice name='…'&gt;text&lt;/voice&gt;&lt;/speak&gt;</c>, the headers
/// <c>Ocp-Apim-Subscription-Key</c> (or <c>Authorization: Bearer</c>), <c>X-Microsoft-OutputFormat:
/// raw-{rate}-16bit-mono-pcm</c> and <c>User-Agent</c>; the response is signed 16-bit little-endian mono PCM.</para>
/// <para><b>For Beginners:</b> <c>new AzureNeuralTTS&lt;float&gt;(new AzureNeuralTTSOptions { ApiKey = "…", Region = "westeurope" })
/// .Synthesize("Hello")</c> returns the spoken audio.</para>
/// </remarks>
public class AzureNeuralTTS<T> : CloudTtsClientBase<T>
{
    /// <summary>Creates the client; a supplied <see cref="HttpClient"/> is used as is and not disposed.</summary>
    public AzureNeuralTTS(AzureNeuralTTSOptions? options = null, HttpClient? httpClient = null)
        : base(options ?? new AzureNeuralTTSOptions(), httpClient)
    {
    }

    private AzureNeuralTTSOptions Options => (AzureNeuralTTSOptions)Settings;

    /// <inheritdoc />
    public override string ProviderName => "Azure AI Speech";

    /// <inheritdoc />
    protected override void ValidateConfiguration()
    {
        var o = Options;
        if (string.IsNullOrWhiteSpace(o.ApiKey) && string.IsNullOrWhiteSpace(o.BearerToken))
            throw new InvalidOperationException("Azure AI Speech needs a resource key (ApiKey) or a BearerToken.");
        RequireSampleRate(8000, 16000, 22050, 24000, 44100, 48000);
        if (!o.TextIsSsml && string.IsNullOrWhiteSpace(o.VoiceId)) throw new InvalidOperationException("Azure AI Speech needs a voice name.");
    }

    /// <summary>The <c>X-Microsoft-OutputFormat</c> of a sample rate.</summary>
    internal static string OutputFormat(int sampleRate) => sampleRate switch
    {
        8000 => "raw-8khz-16bit-mono-pcm",
        16000 => "raw-16khz-16bit-mono-pcm",
        22050 => "raw-22050hz-16bit-mono-pcm",
        24000 => "raw-24khz-16bit-mono-pcm",
        44100 => "raw-44100hz-16bit-mono-pcm",
        48000 => "raw-48khz-16bit-mono-pcm",
        _ => throw new ArgumentOutOfRangeException(nameof(sampleRate)),
    };

    /// <summary>The SSML document of plain text for a voice (its locale is the voice name's first two parts).</summary>
    internal static string Ssml(string text, string voice)
    {
        var parts = voice.Split('-');
        string locale = parts.Length >= 2 ? $"{parts[0]}-{parts[1]}" : "en-US";
        return $"<speak version='1.0' xml:lang='{locale}'><voice name='{System.Security.SecurityElement.Escape(voice)}'>"
            + $"{System.Security.SecurityElement.Escape(text)}</voice></speak>";
    }

    /// <inheritdoc />
    protected override HttpRequestMessage CreateRequest(string text)
    {
        var o = Options;
        var request = new HttpRequestMessage(HttpMethod.Post, $"{BaseUrl($"https://{o.Region}.tts.speech.microsoft.com")}/cognitiveservices/v1")
        {
            Content = new StringContent(o.TextIsSsml ? text : Ssml(text, o.VoiceId), Encoding.UTF8),
        };
        // The service accepts exactly application/ssml+xml (415 otherwise); the body is UTF-8.
        request.Content.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("application/ssml+xml");
        if (!string.IsNullOrWhiteSpace(o.BearerToken))
            request.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", o.BearerToken);
        else
            request.Headers.Add("Ocp-Apim-Subscription-Key", o.ApiKey);
        request.Headers.Add("X-Microsoft-OutputFormat", OutputFormat(o.SampleRate));
        request.Headers.TryAddWithoutValidation("User-Agent", o.UserAgent);
        return request;
    }

    /// <inheritdoc />
    protected override Tensor<T> DecodeAudio(byte[] body, string? mediaType) => FromPcm16(body);
}

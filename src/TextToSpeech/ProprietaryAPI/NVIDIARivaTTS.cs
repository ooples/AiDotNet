using System.Net.Http;
using System.Text;

namespace AiDotNet.TextToSpeech.ProprietaryAPI;

/// <summary>A client of an NVIDIA Riva / Speech NIM text-to-speech server's HTTP API.</summary>
/// <typeparam name="T">The numeric type of the returned waveform.</typeparam>
/// <remarks>
/// <para>Sends <c>POST {endpoint}/v1/audio/synthesize</c> as multipart form data with the fields <c>text</c>,
/// <c>language</c>, <c>voice</c> and <c>sample_rate_hz</c>; the response is a 16-bit mono WAV file.</para>
/// <para><b>For Beginners:</b> <c>new NVIDIARivaTTS&lt;float&gt;(new NVIDIARivaTTSOptions { ApiEndpoint = "http://gpu-box:9000" })
/// .Synthesize("Hello")</c> returns the spoken audio.</para>
/// </remarks>
public class NVIDIARivaTTS<T> : CloudTtsClientBase<T>
{
    /// <summary>Creates the client; a supplied <see cref="HttpClient"/> is used as is and not disposed.</summary>
    public NVIDIARivaTTS(NVIDIARivaTTSOptions? options = null, HttpClient? httpClient = null)
        : base(options ?? new NVIDIARivaTTSOptions(), httpClient)
    {
    }

    private NVIDIARivaTTSOptions Options => (NVIDIARivaTTSOptions)Settings;

    /// <inheritdoc />
    public override string ProviderName => "NVIDIA Riva";

    /// <inheritdoc />
    protected override void ValidateConfiguration()
    {
        if (string.IsNullOrWhiteSpace(Options.ApiEndpoint))
            throw new InvalidOperationException("NVIDIA Riva is self-hosted: set ApiEndpoint to the TTS NIM server.");
    }

    /// <inheritdoc />
    protected override HttpRequestMessage CreateRequest(string text)
    {
        var o = Options;
        var form = new MultipartFormDataContent
        {
            { new StringContent(text, Encoding.UTF8), "text" },
            { new StringContent(o.LanguageCode), "language" },
            { new StringContent(o.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture)), "sample_rate_hz" },
        };
        if (!string.IsNullOrWhiteSpace(o.VoiceId)) form.Add(new StringContent(o.VoiceId), "voice");
        var request = new HttpRequestMessage(HttpMethod.Post, $"{BaseUrl("http://localhost:9000")}/v1/audio/synthesize") { Content = form };
        if (!string.IsNullOrWhiteSpace(o.ApiKey))
            request.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", o.ApiKey);
        return request;
    }

    /// <inheritdoc />
    protected override Tensor<T> DecodeAudio(byte[] body, string? mediaType) => FromWav(body);
}

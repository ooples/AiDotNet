using System.Net.Http;
using System.Text;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.ProprietaryAPI;

/// <summary>A client of the Google Cloud Text-to-Speech API.</summary>
/// <typeparam name="T">The numeric type of the returned waveform.</typeparam>
/// <remarks>
/// <para>Sends <c>POST https://texttospeech.googleapis.com/v1/text:synthesize</c> with
/// <c>{ input: { text | ssml }, voice: { languageCode, name }, audioConfig: { audioEncoding: "LINEAR16", sampleRateHertz } }</c>;
/// the response's <c>audioContent</c> is a base64 WAV file (LINEAR16 includes the header).</para>
/// <para><b>For Beginners:</b> <c>new GoogleCloudTTS&lt;float&gt;(new GoogleCloudTTSOptions { ApiKey = "…" }).Synthesize("Hello")</c>
/// returns the spoken audio.</para>
/// </remarks>
public class GoogleCloudTTS<T> : CloudTtsClientBase<T>
{
    /// <summary>Creates the client; a supplied <see cref="HttpClient"/> is used as is and not disposed.</summary>
    public GoogleCloudTTS(GoogleCloudTTSOptions? options = null, HttpClient? httpClient = null)
        : base(options ?? new GoogleCloudTTSOptions(), httpClient)
    {
    }

    private GoogleCloudTTSOptions Options => (GoogleCloudTTSOptions)Settings;

    /// <inheritdoc />
    public override string ProviderName => "Google Cloud Text-to-Speech";

    /// <inheritdoc />
    protected override void ValidateConfiguration()
    {
        var o = Options;
        if (string.IsNullOrWhiteSpace(o.ApiKey) && string.IsNullOrWhiteSpace(o.AccessToken))
            throw new InvalidOperationException("Google Cloud Text-to-Speech needs an ApiKey or an AccessToken.");
        if (o.SampleRate < 8000 || o.SampleRate > 48000)
            throw new InvalidOperationException($"Google Cloud Text-to-Speech resamples to 8–48 kHz; {o.SampleRate} Hz is outside it.");
    }

    /// <inheritdoc />
    protected override HttpRequestMessage CreateRequest(string text)
    {
        var o = Options;
        if (Encoding.UTF8.GetByteCount(text) > 5000)
            throw new ArgumentException("Google Cloud Text-to-Speech accepts at most 5,000 bytes of input per request.", nameof(text));
        var parts = o.VoiceId.Split('-');
        string language = o.LanguageCode ?? (parts.Length >= 2 ? $"{parts[0]}-{parts[1]}" : "en-US");
        var audio = new JObject { ["audioEncoding"] = "LINEAR16", ["sampleRateHertz"] = o.SampleRate };
        if (o.SpeakingRate is double rate) audio["speakingRate"] = rate;
        if (o.Pitch is double pitch) audio["pitch"] = pitch;
        var voice = new JObject { ["languageCode"] = language };
        if (!string.IsNullOrWhiteSpace(o.VoiceId)) voice["name"] = o.VoiceId;
        var body = new JObject
        {
            ["input"] = new JObject { [o.TextIsSsml ? "ssml" : "text"] = text },
            ["voice"] = voice,
            ["audioConfig"] = audio,
        };
        var request = new HttpRequestMessage(HttpMethod.Post, $"{BaseUrl("https://texttospeech.googleapis.com")}/v1/text:synthesize")
        {
            Content = new StringContent(body.ToString(Newtonsoft.Json.Formatting.None), Encoding.UTF8, "application/json"),
        };
        if (!string.IsNullOrWhiteSpace(o.AccessToken))
            request.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", o.AccessToken);
        else
            request.Headers.Add("X-Goog-Api-Key", o.ApiKey);
        return request;
    }

    /// <inheritdoc />
    protected override Tensor<T> DecodeAudio(byte[] body, string? mediaType)
    {
        var json = JObject.Parse(Encoding.UTF8.GetString(body));
        var content = json["audioContent"]?.ToString()
            ?? throw new InvalidDataException("Google Cloud Text-to-Speech returned no audioContent.");
        return FromWav(Convert.FromBase64String(content));
    }
}

using System.Net.Http;
using System.Text;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.ProprietaryAPI;

/// <summary>A client of the Murf text-to-speech API.</summary>
/// <typeparam name="T">The numeric type of the returned waveform.</typeparam>
/// <remarks>
/// <para>Sends <c>POST https://api.murf.ai/v1/speech/generate</c> with the <c>api-key</c> header and
/// <c>{ text, voiceId, format: "WAV", sampleRate, channelType: "MONO", encodeAsBase64: true[, style, rate, pitch,
/// multiNativeLocale] }</c>; the response's <c>encodedAudio</c> is a base64 WAV file.</para>
/// <para><b>For Beginners:</b> <c>new Murf&lt;float&gt;(new MurfOptions { ApiKey = "…" }).Synthesize("Hello")</c> returns the
/// spoken audio.</para>
/// </remarks>
public class Murf<T> : CloudTtsClientBase<T>
{
    /// <summary>Creates the client; a supplied <see cref="HttpClient"/> is used as is and not disposed.</summary>
    public Murf(MurfOptions? options = null, HttpClient? httpClient = null)
        : base(options ?? new MurfOptions(), httpClient)
    {
    }

    private MurfOptions Options => (MurfOptions)Settings;

    /// <inheritdoc />
    public override string ProviderName => "Murf";

    /// <inheritdoc />
    protected override void ValidateConfiguration()
    {
        RequireApiKey();
        RequireSampleRate(8000, 24000, 44100, 48000);
        if (string.IsNullOrWhiteSpace(Options.VoiceId)) throw new InvalidOperationException("Murf needs a voice id.");
    }

    /// <inheritdoc />
    protected override HttpRequestMessage CreateRequest(string text)
    {
        var o = Options;
        var body = new JObject
        {
            ["text"] = text,
            ["voiceId"] = o.VoiceId,
            ["format"] = "WAV",
            ["sampleRate"] = o.SampleRate,
            ["channelType"] = "MONO",
            ["encodeAsBase64"] = true,
        };
        if (!string.IsNullOrWhiteSpace(o.Style)) body["style"] = o.Style;
        if (o.Rate is int rate) body["rate"] = rate;
        if (o.Pitch is int pitch) body["pitch"] = pitch;
        if (!string.IsNullOrWhiteSpace(o.Locale)) body["multiNativeLocale"] = o.Locale;
        var request = new HttpRequestMessage(HttpMethod.Post, $"{BaseUrl("https://api.murf.ai")}/v1/speech/generate")
        {
            Content = new StringContent(body.ToString(Newtonsoft.Json.Formatting.None), Encoding.UTF8, "application/json"),
        };
        request.Headers.Add("api-key", o.ApiKey);
        return request;
    }

    /// <inheritdoc />
    protected override Tensor<T> DecodeAudio(byte[] body, string? mediaType)
    {
        var json = JObject.Parse(Encoding.UTF8.GetString(body));
        var encoded = json["encodedAudio"]?.ToString();
        if (string.IsNullOrEmpty(encoded))
            throw new InvalidDataException("Murf returned no encodedAudio.");
        return FromWav(Convert.FromBase64String(encoded!));
    }
}

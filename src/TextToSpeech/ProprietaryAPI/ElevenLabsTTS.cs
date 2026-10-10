using System.Net.Http;
using System.Text;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.ProprietaryAPI;

/// <summary>A client of the ElevenLabs text-to-speech API.</summary>
/// <typeparam name="T">The numeric type of the returned waveform.</typeparam>
/// <remarks>
/// <para>Sends <c>POST {endpoint}/v1/text-to-speech/{voice_id}?output_format=pcm_{rate}</c> with the <c>xi-api-key</c>
/// header and the JSON body <c>{ text, model_id, voice_settings }</c>; the response is signed 16-bit little-endian mono
/// PCM at the requested rate.</para>
/// <para><b>For Beginners:</b> <c>new ElevenLabsTTS&lt;float&gt;(new ElevenLabsTTSOptions { ApiKey = "…" }).Synthesize("Hello")</c>
/// returns the spoken audio.</para>
/// </remarks>
public class ElevenLabsTTS<T> : CloudTtsClientBase<T>
{
    /// <summary>Creates the client; a supplied <see cref="HttpClient"/> is used as is and not disposed.</summary>
    public ElevenLabsTTS(ElevenLabsTTSOptions? options = null, HttpClient? httpClient = null)
        : base(options ?? new ElevenLabsTTSOptions(), httpClient)
    {
    }

    private ElevenLabsTTSOptions Options => (ElevenLabsTTSOptions)Settings;

    /// <inheritdoc />
    public override string ProviderName => "ElevenLabs";

    /// <inheritdoc />
    protected override void ValidateConfiguration()
    {
        RequireApiKey();
        RequireSampleRate(8000, 16000, 22050, 24000, 32000, 44100, 48000);
        if (string.IsNullOrWhiteSpace(Options.VoiceId)) throw new InvalidOperationException("ElevenLabs needs a voice id.");
    }

    /// <inheritdoc />
    protected override HttpRequestMessage CreateRequest(string text)
    {
        var o = Options;
        var body = new JObject { ["text"] = text, ["model_id"] = o.ModelId };
        var settings = new JObject();
        if (o.Stability is double stability) settings["stability"] = stability;
        if (o.SimilarityBoost is double similarity) settings["similarity_boost"] = similarity;
        if (o.Style is double style) settings["style"] = style;
        if (o.Speed is double speed) settings["speed"] = speed;
        if (o.UseSpeakerBoost is bool boost) settings["use_speaker_boost"] = boost;
        if (settings.Count > 0) body["voice_settings"] = settings;
        var url = $"{BaseUrl("https://api.elevenlabs.io")}/v1/text-to-speech/{Uri.EscapeDataString(o.VoiceId)}?output_format=pcm_{o.SampleRate}";
        var request = new HttpRequestMessage(HttpMethod.Post, url)
        {
            Content = new StringContent(body.ToString(Newtonsoft.Json.Formatting.None), Encoding.UTF8, "application/json"),
        };
        request.Headers.Add("xi-api-key", o.ApiKey);
        return request;
    }

    /// <inheritdoc />
    protected override Tensor<T> DecodeAudio(byte[] body, string? mediaType) => FromPcm16(body);
}

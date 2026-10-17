using System.Net.Http;
using System.Text;
using AiDotNet.Audio.Codecs;
using AiDotNet.Helpers;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.ProprietaryAPI;

/// <summary>A client of the WellSaid Labs text-to-speech API.</summary>
/// <typeparam name="T">The numeric type of the returned waveform.</typeparam>
/// <remarks>
/// <para>Sends <c>POST https://api.wellsaidlabs.com/v1/tts/stream</c> with the <c>X-Api-Key</c> header,
/// <c>Accept: audio/mpeg</c> and <c>{ text, speaker_id[, model] }</c>; the response is MP3, decoded by
/// <see cref="Mp3Decoder"/>, mixed to mono and resampled to the requested rate when needed.</para>
/// <para><b>For Beginners:</b> <c>new WellSaidLabs&lt;float&gt;(new WellSaidLabsOptions { ApiKey = "…" }).Synthesize("Hello")</c>
/// returns the spoken audio.</para>
/// </remarks>
public class WellSaidLabs<T> : CloudTtsClientBase<T>
{
    /// <summary>Creates the client; a supplied <see cref="HttpClient"/> is used as is and not disposed.</summary>
    public WellSaidLabs(WellSaidLabsOptions? options = null, HttpClient? httpClient = null)
        : base(options ?? new WellSaidLabsOptions(), httpClient)
    {
    }

    private WellSaidLabsOptions Options => (WellSaidLabsOptions)Settings;

    /// <inheritdoc />
    public override string ProviderName => "WellSaid Labs";

    /// <inheritdoc />
    protected override void ValidateConfiguration()
    {
        RequireApiKey();
        if (string.IsNullOrWhiteSpace(Options.VoiceId)) throw new InvalidOperationException("WellSaid Labs needs a speaker id.");
        if (Options.SampleRate <= 0) throw new InvalidOperationException("The sample rate must be positive.");
    }

    /// <inheritdoc />
    protected override HttpRequestMessage CreateRequest(string text)
    {
        var o = Options;
        var body = new JObject { ["text"] = text, ["speaker_id"] = o.VoiceId };
        if (!string.IsNullOrWhiteSpace(o.Model)) body["model"] = o.Model;
        var request = new HttpRequestMessage(HttpMethod.Post, $"{BaseUrl("https://api.wellsaidlabs.com")}/v1/tts/stream")
        {
            Content = new StringContent(body.ToString(Newtonsoft.Json.Formatting.None), Encoding.UTF8, "application/json"),
        };
        request.Headers.Add("X-Api-Key", o.ApiKey);
        request.Headers.Accept.Add(new System.Net.Http.Headers.MediaTypeWithQualityHeaderValue("audio/mpeg"));
        return request;
    }

    /// <inheritdoc />
    protected override Tensor<T> DecodeAudio(byte[] body, string? mediaType)
    {
        var decoded = Mp3Decoder.Decode(body);
        int channels = decoded.Channels, n = decoded.Samples.Length / channels;
        var wave = new Tensor<T>(new[] { 1, 1, n });
        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            for (int c = 0; c < channels; c++) sum += decoded.Samples[i * channels + c];
            wave[0, 0, i] = NumOps.FromDouble(sum / channels);
        }
        if (decoded.SampleRate != Settings.SampleRate)
            wave = AudioHelper<T>.Resample(wave, decoded.SampleRate, Settings.SampleRate);
        return Mono(wave);
    }
}

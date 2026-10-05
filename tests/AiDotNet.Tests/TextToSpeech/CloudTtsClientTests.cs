using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.TextToSpeech.ProprietaryAPI;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// The hosted text-to-speech clients send each vendor's documented REST request and decode its audio response; the
/// shared base retries rate limits and server errors and reports failures. No request leaves the process: a fake
/// <see cref="HttpMessageHandler"/> records the request and returns a canned response.
/// </summary>
/// <remarks>
/// Before this change these classes were local neural networks named after the vendors: they never called the
/// services, so their "synthesis" was an untrained layer stack's output.
/// </remarks>
public class CloudTtsClientTests
{
    private sealed class FakeHandler : HttpMessageHandler
    {
        private readonly Queue<Func<HttpResponseMessage>> _responses = new();

        public List<(HttpRequestMessage Request, byte[] Body)> Requests { get; } = new();

        public FakeHandler Respond(HttpStatusCode status, byte[] body, string mediaType = "application/octet-stream",
            TimeSpan? retryAfter = null)
        {
            _responses.Enqueue(() =>
            {
                var response = new HttpResponseMessage(status) { Content = new ByteArrayContent(body) };
                response.Content.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue(mediaType);
                if (retryAfter is TimeSpan delay) response.Headers.RetryAfter = new System.Net.Http.Headers.RetryConditionHeaderValue(delay);
                return response;
            });
            return this;
        }

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var body = request.Content is null ? Array.Empty<byte>() : await request.Content.ReadAsByteArrayAsync().ConfigureAwait(false);
            Requests.Add((request, body));
            return _responses.Dequeue()();
        }
    }

    private static byte[] Pcm16(params short[] samples)
    {
        var bytes = new byte[samples.Length * 2];
        for (int i = 0; i < samples.Length; i++)
        {
            bytes[2 * i] = (byte)(samples[i] & 0xFF);
            bytes[2 * i + 1] = (byte)((samples[i] >> 8) & 0xFF);
        }
        return bytes;
    }

    private static byte[] Wav(int sampleRate, params short[] samples)
    {
        var data = Pcm16(samples);
        var wav = new List<byte>();
        void Add(string s) => wav.AddRange(Encoding.ASCII.GetBytes(s));
        void Int(int v) => wav.AddRange(BitConverter.GetBytes(v));
        void Short(short v) => wav.AddRange(BitConverter.GetBytes(v));
        Add("RIFF"); Int(36 + data.Length); Add("WAVE");
        Add("fmt "); Int(16); Short(1); Short(1); Int(sampleRate); Int(sampleRate * 2); Short(2); Short(16);
        Add("data"); Int(data.Length); wav.AddRange(data);
        return wav.ToArray();
    }

    private static string Header(HttpRequestMessage request, string name)
        => request.Headers.TryGetValues(name, out var values) ? string.Join(",", values)
            : request.Content is not null && request.Content.Headers.TryGetValues(name, out var content) ? string.Join(",", content) : "";

    [Fact(Timeout = 30000)]
    public async Task ElevenLabs_SendsTheConvertRequest_AndDecodesPcm()
    {
        var handler = new FakeHandler().Respond(HttpStatusCode.OK, Pcm16(16384, -32768, 0), "audio/pcm");
        using var client = new ElevenLabsTTS<double>(new ElevenLabsTTSOptions { ApiKey = "key", Stability = 0.5, SampleRate = 22050 },
            new HttpClient(handler));
        var wave = await client.SynthesizeAsync("Hello there");
        var (request, body) = handler.Requests.Single();
        Assert.Equal(HttpMethod.Post, request.Method);
        Assert.Equal("https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM?output_format=pcm_22050", request.RequestUri!.ToString());
        Assert.Equal("key", Header(request, "xi-api-key"));
        var json = JObject.Parse(Encoding.UTF8.GetString(body));
        Assert.Equal("Hello there", (string?)json["text"]);
        Assert.Equal("eleven_multilingual_v2", (string?)json["model_id"]);
        Assert.Equal(0.5, (double)json["voice_settings"]!["stability"]!);
        Assert.Equal(new[] { 0.5, -1.0, 0.0 }, wave.ToVector().ToArray());
        Assert.Equal(22050, client.SampleRate);
    }

    [Fact]
    public void AwsSignatureV4_ReproducesTheAwsTestSuitesGetVanilla()
    {
        // AWS Signature Version 4 test suite, "get-vanilla".
        var request = new HttpRequestMessage(HttpMethod.Get, "https://example.amazonaws.com/");
        AwsSignatureV4.Sign(request, Array.Empty<byte>(), "service", "us-east-1", "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY", null, new DateTime(2015, 8, 30, 12, 36, 0, DateTimeKind.Utc), includeContentHash: false);
        Assert.Equal("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, SignedHeaders=host;x-amz-date, "
            + "Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31", Header(request, "Authorization"));
    }

    [Fact(Timeout = 30000)]
    public async Task AmazonPolly_SendsASignedSynthesizeSpeechRequest_AndDecodesPcm()
    {
        var handler = new FakeHandler().Respond(HttpStatusCode.OK, Pcm16(8192), "audio/pcm");
        using var client = new AmazonPolly<double>(new AmazonPollyOptions
        {
            AccessKeyId = "AKID", SecretAccessKey = "secret", Region = "eu-west-1", SessionToken = "token",
        }, new HttpClient(handler));
        var wave = await client.SynthesizeAsync("Hi");
        var (request, body) = handler.Requests.Single();
        Assert.Equal("https://polly.eu-west-1.amazonaws.com/v1/speech", request.RequestUri!.ToString());
        var json = JObject.Parse(Encoding.UTF8.GetString(body));
        Assert.Equal("pcm", (string?)json["OutputFormat"]);
        Assert.Equal("16000", (string?)json["SampleRate"]);
        Assert.Equal("neural", (string?)json["Engine"]);
        Assert.Equal("Joanna", (string?)json["VoiceId"]);
        Assert.Equal("text", (string?)json["TextType"]);
        string authorization = Header(request, "Authorization");
        Assert.StartsWith("AWS4-HMAC-SHA256 Credential=AKID/", authorization);
        Assert.Contains("/eu-west-1/polly/aws4_request", authorization);
        Assert.Contains("x-amz-security-token", authorization);
        Assert.Equal("token", Header(request, "x-amz-security-token"));
        Assert.Equal(0.25, wave[0]);
    }

    [Fact(Timeout = 30000)]
    public async Task AmazonPolly_RejectsRatesItCannotReturnAsPcm()
    {
        using var client = new AmazonPolly<double>(new AmazonPollyOptions { AccessKeyId = "a", SecretAccessKey = "b", SampleRate = 24000 },
            new HttpClient(new FakeHandler()));
        await Assert.ThrowsAsync<InvalidOperationException>(() => client.SynthesizeAsync("Hi"));
    }

    [Fact(Timeout = 30000)]
    public async Task Azure_SendsEscapedSsmlWithTheOutputFormat()
    {
        var handler = new FakeHandler().Respond(HttpStatusCode.OK, Pcm16(-16384), "audio/x-wav");
        using var client = new AzureNeuralTTS<double>(new AzureNeuralTTSOptions { ApiKey = "k", Region = "westeurope", VoiceId = "de-DE-KatjaNeural" },
            new HttpClient(handler));
        var wave = await client.SynthesizeAsync("Tom & Jerry <3");
        var (request, body) = handler.Requests.Single();
        Assert.Equal("https://westeurope.tts.speech.microsoft.com/cognitiveservices/v1", request.RequestUri!.ToString());
        Assert.Equal("k", Header(request, "Ocp-Apim-Subscription-Key"));
        Assert.Equal("raw-24khz-16bit-mono-pcm", Header(request, "X-Microsoft-OutputFormat"));
        Assert.Equal("application/ssml+xml", request.Content!.Headers.ContentType!.MediaType);
        Assert.Equal("<speak version='1.0' xml:lang='de-DE'><voice name='de-DE-KatjaNeural'>Tom &amp; Jerry &lt;3</voice></speak>",
            Encoding.UTF8.GetString(body));
        Assert.Equal(-0.5, wave[0]);
    }

    [Fact(Timeout = 30000)]
    public async Task Google_SendsLinear16_AndDecodesTheBase64Wav()
    {
        var response = new JObject { ["audioContent"] = Convert.ToBase64String(Wav(24000, 16384, 0)) };
        var handler = new FakeHandler().Respond(HttpStatusCode.OK, Encoding.UTF8.GetBytes(response.ToString()), "application/json");
        using var client = new GoogleCloudTTS<double>(new GoogleCloudTTSOptions { ApiKey = "g", Pitch = 2 }, new HttpClient(handler));
        var wave = await client.SynthesizeAsync("Hello");
        var (request, body) = handler.Requests.Single();
        Assert.Equal("https://texttospeech.googleapis.com/v1/text:synthesize", request.RequestUri!.ToString());
        Assert.Equal("g", Header(request, "X-Goog-Api-Key"));
        var json = JObject.Parse(Encoding.UTF8.GetString(body));
        Assert.Equal("Hello", (string?)json["input"]!["text"]);
        Assert.Equal("en-US", (string?)json["voice"]!["languageCode"]);
        Assert.Equal("en-US-Neural2-F", (string?)json["voice"]!["name"]);
        Assert.Equal("LINEAR16", (string?)json["audioConfig"]!["audioEncoding"]);
        Assert.Equal(24000, (int)json["audioConfig"]!["sampleRateHertz"]!);
        Assert.Equal(2.0, (double)json["audioConfig"]!["pitch"]!);
        Assert.Equal(new[] { 0.5, 0.0 }, wave.ToVector().ToArray());
    }

    [Fact(Timeout = 30000)]
    public async Task Murf_RequestsBase64Wav_AndDecodesIt()
    {
        var response = new JObject { ["encodedAudio"] = Convert.ToBase64String(Wav(24000, -8192)), ["audioLengthInSeconds"] = 0.1 };
        var handler = new FakeHandler().Respond(HttpStatusCode.OK, Encoding.UTF8.GetBytes(response.ToString()), "application/json");
        using var client = new Murf<double>(new MurfOptions { ApiKey = "m" }, new HttpClient(handler));
        var wave = await client.SynthesizeAsync("Hello");
        var (request, body) = handler.Requests.Single();
        Assert.Equal("https://api.murf.ai/v1/speech/generate", request.RequestUri!.ToString());
        Assert.Equal("m", Header(request, "api-key"));
        var json = JObject.Parse(Encoding.UTF8.GetString(body));
        Assert.Equal("en-US-natalie", (string?)json["voiceId"]);
        Assert.Equal("WAV", (string?)json["format"]);
        Assert.True((bool)json["encodeAsBase64"]!);
        Assert.Equal("MONO", (string?)json["channelType"]);
        Assert.Equal(-0.25, wave[0]);
    }

    [Fact(Timeout = 30000)]
    public async Task Riva_SendsTheMultipartForm_AndDecodesWav()
    {
        var handler = new FakeHandler().Respond(HttpStatusCode.OK, Wav(22050, 32767 / 2 + 1), "audio/wav");
        using var client = new NVIDIARivaTTS<double>(new NVIDIARivaTTSOptions { ApiEndpoint = "http://gpu:9000/" }, new HttpClient(handler));
        var wave = await client.SynthesizeAsync("Hi");
        var (request, body) = handler.Requests.Single();
        Assert.Equal("http://gpu:9000/v1/audio/synthesize", request.RequestUri!.ToString());
        Assert.Equal("multipart/form-data", request.Content!.Headers.ContentType!.MediaType);
        string form = Encoding.UTF8.GetString(body);
        foreach (var (name, value) in new[] { ("text", "Hi"), ("language", "en-US"), ("sample_rate_hz", "22050"), ("voice", "Magpie-Multilingual.EN-US.Aria") })
            Assert.Contains($"name={name}\r\n\r\n{value}\r\n", form.Replace("\"", ""));
        Assert.Equal(0.5, wave[0], 6);
    }

    [Fact(Timeout = 30000)]
    public async Task WellSaid_DecodesTheMp3Stream_AtTheRequestedRate()
    {
        var mp3 = Convert.FromBase64String(AiDotNet.Tests.UnitTests.Audio.Mp3DecoderTests.Mono24kVector);
        var handler = new FakeHandler().Respond(HttpStatusCode.OK, mp3, "audio/mpeg");
        using var client = new WellSaidLabs<double>(new WellSaidLabsOptions { ApiKey = "w", SampleRate = 16000 }, new HttpClient(handler));
        var wave = await client.SynthesizeAsync("Hi");
        var (request, body) = handler.Requests.Single();
        Assert.Equal("https://api.wellsaidlabs.com/v1/tts/stream", request.RequestUri!.ToString());
        Assert.Equal("w", Header(request, "X-Api-Key"));
        Assert.Contains("audio/mpeg", Header(request, "Accept"));
        var json = JObject.Parse(Encoding.UTF8.GetString(body));
        Assert.Equal("3", (string?)json["speaker_id"]);
        // Half a second of 24 kHz audio (plus the encoder's padding) resampled to 16 kHz.
        Assert.InRange(wave.Length, 8000, 8000 + 2400);
        double peak = wave.ToVector().ToArray().Max(Math.Abs);
        Assert.InRange(peak, 0.3, 0.5);
    }

    [Fact(Timeout = 30000)]
    public async Task RateLimitsAndServerErrors_AreRetried_ThenTheResponseIsDecoded()
    {
        var handler = new FakeHandler()
            .Respond((HttpStatusCode)429, Encoding.UTF8.GetBytes("slow down"), "text/plain", TimeSpan.Zero)
            .Respond(HttpStatusCode.ServiceUnavailable, Encoding.UTF8.GetBytes("busy"), "text/plain", TimeSpan.Zero)
            .Respond(HttpStatusCode.OK, Pcm16(16384), "audio/pcm");
        using var client = new ElevenLabsTTS<double>(new ElevenLabsTTSOptions { ApiKey = "k", InitialRetryDelayMs = 0 }, new HttpClient(handler));
        var wave = await client.SynthesizeAsync("Hello");
        Assert.Equal(3, handler.Requests.Count);
        Assert.Equal(0.5, wave[0]);
    }

    [Fact(Timeout = 30000)]
    public async Task ClientErrors_AreReportedWithTheVendorsMessage()
    {
        var handler = new FakeHandler().Respond(HttpStatusCode.Unauthorized, Encoding.UTF8.GetBytes("{\"detail\":\"invalid api key\"}"), "application/json");
        using var client = new ElevenLabsTTS<double>(new ElevenLabsTTSOptions { ApiKey = "bad" }, new HttpClient(handler));
        var error = await Assert.ThrowsAsync<CloudTtsException>(() => client.SynthesizeAsync("Hello"));
        Assert.Equal(HttpStatusCode.Unauthorized, error.StatusCode);
        Assert.Contains("invalid api key", error.Message);
        Assert.Single(handler.Requests);
    }

    [Fact(Timeout = 30000)]
    public async Task MissingCredentialsAndOverlongText_FailBeforeAnyRequest()
    {
        var handler = new FakeHandler();
        using var noKey = new ElevenLabsTTS<double>(new ElevenLabsTTSOptions(), new HttpClient(handler));
        await Assert.ThrowsAsync<InvalidOperationException>(() => noKey.SynthesizeAsync("Hello"));
        using var google = new GoogleCloudTTS<double>(new GoogleCloudTTSOptions { ApiKey = "g" }, new HttpClient(handler));
        await Assert.ThrowsAsync<ArgumentException>(() => google.SynthesizeAsync(new string('a', 5001)));
        Assert.Empty(handler.Requests);
    }
}

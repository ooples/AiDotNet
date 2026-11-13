using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.Whisper;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// The openai-whisper port against openai-whisper 20240930 (tools/reference-data/openai_whisper_reference.py), a tiny
/// random model of large-v3-turbo's layout: log-mel, encoder, decoder, language detection, the decoding logit filters,
/// greedy decoding, DTW word timing (English with punctuation, Chinese per character), add_word_timestamps and a whole
/// transcribe with word timestamps.
/// </summary>
public class OpenAiWhisperTests
{
    private static JObject? _fixture;

    private static JObject Reference()
    {
        if (_fixture is not null) return _fixture;
        const string fileName = "openai_whisper_reference.json";
        string output = Path.Combine(AppContext.BaseDirectory, "Audio", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(output) && dir is not null; dir = dir.Parent)
            output = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "Audio", "ReferenceData", fileName);
        return _fixture = JObject.Parse(File.ReadAllText(output));
    }

    private static double[] Values(JToken token) => token.Select(v => (double)v).ToArray();

    private static double[] Floats(JToken token)
    {
        var bytes = Convert.FromBase64String((string)token!);
        return Enumerable.Range(0, bytes.Length / 4).Select(i => (double)BitConverter.ToSingle(bytes, 4 * i)).ToArray();
    }

    private static double[] Doubles(JToken token)
    {
        var bytes = Convert.FromBase64String((string)token!);
        return Enumerable.Range(0, bytes.Length / 8).Select(i => BitConverter.ToDouble(bytes, 8 * i)).ToArray();
    }

    private static void Close(double[] expected, double[] actual, string what, double tolerance = 1e-8)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance * Math.Max(1, Math.Abs(expected[i])),
                $"{what}[{i}]: reference {expected[i]:R}, port {actual[i]:R}.");
    }

    private static (OpenAiWhisperNetwork<double> Model, OpenAiWhisperTranscriber<double> Transcriber) Model()
    {
        var f = Reference();
        var dims = f["dims"]!;
        var d = new WhisperDimensions((int)dims["n_mels"]!, (int)dims["n_audio_ctx"]!, (int)dims["n_audio_state"]!, (int)dims["n_audio_head"]!,
            (int)dims["n_audio_layer"]!, (int)dims["n_vocab"]!, (int)dims["n_text_ctx"]!, (int)dims["n_text_state"]!, (int)dims["n_text_head"]!,
            (int)dims["n_text_layer"]!);
        var engine = AiDotNetEngine.Current;
        var model = new OpenAiWhisperNetwork<double>(engine, new List<LayerBase<double>>(), d, new Random(0));
        string path = Path.Combine(Path.GetTempPath(), $"whisper_{Guid.NewGuid():N}.safetensors");
        File.WriteAllBytes(path, Convert.FromBase64String((string)f["safetensors_base64"]!));
        var weights = new WeightLoader().LoadWeights(path);
        File.Delete(path);
        var used = new HashSet<string>();
        model.LoadTorchWeights((name, shape) =>
        {
            Assert.True(weights.ContainsKey(name), $"no tensor {name}");
            used.Add(name);
            var v = weights[name].ToVector().Select(x => (double)x).ToArray();
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), v.Length);
            return v;
        });
        Assert.Empty(weights.Keys.Where(k => !used.Contains(k)));
        return (model, new OpenAiWhisperTranscriber<double>(engine, model));
    }

    private static Tensor<double> Window(double[] audio, int mels)
    {
        var mel = OpenAiWhisperTranscriber<double>.LogMelSpectrogram(audio, mels, OpenAiWhisperTranscriber<double>.ChunkSamples);
        var window = new Tensor<double>(new[] { mels, OpenAiWhisperTranscriber<double>.ChunkFrames });
        for (int m = 0; m < mels; m++)
            for (int t = 0; t < OpenAiWhisperTranscriber<double>.ChunkFrames; t++) window[m, t] = mel[m, t];
        return window;
    }

    private static void SameWords(JToken expected, IReadOnlyList<WhisperWordTiming> actual, string what)
    {
        var e = expected.ToArray();
        Assert.True(e.Length == actual.Count, $"{what}: {e.Length} words vs {actual.Count}: {string.Join("|", actual.Select(w => w.Word))}");
        for (int i = 0; i < e.Length; i++)
        {
            Assert.Equal((string)e[i]["word"]!, actual[i].Word);
            Assert.True(Math.Abs((double)e[i]["start"]! - actual[i].Start) < 1e-9 && Math.Abs((double)e[i]["end"]! - actual[i].End) < 1e-9,
                $"{what} '{actual[i].Word}': {(double)e[i]["start"]!}-{(double)e[i]["end"]!} vs {actual[i].Start}-{actual[i].End}");
            Close(new[] { (double)e[i]["probability"]! }, new[] { actual[i].Probability }, what + " probability");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task LogMel_MatchesOpenAiWhisper()
    {
        await Task.Yield();
        var f = Reference();
        int mels = (int)f["dims"]!["n_mels"]!;
        var mel = OpenAiWhisperTranscriber<double>.LogMelSpectrogram(Floats(f["short_audio"]!), mels);
        var flat = new double[mel.Length];
        int frames = mel.GetLength(1);
        for (int m = 0; m < mels; m++) for (int t = 0; t < frames; t++) flat[m * frames + t] = mel[m, t];
        Close(Values(f["short_mel"]!), flat, "mel");
    }

    [Fact(Timeout = 600000)]
    public async Task Network_MatchesOpenAiWhisper()
    {
        await Task.Yield();
        var f = Reference();
        var (model, transcriber) = Model();
        using var _ = new NoGradScope<double>();
        var features = model.Encode(Window(Floats(f["audio"]!), model.Dimensions.Mels));
        Close(Values(f["encoder"]!), features.ToArray().Take(40 * model.Dimensions.AudioState).ToArray(), "encoder");
        Close(new[] { (double)f["encoder_sum"]! }, new[] { features.ToArray().Sum() }, "encoder sum");
        var tokens = f["decoder_tokens"]!.Select(v => (int)v).ToArray();
        var logits = model.Logits(tokens, features);
        int v = model.Dimensions.Vocabulary;
        Close(Doubles(f["decoder_last_logits"]!), Enumerable.Range(0, v).Select(i => logits[tokens.Length - 1, i]).ToArray(), "logits");
        var probs = transcriber.DetectLanguage(features);
        foreach (var pair in (JObject)f["language_probs"]!) Close(new[] { (double)pair.Value! }, new[] { probs[pair.Key] }, "language " + pair.Key);
    }

    [Fact(Timeout = 120000)]
    public async Task LogitFilters_MatchDecodingTask()
    {
        await Task.Yield();
        var f = Reference();
        var (model, transcriber) = Model();
        var initial = transcriber.SotSequence("en");
        int k = 0;
        foreach (var c in f["filters"]!)
        {
            var sampled = c["sampled"]!.Select(x => (int)x).ToList();
            var scores = Enumerable.Range(0, model.Dimensions.Vocabulary).Select(i => 4 * Math.Sin(i * 0.618034 + k)).ToArray();
            transcriber.ApplyFilters(scores, initial.Concat(sampled).ToList(), initial.Length, 50);
            var runs = new List<int[]>();
            int? start = null;
            for (int i = 0; i <= scores.Length; i++)
            {
                bool allowed = i < scores.Length && !double.IsNegativeInfinity(scores[i]);
                if (allowed && start is null) start = i;
                if (!allowed && start is not null) { runs.Add(new[] { start.Value, i }); start = null; }
            }
            var expected = c["allowed"]!.Select(r => r.Select(x => (int)x).ToArray()).ToList();
            Assert.True(expected.Count == runs.Count && expected.Zip(runs, (a, b) => a.SequenceEqual(b)).All(x => x),
                $"case {k}: expected {string.Join(" ", expected.Take(6).Select(r => $"[{r[0]},{r[1]})"))}, got {string.Join(" ", runs.Take(6).Select(r => $"[{r[0]},{r[1]})"))}");
            k++;
        }
    }

    [Fact(Timeout = 600000)]
    public async Task DecodingAndWordTiming_MatchOpenAiWhisper()
    {
        await Task.Yield();
        var f = Reference();
        var (model, transcriber) = Model();
        var window = Window(Floats(f["audio"]!), model.Dimensions.Mels);
        var result = transcriber.Decode(window, 0.0, "en", null, new Random(0));
        var d = f["decode"]!;
        Assert.Equal(d["tokens"]!.Select(x => (int)x).ToList(), result.Tokens);
        Assert.Equal((string)d["text"]!, result.Text);
        Close(new[] { (double)d["avg_logprob"]!, (double)d["no_speech_prob"]! }, new[] { result.AvgLogprob, result.NoSpeechProb }, "decode");
        Close(new[] { (double)d["compression_ratio"]! }, new[] { result.CompressionRatio }, "compression ratio");
        SameWords(f["alignment_en"]!, transcriber.FindAlignment("en", f["alignment_en_tokens"]!.Select(x => (int)x).ToList(), window, 700), "english");
        SameWords(f["alignment_zh"]!, transcriber.FindAlignment("zh", f["alignment_zh_tokens"]!.Select(x => (int)x).ToList(), window, 700), "chinese");
        SameWords(f["alignment_decoded"]!, transcriber.FindAlignment("en", f["alignment_decoded_tokens"]!.Select(x => (int)x).ToList(), window, 700), "decoded");
        var segments = f["word_segments_in"]!.Select(s => new WhisperTranscriptSegment
        {
            Seek = (int)s["seek"]!, Start = (double)s["start"]!, End = (double)s["end"]!, Tokens = s["tokens"]!.Select(x => (int)x).ToList(),
        }).ToList();
        transcriber.AddWordTimestamps(segments, "en", window, 700, 0.0);
        var expected = f["word_timestamps"]!.ToArray();
        for (int i = 0; i < segments.Count; i++)
        {
            Assert.True(Math.Abs((double)expected[i]["start"]! - segments[i].Start) < 1e-9 && Math.Abs((double)expected[i]["end"]! - segments[i].End) < 1e-9,
                $"segment {i}: {(double)expected[i]["start"]!}-{(double)expected[i]["end"]!} vs {segments[i].Start}-{segments[i].End}");
            SameWords(expected[i]["words"]!, segments[i].Words, $"segment {i}");
        }
    }

    [Fact(Timeout = 900000)]
    public async Task Transcribe_MatchesOpenAiWhisper()
    {
        await Task.Yield();
        var f = Reference();
        var (_, transcriber) = Model();
        var result = transcriber.TranscribeAudio(Floats(f["audio"]!), new[] { 0.0 }, true, null, new Random(0));
        var t = f["transcribe"]!;
        Assert.Equal((string)t["language"]!, result.Language);
        Assert.Equal((string)t["text"]!, result.Text);
        var segments = t["segments"]!.ToArray();
        Assert.Equal(segments.Length, result.Segments.Count);
        for (int i = 0; i < segments.Length; i++)
        {
            var e = segments[i];
            var a = result.Segments[i];
            Assert.Equal((int)e["seek"]!, a.Seek);
            Assert.Equal((string)e["text"]!, a.Text);
            Assert.Equal(e["tokens"]!.Select(x => (int)x).ToList(), a.Tokens);
            Assert.True(Math.Abs((double)e["start"]! - a.Start) < 1e-9 && Math.Abs((double)e["end"]! - a.End) < 1e-9, $"segment {i} bounds");
            Close(new[] { (double)e["avg_logprob"]!, (double)e["no_speech_prob"]!, (double)e["compression_ratio"]! },
                new[] { a.AvgLogprob, a.NoSpeechProb, a.CompressionRatio }, $"segment {i}");
            SameWords(e["words"]!, a.Words, $"segment {i}");
        }
    }
}

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Audio.Foundations;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Vocoders;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// FireRedTTS's networks against the release's code on tiny sizes (tools/reference-data/fireredtts_reference.py): the
/// speaker mel and ECAPA-TDNN, the GPT's XTTS losses and logits, its generation and Hugging Face's logits processors,
/// the flow decoder's prompt features, μ, inference mel and loss, BigVGAN, the resamplers, and the 1S speech tokenizer's
/// SSL features, encodings, tokens and EMA step.
/// </summary>
public class FireRedTtsModuleTests
{
    private static readonly int[] Text = { 3, 7, 11, 4, 9 };
    private static readonly int[] Codes = { 5, 17, 2, 2, 9, 13, 8, 6 };
    private static readonly int[] FlowTokens = { 5, 300, 16383, 128, 77 };

    private static readonly Lazy<JObject> Data = new(() =>
    {
        const string fileName = "fireredtts_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(path));
    });

    private static JObject F => Data.Value;
    private static IEngine Engine => AiDotNetEngine.Current;
    private static double[] Doubles(JToken t) => t.Select(v => (double)v).ToArray();
    private static Tensor<double> Tensor(JToken t, params int[] shape) => new(shape, new Vector<double>(Doubles(t)));

    private static void Close(JToken expected, Tensor<double> actual, string what, double tolerance = 1e-9) =>
        Close(Doubles(expected), actual.ToArray(), what, tolerance);

    private static void Close(double[] e, double[] actual, string what, double tolerance = 1e-9)
    {
        Assert.True(e.Length == actual.Length, $"{what}: {e.Length} values expected, {actual.Length} computed");
        for (int i = 0; i < e.Length; i++)
            Assert.True(Math.Abs(e[i] - actual[i]) <= tolerance * Math.Max(1, Math.Abs(e[i])), $"{what}[{i}]: {e[i]} vs {actual[i]}");
    }

    private static Func<string, int[], double[]> Reader(string prefix) => (name, shape) =>
    {
        var token = ((JObject)F["state"]!)[prefix + name] ?? throw new InvalidDataException($"No tensor '{prefix + name}'.");
        var values = Doubles(token);
        if (shape.Length > 0 && shape.Aggregate(1, (a, b) => a * b) != values.Length)
            throw new InvalidDataException($"'{prefix + name}' has {values.Length} values, not [{string.Join(", ", shape)}].");
        return values;
    };

    private static Tensor<double> Transposed(Tensor<double> x) => Engine.TensorTranspose(x);

    private static double CrossEntropy(Tensor<double> logits, IReadOnlyList<int> targets)
    {
        double total = 0;
        for (int i = 0; i < targets.Count; i++)
        {
            double max = double.NegativeInfinity;
            for (int v = 0; v < logits.Shape[1]; v++) max = Math.Max(max, logits[i, v]);
            double sum = 0;
            for (int v = 0; v < logits.Shape[1]; v++) sum += Math.Exp(logits[i, v] - max);
            total += -(logits[i, targets[i]] - max - Math.Log(sum));
        }
        return total / targets.Count;
    }

    internal static IndexTtsEcapa<double> Speaker()
    {
        var layers = new List<LayerBase<double>>();
        var ecapa = new IndexTtsEcapa<double>(Engine, layers, new IndexTtsEcapaConfiguration(80, 8,
            new[] { 16, 16, 16, 16, 48 }, new[] { 5, 3, 3, 3, 1 }, new[] { 1, 2, 3, 4, 1 }, 8, 8, 4), new Random(1));
        ecapa.Load(FireRedTTS<double>.EcapaReader(Reader("speaker.")), "");
        foreach (var layer in layers) layer.SetTrainingMode(false);            // BatchNorm on its running statistics
        return ecapa;
    }

    internal static FireRedTtsGpt<double> Gpt()
    {
        var gpt = new FireRedTtsGpt<double>(Engine, new List<LayerBase<double>>(), new FireRedTtsGptConfiguration(16, 2, 2, 20, 40, 30, 16386, 27, 28,
            16384, 16385, 8, 6, 0.1, 0, 0), new Random(2));
        gpt.Load(Reader("gpt."), 30);
        return gpt;
    }

    // The flow's frozen input embedding (its own copy of the tokenizer's codebooks, as fireredtts_token2wav.pt holds it).
    private static Tensor<double>[] FlowCodebook() => Enumerable.Range(0, 2)
        .Select(b => new Tensor<double>(new[] { 128, 4 }, new Vector<double>(Doubles(F["state"]![$"flow.input_embedding.codebook.{b}.weight"]!)))).ToArray();

    internal static FireRedTtsFlow<double> Flow()
    {
        var books = FlowCodebook();
        var flow = new FireRedTtsFlow<double>(Engine, new List<LayerBase<double>>(), new FireRedTtsFlowConfiguration(100, 128, 4, 8, 2, 16, 2, 1, 0.01,
            1, 4, 8, 4, 1, 1, 3, new[] { 8, 8 }, 2, 4, 1, 1, 0.7, 0.2, 1e-6), b => books[b], new Random(3));
        flow.Load(Reader("flow."), "");
        return flow;
    }

    [Fact]
    public void SpeakerEncoder_MatchesTheRelease()
    {
        using var _ = new NoGradScope<double>();
        var mel = new DifferentiableMel<double>(Engine, 16000, 1024, 160, 640, 80, 0, 8000, singlePrecisionBasis: true).Forward(Tensor(F["wave16k"]!, 16000));
        Close(F["speaker_mel"]!, mel, "speaker mel");
        Close(F["speaker_embedding"]!, Speaker().Forward(mel), "speaker embedding");
    }

    [Fact]
    public void Gpt_TrainingForward_MatchesXtts()
    {
        using var _ = new NoGradScope<double>();
        var gpt = Gpt();
        var prefix = gpt.Prefix(Tensor(F["speaker_embedding"]!, 8));
        Close(F["cond_latent"]!, prefix, "conditioning latent");
        var (inputs, targets, textTargets) = gpt.XttsSequences(Text, Codes);
        var (textLogits, audioLogits) = gpt.Forward(prefix, Text, null, inputs, maskFinalKeys: true);
        Close(F["mel_logits"]!, audioLogits, "audio logits");
        Close(new[] { (double)F["loss_text"]![0]! }, new[] { CrossEntropy(textLogits, textTargets) }, "text cross-entropy");
        Close(new[] { (double)F["loss_mel"]![0]! }, new[] { CrossEntropy(audioLogits, targets) }, "audio cross-entropy");
    }

    [Fact]
    public void Gpt_Generation_MatchesHuggingFace()
    {
        using var _ = new NoGradScope<double>();
        var gpt = Gpt();
        var prefix = gpt.Prefix(Tensor(F["speaker_embedding"]!, 8));
        var greedy = gpt.Generate(prefix, Text, null, new FireRedTtsDecoding(false, 30, 0.85, 0.75, 2.0, 1, 0, 627), new Random(4));
        Assert.Equal(F["greedy"]!.Select(v => (int)v).ToArray(), greedy[0]);
        var sampled = gpt.Generate(prefix, Text, null, new FireRedTtsDecoding(true, 1, 0.85, 0.75, 2.0, 7, 1, 627), new Random(5));
        var want = F["sampled"]!.Select(s => s.Select(v => (int)v).ToArray()).ToArray();
        Assert.Equal(want.Length, sampled.Count);
        for (int i = 0; i < want.Length; i++) Assert.Equal(want[i], sampled[i]);
    }

    [Fact]
    public void LogitsProcessors_MatchHuggingFace()
    {
        var row = Doubles(F["logits"]!);
        FireRedTtsGpt<double>.Process(row, new HashSet<int> { 1, 3, 7, 12 }, new FireRedTtsDecoding(true, 30, 0.85, 0.75, 2.0, 7, 1, 627));
        Close(Doubles(F["processed"]!), row.Select(v => Math.Max(v, -1e30)).ToArray(), "processed logits");
    }

    [Fact]
    public void Choose_KeepsTheSecondShortest()
    {
        var picked = FireRedTtsGpt<double>.Choose(new List<int[]> { new[] { 1, 2, 3 }, new[] { 4 }, new[] { 5, 6 }, new[] { 7, 8 } }, 1);
        Assert.Equal(new[] { 5, 6 }, picked);
    }

    [Fact]
    public void Flow_MatchesTheRelease()
    {
        using var _ = new NoGradScope<double>();
        var flow = Flow();
        var promptMel = new DifferentiableMel<double>(Engine, 16000, 1024, 160, 1024, 100, 0, 8000, singlePrecisionBasis: true).Forward(Tensor(F["wave16k"]!, 16000));
        Close(F["prompt_mel"]!, promptMel, "prompt mel");
        var prompt = flow.EncodePrompt(promptMel);
        Close(F["prompt_features"]!, prompt, "prompt features");
        var embedded = flow.Embed(FlowTokens);
        Close(F["embedded"]!, embedded, "token embedding");
        var encoded = flow.Encode(embedded, prompt);
        Close(F["encoded"]!, encoded, "encoder");
        var mu = flow.Condition(encoded);
        Close(F["mu"]!, mu, "mu");
        int frames = 4 * FlowTokens.Length;
        var mel = flow.Inference(FlowTokens, promptMel, 3, Transposed(Tensor(F["flow_noise"]!, 100, frames)));
        Close(F["flow_mel"]!, mel, "inference mel");
        var target = Transposed(Tensor(F["target_mel"]!, 100, frames));
        var noise = Transposed(Tensor(F["loss_noise"]!, 100, frames));
        Close(F["flow_loss"]!, flow.Loss(mu, target, 0.37, noise, false), "flow loss");
        Close(F["flow_loss_dropped"]!, flow.Loss(mu, target, 0.37, noise, true), "flow loss without mu");
    }

    [Fact]
    public void Vocoder_MatchesTheRelease()
    {
        using var _ = new NoGradScope<double>();
        var vocoder = new BigVganGenerator<double>(Engine, new Random(6), 100, 64, new[] { 5, 3, 2, 2, 2, 2 }, new[] { 11, 7, 4, 4, 4, 4 }, new[] { 3 },
            new[] { new[] { 1, 3, 5 } }, snakeBeta: true, tanhAtFinal: false, biasAtFinal: false);
        var state = (JObject)F["state"]!;
        vocoder.LoadTorch(name => state["bigvgan." + name] is { } t ? Doubles(t) : null);
        int frames = 4 * FlowTokens.Length;
        var mel = Engine.Reshape(Engine.TensorTranspose(Tensor(F["flow_mel"]!, frames, 100)), new[] { 1, 100, frames });
        Close(F["waveform"]!, vocoder.Forward(mel), "waveform");
    }

    [Fact]
    public void Resamplers_MatchTorchaudio()
    {
        using var _ = new NoGradScope<double>();
        Close(F["kaiser_up"]!, AudioHelper<double>.ResampleKaiser(Engine, Tensor(F["wave16k"]!, 16000), 16000, 24000), "kaiser 16k to 24k");
        Close(F["kaiser_down"]!, AudioHelper<double>.ResampleKaiser(Engine, Tensor(F["wave24k"]!, 6000), 24000, 16000), "kaiser 24k to 16k");
        Close(F["hann_down"]!, AudioHelper<double>.ResampleBandLimited(Engine, Tensor(F["wave24k"]!, 6000), 24000, 16000), "hann 24k to 16k");
    }

    private static FireRedTtsSpeechTokenizer<double> Tokenizer(double deadCodeThreshold)
    {
        var configuration = new FireRedTtsTokenizerConfiguration(16, 8, new[] { 1, 1, 1, 2 }, 128, 4, 2, 80, new[] { new[] { 2, 1 }, new[] { 2, 1, 1, 1 } },
            640, 320, 0.9, deadCodeThreshold, 1.0, true);
        var hubert = new HubertConfiguration
        {
            ConvDims = Enumerable.Repeat(8, 7).ToArray(), HiddenSize = 16, Layers = 2, Heads = 2, IntermediateSize = 32,
            PositionalConvKernel = 16, PositionalConvGroups = 2,
        };
        var tokenizer = new FireRedTtsSpeechTokenizer<double>(Engine, new List<LayerBase<double>>(), new List<LayerBase<double>>(), configuration, hubert, new Random(7));
        var state = (JObject)F["state"]!;
        tokenizer.Load(Reader("sast."), name => state["sast." + name] is not null);
        tokenizer.Hubert.LoadTorchWeights(name => state["hubert." + name] is { } t ? Doubles(t) : null);
        return tokenizer;
    }

    [Fact]
    public void SpeechTokenizer_MatchesThe1SRelease()
    {
        using var _ = new NoGradScope<double>();
        var tokenizer = Tokenizer(1.0);
        var wave = Doubles(F["wave16k"]!).Take(9000).ToArray();
        var padded = tokenizer.PadToTokens(wave);
        var ssl = tokenizer.SslFeatures(padded);
        Close(F["ssl"]!, ssl, "SSL features");
        var encoded = tokenizer.Encode(ssl);
        Close(F["sast_encodings"]!, encoded, "encodings");
        var tokens = tokenizer.Tokenize(wave);
        Assert.Equal(F["tokens"]!.Select(v => (int)v).ToArray(), tokens);
        Close(F["sast_quantized"]!, tokenizer.Quantizer.ProjectOut.Forward(tokenizer.Quantizer.Lookup(tokens)), "quantized");
    }

    [Fact]
    public void Quantizer_EmaStep_MatchesThe1SRelease()
    {
        var tokenizer = Tokenizer(0.0);
        var (tokens, _, _, commitment) = tokenizer.Quantizer.Forward(Tensor(F["ema_encodings"]!, 6, 8), update: true, new Random(8));
        Assert.Equal(F["ema_tokens"]!.Select(v => (int)v).ToArray(), Enumerable.Range(0, tokens.GetLength(0)).Select(t => tokens[t, 0]).ToArray());
        Close(F["ema_commitment"]!, commitment, "commitment loss");
        Close(F["ema_codebook0"]!, tokenizer.Quantizer.Codebooks[0], "codebook 0 after the EMA step");
        Close(F["ema_codebook1"]!, tokenizer.Quantizer.Codebooks[1], "codebook 1 after the EMA step");
    }

    /// <summary>
    /// The released FireRedTTS-1 against the release's code in float64 (tools/reference-data/fireredtts_realweights_reference.py)
    /// where the checkpoint is present (<c>FIRERED_DIR</c>: fireredteam/FireRedTTS with fireredtts_gpt.pt,
    /// fireredtts_token2wav.pt and fireredtts_speaker.bin): the text's ids, the speaker embedding and prompt mel of a chirp,
    /// the GPT's logits and greedy tokens, the flow's mel from fixed noise and BigVGAN's audio.
    /// </summary>
    [Fact(Timeout = 3600000)]
    public async System.Threading.Tasks.Task ReleasedCheckpoint_MatchesTheRelease_WhenPresent()
    {
        await System.Threading.Tasks.Task.Yield();
        string? dir = Environment.GetEnvironmentVariable("FIRERED_DIR");
        if (dir is null || !File.Exists(Path.Combine(dir, "fireredtts_gpt.pt"))) return;
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", "fireredtts_realweights_reference.json");
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && d is not null; d = d.Parent)
            path = Path.Combine(d.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "fireredtts_realweights_reference.json");
        var f = JObject.Parse(File.ReadAllText(path));
        // The release's sizes; HuBERT and the streamable decoder's acoustic LM are not part of the release and are kept small.
        var options = new FireRedTTSOptions
        {
            HubertLayers = 1, AcousticLmDim = 16, AcousticLmHeads = 2, AcousticLmLayers = 1, AcousticLmFeedForward = 32,
            Sample = false, Candidates = 1, CandidateRank = 0, MaxNewTokens = 4,
        };
        var model = new FireRedTTS<double>(new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<double>(AiDotNet.Enums.InputType.OneDimensional,
            AiDotNet.Enums.NeuralNetworkTaskType.Generative, inputSize: 1, outputSize: 1), options);
        model.DisableAutoStreaming();
        model.LoadReferenceWeights(dir);
        using var _ = new NoGradScope<double>();
        var ids = model.TokenizeText("\u5C0F\u7EA2\u4E66\uFF0C\u662F\u4E2D\u56FD\u5927\u9646\u7684\u7F51\u7EDC\u8D2D\u7269\u548C\u793E\u4EA4\u5E73\u53F0\u3002");
        Assert.Equal(f["ids"]!.Select(v => (int)v).ToArray(), ids);
        var chirp = Doubles(f["chirp"]!);
        var speaker = model.SpeakerEmbedding(chirp);
        Close(f["speaker"]!, speaker, "speaker embedding", 1e-7);
        var promptMel = model.FlowMel(new Tensor<double>(new[] { chirp.Length }, new Vector<double>(chirp)), 16000);
        Close(f["prompt_mel"]!, promptMel, "prompt mel", 1e-7);
        var gpt = model.Gpt;
        var prefix = gpt.Prefix(speaker);
        var (_, audioLogits) = gpt.Forward(prefix, ids, null, new[] { gpt.Configuration.StartAudio });
        Close(f["logits"]!, audioLogits, "audio logits", 1e-7);
        var generated = gpt.Generate(prefix, ids, null, model.Decoding(), new Random(1));
        Assert.Equal(f["generated"]!.Select(v => (int)v).ToArray(), generated[0]);
        int[] flowTokens = { 1205, 9876, 233, 16000, 4411, 77, 12034, 8191, 5555, 300 };
        var noise = Engine.TensorTranspose(Tensor(f["noise"]!, 100, 4 * flowTokens.Length));
        var mel = model.Flow.Inference(flowTokens, promptMel, 3, noise);
        Close(f["flow_mel"]!, mel, "flow mel", 1e-7);
        var audio = model.Vocoder.Forward(Engine.Reshape(Engine.TensorTranspose(mel), new[] { 1, 100, mel.Shape[0] }));
        Close(f["audio"]!, audio, "audio", 1e-6);
    }
}

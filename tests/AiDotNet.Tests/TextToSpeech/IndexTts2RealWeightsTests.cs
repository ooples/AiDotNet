using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Latest;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// IndexTTS2 with the released weights against the release's own classes (tools/reference-data/indextts2_realweights_reference.py),
/// where the checkpoints are present: <c>INDEXTTS2_DIR</c> (IndexTeam/IndexTTS-2), <c>W2VBERT_DIR</c>
/// (facebook/w2v-bert-2.0), <c>MASKGCT_DIR</c> (amphion/MaskGCT, for its semantic codec), <c>CAMPPLUS_PATH</c>
/// (funasr/campplus's campplus_cn_common.bin) and <c>BIGVGAN_DIR</c> (nvidia/bigvgan_v2_22khz_80band_256x). Every stage
/// reads the release's inputs to it, so each is compared on its own: the prompt's features, codes, mel and style; the
/// speaker latents and emotion vector; the best-first beam codes; the GPT latents; the flow's mel from the release's noise;
/// BigVGAN's waveform; and the text-to-emotion model's prompt and answer. The model runs in float, so values compare
/// relatively and codes as a share. It needs about 24 GB of memory at its peak (the float model beside each checkpoint as it
/// loads); run it alone.
/// </summary>
public class IndexTts2RealWeightsTests
{
    private readonly ITestOutputHelper _output;

    public IndexTts2RealWeightsTests(ITestOutputHelper output) => _output = output;

    private static JsonElement Load(string name)
    {
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    private static float[] Floats(JsonElement root, string key)
    {
        var bytes = Convert.FromBase64String(root.GetProperty(key).GetString()!);
        var values = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    private static int[] Ints(JsonElement root, string key) => root.GetProperty(key).EnumerateArray().Select(v => v.GetInt32()).ToArray();

    private static int[] Shape(JsonElement root, string key) => root.GetProperty("shapes").GetProperty(key).EnumerateArray().Select(v => v.GetInt32()).ToArray();

    private static Tensor<float> Tensor(JsonElement root, string key, params int[] shape) => new(shape, new Vector<float>(Floats(root, key)));

    private double RelativeError(string what, float[] expected, Tensor<float> actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        double scale = expected.Max(v => Math.Abs((double)v)), worst = 0;
        for (int i = 0; i < expected.Length; i++) worst = Math.Max(worst, Math.Abs(expected[i] - actual.GetFlat(i)));
        double relative = worst / scale;
        _output.WriteLine($"{what}: max |difference| / max |value| = {relative:g3}");
        return relative;
    }

    [Fact(Timeout = 3600000)]
    public async Task ReleasedWeights_MatchTheRelease_WhenPresent()
    {
        await Task.Yield();
        string? dir = Environment.GetEnvironmentVariable("INDEXTTS2_DIR");
        string? w2v = Environment.GetEnvironmentVariable("W2VBERT_DIR");
        string? maskgct = Environment.GetEnvironmentVariable("MASKGCT_DIR");
        string? campplus = Environment.GetEnvironmentVariable("CAMPPLUS_PATH");
        string? bigvgan = Environment.GetEnvironmentVariable("BIGVGAN_DIR");
        if (dir is null || w2v is null || maskgct is null || campplus is null || bigvgan is null || !File.Exists(Path.Combine(dir, "gpt.pth")))
            return;
        var f = Load("indextts2_realweights_reference.json");
        var options = IndexTTS2Options.Release();
        var model = new IndexTTS2<float>(new NeuralNetworkArchitecture<float>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), options);
        model.LoadReferenceWeights(new IndexTts2Checkpoints
        {
            Directory = dir, SemanticModel = Path.Combine(w2v, "model.safetensors"),
            SemanticCodec = Path.Combine(maskgct, "semantic_codec", "model.safetensors"), SpeakerEncoder = campplus,
            Vocoder = Path.Combine(bigvgan, "bigvgan_generator.pt"),
        });
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        using var _ = new NoGradScope<float>();

        // The prompt: torchaudio-resampled features, their codes, the reference mel and the CAM++ style.
        var prompt = Floats(f, "prompt").Select(v => (double)v).ToArray();
        int rate = f.GetProperty("prompt_rate").GetInt32();
        var audio16k = AudioHelper<float>.ResampleBandLimited(engine, new Tensor<float>(new[] { prompt.Length }, new Vector<float>(prompt.Select(v => (float)v).ToArray())), rate, 16000);
        var audio22k = AudioHelper<float>.ResampleBandLimited(engine, new Tensor<float>(new[] { prompt.Length }, new Vector<float>(prompt.Select(v => (float)v).ToArray())), rate, 22050);
        var features = model.SemanticFeatures(audio16k.ToArray().Select(v => (double)v).ToArray());
        Assert.True(RelativeError("features", Floats(f, "features"), features) < 2e-3);
        var referenceFeatures = Tensor(f, "features", Shape(f, "features")[0], Shape(f, "features")[1]);
        var promptCodes = model.SemanticCodec.EncodeNormalized(referenceFeatures);
        var expectedCodes = Ints(f, "prompt_codes");
        double codeShare = promptCodes.Zip(expectedCodes, (a, b) => a == b ? 1.0 : 0.0).Average();
        _output.WriteLine($"prompt codes: {codeShare:P1} equal");
        Assert.True(codeShare >= 0.95);
        Assert.True(RelativeError("reference mel", Floats(f, "ref_mel"), model.MelSpectrogram(audio22k)) < 1e-3);
        Assert.True(RelativeError("style", Floats(f, "style"), model.Style(audio16k)) < 2e-3);

        // The conditioners on the release's features.
        var gpt = model.TextToSemantic;
        var speaker = gpt.SpeakerConditioning(referenceFeatures);
        Assert.True(RelativeError("speaker latents", Floats(f, "speaker"), speaker) < 2e-3);
        var emotionFeatures = Tensor(f, "emotion_features", Shape(f, "emotion_features")[0], Shape(f, "emotion_features")[1]);
        var baseVector = gpt.EmotionVector(referenceFeatures);
        var emotion = engine.TensorAdd(baseVector, engine.TensorSubtract(gpt.EmotionVector(emotionFeatures), baseVector));
        Assert.True(RelativeError("emotion vector", Floats(f, "emotion"), emotion) < 2e-3);

        // Generation, then the GPT latents of the release's codes.
        var text = Ints(f, "text");
        var codes = Ints(f, "codes");
        var referenceSpeaker = Tensor(f, "speaker", Shape(f, "speaker")[0], Shape(f, "speaker")[1]);
        var referenceEmotion = Tensor(f, "emotion", 1, Shape(f, "emotion")[0]);
        var conditions = gpt.Conditions(referenceSpeaker, referenceEmotion, null);
        var generated = gpt.Generate(conditions, text, new IndexTtsDecoding(false, 3, 30, 0.8, 0.8, 10.0, 200), new Random(1));
        int prefix = 0;
        while (prefix < Math.Min(generated.Length, codes.Length) && generated[prefix] == codes[prefix]) prefix++;
        _output.WriteLine($"generated {generated.Length} codes, the release {codes.Length}; the first {prefix} agree");
        Assert.True(prefix >= Math.Min(20, codes.Length));
        var latents = engine.TensorNarrow(gpt.Latents(conditions, text, codes), 0, 0, codes.Length);
        Assert.True(RelativeError("GPT latents", Floats(f, "latents"), latents) < 2e-3);

        // The flow from the release's noise, and the vocoder on the release's mel.
        var s2mel = model.SemanticToMel;
        var refMel = Tensor(f, "ref_mel", Shape(f, "ref_mel")[0], Shape(f, "ref_mel")[1]);
        var promptQuantized = model.SemanticCodec.Embed(expectedCodes);
        var promptCondition = s2mel.Regulator.Forward(promptQuantized, refMel.Shape[0]);
        var semantic = s2mel.Fuse(model.SemanticCodec.Embed(codes), Tensor(f, "latents", codes.Length, options.HiddenDim));
        var condition = engine.TensorConcatenate(new[] { promptCondition, s2mel.Regulator.Forward(semantic, (int)(codes.Length * 1.72)) }, 0);
        var noise = Tensor(f, "noise", Shape(f, "noise")[0], Shape(f, "noise")[1]);
        var mel = s2mel.Solve(condition, refMel, Tensor(f, "style", options.StyleDim), noise, 25, 0.7);
        Assert.True(RelativeError("mel", Floats(f, "mel"), mel) < 5e-3);
        var referenceMel = Tensor(f, "mel", Shape(f, "mel")[0], Shape(f, "mel")[1]);
        var wave = model.Vocoder.Forward(engine.Reshape(engine.TensorTranspose(referenceMel), new[] { 1, 80, referenceMel.Shape[0] }));
        Assert.True(RelativeError("waveform", Floats(f, "waveform"), wave) < 5e-3);
    }

    [Fact(Timeout = 3600000)]
    public async Task ReleasedTextToEmotion_AnswersAsTheRelease_WhenPresent()
    {
        await Task.Yield();
        string? dir = Environment.GetEnvironmentVariable("INDEXTTS2_DIR");
        if (dir is null || !File.Exists(Path.Combine(dir, "qwen0.6bemo4-merge", "model.safetensors"))) return;
        var f = Load("indextts2_realweights_reference.json");
        string sentence = f.GetProperty("emotion_text").GetString()!;
        Assert.Equal(f.GetProperty("emotion_prompt").GetString(), IndexTtsTextToEmotion<float>.Prompt(sentence));
        var options = IndexTTS2Options.Release();
        string path = Path.Combine(dir, "qwen0.6bemo4-merge");
        var t2e = new IndexTtsTextToEmotion<float>(AiDotNet.Tensors.Engines.AiDotNetEngine.Current, new System.Collections.Generic.List<AiDotNet.NeuralNetworks.Layers.LayerBase<float>>(),
            IndexTTS2<float>.EmotionConfiguration(options), OuteTTSTokenizers.FromFile(Path.Combine(path, "tokenizer.json")), options.EmotionLabels, new Random(1));
        IndexTTS2<float>.LoadTextToEmotionState(t2e, path);
        using var _ = new NoGradScope<float>();
        string answer = t2e.Answer(sentence, 256);
        _output.WriteLine(answer);
        Assert.Equal(f.GetProperty("emotion_answer").GetString(), answer);
        var scores = t2e.Parse(answer, sentence);
        _output.WriteLine(string.Join(", ", options.EmotionLabels.Zip(scores, (l, v) => $"{l} {v}")));
        // The release's answer: {"愤怒":0.00,"高兴":0.95,...,"惊讶":0.03,"自然":0.02}.
        Assert.Equal(0.95, scores[0], 6);
        Assert.Equal(0.03, scores[6], 6);
        Assert.Equal(0.02, scores[7], 6);
    }
}

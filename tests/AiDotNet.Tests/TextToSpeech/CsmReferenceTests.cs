using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// CSM against SesameAILabs/csm's models.Model (torchtune) at a tiny size and Hugging Face's CsmForConditionalGeneration on
/// the same weights (tools/reference-data/csm_reference.py): every release tensor has a place, the backbone's hidden states
/// and codebook-0 logits over a conversation match, greedy generation samples the same frames from the same logits, and the
/// training loss — the backbone's codebook-0 loss plus the decoder's — matches Hugging Face's.
/// </summary>
public class CsmReferenceTests
{
    private static JsonElement Load(string name)
    {
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    private static double[] ClosedForm(string name, int[] shape)
    {
        int count = shape.Aggregate(1, (a, b) => a * b);
        var values = new double[count];
        for (int i = 0; i < count; i++)
        {
            long k = i;
            double s = Math.Sin(0.37 * i + 0.11 * name.Length + 0.0013 * (k * k % 9973));
            values[i] = name.EndsWith(".scale", StringComparison.Ordinal) ? 1 + 0.2 * s
                : name.EndsWith("embeddings.weight", StringComparison.Ordinal) || name.EndsWith("audio_head", StringComparison.Ordinal) ? 0.5 * s
                : 0.5 * s / Math.Sqrt(count / (double)shape[0]);
        }
        return values;
    }

    private static double[] Doubles(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();
    private static int[] Ints(JsonElement e) => e.EnumerateArray().Select(v => v.GetInt32()).ToArray();

    private static void Close(double[] expected, double[] actual, double tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        double scale = Math.Max(1e-300, expected.Max(Math.Abs)), worst = 0;
        for (int i = 0; i < expected.Length; i++) worst = Math.Max(worst, Math.Abs(expected[i] - actual[i]));
        Assert.True(worst <= tolerance * scale, $"{what} differs by {worst} (scale {scale})");
    }

    internal static CSMOptions TinyOptions(JsonElement reference, int generatedFrames) => new()
    {
        VocabSize = reference.GetProperty("text_vocab").GetInt32(), HiddenDim = reference.GetProperty("width").GetInt32(),
        BackboneLayers = 1, BackboneHeads = 2, BackboneKeyValueHeads = 1, BackboneIntermediate = 32,
        DecoderHidden = reference.GetProperty("width").GetInt32(), DecoderLayers = 1, DecoderHeads = 2, DecoderKeyValueHeads = 1, DecoderIntermediate = 32,
        AudioCodebooks = reference.GetProperty("codebooks").GetInt32(), AudioVocabSize = reference.GetProperty("audio_vocab").GetInt32(),
        MimiCodebookSize = 16, MimiCodebookDim = 8, MimiDimension = 16, MimiFilters = 2, MimiRatios = new[] { 4, 4, 4, 5 },
        MimiTransformerLayers = 1, MimiHeads = 2, MimiFeedForward = 32,
        // 640 samples a frame at 24 kHz: half a frame past the count keeps the floor at it.
        MaxAudioLengthMs = (generatedFrames + 0.5) * 1000.0 * 640 / 24000, TopK = 1, Watermark = false,
    };

    [Fact(Timeout = 600000)]
    public async Task ForwardGenerationAndLoss_MatchTheReleaseAndHuggingFace()
    {
        await Task.Yield();
        var reference = Load("csm_reference.json");
        var generated = reference.GetProperty("generated").EnumerateArray().Select(Ints).ToArray();
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
        var model = new CSM<double>(architecture, TinyOptions(reference, generated.Length));
        var shapes = reference.GetProperty("shapes").EnumerateObject().ToDictionary(p => p.Name, p => Ints(p.Value), StringComparer.Ordinal);
        var used = model.LoadTorchWeights(name => shapes.TryGetValue(name, out var shape) ? ClosedForm(name, shape) : null);
        Assert.Empty(shapes.Keys.Where(n => !used.Contains(n)));

        int books = reference.GetProperty("codebooks").GetInt32();
        var audio = reference.GetProperty("audio").EnumerateArray().Select(Ints).ToArray();
        var firstTurn = Ints(reference.GetProperty("first_text")).Select(id => new CsmFrame(id, null)).ToList();
        firstTurn.AddRange(audio.Select(codes => new CsmFrame(0, codes)));
        firstTurn.Add(new CsmFrame(0, new int[books]));
        var conversation = firstTurn.Concat(Ints(reference.GetProperty("second_text")).Select(id => new CsmFrame(id, null))).ToList();

        var hidden = model.Backbone.Forward(model.EmbedFrames(conversation)).Hidden;
        Close(Doubles(reference.GetProperty("hidden")), hidden.ToArray(), 1e-9, "backbone hidden states");
        Close(Doubles(reference.GetProperty("logits0")), model.Codebook0Logits(hidden).ToArray(), 1e-9, "codebook-0 logits");

        var observed = new List<double[]>();
        model.SampledLogits = observed.Add;
        var frames = model.GenerateFrames(conversation);
        model.SampledLogits = null;
        Assert.Equal(generated.Length, frames.Count);
        for (int f = 0; f < generated.Length; f++) Assert.Equal(generated[f], frames[f]);
        var expectedLogits = reference.GetProperty("sampled_logits").EnumerateArray().Select(Doubles).ToArray();
        Assert.Equal(expectedLogits.Length, observed.Count);
        for (int i = 0; i < expectedLogits.Length; i++) Close(expectedLogits[i], observed[i], 1e-9, $"sampled logits {i}");

        // Hugging Face upcasts the logits to float32 for its cross-entropy, so its losses carry single precision.
        int firstAudio = Ints(reference.GetProperty("first_text")).Length;
        var labels = firstTurn.Select((frame, p) => p >= firstAudio && p < firstAudio + audio.Length ? frame.Codes![0] : -100).ToArray();
        var row = new CsmTrainingRow(firstTurn, labels, Ints(reference.GetProperty("decoder_frames")));
        double loss = model.Loss(row).ToArray()[0];
        double expected = reference.GetProperty("loss").GetDouble();
        Assert.True(Math.Abs(loss - expected) <= 1e-6 * expected, $"loss {loss} vs {expected}");
    }

    // The fixture's conversion to transformers' names (its to_hf): torchtune modules renamed, query and key rows reordered
    // from interleaved pairs to halves, and the decoder's code table tied to the backbone's.
    private static Dictionary<string, double[]> TransformersState(Dictionary<string, int[]> shapes, int headDim)
    {
        var state = new Dictionary<string, double[]>(StringComparer.Ordinal);
        foreach (var (name, shape) in shapes)
        {
            var values = ClosedForm(name, shape);
            string port;
            if (name.StartsWith("backbone.", StringComparison.Ordinal) || name.StartsWith("decoder.", StringComparison.Ordinal))
            {
                bool backbone = name.StartsWith("backbone.", StringComparison.Ordinal);
                string rest = name.Substring(backbone ? "backbone.".Length : "decoder.".Length)
                    .Replace("sa_norm.scale", "input_layernorm.weight").Replace("mlp_norm.scale", "post_attention_layernorm.weight")
                    .Replace("attn.output_proj", "attn.o_proj").Replace("attn.", "self_attn.")
                    .Replace("mlp.w1", "mlp.gate_proj").Replace("mlp.w3", "mlp.up_proj").Replace("mlp.w2", "mlp.down_proj").Replace("norm.scale", "norm.weight");
                if (rest.EndsWith("q_proj.weight", StringComparison.Ordinal) || rest.EndsWith("k_proj.weight", StringComparison.Ordinal))
                {
                    int rows = shape[0], inputs = shape[1], half = headDim / 2;
                    var halves = new double[values.Length];
                    for (int head = 0; head < rows / headDim; head++)
                        for (int j = 0; j < headDim; j++)
                            Array.Copy(values, (head * headDim + (j < half ? 2 * j : 2 * (j - half) + 1)) * inputs, halves, (head * headDim + j) * inputs, inputs);
                    values = halves;
                }
                port = (backbone ? "backbone_model." : "depth_decoder.model.") + rest;
            }
            else
            {
                port = name switch
                {
                    "text_embeddings.weight" => "embed_text_tokens.weight",
                    "audio_embeddings.weight" => "backbone_model.embed_tokens.embed_audio_tokens.weight",
                    "projection.weight" => "depth_decoder.model.inputs_embeds_projector.weight",
                    "codebook0_head.weight" => "lm_head.weight",
                    "audio_head" => "depth_decoder.codebooks_head.weight",
                    _ => throw new InvalidOperationException(name),
                };
            }
            state[port] = values;
        }
        state["depth_decoder.model.embed_tokens.weight"] = state["backbone_model.embed_tokens.embed_audio_tokens.weight"];
        return state;
    }

    [Fact(Timeout = 600000)]
    public async Task TransformersNames_LoadTheSameModel()
    {
        await Task.Yield();
        var reference = Load("csm_reference.json");
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
        var model = new CSM<double>(architecture, TinyOptions(reference, 1));
        var shapes = reference.GetProperty("shapes").EnumerateObject().ToDictionary(p => p.Name, p => Ints(p.Value), StringComparer.Ordinal);
        var state = TransformersState(shapes, reference.GetProperty("width").GetInt32() / 2);
        var used = model.LoadTorchWeights(name => state.TryGetValue(name, out var v) ? v : null);
        Assert.Empty(state.Keys.Where(n => !used.Contains(n)));

        int books = reference.GetProperty("codebooks").GetInt32();
        var frames = Ints(reference.GetProperty("first_text")).Select(id => new CsmFrame(id, null)).ToList();
        frames.AddRange(reference.GetProperty("audio").EnumerateArray().Select(codes => new CsmFrame(0, Ints(codes))));
        frames.Add(new CsmFrame(0, new int[books]));
        frames.AddRange(Ints(reference.GetProperty("second_text")).Select(id => new CsmFrame(id, null)));
        var hidden = model.Backbone.Forward(model.EmbedFrames(frames)).Hidden;
        Close(Doubles(reference.GetProperty("hidden")), hidden.ToArray(), 1e-9, "backbone hidden states");
        Close(Doubles(reference.GetProperty("logits0")), model.Codebook0Logits(hidden).ToArray(), 1e-9, "codebook-0 logits");
    }

    /// <summary>With SilentCipher's released weights (<c>SILENTCIPHER_DIR</c>), CSM's output is Mimi's audio watermarked as
    /// the release watermarks it, and the release's key reads back from it.</summary>
    [Fact(Timeout = 3600000)]
    public async Task Synthesis_IsWatermarkedWithTheReleasesKey_WhenTheWatermarkModelIsPresent()
    {
        await Task.Yield();
        string? dir = Environment.GetEnvironmentVariable("SILENTCIPHER_DIR");
        if (dir is null || !File.Exists(Path.Combine(dir, "enc_c.ckpt"))) return;
        var reference = Load("csm_reference.json");
        var options = TinyOptions(reference, 120);
        options.TopK = 50;
        options.ConstrainToCodebook = true;
        options.Watermark = true;
        options.WatermarkModelPath = dir;
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
        var model = new CSM<double>(architecture, options);
        var frames = model.GenerateFrames(CSM<double>.TextFrames("Watermarked speech.", 0));
        Assert.True(frames.Count >= 60, $"only {frames.Count} frames were generated");
        var codes = new int[frames.Count, options.AudioCodebooks];
        for (int f = 0; f < frames.Count; f++)
            for (int k = 0; k < options.AudioCodebooks; k++) codes[f, k] = frames[f][k];
        var plain = model.Codec.Decode(codes);
        var expected = AiDotNet.Safety.Watermarking.SilentCipherWatermark.Apply(new AiDotNet.Tensors.Engines.CpuEngine(), model.Watermarker, plain, 24000,
            options.WatermarkKey, options.WatermarkSdr);
        var audio = model.Generate("Watermarked speech.", 0, Array.Empty<CsmSegment<double>>()).ToArray();
        Close(expected, audio, 1e-12, "watermarked audio");
        Assert.True(AiDotNet.Safety.Watermarking.SilentCipherWatermark.VerifyWithoutPhaseSearch(new AiDotNet.Tensors.Engines.CpuEngine(), model.Watermarker, audio, 24000, options.WatermarkKey));
        Assert.False(AiDotNet.Safety.Watermarking.SilentCipherWatermark.VerifyWithoutPhaseSearch(new AiDotNet.Tensors.Engines.CpuEngine(), model.Watermarker, plain, 24000, options.WatermarkKey));
    }

    /// <summary>With the release's checkpoint (<c>CSM_DIR</c>: sesame/csm-1b's ckpt.pt, or a directory holding it), prompted
    /// as run_csm.py prompts it (tools/reference-data/csm_realweights_reference.py): the turns tokenize to the release's ids,
    /// the backbone's last hidden state and codebook-0 logits match, and greedy generation samples the same frames from the
    /// same logits.</summary>
    [Fact(Timeout = 3600000)]
    public async Task ReleasedCheckpoint_MatchesTheRelease_WhenPresent()
    {
        await Task.Yield();
        string? path = Environment.GetEnvironmentVariable("CSM_DIR");
        if (path is null || !(File.Exists(path) || File.Exists(Path.Combine(path, "ckpt.pt")) || File.Exists(Path.Combine(path, "model.safetensors")))) return;
        var reference = Load("csm_realweights_reference.json");
        var generated = reference.GetProperty("generated").EnumerateArray().Select(Ints).ToArray();
        // 80 ms a frame: half a frame past the count keeps the floor at it.
        var options = new CSMOptions { TopK = 1, Watermark = false, MaxAudioLengthMs = (generated.Length + 0.5) * 80 };
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
        var model = new CSM<double>(architecture, options);
        // Above 500M parameters the model would stream its weights, paging and compressing them on every step; the test
        // machine holds the 12 GB of double weights resident.
        model.DisableAutoStreaming();
        GC.Collect();
        model.LoadReferenceWeights(path);
        // The 1B model in double is 12 GB; reclaim the checkpoint and the conversions before running it.
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        const string Context = "like revising for an exam I'd have to try and like keep up the momentum because I'd "
            + "start really early I'd be like okay I'm gonna start revising now and then like "
            + "you're revising for ages and then I just like start losing steam I didn't do that "
            + "for the exam we had recently to be fair that was a more of a last minute scenario "
            + "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
            + "sort of start the day with this not like a panic but like a";
        var contextText = CSM<double>.TextFrames(Context, 0);
        var targetText = CSM<double>.TextFrames("Hey how are you doing?", 0);
        Assert.Equal(Ints(reference.GetProperty("context_text")), contextText.Select(f => f.Text).ToArray());
        Assert.Equal(Ints(reference.GetProperty("target_text")), targetText.Select(f => f.Text).ToArray());

        var codeRows = reference.GetProperty("codes").EnumerateArray().Select(Ints).ToArray();
        var codes = new int[codeRows.Length, options.AudioCodebooks];
        for (int f = 0; f < codeRows.Length; f++)
            for (int k = 0; k < options.AudioCodebooks; k++) codes[f, k] = codeRows[f][k];
        var prompt = contextText.Concat(CSM<double>.AudioFrames(codes)).Concat(targetText).ToList();

        var hidden = model.Backbone.Forward(model.EmbedFrames(prompt)).Hidden;
        var logits = model.Codebook0Logits(hidden).ToArray();
        int width = options.HiddenDim, vocab = options.AudioVocabSize;
        Close(Doubles(reference.GetProperty("hidden")), hidden.ToArray().Skip(hidden.Length - width).ToArray(), 1e-9, "last hidden state");
        Close(Doubles(reference.GetProperty("logits0")), logits.Skip(logits.Length - vocab).ToArray(), 1e-9, "codebook-0 logits");

        var observed = new List<double[]>();
        var frames = model.GenerateFrames(prompt);
        model.SampledLogits = null;
        Assert.Equal(generated.Length, frames.Count);
        for (int f = 0; f < generated.Length; f++) Assert.Equal(generated[f], frames[f]);
        var expectedLogits = reference.GetProperty("sampled_logits").EnumerateArray().Select(Doubles).ToArray();
        Assert.Equal(expectedLogits.Length, observed.Count);
        for (int i = 0; i < expectedLogits.Length; i++) Close(expectedLogits[i], observed[i], 1e-9, $"sampled logits {i}");
        Assert.False(model.IsWeightStreamingActive);
    }
}

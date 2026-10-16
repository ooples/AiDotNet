using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.Audio.Fingerprinting;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// CLAP reproduces LAION-CLAP (<c>tools/reference-data/clap_reference.py</c>): with the reference's weights loaded
/// through <see cref="CLAPModel{T}.LoadLaionWeights"/>, a tiny HTSAT + RoBERTa configuration gives the reference's
/// log-mel inputs (both filterbanks), its feature-fusion views, its audio and text embeddings and its logits.
/// </summary>
/// <remarks>
/// The old network (one patch embedding, single-width Swin blocks, a sinusoidal-position transformer, single dense
/// projections, 0 Hz to Nyquist mel bands) cannot load this state dict at all: it has no bn0, no patch merging, no
/// RoBERTa pooler and no fusion module for the names to land in.
/// </remarks>
public class ClapReferenceTests
{
    private const double Tolerance = 1e-8;

    private static JsonElement Fixture()
    {
        const string name = "clap_reference.json";
        string relative = Path.Combine("Audio", "ReferenceData", name);
        string path = Path.Combine(AppContext.BaseDirectory, relative);
        if (!File.Exists(path))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
            path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        }
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    internal static CLAPModelOptions TinyOptions(bool fusion) => new()
    {
        SampleRate = 48000, StftWindowSize = 512, HopLength = 64, NumMelBands = 16, MelMinFrequency = 50,
        MelMaxFrequency = 14000, ClipSamples = 8192, EnableFeatureFusion = fusion,
        SpecSize = 64, AudioPatchSize = 4, AudioEmbedDim = 8, AudioStageDepths = new[] { 2, 2, 2 },
        AudioStageHeads = new[] { 1, 2, 4 }, SwinWindowSize = 8, AudioMlpRatio = 4, AudioDropPathRate = 0.1,
        VocabSize = 64, MaxTextLength = 10, TextMaxPositions = 32, TextHiddenDim = 16, TextEncoderLayers = 2,
        TextEncoderHeads = 2, TextIntermediateDim = 32, TextPaddingTokenId = 1, TextDropoutRate = 0.1,
        TextLayerNormEpsilon = 1e-5, ProjectionDim = 12, Seed = 11,
        // HTS-AT's SpecAugment widths (64 of 1001 frames, 8 of 64 bins) scaled to this 129 x 16 spectrogram.
        SpecAugmentTimeDropWidth = 8, SpecAugmentFreqDropWidth = 2,
    };

    internal static CLAPModel<double> Build(bool fusion) =>
        new(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Embedding,
            inputSize: 8192, outputSize: 12), TinyOptions(fusion));

    private static CLAPModel<double> LoadReference(JsonElement f, bool fusion, out IReadOnlyCollection<string> read)
    {
        var model = Build(fusion);
        var state = f.GetProperty("state_dict");
        var shapes = f.GetProperty("shapes");
        read = model.LoadLaionWeights((name, shape) =>
        {
            Assert.True(state.TryGetProperty(name, out var values), $"The reference has no '{name}'.");
            Assert.Equal(shapes.GetProperty(name).EnumerateArray().Select(e => e.GetInt32()).ToArray(), shape);
            return values.EnumerateArray().Select(e => e.GetDouble()).ToArray();
        });
        model.SetTrainingMode(false);
        return model;
    }

    private static double[] Vector(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();

    private static double[][] Matrix(JsonElement e) => e.EnumerateArray().Select(Vector).ToArray();

    private static Tensor<double> Row(double[] values) => new(new[] { 1, values.Length }, new Vector<double>(values));

    private static void Close(double[] expected, Tensor<double> actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual.GetFlat(i)) <= Tolerance,
                $"{what}[{i}]: {actual.GetFlat(i):R} vs the reference's {expected[i]:R}");
    }

    private static double[] Flatten(double[][] rows) => rows.SelectMany(r => r).ToArray();

    // get_audio_features' "repeatpad": int(clip / len) whole copies, then zeros.
    private static double[] RepeatPad(double[] audio, int clip)
    {
        var result = new double[clip];
        int copies = clip / audio.Length;
        for (int i = 0; i < copies * audio.Length; i++) result[i] = audio[i % audio.Length];
        return result;
    }

    [Fact(Timeout = 120000)]
    public async Task LogMel_MatchesBothReferenceFrontEnds()
    {
        await Task.Yield();
        var f = Fixture();
        var fitted = Row(RepeatPad(Vector(f.GetProperty("short_waveform")), 8192));
        // Non-fusion models read torchlibrosa's mel (librosa's Slaney filterbank, 50 Hz to 14 kHz).
        using var plain = Build(fusion: false);
        Close(Flatten(Matrix(f.GetProperty("plain_mel"))), plain.ComputeLogMel(fitted), "slaney mel");
        // Fusion models read torchaudio's (HTK scale, unnormalized).
        using var fused = Build(fusion: true);
        Close(Flatten(Matrix(f.GetProperty("fusion_short_mel"))), fused.ComputeLogMel(fitted), "htk mel");
        Close(Flatten(Matrix(f.GetProperty("long_mel"))), fused.ComputeLogMel(Row(Vector(f.GetProperty("long_waveform")))), "long htk mel");
    }

    [Fact(Timeout = 120000)]
    public async Task FusionViews_ShrinkAndCropTheLongClipAsTheReferenceDoes()
    {
        await Task.Yield();
        var f = Fixture();
        var mel = Matrix(f.GetProperty("long_mel"));
        var views = f.GetProperty("long_views").EnumerateArray().Select(Matrix).ToArray();
        int total = mel.Length, chunk = views[0].Length, mels = mel[0].Length;
        Assert.Equal(8192 / 64 + 1, chunk);

        // View 0: torchvision Resize (antialiased bilinear) of the whole mel to the clip's frame count.
        var melTensor = new Tensor<double>(new[] { total, mels });
        for (int s = 0; s < total; s++)
            for (int m = 0; m < mels; m++) melTensor[s, m] = mel[s][m];
        var shrunk = ClapAudioFrontEnd<double>.AntialiasedBilinearShrink(melTensor, chunk);
        for (int t = 0; t < chunk; t++)
            for (int m = 0; m < mels; m++)
            {
                double value = shrunk[t, m];
                Assert.True(Math.Abs(views[0][t][m] - value) <= Tolerance, $"shrunk view [{t}, {m}]: {value:R} vs {views[0][t][m]:R}");
            }

        // Views 1-3: crops starting at one offset from each third of the possible starts.
        var offsets = f.GetProperty("long_crop_offsets").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        var thirds = ClapAudioFrontEnd<double>.SplitOffsets(total - chunk + 1);
        for (int part = 0; part < 3; part++)
        {
            Assert.Contains(offsets[part], thirds[part]);
            for (int t = 0; t < chunk; t++)
                for (int m = 0; m < mels; m++) Assert.Equal(mel[offsets[part] + t][m], views[part + 1][t][m]);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task AudioEmbeddings_MatchTheReference()
    {
        await Task.Yield();
        var f = Fixture();
        var shortAudio = Row(Vector(f.GetProperty("short_waveform")));

        using var plain = LoadReference(f, fusion: false, out var plainRead);
        Assert.DoesNotContain(plainRead, n => n.Contains("fusion_model") || n.Contains("mel_conv2d"));
        Close(Vector(f.GetProperty("plain_audio_embedding")), plain.EncodeAudio(shortAudio), "non-fusion embedding");

        using var fused = LoadReference(f, fusion: true, out var fusedRead);
        Assert.Contains(fusedRead, n => n.StartsWith("audio_branch.patch_embed.fusion_model.global_att", StringComparison.Ordinal));
        // A clip shorter than 10 s: repeat-padded, four identical views, not "longer".
        Close(Vector(f.GetProperty("fusion_short_audio_embedding")), fused.EncodeAudio(shortAudio), "fusion embedding (short)");

        // A long clip through the fusion branch, from the reference's own four views.
        var views = f.GetProperty("long_views").EnumerateArray().Select(Matrix).ToArray();
        int frames = views[0].Length, mels = views[0][0].Length;
        var input = new Tensor<double>(new[] { 1, 4, frames, mels });
        for (int v = 0; v < 4; v++)
            for (int t = 0; t < frames; t++)
                for (int m = 0; m < mels; m++) input[0, v, t, m] = views[v][t][m];
        var longer = new Tensor<double>(new[] { 1 });
        longer[0] = 1.0;
        var fusedLong = fused.EncodeLogMel(input, longer);
        Close(Vector(f.GetProperty("fusion_long_audio_embedding")), fusedLong, "fusion embedding (long)");
        // The fusion branch is live: without the mask the same views give a different embedding.
        var unfused = fused.EncodeLogMel(input);
        Assert.True(Enumerable.Range(0, 12).Max(i => Math.Abs(unfused.GetFlat(i) - fusedLong.GetFlat(i))) > 1e-4);
    }

    [Fact(Timeout = 120000)]
    public async Task TextEmbeddings_AndLogits_MatchTheReference()
    {
        await Task.Yield();
        var f = Fixture();
        using var model = LoadReference(f, fusion: true, out var read);
        Assert.Contains("text_branch.pooler.dense.weight", read);
        var captions = f.GetProperty("captions").EnumerateArray().Select(r => r.EnumerateArray().Select(e => (double)e.GetInt32()).ToArray()).ToArray();
        var ids = new Tensor<double>(new[] { captions.Length, captions[0].Length });
        for (int c = 0; c < captions.Length; c++)
            for (int i = 0; i < captions[c].Length; i++) ids[c, i] = captions[c][i];
        var text = model.EncodeText(ids);
        Close(Flatten(Matrix(f.GetProperty("text_embeddings"))), text, "text embedding");

        var audio = model.EncodeAudio(Row(Vector(f.GetProperty("short_waveform"))));
        var expected = Matrix(f.GetProperty("logits"))[0];
        for (int c = 0; c < captions.Length; c++)
        {
            double dot = 0;
            for (int d = 0; d < 12; d++) dot += audio[0, d] * text[c, d];
            Assert.True(Math.Abs(expected[c] - model.LogitScale * dot) <= Tolerance, $"logit {c}: {model.LogitScale * dot:R} vs {expected[c]:R}");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task PaddingPositions_DoNotChangeTheTextEmbedding()
    {
        await Task.Yield();
        var f = Fixture();
        using var model = LoadReference(f, fusion: true, out _);
        var shortIds = new Tensor<double>(new[] { 1, 4 }, new Vector<double>(new double[] { 0, 33, 41, 2 }));
        var padded = new Tensor<double>(new[] { 1, 10 }, new Vector<double>(new double[] { 0, 33, 41, 2, 1, 1, 1, 1, 1, 1 }));
        var a = model.EncodeText(shortIds);
        var b = model.EncodeText(padded);
        for (int d = 0; d < 12; d++) Assert.True(Math.Abs(a.GetFlat(d) - b.GetFlat(d)) <= Tolerance, $"dim {d}");
    }
}

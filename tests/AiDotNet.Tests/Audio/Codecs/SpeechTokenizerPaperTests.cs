using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// SpeechTokenizer encodes, quantizes, decodes and trains as its paper specifies (Zhang et al., ICLR 2024), and loads the
/// released checkpoint's layout exactly.
/// </summary>
/// <remarks>
/// <c>ReferenceData/speechtokenizer_official_layout_reference.json</c> holds a tiny model built from the reference
/// implementation (ZhangXInFD/SpeechTokenizer) with random weights and codebooks, saved with <c>torch.save</c> like the
/// released <c>SpeechTokenizer.pt</c> (generator: tts-paper-specs/drafts/codecs/speechtokenizer_parity_ref.py), with the
/// latent, codes and decoded audio of a fixed input.
/// </remarks>
public class SpeechTokenizerPaperTests
{
    private static JObject Reference()
    {
        const string fileName = "speechtokenizer_official_layout_reference.json";
        string output = Path.Combine(AppContext.BaseDirectory, "Audio", "Codecs", "ReferenceData", fileName);
        if (!File.Exists(output))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(output))
            {
                output = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "Audio", "Codecs", "ReferenceData", fileName);
                dir = dir.Parent;
            }
        }
        return JObject.Parse(File.ReadAllText(output));
    }

    private static NeuralNetworkArchitecture<double> Architecture()
        => new(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1) { RandomSeed = 19 };

    private static SpeechTokenizerOptions TinyOptions() => new()
    {
        Filters = 4,
        Ratios = [4, 2],
        Dimension = 8,
        SemanticDimension = 6,
        NumQuantizers = 2,
        CodebookSize = 16,
        TargetBandwidthKbps = 16.0,
        SegmentSize = 96,
        KMeansIterations = 5,
        MelScales = [5, 6],
        MelBins = 8,
        StftDiscriminatorWindows = [32, 16],
        StftDiscriminatorFilters = 4,
        DiscriminatorWidthDivisor = 64,
        SamplingSeed = 3,
    };

    private static Tensor<double> Teacher(int frames, int dim)
    {
        var s = new Tensor<double>(new[] { frames, dim });
        for (int t = 0; t < frames; t++)
            for (int d = 0; d < dim; d++) s[t, d] = Math.Sin(0.7 * t * (d + 1) + d);
        return s;
    }

    [Fact(Timeout = 300000)]
    public async Task OfficialLayout_LoadsAndReproducesTheReferenceModel()
    {
        await Task.Yield();
        var fixture = Reference();
        using var model = new SpeechTokenizer<double>(Architecture(), TinyOptions());
        string path = Path.Combine(Path.GetTempPath(), $"speechtokenizer_reference_{Guid.NewGuid():N}.pt");
        File.WriteAllBytes(path, Convert.FromBase64String((string)fixture["checkpoint_base64"]!));
        try
        {
            model.LoadPretrainedWeights(path);
        }
        finally
        {
            File.Delete(path);
        }

        var samples = fixture["audio"]!.Select(v => (double)v).ToArray();
        var audio = new Tensor<double>(new[] { 1, 1, samples.Length }, new Vector<double>(samples));
        var latent = model.EncodeEmbeddings(audio);
        var expectedLatent = fixture["latent"]!.Select(v => (double)v).ToArray();
        Assert.Equal(expectedLatent.Length, latent.Length);
        for (int i = 0; i < latent.Length; i++)
            Assert.True(Math.Abs(latent[i] - expectedLatent[i]) < 1e-4 * (1 + Math.Abs(expectedLatent[i])), $"latent[{i}] = {latent[i]}, reference {expectedLatent[i]}");

        var codes = model.Encode(audio);
        var expectedCodes = fixture["codes"]!.Select(row => row.Select(v => (int)v).ToArray()).ToArray();
        for (int q = 0; q < expectedCodes.Length; q++)
            for (int t = 0; t < expectedCodes[q].Length; t++)
                Assert.Equal(expectedCodes[q][t], codes[q, t]);
        Assert.Equal(expectedCodes[0], model.EncodeSemantic(audio));

        var decoded = model.Decode(codes);
        var expectedDecoded = fixture["decoded"]!.Select(v => (double)v).ToArray();
        Assert.Equal(expectedDecoded.Length, decoded.Length);
        for (int i = 0; i < decoded.Length; i++)
            Assert.True(Math.Abs(decoded[i] - expectedDecoded[i]) < 1e-4 * (1 + Math.Abs(expectedDecoded[i])), $"decoded[{i}] = {decoded[i]}, reference {expectedDecoded[i]}");
    }

    [Fact(Timeout = 300000)]
    public async Task PaperModel_Codes16kHzSpeechAt50FramesPerSecond()
    {
        await Task.Yield();
        // App. D / §4.1: strides (2, 4, 5, 8) = 320 at 16 kHz, 50 frames per second; eight codebooks of 1024.
        using var model = new SpeechTokenizer<double>(Architecture(), new SpeechTokenizerOptions
        {
            StftDiscriminatorFilters = 4, DiscriminatorWidthDivisor = 64, Dimension = 64, SemanticDimension = 32,
        });
        Assert.Equal(320, model.HopLength);
        Assert.Equal(50, model.TokenFrameRate);
        Assert.Equal(8, model.NumQuantizers);
        Assert.Equal(8, model.QuantizersForBandwidth(4.0));
    }

    [Fact(Timeout = 120000)]
    public async Task Training_NeedsTheSemanticTeacher()
    {
        await Task.Yield();
        // §3.2: the first codebook's role comes from the distillation, so training without the teacher's representations is
        // refused rather than silently dropping it.
        using var model = new SpeechTokenizer<double>(Architecture(), TinyOptions());
        var audio = new Tensor<double>(new[] { 96 });
        Assert.Throws<NotSupportedException>(() => model.Train(audio, audio));
    }

    [Fact(Timeout = 600000)]
    public async Task Distillation_AlignsTheFirstCodebookWithTheTeacher()
    {
        await Task.Yield();
        // §3.2's D-axis loss pulls A·Q1 toward the teacher along time, per teacher dimension. Two identical models train on
        // the same speech and teacher, one without the distillation term; only the distilled one's first codebook comes to
        // agree with the teacher. (Adam at 1e-2 so that twenty steps show it; the paper's 4e-4 needs far more.)
        double Agreement(double distillationWeight)
        {
            var options = TinyOptions();
            options.DistillationLossWeight = distillationWeight;
            var optimizer = new AiDotNet.Optimizers.AdamOptimizer<double, Tensor<double>, Tensor<double>>(null,
                new AiDotNet.Models.Options.AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 1e-2 });
            using var model = new SpeechTokenizer<double>(Architecture(), options, optimizer);
            var audio = new Tensor<double>(new[] { 96 });
            for (int i = 0; i < audio.Length; i++) audio[i] = 0.5 * Math.Sin(2 * Math.PI * 900 * i / 16000.0);
            var teacher = Teacher(12, 6);
            for (int i = 0; i < 20; i++) model.Train(audio, teacher);
            return model.DistillationAgreement(audio, teacher);
        }
        double distilled = Agreement(120.0);
        double control = Agreement(0.0);
        Assert.True(distilled > 0.2, $"The distilled first codebook's agreement with the teacher is only {distilled}.");
        Assert.True(distilled > control + 0.1, $"Distillation did not raise the agreement ({control} without, {distilled} with).");
    }
}

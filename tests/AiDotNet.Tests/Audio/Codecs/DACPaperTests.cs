using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Audio.Codecs;
using AiDotNet.Audio.Effects;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// DAC encodes, quantizes, decodes and trains as its paper specifies (Kumar et al. 2023), and loads the released
/// checkpoints' layout exactly.
/// </summary>
/// <remarks>
/// Before this change DAC was a stack of generic layers with a magnitude-bucketing "quantizer"; it had no Snake
/// activations, no factorized codebooks, no discriminators and trained by regression.
/// <c>ReferenceData/dac_official_layout_reference.json</c> holds a tiny Hugging Face DacModel with random weights
/// (generator: tools/reference-data/dac_reference.py) in the layout of descript/dac_44khz, with the latent,
/// codes and decoded audio of a fixed input.
/// </remarks>
public class DACPaperTests
{
    private static JObject Reference()
    {
        const string fileName = "dac_official_layout_reference.json";
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
        => new(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1) { RandomSeed = 17 };

    private static DACOptions TinyOptions() => new()
    {
        SampleRate = 16000,
        EncoderDim = 4,
        EncoderRates = [2, 4],
        DecoderDim = 16,
        DecoderRates = [4, 2],
        NumQuantizers = 2,
        CodebookSize = 16,
        CodebookDim = 2,
        TargetBandwidthKbps = 16.0,
        SegmentSize = 256,
        MelWindows = [32, 64],
        MelBins = [5, 10],
        DiscriminatorWindows = [64, 32],
        DiscriminatorChannels = 4,
        DiscriminatorWidthDivisor = 32,
        SamplingSeed = 7,
    };

    [Fact(Timeout = 300000)]
    public async Task OfficialLayout_LoadsAndReproducesTheReferenceModel()
    {
        await Task.Yield();
        var fixture = Reference();
        using var model = new DAC<double>(Architecture(), TinyOptions());
        string path = Path.Combine(Path.GetTempPath(), $"dac_reference_{Guid.NewGuid():N}.safetensors");
        File.WriteAllBytes(path, Convert.FromBase64String((string)fixture["safetensors_base64"]!));
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

        var decoded = model.Decode(codes);
        var expectedDecoded = fixture["decoded"]!.Select(v => (double)v).ToArray();
        Assert.Equal(expectedDecoded.Length, decoded.Length);
        for (int i = 0; i < decoded.Length; i++)
            Assert.True(Math.Abs(decoded[i] - expectedDecoded[i]) < 1e-4 * (1 + Math.Abs(expectedDecoded[i])), $"decoded[{i}] = {decoded[i]}, reference {expectedDecoded[i]}");
    }

    [Fact(Timeout = 600000)]
    public async Task PaperModel_Has86FramesPerSecondOfNineCodebooks()
    {
        await Task.Yield();
        // §4.3: strides (2, 4, 8, 8) = 512 at 44.1 kHz, about 86 frames per second; nine 10-bit codebooks.
        using var model = new DAC<double>(Architecture(), new DACOptions { DiscriminatorWidthDivisor = 64, DiscriminatorChannels = 2, DecoderDim = 96 });
        Assert.Equal(512, model.HopLength);
        Assert.Equal(86, model.TokenFrameRate);
        Assert.Equal(9, model.NumQuantizers);
        Assert.Equal(9, model.QuantizersForBandwidth(8.0));
    }

    [Fact(Timeout = 60000)]
    public async Task Codes_AreChosenByCosineSimilarity()
    {
        await Task.Yield();
        // App. A: k = argmin ||l2(W_in z) - l2(e_j)||, so scaling the latent leaves every code unchanged.
        var quantizer = new FactorizedVectorQuantizerLayer<double>(4, 2, 8, 2);
        var random = new Random(1);
        var z = new Tensor<double>(new[] { 1, 4, 10 });
        for (int i = 0; i < z.Length; i++) z[i] = random.NextDouble() - 0.5;
        var scaled = new Tensor<double>(z._shape, z.ToVector());
        for (int i = 0; i < scaled.Length; i++) scaled[i] *= 37.0;
        var a = quantizer.Encode(z, 1);
        var b = quantizer.Encode(scaled, 1);
        for (int t = 0; t < 10; t++) Assert.Equal(a[0, t], b[0, t]);
    }

    [Fact(Timeout = 60000)]
    public async Task CodebookLosses_UseNormalizedVectorsByDefault()
    {
        await Task.Yield();
        // App. A: both losses compare l2(z_proj) and l2(e_k), so each is at most 4 (two unit vectors); the reference code's raw
        // option compares unnormalized vectors.
        var normalized = new FactorizedVectorQuantizerLayer<double>(4, 1, 8, 2);
        var raw = new FactorizedVectorQuantizerLayer<double>(4, 1, 8, 2, lossesOnRawVectors: true);
        var z = new Tensor<double>(new[] { 1, 4, 6 });
        for (int i = 0; i < z.Length; i++) z[i] = 50.0 * Math.Sin(i + 1);
        normalized.SetTrainingMode(true);
        raw.SetTrainingMode(true);
        using (new NoGradScope<double>())
        {
            normalized.Forward(z);
            raw.Forward(z);
        }
        Assert.InRange(normalized.CommitmentLoss![0], 0.0, 4.0);
        Assert.Equal(normalized.CommitmentLoss[0], normalized.CodebookLoss![0], 12);
        Assert.True(raw.CommitmentLoss![0] > 4.0, $"The raw commitment loss {raw.CommitmentLoss[0]} should see the unnormalized scale.");
    }

    [Fact(Timeout = 600000)]
    public async Task Training_ReducesTheMelDistance()
    {
        await Task.Yield();
        using var model = new DAC<double>(Architecture(), TinyOptions());
        var provider = (AiDotNet.Interfaces.ITrainingObjectiveProvider<double>)model;
        var audio = new Tensor<double>(new[] { 256 });
        for (int i = 0; i < audio.Length; i++) audio[i] = 0.5 * Math.Sin(2 * Math.PI * 1000 * i / 16000.0);
        double before = provider.EvaluateTrainingObjective(audio, audio);
        for (int i = 0; i < 30; i++) model.Train(audio, audio);
        double after = provider.EvaluateTrainingObjective(audio, audio);
        Assert.True(after < before, $"The mel distance did not fall ({before} -> {after}).");
    }
}

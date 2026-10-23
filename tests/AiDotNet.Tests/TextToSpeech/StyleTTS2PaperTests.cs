using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.StyleEmotion;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// StyleTTS 2 (Li et al. 2023) reproduces yl4579/StyleTTS2 in both released configurations — multispeaker with the
/// HiFi-GAN decoder (LibriTTS) and single-speaker with the iSTFTNet decoder (LJSpeech): the text encoder, PL-BERT and its
/// projection, the reference style, the style denoiser with classifier-free guidance, ancestral DPM-2 sampling, the duration
/// bins, pitch and energy, the waveform, and the SLM discriminator head.
/// </summary>
/// <remarks><c>ReferenceData/styletts2_reference.json</c> holds tiny seeded models' float32 weights and their float64
/// results (<c>tools/reference-data/styletts2_reference.py</c>); the harmonic source's random phases and noise are off on
/// both sides and the sampler's draws are recorded.</remarks>
public class StyleTTS2PaperTests
{
    private static JObject Reference()
    {
        const string fileName = "styletts2_reference.json";
        string output = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(output) && dir is not null; dir = dir.Parent)
            output = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(output));
    }

    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);

    private static int[] Ints(JToken token) => token.Select(v => (int)v).ToArray();

    private static double[] Values(JToken token) => token.Select(v => (double)v).ToArray();

    private static (StyleTTS2<double> Model, JObject Shared, JObject Variant) Load(string name)
    {
        var fixture = Reference();
        var shared = (JObject)fixture["shared"]!;
        var variant = (JObject)fixture[name]!;
        var c = variant["config"]!;
        var decoder = c["decoder"]!;
        var transformer = c["diffusion"]!["transformer"]!;
        var bert = shared["bert"]!;
        var asr = shared["asr"]!;
        bool istft = (string)decoder["type"]! == "istftnet";
        var rates = Ints(decoder["upsample_rates"]!);
        int hop = 2 * rates.Aggregate(1, (a, b) => a * b) * (istft ? (int)decoder["gen_istft_hop_size"]! : 1);
        var model = new StyleTTS2<double>(Architecture(), new StyleTTS2Options
        {
            SampleRate = 24000, HopSize = hop,
            HiddenDim = (int)c["hidden_dim"]!, VocabSize = (int)c["n_token"]!, StyleDim = (int)c["style_dim"]!,
            TextLayers = (int)c["n_layer"]!, StyleEncoderDim = (int)c["dim_in"]!, MaxConvDim = (int)c["max_conv_dim"]!,
            MelChannels = (int)c["n_mels"]!, DropoutRate = (double)c["dropout"]!, DurationBins = (int)c["max_dur"]!,
            Multispeaker = (bool)c["multispeaker"]!,
            DecoderType = istft ? StyleTts2DecoderType.IstftNet : StyleTts2DecoderType.HiFiGan,
            DecoderBottleneck = (int)shared["bottleneck"]!, ResidualDim = (int)shared["residual"]!,
            ResBlockKernels = Ints(decoder["resblock_kernel_sizes"]!),
            ResBlockDilations = decoder["resblock_dilation_sizes"]!.Select(Ints).ToArray(),
            UpsampleRates = rates, UpsampleKernels = Ints(decoder["upsample_kernel_sizes"]!),
            UpsampleInitialChannels = (int)decoder["upsample_initial_channel"]!,
            IstftFftSize = istft ? (int)decoder["gen_istft_n_fft"]! : 20, IstftHopSize = istft ? (int)decoder["gen_istft_hop_size"]! : 5,
            BertHidden = (int)bert["hidden_size"]!, BertLayers = (int)bert["num_hidden_layers"]!, BertHeads = (int)bert["num_attention_heads"]!,
            BertIntermediate = (int)bert["intermediate_size"]!, BertEmbedding = (int)bert["embedding_size"]!,
            BertMaxPositions = (int)bert["max_position_embeddings"]!,
            DiffusionLayers = (int)transformer["num_layers"]!, DiffusionHeads = (int)transformer["num_heads"]!,
            DiffusionHeadFeatures = (int)transformer["head_features"]!, DiffusionMultiplier = (int)transformer["multiplier"]!,
            SigmaData = (double)c["diffusion"]!["dist"]!["sigma_data"]!,
            SlmHidden = (int)c["slm"]!["hidden"]!, SlmLayers = (int)c["slm"]!["nlayers"]!, SlmDiscriminatorChannels = (int)c["slm"]!["initial_channel"]!,
            AlignerHidden = (int)asr["hidden_dim"]!, AlignerTokenEmbedding = (int)asr["token_embedding_dim"]!, AlignerConvLayers = (int)asr["n_layers"]!,
            PitchExtractorChannels = Ints(shared["jdc_widths"]!), PitchExtractorLstm = (int)shared["jdc_lstm"]!,
            DiffusionSteps = (int)shared["steps"]!, Alpha = (double)shared["inference"]!["alpha"]!, Beta = (double)shared["inference"]!["beta"]!,
            TrimEndSamples = (bool)c["multispeaker"]! ? (int)shared["inference"]!["trim"]! : 0,
            FinalPhonemeExtraFrames = (int)shared["inference"]!["final_extra"]!,
        });
        string path = Path.Combine(Path.GetTempPath(), $"styletts2_reference_{Guid.NewGuid():N}.safetensors");
        File.WriteAllBytes(path, Convert.FromBase64String((string)variant["safetensors_base64"]!));
        try
        {
            var weights = new WeightLoader().LoadWeights(path);
            var used = new HashSet<string>();
            model.LoadReferenceWeights((tensorName, shape) =>
            {
                Assert.True(weights.ContainsKey(tensorName), $"The reference has no tensor '{tensorName}'.");
                var tensor = weights[tensorName];
                Assert.Equal(shape.Aggregate(1, (a, b) => a * b), tensor.Length);
                used.Add(tensorName);
                return tensor.ToVector().Select(v => (double)v).ToArray();
            });
            model.SlmDiscriminator.Load((tensorName, shape) =>
            {
                used.Add(tensorName);
                return weights[tensorName].ToVector().Select(v => (double)v).ToArray();
            }, "wd");
            // Batch-norm step counters are not weights; diffusion.unet is the same module as diffusion.diffusion.net; PL-BERT's
            // pooler is never read (the reference's CustomAlbert returns the last hidden state).
            var unused = weights.Keys.Where(k => !used.Contains(k) && !k.EndsWith("num_batches_tracked", StringComparison.Ordinal)
                && !k.StartsWith("diffusion.unet.", StringComparison.Ordinal) && !k.StartsWith("bert.pooler.", StringComparison.Ordinal)).ToList();
            Assert.True(unused.Count == 0, "Unread reference tensors: " + string.Join(", ", unused));
        }
        finally
        {
            File.Delete(path);
        }
        model.SetTrainingMode(false);
        model.DecoderNetwork.Generator.Source.Stochastic = false;
        return (model, shared, variant);
    }

    private static Tensor<double> Matrix(double[] values, params int[] shape) => new(shape, new Vector<double>(values));

    private static void Close(double[] expected, Tensor<double> actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-8 * Math.Max(1, Math.Abs(expected[i])),
                $"{what}[{i}]: reference {expected[i]:R}, port {actual[i]:R}.");
    }

    [Theory(Timeout = 300000)]
    [InlineData("libritts")]
    [InlineData("ljspeech")]
    public async Task Inference_MatchesTheReference(string variant)
    {
        await Task.Yield();
        var (model, shared, fixture) = Load(variant);
        using var _ = model;
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var random = new Random(0);
        var ids = Ints(shared["tokens"]!);
        int length = ids.Length, width = 2 * (int)fixture["config"]!["style_dim"]!;
        bool multispeaker = (bool)fixture["config"]!["multispeaker"]!;
        var tokens = Matrix(ids.Select(v => (double)v).ToArray(), length);

        Close(Values(fixture["text"]!), model.TextEncoderNetwork.Forward(tokens, false, random), "text features");
        var bert = model.BertNetwork.Encode(ids, false, random);
        Close(Values(fixture["bert"]!), bert, "PL-BERT");
        Close(Values(fixture["bert_encoder"]!), engine.TensorTranspose(model.BertEncoderLayer.Forward(bert)), "PL-BERT projection");

        Tensor<double>? reference = null;
        if (multispeaker)
        {
            var mel = Matrix(Values(shared["reference"]!), 80, (int)shared["reference_frames"]!);
            reference = model.ComputeStyle(engine.TensorTranspose(mel));
            Close(Values(fixture["reference_style"]!), reference, "reference style");
        }

        var probe = Matrix(Values(shared["probe"]!), 1, width);
        Close(Values(fixture["denoised"]!), model.DiffusionNetwork.Denoise(probe, (double)shared["sigma"]!, bert, reference,
            (double)shared["guidance"]!), "denoised style");

        var noise = Matrix(Values(shared["noise"]!), 1, width);
        var draws = shared["step_noise"]!.Select(d => Matrix(Values(d), 1, width)).ToArray();
        var sampled = model.DiffusionNetwork.Sample(noise, bert, reference, (int)shared["steps"]!, 1.0, random, i => draws[i]);
        Close(Values(fixture["sampled_style"]!), sampled, "sampled style");

        var wave = model.SynthesizeWave(ids, reference, noise, random, i => draws[i]);
        Close(Values(fixture["wave"]!), wave, "waveform");

        var hidden = Matrix(Values(shared["hidden"]!), 1, (int)fixture["config"]!["slm"]!["hidden"]! * (int)fixture["config"]!["slm"]!["nlayers"]!, 12);
        Close(Values(fixture["wd"]!), model.SlmDiscriminator.Forward(hidden), "SLM discriminator");
    }

    [Theory(Timeout = 300000)]
    [InlineData("libritts")]
    [InlineData("ljspeech")]
    public async Task Prosody_MatchesTheReference(string variant)
    {
        await Task.Yield();
        var (model, shared, fixture) = Load(variant);
        using var _ = model;
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var random = new Random(0);
        var ids = Ints(shared["tokens"]!);
        int length = ids.Length, s = (int)fixture["config"]!["style_dim"]!, hiddenDim = (int)fixture["config"]!["hidden_dim"]!;
        bool multispeaker = (bool)fixture["config"]!["multispeaker"]!;
        var sampled = Matrix(Values(fixture["sampled_style"]!), 1, 2 * s);
        var prosodic = engine.TensorSlice(sampled, new[] { 0, s }, new[] { 1, s });
        if (multispeaker)
        {
            double beta = (double)shared["inference"]!["beta"]!;
            var reference = Values(fixture["reference_style"]!);
            var mixed = new double[s];
            for (int i = 0; i < s; i++) mixed[i] = beta * prosodic[0, i] + (1 - beta) * reference[s + i];
            prosodic = Matrix(mixed, 1, s);
        }
        var bert = model.BertNetwork.Encode(ids, false, random);
        var dEn = engine.Reshape(engine.TensorTranspose(model.BertEncoderLayer.Forward(bert)), new[] { 1, hiddenDim, length });
        var encoded = model.PredictorNetwork.EncodeText(dEn, prosodic, false, random);
        var logits = model.PredictorNetwork.Durations(encoded, false, random);
        Close(Values(fixture["duration_logits"]!), logits, "duration logits");
        Assert.Equal(Ints(fixture["rounded_durations"]!), model.FramesOf(logits));
    }
}

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AiDotNet.Audio.Foundations;
using AiDotNet.Audio.Pitch;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.MultiModal;
using AiDotNet.TextToSpeech.Vocoders;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// SpiritLM's speech tokenizer and vocoders against the release (<c>tools/reference-data/spiritlm_speech_reference.py</c>):
/// the 25 Hz mHuBERT and its CTC linear quantizer, FCPE, pYAAPT, the F0 VQ-VAE, the Base and Expressive unit vocoders, the
/// wav2vec 2.0 style classifier and the token strings, on tiny seeded modules; the released checkpoints when
/// SPIRITLM_SPEECH_TOKENIZER names the release's <c>speech_tokenizer</c> directory.
/// </summary>
public class SpiritLmSpeechReferenceTests
{
    private const double Tolerance = 1e-9;

    private static readonly Lazy<JObject> Data = new(() =>
    {
        const string name = "spiritlm_speech_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", name);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", name);
        return JObject.Parse(File.ReadAllText(path));
    });

    private static double[] Doubles(JToken t) => t.Select(v => (double)v).ToArray();
    private static int[] Ints(JToken t) => t.Select(v => (int)v).ToArray();

    /// <summary>A safetensors blob's floating tensors in float64 (integer tensors are skipped).</summary>
    private static Dictionary<string, double[]> State(string base64)
    {
        var bytes = Convert.FromBase64String(base64);
        long headerLength = BitConverter.ToInt64(bytes, 0);
        var header = JObject.Parse(Encoding.UTF8.GetString(bytes, 8, (int)headerLength));
        var result = new Dictionary<string, double[]>();
        int start = 8 + (int)headerLength;
        foreach (var entry in header.Properties())
        {
            if (entry.Name == "__metadata__") continue;
            string dtype = (string)entry.Value["dtype"]!;
            var offsets = Ints(entry.Value["data_offsets"]!);
            int begin = start + offsets[0], length = offsets[1] - offsets[0];
            if (dtype == "F64") result[entry.Name] = Enumerable.Range(0, length / 8).Select(i => BitConverter.ToDouble(bytes, begin + 8 * i)).ToArray();
            else if (dtype == "F32") result[entry.Name] = Enumerable.Range(0, length / 4).Select(i => (double)BitConverter.ToSingle(bytes, begin + 4 * i)).ToArray();
        }
        return result;
    }

    private static Func<string, double[]?> Reader(Dictionary<string, double[]> state) => name => state.TryGetValue(name, out var v) ? v : null;

    private static void Close(JToken expected, IReadOnlyList<double> actual, string what, double tolerance = Tolerance)
    {
        var values = Doubles(expected);
        Assert.Equal(values.Length, actual.Count);
        for (int i = 0; i < values.Length; i++)
            Assert.True(Math.Abs(values[i] - actual[i]) <= tolerance * Math.Max(1, Math.Abs(values[i])), $"{what}[{i}]: {values[i]:R} vs {actual[i]:R}");
    }

    private static double[] Flat(Tensor<double> t) => t.ToArray();

    private static HubertConfiguration TinyHubert(JToken c) => new()
    {
        ConvDims = new[] { 12, 12, 12, 12 }, ConvKernels = new[] { 10, 3, 3, 2 }, ConvStrides = new[] { 5, 2, 2, 2 }, LayerNormFeatureEncoder = false,
        HiddenSize = (int)c["encoder_embed_dim"]!, Layers = 2, Heads = (int)c["encoder_attention_heads"]!, IntermediateSize = (int)c["encoder_ffn_embed_dim"]!,
        PositionalConvKernel = (int)c["conv_pos"]!, PositionalConvGroups = (int)c["conv_pos_groups"]!, LayerNormFirst = false, PositionalConvBatchNorm = true,
    };

    [Fact(Timeout = 120000)]
    public async Task Hubert25HzAndLinearQuantizer_MatchTheRelease()
    {
        await Task.Yield();
        var f = Data.Value;
        var layers = new List<LayerBase<double>>();
        var extractor = new HubertUnitExtractor<double>(AiDotNetEngine.Current, layers, TinyHubert(f["hubert_config"]!), 2, 6);
        extractor.LoadFairseqWeights(Reader(State((string)f["hubert_state"]!)));
        var quantizer = new HubertLinearQuantizer<double>(AiDotNetEngine.Current, layers, 16, 16, 6);
        quantizer.LoadTorchWeights(Reader(State((string)f["quantizer_state"]!)));
        extractor.Quantizer = quantizer;
        var audio = Doubles(f["hubert_audio"]!);
        using (new NoGradScope<double>())
        {
            var features = extractor.Network.Forward(new Tensor<double>(new[] { audio.Length }, new Vector<double>(audio)), 2)[2];
            Close(f["hubert_features"]!, Flat(features), "features");
            Close(f["quantizer_logits"]!, Flat(quantizer.Logits(features)), "quantizer logits");
        }
        Assert.Equal(Ints(f["units"]!), extractor.Units(audio, deduplicate: false));
        var cases = f["blank_cases"]!.Select(Ints).ToArray();
        var filled = f["blank_filled"]!.Select(Ints).ToArray();
        for (int i = 0; i < cases.Length; i++) Assert.Equal(filled[i], HubertLinearQuantizer<double>.FillBlanks(cases[i], 4));
    }

    [Fact(Timeout = 120000)]
    public async Task Fcpe_MatchesTorchfcpe()
    {
        await Task.Yield();
        var f = Data.Value;
        var m = f["fcpe_config"]!["model"]!;
        var fcpe = new FcpePitchEstimator<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), (int)f["fcpe_config"]!["mel"]!["num_mels"]!,
            (int)m["hidden_dims"]!, (int)m["n_layers"]!, (int)m["out_dims"]!);
        fcpe.LoadTorchWeights(Reader(State((string)f["fcpe_state"]!)));
        var audio = Doubles(f["fcpe_audio"]!);
        var mel = fcpe.LogMel(audio);
        Close(f["fcpe_mel"]!, mel.Cast<double>().ToArray(), "log-mel");
        using (new NoGradScope<double>()) Close(f["fcpe_saliences"]!, Flat(fcpe.Saliences(mel)), "saliences");
        Close(f["fcpe_f0"]!, fcpe.F0(audio), "F0");
        var (f0, voiced) = fcpe.SpiritLmF0(audio);
        Close(f["fcpe_f0_interpolated"]!, f0, "interpolated F0");
        Close(f["fcpe_voiced"]!, voiced, "voicing", 0);
    }

    [Fact(Timeout = 120000)]
    public async Task Yaapt_MatchesAmfmDecompy()
    {
        await Task.Yield();
        var f = Data.Value;
        var audio = Doubles(f["yaapt_audio"]!);
        var track = YaaptPitchTracker.SpiritLmTrack(audio);
        Close(f["yaapt_energy"]!, track.Energy, "NLFER energy");
        Close(f["yaapt_raw"]!, track.Pitch, "tracked pitch");
        var (f0, voiced) = YaaptPitchTracker.SpiritLmF0(audio);
        Close(f["yaapt_f0"]!, f0, "interpolated F0");
        Close(f["yaapt_voiced"]!, voiced, "voicing", 0);
    }

    [Fact(Timeout = 120000)]
    public async Task PitchVqVae_MatchesTheRelease()
    {
        await Task.Yield();
        var f = Data.Value;
        var c = f["vq_config"]!;
        var vq = new SpiritLmPitchQuantizer<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), new SpiritLmPitchQuantizerConfiguration
        {
            Codes = (int)c["vq_params"]!["l_bins"]!, LatentDim = (int)c["vq_params"]!["emb_width"]!, Width = (int)c["encoder_params"]!["width"]!,
            Depth = (int)c["encoder_params"]!["depth"]!, DilationGrowth = (int)c["encoder_params"]!["dilation_growth_rate"]!,
        }, new Random(0));
        vq.LoadTorchWeights(Reader(State((string)f["vqvae_state"]!)));
        var input = Doubles(f["vq_input"]!);
        int frames = input.Length / 2;
        var x = new Tensor<double>(new[] { 1, 2, frames }, new Vector<double>(input));
        using var _ = new NoGradScope<double>();
        var latents = vq.Encode(x);
        Close(f["vq_latents"]!, Flat(latents), "latents");
        Assert.Equal(Ints(f["vq_codes"]!), vq.Nearest(latents));
        Close(f["vq_reconstruction"]!, Flat(vq.Reconstruct(x)), "reconstruction");
    }

    private static CodeHiFiGan<double> TinyVocoder(JToken c, string state)
    {
        var configuration = new CodeHiFiGanConfiguration
        {
            Units = (int)c["num_embeddings"]!, EmbeddingDim = (int)c["embedding_dim"]!, InitialChannels = (int)c["upsample_initial_channel"]!,
            UpsampleRates = Ints(c["upsample_rates"]!), UpsampleKernels = Ints(c["upsample_kernel_sizes"]!), ResblockKernels = Ints(c["resblock_kernel_sizes"]!),
            ResblockDilations = c["resblock_dilation_sizes"]!.Select(Ints).ToArray(), DurationHidden = (int)c["dur_predictor_params"]!["var_pred_hidden_dim"]!,
            Speakers = (int)c["num_speakers"]!, Styles = c["num_styles"] is { } s ? (int)s : 0,
            PitchTokens = c["num_f0_tokens"] is { } p ? (int)p : 0, StyleTokens = c["num_style_tokens"] is { } t ? (int)t : 0,
        };
        var vocoder = new CodeHiFiGan<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), configuration, new Random(0));
        vocoder.LoadTorchWeights(Reader(State(state)));
        return vocoder;
    }

    [Fact(Timeout = 120000)]
    public async Task Vocoders_MatchTheReleaseCodeGenerator()
    {
        await Task.Yield();
        var f = Data.Value;
        var units = Ints(f["vocoder_units"]!);
        using var _ = new NoGradScope<double>();
        var baseVocoder = TinyVocoder(f["vocoder_base_config"]!, (string)f["vocoder_base_state"]!);
        Close(f["vocoder_base_log_durations"]!, Flat(baseVocoder.LogDurations(units)), "Base log durations");
        Close(f["vocoder_base_wave"]!, Flat(baseVocoder.Forward(units, true, speaker: 1, style: 3)), "Base waveform");
        var expressive = TinyVocoder(f["vocoder_expressive_config"]!, (string)f["vocoder_expressive_state"]!);
        Close(f["vocoder_expressive_wave"]!, Flat(expressive.Forward(units, true, speaker: 1, pitch: Ints(f["pitch_tokens"]!), styleTokens: Ints(f["style_tokens"]!))),
            "Expressive waveform");
        var cases = f["upsample_cases"]!.ToArray();
        var expected = f["upsampled"]!.Select(Ints).ToArray();
        for (int i = 0; i < cases.Length; i++)
            Assert.Equal(expected[i], CodeHiFiGan<double>.Upsample(Ints(cases[i][0]!), (int)cases[i][1]!, (int)cases[i][2]!));
    }

    [Fact(Timeout = 120000)]
    public async Task StyleClassifier_MatchesTheRelease()
    {
        await Task.Yield();
        var f = Data.Value;
        var c = f["style_config"]!;
        var configuration = new HubertConfiguration
        {
            ConvDims = Ints(c["conv_dim"]!), ConvKernels = Ints(c["conv_kernel"]!), ConvStrides = Ints(c["conv_stride"]!), LayerNormFeatureEncoder = false,
            HiddenSize = (int)c["hidden_size"]!, Layers = (int)c["num_hidden_layers"]!, Heads = (int)c["num_attention_heads"]!,
            IntermediateSize = (int)c["intermediate_size"]!, PositionalConvKernel = (int)c["num_conv_pos_embeddings"]!,
            PositionalConvGroups = (int)c["num_conv_pos_embedding_groups"]!, LayerNormFirst = false,
        };
        var encoder = new SpiritLmStyleEncoder<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), configuration, (int)c["classifier_proj_size"]!, 5, 10);
        encoder.LoadTorchWeights(Reader(State((string)f["style_state"]!)));
        var audio = Doubles(f["style_audio"]!);
        using (new NoGradScope<double>()) Close(f["style_logits"]!, Flat(encoder.Logits(audio)), "style logits");
        Assert.Equal(Ints(f["style_pred"]!), encoder.Tokens(audio));
    }

    [Fact(Timeout = 60000)]
    public async Task TokenStrings_MatchTheRelease()
    {
        await Task.Yield();
        var f = Data.Value;
        var cases = f["string_cases"]!.ToArray();
        var strings = f["strings"]!.Select(s => (string)s!).ToArray();
        int[]? Field(JToken c, string key) => c[key] is { } v ? ((string)v!).Split(' ').Select(int.Parse).ToArray() : null;
        for (int i = 0; i < cases.Length; i++)
            Assert.Equal(strings[i], SpiritLmTokens.UnitsToString(Field(cases[i], "hubert")!, Field(cases[i], "pitch"), Field(cases[i], "style")));
        var parseCases = f["parse_cases"]!.Select(s => (string)s!).ToArray();
        void Check(JToken expected, (int[] Hubert, int[]? Pitch, int[]? Style) actual)
        {
            Assert.Equal(Field(expected, "hubert"), actual.Hubert);
            Assert.Equal(Field(expected, "pitch"), actual.Pitch);
            Assert.Equal(Field(expected, "style"), actual.Style);
        }
        for (int i = 0; i < parseCases.Length; i++)
        {
            Check(f["parsed"]![i]!, SpiritLmTokens.StringToUnits(parseCases[i]));
            Check(f["parsed_duplicated"]![i]!, SpiritLmTokens.StringToUnits(parseCases[i], duplicateHubertForMultiplePitch: true));
        }
    }

    // ---------------------------------------------------------------- released checkpoints

    private static string? ReleaseDirectory => Environment.GetEnvironmentVariable("SPIRITLM_SPEECH_TOKENIZER") is { Length: > 0 } d && Directory.Exists(d) ? d : null;

    private static Func<string, double[]?> Checkpoint(string path, string prefix = "")
    {
        var weights = new WeightLoader().LoadWeights(path);
        return name => weights.TryGetValue(prefix + name, out var t) ? t.ToVector().Select(v => (double)v).ToArray() : null;
    }

    private static CodeHiFiGanConfiguration ReleaseVocoder(bool expressive) => new()
    {
        Units = 501, EmbeddingDim = 128, InitialChannels = 512, UpsampleRates = new[] { 5, 4, 4, 4, 2 }, UpsampleKernels = new[] { 11, 8, 8, 8, 4 },
        DurationHidden = 128, Speakers = 4, Styles = expressive ? 0 : 34, PitchTokens = expressive ? 64 : 0, StyleTokens = expressive ? 100 : 0,
    };

    /// <summary>The released 25 Hz mHuBERT and quantizer, FCPE (FCPE_CHECKPOINT: torchfcpe's bundled
    /// <c>fcpe_c_v001.pt</c>), the F0 VQ-VAE, both vocoders and the style classifier, on two seconds of LibriSpeech.</summary>
    [Fact(Timeout = 1800000)]
    public async Task RealSpeechTokenizer_MatchesTheRelease()
    {
        await Task.Yield();
        if (ReleaseDirectory is not { } dir) return;
        var f = Data.Value;
        Assert.True(f["real_audio"] is not null, "The fixture was written without the released checkpoints.");
        var audio = Doubles(f["real_audio"]!);
        var engine = AiDotNetEngine.Current;
        using var _ = new NoGradScope<double>();

        var layers = new List<LayerBase<double>>();
        var hubert = HubertUnitExtractor<double>.MHubertBase(11) with
        {
            ConvDims = Enumerable.Repeat(512, 8).ToArray(), ConvKernels = new[] { 10, 3, 3, 3, 3, 2, 2, 2 }, ConvStrides = new[] { 5, 2, 2, 2, 2, 2, 2, 2 },
            PositionalConvBatchNorm = true,
        };
        var extractor = new HubertUnitExtractor<double>(engine, layers, hubert, 11, 500) { MinChunk = 1280 };
        extractor.LoadFairseqWeights(Checkpoint(Path.Combine(dir, "hubert_25hz", "mhubert_base_25hz.pt"), "model_weight."));
        var quantizer = new HubertLinearQuantizer<double>(engine, layers, 768, 768, 500);
        quantizer.LoadTorchWeights(Checkpoint(Path.Combine(dir, "hubert_25hz", "L11_quantizer_500.pt"), "model_weight."));
        extractor.Quantizer = quantizer;
        var features = extractor.Network.Forward(new Tensor<double>(new[] { audio.Length }, new Vector<double>(audio)), 11)[11];
        Assert.Equal((int)f["real_hubert_frames"]!, features.Shape[0]);
        Close(f["real_hubert_head"]!, Flat(features).Take(4 * 768).ToArray(), "real features", 1e-8);
        var units = extractor.Units(audio, deduplicate: false);
        Assert.Equal(Ints(f["real_units"]!), units);

        var vq = new SpiritLmPitchQuantizer<double>(engine, new List<LayerBase<double>>(), new SpiritLmPitchQuantizerConfiguration(), new Random(0));
        vq.LoadTorchWeights(Checkpoint(Path.Combine(dir, "vqvae_f0_quantizer", "model.pt"), "model."));
        if (Environment.GetEnvironmentVariable("FCPE_CHECKPOINT") is { Length: > 0 } fcpePath && File.Exists(fcpePath))
        {
            var fcpe = new FcpePitchEstimator<double>(engine, new List<LayerBase<double>>());
            fcpe.LoadTorchWeights(Checkpoint(fcpePath, "model."));
            Close(f["real_fcpe_f0"]!, fcpe.F0(audio), "real FCPE F0", 1e-8);
            var (f0, voiced) = fcpe.SpiritLmF0(audio);
            Close(f["real_fcpe_interpolated"]!, f0, "real interpolated F0", 1e-8);
            Close(f["real_fcpe_voiced"]!, voiced, "real voicing", 0);
            // norm_f0_interp: the voiced mean removed from the nonzero frames.
            double mean = f0.Where((v, i) => voiced[i] != 0).Average();
            var normalized = f0.Select(v => v != 0 ? v - mean : v).ToArray();
            Assert.Equal(Ints(f["real_pitch_codes"]!), vq.Codes(normalized, voiced));
        }

        var codes = Ints(f["real_pitch_codes"]!);
        var vocoderUnits = units.Take(40).ToArray();
        foreach (bool expressive in new[] { false, true })
        {
            string name = expressive ? "expressive" : "base";
            var vocoder = new CodeHiFiGan<double>(engine, new List<LayerBase<double>>(), ReleaseVocoder(expressive), new Random(0));
            vocoder.LoadTorchWeights(Checkpoint(Path.Combine(dir, expressive ? "hifigan_spiritlm_expressive_w2v2" : "hifigan_spiritlm_base", "generator.pt"), "generator."));
            var wave = expressive
                ? vocoder.Forward(vocoderUnits, true, speaker: 2, pitch: Enumerable.Range(0, vocoderUnits.Length).Select(i => codes[Math.Min(i / 2, codes.Length - 1)]).ToArray(),
                    styleTokens: Enumerable.Repeat(71, vocoderUnits.Length).ToArray())
                : vocoder.Forward(vocoderUnits, true, speaker: 2, style: 8);
            Assert.Equal((int)f[$"real_vocoder_{name}_length"]!, wave.Length);
            Close(f[$"real_vocoder_{name}_head"]!, Flat(wave).Take(2000).ToArray(), $"real {name} waveform", 1e-8);
        }

        var style = new SpiritLmStyleEncoder<double>(engine, new List<LayerBase<double>>(), SpiritLmStyleEncoder<double>.Wav2Vec2Base());
        style.LoadTorchWeights(Checkpoint(Path.Combine(dir, "style_encoder_w2v2", "pytorch_model.bin")));
        Close(f["real_style_logits"]!, Flat(style.Logits(audio)), "real style logits", 1e-8);
        Assert.Equal(Ints(f["real_style"]!), style.Tokens(audio));
    }
}

using System.Reflection;
using System.Text.Json;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.VoiceCloning;

internal static class XttsAudit
{
    internal static void Run(string directory, string fixture, bool canonical = false)
    {
        Environment.SetEnvironmentVariable("OMP_NUM_THREADS", "1");
        AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = 1;
        AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(fixture));
        var root = document.RootElement;
        var failures = new List<string>();
        static double[] Values(JsonElement array) => array.EnumerateArray().Select(x => x.GetDouble()).ToArray();
        static Tensor<double> Tensor(JsonElement array, params int[] shape) => new(shape, new Vector<double>(Values(array)));
        void Check(string label, Tensor<double> actual, JsonElement expected, double tolerance)
        {
            var values = Values(expected);
            if (actual.Length != values.Length) throw new InvalidDataException($"{label}: output lengths differ.");
            double error = actual.ToArray().Zip(values, (a,b) => Math.Abs(a-b)).Max();
            Console.WriteLine($"{label}: maximum error {error:R}.");
            if (!double.IsFinite(error) || error > tolerance) failures.Add($"{label}: {error:R} exceeds {tolerance:R}");
        }
        Console.WriteLine($"XTTS checkpoint {root.GetProperty("revision").GetString()}, code {root.GetProperty("code_revision").GetString()}.");
        Console.WriteLine(canonical ? "Creating complete native canonical XTTS-v2." : "Creating complete native XTTS-v2 cloning facade.");
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
        using XTTSv2<double> model = canonical
            ? new XTTSv2<double>(architecture, new XTTSv2Options { TokenizerPath = Path.Combine(directory, "vocab.json") })
            : new XTTSv2Clone<double>(architecture, new XTTSv2CloneOptions
            {
                TokenizerPath = Path.Combine(directory, "vocab.json"), MinReferenceDurationSec = .33,
            });
        model.DisableAutoStreaming();
        foreach (var item in root.GetProperty("tokenizer_cases").EnumerateArray())
        {
            var expected = item.GetProperty("ids").EnumerateArray().Select(x => x.GetInt32()).ToArray();
            var actual = model.TokenizeText(item.GetProperty("text").GetString()!,item.GetProperty("language").GetString()!);
            if (!actual.SequenceEqual(expected)) throw new InvalidDataException(
                $"Original XTTS tokenizer IDs differ for {item.GetProperty("text").GetString()}: expected [{string.Join(",", expected)}], native [{string.Join(",", actual)}].");
        }
        Console.WriteLine("Released English/Hindi tokenizer IDs passed.");
        Console.WriteLine("Importing the original wrapped model.pth, dVAE, tokenizer and normalization through LoadAuthorWeights.");
        model.LoadAuthorWeights(Path.Combine(directory,"model.pth"),Path.Combine(directory,"dvae.pth"),
            Path.Combine(directory,"vocab.json"),Path.Combine(directory,"mel_stats.pth"));
        model.SetTrainingMode(false);
        GC.Collect(); GC.WaitForPendingFinalizers();
        T Field<T>(string name) => (T)typeof(XTTSv2<double>).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
        var gpt = Field<XttsLanguageModel<double>>("_languageModel");
        var expectedGpt = root.GetProperty("gpt");
        var condition = gpt.Condition(Tensor(expectedGpt.GetProperty("mel"),80,expectedGpt.GetProperty("frames").GetInt32()));
        int[] text = expectedGpt.GetProperty("text").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        int[] audio = expectedGpt.GetProperty("audio").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        var hidden = gpt.Forward(text,audio,condition);
        var textHead = (DenseLayer<double>)typeof(XttsLanguageModel<double>).GetField("_textHead",BindingFlags.Instance|BindingFlags.NonPublic)!.GetValue(gpt)!;
        var audioHead = (DenseLayer<double>)typeof(XttsLanguageModel<double>).GetField("_audioHead",BindingFlags.Instance|BindingFlags.NonPublic)!.GetValue(gpt)!;
        var textLogits=textHead.Forward(hidden.Text);var audioLogits=audioHead.Forward(hidden.Audio);
        foreach (string precision in new[] {"float64","float32"})
        {
            var expected=expectedGpt.GetProperty(precision);
            Check($"Released XTTS conditioner / {precision}",condition,expected.GetProperty("condition"),2e-4);
            Check($"Released XTTS text hidden / {precision}",hidden.Text,expected.GetProperty("text_hidden"),2e-4);
            Check($"Released XTTS audio hidden / {precision}",hidden.Audio,expected.GetProperty("audio_hidden"),2e-4);
            Check($"Released XTTS text logits / {precision}",textLogits,expected.GetProperty("text_logits"),2e-4);
            Check($"Released XTTS audio logits / {precision}",audioLogits,expected.GetProperty("audio_logits"),2e-4);
        }
        var dvae=Field<XttsDiscreteVae<double>>("_dvae");
        var expectedVae=root.GetProperty("dvae");
        var codes=dvae.Encode(Tensor(expectedVae.GetProperty("mel"),80,20));
        Check("Released XTTS dVAE code IDs",codes,expectedVae.GetProperty("codes"),0);
        Check("Released XTTS dVAE reconstruction",dvae.Decode(codes),expectedVae.GetProperty("reconstruction"),2e-6);
        var expectedSpeaker=root.GetProperty("speaker");
        var referenceAudio=Tensor(expectedSpeaker.GetProperty("audio"),Values(expectedSpeaker.GetProperty("audio")).Length);
        var speaker=model.ExtractSpeakerEmbedding(referenceAudio);
        foreach(string precision in new[]{"float64","float32"})
            Check($"Released XTTS H/ASP waveform speaker / {precision}",speaker,expectedSpeaker.GetProperty(precision),2e-4);
        var decoder=Field<XttsWaveformDecoder<double>>("_decoder");
        var expectedDecoder=root.GetProperty("decoder");
        var latents=Tensor(expectedDecoder.GetProperty("latents"),expectedDecoder.GetProperty("frames").GetInt32(),1024);
        Check("Released XTTS HiFi-GAN waveform / author features",decoder.Forward(latents,Tensor(expectedDecoder.GetProperty("speaker"),512)),
            expectedDecoder.GetProperty("waveform"),2e-6);
        Check("Released XTTS HiFi-GAN waveform / native GPT and speaker",decoder.Forward(hidden.Audio,speaker),
            expectedDecoder.GetProperty("waveform"),2e-4);
        if(failures.Count>0)throw new InvalidDataException(string.Join(Environment.NewLine,failures));
        Console.WriteLine(canonical ? "Original full canonical XTTS-v2 checkpoint audit passed." : "Original full XTTS-v2 cloning-facade checkpoint audit passed.");
    }
}

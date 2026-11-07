using System.Text.Json;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.VoiceCloning;

if (args.Length == 2 && args[0] == "metavoice-output")
{
    MetaVoiceOutputAudit.Run(args[1]);
    return;
}

if (args.Length == 4 && args[0] == "metavoice-public-release")
{
    MetaVoicePublicAudit.Run(args[1], args[2], args[3]);
    return;
}

if (args.Length == 2 && args[0] == "metavoice-deepfilter")
{
    MetaVoiceDeepFilterAudit.Run(args[1]);
    return;
}

if (args.Length == 3 && args[0] == "metavoice-public")
{
    MetaVoicePublicAudit.Run(args[1], args[2]);
    return;
}

if (args.Length == 2 && args[0] == "metavoice-speaker")
{
    MetaVoiceSpeakerAudit.Run(args[1]);
    return;
}

if (args.Length == 2 && args[0] == "metavoice-mbd-pipeline")
{
    MetaVoiceMultibandPipelineAudit.Run(args[1]);
    return;
}

if (args.Length == 2 && args[0] == "metavoice-tokenizer")
{
    MetaVoiceTokenizerAudit.Run(args[1]);
    return;
}

if (args.Length == 2 && args[0] == "metavoice-encodec")
{
    MetaVoiceEncodecAudit.Run(args[1]);
    return;
}

if (args.Length == 2 && args[0] == "metavoice-mbd")
{
    MetaVoiceMultibandAudit.Run(args[1]);
    return;
}

if (args.Length == 3 && args[0] == "metavoice-gpt")
{
    MetaVoiceGptAudit.Run(args[1], args[2]);
    return;
}

if (args.Length == 4 && args[0] == "kokoro-gradients")
{
    KokoroGradientAudit.Run(args[1], args[2], args[3]);
    return;
}
if (args.Length == 3 && (args[0] == "kokoro" || args[0] == "kokoro-author-spectrum"))
{
    KokoroAudit.Run(args[1], args[2], args[0] == "kokoro-author-spectrum");
    return;
}
if (args.Length == 5 && (args[0] == "kokoro-public" || args[0] == "kokoro-public-clone" || args[0] == "kokoro-public-state"))
{
    KokoroPublicAudit.Run(args[1], args[2], args[3], args[4], checkClone: args[0] == "kokoro-public-clone", checkRestore: args[0] == "kokoro-public-state");
    return;
}

if (args.Length == 3 && args[0] == "openvoice")
{
    OpenVoiceAudit.Run(args[1], args[2]);
    return;
}

if (args.Length == 3 && args[0] == "cosyvoice1-all")
{
    CosyVoiceReleasedAudit.All(args[1], args[2]);
    return;
}
if (args.Length == 3 && args[0] == "cosyvoice1-flow")
{
    CosyVoiceReleasedAudit.Flow(args[1], args[2]);
    return;
}
if (args.Length == 3 && args[0] == "cosyvoice1-vocoder")
{
    CosyVoiceReleasedAudit.Vocoder(args[1], args[2]);
    return;
}
if (args.Length == 3 && (args[0] == "cosyvoice1-lm" || args[0] == "cosyvoice2-lm"))
{
    CosyVoiceReleasedAudit.LanguageModel(args[0] == "cosyvoice1-lm" ? 1 : 2, args[1], args[2]);
    return;
}
if (args.Length == 3 && (args[0] == "cosyvoice1-frontend" || args[0] == "cosyvoice2-frontend"))
{
    CosyVoiceReleasedAudit.Frontend(args[0] == "cosyvoice1-frontend" ? 1 : 2, args[1], args[2]);
    return;
}
if (args.Length == 3 && args[0] == "vallex-community")
{
    VallexAudit.Run(args[1], args[2]);
    return;
}
if (args.Length == 3 && args[0] == "uniaudio-critics")
{
    UniAudioCriticAudit.Run(args[1], args[2]);
    return;
}
if (args.Length == 4 && args[0] == "uniaudio")
{
    UniAudioAudit.Run(args[1], args[2], args[3]);
    return;
}
if (args.Length == 2 && args[0] == "uniaudio-inspect")
{
    UniAudioAudit.Inspect(args[1]);
    return;
}
if (args.Length == 4 && args[0] == "voicecraft")
{
    VoiceCraftAudit.Run(args[1], args[2], args[3]);
    return;
}
if (args.Length == 2 && args[0] == "inspect")
{
    var state = new WeightLoader().LoadWeights(args[1]);
    Console.WriteLine($"Original checkpoint contains {state.Count} readable named tensors.");
    foreach (var item in state.Take(12)) Console.WriteLine($"{item.Key}: [{string.Join(",", item.Value.Shape.ToArray())}]");
    return;
}
if (args.Length == 3 && args[0] == "cosyvoice3-frontend")
{
    CosyVoice3FrontendAudit.Run(args[1], args[2]);
    return;
}
if (args.Length == 3 && args[0] == "cosyvoice3-vocoder")
{
    CosyVoice3VocoderAudit.Run(args[1], args[2]);
    return;
}
if (args.Length == 3 && args[0] == "cosyvoice3-flow")
{
    CosyVoice3FlowAudit.Run(args[1], args[2]);
    return;
}
if (args.Length == 3 && args[0] == "cosyvoice3-lm")
{
    CosyVoice3Audit.Run(args[1], args[2]);
    return;
}
if (args.Length == 3 && args[0] == "spark-speaker")
{
    SparkSpeakerAudit.Run(args[1], args[2]);
    return;
}
if (args.Length == 3 && args[0] == "spark")
{
    SparkAudit.Run(args[1], args[2]);
    return;
}
if (args.Length == 3 && (args[0] == "xtts" || args[0] == "xtts-canonical"))
{
    XttsAudit.Run(args[1], args[2], canonical: args[0] == "xtts-canonical");
    return;
}
if (args.Length == 3 && args[0] == "tortoise")
{
    TortoiseAudit.Run(args[1], args[2]);
    return;
}
if (args.Length == 3 && args[0] == "tortoise-sampling")
{
    TortoiseSamplingAudit.Run(args[1], args[2]);
    return;
}
if (args.Length == 3 && args[0] == "chatterbox-cam")
{
    S3GenAudit.Cam(args[1], args[2]);
    return;
}
if (args.Length == 3 && args[0] == "chatterbox-mel")
{
    S3GenAudit.Mel(args[2]);
    return;
}
if (args.Length == 3 && args[0] == "chatterbox-s3gen")
{
    S3GenAudit.Run(args[1], args[2]);
    return;
}
if (args.Length != 3 || args[0] != "chatterbox")
    throw new ArgumentException("Usage: TtsCheckpointAudit <mode> <original-weight-directory> <author-reference.json> [codec-reference.json]. See tools/TtsCheckpointAudit/README.md for modes, including cosyvoice1-all's reference-directory argument.");
Environment.SetEnvironmentVariable("OMP_NUM_THREADS", "1");
Environment.SetEnvironmentVariable("OPENBLAS_NUM_THREADS", "1");
Environment.SetEnvironmentVariable("MKL_NUM_THREADS", "1");
AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = 1;
AiDotNetEngine.ResetToCpu();
using var document = JsonDocument.Parse(File.ReadAllText(args[2]));
var reference = document.RootElement;
var directory = Path.GetFullPath(args[1]);
Console.WriteLine($"Author checkpoint revision: {reference.GetProperty("revision").GetString()}");
Console.WriteLine($"Author code revision: {reference.GetProperty("code_revision").GetString()}");
var tokenizer = ChatterboxTokenizer.Load(Path.Combine(directory, "tokenizer.json"));
foreach (var item in reference.GetProperty("tokenizer_cases").EnumerateArray())
{
    var expected = item.GetProperty("ids").EnumerateArray().Select(value => value.GetInt32()).ToArray();
    var actual = tokenizer.Encode(item.GetProperty("text").GetString()!);
    if (!actual.SequenceEqual(expected))
        throw new InvalidDataException($"Released tokenizer mismatch for {item.GetProperty("text").GetString()}: "
            + $"expected [{string.Join(",", expected)}], actual [{string.Join(",", actual)}].");
}
Console.WriteLine("Released tokenizer cases passed.");

var options = new ChatterboxOptions();
var voice = new ChatterboxVoiceEncoder<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), options);
voice.LoadTorchWeights(Reader(Path.Combine(directory, "ve.safetensors")));
Tensor<double>? nativeSpeaker = null;
foreach (var item in reference.GetProperty("voice_cases").EnumerateArray())
{
    var samples = Values(item.GetProperty("audio"));
    nativeSpeaker = voice.Embed(new Tensor<double>(new[] { samples.Length }, new Vector<double>(samples)), 16000);
    var error = MaximumError(nativeSpeaker, Values(item.GetProperty("embedding")));
    Console.WriteLine($"Released VE, {samples.Length} samples: maximum embedding error {error:R}.");
    if (error >= 2e-6) throw new InvalidDataException("Released voice encoder differs from the author.");
}
Console.WriteLine("Creating the full 520M native T3 graph.");
var layers = new List<LayerBase<double>>();
var t3 = new ChatterboxT3<double>(AiDotNetEngine.Current, layers, options, new Random(1));
Console.WriteLine("Loading original T3 named tensors.");
t3.LoadTorchWeights(Reader(Path.Combine(directory, "t3_cfg.safetensors")));
foreach (var layer in layers) layer.SetTrainingMode(false);
GC.Collect(); GC.WaitForPendingFinalizers();
int[] Ids(string key) => reference.GetProperty(key).EnumerateArray().Select(value => value.GetInt32()).ToArray();
var speaker = new Tensor<double>(new[] { 256 }, new Vector<double>(Values(reference.GetProperty("speaker"))));
var condition = new ChatterboxT3Condition<double>(speaker, Ids("prompt"), .7);
var logits = t3.TeacherForcedLogits(Ids("text"), Ids("speech"), condition);
CheckT3(logits.Text, logits.Speech, "author speaker embedding");
if (nativeSpeaker is null) throw new InvalidDataException("No reference voice case was supplied.");
logits = t3.TeacherForcedLogits(Ids("text"), Ids("speech"), condition with { SpeakerEmbedding = nativeSpeaker });
CheckT3(logits.Text, logits.Speech, "native waveform-to-speaker chain");
Console.WriteLine("Released Chatterbox tokenizer/VE/T3 checkpoint audit passed.");

void CheckT3(Tensor<double> text, Tensor<double> speech, string label)
{
    foreach (string precision in new[] { "float64", "float32" })
    {
        var expected = reference.GetProperty(precision);
        double textError = MaximumError(text, Values(expected.GetProperty("text_logits")));
        double speechError = MaximumError(speech, Values(expected.GetProperty("speech_logits")));
        Console.WriteLine($"Released T3 ({label}), author {precision}: text {textError:R}, speech {speechError:R}.");
        // HF forces float32 RMS variance and attention accumulation, including in its double model.
        if (Math.Max(textError, speechError) >= 2e-4)
            throw new InvalidDataException($"Released T3 differs from the author ({precision}).");
    }
    var expectedIds = reference.GetProperty("float64").GetProperty("speech_argmax").EnumerateArray().Select(value => value.GetInt32()).ToArray();
    for (int row = 0; row < expectedIds.Length; row++)
    {
        int best = 0;
        for (int code = 1; code < speech.Shape[2]; code++) if (speech[0, row, code] > speech[0, row, best]) best = code;
        if (best != expectedIds[row]) throw new InvalidDataException($"Released T3 argmax differs at speech position {row}.");
    }
}

static double[] Values(JsonElement array) => array.EnumerateArray().Select(value => value.GetDouble()).ToArray();
static double MaximumError(Tensor<double> tensor, double[] expected)
{
    if (tensor.Length != expected.Length) throw new InvalidDataException("Reference output shape/length differs.");
    return tensor.ToArray().Zip(expected, (a, b) => Math.Abs(a - b)).Max();
}
static Func<string, int[], double[]> Reader(string path)
{
    var state = new WeightLoader().LoadWeights(path);
    return (name, shape) =>
    {
        if (!state.TryGetValue(name, out var tensor)) throw new InvalidDataException($"Missing original tensor '{name}'.");
        if (!tensor.Shape.ToArray().SequenceEqual(shape)) throw new InvalidDataException($"Original tensor '{name}' has incompatible axes.");
        return tensor.ToArray().Select(value => (double)value).ToArray();
    };
}

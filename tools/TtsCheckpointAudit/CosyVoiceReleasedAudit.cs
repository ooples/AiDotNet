using System.Text.Json;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.VoiceCloning;
using AiDotNet.TextToSpeech.Vocoders;
using System.Reflection;

internal static class CosyVoiceReleasedAudit
{
    private sealed class RecordedRandom : Random
    {
        private readonly Queue<double> _values = new();
        internal RecordedRandom(IEnumerable<double> gaussian, IEnumerable<double>? phases = null)
        {
            if (phases is not null)
                foreach (double phase in phases.Skip(1)) _values.Enqueue(phase / (2 * Math.PI) + .5);
            foreach (double value in gaussian)
            {
                _values.Enqueue(1 - Math.Exp(-value * value / 2));
                _values.Enqueue(value < 0 ? .5 : 0);
            }
        }
        public override double NextDouble() => _values.Dequeue();
        internal void VerifyConsumed() { if (_values.Count != 0) throw new InvalidDataException("Sampling draw layout differs."); }
    }

    internal static void All(string directory, string referenceDirectory)
    {
        AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = 1; AiDotNetEngine.ResetToCpu();
        using var lmDocument = JsonDocument.Parse(File.ReadAllText(Path.Combine(referenceDirectory, "cosyvoice1-pretrained-reference.json")));
        using var flowDocument = JsonDocument.Parse(File.ReadAllText(Path.Combine(referenceDirectory, "cosyvoice1-flow-pretrained-reference.json")));
        using var hiftDocument = JsonDocument.Parse(File.ReadAllText(Path.Combine(referenceDirectory, "cosyvoice1-hift-pretrained-reference.json")));
        static double[] Values(JsonElement x) => x.EnumerateArray().Select(v => v.GetDouble()).ToArray();
        static int[] Ids(JsonElement x) => x.EnumerateArray().Select(v => v.GetInt32()).ToArray();
        void Check(string label, Tensor<double> actual, JsonElement expected, double tolerance = 2e-8)
        {
            var values = Values(expected);
            if (actual.Length != values.Length) throw new InvalidDataException(label + " shape differs.");
            double error = actual.ToArray().Zip(values, (a, b) => Math.Abs(a - b)).Max();
            Console.WriteLine($"Complete original 300M facade {label}: maximum error {error:R}.");
            if (!double.IsFinite(error) || error > tolerance) throw new InvalidDataException(label + " differs.");
        }
        Console.WriteLine("Creating the complete released CosyVoice-300M cloning facade.");
        using var model = new CosyVoiceClone<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Generative, inputSize: 1, outputSize: 1));
        model.DisableAutoStreaming();
        Console.WriteLine("Loading all original llm.pt/flow.pt/hift.pt and frozen ONNX components through LoadPretrainedWeights.");
        model.LoadPretrainedWeights(directory); model.SetTrainingMode(false); GC.Collect();
        using var scope = new NoGradScope<double>();
        var lmRoot = lmDocument.RootElement; var lmReference = lmRoot.GetProperty("float64_derived_positions");
        var speaker = new Tensor<double>(new[] { 192 }, new Vector<double>(Values(lmReference.GetProperty("xvector"))));
        var lmInput = model.LanguageModel!.Input(Ids(lmRoot.GetProperty("text")), Ids(lmRoot.GetProperty("speech")), speaker, false, new Random(0));
        Check("LM input", lmInput, lmReference.GetProperty("inputs"));
        var hidden = model.LanguageModel.Llm.Forward(lmInput, false, new Random(0));
        Check("LM logits", model.LanguageModel.Head.Forward(hidden), lmReference.GetProperty("logits"));
        var flowRoot = flowDocument.RootElement; var flowReference = flowRoot.GetProperty("float64_operators_derived_buffers");
        model.Voice = new TtsVoice<double>
        {
            SpeakerEmbedding = new Tensor<double>(new[] { 192 }, new Vector<double>(Values(flowRoot.GetProperty("speaker")))),
            Reference = new Tensor<double>(new[] { 5, 80 }, new Vector<double>(Values(flowRoot.GetProperty("prompt_mel")))),
            ReferenceCodecTokens = new Tensor<double>(new[] { 3 }, new Vector<double>(Values(flowRoot.GetProperty("prompt_tokens")))),
        };
        var flowRandom = new RecordedRandom(Values(flowRoot.GetProperty("noise")));
        var mel = model.TokensToMel(Ids(flowRoot.GetProperty("tokens")), flowRandom); flowRandom.VerifyConsumed();
        Check("public prompted TokensToMel", mel, flowReference.GetProperty("output"));
        var hiftReference = hiftDocument.RootElement.GetProperty("float64");
        var vocoderRandom = new RecordedRandom(Values(hiftReference.GetProperty("harmonic_noise")), Values(hiftReference.GetProperty("phases")));
        var waveform = model.MelToAudio(mel, vocoderRandom); vocoderRandom.VerifyConsumed();
        Check("public MelToAudio with replayed original sampling draws", waveform, hiftReference.GetProperty("waveform"));
        Console.WriteLine("PASS: full original-file 300M facade import, LM, prompted flow and actual stochastic vocoder pipeline.");
    }

    internal static void Vocoder(string directory, string fixturePath)
    {
        AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = 1; AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(fixturePath)); var root = document.RootElement;
        var reference = root.GetProperty("float64");
        static double[] Values(JsonElement x) => x.EnumerateArray().Select(v => v.GetDouble()).ToArray();
        var owner = new List<LayerBase<double>>();
        var configuration = new HiftConfiguration(80, 512, 8, 22050, .1, .003, 10,
            [8, 8], [16, 16], 16, 4, [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            [7, 11], [[1, 3, 5], [1, 3, 5]], .1, .99, 512);
        var vocoder = new HiftGenerator<double>(AiDotNetEngine.Current, owner, configuration, new Random(0));
        // Use the same original-file reader as the public model release loader.
        var read = (Func<string, int[], double[]>)typeof(CosyVoiceBase<double>).GetMethod("CheckpointReader",
            BindingFlags.Static | BindingFlags.NonPublic)!.Invoke(null, new object[] { Path.Combine(directory, "hift.pt") })!;
        vocoder.LoadTorchWeights(read);
        int frames = root.GetProperty("frames").GetInt32();
        var mel = new Tensor<double>(new[] { 1, 80, frames }, new Vector<double>(Values(root.GetProperty("mel"))));
        var source = new Tensor<double>(new[] { 1, 1, frames * 256 }, new Vector<double>(Values(reference.GetProperty("source"))));
        void Check(string label, Tensor<double> actual, JsonElement expected)
        {
            var values = Values(expected);
            if (actual.Length != values.Length) throw new InvalidDataException(label + " shape differs.");
            double error = actual.ToArray().Zip(values, (a, b) => Math.Abs(a - b)).Max();
            Console.WriteLine($"Full original CosyVoice HiFT {label}: maximum error {error:R}.");
            if (!double.IsFinite(error) || error > 2e-8) throw new InvalidDataException(label + " differs.");
        }
        using var scope = new NoGradScope<double>();
        Check("F0 predictor", vocoder.PredictF0(mel), reference.GetProperty("f0"));
        var harmonicRows = new Tensor<double>(new[] { 1, frames * 256, 9 },
            new Vector<double>(Values(reference.GetProperty("harmonics"))));
        Check("trained harmonic merge", vocoder.Merge(AiDotNetEngine.Current.TensorPermute(harmonicRows, [0, 2, 1])), reference.GetProperty("source"));
        Check("waveform with original stochastic source", vocoder.Decode(mel, source), reference.GetProperty("waveform"));
        Console.WriteLine("PASS: complete original CosyVoice-300M HiFT checkpoint, F0 and waveform.");
    }

    internal static void Flow(string directory, string fixturePath)
    {
        AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = 1; AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(fixturePath)); var root = document.RootElement;
        var reference = root.GetProperty("float64_operators_derived_buffers");
        static double[] Values(JsonElement x) => x.EnumerateArray().Select(v => v.GetDouble()).ToArray();
        static int[] Ids(JsonElement x) => x.EnumerateArray().Select(v => v.GetInt32()).ToArray();
        var configuration = new CosyVoiceFlowConfiguration(4096, 512, 80, 192, 50, 22050, 256,
            new WenetEncoderConfiguration(512, 512, 8, 2048, 6, WenetInputLayer.Linear, Swish: true, Conformer: true),
            4, [256, 256], 8, 64, 4, 12, 1e-6, .2, .7, 10);
        var owner = new List<LayerBase<double>>(); var engine = AiDotNetEngine.Current;
        var flow = new CosyVoiceFlow<double>(engine, owner, configuration, new Random(0));
        var state = new WeightLoader().LoadWeights(Path.Combine(directory, "flow.pt"));
        flow.LoadTorchWeights((name, shape) =>
        {
            if (!state.TryGetValue(name, out var value) || (shape.Length > 0 && !value.Shape.ToArray().SequenceEqual(shape)))
                throw new InvalidDataException("Missing or incompatible original flow tensor: " + name);
            return value.ToArray().Select(x => (double)x).ToArray();
        }, "");
        state.Clear(); GC.Collect();
        var tokens = Ids(root.GetProperty("tokens")); var prompt = Ids(root.GetProperty("prompt_tokens"));
        var promptMel = new Tensor<double>(new[] { 5, 80 }, new Vector<double>(Values(root.GetProperty("prompt_mel"))));
        var speaker = new Tensor<double>(new[] { 192 }, new Vector<double>(Values(root.GetProperty("speaker"))));
        var noise = new Tensor<double>(new[] { 1, 80, 15 }, new Vector<double>(Values(root.GetProperty("noise"))));
        void Check(string label, Tensor<double> actual, double[] expected)
        {
            if (actual.Length != expected.Length) throw new InvalidDataException(label + " shape differs.");
            double error = actual.ToArray().Zip(expected, (a, b) => Math.Abs(a - b)).Max();
            Console.WriteLine($"Full original CosyVoice flow {label}: maximum error {error:R}.");
            if (!double.IsFinite(error) || error > 2e-8) throw new InvalidDataException(label + " differs.");
        }
        using var scope = new NoGradScope<double>();
        var encoded = flow.Encode(prompt.Concat(tokens).ToArray(), false, new Random(0));
        var mu = flow.StretchForInference(engine.TensorSlice(encoded, [0, 0], [3, 80]),
            engine.TensorSlice(encoded, [3, 0], [6, 80]), 5, 10);
        Check("conditioned encoder/regulator", mu, Values(reference.GetProperty("mu")));
        var condition = new Tensor<double>(new[] { 1, 80, 15 }, new Vector<double>(Values(reference.GetProperty("condition"))));
        var mask = new Tensor<double>(new[] { 1, 1, 15 }, new Vector<double>(Enumerable.Repeat(1.0, 15).ToArray()));
        Check("first guided field", flow.Field(noise, mu, flow.ProjectSpeaker(speaker), condition, 0, mask),
            Values(reference.GetProperty("estimator_steps")[0]).Take(80 * 15).ToArray());
        Check("ten-step prompted inference", flow.Inference(tokens, prompt, promptMel, speaker, noise),
            Values(reference.GetProperty("output")));
        Console.WriteLine("PASS: complete original CosyVoice-300M trained flow and ten-step prompted inference.");
    }

    internal static void LanguageModel(int variant, string directory, string fixturePath)
    {
        AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = 1; AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(fixturePath));
        var root = document.RootElement;
        var reference = root.GetProperty(variant == 1 ? "float64_derived_positions" : "float64_operators");
        static double[] Values(JsonElement value) => value.EnumerateArray().Select(x => x.GetDouble()).ToArray();
        static int[] Ids(JsonElement value) => value.EnumerateArray().Select(x => x.GetInt32()).ToArray();
        void Check(string name, Tensor<double> actual, JsonElement expected, double tolerance = 2e-8)
        {
            var values = Values(expected);
            if (actual.Length != values.Length) throw new InvalidDataException(name + " shape differs.");
            double error = actual.ToArray().Zip(values, (a, b) => Math.Abs(a - b)).Max();
            Console.WriteLine($"Full original CosyVoice {variant} {name}: maximum error {error:R}.");
            if (!double.IsFinite(error) || error > tolerance) throw new InvalidDataException(name + " differs.");
        }
        var state = new WeightLoader().LoadWeights(Path.Combine(directory, "llm.pt"));
        double[] Read(string name, int[] shape)
        {
            if (!state.TryGetValue(name, out var value) || !value.Shape.ToArray().SequenceEqual(shape))
                throw new InvalidDataException("Missing or incompatible original LM tensor: " + name);
            return value.ToArray().Select(x => (double)x).ToArray();
        }
        var owner = new List<LayerBase<double>>(); var engine = AiDotNetEngine.Current;
        int[] text = Ids(root.GetProperty("text")), speech = Ids(root.GetProperty("speech"));
        using var scope = new NoGradScope<double>();
        if (variant == 1)
        {
            var configuration = new CosyVoiceLmConfiguration(51866, 4096, 512, 1024, 192,
                new WenetEncoderConfiguration(512, 1024, 16, 4096, 6, WenetInputLayer.Linear,
                    Swish: true, Conformer: true, StaticChunkSize: 1),
                new WenetEncoderConfiguration(1024, 1024, 16, 4096, 14, WenetInputLayer.LinearLegacy,
                    Swish: false, Conformer: false, StaticChunkSize: 1));
            var lm = new CosyVoiceLm<double>(engine, owner, configuration, new Random(0));
            lm.LoadTorchWeights(Read, ""); state.Clear(); GC.Collect();
            var speaker = new Tensor<double>(new[] { 192 }, new Vector<double>(Values(reference.GetProperty("xvector"))));
            var input = lm.Input(text, speech, speaker, false, new Random(0));
            Check("LM input", input, reference.GetProperty("inputs"));
            var hidden = lm.Llm.Forward(input, false, new Random(0));
            Check("hidden states", hidden, reference.GetProperty("hidden"));
            Check("vocabulary logits", lm.Head.Forward(hidden), reference.GetProperty("logits"));
            Check("causal objective", lm.Loss(text, speech, speaker, false, new Random(0)),
                JsonSerializer.SerializeToElement(new[] { reference.GetProperty("loss").GetDouble() }), 1e-10);
        }
        else
        {
            var lm = new CosyVoice2Lm<double>(engine, owner, new CosyVoice2LmConfiguration(new LlamaConfiguration
            { VocabSize = 151936, Hidden = 896, Layers = 24, Heads = 14, KeyValueHeads = 2, Intermediate = 4864,
                RmsEpsilon = 1e-6, RopeTheta = 1000000, AttentionBias = true, IncludeHead = false }, 6561), new Random(0));
            lm.LoadTorchWeights(Read, ""); state.Clear(); GC.Collect();
            var sequence = lm.TrainingSequence(text, speech, new Random(0));
            Check("LM input", sequence.Input, reference.GetProperty("inputs"));
            if (!sequence.Targets.SequenceEqual(Ids(reference.GetProperty("targets"))))
                throw new InvalidDataException("Original Qwen2LM target layout differs.");
            Check("vocabulary logits", lm.LogitsOf(sequence.Input), reference.GetProperty("logits"));
            Check("causal objective", lm.Loss(sequence),
                JsonSerializer.SerializeToElement(new[] { reference.GetProperty("loss").GetDouble() }), 1e-10);
        }
        Console.WriteLine($"PASS: complete original CosyVoice {variant} trained LM at explicitly matched operator precision.");
    }

    internal static void Frontend(int variant, string directory, string fixture)
    {
        AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = 1;
        AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(fixture));
        // Preserve actual frozen frontend graphs and output widths. Unused neural networks are small.
        CosyVoiceFamilyOptions options = variant == 1
            ? new CosyVoiceCloneOptions { TextEncoderDim = 16, LLMDim = 16, NumEncoderLayers = 1,
                NumLLMLayers = 1, NumHeads = 2, LmFeedForwardDim = 32 }
            : new CosyVoice2Options { LLMDim = 16, NumLLMLayers = 1, NumHeads = 2, KeyValueHeads = 1, LmFeedForwardDim = 32 };
        options.FlowDim = 16; options.FlowHeads = 2; options.FlowFeedForwardDim = 32; options.FlowEncoderLayers = 1;
        options.DecoderChannels = [16]; options.DecoderHeads = 2; options.DecoderHeadDim = 8;
        options.DecoderBlocks = options.DecoderMidBlocks = 1;
        options.HiftBaseChannels = 16; options.HiftHarmonics = 2;
        options.SpeechTokenizerState = variant == 1 ? 16 : 64;
        options.SpeechTokenizerHeads = 1; options.SpeechTokenizerLayers = 1;
        options.SpeakerEncoderChannels = 8; options.SpeakerEncoderGrowthRate = 4;
        options.SpeechTokenizerPath = Path.Combine(directory, $"speech_tokenizer_v{variant}.onnx");
        options.SpeakerEncoderOnnxPath = Path.Combine(directory, "campplus.onnx");
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Generative,
            inputSize: 1, outputSize: 1);
        using CosyVoiceBase<double> model = variant == 1 ? new CosyVoiceClone<double>(architecture, (CosyVoiceCloneOptions)options)
            : new CosyVoice2<double>(architecture, (CosyVoice2Options)options);
        model.DisableAutoStreaming(); model.SetTrainingMode(false);
        static double[] Values(JsonElement value) => value.EnumerateArray().Select(x => x.GetDouble()).ToArray();
        static void Check(string label, double[] actual, double[] expected, double tolerance)
        {
            if (actual.Length != expected.Length) throw new InvalidDataException(label + " shape differs.");
            double error = actual.Zip(expected, (a, b) => Math.Abs(a - b)).Max();
            Console.WriteLine($"{label}: maximum error {error:R}.");
            if (!double.IsFinite(error) || error > tolerance) throw new InvalidDataException(label + " differs.");
        }
        foreach (var item in document.RootElement.GetProperty("cases").EnumerateArray())
        {
            var samples = Values(item.GetProperty("audio")); int rate = item.GetProperty("rate").GetInt32();
            var audio = new Tensor<double>(new[] { samples.Length }, new Vector<double>(samples));
            var codes = model.SpeechTokens(audio, rate);
            if (!codes.SequenceEqual(item.GetProperty("tokens").EnumerateArray().Select(x => x.GetInt32())))
                throw new InvalidDataException("Original frozen speech tokenizer codes differ.");
            Console.WriteLine($"CosyVoice {variant} / {rate} Hz: all {codes.Length} speech codes exact.");
            var speaker = model.SpeakerEmbedding(audio, rate).ToArray();
            Check("Original CAM++ with matched double frontend / different ORT versions", speaker,
                Values(item.GetProperty("speaker_float64_buffers")), 2e-5);
            var voice = model.CreateVoice(audio, rate, "Hello, research world!");
            if (!voice.ReferenceCodecTokens!.ToArray().SequenceEqual(Values(item.GetProperty("voice_tokens"))) ||
                !voice.ReferenceTokens!.ToArray().SequenceEqual(Values(item.GetProperty("voice_text_tokens"))))
                throw new InvalidDataException("Original zero-shot transcript or trimmed speech codes differ.");
            if (voice.Reference!.Shape[0] != item.GetProperty("voice_mel_frames").GetInt32() || voice.Reference.Shape[1] != 80)
                throw new InvalidDataException("Original zero-shot mel shape differs.");
            Check("Original Matcha mel with double derived buffers", voice.Reference.ToArray(),
                Values(item.GetProperty("voice_mel_float64_buffers")), 2e-8);
            Console.WriteLine("Public CreateVoice: exact original transcript/codec IDs and zero-shot feature alignment.");
        }
        Console.WriteLine($"PASS: CosyVoice {variant} original frozen frontend and public zero-shot prompt.");
    }
}

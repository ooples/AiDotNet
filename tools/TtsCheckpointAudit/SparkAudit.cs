using System.Reflection;
using System.Text.Json;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;

internal static class SparkAudit
{
    internal static void Run(string directory, string fixture)
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
            if (actual.Length != values.Length) throw new InvalidDataException($"{label}: lengths {actual.Length} versus {values.Length}.");
            double error = actual.ToArray().Zip(values, (a,b) => Math.Abs(a-b)).Max();
            Console.WriteLine($"{label}: maximum error {error:R}.");
            if (!double.IsFinite(error) || error > tolerance) failures.Add($"{label}: {error:R} exceeds {tolerance:R}");
        }
        Console.WriteLine($"Spark original checkpoint {root.GetProperty("signature").GetProperty("revision").GetString()}.");
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
        using var model = new SparkTTS<double>(architecture, new SparkTTSOptions());
        model.DisableAutoStreaming();
        foreach (var item in root.GetProperty("tokenizer_cases").EnumerateArray())
        {
            var actual = model.TextToTokens(item.GetProperty("text").GetString()!).ToArray();
            var expected = Values(item.GetProperty("ids"));
            if (!actual.SequenceEqual(expected)) throw new InvalidDataException("Original Spark tokenizer IDs differ.");
        }
        Console.WriteLine("Original Spark tokenizer IDs passed; importing all three original checkpoints.");
        model.LoadAuthorWeights(Path.Combine(directory,"LLM","model.safetensors"),
            Path.Combine(directory,"BiCodec","model.safetensors"),
            Path.Combine(directory,"wav2vec2-large-xlsr-53","pytorch_model.bin"));
        model.SetTrainingMode(false);
        GC.Collect(); GC.WaitForPendingFinalizers();
        T Field<T>(string name) => (T)typeof(SparkTTS<double>).GetField(name,BindingFlags.Instance|BindingFlags.NonPublic)!.GetValue(model)!;
        var lm=Field<SparkTokenLanguageModel<double>>("_languageModel");
        var bi=Field<SparkBiCodec<double>>("_biCodec");
        var expectedLm=root.GetProperty("llm");
        var ids=expectedLm.GetProperty("ids").EnumerateArray().Select(x=>x.GetInt32()).ToArray();
        var logits=lm.Logits(ids).ToArray();
        var last=new Tensor<double>(new[]{166000},new Vector<double>(logits.Skip(logits.Length-166000).ToArray()));
        foreach(string precision in new[]{"float64","float32"})
            Check($"Qwen last-token logits / {precision}",last,expectedLm.GetProperty(precision).GetProperty("logits"),2e-4);
        int best=Array.IndexOf(last.ToArray(),last.ToArray().Max());
        if(best!=expectedLm.GetProperty("float64").GetProperty("argmax").GetInt32()) failures.Add("Qwen argmax differs.");
        var xlsr=root.GetProperty("xlsr");
        int frames=xlsr.GetProperty("frames").GetInt32();
        var wave=Tensor(xlsr.GetProperty("audio"),16000);
        var mixed=bi.ExtractXlsrFeatures(wave).ToArray();
        var mixedTensor=new Tensor<double>(new[]{1,frames,1024},new Vector<double>(mixed));
        var crossPrecision=Values(xlsr.GetProperty("features"));
        Console.WriteLine($"XLSR native float64 versus original float32: maximum error {mixed.Zip(crossPrecision,(a,b)=>Math.Abs(a-b)).Max():R}; reported precision diagnostic, not a float64 equivalence assertion.");
        var xlsrDouble=root.GetProperty("xlsr_float64");
        Check("Original XLSR float64 normalization",bi.NormalizeXlsrWaveform(wave),xlsrDouble.GetProperty("normalized"),2e-13);
        Check("XLSR layer 11/14/16 mean / original float64",mixedTensor,xlsrDouble.GetProperty("features"),2e-8);
        var features=Tensor(xlsr.GetProperty("features"),1,frames,1024).Transpose(new[]{0,2,1});
        var encoded=bi.EncodeFeatures(features);
        var reference=root.GetProperty("bicodec").GetProperty("float64");
        Check("BiCodec encoder, author XLSR features",encoded,reference.GetProperty("encoded"),2e-4);
        var melObject=typeof(SparkSpeakerTokenizer<double>).GetField("_mel",BindingFlags.Instance|BindingFlags.NonPublic)!.GetValue(bi.Speaker)!;
        int referenceSamples=root.GetProperty("bicodec").GetProperty("reference_samples").GetInt32();
        var repeated=new double[referenceSamples];var raw=wave.ToArray();
        for(int i=0;i<repeated.Length;i++) repeated[i]=raw[i%raw.Length];
        var mel=(Tensor<double>)melObject.GetType().GetMethod("Forward")!.Invoke(melObject,new object?[]{new Tensor<double>(new[]{referenceSamples},new Vector<double>(repeated)),null,null})!;
        Check("BiCodec raw Slaney magnitude mel / author upcast float32 buffers",mel,reference.GetProperty("mel"),2e-6);
        Check("BiCodec raw Slaney magnitude mel / original float64 buffers",mel,root.GetProperty("mel_float64_buffers").GetProperty("mel"),2e-8);
        var speaker=bi.Speaker.Encode(wave);
        Check("BiCodec global IDs",speaker.GlobalTokens,reference.GetProperty("global_tokens"),0);
        Check("BiCodec speaker vector",speaker.DVector,reference.GetProperty("speaker"),2e-4);
        var semantic=Tensor(reference.GetProperty("semantic"),reference.GetProperty("semantic").GetArrayLength());
        var global=Tensor(reference.GetProperty("global_tokens"),32);
        var codes=bi.SemanticQuantizer.Encode(encoded);
        var expectedCodes=Values(reference.GetProperty("semantic"));
        for(int i=0;i<expectedCodes.Length;i++)
            if(codes[0,i]!=expectedCodes[i]) failures.Add($"Semantic code differs at {i}.");
        var authorCodes=new int[1,expectedCodes.Length];
        for(int i=0;i<expectedCodes.Length;i++) authorCodes[0,i]=(int)expectedCodes[i];
        var quantized=bi.SemanticQuantizer.Decode(authorCodes);
        var authorSpeaker=bi.Speaker.Decode(global);
        var prenet=bi.PrenetFeatures(quantized,authorSpeaker);
        Check("BiCodec prenet",prenet,reference.GetProperty("prenet"),2e-4);
        Check("BiCodec postnet",bi.PostnetFeatures(prenet),reference.GetProperty("postnet"),2e-4);
        Check("BiCodec decoded waveform, author codes",bi.Decode(semantic,global),reference.GetProperty("waveform"),2e-4);
        var nativeFeatures=new Tensor<double>(new[]{1,frames,1024},new Vector<double>(mixed)).Transpose(new[]{0,2,1});
        var nativeCodes=bi.SemanticQuantizer.Encode(bi.EncodeFeatures(nativeFeatures));
        var nativeSemantic=new double[expectedCodes.Length];
        for(int i=0;i<nativeSemantic.Length;i++)
        {
            nativeSemantic[i]=nativeCodes[0,i];
            if(nativeSemantic[i]!=expectedCodes[i]) failures.Add($"Native-waveform semantic code differs at {i}.");
        }
        Check("BiCodec decoded waveform, native-waveform codes",bi.Decode(new Tensor<double>(new[]{nativeSemantic.Length},new Vector<double>(nativeSemantic)),speaker.GlobalTokens),reference.GetProperty("waveform"),2e-4);
        if(failures.Count>0) throw new InvalidDataException(string.Join(Environment.NewLine,failures));
        Console.WriteLine("Original Spark full-checkpoint audit passed.");
    }
}

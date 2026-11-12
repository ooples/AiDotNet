using System.Text.Json;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;

internal static class CosyVoice3Audit
{
    internal static void Run(string directory,string fixture)
    {
        AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism=1;
        AiDotNetEngine.ResetToCpu();
        using var document=JsonDocument.Parse(File.ReadAllText(fixture));
        var root=document.RootElement;
        static double[] Values(JsonElement x)=>x.EnumerateArray().Select(v=>v.GetDouble()).ToArray();
        static int[] Ids(JsonElement x)=>x.EnumerateArray().Select(v=>v.GetInt32()).ToArray();
        var failures=new List<string>();
        void Check(string label,Tensor<double> actual,JsonElement target,double tolerance=2e-4)
        {
            var values=Values(target);
            if(actual.Length!=values.Length)throw new InvalidDataException(label+": shape differs.");
            double error=actual.ToArray().Zip(values,(a,b)=>Math.Abs(a-b)).Max();
            Console.WriteLine($"{label}: maximum error {error:R}.");
            if(!double.IsFinite(error)||error>tolerance)failures.Add($"{label}: {error:R} exceeds {tolerance:R}.");
        }
        Console.WriteLine($"CosyVoice3 original LM checkpoint {root.GetProperty("revision").GetString()}.");
        foreach(var item in root.GetProperty("tokenizer_cases").EnumerateArray())
        {
            var actual=AiDotNet.Tokenization.Algorithms.QwenTokenizer.CosyVoice3.Encode(item.GetProperty("text").GetString()!);
            var expected=Ids(item.GetProperty("ids"));
            if(!actual.SequenceEqual(expected))throw new InvalidDataException($"Original CosyVoice3 tokenizer IDs differ for {item.GetProperty("text").GetString()}: expected [{string.Join(",",expected)}], native [{string.Join(",",actual)}].");
        }
        using var model=new CosyVoice3<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression,inputSize:1,outputSize:1),new CosyVoice3Options());
        model.DisableAutoStreaming();
        foreach(var item in root.GetProperty("tokenizer_cases").EnumerateArray())
            if(!model.Tokenize(item.GetProperty("text").GetString()!).SequenceEqual(Ids(item.GetProperty("ids"))))
                throw new InvalidDataException("Original CosyVoice3 tokenizer IDs differ.");
        Console.WriteLine("Original tokenizer IDs passed; importing all original PyTorch/ONNX components through LoadPretrainedWeights.");
        var lm=model.LanguageModel!;
        model.LoadPretrainedWeights(directory);
        model.SetTrainingMode(false);GC.Collect();GC.WaitForPendingFinalizers();
        Tensor<double> Embed(int[] ids)
        {
            var matrix=new int[1,ids.Length];for(int i=0;i<ids.Length;i++)matrix[0,i]=ids[i];
            return AiDotNetEngine.Current.Reshape(lm.Qwen.Embed(matrix),new[]{ids.Length,lm.Configuration.Qwen.Hidden});
        }
        var sequence=lm.TrainingSequence(Embed(Ids(root.GetProperty("text"))),Ids(root.GetProperty("speech")),
            new Random(1),prefix:Embed(Ids(root.GetProperty("instruct"))),bistream:false);
        if(!sequence.Targets.SequenceEqual(Ids(root.GetProperty("float64").GetProperty("targets"))))
            throw new InvalidDataException("Original CosyVoice3 causal loss targets differ.");
        var logits=lm.LogitsOf(sequence.Input);
        var loss=lm.Loss(sequence);
        foreach(string precision in new[]{"float64","float32"})
        {
            var expected=root.GetProperty(precision);
            Check($"CosyVoice3 pretrained input / {precision}",sequence.Input,expected.GetProperty("inputs"),1e-12);
            Check($"CosyVoice3 pretrained logits / {precision}",logits,expected.GetProperty("logits"));
            double error=Math.Abs(loss[0]-expected.GetProperty("loss").GetDouble());
            Console.WriteLine($"CosyVoice3 pretrained loss / {precision}: error {error:R}.");
            if(!double.IsFinite(error)||error>2e-5)failures.Add($"CosyVoice3 loss {precision} differs by {error:R}.");
        }
        if(failures.Count>0)throw new InvalidDataException(string.Join(Environment.NewLine,failures));
        string frontend=Path.Combine(Path.GetDirectoryName(Path.GetFullPath(fixture))!,"cosyvoice3-frontend-pretrained-reference.json");
        if(File.Exists(frontend))
        {
            using var voiceReference=JsonDocument.Parse(File.ReadAllText(frontend));
            CosyVoice3FrontendAudit.ValidateVoice(model,voiceReference.RootElement);
        }
        Console.WriteLine("Original CosyVoice3 full pretrained LM audit passed; flow and waveform rendering are separate checks.");
    }
}

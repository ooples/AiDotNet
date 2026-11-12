using System.Text.Json;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;

internal static class CosyVoice3FlowAudit
{
    internal static void Run(string directory,string fixture)
    {
        AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism=1;AiDotNetEngine.ResetToCpu();
        using var document=JsonDocument.Parse(File.ReadAllText(fixture));var root=document.RootElement;
        static double[] Values(JsonElement x)=>x.EnumerateArray().Select(v=>v.GetDouble()).ToArray();
        static int[] Ids(JsonElement x)=>x.EnumerateArray().Select(v=>v.GetInt32()).ToArray();
        static Tensor<double> Tensor(JsonElement x,params int[] shape)=>new(shape,new Vector<double>(Values(x)));
        var failures=new List<string>();
        void Check(string label,Tensor<double> actual,JsonElement expected,double tolerance=2e-4)
        {
            var values=Values(expected);if(actual.Length!=values.Length)throw new InvalidDataException(label+": shape differs.");
            double error=actual.ToArray().Zip(values,(a,b)=>Math.Abs(a-b)).Max();Console.WriteLine($"{label}: maximum error {error:R}.");
            if(!double.IsFinite(error)||error>tolerance)failures.Add($"{label}: {error:R} exceeds {tolerance:R}.");
        }
        var layers=new List<LayerBase<double>>();
        var flow=new CosyVoice3Flow<double>(AiDotNetEngine.Current,layers,new CosyVoice3Options(),new Random(1));
        var state=new WeightLoader().LoadWeights(Path.Combine(directory,"flow.pt"));
        flow.LoadTorchWeights((name,shape)=>
        {
            if(!state.TryGetValue(name,out var value))throw new InvalidDataException("Missing original flow tensor "+name);
            // The shared DiT loader requests values with an empty shape and checks each layer's expected size itself.
            if(shape.Length>0&&!value.Shape.ToArray().SequenceEqual(shape))throw new InvalidDataException("Original flow tensor axes differ: "+name);
            return value.ToArray().Select(v=>(double)v).ToArray();
        });
        state.Clear();foreach(var layer in layers)layer.SetTrainingMode(false);GC.Collect();GC.WaitForPendingFinalizers();
        var tokens=Ids(root.GetProperty("tokens"));var prompt=Ids(root.GetProperty("prompt_tokens"));
        var speaker=Tensor(root.GetProperty("speaker"),192);var mel=Tensor(root.GetProperty("prompt_mel"),4,80);
        var noise=Tensor(root.GetProperty("noise"),80,56);
        var encoded=flow.Encode(prompt.Concat(tokens).ToArray());var projected=flow.ProjectSpeaker(speaker);
        foreach(string precision in new[]{"float64","float32"})
        {
            var expected=root.GetProperty(precision);
            Check("CosyVoice3 pretrained flow encoder / "+precision,encoded,expected.GetProperty("encoded"),precision=="float64"?2e-10:2e-5);
            Check("CosyVoice3 pretrained speaker projection / "+precision,projected,expected.GetProperty("projected_speaker"),precision=="float64"?2e-12:2e-6);
        }
        foreach(bool streaming in new[]{false,true})
        {
            var result=flow.Inference(tokens,prompt,mel,speaker,noise,streaming);
            string mode=streaming?"true":"false";
            var originalDouble=root.GetProperty("float64").GetProperty("outputs").GetProperty(mode);
            var untrimmed=Values(originalDouble.GetProperty("untrimmed"));
            var trimmed=Enumerable.Range(0,80).SelectMany(channel=>untrimmed.Skip(channel*56+4).Take(52)).ToArray();
            Check($"CosyVoice3 full pretrained ten-step flow / streaming {streaming} / original uncast float64 with upcast float32 rotary buffer",result,
                JsonSerializer.SerializeToElement(trimmed),2e-6);
            var matchedUntrimmed=Values(root.GetProperty("float64_rotary_buffer").GetProperty("outputs").GetProperty(mode).GetProperty("untrimmed"));
            var matchedTrimmed=Enumerable.Range(0,80).SelectMany(channel=>matchedUntrimmed.Skip(channel*56+4).Take(52)).ToArray();
            Check($"CosyVoice3 full pretrained ten-step flow / streaming {streaming} / original methods with float64 rotary buffer",result,
                JsonSerializer.SerializeToElement(matchedTrimmed),2e-8);
            Check($"CosyVoice3 full pretrained ten-step flow / streaming {streaming} / author float64 final float cast",result,
                originalDouble.GetProperty("output"),2e-6);
            var originalFloat=Values(root.GetProperty("float32").GetProperty("outputs").GetProperty(mode).GetProperty("output"));
            double delta=result.ToArray().Zip(originalFloat,(a,b)=>Math.Abs(a-b)).Max();
            Console.WriteLine($"CosyVoice3 native float64 versus original float32 / streaming {streaming}: maximum delta {delta:R}; cross-precision diagnostic, not a float64 equivalence assertion.");
        }
        if(failures.Count>0)throw new InvalidDataException(string.Join(Environment.NewLine,failures));
        Console.WriteLine("Original CosyVoice3 full pretrained flow audit passed.");
    }
}

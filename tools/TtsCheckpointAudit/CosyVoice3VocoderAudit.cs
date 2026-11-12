using System.Text.Json;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.Vocoders;

internal static class CosyVoice3VocoderAudit
{
    internal static void Run(string directory,string fixture)
    {
        AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism=1;AiDotNetEngine.ResetToCpu();
        using var document=JsonDocument.Parse(File.ReadAllText(fixture));var root=document.RootElement;
        static double[] Values(JsonElement x)=>x.EnumerateArray().Select(v=>v.GetDouble()).ToArray();
        static Tensor<double> Tensor(JsonElement x,params int[] shape)=>new(shape,new Vector<double>(Values(x)));
        var failures=new List<string>();
        void Check(string label,Tensor<double> actual,JsonElement target,double tolerance)
        {
            var expected=Values(target);if(actual.Length!=expected.Length)throw new InvalidDataException(label+": lengths differ.");
            double error=actual.ToArray().Zip(expected,(a,b)=>Math.Abs(a-b)).Max();Console.WriteLine($"{label}: maximum error {error:R}.");
            if(!double.IsFinite(error)||error>tolerance)failures.Add($"{label}: {error:R} exceeds {tolerance:R}.");
        }
        var layers=new List<LayerBase<double>>();
        var configuration=new HiftConfiguration(80,512,8,24000,.1,.003,10,new[]{8,5,3},new[]{16,11,7},16,4,
            new[]{3,7,11},new[]{new[]{1,3,5},new[]{1,3,5},new[]{1,3,5}},new[]{7,7,11},
            new[]{new[]{1,3,5},new[]{1,3,5},new[]{1,3,5}},.1,.99,512,Causal:true,ConvPreLookRight:4);
        var vocoder=new HiftGenerator<double>(AiDotNetEngine.Current,layers,configuration,new Random(1));
        var state=new WeightLoader().LoadWeights(Path.Combine(directory,"hift.pt"));
        vocoder.LoadTorchWeights((name,shape)=>
        {
            if(!state.TryGetValue(name,out var value))throw new InvalidDataException("Missing original CausalHiFT tensor "+name);
            if(!value.Shape.ToArray().SequenceEqual(shape))throw new InvalidDataException("Original CausalHiFT axes differ: "+name);
            return value.ToArray().Select(v=>(double)v).ToArray();
        });
        state.Clear();foreach(var layer in layers)layer.SetTrainingMode(false);
        int frames=root.GetProperty("frames").GetInt32();var mel=Tensor(root.GetProperty("mel"),1,80,frames);
        var expected=root.GetProperty("float64");
        Check("Original pretrained CausalHiFT F0",vocoder.PredictF0(mel),expected.GetProperty("f0"),2e-8);
        Check("Original pretrained CausalHiFT waveform / identical author excitation",vocoder.Decode(mel,
            Tensor(expected.GetProperty("source"),1,1,frames*480)),expected.GetProperty("waveform"),2e-6);
        if(failures.Count>0)throw new InvalidDataException(string.Join(Environment.NewLine,failures));
        Console.WriteLine("Original pretrained CosyVoice3 CausalHiFT audit passed.");
    }
}

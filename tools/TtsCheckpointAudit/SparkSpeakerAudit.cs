using System.Reflection;
using System.Text.Json;
using AiDotNet.Audio;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Latest;
using AiDotNet.TextToSpeech.Vocoders;

internal static class SparkSpeakerAudit
{
    internal static void Run(string directory,string fixture)
    {
        AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism=1;
        AiDotNetEngine.ResetToCpu();
        using var document=JsonDocument.Parse(File.ReadAllText(fixture));
        var root=document.RootElement;
        var expected=root.GetProperty("bicodec").GetProperty("float64");
        static double[] Values(JsonElement x)=>x.EnumerateArray().Select(v=>v.GetDouble()).ToArray();
        static Tensor<double> Tensor(JsonElement x,params int[] shape)=>new(shape,new Vector<double>(Values(x)));
        var failures=new List<string>();
        void Check(string label,Tensor<double> actual,JsonElement target,double tolerance=2e-8)
        {
            var values=Values(target);
            if(actual.Length!=values.Length)throw new InvalidDataException(label+": shape differs.");
            double error=actual.ToArray().Zip(values,(a,b)=>Math.Abs(a-b)).Max();
            Console.WriteLine($"{label}: {error:R}");
            if(!double.IsFinite(error)||error>tolerance)failures.Add($"{label}: {error:R} exceeds {tolerance:R}.");
        }
        var layers=new List<LayerBase<double>>();
        var speaker=new SparkSpeakerTokenizer<double>(AiDotNetEngine.Current,layers,new SparkTTSOptions(),new Random(1));
        speaker.LoadTorchWeights(SparkTTS<double>.AuthorStateReader(new WeightLoader().LoadWeights(Path.Combine(directory,"BiCodec","model.safetensors"))));
        foreach(var layer in layers)layer.SetTrainingMode(false);
        T Field<T>(string name)=>(T)typeof(SparkSpeakerTokenizer<double>).GetField(name,BindingFlags.Instance|BindingFlags.NonPublic)!.GetValue(speaker)!;
        var melTransform=Field<CenteredLogMel<double>>("_mel");
        var ecapa=Field<EcapaTdnnBackbone<double>>("_ecapa");
        var perceiver=Field<IndexTtsPerceiver<double>>("_perceiver");
        int samples=root.GetProperty("bicodec").GetProperty("reference_samples").GetInt32();
        var raw=Values(root.GetProperty("xlsr").GetProperty("audio"));var repeated=new double[samples];
        for(int i=0;i<samples;i++)repeated[i]=raw[i%raw.Length];
        var mel=melTransform.Forward(new Tensor<double>(new[]{samples},new Vector<double>(repeated)));
        Check("Raw mel / author upcast float32 buffers",mel,expected.GetProperty("mel"),2e-6);
        Check("Raw mel / original float64 buffers",mel,root.GetProperty("mel_float64_buffers").GetProperty("mel"));
        int frames=mel.Shape[2];
        var features=ecapa.ForwardFrameFeatures(Tensor(expected.GetProperty("mel"),1,128,frames));
        Check("ECAPA on author mel",features,expected.GetProperty("speaker_features"));
        var nativeFeatures=ecapa.ForwardFrameFeatures(mel);
        Check("ECAPA on native mel / author upcast float32 mel buffers",nativeFeatures,expected.GetProperty("speaker_features"),3e-5);
        var context=AiDotNetEngine.Current.Reshape(Tensor(expected.GetProperty("speaker_features"),1,1536,frames).Transpose(new[]{0,2,1}),new[]{frames,1536});
        Check("Perceiver on author ECAPA",perceiver.Forward(context),expected.GetProperty("speaker_latents"));
        Check("FSQ on author ECAPA",speaker.EncodeContext(context).GlobalTokens,expected.GetProperty("global_tokens"),0);
        Check("Complete global IDs",speaker.Encode(new Tensor<double>(new[]{raw.Length},new Vector<double>(raw))).GlobalTokens,expected.GetProperty("global_tokens"),0);
        Check("Speaker decoder on author IDs",speaker.Decode(Tensor(expected.GetProperty("global_tokens"),32)),expected.GetProperty("speaker"));
        if(failures.Count>0)throw new InvalidDataException(string.Join(Environment.NewLine,failures));
    }
}

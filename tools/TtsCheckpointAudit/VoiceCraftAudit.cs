using System.Reflection;
using System.Text.Json;
using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;

internal static class VoiceCraftAudit
{
    internal static void Run(string directory,string fixture,string codecFixture)
    {
        AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism=1;AiDotNetEngine.ResetToCpu();
        using var document=JsonDocument.Parse(File.ReadAllText(fixture));var root=document.RootElement;
        using var codecDocument=JsonDocument.Parse(File.ReadAllText(codecFixture));
        static double[] Values(JsonElement x)=>x.EnumerateArray().Select(v=>v.GetDouble()).ToArray();
        static int[] Ids(JsonElement x)=>x.EnumerateArray().Select(v=>v.GetInt32()).ToArray();
        var failures=new List<string>();
        void Check(string label,Tensor<double> actual,JsonElement target,double? tolerance)
        {
            var expected=Values(target);if(actual.Length!=expected.Length)throw new InvalidDataException($"{label}: lengths {actual.Length}/{expected.Length} differ.");
            double error=actual.ToArray().Zip(expected,(a,b)=>Math.Abs(a-b)).Max();Console.WriteLine($"{label}: maximum error {error:R}.");
            if(!double.IsFinite(error)||(tolerance.HasValue&&error>tolerance.Value))
                failures.Add($"{label}: {error:R} exceeds {tolerance:R}.");
        }
        foreach(var phone in root.GetProperty("phn2num").EnumerateObject())
        {
            var ids=VoiceCraftPhoneTokenizer.Tokenize("checkpoint inventory",1,phonemizer:_=>new[]{phone.Name});
            if(ids.Length!=1||ids[0]!=phone.Value.GetInt32())throw new InvalidDataException("Original VoiceCraft phone inventory differs for "+phone.Name);
        }
        Console.WriteLine("All original VoiceCraft phone inventory IDs match.");
        Console.WriteLine($"Creating complete native VoiceCraft 830M and original GigaSpeech codec, checkpoint {root.GetProperty("revision").GetString()}.");
        using var model=new VoiceCraft<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression,inputSize:1,outputSize:1),VoiceCraftOptions.Official830M());
        model.DisableAutoStreaming();
        T Field<T>(string name)=>(T)typeof(VoiceCraft<double>).GetField(name,BindingFlags.Instance|BindingFlags.NonPublic)!.GetValue(model)!;
        var codec=Field<EnCodec<double>>("_codecNetwork");codec.DisableAutoStreaming();
        Console.WriteLine("Loading original wrapped giga830M.pth and best_state.model codec through public LoadAuthorWeights.");
        model.LoadAuthorWeights(Path.Combine(directory,"giga830M.pth"),Path.Combine(directory,"encodec_4cb2048_giga.th"));
        model.SetTrainingMode(false);codec.SetTrainingMode(false);GC.Collect();GC.WaitForPendingFinalizers();
        var network=Field<VoiceCraftNetwork<double>>("_network");
        Console.WriteLine("The original float64 run retains float32 positional tables; its cross-precision delta is reported separately. The unchanged author network with float64 non-learned positional tables must meet 2e-8.");
        int caseIndex=0;
        foreach(var item in root.GetProperty("cases").EnumerateArray())
        {
            var text=Ids(item.GetProperty("text"));var rows=item.GetProperty("rows").EnumerateArray().Select(Ids).ToArray();
            var logits=network.Forward(text,rows,Ids(item.GetProperty("masks")));
            foreach(string precision in new[]{"float64_position_buffers","float64","float32"})
                for(int q=0;q<4;q++)Check($"Full VoiceCraft case {caseIndex}, head {q} / {precision}",logits[q],
                    item.GetProperty(precision).GetProperty("logits")[q],
                    precision=="float64_position_buffers"?2e-8:precision=="float32"?2e-4:null);
            for(int q=0;q<4;q++)
            {
                var expected=Ids(item.GetProperty("float64").GetProperty("argmax")[q]);
                for(int t=0;t<expected.Length;t++)
                {
                    int best=0;for(int v=1;v<2052;v++)if(logits[q][t,v]>logits[q][t,best])best=v;
                    if(best!=expected[t])failures.Add($"VoiceCraft head {q} argmax differs at row {t}.");
                }
            }
            caseIndex++;
        }
        foreach(var item in codecDocument.RootElement.GetProperty("cases").EnumerateArray())
        {
            var audio=Values(item.GetProperty("audio"));var expected=item.GetProperty("float64");
            var codes=codec.Encode(new Tensor<double>(new[]{audio.Length},new Vector<double>(audio)));
            var expectedCodes=Ids(expected.GetProperty("codes"));
            if(codes.Length!=expectedCodes.Length)throw new InvalidDataException("Original GigaSpeech codec token shape differs.");
            for(int q=0;q<codes.GetLength(0);q++)for(int t=0;t<codes.GetLength(1);t++)
                if(codes[q,t]!=expectedCodes[q*codes.GetLength(1)+t])failures.Add($"Codec {audio.Length} samples, code ({q},{t}) differs.");
            Check($"Original GigaSpeech codec round-trip / {audio.Length} samples",codec.Decode(codes),expected.GetProperty("waveform"),2e-8);
            Console.WriteLine($"Original GigaSpeech codec {audio.Length} samples: compared all {codes.Length} code IDs.");
        }
        if(failures.Count>0)throw new InvalidDataException(string.Join(Environment.NewLine,failures));
        Console.WriteLine("Original full VoiceCraft Transformer and GigaSpeech codec checkpoint audit passed.");
    }
}

using System.Text.Json;
using System.Reflection;
using AiDotNet.Audio.Features;
using AiDotNet.Onnx;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;

internal static class CosyVoice3FrontendAudit
{
    internal static void Run(string directory,string fixture)
    {
        AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism=1;AiDotNetEngine.ResetToCpu();
        using var document=JsonDocument.Parse(File.ReadAllText(fixture));
        // Only the actual frozen prompt tokenizer is exercised here. Unused trainable networks are narrowed.
        var options=new CosyVoice3Options{LLMDim=16,NumLLMLayers=1,NumHeads=2,KeyValueHeads=1,LmFeedForwardDim=32,
            DiTDim=32,DiTDepth=1,DiTHeads=2,DiTHeadDim=16,PreLookaheadChannels=32,HiftBaseChannels=16,
            SpeakerEncoderChannels=16,SpeakerEncoderGrowthRate=8,DropoutRate=0,
            SpeechTokenizerPath=Path.Combine(directory,"speech_tokenizer_v3.onnx"),
            SpeakerEncoderOnnxPath=Path.Combine(directory,"campplus.onnx")};
        using var model=new CosyVoice3<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression,inputSize:1,outputSize:1),options);
        model.DisableAutoStreaming();model.SetTrainingMode(false);
        using var directSpeaker=new OnnxModel<double>(Path.Combine(directory,"campplus.onnx"));
        var failures=new List<string>();
        foreach(var item in document.RootElement.GetProperty("cases").EnumerateArray())
        {
            var audio=item.GetProperty("audio").EnumerateArray().Select(v=>v.GetDouble()).ToArray();
            int rate=item.GetProperty("rate").GetInt32();
            var expected=item.GetProperty("tokens").EnumerateArray().Select(v=>v.GetInt32()).ToArray();
            var actual=model.SpeechTokens(new Tensor<double>(new[]{audio.Length},new Vector<double>(audio)),rate);
            if(!actual.SequenceEqual(expected))throw new InvalidDataException($"Original CosyVoice3 frozen tokenizer differs at {rate} Hz / {audio.Length} samples: expected [{string.Join(",",expected)}], native [{string.Join(",",actual)}].");
            Console.WriteLine($"Original CosyVoice3 frozen tokenizer {rate} Hz / {audio.Length} samples: all {actual.Length} IDs exact.");
            var speaker=model.SpeakerEmbedding(new Tensor<double>(new[]{audio.Length},new Vector<double>(audio)),rate).ToArray();
            var expectedSpeaker=item.GetProperty("speaker").EnumerateArray().Select(v=>v.GetDouble()).ToArray();
            double speakerError=speaker.Zip(expectedSpeaker,(a,b)=>Math.Abs(a-b)).Max();
            Console.WriteLine($"Original CosyVoice3 public CAM++ frontend {rate} Hz / {audio.Length} samples: maximum error {speakerError:R}.");
            Console.WriteLine("The preceding delta retains the original float32 frontend; checking independently executed float64 frontend next.");
            var precise=item.GetProperty("speaker_float64_buffers").EnumerateArray().Select(v=>v.GetDouble()).ToArray();
            double preciseError=speaker.Zip(precise,(a,b)=>Math.Abs(a-b)).Max();
            Console.WriteLine($"Original CAM++ ONNX / float64 frontend: maximum error {preciseError:R}.");
            if(!double.IsFinite(preciseError)||preciseError>2e-5)failures.Add("Original CAM++ prompt embedding differs with matched frontend precision.");
            var resampled=(Tensor<double>)typeof(CosyVoiceBase<double>).GetMethod("ResampleAudio",BindingFlags.Instance|BindingFlags.NonPublic)!
                .Invoke(model,new object[]{new Tensor<double>(new[]{audio.Length},new Vector<double>(audio)),rate,16000})!;
            var fbank=new KaldiFbank<double>(AiDotNetEngine.Current,16000,80,WindowType.Povey).Forward(resampled);
            fbank=AiDotNetEngine.Current.TensorSubtract(fbank,AiDotNetEngine.Current.ReduceMean(fbank,new[]{0},true));
            var expectedFbank=item.GetProperty("fbank_float64_buffers").EnumerateArray().Select(v=>v.GetDouble()).ToArray();
            double fbankError=fbank.ToArray().Zip(expectedFbank,(a,b)=>Math.Abs(a-b)).Max();
            Console.WriteLine($"Original Kaldi float64 frontend: maximum error {fbankError:R}.");
            if(!double.IsFinite(fbankError)||fbankError>2e-8)throw new InvalidDataException("Original Kaldi frontend differs.");
            var direct=directSpeaker.Run(new Dictionary<string,Tensor<double>>{
                [directSpeaker.Metadata.Inputs[0].Name]=new Tensor<double>(new[]{1,item.GetProperty("fbank_frames").GetInt32(),80},new Vector<double>(expectedFbank))}).Values.First().ToArray();
            Console.WriteLine($"Original ONNX with identical author features / native runtime versus author runtime: {direct.Zip(precise,(a,b)=>Math.Abs(a-b)).Max():R}.");
            Console.WriteLine($"Public speaker path versus direct original ONNX on identical-precision author features: {speaker.Zip(direct,(a,b)=>Math.Abs(a-b)).Max():R}.");
        }
        if(failures.Count>0)throw new InvalidDataException(string.Join(Environment.NewLine,failures));
        ValidateVoice(model,document.RootElement);
        Console.WriteLine("Original CosyVoice3 public frozen speech-tokenizer audit passed.");
    }

    internal static void ValidateVoice(CosyVoice3<double> model,JsonElement root)
    {
        foreach(var item in root.GetProperty("cases").EnumerateArray())
        {
            var raw=item.GetProperty("audio").EnumerateArray().Select(v=>v.GetDouble()).ToArray();
            var voice=model.CreateVoice(new Tensor<double>(new[]{raw.Length},new Vector<double>(raw)),item.GetProperty("rate").GetInt32(),"Hello, research world!");
            var expectedTokens=item.GetProperty("voice_tokens").EnumerateArray().Select(v=>v.GetDouble()).ToArray();
            var expectedText=item.GetProperty("voice_text_tokens").EnumerateArray().Select(v=>v.GetDouble()).ToArray();
            if(!voice.ReferenceCodecTokens!.ToArray().SequenceEqual(expectedTokens)||!voice.ReferenceTokens!.ToArray().SequenceEqual(expectedText))
                throw new InvalidDataException("Original zero-shot prompt trimming or transcript token IDs differ.");
            if(voice.Reference!.Shape[0]!=item.GetProperty("voice_mel_frames").GetInt32()||voice.Reference.Shape[1]!=80)
                throw new InvalidDataException("Original zero-shot prompt mel shape differs.");
            var expectedMel=item.GetProperty("voice_mel").EnumerateArray().Select(v=>v.GetDouble()).ToArray();
            double melError=voice.Reference.ToArray().Zip(expectedMel,(a,b)=>Math.Abs(a-b)).Max();
            Console.WriteLine($"Original frontend_zero_shot versus public CreateVoice: exact codec/text IDs and mel shape; float64 versus float32 mel delta {melError:R}.");
        }
    }
}

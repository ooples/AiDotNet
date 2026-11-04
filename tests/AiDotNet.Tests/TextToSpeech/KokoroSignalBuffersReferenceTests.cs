using System.Text.Json;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.EndToEnd;
using AiDotNet.TextToSpeech.StyleEmotion;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class KokoroSignalBuffersReferenceTests
{
    [Fact]
    public void ApproximateTextFrontend_RequiresExplicitOptIn_AndOptionCopyPreservesIt()
    {
        Assert.False(new KokoroOptions().UseNativeEnglishFallback);
        Assert.True(new KokoroOptions(new KokoroOptions { UseNativeEnglishFallback=true }).UseNativeEnglishFallback);
    }
    private static JsonDocument Reference() => JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
        "TextToSpeech","ReferenceData","kokoro_signal_buffers_reference.json")));
    private static double[] Values(JsonElement value) => value.EnumerateArray().Select(x => x.GetDouble()).ToArray();
    private static void Close(double[] actual,double[] expected,string name,double tolerance=1e-12)
    {
        Assert.Equal(expected.Length,actual.Length);
        double error=actual.Zip(expected,(a,b) => Math.Abs(a-b)).Max();
        Assert.True(!double.IsNaN(error) && !double.IsInfinity(error) && error<tolerance,name+" error "+error.ToString("R"));
    }

    [Fact]
    public void OriginalFloat32Window_MatchesRawSpectrum_WithoutPhaseCanonicalization()
    {
        AiDotNetEngine.ResetToCpu(); using var document=Reference(); var root=document.RootElement;
        var window=KokoroSignalBuffers.HannFloat32(); Assert.Equal(Values(root.GetProperty("window")),window);
        var signal=Values(root.GetProperty("signal")); var input=new Tensor<double>(new[] {signal.Length},new Vector<double>(signal));
        var result=new KokoroHarmonicStft<double>(window:window).Forward(input);
        Assert.Equal(root.GetProperty("spectrum_shape").EnumerateArray().Select(x => x.GetInt32()),result.Shape.ToArray());
        Close(result.ToArray(),Values(root.GetProperty("spectrum")),"raw magnitude/phase",2e-9);
        // A promoted window is a different source computation, not a precision fix.
        var promoted=new KokoroHarmonicStft<double>().Forward(input);
        Assert.True(promoted.ToArray().Zip(Values(root.GetProperty("spectrum")),(a,b) => Math.Abs(a-b)).Max()>2e-9);
    }

    [Fact]
    public void OriginalFloat32UvNoiseAmplitude_MatchesMixedVoicedSourceAndMergeGradients()
    {
        AiDotNetEngine.ResetToCpu(); using var document=Reference(); var root=document.RootElement;
        var engine=AiDotNetEngine.Current; var layers=new List<LayerBase<double>>();
        var source=new StyleTts2Source<double>(engine,layers,24000,300) { UseFloat32NoiseAmplitude=true };
        source.Load((name,shape) => Values(root.GetProperty(name.EndsWith("weight",StringComparison.Ordinal)?"weight":"bias")),"source");
        var draws=root.GetProperty("draws").EnumerateArray().ToArray();
        source.InitialPhases=Values(draws.Single(x => x.GetProperty("kind").GetString()=="uniform").GetProperty("values"));
        source.GaussianNoise=Values(draws.Single(x => x.GetProperty("kind").GetString()=="normal" && x.GetProperty("shape").EnumerateArray().Last().GetInt32()==9).GetProperty("values"));
        var pitchValues=Values(root.GetProperty("pitch")); var pitch=new Tensor<double>(new[] {1,pitchValues.Length},new Vector<double>(pitchValues));
        using var tape=new GradientTape<double>(); var harmonic=source.Forward(pitch,new Random(1));
        Close(harmonic.ToArray(),Values(root.GetProperty("harmonic")),"original mixed-precision source");
        var probe=new Tensor<double>(harmonic.Shape.ToArray(),new Vector<double>(Values(root.GetProperty("probe"))));
        var loss=engine.Reshape(engine.ReduceSum(engine.TensorMultiply(harmonic,probe),new[] {0,1,2},true),new[] {1});
        Close(loss.ToArray(),new[] {root.GetProperty("loss").GetDouble()},"source loss",2e-9);
        var parameters=new[] {source.Merge.GetWeights(),source.Merge.GetBiases()};
        var gradients=tape.ComputeGradients(loss,sources:parameters.Concat(new[] {pitch}).ToList());
        Close(gradients[parameters[0]].ToArray(),Values(root.GetProperty("weight_gradient")),"merge weights",2e-9);
        Close(gradients[parameters[1]].ToArray(),Values(root.GetProperty("bias_gradient")),"merge bias",2e-9);
        if(gradients.TryGetValue(pitch,out var grad)) Assert.All(grad.ToArray(),value => Assert.Equal(0,value));
    }
}

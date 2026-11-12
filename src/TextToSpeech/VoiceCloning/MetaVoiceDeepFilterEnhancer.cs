using AiDotNet.Helpers;
using AiDotNet.ModelLoading;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Original DeepFilterNet3 checkpoint and mono 48 kHz waveform path.</summary>
internal sealed class MetaVoiceDeepFilterEnhancer<T>
{
    internal MetaVoiceDeepFilterNet<T> Network { get; }
    private readonly MetaVoiceDeepFilterFrontend _frontend;
    private static readonly INumericOperations<T> Numeric = MathHelper.GetNumericOperations<T>();

    internal MetaVoiceDeepFilterEnhancer(string checkpoint)
    {
        Network = new MetaVoiceDeepFilterNet<T>();
        using (var file = TorchCheckpointFile.Open(checkpoint, strictAxes: true))
        {
            if (file.Names.Count != 133) throw new InvalidDataException("The released DeepFilterNet3 checkpoint requires exactly 133 consumed tensors.");
            Network.LoadTorch(file);
            file.RequireAllUsed("released DeepFilterNet3");
        }
        _frontend = new MetaVoiceDeepFilterFrontend(AiDotNetEngine.Current, Network.ErbWidths());
        Network.SetTrainingMode(false);
    }

    internal MetaVoiceDeepFilterEnhancer(MetaVoiceDeepFilterNet<T> network)
    {
        Network = network;
        _frontend = new MetaVoiceDeepFilterFrontend(AiDotNetEngine.Current, Network.ErbWidths());
        Network.SetTrainingMode(false);
    }

    internal Tensor<T> Enhance48k(Tensor<T> waveform)
    {
        if (waveform.Rank != 1 || waveform.Length == 0) throw new ArgumentException("A nonempty mono 48 kHz waveform is required.", nameof(waveform));
        using var noGrad = new NoGradScope<T>();
        var samples = waveform.ToArray().Select(x => (float)Numeric.ToDouble(x)).ToArray();
        var inputs = _frontend.Analyze(samples, pad: true);
        Tensor<T> Convert(Tensor<float> value) => new(value.Shape.ToArray(),
            new Vector<T>(value.ToArray().Select(x => Numeric.FromDouble(x))));
        var output = Network.Forward(Convert(inputs.Spectrum), Convert(inputs.Erb), Convert(inputs.Features)).Spectrum;
        var spectrum = new Tensor<float>(output.Shape.ToArray(), new Vector<float>(output.ToArray().Select(x => (float)Numeric.ToDouble(x))));
        var enhanced = _frontend.Synthesize(spectrum, samples.Length);
        return new Tensor<T>(new[] { enhanced.Length }, new Vector<T>(enhanced.Select(x => Numeric.FromDouble(x))));
    }
}

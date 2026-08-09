using System;
using AiDotNet.Audio.SourceSeparation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests;

/// <summary>
/// Splits MusicSourceSeparator's clone divergence between Clone()'s copy-on-write fast path and the
/// serialize/deserialize path they are both supposed to agree with.
/// </summary>
/// <remarks>
/// Established already: the clone's parameters are bit-identical (0/5892), it takes the same Demucs
/// forward path (identical activation keys), Predict is deterministic on one instance, and Encoder_0 —
/// the FIRST layer — already differs by 7.93e-01. NeuralNetworkBase.DeepCopy has a COW branch that
/// returns early before the serialize path runs, so if an explicit round-trip MATCHES while Clone does
/// not, the defect is in that branch rather than in any layer's restore.
/// </remarks>
public class MssCloneVsSerializeLocalization
{
    private readonly ITestOutputHelper _out;
    public MssCloneVsSerializeLocalization(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Localize()
    {
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 64, outputSize: 64);
        using var model = new MusicSourceSeparator<double>(arch);

        var rng = new Random(7);
        var input = new Tensor<double>([1, 64]);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() * 2 - 1;

        var baseline = model.Predict(input);

        static double Worst(Tensor<double> a, Tensor<double> b)
        {
            double w = 0;
            for (int i = 0; i < Math.Min(a.Length, b.Length); i++) w = Math.Max(w, Math.Abs(a[i] - b[i]));
            return w;
        }

        var cloned = (MusicSourceSeparator<double>)model.Clone();
        _out.WriteLine($"CLONE            worstDelta={Worst(baseline, cloned.Predict(input)):E3}");

        byte[] bytes = model.Serialize();
        using var restored = new MusicSourceSeparator<double>(arch);
        restored.Deserialize(bytes);
        _out.WriteLine($"SERIALIZE/DESER  worstDelta={Worst(baseline, restored.Predict(input)):E3}");

        // A fresh instance handed the original's parameter vector directly — no serialization at all.
        using var viaParams = new MusicSourceSeparator<double>(arch);
        viaParams.UpdateParameters(model.GetParameters());
        _out.WriteLine($"SETPARAMS ONLY   worstDelta={Worst(baseline, viaParams.Predict(input)):E3}");

        Assert.True(true, "localization only");
    }
}

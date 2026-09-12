using System;
using AiDotNet.Audio.SourceSeparation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests;

/// <summary>
/// MusicSourceSeparator's Clone(), a serialize/deserialize round-trip, and a bare parameter copy must
/// all reproduce the original model's predictions.
/// </summary>
/// <remarks>
/// <para>
/// This began as a localization scaffold for a clone divergence (the clone's parameters were
/// bit-identical, yet Encoder_0's output differed by 7.93e-01) and asserted nothing. It also never
/// ran in CI: no shard filter selected this class, so when the paper-default configuration began
/// requiring at least 4096 samples, its 64-sample input went stale without anyone noticing.
/// </para>
/// <para>
/// It now asserts what the scaffold was measuring, so a regression in any of the three copy paths
/// fails instead of printing a number nobody reads.
/// </para>
/// </remarks>
public class MssCloneVsSerializeLocalization
{
    private const int Samples = 64;
    private const double Tolerance = 1e-9;

    private readonly ITestOutputHelper _out;
    public MssCloneVsSerializeLocalization(ITestOutputHelper o) => _out = o;

    // The same smoke-scale Demucs the generated ModelFamily scaffold uses (TestScaffoldGenerator):
    // the paper defaults build a 100.6M-parameter double fixture that needs >= 4096 samples, peaks at
    // several GiB, and would hold four of them here.
    private static MusicSourceSeparator<double> Create(NeuralNetworkArchitecture<double> arch) =>
        new(arch, new SourceSeparationOptions
        {
            DemucsDepth = 2, DemucsBaseChannels = 8, DemucsKernelSize = 8,
            DemucsStride = 4, DemucsPadding = 2, StemCount = 4,
        });

    [Fact]
    public void CloneSerializeAndParameterCopy_ReproduceTheOriginalPrediction()
    {
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: Samples, outputSize: 256);
        using var model = Create(arch);

        var rng = new Random(7);
        var input = new Tensor<double>([1, Samples]);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() * 2 - 1;

        var baseline = model.Predict(input);

        static double Worst(Tensor<double> a, Tensor<double> b)
        {
            Assert.Equal(a.Length, b.Length);
            double w = 0;
            for (int i = 0; i < a.Length; i++) w = Math.Max(w, Math.Abs(a[i] - b[i]));
            return w;
        }

        using var cloned = (MusicSourceSeparator<double>)model.Clone();
        double cloneDelta = Worst(baseline, cloned.Predict(input));

        byte[] bytes = model.Serialize();
        using var restored = Create(arch);
        restored.Deserialize(bytes);
        double serializeDelta = Worst(baseline, restored.Predict(input));

        // A fresh instance handed the original's parameter vector directly - no serialization at all.
        // Its lazy layers have no shapes until they see an input, so they are resolved first, exactly
        // as the SetParameters error directs for a fresh model receiving trained weights.
        using var viaParams = Create(arch);
        viaParams.ResolveShapes(input);
        viaParams.UpdateParameters(model.GetParameters());
        double parameterDelta = Worst(baseline, viaParams.Predict(input));

        _out.WriteLine($"clone {cloneDelta:E3}, serialize/deserialize {serializeDelta:E3}, parameters only {parameterDelta:E3}");
        Assert.True(cloneDelta <= Tolerance, $"Clone() diverged from the original by {cloneDelta:E3}.");
        Assert.True(serializeDelta <= Tolerance, $"A serialize/deserialize round-trip diverged by {serializeDelta:E3}.");
        Assert.True(parameterDelta <= Tolerance, $"Copying the parameter vector alone diverged by {parameterDelta:E3}.");
    }
}

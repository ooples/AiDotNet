using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for audio classification models (genre, event, scene classification).
/// Inherits audio NN invariants and adds classification-specific: valid class outputs
/// and silence classification behavior.
/// </summary>
/// <remarks>
/// Generic over <typeparamref name="T"/> so heavy paper-scale audio classifiers (e.g. PANNs
/// CNN14) can run their generated scaffold in <c>&lt;float&gt;</c> — half the activation/tape
/// footprint and ~2x the throughput, which the double-precision variant needs to fit the
/// per-test timeout. The non-generic <see cref="AudioClassifierTestBase"/> alias below preserves
/// the default <c>&lt;double&gt;</c> behavior for every model not selected for a float scaffold.
/// </remarks>
public abstract class AudioClassifierTestBase<T> : AudioNNModelTestBase<T>
{
    [Fact(Timeout = 60000)]
    public async Task ClassOutput_ShouldBeNonNegative()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var output = network.Predict(input);

        for (int i = 0; i < output.Length; i++)
        {
            double v = ConvertToDouble(output[i]);
            Assert.True(v >= -1e-10,
                $"Audio class output[{i}] = {v:F4} is negative — invalid class score.");
        }
    }

    [Fact(Timeout = 60000)]
    public async Task SilenceClassification_ShouldNotCrash()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var network = CreateNetwork();
        var silence = CreateConstantTensor(InputShape, 0.0);
        var output = network.Predict(silence);
        Assert.True(output.Length > 0, "Audio classifier produced empty output for silence.");
        for (int i = 0; i < output.Length; i++)
            Assert.False(double.IsNaN(ConvertToDouble(output[i])), $"Audio class output[{i}] is NaN for silence.");
    }
}

/// <summary>Double-precision default for <see cref="AudioClassifierTestBase{T}"/>.</summary>
public abstract class AudioClassifierTestBase : AudioClassifierTestBase<double> { }

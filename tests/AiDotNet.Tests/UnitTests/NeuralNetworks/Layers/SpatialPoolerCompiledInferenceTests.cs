using System.Threading.Tasks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// The spatial pooler min-max normalizes its column overlaps before thresholding. It read the extremes back to the
/// host, so a compiled inference plan froze the traced input's minimum and range and thresholded every later input
/// against them. The extremes are now tensor reductions the plan recomputes, so a plan traced on one input and
/// replayed on another, at a different level and scale, gives that input's eager SDR.
/// </summary>
public class SpatialPoolerCompiledInferenceTests
{
    [Fact(Timeout = 60000)]
    public async Task A_compiled_plan_replayed_on_a_new_input_gives_its_eager_sdr()
    {
        await Task.Yield();
        const int inputs = 12, columns = 24;
        var layer = new SpatialPoolerLayer<double>(columns, sparsityThreshold: 0.3) { RandomSeed = 5 };
        var random = RandomHelper.CreateSeededRandom(8);
        var traced = new Tensor<double>(new[] { inputs });
        var replayed = new Tensor<double>(new[] { inputs });
        for (int i = 0; i < inputs; i++)
        {
            traced[i] = random.NextDouble();
            replayed[i] = 40.0 + 25.0 * random.NextDouble() * random.NextDouble();
        }
        layer.Forward(traced);
        var expected = layer.Forward(replayed).ToArray();
        Assert.NotEqual(expected, layer.Forward(traced).ToArray());

        using var cache = new CompiledModelCache<double>();
        var plan = cache.GetOrCompileInference(traced, () => layer.Forward(traced));
        var rebindable = Assert.IsAssignableFrom<ICompiledPlan<double>>(plan);
        rebindable.SetInputs(new[] { replayed });
        Assert.Equal(expected, plan.Execute().ToArray());
    }
}

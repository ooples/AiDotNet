using System;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers.SSM;

/// <summary>
/// The incremental <c>Step</c> of <see cref="GatedDeltaNetLayer{T}"/> and <see cref="MultiLatentAttentionLayer{T}"/> on
/// the GPU engine, eagerly and as a compiled decode plan. Their states are updated in place (the convolution history
/// and delta-rule memory; the latent cache at a device-side position), so a plan traced on one step replays every
/// later step and must give the full causal forward's output at every position.
/// </summary>
[Collection("DiffusionGpuCuda")]
public class IncrementalStepGpuTests
{
    private const int Batch = 2, SeqLen = 10, ModelDim = 32, Heads = 4;

    private static Tensor<float> Position(Tensor<float> x, int t)
    {
        var slice = new Tensor<float>(new[] { Batch, 1, ModelDim });
        for (int b = 0; b < Batch; b++)
            for (int d = 0; d < ModelDim; d++)
                slice[b, 0, d] = x[b, t, d];
        return slice;
    }

    // Runs `step` per position on the GPU engine (eagerly, or replaying a plan traced on the first call) against the CPU
    // forward of the same layer; returns the host-side step count, which a replayed plan does not advance.
    private static int Run(LayerBase<float> layer, Func<Tensor<float>, Tensor<float>> step, Func<int> hostPosition, bool compiled)
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch { /* no GPU backend on this host */ }
        var previous = AiDotNetEngine.Current;
        try
        {
            Skip.If(gpu is null || !gpu.SupportsGpu || !gpu.IsGpuAvailable, "No GPU engine available on this host.");
            AiDotNetEngine.Current = new CpuEngine();
            layer.SetTrainingMode(false);
            var rng = new Random(11);
            var data = new float[Batch * SeqLen * ModelDim];
            for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
            var x = new Tensor<float>(new[] { Batch, SeqLen, ModelDim }, new Vector<float>(data));
            var full = layer.Forward(x);

            AiDotNetEngine.Current = gpu;
            var cache = new CompiledModelCache<float>();
            double worst = 0, scale = 0;
            for (int t = 0; t < SeqLen; t++)
            {
                var y = compiled ? cache.GetOrCompileInference(Position(x, t), step).Execute() : step(Position(x, t));
                for (int b = 0; b < Batch; b++)
                    for (int d = 0; d < ModelDim; d++)
                    {
                        worst = Math.Max(worst, Math.Abs(full[b, t, d] - y[b, 0, d]));
                        scale = Math.Max(scale, Math.Abs(full[b, t, d]));
                    }
            }
            Assert.True(scale > 0, "the forward output is identically zero");
            Assert.True(worst / scale < 1e-4, $"step vs forward relative error {worst / scale:G4}");
            return hostPosition();
        }
        finally
        {
            AiDotNetEngine.Current = previous;
            gpu?.Dispose();
        }
    }

    [SkippableTheory]
    [Trait("category", "gpu")]
    [InlineData(false)]
    [InlineData(true)]
    public void GatedDeltaNet_Step_on_the_gpu_matches_the_forward_at_every_position(bool compiled)
    {
        var layer = new GatedDeltaNetLayer<float>(SeqLen, ModelDim, Heads, convKernelSize: 4);
        var state = layer.CreateStepState(Batch);
        // Eagerly the host runs every step; a replayed plan runs it once, on the trace.
        Assert.Equal(compiled ? 1 : SeqLen, Run(layer, xt => layer.Step(xt, state), () => state.Position, compiled));
    }

    [SkippableTheory]
    [Trait("category", "gpu")]
    [InlineData(false)]
    [InlineData(true)]
    public void MultiLatentAttention_Step_on_the_gpu_matches_the_forward_at_every_position(bool compiled)
    {
        var layer = new MultiLatentAttentionLayer<float>(SeqLen, ModelDim, Heads, latentDimension: 8);
        var state = layer.CreateStepState(Batch);
        Assert.Equal(compiled ? 1 : SeqLen, Run(layer, xt => layer.Step(xt, state), () => state.Position, compiled));
    }
}

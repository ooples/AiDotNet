using System;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// Incremental decoding through <see cref="GroupedQueryAttentionLayer{T}.Step"/> (a growing key-value cache) gives
/// the outputs the full forward over the whole sequence gives at the same positions: a prompt in one step, then one
/// token per step, as a B = 1 KV-cached decode runs.
/// </summary>
public sealed class GroupedQueryAttentionStepTests
{
    private const int Seq = 9, Dim = 32, Heads = 4, KvHeads = 2, Prompt = 4;

    private static GroupedQueryAttentionLayer<float> Layer(bool rope, bool causal, double softcap, bool bias)
    {
        var layer = new GroupedQueryAttentionLayer<float>(Seq, Dim, Heads, KvHeads,
            useProjectionBias: bias, attnLogitSoftcap: softcap, useCausalMask: causal);
        if (rope) layer.ConfigurePositionalEncoding(PositionalEncodingType.Rotary, maxSequenceLength: 64);
        var p = layer.GetParameters();
        var rng = new Random(7);
        var values = new float[p.Length];
        for (int i = 0; i < values.Length; i++) values[i] = (float)((rng.NextDouble() - 0.5) * 0.4);
        layer.SetParameters(new AiDotNet.Tensors.LinearAlgebra.Vector<float>(values));
        layer.SetTrainingMode(false);
        return layer;
    }

    private static Tensor<float> Input(int batch)
    {
        var rng = new Random(11);
        var x = new Tensor<float>(new[] { batch, Seq, Dim });
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        return x;
    }

    private static Tensor<float> Slice(Tensor<float> x, int from, int count)
    {
        int b = x.Shape[0];
        var r = new Tensor<float>(new[] { b, count, Dim });
        for (int n = 0; n < b; n++)
            for (int t = 0; t < count; t++)
                for (int c = 0; c < Dim; c++)
                    r[n, t, c] = x[n, from + t, c];
        return r;
    }

    private static void AssertStepsMatchTheFullForward(GroupedQueryAttentionLayer<float> layer, int batch)
    {
        var x = Input(batch);

        var state = layer.CreateStepState(batch);
        var outputs = new float[batch * Seq * Dim];
        var full = new float[batch * Seq * Dim];
        // A step's rows are the last rows of the full forward over every token seen so far (with no causal mask a
        // position attends to the tokens after it, which a step has not seen; with one, this is the whole sequence's
        // forward).
        void Put(float[] into, Tensor<float> y, int from, int count)
        {
            var a = y.ToArray();
            int rows = a.Length / (batch * Dim);
            for (int n = 0; n < batch; n++)
                for (int t = 0; t < count; t++)
                    Array.Copy(a, (n * rows + rows - count + t) * Dim, into, (n * Seq + from + t) * Dim, Dim);
        }
        Put(outputs, layer.Step(Slice(x, 0, Prompt), state), 0, Prompt);          // the prompt in one step
        Put(full, layer.Forward(Slice(x, 0, Prompt)), 0, Prompt);
        for (int t = Prompt; t < Seq; t++)
        {
            var token = t % 2 == 0 ? Slice(x, t, 1) : Slice(x, t, 1).Reshape(batch, Dim);   // [B, 1, D] and [B, D]
            Put(outputs, layer.Step(token, state), t, 1);
            Put(full, layer.Forward(Slice(x, 0, t + 1)), t, 1);
        }
        Assert.Equal(Seq, state.Position);

        for (int i = 0; i < full.Length; i++)
            Assert.True(Math.Abs(outputs[i] - full[i]) <= 1e-5f * (1f + Math.Abs(full[i])),
                $"[{i}] (position {i / Dim % Seq}): step {outputs[i]} vs full forward {full[i]}");
    }

    [Theory]
    [InlineData(true, true, 0.0, false, 1)]
    [InlineData(true, true, 0.0, true, 2)]
    [InlineData(true, true, 30.0, false, 1)]
    [InlineData(false, true, 0.0, false, 1)]
    [InlineData(true, false, 0.0, false, 1)]
    public void Steps_over_a_growing_cache_give_the_full_forward_on_the_cpu(bool rope, bool causal, double softcap, bool bias, int batch)
    {
        using var cpu = AiDotNetEngine.UseInCurrentContext(new CpuEngine());
        AssertStepsMatchTheFullForward(Layer(rope, causal, softcap, bias), batch);
    }

    [SkippableTheory]
    [InlineData(true, true, 0.0, false, 1)]
    [InlineData(true, true, 30.0, true, 2)]
    public void Steps_over_a_growing_cache_give_the_full_forward_on_the_gpu(bool rope, bool causal, double softcap, bool bias, int batch)
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is not OutOfMemoryException) { Skip.If(true, "No GPU backend"); return; }
        if (!gpu.IsGpuAvailable) { gpu.Dispose(); Skip.If(true, "No GPU available"); return; }
        try
        {
            using var scope = AiDotNetEngine.UseInCurrentContext(gpu);
            AssertStepsMatchTheFullForward(Layer(rope, causal, softcap, bias), batch);
        }
        finally { gpu.Dispose(); }
    }

    [Fact]
    public void A_state_from_a_layer_of_another_shape_is_refused()
    {
        var layer = Layer(true, true, 0.0, false);
        var other = new GroupedQueryAttentionLayer<float>(Seq, Dim, Heads, Heads, useCausalMask: true);
        Assert.Throws<ArgumentException>(() => layer.Step(Slice(Input(1), 0, 1), other.CreateStepState(1)));
    }
}

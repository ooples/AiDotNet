using AiDotNet.Attributes;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// A layer's persistent state outlives any <see cref="TensorScope"/> that happens to be open when the layer creates it
/// lazily on its first forward: a lazily allocated weight (<c>AllocateLazyWeight</c>) and a registered non-trainable
/// buffer (<c>RegisterBuffer</c>). The detached GAN forward (<c>ForwardForTrainingDetached</c>) and the compiled step's
/// sizing forward both run a model's first forward inside a scope that frees every GPU result it saw created, so state
/// made there as a GPU result was freed with the activations and the next forward read a released buffer.
/// </summary>
[Collection("NonParallelIntegration")]
public sealed class LayerPersistentStateScopeTests
{
    [SkippableFact]
    [Trait("Category", "GPU")]
    public void LazyWeightAndRegisteredBuffer_CreatedInsideAScope_SurviveIt()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch { /* no GPU backend */ }
        var previous = AiDotNetEngine.Current;
        try
        {
            Skip.If(gpu is null || !gpu.SupportsGpu || !gpu.IsGpuAvailable, "No DirectGpuTensorEngine available on this machine.");
            AiDotNetEngine.Current = gpu;

            var source = new Tensor<float>([64]);
            for (int i = 0; i < source.Length; i++) source[i] = 0.25f * i - 3f;
            using var layer = new LazyStateProbeLayer();

            Tensor<float> weight, buffer, activation;
            using (TensorScope.Begin())
            {
                // Each is a GPU result (an engine op on the device) created while the scope is open, as a lazy layer
                // that builds its state from its first input does.
                weight = layer.AllocateWeightFrom(source);
                buffer = layer.RegisterBufferFrom(source);
                activation = gpu!.TensorMultiplyScalar(source, 2f);
            }

            // The scope frees what it saw created and nothing exempted: an ordinary activation is gone.
            Assert.IsType<InvalidOperationException>(Record.Exception(() => activation.ToArray()));

            for (int i = 0; i < source.Length; i++)
            {
                Assert.Equal(2f * source[i], weight[i]);
                Assert.Equal(2f * source[i], buffer[i]);
            }
            Assert.Same(buffer, Assert.Single(layer.GetRegisteredBuffers()).Tensor);
        }
        finally
        {
            AiDotNetEngine.Current = previous;
            gpu?.Dispose();
        }
    }

    [ElementWiseShape]
    private sealed class LazyStateProbeLayer : LayerBase<float>
    {
        public LazyStateProbeLayer() : base([1], [1])
        {
        }

        public Tensor<float> AllocateWeightFrom(Tensor<float> source)
            => AllocateLazyWeight(source.Shape.ToArray(), () => Engine.TensorMultiplyScalar(source, 2f));

        public Tensor<float> RegisterBufferFrom(Tensor<float> source)
        {
            var buffer = Engine.TensorMultiplyScalar(source, 2f);
            RegisterBuffer(buffer, "lazy_state");
            return buffer;
        }

        protected override Tensor<float> ForwardTraced(Tensor<float> input) => input;

        public override bool SupportsTraining => false;

        public override void ResetState()
        {
        }
    }
}

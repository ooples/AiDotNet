using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

[Collection("FusedTrainingSerial")]
public sealed class SpectralStateSnapshotOwnershipTests
{
    [Fact]
    public void CpuZeroAdditionSnapshotsOwnTheirPhysicalStorage() => Check(new CpuEngine());

    [SkippableFact]
    public void GpuZeroAdditionSnapshotsOwnTheirNativeRoots()
    {
        using var engine = new DirectGpuTensorEngine();
        Skip.IfNot(engine.IsGpuAvailable, "No direct GPU backend available.");
        Check(engine);
    }

    private static void Check(IEngine engine)
    {
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            foreach (var original in new[] { new[] { .6f, .8f }, new[] { -.2f, .4f, -.7f, .5f } })
            {
                var state = new Tensor<float>((float[])original.Clone(), new[] { original.Length });
                // Materialize the same resident state consumed by later spectral power iterations on a GPU.
                if (engine.SupportsGpu) state = engine.TensorMultiplyScalar(state, 1f);
                Tensor<float> snapshot;
                using var tape = new GradientTape<float>();
                using (new NoGradScope<float>())
                    snapshot = engine.Reshape(engine.TensorAddScalar(state, 0f), new[] { 1, original.Length });
                Assert.NotSame(state.StorageIdentity, snapshot.StorageIdentity);
                if (engine.SupportsGpu)
                {
                    var sourceBuffer = Assert.IsAssignableFrom<IGpuBuffer>(state.TryGetGpuBuffer());
                    var snapshotBuffer = Assert.IsAssignableFrom<IGpuBuffer>(snapshot.TryGetGpuBuffer());
                    var sourceRoot = GpuBufferViewRoot.Of(sourceBuffer);
                    var snapshotRoot = GpuBufferViewRoot.Of(snapshotBuffer);
                    Assert.NotSame(sourceRoot, snapshotRoot);
                    Assert.NotEqual(sourceRoot.Handle, snapshotRoot.Handle);
                }
                using (new NoGradScope<float>())
                    engine.TensorCopyInPlace(state,
                        new Tensor<float>(original.Select(x => x * -.5f + .1f).ToArray(), new[] { original.Length }));
                Assert.Equal(original, snapshot.ToArray());
                Assert.Equal(original.Select(x => x * -.5f + .1f).ToArray(), state.ToArray());
            }
        }
        finally { AiDotNetEngine.Current = previous; }
    }
}

using AiDotNet.Diffusion;
using AiDotNet.NeuralNetworks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Diffusion;

/// <summary>
/// Forward-caching-allocator tests for the diffusion denoising loop
/// (<see cref="DiffusionModelBase{T}.Generate(int[], int, int?)"/>) — the second
/// consumer-side boundary named in Tensors #661 ("NeuralNetworkBase.Predict / the
/// diffusion denoise loop"). The loop now runs inside one <c>TensorArena</c> with a
/// <c>Reset()</c> per step (gated by <see cref="InferenceArenaSettings.Enabled"/>),
/// recycling each step's noise-predictor + scheduler intermediates instead of
/// GC-churning them — the dominant inference-time allocation source (~50 forwards
/// per generation).
///
/// <para>Unlike a classifier <c>Predict</c>, a <see cref="DiffusionModelBase{T}"/>
/// carries a latent (<c>sample</c>) across every step; the loop detaches it to a
/// GC-owned buffer before each <c>Reset</c>, so these tests guard that the carried
/// latent never points into recycled arena scratch.</para>
/// </summary>
/// <remarks>The class flips the process-global <see cref="InferenceArenaSettings.Enabled"/>,
/// so it shares the serialized "InferenceArena" collection and always restores the flag.</remarks>
[Collection("InferenceArena")]
public class DiffusionGenerateArenaIntegrationTests
{
    private static T WithArena<T>(bool enabled, System.Func<T> body)
    {
        bool prev = InferenceArenaSettings.Enabled;
        InferenceArenaSettings.Enabled = enabled;
        try { return body(); }
        finally { InferenceArenaSettings.Enabled = prev; }
    }

    /// <summary>
    /// The denoise loop's output must be bit-exact with the arena ON vs OFF. The
    /// arena hands back <c>RentUninitialized</c> buffers holding the prior step's
    /// bytes, so any step that relies on zero-initialised scratch — or any missing
    /// detach of the carried latent — would diverge here.
    /// </summary>
    [Fact]
    public void Generate_ArenaOnOff_IsBitIdentical()
    {
        const int seed = 42;
        const int steps = 10;
        var shape = new[] { 1, 4, 8, 8 };

        // Fresh model per path: same seed → same initial noise + same deterministic
        // scheduler trajectory, so the only variable is the arena.
        var off = WithArena(false, () => new DDPMModel<float>(seed).Generate(shape, steps, seed));
        var on = WithArena(true, () => new DDPMModel<float>(seed).Generate(shape, steps, seed));

        Assert.Equal(off.Shape, on.Shape);
        Assert.Equal(off.Length, on.Length);
        for (int i = 0; i < off.Length; i++)
            Assert.Equal(off[i], on[i]); // bit-exact (float, no tolerance)
    }

    /// <summary>
    /// Replay determinism with the arena ON: two generations from the same seed on
    /// the same instance must be bit-identical. Catches arena state bleeding between
    /// full generations (a Reset that leaves the carried latent aliasing scratch, or
    /// a buffer not returned to the pool).
    /// </summary>
    [Fact]
    public void Generate_ArenaOn_ReplayIsDeterministic()
    {
        const int seed = 7;
        var shape = new[] { 1, 4, 4, 4 };

        var (first, second) = WithArena(true, () =>
        {
            var model = new DDPMModel<float>(seed);
            var a = model.Generate(shape, numInferenceSteps: 5, seed: seed);
            var b = model.Generate(shape, numInferenceSteps: 5, seed: seed);
            return (a, b);
        });

        Assert.Equal(first.Length, second.Length);
        for (int i = 0; i < first.Length; i++)
            Assert.Equal(first[i], second[i]);
    }
}

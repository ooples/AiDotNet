using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Thread-local source of deterministic per-layer initialization seeds, reset at
/// the start of each model's construction.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why this exists.</b> Layers initialize their weights inside their
/// constructors, which run BEFORE <see cref="LayerBase{T}.RandomSeed"/> is wired
/// (<c>LayerHelper.Wire</c> / <c>NeuralNetworkBase.EnsureLayerRandomSeedsWired</c>
/// both set it AFTER the layer object already exists). With no seed set at
/// construction time, weight initialization falls back to the process-shared,
/// fixed-seed <see cref="RandomHelper.ThreadSafeRandom"/>, whose state advances
/// cumulatively across every prior draw in the process. That made a model's
/// INITIAL WEIGHTS depend on how much unrelated work ran first — so two
/// constructions of the same architecture at the same <c>architecture.RandomSeed</c>
/// produced DIFFERENT initial weights depending on execution order (the systemic
/// cross-process / cross-test training-determinism bug: an invariant that holds in
/// isolation flips once other work has advanced the shared RNG).
/// </para>
/// <para>
/// <b>How it fixes it.</b> <see cref="NeuralNetworkBase{T}"/>'s constructor calls
/// <see cref="ResetForModelConstruction"/> with the architecture's seed BEFORE the
/// derived constructor builds its layers. Each <see cref="LayerBase{T}"/>
/// constructor then pulls a deterministic seed from this scope via
/// <see cref="NextSeedOrNull"/> when no explicit seed was set. Because the
/// sequence is RESET at the start of every model construction, the per-layer init
/// seeds depend only on the model's own (deterministic) layer-construction order
/// and its architecture seed — never on prior, unrelated work. When the
/// architecture has no seed (the production default), the scope is inert and
/// layers keep their existing non-reproducible <see cref="RandomHelper.ThreadSafeRandom"/>
/// behaviour, preserving the "reproducible iff a seed was requested" contract.
/// </para>
/// </remarks>
internal static class LayerInitializationSeedScope
{
    [ThreadStatic]
    private static Random? _rng;

    [ThreadStatic]
    private static int? _ambientFallbackSeed;

    /// <summary>
    /// Test-only ambient fallback init seed for the current thread. When set,
    /// model constructions whose architecture carries NO explicit
    /// <see cref="NeuralNetworks.NeuralNetworkArchitecture{T}.RandomSeed"/> derive
    /// their per-layer weight init from THIS seed instead of the process-shared,
    /// order-dependent <see cref="RandomHelper.ThreadSafeRandom"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This exists solely so the ModelFamily test harness can pin weight init for
    /// deep, init-sensitive models (e.g. the end-to-end TTS family — VITS, VITS2,
    /// NaturalSpeech, Kokoro, MeloTTS, Piper, YourTTS — whose VAE+flow+decoder
    /// stack diverges from a poorly-scaled init). Those models otherwise inherit a
    /// non-reproducible init whose scale depends on how many sibling tests ran on
    /// the same xUnit worker thread first, so an invariant like
    /// <c>MoreData_ShouldNotDegrade</c> passes in isolation but fails when
    /// interleaved with other classes. The generated test's factory sets this seed
    /// around construction (and clears it in a <c>finally</c>) so the fix is scoped
    /// to exactly those tests.
    /// </para>
    /// <para>
    /// Production code never sets this (the property is internal and only the test
    /// assembly is granted access via <c>InternalsVisibleTo</c>), so the
    /// "reproducible iff a seed was requested" contract is preserved: when neither
    /// an architecture seed nor this ambient seed is present, the scope stays inert
    /// and layers keep their existing non-reproducible behaviour.
    /// </para>
    /// </remarks>
    internal static int? AmbientFallbackSeed
    {
        get => _ambientFallbackSeed;
        set => _ambientFallbackSeed = value;
    }

    /// <summary>
    /// Begins a fresh deterministic per-layer init-seed sequence for the model
    /// about to be constructed on this thread. Pass the architecture's resolved
    /// seed (or <c>null</c> to fall back to <see cref="AmbientFallbackSeed"/>, then
    /// to the existing non-reproducible initialization when neither is set). Called
    /// from the <see cref="NeuralNetworkBase{T}"/> constructor, which runs before
    /// the derived model constructor builds its layers.
    /// </summary>
    internal static void ResetForModelConstruction(int? architectureSeed)
    {
        int? effectiveSeed = architectureSeed ?? _ambientFallbackSeed;
        _rng = effectiveSeed.HasValue
            ? RandomHelper.CreateSeededRandom(effectiveSeed.Value)
            : null;
    }

    /// <summary>
    /// Returns the next deterministic per-layer init seed, or <c>null</c> when no
    /// seeded construction scope is active (no architecture seed was set). A
    /// <c>null</c> return tells the layer to keep its existing initialization
    /// behaviour.
    /// </summary>
    internal static int? NextSeedOrNull()
    {
        // ARM LAZILY FROM THE AMBIENT SEED.
        //
        // ResetForModelConstruction runs inside the model's constructor, but a layer can be built
        // BEFORE that: an architecture carrying explicit Layers is constructed as an ARGUMENT to
        // the model constructor, and C# evaluates arguments first. Those layers called this, found
        // _rng still null, took no seed, and fell back to the process-shared
        // RandomHelper.ThreadSafeRandom -- whose position depends on every draw made before it on
        // the thread.
        //
        // Measured on the Dessurt fixture, whose architecture supplies FlattenLayer + DenseLayer
        // (12,292 parameters): archSeed=1234 yet 0 of 2 layers carried a RandomSeed, and the
        // initial parameter sum was 3.27575872 running alone against 0.739498765 with one sibling
        // fixture ahead of it -- which carried training loss from 4.80392122 to 0.0138604036 and
        // decided a pass/fail purely on ordering.
        //
        // AmbientFallbackSeed was already being set around construction for this exact purpose, but
        // only ResetForModelConstruction consumed it, so it never reached anything built ahead of
        // the constructor body. Arming here on first use closes that window without changing the
        // contract: with no ambient seed and no reset, this still returns null and initialization
        // stays exactly as it was.
        if (_rng is null && _ambientFallbackSeed.HasValue)
            _rng = RandomHelper.CreateSeededRandom(_ambientFallbackSeed.Value);

        return _rng is null ? (int?)null : _rng.Next();
    }
}

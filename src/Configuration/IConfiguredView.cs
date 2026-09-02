namespace AiDotNet.Configuration;

/// <summary>
/// Read-only test-verification view over the post-Configure*() state of an
/// <see cref="AiModelBuilder{T,TInput,TOutput}"/>. Used exclusively by the
/// integration-test bucket suite to assert that configuration values land on
/// the builder's slot (the "stored on the builder but never consumed"
/// regression PR #1357/#1361/#1368 hunts for).
/// </summary>
/// <remarks>
/// <para>
/// Implemented EXPLICITLY on <see cref="AiModelBuilder{T,TInput,TOutput}"/>
/// so the accessors do NOT appear on the production type's regular surface
/// (review #1368 C6WRW). Test code casts to <see cref="IConfiguredView{T,TInput,TOutput}"/>
/// to access the values:
/// </para>
/// <code>
/// var builder = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
///     .ConfigureModel(model)
///     .ConfigureCaching(new CacheConfig { MaxCacheSize = 99 });
/// var view = (IConfiguredView&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;)builder;
/// Assert.Equal(99, view.ConfiguredCaching!.MaxCacheSize);
/// </code>
/// <para>
/// The interface is marked <c>internal</c> and the AiDotNet.Tests assembly
/// reaches it via <see cref="System.Runtime.CompilerServices.InternalsVisibleToAttribute"/>.
/// Production callers cannot see (or accidentally bind against) the
/// accessors because the interface symbol isn't visible to them.
/// </para>
/// </remarks>
/// <typeparam name="T">Numeric type the builder operates on.</typeparam>
/// <typeparam name="TInput">Input tensor / matrix / sample type.</typeparam>
/// <typeparam name="TOutput">Output tensor / scalar / sequence type.</typeparam>
internal interface IConfiguredView<T, TInput, TOutput>
{
    /// <summary>The active <see cref="AiDotNet.Interfaces.IOptimizer{T,TInput,TOutput}"/> picked by <c>ConfigureOptimizer</c>.</summary>
    AiDotNet.Interfaces.IOptimizer<T, TInput, TOutput>? ConfiguredOptimizer { get; }

    /// <summary>The cache config wired via <c>ConfigureCaching</c>.</summary>
    AiDotNet.Deployment.Configuration.CacheConfig? ConfiguredCaching { get; }

    /// <summary>The inference-optimization config wired via <c>ConfigureInferenceOptimizations</c>.</summary>
    AiDotNet.Configuration.InferenceOptimizationConfig? ConfiguredInferenceOptimizations { get; }

    /// <summary>The JIT compilation config wired via <c>ConfigureJitCompilation</c>.</summary>
    AiDotNet.Configuration.JitCompilationConfig? ConfiguredJitCompilation { get; }

    /// <summary>The interpretability options wired via <c>ConfigureInterpretability</c>.</summary>
    AiDotNet.Models.Options.InterpretabilityOptions? ConfiguredInterpretability { get; }

    /// <summary>The training memory config wired via <c>ConfigureTrainingMemoryManagement</c>.</summary>
    AiDotNet.Training.Memory.TrainingMemoryConfig? ConfiguredMemoryManagement { get; }

    /// <summary>The license-key payload wired via <c>ConfigureLicenseKey</c>.</summary>
    AiDotNet.Models.AiDotNetLicenseKey? ConfiguredLicenseKey { get; }

    /// <summary>The evolution run settings wired via <c>ConfigureEvolution</c>.</summary>
    EvolutionOptions? ConfiguredEvolution { get; }

    /// <summary>The program seeds wired via <c>ConfigureEvolutionSeeds(EvolutionSeedOptions)</c>.</summary>
    EvolutionSeedOptions? ConfiguredEvolutionSeeds { get; }

    /// <summary>
    /// How many typed candidates were captured by <c>ConfigureEvolutionSeeds&lt;TGenome&gt;</c>.
    /// </summary>
    /// <remarks>
    /// The seeds are held under a genome type this interface cannot name, so their presence is reported as a count
    /// rather than exposed directly.
    /// </remarks>
    int ConfiguredEvolutionSeedCount { get; }

    /// <summary>The program-evolution options wired via <c>ConfigureProgramEvolution</c>.</summary>
    ProgramEvolutionOptions? ConfiguredProgramEvolution { get; }

    /// <summary>The chat client wired via <c>ConfigureChatClient</c> or <c>ConfigureChatClientEnsemble</c>.</summary>
    AiDotNet.Agentic.Models.IChatClient<T>? ConfiguredChatClient { get; }

    /// <summary>The chat-client pipeline settings wired alongside the client.</summary>
    ChatClientOptions? ConfiguredChatClientOptions { get; }

    /// <summary>The sandbox settings wired via <c>ConfigureProgramSandbox</c>.</summary>
    ProgramSandboxOptions? ConfiguredProgramSandbox { get; }

    /// <summary>The execution engine wired via <c>ConfigureProgramExecutionEngine</c>.</summary>
    AiDotNet.ProgramSynthesis.Interfaces.IProgramExecutionEngine? ConfiguredProgramExecutionEngine { get; }

    /// <summary>
    /// Whether a typed-genome run was captured by <c>ConfigureEvolution&lt;TGenome&gt;</c>.
    /// </summary>
    /// <remarks>
    /// The captured run is a delegate closed over a genome type this interface cannot name, so its presence is
    /// reported as a flag rather than exposed directly.
    /// </remarks>
    bool HasConfiguredEvolutionRun { get; }
}

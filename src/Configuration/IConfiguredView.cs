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

    /// <summary>The agent-assistance config wired via <c>ConfigureAgentAssistance</c>.</summary>
    AiDotNet.Models.AgentConfiguration<T>? ConfiguredAgentAssistance { get; }
}

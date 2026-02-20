using AiDotNet.Interfaces;

namespace AiDotNet.Safety;

/// <summary>
/// Fluent builder for constructing a <see cref="SafetyPipeline{T}"/> with custom modules.
/// </summary>
/// <remarks>
/// <para>
/// The SafetyPipelineBuilder provides a fluent API for manually assembling a safety pipeline
/// with specific modules. For most users, <see cref="SafetyPipelineFactory{T}"/> with a
/// <see cref="SafetyConfig"/> is simpler. Use this builder when you need fine-grained control
/// over which modules are included and their order.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is an advanced builder for when you want to hand-pick exactly
/// which safety modules to use. Most users should use <c>ConfigureSafety()</c> on the
/// AiModelBuilder instead, which automatically sets up the right modules.
/// </para>
/// <para>
/// <b>Example:</b>
/// <code>
/// var pipeline = new SafetyPipelineBuilder&lt;double&gt;()
///     .WithConfig(config)
///     .AddModule(new EnsembleToxicityDetector&lt;double&gt;(0.5))
///     .AddModule(new CompositePIIDetector&lt;double&gt;())
///     .AddModule(new InputGuardrail&lt;double&gt;())
///     .Build();
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SafetyPipelineBuilder<T>
{
    private SafetyConfig _config = new();
    private readonly List<ISafetyModule<T>> _modules = new();

    /// <summary>
    /// Sets the safety configuration for the pipeline.
    /// </summary>
    /// <param name="config">The safety configuration.</param>
    /// <returns>This builder for fluent chaining.</returns>
    public SafetyPipelineBuilder<T> WithConfig(SafetyConfig config)
    {
        if (config is null) throw new ArgumentNullException(nameof(config));
        _config = config;
        return this;
    }

    /// <summary>
    /// Configures the safety settings using an action delegate.
    /// </summary>
    /// <param name="configure">Action to configure the safety settings.</param>
    /// <returns>This builder for fluent chaining.</returns>
    public SafetyPipelineBuilder<T> Configure(Action<SafetyConfig> configure)
    {
        if (configure is null) throw new ArgumentNullException(nameof(configure));
        configure(_config);
        return this;
    }

    /// <summary>
    /// Adds a safety module to the pipeline.
    /// </summary>
    /// <param name="module">The safety module to add.</param>
    /// <returns>This builder for fluent chaining.</returns>
    public SafetyPipelineBuilder<T> AddModule(ISafetyModule<T> module)
    {
        _modules.Add(module);
        return this;
    }

    /// <summary>
    /// Adds multiple safety modules to the pipeline.
    /// </summary>
    /// <param name="modules">The safety modules to add.</param>
    /// <returns>This builder for fluent chaining.</returns>
    public SafetyPipelineBuilder<T> AddModules(IEnumerable<ISafetyModule<T>> modules)
    {
        _modules.AddRange(modules);
        return this;
    }

    /// <summary>
    /// Adds a text safety module to the pipeline.
    /// </summary>
    /// <param name="module">The text safety module to add.</param>
    /// <returns>This builder for fluent chaining.</returns>
    public SafetyPipelineBuilder<T> AddTextModule(ITextSafetyModule<T> module)
    {
        _modules.Add(module);
        return this;
    }

    /// <summary>
    /// Adds an image safety module to the pipeline.
    /// </summary>
    /// <param name="module">The image safety module to add.</param>
    /// <returns>This builder for fluent chaining.</returns>
    public SafetyPipelineBuilder<T> AddImageModule(IImageSafetyModule<T> module)
    {
        _modules.Add(module);
        return this;
    }

    /// <summary>
    /// Adds an audio safety module to the pipeline.
    /// </summary>
    /// <param name="module">The audio safety module to add.</param>
    /// <returns>This builder for fluent chaining.</returns>
    public SafetyPipelineBuilder<T> AddAudioModule(IAudioSafetyModule<T> module)
    {
        _modules.Add(module);
        return this;
    }

    /// <summary>
    /// Adds a video safety module to the pipeline.
    /// </summary>
    /// <param name="module">The video safety module to add.</param>
    /// <returns>This builder for fluent chaining.</returns>
    public SafetyPipelineBuilder<T> AddVideoModule(IVideoSafetyModule<T> module)
    {
        _modules.Add(module);
        return this;
    }

    /// <summary>
    /// Builds the safety pipeline with the configured modules.
    /// </summary>
    /// <returns>A configured safety pipeline ready for content evaluation.</returns>
    public SafetyPipeline<T> Build()
    {
        var pipeline = new SafetyPipeline<T>(_config);

        foreach (var module in _modules)
        {
            pipeline.AddModule(module);
        }

        return pipeline;
    }
}

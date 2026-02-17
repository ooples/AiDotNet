using System.Diagnostics;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety;

/// <summary>
/// Composable safety pipeline that runs multiple safety modules and aggregates their findings.
/// </summary>
/// <remarks>
/// <para>
/// The SafetyPipeline is the runtime orchestrator for content safety. It holds a list of
/// <see cref="ISafetyModule{T}"/> instances, runs them against content, and produces a
/// unified <see cref="SafetyReport"/>. It respects the <see cref="SafetyConfig"/> to
/// determine enabled modules, action thresholds, and exception behavior.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of the safety pipeline as an assembly line of inspectors.
/// Content flows through each inspector (module), and at the end you get a combined report
/// of everything they found. If anything dangerous is detected, the pipeline can block
/// the content or throw an exception, depending on your configuration.
/// </para>
/// <para>
/// <b>Usage via facade:</b>
/// <code>
/// var result = await new AiModelBuilder&lt;double, double[], double&gt;()
///     .ConfigureSafety(safety =&gt;
///     {
///         safety.Text.ToxicityDetection = true;
///         safety.Text.PIIDetection = true;
///     })
///     .BuildAsync();
/// </code>
/// The pipeline is constructed automatically based on your SafetyConfig.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SafetyPipeline<T>
{
    private readonly SafetyConfig _config;
    private readonly List<ISafetyModule<T>> _modules = new();

    /// <summary>
    /// Initializes a new safety pipeline with the given configuration.
    /// </summary>
    /// <param name="config">The safety configuration. If null, defaults are used.</param>
    public SafetyPipeline(SafetyConfig? config = null)
    {
        _config = config ?? new SafetyConfig();
    }

    /// <summary>
    /// Gets the configuration for this pipeline.
    /// </summary>
    public SafetyConfig Config => _config;

    /// <summary>
    /// Gets the registered safety modules.
    /// </summary>
    public IReadOnlyList<ISafetyModule<T>> Modules => _modules;

    /// <summary>
    /// Adds a safety module to the pipeline.
    /// </summary>
    /// <param name="module">The safety module to add.</param>
    /// <returns>This pipeline for fluent chaining.</returns>
    /// <exception cref="ArgumentNullException">Thrown when module is null.</exception>
    public SafetyPipeline<T> AddModule(ISafetyModule<T> module)
    {
        if (module == null)
        {
            throw new ArgumentNullException(nameof(module));
        }

        _modules.Add(module);
        return this;
    }

    /// <summary>
    /// Evaluates content represented as a numeric vector through all registered modules.
    /// </summary>
    /// <param name="content">The content to evaluate.</param>
    /// <returns>A unified safety report from all modules.</returns>
    public SafetyReport EvaluateVector(Vector<T> content)
    {
        if (!_config.EffectiveEnabled)
        {
            return SafetyReport.Safe(Array.Empty<string>());
        }

        var sw = Stopwatch.StartNew();
        var allFindings = new List<SafetyFinding>();
        var modulesRun = new List<string>();

        foreach (var module in _modules)
        {
            if (!module.IsReady)
            {
                continue;
            }

            modulesRun.Add(module.ModuleName);
            var findings = module.Evaluate(content);
            allFindings.AddRange(findings);
        }

        sw.Stop();
        return SafetyReport.FromFindings(allFindings, modulesRun, sw.Elapsed.TotalMilliseconds);
    }

    /// <summary>
    /// Evaluates text content through all registered text safety modules.
    /// </summary>
    /// <param name="text">The text content to evaluate.</param>
    /// <returns>A unified safety report from all text modules.</returns>
    public SafetyReport EvaluateText(string text)
    {
        if (!_config.EffectiveEnabled)
        {
            return SafetyReport.Safe(Array.Empty<string>());
        }

        var sw = Stopwatch.StartNew();
        var allFindings = new List<SafetyFinding>();
        var modulesRun = new List<string>();

        foreach (var module in _modules)
        {
            if (!module.IsReady)
            {
                continue;
            }

            if (module is ITextSafetyModule<T> textModule)
            {
                modulesRun.Add(module.ModuleName);
                var findings = textModule.EvaluateText(text);
                allFindings.AddRange(findings);
            }
        }

        sw.Stop();
        return SafetyReport.FromFindings(allFindings, modulesRun, sw.Elapsed.TotalMilliseconds);
    }

    /// <summary>
    /// Evaluates image content through all registered image safety modules.
    /// </summary>
    /// <param name="image">The image tensor to evaluate.</param>
    /// <returns>A unified safety report from all image modules.</returns>
    public SafetyReport EvaluateImage(Tensor<T> image)
    {
        if (!_config.EffectiveEnabled)
        {
            return SafetyReport.Safe(Array.Empty<string>());
        }

        var sw = Stopwatch.StartNew();
        var allFindings = new List<SafetyFinding>();
        var modulesRun = new List<string>();

        foreach (var module in _modules)
        {
            if (!module.IsReady)
            {
                continue;
            }

            if (module is IImageSafetyModule<T> imageModule)
            {
                modulesRun.Add(module.ModuleName);
                var findings = imageModule.EvaluateImage(image);
                allFindings.AddRange(findings);
            }
        }

        sw.Stop();
        return SafetyReport.FromFindings(allFindings, modulesRun, sw.Elapsed.TotalMilliseconds);
    }

    /// <summary>
    /// Evaluates audio content through all registered audio safety modules.
    /// </summary>
    /// <param name="audioSamples">The audio waveform.</param>
    /// <param name="sampleRate">The sample rate in Hz.</param>
    /// <returns>A unified safety report from all audio modules.</returns>
    public SafetyReport EvaluateAudio(Vector<T> audioSamples, int sampleRate)
    {
        if (!_config.EffectiveEnabled)
        {
            return SafetyReport.Safe(Array.Empty<string>());
        }

        var sw = Stopwatch.StartNew();
        var allFindings = new List<SafetyFinding>();
        var modulesRun = new List<string>();

        foreach (var module in _modules)
        {
            if (!module.IsReady)
            {
                continue;
            }

            if (module is IAudioSafetyModule<T> audioModule)
            {
                modulesRun.Add(module.ModuleName);
                var findings = audioModule.EvaluateAudio(audioSamples, sampleRate);
                allFindings.AddRange(findings);
            }
        }

        sw.Stop();
        return SafetyReport.FromFindings(allFindings, modulesRun, sw.Elapsed.TotalMilliseconds);
    }

    /// <summary>
    /// Evaluates video content through all registered video safety modules.
    /// </summary>
    /// <param name="frames">The video frames in temporal order.</param>
    /// <param name="frameRate">The frame rate in FPS.</param>
    /// <returns>A unified safety report from all video modules.</returns>
    public SafetyReport EvaluateVideo(IReadOnlyList<Tensor<T>> frames, double frameRate)
    {
        if (!_config.EffectiveEnabled)
        {
            return SafetyReport.Safe(Array.Empty<string>());
        }

        var sw = Stopwatch.StartNew();
        var allFindings = new List<SafetyFinding>();
        var modulesRun = new List<string>();

        foreach (var module in _modules)
        {
            if (!module.IsReady)
            {
                continue;
            }

            if (module is IVideoSafetyModule<T> videoModule)
            {
                modulesRun.Add(module.ModuleName);
                var findings = videoModule.EvaluateVideo(frames, frameRate);
                allFindings.AddRange(findings);
            }
        }

        sw.Stop();
        return SafetyReport.FromFindings(allFindings, modulesRun, sw.Elapsed.TotalMilliseconds);
    }

    /// <summary>
    /// Checks a safety report and throws <see cref="SafetyViolationException"/> if the content
    /// should be blocked based on the current configuration.
    /// </summary>
    /// <param name="report">The safety report to check.</param>
    /// <param name="isInput">Whether this is an input check (true) or output check (false).</param>
    /// <exception cref="SafetyViolationException">
    /// Thrown when the report indicates unsafe content and the configuration requires throwing.
    /// </exception>
    public void EnforcePolicy(SafetyReport report, bool isInput)
    {
        if (report.IsSafe)
        {
            return;
        }

        bool shouldThrow = isInput
            ? _config.EffectiveThrowOnUnsafeInput
            : _config.EffectiveThrowOnUnsafeOutput;

        if (shouldThrow && report.OverallAction >= SafetyAction.Block)
        {
            throw new SafetyViolationException(report, isInput);
        }
    }
}

using AiDotNet.Enums;
using AiDotNet.Factories;

namespace AiDotNet.Configuration;

/// <summary>
/// Applies a <see cref="YamlModelConfig"/> to an <see cref="AiModelBuilder{T, TInput, TOutput}"/>
/// by calling the appropriate <c>Configure*</c> methods.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This class bridges YAML configuration to the fluent builder API.
/// It reads each section from the parsed YAML and calls the matching builder method.
/// Because this runs in the constructor (before any fluent calls), YAML provides defaults
/// that fluent <c>.Configure*()</c> calls can override afterwards.</para>
/// </remarks>
internal static partial class YamlConfigApplier<T, TInput, TOutput>
{
    /// <summary>
    /// Applies all non-null sections of the YAML config to the builder.
    /// </summary>
    /// <param name="config">The parsed YAML configuration.</param>
    /// <param name="builder">The builder to apply configuration to.</param>
    internal static void Apply(YamlModelConfig config, AiModelBuilder<T, TInput, TOutput> builder)
    {
        // Optimizer: parse enum, use OptimizerFactory to create instance with default options
        if (config.Optimizer is not null && !string.IsNullOrWhiteSpace(config.Optimizer.Type))
        {
            if (!Enum.TryParse<OptimizerType>(config.Optimizer.Type, ignoreCase: true, out var optimizerType))
            {
                throw new ArgumentException(
                    $"Unknown optimizer type: '{config.Optimizer.Type}'. " +
                    $"Valid values are: {string.Join(", ", Enum.GetNames(typeof(OptimizerType)))}");
            }

            var optimizer = OptimizerFactory<T, TInput, TOutput>.CreateOptimizer(optimizerType);
            builder.ConfigureOptimizer(optimizer);
        }

        // Time series model: parse enum, create via factory, and configure via ConfigureModel.
        // ITimeSeriesModel<T> extends IFullModel<T, Matrix<T>, Vector<T>>, so this only works
        // when TInput=Matrix<T> and TOutput=Vector<T>. A runtime cast is used to handle the
        // generic type mismatch safely.
        if (config.TimeSeriesModel is not null && !string.IsNullOrWhiteSpace(config.TimeSeriesModel.Type))
        {
            if (!Enum.TryParse<TimeSeriesModelType>(config.TimeSeriesModel.Type, ignoreCase: true, out var tsModelType))
            {
                throw new ArgumentException(
                    $"Unknown time series model type: '{config.TimeSeriesModel.Type}'. " +
                    $"Valid values are: {string.Join(", ", Enum.GetNames(typeof(TimeSeriesModelType)))}");
            }

            // Create the time series model using the factory with matching generic types.
            // TimeSeriesModelFactory needs Matrix<T>/Vector<T> type args, but the builder
            // may use different TInput/TOutput. We create with the concrete TS types and cast.
            var tsModel = TimeSeriesModelFactory<T, Matrix<T>, Vector<T>>.CreateModel(tsModelType);

            if (tsModel is IFullModel<T, TInput, TOutput> fullModel)
            {
                builder.ConfigureModel(fullModel);
            }
            else
            {
                throw new InvalidOperationException(
                    $"Time series model type '{config.TimeSeriesModel.Type}' was created successfully, " +
                    $"but cannot be used with AiModelBuilder<{typeof(T).Name}, {typeof(TInput).Name}, {typeof(TOutput).Name}>. " +
                    $"Time series models require TInput=Matrix<T> and TOutput=Vector<T>.");
            }
        }

        // Deployment POCO configs
        if (config.Quantization is not null)
        {
            builder.ConfigureQuantization(config.Quantization);
        }

        if (config.Compression is not null)
        {
            builder.ConfigureCompression(config.Compression);
        }

        if (config.Caching is not null)
        {
            builder.ConfigureCaching(config.Caching);
        }

        if (config.Versioning is not null)
        {
            builder.ConfigureVersioning(config.Versioning);
        }

        if (config.AbTesting is not null)
        {
            builder.ConfigureABTesting(config.AbTesting);
        }

        if (config.Telemetry is not null)
        {
            builder.ConfigureTelemetry(config.Telemetry);
        }

        if (config.Export is not null)
        {
            builder.ConfigureExport(config.Export);
        }

        if (config.GpuAcceleration is not null)
        {
            builder.ConfigureGpuAcceleration(config.GpuAcceleration);
        }

        if (config.Profiling is not null)
        {
            builder.ConfigureProfiling(config.Profiling);
        }

        // Infrastructure POCO configs
        if (config.JitCompilation is not null)
        {
            builder.ConfigureJitCompilation(config.JitCompilation);
        }

        if (config.MixedPrecision is not null)
        {
            builder.ConfigureMixedPrecision(config.MixedPrecision);
        }

        if (config.Reasoning is not null)
        {
            builder.ConfigureReasoning(config.Reasoning);
        }

        if (config.Benchmarking is not null)
        {
            builder.ConfigureBenchmarking(config.Benchmarking);
        }

        if (config.InferenceOptimizations is not null)
        {
            builder.ConfigureInferenceOptimizations(config.InferenceOptimizations);
        }

        if (config.Interpretability is not null)
        {
            builder.ConfigureInterpretability(config.Interpretability);
        }

        if (config.MemoryManagement is not null)
        {
            builder.ConfigureMemoryManagement(config.MemoryManagement);
        }

        // Apply all auto-generated sections discovered by the source generator.
        ApplyGenerated(config, builder);
    }
}

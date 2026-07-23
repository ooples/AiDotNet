using AiDotNet.Data.Loaders;
using AiDotNet.LinearAlgebra;
using AiDotNet.Training.Configuration;

namespace AiDotNet.Training.Factories;

/// <summary>
/// Factory for creating data loaders from <see cref="DatasetConfig"/> objects.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This factory creates the appropriate data loader based on your dataset
/// configuration. Currently it supports CSV files. If no file path is specified, it returns null
/// so you can provide data programmatically.
/// </para>
/// </remarks>
internal static class DatasetFactory<T>
{
    /// <summary>
    /// Creates a data loader from a dataset configuration.
    /// </summary>
    /// <param name="config">The dataset configuration with path and loading settings.</param>
    /// <returns>
    /// A <see cref="CsvDataLoader{T}"/> if a file path is specified; null otherwise.
    /// The caller is responsible for calling <c>LoadAsync()</c> on the returned loader.
    /// </returns>
    public static CsvDataLoader<T>? Create(DatasetConfig? config)
    {
        if (config is null || string.IsNullOrWhiteSpace(config.Path))
        {
            return null;
        }

        return new CsvDataLoader<T>(
            config.Path,
            config.HasHeader,
            config.LabelColumn,
            config.BatchSize);
    }
}

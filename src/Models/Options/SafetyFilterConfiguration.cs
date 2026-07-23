using AiDotNet.Interfaces;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration for safety filtering during inference.
/// </summary>
/// <remarks>
/// <para>
/// This configuration controls whether safety filtering is enabled and which implementation is used.
/// When enabled and no custom filter is provided, a default filter is created using the provided options.
/// </para>
/// <para><b>For Beginners:</b> This is the safety "on/off switch" and settings bundle.
/// You can leave it alone to use safe defaults, or customize the filter for expert deployments.</para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public class SafetyFilterConfiguration<T>
{
    /// <summary>
    /// Gets or sets whether safety filtering is enabled.
    /// </summary>
    /// <remarks>
    /// When false, safety validation and output filtering are skipped.
    /// </remarks>
    public bool Enabled { get; set; } = true;

    /// <summary>
    /// Gets or sets the default options used when constructing the standard safety filter.
    /// </summary>
    public SafetyFilterOptions<T> Options { get; set; } = new();

    /// <summary>
    /// Gets or sets an optional custom safety filter implementation.
    /// </summary>
    /// <remarks>
    /// When provided, this filter is used instead of the standard default implementation.
    /// </remarks>
    public ISafetyFilter<T>? Filter { get; set; }
}


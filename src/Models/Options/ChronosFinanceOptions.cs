using System;
using AiDotNet.TimeSeries;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for <c>Finance.Forecasting.Foundation.Chronos</c>, the tensor-API Chronos-T5 (Ansari et al.
/// 2024). It shares its network, and so every setting, with <see cref="AiDotNet.TimeSeries.ChronosFoundationModel{T}"/>;
/// see <see cref="ChronosOptions{T}"/> for each one. The defaults are Chronos-T5 (Small) and the paper's training
/// recipe.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
public class ChronosFinanceOptions<T> : ChronosOptions<T>
{
    /// <summary>Initializes the options with the defaults.</summary>
    public ChronosFinanceOptions()
    {
    }

    /// <summary>Copies <paramref name="other"/>.</summary>
    public ChronosFinanceOptions(ChronosOptions<T> other)
        : base(other)
    {
    }
}

using System;

namespace AiDotNet.Agentic.Graph;

/// <summary>
/// Per-run settings for executing a <see cref="CompiledStateGraph{TState}"/>.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Knobs for a single graph run. The most important one is
/// <see cref="MaxSteps"/>, which caps how many node executions a run may take before giving up — a
/// safety net against cycles that never end.
/// </para>
/// </remarks>
public sealed class GraphRunOptions
{
    private int _maxSteps = 25;

    /// <summary>
    /// Gets or sets the maximum number of node executions allowed in a single run before a
    /// <see cref="GraphRecursionException"/> is thrown. Default: 25. Must be positive.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when set to a value &lt;= 0.</exception>
    public int MaxSteps
    {
        get => _maxSteps;
        set
        {
            if (value <= 0)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(value), value, "MaxSteps must be positive (a run needs at least one step).");
            }

            _maxSteps = value;
        }
    }
}

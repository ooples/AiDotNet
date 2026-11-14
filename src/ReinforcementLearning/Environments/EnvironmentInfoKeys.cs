using System.Collections.Generic;

namespace AiDotNet.ReinforcementLearning.Environments;

/// <summary>
/// Keys of the info dictionary returned by <see cref="AiDotNet.Interfaces.IEnvironment{T}.Step"/>.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> An episode can end in two ways. It terminates when the task reaches a real end state
/// (the pole fell, the goal was reached): nothing more can be earned. It is truncated when it is cut off by a time
/// limit while the task could have gone on: the agent should still count the value of the state it was in.
/// Gymnasium reports these as <c>terminated</c> and <c>truncated</c>; here an environment returns
/// <c>Done = true</c> for both and sets <see cref="Truncated"/> in the info dictionary for a truncation.
/// </remarks>
public static class EnvironmentInfoKeys
{
    /// <summary>
    /// Info key whose <see cref="bool"/> value is true when the episode ended by truncation (a time limit), not by
    /// reaching a terminal state.
    /// </summary>
    public const string Truncated = "truncated";

    /// <summary>
    /// Whether <paramref name="info"/> reports that the step ended its episode by truncation.
    /// </summary>
    /// <param name="info">The info dictionary returned by a step; may be null.</param>
    /// <returns>True when <see cref="Truncated"/> is present and set to true.</returns>
    public static bool IsTruncated(IReadOnlyDictionary<string, object>? info)
        => info is not null && info.TryGetValue(Truncated, out var value) && value is bool truncated && truncated;
}

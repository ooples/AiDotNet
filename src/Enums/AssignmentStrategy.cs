namespace AiDotNet.Enums;

/// <summary>
/// Strategy for assigning requests to model versions during A/B testing.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This determines how traffic is distributed between different model versions:
///
/// - **Random**: Each request is randomly assigned based on traffic split percentages.
///   Use when you want pure statistical randomness.
///
/// - **Sticky**: Users consistently get the same version (based on user ID hash).
///   Use when you want each user to have a consistent experience across sessions.
///
/// - **Gradual**: Gradually shifts traffic from old to new version over time.
///   Use when you want to slowly roll out a new version to minimize risk.
/// </para>
/// </remarks>
public enum AssignmentStrategy
{
    /// <summary>
    /// Each request randomly assigned based on traffic split.
    /// </summary>
    Random = 0,

    /// <summary>
    /// Users consistently get the same version (based on user ID hash).
    /// </summary>
    Sticky = 1,

    /// <summary>
    /// Gradually shift traffic from old to new version over time.
    /// </summary>
    Gradual = 2
}

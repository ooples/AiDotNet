using AiDotNet.Safety.Text;
using AiDotNet.Safety.Compliance;

namespace AiDotNet.Safety.Compliance;

/// <summary>
/// Abstract base class for regulatory compliance checking modules.
/// </summary>
/// <remarks>
/// <para>
/// Provides shared infrastructure for compliance modules including configuration
/// access and common compliance check utilities. Concrete implementations provide
/// the actual compliance checking logic (EU AI Act, GDPR, SOC2).
/// </para>
/// <para>
/// <b>For Beginners:</b> This base class provides common code for all compliance checkers.
/// Each checker type extends this and adds its own checks for specific laws and regulations.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class ComplianceModuleBase<T> : TextSafetyModuleBase<T>, IComplianceModule<T>
{
    /// <summary>
    /// The safety configuration to evaluate for compliance.
    /// </summary>
    protected readonly SafetyConfig Config;

    /// <summary>
    /// Initializes the compliance module base.
    /// </summary>
    /// <param name="config">The safety configuration to evaluate.</param>
    protected ComplianceModuleBase(SafetyConfig config)
    {
        Config = config;
    }

    /// <inheritdoc />
    public abstract string RegulationName { get; }

    /// <inheritdoc />
    public abstract IReadOnlyList<SafetyFinding> EvaluateCompliance(SafetyConfig config);
}

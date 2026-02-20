using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety;

/// <summary>
/// Abstract base class for all safety modules, providing common infrastructure.
/// </summary>
/// <remarks>
/// <para>
/// Provides shared functionality for all safety modules including the module name,
/// readiness state, and default numeric operations. Concrete safety modules should
/// typically inherit from a more specific base class (TextSafetyModuleBase,
/// ImageSafetyModuleBase, etc.) rather than this one directly.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is the common ancestor of all safety modules. It handles
/// boilerplate like module naming and readiness checks so each module can focus on
/// its specific detection logic.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class SafetyModuleBase<T> : ISafetyModule<T>
{
    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc />
    public abstract string ModuleName { get; }

    /// <inheritdoc />
    public virtual bool IsReady => true;

    /// <inheritdoc />
    public abstract IReadOnlyList<SafetyFinding> Evaluate(Vector<T> content);
}

using AiDotNet.Helpers;

namespace AiDotNet.NestedLearning;

/// <summary>
/// Base class for nested learning components providing Engine and NumOps.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public abstract class NestedLearningBase<T>
{
    /// <summary>
    /// The tensor computation engine.
    /// </summary>
    protected static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Numeric operations for scalar math on type T.
    /// </summary>
    protected static INumericOperations<T> NumOps { get; } = MathHelper.GetNumericOperations<T>();
}

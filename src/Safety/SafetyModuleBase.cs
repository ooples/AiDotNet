using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
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
public abstract class SafetyModuleBase<T> : ModelBase<T, Vector<T>, Vector<T>>, ISafetyModule<T>
{
    // NumOps and Engine inherited from ModelBase

    /// <inheritdoc />
    public abstract string ModuleName { get; }

    /// <inheritdoc />
    public virtual bool IsReady => true;

    /// <inheritdoc />
    public abstract IReadOnlyList<SafetyFinding> Evaluate(Vector<T> content);

    /// <summary>
    /// Predicts safety scores from content by delegating to Evaluate and converting findings to a score vector.
    /// </summary>
    public override Vector<T> Predict(Vector<T> input)
    {
        var findings = Evaluate(input);
        if (findings.Count == 0)
            return new Vector<T>(new[] { NumOps.Zero });

        var scores = new T[findings.Count];
        for (int i = 0; i < findings.Count; i++)
        {
            scores[i] = NumOps.FromDouble(findings[i].Confidence);
        }

        return new Vector<T>(scores);
    }

    /// <summary>
    /// Training is not typically used for safety modules. Override in subclasses that support fine-tuning.
    /// </summary>
    public override void Train(Vector<T> input, Vector<T> expectedOutput) { }

    /// <inheritdoc />
    public override ILossFunction<T> DefaultLossFunction => new MeanSquaredErrorLoss<T>();

    /// <inheritdoc />
    public override Vector<T> GetParameters() => new Vector<T>(0);

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters) { }

    /// <inheritdoc />
    public override IFullModel<T, Vector<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var copy = DeepCopy();
        copy.SetParameters(parameters);
        return copy;
    }

    /// <inheritdoc />
    public override IFullModel<T, Vector<T>, Vector<T>> DeepCopy()
        => (SafetyModuleBase<T>)MemberwiseClone();
}

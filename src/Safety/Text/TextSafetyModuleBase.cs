using AiDotNet.Interfaces;
using AiDotNet.Safety;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Safety.Text;

/// <summary>
/// Abstract base class for text safety modules providing common text processing utilities.
/// </summary>
/// <remarks>
/// <para>
/// Provides shared infrastructure for all text safety modules, including the
/// vector-to-text conversion needed to satisfy <see cref="ISafetyModule{T}"/>.
/// Concrete modules implement <see cref="EvaluateText(string)"/> and this base
/// class handles the <see cref="ISafetyModule{T}.Evaluate(Vector{T})"/> bridge.
/// </para>
/// <para>
/// <b>For Beginners:</b> This base class handles the boring plumbing so that each
/// text safety module only needs to implement one method: <c>EvaluateText(string)</c>.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class TextSafetyModuleBase<T> : ITextSafetyModule<T>
{
    /// <inheritdoc />
    public abstract string ModuleName { get; }

    /// <inheritdoc />
    public virtual bool IsReady => true;

    /// <inheritdoc />
    public abstract IReadOnlyList<SafetyFinding> EvaluateText(string text);

    /// <inheritdoc />
    /// <remarks>
    /// The base implementation interprets the vector as character codes and converts
    /// to a string, then delegates to <see cref="EvaluateText(string)"/>.
    /// For modules that work directly on embeddings, override this method.
    /// </remarks>
    public virtual IReadOnlyList<SafetyFinding> Evaluate(Vector<T> content)
    {
        if (content is null)
        {
            throw new ArgumentNullException(nameof(content));
        }

        // Convert numeric vector to string representation for text analysis.
        // Subclasses that work on embeddings directly should override this.
        var numOps = MathHelper.GetNumericOperations<T>();
        var chars = new char[content.Length];
        for (int i = 0; i < content.Length; i++)
        {
            int code = (int)Math.Round(numOps.ToDouble(content[i]));
            chars[i] = code is >= 0 and <= 65535 ? (char)code : '?';
        }

        return EvaluateText(new string(chars));
    }
}

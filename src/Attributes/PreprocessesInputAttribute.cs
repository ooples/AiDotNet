using System;

namespace AiDotNet.Attributes;

/// <summary>
/// Declares that the model transforms its input before the layer stack sees it, so the tensor reaching
/// <c>Layers[0]</c> is NOT the tensor the caller passed to <c>Predict</c>.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> most models feed their input straight into the first layer — the tensor you pass
/// to <c>Predict</c> is exactly the tensor that layer receives. A few do a conversion step first. A
/// time-series foundation model, for example, may turn a run of real-valued measurements into a list of
/// whole-number token IDs before any layer runs, the same way a language model turns text into tokens.
/// After that step the shape has changed completely, and it is the CONVERTED shape the first layer must
/// accept — not the one the model advertises to callers.
/// </para>
/// <para>
/// <b>What it is for.</b> The declared-input boundary check reads the model's own
/// <c>[TensorLayout(Direction = Input)]</c> and compares it against the first layer's, to catch a model
/// advertising an input rank its own stack rejects — a shape on which <c>Predict</c> can never succeed.
/// That comparison is only meaningful when both declarations describe the SAME tensor. Where a conversion
/// step sits between them they describe different tensors, both correctly, and the check would report a
/// defect that is not there. This attribute is how a model says so.
/// </para>
/// <para>
/// <b>Why an attribute and not a virtual property.</b> This is a fact about the TYPE, not about any
/// instance: whether a conversion step exists is fixed by the code, never by a constructor argument. As an
/// attribute it can be read off the open generic type — before anything is constructed, and without
/// closing the type over some particular numeric type just to reach a <c>bool</c>. A checker that had to
/// build an instance first would silently stop exempting any model whose construction failed for an
/// unrelated reason, which is the quiet kind of coverage loss this shape work exists to eliminate.
/// </para>
/// <para>
/// <b>Why the reason is required.</b> This attribute REMOVES a check, so an unexplained one is
/// indistinguishable from a way to silence an inconvenient failure. The reason should name the conversion
/// — the method that performs it — so the next reader can confirm the exemption still holds instead of
/// taking it on trust. It is surfaced in the boundary check's report, not just in the source.
/// </para>
/// <para>
/// <b>When NOT to use this.</b> If the model folds its input directly into the stack
/// (<c>foreach (var layer in Layers) current = layer.Forward(current);</c>), a mismatch between the two
/// declarations is a REAL defect and this attribute would hide it. Fix whichever declaration is wrong
/// instead: either the model advertises a rank it cannot serve, or the first layer is the wrong layer for
/// the input the model accepts. Reshaping alone does not qualify either — a rank change is exactly what
/// the two declarations are supposed to disagree about visibly.
/// </para>
/// <example>
/// <code>
/// // Chronos.PredictCore calls Forward(Tokenize(input)): the embedding layer receives token indices,
/// // never the raw series, so the model's rank-3 input and the layer's rank-1/2 input are both correct.
/// [PreprocessesInput("Tokenize converts the real-valued series to token IDs before the stack runs.")]
/// public class Chronos&lt;T&gt; : NeuralNetworkBase&lt;T&gt;
/// </code>
/// </example>
/// </remarks>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = true)]
public sealed class PreprocessesInputAttribute : Attribute
{
    /// <summary>
    /// Creates the declaration.
    /// </summary>
    /// <param name="reason">
    /// Why the layer stack does not see <c>Predict</c>'s input, naming the conversion that sits between
    /// them so the exemption can be re-checked later rather than trusted.
    /// </param>
    /// <exception cref="ArgumentException">The reason is null, empty, or whitespace.</exception>
    public PreprocessesInputAttribute(string reason)
    {
        if (string.IsNullOrWhiteSpace(reason))
        {
            throw new ArgumentException(
                "A preprocessing declaration must say what transforms the input, because it removes a "
                + "shape check and an unexplained exemption cannot be distinguished from a silenced one.",
                nameof(reason));
        }

        Reason = reason;
    }

    /// <summary>
    /// Why the layer stack does not see <c>Predict</c>'s input.
    /// </summary>
    public string Reason { get; }
}

/// <summary>
/// Declares the tensor layout produced by a model's preprocessing step and consumed by
/// <c>Layers[0]</c>.
/// </summary>
/// <remarks>
/// This is the other half of <see cref="PreprocessesInputAttribute"/>. The caller-facing
/// <see cref="TensorLayoutAttribute"/> remains the public input contract; this declaration makes the
/// transformed stack boundary equally explicit so preprocessing cannot become an unchecked exemption.
/// Multiple declarations are allowed when the preprocessing method supports more than one layout.
/// </remarks>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = true)]
public sealed class StackInputLayoutAttribute : Attribute
{
    /// <summary>Creates a stack-entry layout from its ordered semantic axes.</summary>
    public StackInputLayoutAttribute(params TensorAxis[] axes)
    {
        Axes = axes ?? throw new ArgumentNullException(nameof(axes));
        if (axes.Length == 0)
            throw new ArgumentException("A stack-input layout must declare at least one axis.", nameof(axes));
    }

    /// <summary>The ordered axes delivered to the first layer after preprocessing.</summary>
    public TensorAxis[] Axes { get; }

    /// <summary>Whether the leading batch axis may be omitted.</summary>
    public bool BatchOptional { get; set; }

    /// <summary>Whether the declaration accepts <paramref name="rank"/>.</summary>
    public bool AcceptsRank(int rank)
        => rank == Axes.Length
           || (BatchOptional
               && Axes.Length > 1
               && Axes[0] == TensorAxis.Batch
               && rank == Axes.Length - 1);
}

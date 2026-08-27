namespace AiDotNet.Attributes;

/// <summary>
/// Marks mathematical model parameters that become available through <c>Fit</c> rather than a
/// gradient step. The generator keeps the slot in the manifest before fitting and derives all
/// count, read, write and restore behavior after materialization.
/// </summary>
[AttributeUsage(AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = false, Inherited = false)]
public sealed class FittedParameterAttribute : Attribute
{
    /// <summary>Gets the lifecycle for fitted state.</summary>
    public AiDotNet.Models.Parameters.ParameterAvailability Availability { get; set; }
        = AiDotNet.Models.Parameters.ParameterAvailability.Fit;

    /// <summary>
    /// Declares that this member's extent comes from the DATA the caller supplies rather than from
    /// the layer's construction arguments, and so must be persisted without joining the flat
    /// parameter vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Ordinary fitted state -- a reservoir's fixed weights, a normalization layer's running
    /// statistics -- is sized once at construction and never changes again, so carrying it in the
    /// flat vector is what lets one vector describe the whole layer, which is the guarantee
    /// <c>state_dict()</c> does not offer.
    /// </para>
    /// <para>
    /// Input-sized state cannot honor that guarantee. A graph layer's adjacency matrix is
    /// <c>[numNodes, numNodes]</c> for whatever graph was handed in last, so putting it in the
    /// vector makes <c>ParameterCount</c> a function of the most recent input: the width changes
    /// under a caller who only ran a forward pass, and a checkpoint taken on a ten-node graph
    /// could never restore into a twenty-node one. Both are true of the same weights, which is
    /// what makes it a property of the member rather than of the moment.
    /// </para>
    /// <para>
    /// Marked members are still registered through <c>RegisterBuffer</c>, so they are written and
    /// read by name in the layer's serialized buffer block and copied by <c>DeepCopy</c>. They are
    /// simply absent from the parameter vector, where their width was never meaningful.
    /// </para>
    /// </remarks>
    public bool InputSized { get; set; }
}

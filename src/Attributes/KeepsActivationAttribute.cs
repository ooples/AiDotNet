namespace AiDotNet.Attributes;

/// <summary>
/// Declares that a field deliberately holds an activation across calls, and says why.
/// </summary>
/// <remarks>
/// <para>
/// ADN0065 reports a backward-activation cache written in a forward without a training-pass guard,
/// because an ungated write keeps the activation reachable after the pass and makes peak inference
/// memory O(sum of activations). Some fields keep an activation on purpose: recurrent state carried
/// between frames, embeddings a later public call reads, a value a loss consumes after the forward.
/// Such a field is not a backward cache, so gating it would be a bug.
/// </para>
/// <para>
/// This attribute is the reviewed exemption. It names the field and records why it outlives the
/// forward, so the keep is visible at the declaration rather than silently allowlisted. A blank reason
/// claims nothing, so the analyzer reports it (ADN0068) and the constructor rejects it.
/// </para>
/// <para>
/// A field kept only so a test can read it after the forward is not a deliberate keep. Remove the
/// field, or expose what the test needs through the API, rather than exempting it.
/// </para>
/// <para><b>For Beginners:</b> most layers remember their last input only so the backward pass can
/// use it, and should forget it when no backward is coming. Put this on a field only when the layer
/// genuinely needs the value again on a later call.</para>
/// </remarks>
[AttributeUsage(AttributeTargets.Field, AllowMultiple = false, Inherited = false)]
public sealed class KeepsActivationAttribute : Attribute
{
    /// <summary>Why the field must keep its value beyond the forward pass that wrote it.</summary>
    public string Reason { get; }

    /// <summary>Creates the declaration.</summary>
    /// <param name="reason">Why the field must keep its value beyond the forward pass that wrote it.</param>
    public KeepsActivationAttribute(string reason)
    {
        if (reason is null) throw new ArgumentNullException(nameof(reason));
        if (string.IsNullOrWhiteSpace(reason))
            throw new ArgumentException("Why the field keeps its activation across calls must be stated.", nameof(reason));
        Reason = reason;
    }
}

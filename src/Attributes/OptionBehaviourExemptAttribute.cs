namespace AiDotNet.Attributes;

/// <summary>
/// Declares that an options property legitimately cannot change a model's observable behaviour at
/// test scale, and records why, so the generated option-sensitivity test does not report it.
/// </summary>
/// <remarks>
/// <para>
/// Every public settable property on a model's options class gets a generated test that builds the
/// model twice with one fixed seed - once as configured, once with that single property perturbed -
/// and requires the two to differ in parameter count, in their outputs on a fixed input, or in their
/// parameters after a few fixed training steps. A property that changes nothing is either ignored by
/// the model (the defect the test exists to find: Dreamer, World Models and Decision Transformer all
/// stored options they never read) or genuinely has no effect within a unit-test budget.
/// </para>
/// <para>
/// This attribute is for the second case only, and the reason is mandatory: a blank reason is a build
/// error (ADNOPT004). It mirrors how a cache that survives a clone must carry an exemption attribute
/// with a stated justification - an exemption nobody had to explain is indistinguishable from a
/// defect someone chose to hide.
/// </para>
/// <para>
/// An exemption does NOT silence ADNOPT001. A property that nothing in the library ever reads is dead
/// configuration at every scale, not merely at test scale, so the static analyzer still reports it.
/// </para>
/// <para>
/// <b>For Beginners:</b> Options are the knobs on a model. The tests turn each knob a little and check
/// that the model notices. Some knobs only matter in situations a quick test never reaches - for
/// example, a checkpoint interval of 10,000 steps never fires in a test that trains for three. Put
/// this attribute on such a knob and say, in plain words, why turning it cannot show up in a test.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// using AiDotNet.Attributes;
/// using AiDotNet.Models.Options;
///
/// public class MyTrainerOptions : ModelOptions
/// {
///     [OptionBehaviourExempt("Checkpoints are written every N steps; the generated test trains for fewer steps than the smallest valid N.")]
///     public int CheckpointInterval { get; set; } = 10000;
/// }
/// </code>
/// </example>
[AttributeUsage(AttributeTargets.Property, AllowMultiple = false, Inherited = true)]
public sealed class OptionBehaviourExemptAttribute : Attribute
{
    /// <summary>Exempts a property from the generated option-sensitivity test.</summary>
    /// <param name="reason">
    /// Why perturbing this property cannot change observable behaviour at test scale. Required and
    /// enforced non-blank by ADNOPT004; it is shown as the skip reason of the generated test.
    /// </param>
    public OptionBehaviourExemptAttribute(string reason)
    {
        Reason = reason;
    }

    /// <summary>Why this property cannot change observable behaviour at test scale.</summary>
    public string Reason { get; }
}

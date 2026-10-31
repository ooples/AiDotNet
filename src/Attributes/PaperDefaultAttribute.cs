namespace AiDotNet.Attributes;

/// <summary>
/// Declares the value a model's research paper specifies for one options property, with the place in
/// the paper it comes from, so a generated test can check that the options constructor really ships it.
/// </summary>
/// <remarks>
/// <para>
/// <c>[PaperOptimizer]</c> records a paper's training recipe on the model class, because a recipe is
/// one unit: optimizer, learning rate, schedule and clipping only make sense together. Architecture and
/// algorithm hyperparameters are the opposite - each is an independent claim about one number - so
/// this attribute sits on the individual property it describes. Keeping the declared value next to the
/// default it constrains means a reviewer sees both on one line, a renamed property carries its
/// declaration with it, and no property name is ever spelled as a string.
/// </para>
/// <para>
/// The paper itself is the <c>[ResearchPaper]</c> of the model the options configure, so it is not
/// repeated here. <see cref="Source"/> names WHERE in that paper the value is stated and is required
/// (ADNOPT003), for the same reason <c>PaperOptimizer.Source</c> is: an uncited number reads as
/// authoritative and is never re-checked.
/// </para>
/// <para>
/// Only declare a value the paper actually states. A property the paper is silent on stays
/// undeclared; it is not a defect for a library default to fill a gap the authors left.
/// </para>
/// <para>
/// <b>What is checked.</b> At build time ADNOPT003 rejects a declaration with no source or with a value
/// whose type cannot be assigned to the property. At test time a generated test constructs the options
/// class with its parameterless constructor and requires every declared property to hold exactly the
/// declared value (numbers are compared after conversion to <see cref="double"/>, so a generic
/// <c>T LearningRate</c> can be declared as <c>0.001</c>).
/// </para>
/// <para>
/// <b>For Beginners:</b> A paper says things like "we use a discount factor of 0.99". This attribute
/// writes that sentence down next to the setting, with the page or table it came from, and the tests
/// then make sure the library's default really is 0.99.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// using AiDotNet.Attributes;
/// using AiDotNet.Models.Options;
///
/// public class MyAgentOptions : ModelOptions
/// {
///     [PaperDefault(0.99, Source = "Extended Data Table 1: discount factor")]
///     public double DiscountFactor { get; set; } = 0.99;
///
///     [PaperDefault(new[] { 64, 64 }, Source = "Sec. 5.1: two hidden layers of 64 units")]
///     public int[] HiddenLayers { get; set; } = { 64, 64 };
/// }
/// </code>
/// </example>
[AttributeUsage(AttributeTargets.Property, AllowMultiple = false, Inherited = true)]
public sealed class PaperDefaultAttribute : Attribute
{
    /// <summary>Declares the paper's value for this property.</summary>
    /// <param name="value">
    /// The value the paper states: a number, <see langword="bool"/>, enum member, string, or a
    /// one-dimensional array of those for a list-valued property.
    /// </param>
    public PaperDefaultAttribute(object value)
    {
        Value = value;
    }

    /// <summary>The value the paper states for this property.</summary>
    public object Value { get; }

    /// <summary>
    /// Where in the paper the value is stated, for example <c>"Sec. 4.1, Table 8"</c>. Required and
    /// enforced by ADNOPT003.
    /// </summary>
    public string Source { get; set; } = string.Empty;
}

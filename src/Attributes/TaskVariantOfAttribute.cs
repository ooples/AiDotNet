namespace AiDotNet.Attributes;

/// <summary>
/// Declares that a model is the same network as another model of the same paper, adapted to a different task.
/// </summary>
/// <remarks>
/// <para>
/// One architecture often serves several tasks: SAM is a promptable segmenter and an image encoder, ResNet a classifier
/// and a detection backbone. Each task gets its own model type, and both types cite the paper. That is legitimate only
/// when the two really run one network; two separate implementations of one paper drift apart, and one of them goes
/// untested (ADNDEF001).
/// </para>
/// <para>
/// The analyzer verifies the claim: the two models must both construct (directly or through the types they construct)
/// a type, or call a LayerHelper factory, that no model of another paper uses. That shared piece is the network. A
/// declaration it cannot verify is an error (ADNDEF005).
/// </para>
/// </remarks>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = false)]
public sealed class TaskVariantOfAttribute : Attribute
{
    /// <summary>The model this one shares its network with, as an open generic type (for example <c>typeof(SAM&lt;&gt;)</c>).</summary>
    public Type Canonical { get; }

    /// <summary>The task this model adapts the network to.</summary>
    public string Task { get; }

    /// <summary>Creates the declaration.</summary>
    public TaskVariantOfAttribute(Type canonical, string task)
    {
        if (canonical is null) throw new ArgumentNullException(nameof(canonical));
        if (task is null) throw new ArgumentNullException(nameof(task));
        if (string.IsNullOrWhiteSpace(task))
            throw new ArgumentException("The task this variant adapts the network to must be stated.", nameof(task));
        Canonical = canonical;
        Task = task;
    }
}

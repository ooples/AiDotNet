namespace AiDotNet.Models;

/// <summary>
/// Lets a type finish a configuration copy with state its public setters cannot carry.
/// </summary>
/// <remarks>
/// <see cref="CloneEngine.CopyConfiguration"/> copies through public setters, so a setter that records more than the
/// value -- such as whether the caller chose it -- records it again on the copy. The engine calls
/// <see cref="CompleteConfigurationCopy"/> after every setter has run, so the copy can take that state from its source.
/// </remarks>
internal interface IConfigurationCopyCompletion
{
    /// <summary>Copies from <paramref name="source"/> the state the property setters could not carry.</summary>
    /// <param name="source">The instance this one was copied from; always of the same runtime type.</param>
    void CompleteConfigurationCopy(object source);
}

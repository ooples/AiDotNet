namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for objects that can be cloned or copied.
/// </summary>
/// <typeparam name="T">The type of the object being cloned.</typeparam>
public interface ICloneable<T> where T : class
{
    /// <summary>
    /// Creates a deep copy of this object.
    /// </summary>
    T DeepCopy();

    /// <summary>
    /// Creates a shallow copy of this object.
    /// </summary>
    T Clone();
}

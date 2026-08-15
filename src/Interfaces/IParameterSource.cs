namespace AiDotNet.Interfaces;

/// <summary>
/// Anything that owns trainable parameters and can hand them out and take them back as a flat
/// vector: a model, a sub-model, a layer, or an internal component such as an encoder or projector.
/// </summary>
/// <remarks>
/// <para>
/// This is the minimal contract a parameter registry needs, and deliberately nothing more.
/// <see cref="IParameterizable{T, TInput, TOutput}"/> demands a good deal beyond it — most notably
/// <c>WithParameters</c>, which returns an <c>IFullModel</c> — so requiring it of a component would
/// force internal helpers like <c>ControlNetEncoder</c> or <c>DiffWaveNetwork</c> to pretend to be
/// standalone models. They already have these three members; they simply had no interface to say so.
/// </para>
/// <para>
/// The three are not independent. <see cref="ParameterCount"/> must equal the length of the vector
/// <see cref="GetParameters"/> returns, and <see cref="SetParameters"/> must accept exactly that
/// vector: callers pair them BY LENGTH, so an implementation whose count disagrees with its vector
/// causes a checkpoint to be restored into the wrong tensors, silently, leaving the model on its
/// initial weights. Derive one from the other rather than maintaining both.
/// </para>
/// <para><b>For Beginners:</b> implement this and a containing model can count, save and load your
/// component's numbers without knowing anything else about it.</para>
/// </remarks>
/// <typeparam name="T">The numeric type of the parameters.</typeparam>
public interface IParameterSource<T>
{
    /// <summary>
    /// How many parameters this owns. Must equal <c>GetParameters().Length</c>.
    /// </summary>
    long ParameterCount { get; }

    /// <summary>
    /// All parameters as one flat vector, in a stable order.
    /// </summary>
    /// <remarks>The order is the serialization order: <see cref="SetParameters"/> must read it back
    /// in exactly the same order.</remarks>
    Vector<T> GetParameters();

    /// <summary>
    /// Restores parameters from a vector produced by <see cref="GetParameters"/>.
    /// </summary>
    void SetParameters(Vector<T> parameters);
}

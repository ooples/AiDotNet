using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>Stores transitions with the legal action set of the resulting state.</summary>
/// <typeparam name="T">Numeric element type.</typeparam>
public interface IMaskedExperienceAgent<T>
{
    /// <summary>
    /// Stores a transition. A null mask means unrestricted; terminal transitions do not bootstrap.
    /// Implementations must snapshot non-null masks or reject unsupported masked transitions.
    /// </summary>
    void StoreExperience(Vector<T> state, Vector<T> action, T reward, Vector<T> nextState,
        bool done, bool[]? nextLegalActions);
}

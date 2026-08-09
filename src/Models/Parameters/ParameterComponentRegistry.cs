using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Models.Parameters;

/// <summary>
/// Holds the components a model's parameters live in and folds them into the count, the vector and
/// the restore, so those three cannot describe different things.
/// </summary>
/// <remarks>
/// <para>
/// The library has several unrelated model roots -- <c>ModelBase</c>, <c>ModelWrapperBase</c>,
/// <c>LayerBase</c>, <c>ReinforcementLearningAgentBase</c>, <c>CausalModelBase</c>,
/// <c>SurvivalModelBase</c> and more -- because a causal estimator and a diffusion U-Net genuinely
/// have nothing to inherit from each other. Each still needs the same fold, and C# gives them no
/// common place to put it. Holding this registry as a field is that common place: a base wires
/// three one-line members to it and the fold itself exists ONCE.
/// </para>
/// <para>
/// Registration order is serialization order. Register only what is TRAINED -- a target network, a
/// frozen teacher or a cached projection is not an independent parameter, and registering one hands
/// an optimizer weights that are only ever meant to be written to.
/// </para>
/// </remarks>
public sealed class ParameterComponentRegistry<T>
{
    private readonly List<IParameterSource<T>> _components = new();

    /// <summary>The registered components, in registration order.</summary>
    public IReadOnlyList<IParameterSource<T>> Components => _components;

    /// <summary>True once anything has been registered.</summary>
    public bool HasComponents => _components.Count > 0;

    /// <summary>
    /// Adds a component. Null is tolerated, so a model may register something a configuration did
    /// not build, and registration is idempotent by reference.
    /// </summary>
    public void Register(IParameterSource<T>? component)
    {
        if (component is null) return;
        for (int i = 0; i < _components.Count; i++)
        {
            if (ReferenceEquals(_components[i], component)) return;
        }
        _components.Add(component);
    }

    /// <summary>Sum of the registered components' counts.</summary>
    public long ParameterCount
    {
        get
        {
            long total = 0;
            for (int i = 0; i < _components.Count; i++) total += _components[i].ParameterCount;
            return total;
        }
    }

    /// <summary>Concatenates the components in registration order.</summary>
    public Vector<T> GetParameters()
    {
        if (_components.Count == 0) return new Vector<T>(0);

        var parts = new Vector<T>[_components.Count];
        int total = 0;
        for (int i = 0; i < _components.Count; i++)
        {
            parts[i] = _components[i].GetParameters();
            total += parts[i].Length;
        }

        var result = new Vector<T>(total);
        int offset = 0;
        for (int i = 0; i < parts.Length; i++)
        {
            for (int j = 0; j < parts[i].Length; j++) result[offset++] = parts[i][j];
        }
        return result;
    }

    /// <summary>
    /// The inverse of <see cref="GetParameters"/>: each component takes back the slice it
    /// contributed.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A component that declares <see cref="IVariableLengthParameterSource{T}"/> is sized BY the
    /// incoming vector rather than checked against it, and takes whatever the fixed-size components
    /// leave. At most one may do so, and it must be registered last, or "the remainder" would not
    /// identify a unique slice.
    /// </para>
    /// <para>
    /// That exists because a model can legitimately not know its own width yet. A propensity model
    /// has no coefficients until it is fitted, so a fresh instance declares zero -- and restoring a
    /// checkpoint INTO a fresh instance is the whole point of a checkpoint. Sizing it strictly made
    /// every such load throw.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">
    /// The vector's length does not match what the components declare. Reported rather than
    /// absorbed: a restore that silently takes a wrong-length vector is how a checkpoint goes
    /// missing.
    /// </exception>
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));

        int variableIndex = -1;
        for (int i = 0; i < _components.Count; i++)
        {
            if (_components[i] is IVariableLengthParameterSource<T>)
            {
                if (variableIndex >= 0)
                {
                    throw new InvalidOperationException(
                        "Only one variable-length component may be registered: with two, " +
                        "'the remainder' does not identify a unique slice.");
                }
                variableIndex = i;
            }
        }

        if (variableIndex >= 0 && variableIndex != _components.Count - 1)
        {
            throw new InvalidOperationException(
                "A variable-length component must be registered LAST, so the fixed-size " +
                "components ahead of it determine where its slice begins.");
        }

        long fixedTotal = 0;
        for (int i = 0; i < _components.Count; i++)
        {
            if (i != variableIndex) fixedTotal += _components[i].ParameterCount;
        }

        if (variableIndex < 0)
        {
            if (parameters.Length != fixedTotal)
            {
                throw new ArgumentException(
                    $"Expected {fixedTotal} parameters, got {parameters.Length}.", nameof(parameters));
            }
        }
        else if (parameters.Length < fixedTotal)
        {
            throw new ArgumentException(
                $"Expected at least {fixedTotal} parameters for the fixed-size components, " +
                $"got {parameters.Length}.", nameof(parameters));
        }

        int offset = 0;
        for (int i = 0; i < _components.Count; i++)
        {
            int n = i == variableIndex
                ? parameters.Length - offset
                : checked((int)_components[i].ParameterCount);
            var slice = new Vector<T>(n);
            for (int j = 0; j < n; j++) slice[j] = parameters[offset++];
            _components[i].SetParameters(slice);
        }
    }
}

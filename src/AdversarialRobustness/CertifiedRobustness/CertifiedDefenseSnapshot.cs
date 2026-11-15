using System;
using AiDotNet.AdversarialRobustness.Defenses;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Serialization;
using AiDotNet.Validation;
using Newtonsoft.Json;

namespace AiDotNet.AdversarialRobustness.CertifiedRobustness;

/// <summary>
/// Creates the certification engine named by <see cref="CertifiedDefenseOptions{T}.CertificationMethod"/>.
/// </summary>
internal static class CertifiedDefenseFactory<T, TInput, TOutput>
{
    /// <summary>The engine of <paramref name="options"/>.</summary>
    public static ICertifiedDefense<T, TInput, TOutput> Create(CertifiedDefenseOptions<T> options)
    {
        return options.CertificationMethod switch
        {
            CertifiedDefenseMethod.RandomizedSmoothing => new RandomizedSmoothing<T, TInput, TOutput>(options),
            CertifiedDefenseMethod.IntervalBoundPropagation => new IntervalBoundPropagation<T, TInput, TOutput>(options),
            CertifiedDefenseMethod.Crown => new CROWNVerification<T, TInput, TOutput>(options),
            CertifiedDefenseMethod.AdaptiveRandomizedSmoothing => new AdaptiveRandomizedSmoothing<T, TInput, TOutput>(options),
            _ => throw new InvalidOperationException(
                $"AdversarialRobustnessOptions.CertificationMethod has the undefined value {(int)options.CertificationMethod}.")
        };
    }

    /// <summary>The method whose engine is exactly the type of <paramref name="defense"/>, or null for other types.</summary>
    public static CertifiedDefenseMethod? BuiltInMethodOf(ICertifiedDefense<T, TInput, TOutput> defense)
    {
        var type = defense.GetType();
        if (type == typeof(RandomizedSmoothing<T, TInput, TOutput>))
        {
            return CertifiedDefenseMethod.RandomizedSmoothing;
        }

        if (type == typeof(IntervalBoundPropagation<T, TInput, TOutput>))
        {
            return CertifiedDefenseMethod.IntervalBoundPropagation;
        }

        if (type == typeof(CROWNVerification<T, TInput, TOutput>))
        {
            return CertifiedDefenseMethod.Crown;
        }

        if (type == typeof(AdaptiveRandomizedSmoothing<T, TInput, TOutput>))
        {
            return CertifiedDefenseMethod.AdaptiveRandomizedSmoothing;
        }

        return null;
    }
}

/// <summary>
/// The persisted form of a certified defense: its options, its own serialized state and, for a defense that is not
/// one of the library's engines, its type.
/// </summary>
/// <remarks>
/// <para>
/// Restoring builds the engine from the persisted options (so a seeded randomized-smoothing engine starts from the
/// same seed) and then applies the engine's own <c>IModelSerializer.Deserialize</c>. A custom defense is
/// rebuilt from its type through the <see cref="SafeSerializationBinder"/>, with a public constructor taking
/// <see cref="CertifiedDefenseOptions{T}"/> or a parameterless one.
/// </para>
/// </remarks>
internal sealed class CertifiedDefenseSnapshot<T>
{
    [JsonConstructor]
    public CertifiedDefenseSnapshot(
        CertifiedDefenseMethod? builtInMethod,
        string? customTypeName,
        string? customAssemblyName,
        CertifiedDefenseOptions<T>? options,
        byte[]? state)
    {
        BuiltInMethod = builtInMethod;
        CustomTypeName = customTypeName;
        CustomAssemblyName = customAssemblyName;
        Options = options ?? new CertifiedDefenseOptions<T>();
        State = state ?? Array.Empty<byte>();
    }

    /// <summary>The library engine of the defense, or null for a custom defense.</summary>
    public CertifiedDefenseMethod? BuiltInMethod { get; }

    /// <summary>The full type name of a custom defense.</summary>
    public string? CustomTypeName { get; }

    /// <summary>The assembly of a custom defense.</summary>
    public string? CustomAssemblyName { get; }

    /// <summary>The options of the defense.</summary>
    public CertifiedDefenseOptions<T> Options { get; }

    /// <summary>The defense's own serialized state.</summary>
    public byte[] State { get; }

    /// <summary>Captures <paramref name="defense"/>.</summary>
    public static CertifiedDefenseSnapshot<T> Capture<TInput, TOutput>(ICertifiedDefense<T, TInput, TOutput> defense)
    {
        Guard.NotNull(defense);
        byte[] state;
        using (ModelPersistenceGuard.InternalOperation())
        {
            state = defense.Serialize();
        }

        var builtIn = CertifiedDefenseFactory<T, TInput, TOutput>.BuiltInMethodOf(defense);
        var type = defense.GetType();
        return new CertifiedDefenseSnapshot<T>(
            builtIn,
            builtIn.HasValue ? null : type.FullName,
            builtIn.HasValue ? null : type.Assembly.FullName,
            defense.GetOptions(),
            state);
    }

    /// <summary>Rebuilds the defense.</summary>
    /// <exception cref="InvalidOperationException">The custom defense type cannot be resolved or constructed.</exception>
    public ICertifiedDefense<T, TInput, TOutput> Restore<TInput, TOutput>()
    {
        ICertifiedDefense<T, TInput, TOutput> defense;
        if (BuiltInMethod is CertifiedDefenseMethod method)
        {
            Options.CertificationMethod = method;
            defense = CertifiedDefenseFactory<T, TInput, TOutput>.Create(Options);
        }
        else
        {
            defense = CreateCustom<TInput, TOutput>();
        }

        if (State.Length > 0)
        {
            using (ModelPersistenceGuard.InternalOperation())
            {
                defense.Deserialize(State);
            }
        }

        return defense;
    }

    private ICertifiedDefense<T, TInput, TOutput> CreateCustom<TInput, TOutput>()
    {
        string typeName = CustomTypeName ?? throw new InvalidOperationException(
            "The persisted certified defense names neither a library engine nor a custom type.");

        // The binder admits AiDotNet types only, so a payload cannot name an arbitrary type to construct.
        Type type = new SafeSerializationBinder().BindToType(CustomAssemblyName, typeName);
        if (!typeof(ICertifiedDefense<T, TInput, TOutput>).IsAssignableFrom(type) || type.IsAbstract)
        {
            throw new InvalidOperationException(
                $"The persisted certified defense type '{typeName}' is not a concrete ICertifiedDefense of this model.");
        }

        var withOptions = type.GetConstructor(new[] { typeof(CertifiedDefenseOptions<T>) });
        object? instance = withOptions is not null
            ? withOptions.Invoke(new object[] { Options })
            : type.GetConstructor(Type.EmptyTypes)?.Invoke(Array.Empty<object>());

        return instance as ICertifiedDefense<T, TInput, TOutput>
            ?? throw new InvalidOperationException(
                $"The persisted certified defense type '{typeName}' has no public constructor taking " +
                "CertifiedDefenseOptions or no arguments, so it cannot be rebuilt on load.");
    }
}

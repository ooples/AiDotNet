using System.Reflection;
using AiDotNet.Interfaces;

namespace AiDotNet.Models.Parameters;

/// <summary>
/// Shared fallback for parameter-only model families that do not inherit the generated model
/// registry. It discovers parameter-bearing members declared by a concrete variant while keeping
/// the family base as the sole owner of ordering, counting, reading, and restoring parameters.
/// </summary>
internal static class GeneratedParameterDiscovery
{
    private const BindingFlags DeclaredInstance =
        BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.DeclaredOnly;

    /// <summary>Enumerates parameter sources declared below <paramref name="familyBase"/>.</summary>
    internal static IEnumerable<IParameterSource<T>> EnumerateDerivedSources<T>(
        object owner,
        Type familyBase)
    {
        var seen = new HashSet<object>(AiDotNet.Helpers.TensorReferenceComparer<object>.Instance);

        for (var current = owner.GetType(); current is not null && current != familyBase;
             current = current.BaseType)
        {
            foreach (var field in current.GetFields(DeclaredInstance).OrderBy(field => field.MetadataToken))
            {
                if (IsExcluded(field)) continue;
                if (field.GetValue(owner) is IParameterSource<T> source && seen.Add(source))
                    yield return source;
            }
        }
    }

    /// <summary>Enumerates layer-shaped parameter sources declared by a concrete variant.</summary>
    internal static IEnumerable<ILayer<T>> EnumerateDerivedLayers<T>(object owner, Type familyBase)
    {
        foreach (var source in EnumerateDerivedSources<T>(owner, familyBase))
        {
            if (source is ILayer<T> layer) yield return layer;
        }
    }

    private static bool IsExcluded(FieldInfo field)
    {
        foreach (var attribute in field.GetCustomAttributes(inherit: true))
        {
            if (attribute.GetType().Name is "ScratchAttribute" or "BufferAttribute"
                or "ParameterAliasAttribute" or "ExternalStateAttribute"
                or "ExternalResourceAttribute")
            {
                return true;
            }
        }

        return false;
    }
}

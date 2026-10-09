using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using AiDotNet.Diffusion.ThreeD;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion;

/// <summary>
/// #2087: MVDreamModel's configuration copy received the source's MultiViewUNet by reference, because that
/// type had a public Clone but did not declare ICloneable, which is what CloneEngine looks for. The clone and
/// the original then shared every base U-Net layer, and Clone threw AliasedLayerGraph.
/// </summary>
public class MVDreamCloneTests
{
    [Fact]
    public void Clone_SharesNoLayerWithTheSource()
    {
        var model = new MVDreamModel<float>(
            multiViewUNet: new MultiViewUNet<float>(
                inputChannels: 4, outputChannels: 4, baseChannels: 8, numViews: 2, contextDim: 16, seed: 1),
            imageVAE: new StandardVAE<float>(
                inputChannels: 3, latentChannels: 4, baseChannels: 8,
                channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1, seed: 1),
            seed: 1);

        var clone = model.Clone();

        Assert.Equal(model.GetParameters().ToArray(), clone.GetParameters().ToArray());

        var sourceLayers = ReachableLayers(model);
        Assert.NotEmpty(sourceLayers);
        var shared = new List<string>();
        foreach (var layer in ReachableLayers(clone))
        {
            if (sourceLayers.Contains(layer)) shared.Add(layer.GetType().Name);
        }

        Assert.True(shared.Count == 0,
            $"The clone shares {shared.Count} layer object(s) with its source, first {string.Join(", ", shared.GetRange(0, System.Math.Min(5, shared.Count)))}.");
    }

    [Fact]
    public void DeepCopy_IsAnIndependentCopyWithEqualParameters()
    {
        var source = new MultiViewUNet<float>(
            inputChannels: 4, outputChannels: 4, baseChannels: 8, numViews: 2, contextDim: 16, seed: 1);
        var before = source.GetParameters().ToArray();

        var copy = source.DeepCopy();

        Assert.NotSame(source, copy);
        Assert.Equal(before, copy.GetParameters().ToArray());

        // Writing the copy's weights must not reach the source: shared buffers or layers would.
        var changed = copy.GetParameters().ToArray();
        for (int i = 0; i < changed.Length; i++) changed[i] += 1f;
        copy.SetParameters(new AiDotNet.Tensors.LinearAlgebra.Vector<float>(changed));

        Assert.Equal(changed, copy.GetParameters().ToArray());
        Assert.Equal(before, source.GetParameters().ToArray());
        Assert.Empty(SharedLayers(source, copy));
    }

    private static List<string> SharedLayers(object source, object copy)
    {
        var sourceLayers = ReachableLayers(source);
        var shared = new List<string>();
        foreach (var layer in ReachableLayers(copy))
        {
            if (sourceLayers.Contains(layer)) shared.Add(layer.GetType().Name);
        }
        return shared;
    }

    private static HashSet<object> ReachableLayers(object root)
    {
        var layers = new HashSet<object>(IdentityComparer.Instance);
        var seen = new HashSet<object>(IdentityComparer.Instance);
        var pending = new Stack<object>();
        pending.Push(root);
        while (pending.Count > 0)
        {
            var current = pending.Pop();
            var type = current.GetType();
            // Walk the library's own objects and any collection that may hold them; skip every other
            // framework object, whose internals cannot hold a layer.
            bool libraryType = type.Namespace is not null
                && type.Namespace.StartsWith("AiDotNet", System.StringComparison.Ordinal);
            if (!seen.Add(current) || type.IsPrimitive || current is string || current is System.Delegate
                || (!libraryType && current is not IEnumerable))
            {
                continue;
            }

            if (current is ILayer<float>) layers.Add(current);
            if (type.Namespace is not null && type.Namespace.StartsWith("AiDotNet.Tensors", System.StringComparison.Ordinal)) continue;

            if (current is IEnumerable sequence)
            {
                foreach (var item in sequence)
                {
                    if (item is not null && !item.GetType().IsValueType) pending.Push(item);
                }
                if (type.IsArray) continue;
            }

            for (var declaring = type; declaring is not null && declaring != typeof(object); declaring = declaring.BaseType)
            {
                foreach (var field in declaring.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
                {
                    if (field.FieldType.IsValueType) continue;
                    if (field.GetValue(current) is { } value) pending.Push(value);
                }
            }
        }

        return layers;
    }

    // System.Collections.Generic.ReferenceEqualityComparer is public only from .NET 5; net471, which this project also
    // targets, needs its own.
    private sealed class IdentityComparer : IEqualityComparer<object>
    {
        public static readonly IdentityComparer Instance = new IdentityComparer();
        public new bool Equals(object? x, object? y) => ReferenceEquals(x, y);
        public int GetHashCode(object obj) => System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(obj);
    }
}

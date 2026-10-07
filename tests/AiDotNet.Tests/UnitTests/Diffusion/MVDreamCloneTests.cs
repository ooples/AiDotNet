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

    private static HashSet<object> ReachableLayers(object root)
    {
        var layers = new HashSet<object>(ReferenceEqualityComparer.Instance);
        var seen = new HashSet<object>(ReferenceEqualityComparer.Instance);
        var pending = new Stack<object>();
        pending.Push(root);
        while (pending.Count > 0)
        {
            var current = pending.Pop();
            var type = current.GetType();
            if (!seen.Add(current) || type.IsPrimitive || current is string || current is System.Delegate
                || type.Namespace is null || !type.Namespace.StartsWith("AiDotNet", System.StringComparison.Ordinal)
                && current is not IEnumerable)
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
}

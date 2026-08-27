using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.Parameters;

/// <summary>
/// A tensor may belong to exactly one owner in the parameter traversal.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="LayerBase{T}"/> builds its ordered component list from three sources and deduplicates
/// between them: the generated declarations, then every tensor from the trainable view that was not
/// already declared, then every registered sub-layer that was not already declared. The sub-layer
/// check compares LAYER references, so it cannot see that a child's tensors already entered the list
/// through the parent's own trainable view.
/// </para>
/// <para>
/// That makes one shape of hand-written override silently wrong: a composite whose
/// GetTrainableParameters FLATTENS its children's tensors into its own list, while those same
/// children are also registered as sub-layers. Each child's weights then appear twice — once as the
/// parent's trainable tensors and once inside the child component — and the count, the vector and
/// every offset after them are wrong together. Because the count and the vector are both derived
/// from that one doubled walk, they still agree with each other, which is why nothing reports it.
/// </para>
/// <para>
/// The invariant is checkable from outside: no tensor the parent hands out may be the same object a
/// descendant hands out. It holds for a composite that declares its children and lets the base
/// compose them, and fails for one that flattens them by hand.
/// </para>
/// </remarks>
public class CompositeLayerOwnershipTests
{
    public static IEnumerable<object[]> Composites()
    {
        yield return Row("BiaffineSpanScorer", () => new BiaffineSpanScorerLayer<double>(8, 6, 3));
        yield return Row("GatedFusion", () => new GatedFusionLayer<double>(8));
        yield return Row("ClozeAttention", () => new ClozeAttentionLayer<double>(8));
        yield return Row("Branchformer", () => new BranchformerBlock<double>(8, 2, 16, 3));
        yield return Row("CifAlignment", () => new CifAlignmentLayer<double>(8));
    }

    private static object[] Row(string name, Func<LayerBase<double>> factory)
        => new object[] { name, factory };

    [Theory(Timeout = 120000)]
    [MemberData(nameof(Composites))]
    public async Task NoTensorIsOwnedByBothAParentAndItsChild(string name, Func<LayerBase<double>> factory)
    {
        await Task.Yield();
        var layer = factory();

        var ownTensors = layer.GetTrainableParameters();
        var byIdentity = new HashSet<Tensor<double>>(ReferenceEqualityComparer<Tensor<double>>.Instance);
        for (int i = 0; i < ownTensors.Count; i++)
        {
            if (ownTensors[i] is not null) byIdentity.Add(ownTensors[i]);
        }

        var duplicated = new List<string>();
        CollectDuplicates(layer, byIdentity, duplicated, name);

        Assert.True(duplicated.Count == 0,
            $"{name} hands out {duplicated.Count} tensor(s) that a registered sub-layer also hands " +
            "out, so each is counted once as the parent's own parameter and again inside the child " +
            $"component: {string.Join(", ", duplicated)}.");
    }

    private static void CollectDuplicates(
        LayerBase<double> parent,
        HashSet<Tensor<double>> parentTensors,
        List<string> duplicated,
        string path)
    {
        var children = parent.GetSubLayers();
        if (children is null) return;

        for (int c = 0; c < children.Count; c++)
        {
            if (children[c] is not LayerBase<double> child) continue;

            var childTensors = child.GetTrainableParameters();
            for (int i = 0; i < childTensors.Count; i++)
            {
                if (childTensors[i] is not null && parentTensors.Contains(childTensors[i]))
                    duplicated.Add($"{path}->{child.GetType().Name}[{i}]");
            }

            CollectDuplicates(child, parentTensors, duplicated, $"{path}->{child.GetType().Name}");
        }
    }

    private sealed class ReferenceEqualityComparer<TItem> : IEqualityComparer<TItem>
        where TItem : class
    {
        internal static readonly ReferenceEqualityComparer<TItem> Instance = new();

        public bool Equals(TItem? x, TItem? y) => ReferenceEquals(x, y);

        public int GetHashCode(TItem obj)
            => System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(obj);
    }
}

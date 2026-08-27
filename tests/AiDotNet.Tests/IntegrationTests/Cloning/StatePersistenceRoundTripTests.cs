using System;
using System.Collections.Generic;
using System.Linq.Expressions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Serialization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Cloning;

/// <summary>
/// Round-trip proofs for the two persistence routes added so every layer could get a clone factory.
/// </summary>
/// <remarks>
/// Both exist because a generic route was not enough, and both fail SILENTLY if they regress: the
/// value comes back structurally valid but wrong, so a factory still builds and a clone still looks
/// like a clone.
/// </remarks>
public class StatePersistenceRoundTripTests
{
    /// <summary>The metadata must survive its own SaveState/LoadState exactly.</summary>
    /// <remarks>
    /// EdgeTypeSchema is a Dictionary&lt;string, (string, string)&gt;, and System.Text.Json cannot
    /// round-trip a ValueTuple -- its members are FIELDS, so it serialises as {} and the schema
    /// comes back empty with nothing reporting a problem. That is precisely why this type carries
    /// its own state instead of going through the JSON heuristic, and precisely what this asserts.
    /// </remarks>
    [Fact]
    public void GraphMetadata_RoundTripsIncludingItsTupleValuedSchema()
    {
        var original = new HeterogeneousGraphMetadata
        {
            NodeTypes = new[] { "user", "product" },
            EdgeTypes = new[] { "buys", "views" },
            NodeTypeFeatures = new Dictionary<string, int> { ["user"] = 32, ["product"] = 64 },
            EdgeTypeSchema = new Dictionary<string, (string SourceType, string TargetType)>
            {
                ["buys"] = ("user", "product"),
                ["views"] = ("user", "product"),
            },
        };

        var restored = new HeterogeneousGraphMetadata();
        restored.LoadState(original.SaveState());

        Assert.Equal(original.NodeTypes, restored.NodeTypes);
        Assert.Equal(original.EdgeTypes, restored.EdgeTypes);
        Assert.Equal(original.NodeTypeFeatures, restored.NodeTypeFeatures);

        Assert.Equal(original.EdgeTypeSchema.Count, restored.EdgeTypeSchema.Count);
        foreach (var pair in original.EdgeTypeSchema)
        {
            Assert.True(
                restored.EdgeTypeSchema.TryGetValue(pair.Key, out var got),
                $"edge type '{pair.Key}' did not survive the round trip");
            Assert.Equal(pair.Value.SourceType, got.SourceType);
            Assert.Equal(pair.Value.TargetType, got.TargetType);
        }
    }

    /// <summary>An expression tree must come back computing the same function.</summary>
    /// <remarks>
    /// Compiling an Expression is one-way, so LambdaLayer keeps the tree in order to stay
    /// restorable. Comparing text would not prove much; what matters is that the rebuilt tree
    /// EVALUATES identically, which is the property a clone actually depends on.
    /// </remarks>
    [Fact]
    public void ExpressionState_RoundTripsToAnEquivalentFunction()
    {
        // NOT the identity. `x => x` would pass even if Load returned a default passthrough, which
        // is exactly the vacuous-green shape this suite keeps tripping over. Transposing makes the
        // rebuilt tree observably wrong if the node is dropped: the shape itself changes.
        Expression<Func<Tensor<double>, Tensor<double>>> original = x => x.Transpose();

        string saved = ExpressionState.Save(original);
        Assert.False(string.IsNullOrEmpty(saved), "an expression that can be saved should not save as nothing");

        var restored = ExpressionState.Load<Func<Tensor<double>, Tensor<double>>>(
            saved, nameof(StatePersistenceRoundTripTests), "probe");

        var input = new Tensor<double>(new[] { 2, 3 });
        for (int i = 0; i < input.Length; i++) input[i] = (i + 1) * 0.25;

        var before = original.Compile()(input);
        var after = restored.Compile()(input);

        // Shape too, not just length -- a dropped Transpose keeps the element count identical.
        Assert.Equal(before.Shape, after.Shape);
        Assert.Equal(before.Length, after.Length);
        for (int i = 0; i < before.Length; i++)
        {
            Assert.True(
                before[i] == after[i],
                $"element {i} differs after the round trip: {before[i]:R} became {after[i]:R}");
        }
    }
}

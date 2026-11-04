using System;
using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Serialization;
using Xunit;

namespace AiDotNetTests.UnitTests.Serialization;

public class ExternalLayerReflectionReviewTests
{
    [ElementWiseShape]
    public sealed class ShapeOnlyLayer : LayerBase<double>
    {
        public ShapeOnlyLayer(int[] inputShape) : base(inputShape, inputShape) { }
        public override bool SupportsTraining => false;
        public override void ResetState() { }
        protected override void WriteConstructionState(Dictionary<string, string> metadata)
        {
            metadata["Tag"] = "same";
            metadata["tag"] = "same";
        }
    }

    public sealed class MutableConfig
    {
        public List<int> Values { get; set; } = new List<int>();
    }

    public sealed class ConfigOwner
    {
        public ConfigOwner(MutableConfig configuration) => Configuration = configuration;
        public MutableConfig Configuration { get; }
    }

    [Fact]
    public void ShapeFallbackAcceptsConstructionKeysThatDifferOnlyInCase()
    {
        var source = new ShapeOnlyLayer(new[] { 3 });
        var clone = Assert.IsType<ShapeOnlyLayer>(source.Clone());
        Assert.NotSame(source, clone);
        Assert.Equal(source.GetInputShape(), clone.GetInputShape());
    }

    [Fact]
    public void ReflectionFactoryDoesNotShareMutableConstructorState()
    {
        var configuration = new MutableConfig { Values = new List<int> { 7 } };
        var bag = new LayerStateBag(new Dictionary<string, object> { ["configuration"] = configuration }, "ConfigOwner");
        Assert.True(LayerFactoryRegistry<double>.TryCreate(typeof(ConfigOwner), typeof(ConfigOwner), bag, null, null, out var rebuilt));
        var owner = Assert.IsType<ConfigOwner>(rebuilt);
        Assert.NotSame(configuration, owner.Configuration);
        Assert.NotSame(configuration.Values, owner.Configuration.Values);
        owner.Configuration.Values[0] = 99;
        Assert.Equal(7, configuration.Values[0]);
    }
}

using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Serialization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.Serialization;

/// <summary>
/// A layer declared in the TEST assembly, which the generator never sees.
/// </summary>
/// <remarks>
/// This is the only kind of layer that proves the clone path works for a consumer. Every layer in
/// the 321-type sweep lives in AiDotNet, so the generated factory table already names it; none of
/// them can demonstrate what happens to somebody else's layer.
/// </remarks>
// Shape-preserving: the constructor passes [units] as BOTH the input and the output shape, so
// the layer is element-wise at any rank. ADNSHAPE006 requires every LayerBase to say which it
// is, and this is the form the descriptor prescribes for that case.
[AiDotNet.Attributes.ElementWiseShape]
public sealed partial class ExternalTestLayer<T> : LayerBase<T>
{
    private readonly int _units;
    private readonly bool _useBias;

    /// <summary>Initializes a new instance of the <see cref="ExternalTestLayer{T}"/> class.</summary>
    /// <param name="units">The output width.</param>
    /// <param name="useBias">Whether a bias is added. Defaults to <c>true</c>.</param>
    public ExternalTestLayer(int units, bool useBias = true)
        : base([units], [units])
    {
        _units = units;
        _useBias = useBias;
    }

    /// <summary>The width this layer was built with.</summary>
    public int Units => _units;

    /// <summary>Whether this layer was built with a bias.</summary>
    public bool UseBias => _useBias;

    /// <inheritdoc/>
    public override bool SupportsTraining => false;

    /// <inheritdoc/>
    public override void ResetState()
    {
    }
}

/// <summary>Cloning a layer that AiDotNet's generator never compiled.</summary>
public class ExternalLayerCloneTests
{
    [Fact]
    public void A_layer_from_another_assembly_clones_with_its_construction_state()
    {
        var original = new ExternalTestLayer<double>(units: 7, useBias: false);

        var clone = original.Clone();

        var typed = Assert.IsType<ExternalTestLayer<double>>(clone);
        Assert.NotSame(original, typed);

        // The point of the whole exercise: the arguments survive. Before the registry this threw,
        // telling the author to add [LayerState] -- which cannot help, because the generator does
        // not run in their compilation.
        Assert.Equal(7, typed.Units);
        Assert.False(typed.UseBias);
    }

    [Fact]
    public void An_omitted_optional_argument_keeps_its_declared_default()
    {
        // useBias defaults to true. Rebuilding it as default(bool) would be false, which is the
        // silent-value-loss this work exists to remove.
        var original = new ExternalTestLayer<double>(units: 3);

        var clone = (ExternalTestLayer<double>)original.Clone();

        Assert.Equal(3, clone.Units);
        Assert.True(clone.UseBias);
    }

    [Fact]
    public void An_explicitly_registered_factory_is_preferred_over_reflection()
    {
        LayerFactoryRegistry<float>.Register(
            typeof(ExternalTestLayer<>),
            (state, _, _) => new ExternalTestLayer<float>(state.Int32("units"), state.Boolean("useBias")));

        Assert.True(LayerFactoryRegistry<float>.IsRegistered(typeof(ExternalTestLayer<>)));

        var original = new ExternalTestLayer<float>(units: 5, useBias: false);
        var clone = (ExternalTestLayer<float>)original.Clone();

        Assert.Equal(5, clone.Units);
        Assert.False(clone.UseBias);
    }
}

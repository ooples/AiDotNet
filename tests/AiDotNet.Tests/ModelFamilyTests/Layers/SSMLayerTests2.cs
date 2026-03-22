using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class ABCLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new ABCLayer<double>(4, 8);
    protected override int[] InputShape => [1, 4, 8];
}

public class BASEDLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new BASEDLayer<double>(4, 8);
    protected override int[] InputShape => [1, 4, 8];
}

public class DeltaFormerLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new DeltaFormerLayer<double>(4, 8);
    protected override int[] InputShape => [1, 4, 8];
}

public class DeltaNetLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new DeltaNetLayer<double>(4, 8);
    protected override int[] InputShape => [1, 4, 8];
}

public class DeltaProductLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new DeltaProductLayer<double>(4, 8);
    protected override int[] InputShape => [1, 4, 8];
}

public class ExtendedLSTMLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new ExtendedLSTMLayer<double>(4, 8);
    protected override int[] InputShape => [1, 4, 8];
}

public class GatedDeltaNetLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new GatedDeltaNetLayer<double>(4, 8);
    protected override int[] InputShape => [1, 4, 8];
}

public class GatedDeltaProductLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new GatedDeltaProductLayer<double>(4, 8);
    protected override int[] InputShape => [1, 4, 8];
}

public class GatedLinearAttentionLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new GatedLinearAttentionLayer<double>(4, 8);
    protected override int[] InputShape => [1, 4, 8];
}

public class GatedSlotAttentionLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new GatedSlotAttentionLayer<double>(4, 8);
    protected override int[] InputShape => [1, 4, 8];
}

public class HGRN2LayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new HGRN2Layer<double>(4, 8);
    protected override int[] InputShape => [1, 4, 8];
}

public class HGRNLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new HGRNLayer<double>(4, 8);
    protected override int[] InputShape => [1, 4, 8];
}

public class HedgehogLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new HedgehogLayer<double>(4, 8);
    protected override int[] InputShape => [1, 4, 8];
}

public class KimiLinearAttentionLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new KimiLinearAttentionLayer<double>(4, 8);
    protected override int[] InputShape => [1, 4, 8];
}

public class LogLinearAttentionLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new LogLinearAttentionLayer<double>(4, 8);
    protected override int[] InputShape => [1, 4, 8];
}

public class LonghornLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new LonghornLayer<double>(4, 8);
    protected override int[] InputShape => [1, 4, 8];
    // Group normalization maps constant inputs to same output by design
    protected override bool ExpectsDifferentOutputForConstantInputs => false;
}

public class MEGALayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new MEGALayer<double>(4, 8);
    protected override int[] InputShape => [1, 4, 8];
}

public class Mamba2BlockTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new Mamba2Block<double>(4, 8);
    protected override int[] InputShape => [1, 4, 8];
}

public class MegalodonLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new MegalodonLayer<double>(4, 8);
    protected override int[] InputShape => [1, 4, 8];
}

public class MesaNetLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new MesaNetLayer<double>(4, 8);
    protected override int[] InputShape => [1, 4, 8];
}

public class MixtureOfMambaLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new MixtureOfMambaLayer<double>(4, 8);
    protected override int[] InputShape => [1, 4, 8];
}

public class MixtureOfMemoriesLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new MixtureOfMemoriesLayer<double>(4, 8);
    protected override int[] InputShape => [1, 4, 8];
}

public class MultiLatentAttentionLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new MultiLatentAttentionLayer<double>(4, 8);
    protected override int[] InputShape => [1, 4, 8];
}

public class PaTHAttentionLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new PaTHAttentionLayer<double>(4, 8);
    protected override int[] InputShape => [1, 4, 8];
}

public class RWKV7BlockTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new RWKV7Block<double>(4, 8);
    protected override int[] InputShape => [1, 4, 8];
}

public class RealGatedLinearRecurrenceLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new RealGatedLinearRecurrenceLayer<double>(4, 8);
    protected override int[] InputShape => [1, 4, 8];
}

public class RebasedLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new RebasedLayer<double>(4, 8);
    protected override int[] InputShape => [1, 4, 8];
}

public class RodimusLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new RodimusLayer<double>(4, 8);
    protected override int[] InputShape => [1, 4, 8];
}

public class S5LayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new S5Layer<double>(4, 8);
    protected override int[] InputShape => [1, 4, 8];
}

public class TTTLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new TTTLayer<double>(4, 8);
    protected override int[] InputShape => [1, 4, 8];
}

public class TransNormerLLMLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer() => new TransNormerLLMLayer<double>(4, 8);
    protected override int[] InputShape => [1, 4, 8];
}

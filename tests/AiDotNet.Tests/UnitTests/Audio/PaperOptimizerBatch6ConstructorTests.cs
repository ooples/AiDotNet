using AiDotNet.NeuralNetworks;
using AiDotNet.SpeechRecognition.Streaming;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Audio;

public sealed class PaperOptimizerBatch6ConstructorTests : ConstructorInitializationTestBase
{
    [Fact]
    public void EmformerRNNT_does_not_dispatch_initialization_during_construction()
        => AssertSafeInitialization(architecture => new DerivedEmformerRNNT(architecture));

    [Fact]
    public void FastEmit_does_not_dispatch_initialization_during_construction()
        => AssertSafeInitialization(architecture => new DerivedFastEmit(architecture));

    private sealed class DerivedEmformerRNNT : EmformerRNNT<double>, IInitializationProbe
    {
        private readonly bool _constructed;
        public int InitializationCalls { get; private set; }

        public DerivedEmformerRNNT(NeuralNetworkArchitecture<double> architecture) : base(architecture)
        {
            // Set in the constructor body, not a field initializer that runs before the base constructor.
            _constructed = true;
        }

        protected override void InitializeLayers()
        {
            Assert.True(_constructed, "A virtual initialization hook ran before the derived constructor body.");
            InitializationCalls++;
            base.InitializeLayers();
        }

        public void ReinitializeLayers()
        {
            Layers.Clear();
            InitializeLayers();
        }
    }

    private sealed class DerivedFastEmit : FastEmit<double>, IInitializationProbe
    {
        private readonly bool _constructed;
        public int InitializationCalls { get; private set; }

        public DerivedFastEmit(NeuralNetworkArchitecture<double> architecture) : base(architecture)
        {
            // Set in the constructor body, not a field initializer that runs before the base constructor.
            _constructed = true;
        }

        protected override void InitializeLayers()
        {
            Assert.True(_constructed, "A virtual initialization hook ran before the derived constructor body.");
            InitializationCalls++;
            base.InitializeLayers();
        }

        public void ReinitializeLayers()
        {
            Layers.Clear();
            InitializeLayers();
        }
    }
}

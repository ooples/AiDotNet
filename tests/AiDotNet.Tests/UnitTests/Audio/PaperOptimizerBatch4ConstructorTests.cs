using AiDotNet.NeuralNetworks;
using AiDotNet.SpeechRecognition.LLMIntegrated;
using AiDotNet.SpeechRecognition.Multilingual;
using AiDotNet.SpeechRecognition.NeMo;
using AiDotNet.TextToSpeech.FlowDiffusion;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Audio;

public sealed class PaperOptimizerBatch4ConstructorTests : ConstructorInitializationTestBase
{
    [Fact]
    public void OLMoASR_does_not_dispatch_initialization_during_construction()
        => AssertSafeInitialization(architecture => new DerivedOLMoASR(architecture));

    [Fact]
    public void XLSR_does_not_dispatch_initialization_during_construction()
        => AssertSafeInitialization(architecture => new DerivedXLSR(architecture));

    [Fact]
    public void NeMoCitrinet_does_not_dispatch_initialization_during_construction()
        => AssertSafeInitialization(architecture => new DerivedNeMoCitrinet(architecture));

    [Fact]
    public void DiTToTTS_does_not_dispatch_initialization_during_construction()
        => AssertSafeInitialization(architecture => new DerivedDiTToTTS(architecture));

    private sealed class DerivedOLMoASR : OLMoASR<double>, IInitializationProbe
    {
        private readonly bool _constructed;
        public int InitializationCalls { get; private set; }

        public DerivedOLMoASR(NeuralNetworkArchitecture<double> architecture) : base(architecture)
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

    private sealed class DerivedXLSR : XLSR<double>, IInitializationProbe
    {
        private readonly bool _constructed;
        public int InitializationCalls { get; private set; }

        public DerivedXLSR(NeuralNetworkArchitecture<double> architecture) : base(architecture)
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

    private sealed class DerivedNeMoCitrinet : NeMoCitrinet<double>, IInitializationProbe
    {
        private readonly bool _constructed;
        public int InitializationCalls { get; private set; }

        public DerivedNeMoCitrinet(NeuralNetworkArchitecture<double> architecture) : base(architecture)
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

    private sealed class DerivedDiTToTTS : DiTToTTS<double>, IInitializationProbe
    {
        private readonly bool _constructed;
        public int InitializationCalls { get; private set; }

        public DerivedDiTToTTS(NeuralNetworkArchitecture<double> architecture) : base(architecture)
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

using AiDotNet.NeuralNetworks;
using AiDotNet.SpeechRecognition.Multilingual;
using AiDotNet.SpeechRecognition.Robust;
using AiDotNet.SpeechRecognition.Specialized;
using AiDotNet.SpeechRecognition.Streaming;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Audio;

public sealed class PaperOptimizerBatch5ConstructorTests : ConstructorInitializationTestBase
{
    [Fact]
    public void OWSM_does_not_dispatch_initialization_during_construction()
        => AssertSafeInitialization(architecture => new DerivedOWSM(architecture));

    [Fact]
    public void AVHuBERT_does_not_dispatch_initialization_during_construction()
        => AssertSafeInitialization(architecture => new DerivedAVHuBERT(architecture));

    [Fact]
    public void RobustConformer_does_not_dispatch_initialization_during_construction()
        => AssertSafeInitialization(architecture => new DerivedRobustConformer(architecture));

    [Fact]
    public void VoxtLM_does_not_dispatch_initialization_during_construction()
        => AssertSafeInitialization(architecture => new DerivedVoxtLM(architecture));

    [Fact]
    public void StreamingConformer_does_not_dispatch_initialization_during_construction()
        => AssertSafeInitialization(architecture => new DerivedStreamingConformer(architecture));

    private sealed class DerivedOWSM : OWSM<double>, IInitializationProbe
    {
        private readonly bool _constructed;
        public int InitializationCalls { get; private set; }

        public DerivedOWSM(NeuralNetworkArchitecture<double> architecture) : base(architecture)
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

    private sealed class DerivedAVHuBERT : AVHuBERT<double>, IInitializationProbe
    {
        private readonly bool _constructed;
        public int InitializationCalls { get; private set; }

        public DerivedAVHuBERT(NeuralNetworkArchitecture<double> architecture) : base(architecture)
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

    private sealed class DerivedRobustConformer : RobustConformer<double>, IInitializationProbe
    {
        private readonly bool _constructed;
        public int InitializationCalls { get; private set; }

        public DerivedRobustConformer(NeuralNetworkArchitecture<double> architecture) : base(architecture)
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

    private sealed class DerivedVoxtLM : VoxtLM<double>, IInitializationProbe
    {
        private readonly bool _constructed;
        public int InitializationCalls { get; private set; }

        public DerivedVoxtLM(NeuralNetworkArchitecture<double> architecture) : base(architecture)
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

    private sealed class DerivedStreamingConformer : StreamingConformer<double>, IInitializationProbe
    {
        private readonly bool _constructed;
        public int InitializationCalls { get; private set; }

        public DerivedStreamingConformer(NeuralNetworkArchitecture<double> architecture) : base(architecture)
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

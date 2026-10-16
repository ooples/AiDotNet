using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Diagnostics;

namespace AiDotNet.Generators;

/// <summary>
/// Keeps training in the base class: a concrete neural network may not override <c>Train(Tensor, Tensor)</c>.
/// </summary>
/// <remarks>
/// <para>
/// <c>NeuralNetworkBase.Train</c> is the one training step: the ONNX and disposed guards, batch-dim promotion,
/// micro-batching, the out-of-memory and device-fault retries, the model's optimizer, and the shared fused session.
/// A model supplies what differs through declarations instead: its optimizer (the generated <c>TrainingOptimizer</c>),
/// why it cannot train (<c>TrainingUnavailableReason</c>), how its input or target is prepared
/// (<c>PrepareTrainingPair</c>, <c>PrepareInputForTraining</c>), and a non-standard objective
/// (<c>DefineTrainingObjectives</c>). About 700 overrides that re-did the base step, and skipped half of it, were
/// deleted when this rule was added.
/// </para>
/// <para>
/// Abstract family bases may still override <c>Train</c>. The overrides that predate the rule are listed in
/// <see cref="TrainOverrideBaseline"/>, which may only shrink: a listed type that stops overriding is AIDN108.
/// </para>
/// </remarks>
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public class TrainOverrideAnalyzer : DiagnosticAnalyzer
{
    private const string NeuralNetworkBasePrefix = "AiDotNet.NeuralNetworks.NeuralNetworkBase<";

    private static readonly DiagnosticDescriptor NewTrainOverride = new(
        "AIDN107",
        "A model overrides Train instead of declaring its training step",
        "'{0}' overrides Train. The base class runs every model's training step; declare what differs instead: "
            + "DefineTrainingObjectives for a non-standard objective, PrepareTrainingPair or PrepareInputForTraining for "
            + "the data, TrainingUnavailableReason for a guard. Its optimizer comes from its _optimizer field.",
        "AiDotNet.Design",
        DiagnosticSeverity.Warning,
        isEnabledByDefault: true,
        description: "A Train override bypasses the base step's guards, batch-dim promotion, micro-batching, retries and "
            + "declared objectives.");

    private static readonly DiagnosticDescriptor StaleBaselineEntry = new(
        "AIDN108",
        "Train-override baseline entry no longer overrides",
        "'{0}' is listed in TrainOverrideBaseline but no longer overrides Train; remove its line so the baseline stays an exact inventory",
        "AiDotNet.Design",
        DiagnosticSeverity.Warning,
        isEnabledByDefault: true);

    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics
        => ImmutableArray.Create(NewTrainOverride, StaleBaselineEntry);

    public override void Initialize(AnalysisContext context)
    {
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();
        context.RegisterCompilationStartAction(start =>
        {
            if (start.Compilation.AssemblyName != "AiDotNet") return;
            var overriding = new ConcurrentDictionary<string, byte>(System.StringComparer.Ordinal);
            start.RegisterSymbolAction(symbolContext =>
            {
                var type = (INamedTypeSymbol)symbolContext.Symbol;
                if (type.IsAbstract || type.TypeKind != TypeKind.Class || !IsNeuralNetwork(type)) return;
                var train = type.GetMembers("Train").OfType<IMethodSymbol>().FirstOrDefault(IsTensorTrainOverride);
                if (train is null) return;

                string name = type.ToDisplayString();
                overriding[name] = 0;
                if (TrainOverrideBaseline.Types.Contains(name)) return;
                var location = train.Locations.FirstOrDefault(l => l.IsInSource) ?? Location.None;
                symbolContext.ReportDiagnostic(Diagnostic.Create(NewTrainOverride, location, name));
            }, SymbolKind.NamedType);

            start.RegisterCompilationEndAction(end =>
            {
                foreach (var listed in TrainOverrideBaseline.Types.Where(t => !overriding.ContainsKey(t)).OrderBy(t => t))
                    end.ReportDiagnostic(Diagnostic.Create(StaleBaselineEntry, Location.None, listed));
            });
        });
    }

    private static bool IsTensorTrainOverride(IMethodSymbol method)
        => method.IsOverride
            && method.Parameters.Length == 2
            && method.ReturnsVoid
            && method.Parameters.All(p => p.Type is INamedTypeSymbol { Name: "Tensor", TypeArguments.Length: 1 });

    private static bool IsNeuralNetwork(INamedTypeSymbol type)
    {
        for (var current = type.BaseType; current is not null; current = current.BaseType)
        {
            if (current.OriginalDefinition.ToDisplayString().StartsWith(NeuralNetworkBasePrefix, System.StringComparison.Ordinal))
                return true;
        }

        return false;
    }
}

/// <summary>
/// The concrete networks that overrode <c>Train</c> when <see cref="TrainOverrideAnalyzer"/> was added (2026-10-16), 169 types.
/// May only shrink.
/// </summary>
internal static class TrainOverrideBaseline
{
    internal static readonly HashSet<string> Types = new HashSet<string>(System.StringComparer.Ordinal)
    {
        "AiDotNet.Audio.Classification.AudioLDMClassifier<T>",
        "AiDotNet.Audio.Classification.AudioMAE<T>",
        "AiDotNet.Audio.Emotion.SpeechEmotionRecognizer<T>",
        "AiDotNet.Audio.Enhancement.ConvTasNet<T>",
        "AiDotNet.Audio.Enhancement.DCCRN<T>",
        "AiDotNet.Audio.Enhancement.DeepFilterNet<T>",
        "AiDotNet.Audio.Fingerprinting.ASTModel<T>",
        "AiDotNet.Audio.Fingerprinting.CLAPModel<T>",
        "AiDotNet.Audio.Fingerprinting.GraFPrint<T>",
        "AiDotNet.Audio.Foundations.Data2Vec2<T>",
        "AiDotNet.Audio.Foundations.HuBERT<T>",
        "AiDotNet.Audio.Foundations.MERT<T>",
        "AiDotNet.Audio.LanguageIdentification.ECAPATDNNLanguageIdentifier<T>",
        "AiDotNet.Audio.LanguageIdentification.VoxLingua107Identifier<T>",
        "AiDotNet.Audio.LanguageIdentification.Wav2Vec2LanguageIdentifier<T>",
        "AiDotNet.Audio.MusicGen.MusicGenModel<T>",
        "AiDotNet.Audio.Speaker.PyAnnote<T>",
        "AiDotNet.Audio.SpeechRecognition.Wav2Vec2Model<T>",
        "AiDotNet.Audio.TextToSpeech.Tacotron2Model<T>",
        "AiDotNet.Audio.VoiceActivity.SileroVad<T>",
        "AiDotNet.Audio.Whisper.WhisperModel<T>",
        "AiDotNet.ComputerVision.Detection.Backbones.CSPDarknet<T>",
        "AiDotNet.ComputerVision.Detection.Backbones.EfficientNet<T>",
        "AiDotNet.ComputerVision.Detection.Backbones.ResNet<T>",
        "AiDotNet.ComputerVision.Detection.Backbones.SwinTransformer<T>",
        "AiDotNet.ComputerVision.Detection.Backbones.VGG16BNBackbone<T>",
        "AiDotNet.ComputerVision.Detection.ObjectDetection.YOLO.YOLOv9Backbone<T>",
        "AiDotNet.ComputerVision.Segmentation.Diffusion.DiffCutSegmentation<T>",
        "AiDotNet.ComputerVision.Segmentation.Efficient.EfficientSAM<T>",
        "AiDotNet.ComputerVision.Segmentation.Efficient.FastSAM<T>",
        "AiDotNet.ComputerVision.Segmentation.Efficient.SlimSAM<T>",
        "AiDotNet.ComputerVision.Segmentation.Foundation.EoMT<T>",
        "AiDotNet.ComputerVision.Segmentation.Foundation.Mask2Former<T>",
        "AiDotNet.ComputerVision.Segmentation.Foundation.OneFormer<T>",
        "AiDotNet.ComputerVision.Segmentation.Foundation.SAM<T>",
        "AiDotNet.ComputerVision.Segmentation.Foundation.SAM21<T>",
        "AiDotNet.ComputerVision.Segmentation.Foundation.SAMHQ<T>",
        "AiDotNet.ComputerVision.Segmentation.Foundation.UNINEXT<T>",
        "AiDotNet.ComputerVision.Segmentation.Foundation.XDecoder<T>",
        "AiDotNet.ComputerVision.Segmentation.Interactive.SEEM<T>",
        "AiDotNet.ComputerVision.Segmentation.Mamba.VisionMamba<T>",
        "AiDotNet.ComputerVision.Segmentation.Mamba.VMamba<T>",
        "AiDotNet.ComputerVision.Segmentation.Medical.MedSAM<T>",
        "AiDotNet.ComputerVision.Segmentation.Medical.NnUNet<T>",
        "AiDotNet.ComputerVision.Segmentation.Medical.SwinUNETR<T>",
        "AiDotNet.ComputerVision.Segmentation.Medical.TransUNet<T>",
        "AiDotNet.ComputerVision.Segmentation.Medical.UMamba<T>",
        "AiDotNet.ComputerVision.Segmentation.OpenVocabulary.CATSeg<T>",
        "AiDotNet.ComputerVision.Segmentation.OpenVocabulary.MaskAdapter<T>",
        "AiDotNet.ComputerVision.Segmentation.OpenVocabulary.OpenVocabSAM<T>",
        "AiDotNet.ComputerVision.Segmentation.Panoptic.KMaXDeepLab<T>",
        "AiDotNet.Document.GraphBased.PICK<T>",
        "AiDotNet.Document.LayoutAware.LayoutLM<T>",
        "AiDotNet.Document.LayoutAware.LayoutLMv2<T>",
        "AiDotNet.Document.OCR.TextRecognition.ABINet<T>",
        "AiDotNet.Document.PixelToSequence.Dessurt<T>",
        "AiDotNet.Document.PixelToSequence.Donut<T>",
        "AiDotNet.Document.PixelToSequence.MATCHA<T>",
        "AiDotNet.Document.PixelToSequence.Nougat<T>",
        "AiDotNet.Document.PixelToSequence.Pix2Struct<T>",
        "AiDotNet.Document.VisionLanguage.InfographicVQA<T>",
        "AiDotNet.Document.VisionLanguage.UDOP<T>",
        "AiDotNet.Finance.Forecasting.Foundation.Chronos<T>",
        "AiDotNet.Finance.Forecasting.Foundation.CSDI<T>",
        "AiDotNet.Finance.Forecasting.Foundation.MGTSD<T>",
        "AiDotNet.Finance.Forecasting.Foundation.MOIRAI<T>",
        "AiDotNet.Finance.Forecasting.Neural.NHiTSFinance<T>",
        "AiDotNet.Finance.Graph.DCRNN<T>",
        "AiDotNet.Finance.Graph.GraphWaveNet<T>",
        "AiDotNet.Finance.Graph.MTGNN<T>",
        "AiDotNet.Finance.Graph.RelationalGCN<T>",
        "AiDotNet.Finance.Portfolio.DeepPortfolioManager<T>",
        "AiDotNet.Finance.Risk.NeuralVaR<T>",
        "AiDotNet.Finance.Trading.Factors.FactorVAE<T>",
        "AiDotNet.NER.SequenceLabeling.BiLSTMCRF<T>",
        "AiDotNet.NER.SequenceLabeling.CNNBiLSTMCRF<T>",
        "AiDotNet.NER.SequenceLabeling.LSTMCRF<T>",
        "AiDotNet.NER.SequenceLabeling.WordCharBiLSTMCRF<T>",
        "AiDotNet.NeuralNetworks.ACGAN<T>",
        "AiDotNet.NeuralNetworks.Autoencoder<T>",
        "AiDotNet.NeuralNetworks.Blip2NeuralNetwork<T>",
        "AiDotNet.NeuralNetworks.CapsuleNetwork<T>",
        "AiDotNet.NeuralNetworks.ClipNeuralNetwork<T>",
        "AiDotNet.NeuralNetworks.ConditionalGAN<T>",
        "AiDotNet.NeuralNetworks.ConvolutionalNeuralNetwork<T>",
        "AiDotNet.NeuralNetworks.CycleGAN<T>",
        "AiDotNet.NeuralNetworks.DeepBeliefNetwork<T>",
        "AiDotNet.NeuralNetworks.DeepBoltzmannMachine<T>",
        "AiDotNet.NeuralNetworks.DeepQNetwork<T>",
        "AiDotNet.NeuralNetworks.DifferentiableNeuralComputer<T>",
        "AiDotNet.NeuralNetworks.EchoStateNetwork<T>",
        "AiDotNet.NeuralNetworks.EfficientNetNetwork<T>",
        "AiDotNet.NeuralNetworks.ExtremeLearningMachine<T>",
        "AiDotNet.NeuralNetworks.GenerativeAdversarialNetwork<T>",
        "AiDotNet.NeuralNetworks.GraphAttentionNetwork<T>",
        "AiDotNet.NeuralNetworks.GraphGenerationModel<T>",
        "AiDotNet.NeuralNetworks.GraphIsomorphismNetwork<T>",
        "AiDotNet.NeuralNetworks.GraphNeuralNetwork<T>",
        "AiDotNet.NeuralNetworks.GraphSAGENetwork<T>",
        "AiDotNet.NeuralNetworks.GRUNeuralNetwork<T>",
        "AiDotNet.NeuralNetworks.HopeNetwork<T>",
        "AiDotNet.NeuralNetworks.HopfieldNetwork<T>",
        "AiDotNet.NeuralNetworks.HTMNetwork<T>",
        "AiDotNet.NeuralNetworks.InfoGAN<T>",
        "AiDotNet.NeuralNetworks.LiquidStateMachine<T>",
        "AiDotNet.NeuralNetworks.MemoryNetwork<T>",
        "AiDotNet.NeuralNetworks.MobileNetV2Network<T>",
        "AiDotNet.NeuralNetworks.MobileNetV3Network<T>",
        "AiDotNet.NeuralNetworks.NEAT<T>",
        "AiDotNet.NeuralNetworks.NeuralTuringMachine<T>",
        "AiDotNet.NeuralNetworks.Pix2Pix<T>",
        "AiDotNet.NeuralNetworks.QuantumNeuralNetwork<T>",
        "AiDotNet.NeuralNetworks.RadialBasisFunctionNetwork<T>",
        "AiDotNet.NeuralNetworks.RecurrentNeuralNetwork<T>",
        "AiDotNet.NeuralNetworks.ResidualNeuralNetwork<T>",
        "AiDotNet.NeuralNetworks.ResNetNetwork<T>",
        "AiDotNet.NeuralNetworks.RestrictedBoltzmannMachine<T>",
        "AiDotNet.NeuralNetworks.SelfOrganizingMap<T>",
        "AiDotNet.NeuralNetworks.SpikingNeuralNetwork<T>",
        "AiDotNet.NeuralNetworks.StyleGAN<T>",
        "AiDotNet.NeuralNetworks.SyntheticData.CausalGANGenerator<T>",
        "AiDotNet.NeuralNetworks.SyntheticData.CopulaGANGenerator<T>",
        "AiDotNet.NeuralNetworks.SyntheticData.CTABGANPlusGenerator<T>",
        "AiDotNet.NeuralNetworks.SyntheticData.CTGANGenerator<T>",
        "AiDotNet.NeuralNetworks.SyntheticData.DPCTGANGenerator<T>",
        "AiDotNet.NeuralNetworks.SyntheticData.GOGGLEGenerator<T>",
        "AiDotNet.NeuralNetworks.SyntheticData.MedGANGenerator<T>",
        "AiDotNet.NeuralNetworks.SyntheticData.MisGANGenerator<T>",
        "AiDotNet.NeuralNetworks.SyntheticData.OCTGANGenerator<T>",
        "AiDotNet.NeuralNetworks.SyntheticData.PATEGANGenerator<T>",
        "AiDotNet.NeuralNetworks.SyntheticData.REaLTabFormerGenerator<T>",
        "AiDotNet.NeuralNetworks.SyntheticData.TableGANGenerator<T>",
        "AiDotNet.NeuralNetworks.SyntheticData.TimeGANGenerator<T>",
        "AiDotNet.NeuralNetworks.SyntheticData.TVAEGenerator<T>",
        "AiDotNet.NeuralNetworks.Transformer<T>",
        "AiDotNet.NeuralNetworks.VGGNetwork<T>",
        "AiDotNet.NeuralNetworks.WGAN<T>",
        "AiDotNet.NeuralNetworks.WGANGP<T>",
        "AiDotNet.NeuralRadianceFields.Models.GaussianSplatting<T>",
        "AiDotNet.NeuralRadianceFields.Models.InstantNGP<T>",
        "AiDotNet.PhysicsInformed.NeuralOperators.GraphNeuralOperator<T>",
        "AiDotNet.PhysicsInformed.ScientificML.UniversalDifferentialEquation<T>",
        "AiDotNet.ProgramSynthesis.Engines.CodeBERT<T>",
        "AiDotNet.ProgramSynthesis.Engines.CodeT5<T>",
        "AiDotNet.ProgramSynthesis.Engines.GraphCodeBERT<T>",
        "AiDotNet.SpeechRecognition.AlibabaASR.Paraformer<T>",
        "AiDotNet.SpeechRecognition.AlibabaASR.ParaformerLarge<T>",
        "AiDotNet.SpeechRecognition.AlibabaASR.SeACo<T>",
        "AiDotNet.SpeechRecognition.AlibabaASR.SenseVoice<T>",
        "AiDotNet.SpeechRecognition.AlibabaASR.SenseVoiceLarge<T>",
        "AiDotNet.SpeechRecognition.ConformerFamily.CIFEncoder<T>",
        "AiDotNet.TextToSpeech.CodecBased.BarkModel<T>",
        "AiDotNet.TextToSpeech.DescriptionBased.ParlerTTS<T>",
        "AiDotNet.TextToSpeech.FlowDiffusion.E2TTS<T>",
        "AiDotNet.TextToSpeech.Vocoders.WaveGrad<T>",
        "AiDotNet.UncertaintyQuantification.BayesianNeuralNetworks.BayesianNeuralNetwork<T>",
        "AiDotNet.Video.Depth.DepthAnythingV2<T>",
        "AiDotNet.Video.Enhancement.MIAVSR<T>",
        "AiDotNet.Video.Enhancement.StableVideoSR<T>",
        "AiDotNet.Video.Generation.OpenSora<T>",
        "AiDotNet.Video.RealESRGAN<T>",
        "AiDotNet.Video.Segmentation.SAM2<T>",
        "AiDotNet.VisionLanguage.Document.GOTOCR2<T>",
        "AiDotNet.VisionLanguage.Encoders.Florence2<T>",
        "AiDotNet.VisionLanguage.Generative.KOSMOS1<T>",
        "AiDotNet.VisionLanguage.Generative.KOSMOS2<T>",
        "AiDotNet.VisionLanguage.Generative.PaLI3<T>",
        "AiDotNet.VisionLanguage.InstructionTuned.Phi3Vision<T>",
        "AiDotNet.VisionLanguage.Unified.Transfusion<T>",
    };
}
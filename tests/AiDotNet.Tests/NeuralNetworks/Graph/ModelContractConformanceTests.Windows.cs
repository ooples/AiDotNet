using System;
using System.Collections.Generic;

namespace AiDotNet.Tests.NeuralNetworks.Graph;

/// <summary>
/// The committed window assignment for the CI conformance shards (ADNSHAPE_CONF_WINDOW).
/// </summary>
/// <remarks>
/// <para>
/// The windows used to be ordinal: offset/budget over the name-sorted inventory. Adding one model
/// moved every model after it into the next window, so a coverage map measured before the addition
/// routed all windows wrongly, and the selector had to re-run every window whenever any model
/// declaration changed on master. Here a model keeps its window for good: the table is
/// append-only, and a new model joins the last window with room or opens a new one.
/// </para>
/// <para>
/// <see cref="ConformanceWindowTableTests"/> fails when a windowed model is missing, a listed one no
/// longer exists, or a window is over <see cref="WindowSize"/>, and prints the lines to add.
/// <see cref="WindowCount"/> must equal the number of "Conformance - VisionLanguage" shards in
/// .github/test-shards.yml.
/// </para>
/// </remarks>
public partial class ModelContractConformanceTests
{
    /// <summary>The namespace the CI windows cover.</summary>
    internal const string WindowedNamespace = "VisionLanguage";

    /// <summary>
    /// Models per window: each concrete contract gets up to three minutes and 1 GiB in its own worker,
    /// and five fit the shard's 20-minute hang timeout.
    /// </summary>
    internal const int WindowSize = 5;

    /// <summary>Windows the manifest defines (ADNSHAPE_CONF_WINDOW 0 .. WindowCount-1).</summary>
    internal const int WindowCount = 34;

    /// <summary>Model full name (generic definition) to window. Append only; never renumber.</summary>
    internal static readonly IReadOnlyDictionary<string, int> ConformanceWindows = new Dictionary<string, int>(StringComparer.Ordinal)
    {
        // Seeded 2026-10-01 from the ordinal windows the manifest ran (offset = 5 x window), so no model moved.
        ["AiDotNet.VisionLanguage.Encoders.ALIGN`1"] = 0,
        ["AiDotNet.VisionLanguage.InstructionTuned.AquilaVL`1"] = 0,
        ["AiDotNet.VisionLanguage.InstructionTuned.Aria`1"] = 0,
        ["AiDotNet.VisionLanguage.Encoders.BASIC`1"] = 0,
        ["AiDotNet.VisionLanguage.Generative.BLIP3`1"] = 0,
        ["AiDotNet.VisionLanguage.Encoders.BiomedCLIP`1"] = 1,
        ["AiDotNet.VisionLanguage.Foundational.BridgeTower`1"] = 1,
        ["AiDotNet.VisionLanguage.Encoders.CLIPA`1"] = 1,
        ["AiDotNet.VisionLanguage.InstructionTuned.Cambrian1`1"] = 1,
        ["AiDotNet.VisionLanguage.Unified.Chameleon`1"] = 1,
        ["AiDotNet.VisionLanguage.Proprietary.ClaudeVision`1"] = 2,
        ["AiDotNet.VisionLanguage.Generative.CoCa`1"] = 2,
        ["AiDotNet.VisionLanguage.InstructionTuned.CogVLM2`1"] = 2,
        ["AiDotNet.VisionLanguage.InstructionTuned.CogVLM`1"] = 2,
        ["AiDotNet.VisionLanguage.Encoders.DFNCLIP`1"] = 2,
        ["AiDotNet.VisionLanguage.Grounding.DINOX`1"] = 3,
        ["AiDotNet.VisionLanguage.Encoders.DINOv2`1"] = 3,
        ["AiDotNet.VisionLanguage.Encoders.DINOv3`1"] = 3,
        ["AiDotNet.VisionLanguage.Encoders.DeCLIP`1"] = 3,
        ["AiDotNet.VisionLanguage.InstructionTuned.DeepSeekVL2`1"] = 3,
        ["AiDotNet.VisionLanguage.InstructionTuned.DeepSeekVL`1"] = 4,
        ["AiDotNet.Document.VisionLanguage.DocOwl`1"] = 4,
        ["AiDotNet.VisionLanguage.Document.DocPedia`1"] = 4,
        ["AiDotNet.VisionLanguage.Document.Donut`1"] = 4,
        ["AiDotNet.VisionLanguage.Medical.DragonflyMed`1"] = 4,
        ["AiDotNet.VisionLanguage.InstructionTuned.Dragonfly`1"] = 5,
        ["AiDotNet.VisionLanguage.Encoders.EVACLIP`1"] = 5,
        ["AiDotNet.VisionLanguage.InstructionTuned.Eagle25`1"] = 5,
        ["AiDotNet.VisionLanguage.InstructionTuned.Eagle`1"] = 5,
        ["AiDotNet.VisionLanguage.Generative.Emu2`1"] = 5,
        ["AiDotNet.VisionLanguage.Generative.Emu3`1"] = 6,
        ["AiDotNet.VisionLanguage.Generative.Emu`1"] = 6,
        ["AiDotNet.VisionLanguage.Encoders.FLIP`1"] = 6,
        ["AiDotNet.VisionLanguage.Grounding.FerretV2`1"] = 6,
        ["AiDotNet.VisionLanguage.Grounding.Ferret`1"] = 6,
        ["AiDotNet.VisionLanguage.Encoders.Florence2`1"] = 7,
        ["AiDotNet.VisionLanguage.InstructionTuned.Fuyu`1"] = 7,
        ["AiDotNet.VisionLanguage.Generative.GIT`1"] = 7,
        ["AiDotNet.VisionLanguage.Grounding.GLaMM`1"] = 7,
        ["AiDotNet.VisionLanguage.Document.GOTOCR2`1"] = 7,
        ["AiDotNet.VisionLanguage.ThreeD.GPT4Point`1"] = 8,
        ["AiDotNet.VisionLanguage.Robotics.GR00TN1`1"] = 8,
        ["AiDotNet.VisionLanguage.Proprietary.GeminiVision`1"] = 8,
        ["AiDotNet.VisionLanguage.InstructionTuned.Gemma3`1"] = 8,
        ["AiDotNet.VisionLanguage.RemoteSensing.GeoChat`1"] = 8,
        ["AiDotNet.VisionLanguage.Proprietary.GrokVision`1"] = 9,
        ["AiDotNet.VisionLanguage.Grounding.Groma`1"] = 9,
        ["AiDotNet.VisionLanguage.Grounding.GroundedSAM2`1"] = 9,
        ["AiDotNet.VisionLanguage.Grounding.GroundingDINO15`1"] = 9,
        ["AiDotNet.VisionLanguage.Grounding.GroundingDINO`1"] = 9,
        ["AiDotNet.VisionLanguage.Robotics.Helix`1"] = 10,
        ["AiDotNet.VisionLanguage.Generative.IDEFICS2`1"] = 10,
        ["AiDotNet.VisionLanguage.Generative.IDEFICS3`1"] = 10,
        ["AiDotNet.VisionLanguage.Generative.IDEFICS`1"] = 10,
        ["AiDotNet.Document.VisionLanguage.InfographicVQA`1"] = 10,
        ["AiDotNet.VisionLanguage.Generative.InstructBLIP`1"] = 11,
        ["AiDotNet.VisionLanguage.InstructionTuned.InternVL25`1"] = 11,
        ["AiDotNet.VisionLanguage.InstructionTuned.InternVL2`1"] = 11,
        ["AiDotNet.VisionLanguage.InstructionTuned.InternVL3`1"] = 11,
        ["AiDotNet.VisionLanguage.InstructionTuned.InternVL`1"] = 11,
        ["AiDotNet.VisionLanguage.Encoders.InternViT`1"] = 12,
        ["AiDotNet.VisionLanguage.Unified.JanusPro`1"] = 12,
        ["AiDotNet.VisionLanguage.Unified.Janus`1"] = 12,
        ["AiDotNet.VisionLanguage.Generative.KOSMOS1`1"] = 12,
        ["AiDotNet.VisionLanguage.Generative.KOSMOS2`1"] = 12,
        ["AiDotNet.VisionLanguage.Reasoning.KimiVLThinking`1"] = 13,
        ["AiDotNet.VisionLanguage.Reasoning.KimiVL`1"] = 13,
        ["AiDotNet.VisionLanguage.ThreeD.LEOVL`1"] = 13,
        ["AiDotNet.VisionLanguage.Encoders.LLM2CLIP`1"] = 13,
        ["AiDotNet.VisionLanguage.InstructionTuned.LLaVA15`1"] = 13,
        ["AiDotNet.VisionLanguage.Reasoning.LLaVACoT`1"] = 14,
        ["AiDotNet.VisionLanguage.Medical.LLaVAMed`1"] = 14,
        ["AiDotNet.VisionLanguage.VideoLanguage.LLaVANeXTVideo`1"] = 14,
        ["AiDotNet.VisionLanguage.InstructionTuned.LLaVANeXT`1"] = 14,
        ["AiDotNet.VisionLanguage.InstructionTuned.LLaVAOneVision15`1"] = 14,
        ["AiDotNet.VisionLanguage.InstructionTuned.LLaVAOneVision`1"] = 15,
        ["AiDotNet.VisionLanguage.VideoLanguage.LLaVAVideo`1"] = 15,
        ["AiDotNet.VisionLanguage.Foundational.LXMERT`1"] = 15,
        ["AiDotNet.VisionLanguage.Document.LayoutLMv3`1"] = 15,
        ["AiDotNet.VisionLanguage.Encoders.LiT`1"] = 15,
        ["AiDotNet.VisionLanguage.InstructionTuned.Llama32Vision`1"] = 16,
        ["AiDotNet.VisionLanguage.VideoLanguage.LongVILA`1"] = 16,
        ["AiDotNet.VisionLanguage.Foundational.METER`1"] = 16,
        ["AiDotNet.VisionLanguage.Document.MPLUGDocOwl15`1"] = 16,
        ["AiDotNet.VisionLanguage.Document.MPLUGDocOwl2`1"] = 16,
        ["AiDotNet.VisionLanguage.Document.MPLUGDocOwl`1"] = 17,
        ["AiDotNet.VisionLanguage.InstructionTuned.MPLUGOwl2`1"] = 17,
        ["AiDotNet.VisionLanguage.InstructionTuned.MPLUGOwl3`1"] = 17,
        ["AiDotNet.VisionLanguage.InstructionTuned.MPLUGOwl`1"] = 17,
        ["AiDotNet.VisionLanguage.InstructionTuned.Mantis`1"] = 17,
        ["AiDotNet.VisionLanguage.InstructionTuned.Maya`1"] = 18,
        ["AiDotNet.VisionLanguage.Encoders.MedCLIP`1"] = 18,
        ["AiDotNet.VisionLanguage.Medical.MedFlamingo`1"] = 18,
        ["AiDotNet.VisionLanguage.Encoders.MetaCLIP`1"] = 18,
        ["AiDotNet.VisionLanguage.InstructionTuned.MiniCPMV`1"] = 18,
        ["AiDotNet.VisionLanguage.InstructionTuned.MiniCPMo`1"] = 19,
        ["AiDotNet.VisionLanguage.InstructionTuned.MiniGPT4`1"] = 19,
        ["AiDotNet.VisionLanguage.InstructionTuned.MiniGPTv2`1"] = 19,
        ["AiDotNet.VisionLanguage.InstructionTuned.Molmo`1"] = 19,
        ["AiDotNet.VisionLanguage.InstructionTuned.Monkey`1"] = 19,
        ["AiDotNet.VisionLanguage.InstructionTuned.Moondream`1"] = 20,
        ["AiDotNet.VisionLanguage.InstructionTuned.NVLM`1"] = 20,
        ["AiDotNet.VisionLanguage.Document.Nougat`1"] = 20,
        ["AiDotNet.VisionLanguage.Grounding.OWLViT`1"] = 20,
        ["AiDotNet.VisionLanguage.Grounding.OWLv2`1"] = 20,
        ["AiDotNet.VisionLanguage.Robotics.Octo`1"] = 21,
        ["AiDotNet.VisionLanguage.Unified.OmniGen2`1"] = 21,
        ["AiDotNet.VisionLanguage.Encoders.OpenCLIP`1"] = 21,
        ["AiDotNet.VisionLanguage.Generative.OpenFlamingo`1"] = 21,
        ["AiDotNet.VisionLanguage.Foundational.Oscar`1"] = 21,
        ["AiDotNet.VisionLanguage.InstructionTuned.Ovis`1"] = 22,
        ["AiDotNet.VisionLanguage.VideoLanguage.PLLaVA`1"] = 22,
        ["AiDotNet.VisionLanguage.Generative.PaLI3`1"] = 22,
        ["AiDotNet.VisionLanguage.Generative.PaLIX`1"] = 22,
        ["AiDotNet.VisionLanguage.Generative.PaLI`1"] = 22,
        ["AiDotNet.VisionLanguage.Robotics.PaLME`1"] = 23,
        ["AiDotNet.VisionLanguage.Medical.PathVLM`1"] = 23,
        ["AiDotNet.VisionLanguage.Encoders.PerceptionEncoder`1"] = 23,
        ["AiDotNet.VisionLanguage.InstructionTuned.Phi3Vision`1"] = 23,
        ["AiDotNet.VisionLanguage.InstructionTuned.Phi4Multimodal`1"] = 23,
        ["AiDotNet.VisionLanguage.Robotics.PiZero`1"] = 24,
        ["AiDotNet.VisionLanguage.Document.Pix2Struct`1"] = 24,
        ["AiDotNet.VisionLanguage.InstructionTuned.PixtralLarge`1"] = 24,
        ["AiDotNet.VisionLanguage.InstructionTuned.Pixtral`1"] = 24,
        ["AiDotNet.VisionLanguage.ThreeD.PointLLM`1"] = 24,
        ["AiDotNet.VisionLanguage.Reasoning.QVQ72B`1"] = 25,
        ["AiDotNet.VisionLanguage.InstructionTuned.Qwen25VL`1"] = 25,
        ["AiDotNet.VisionLanguage.InstructionTuned.Qwen2VL`1"] = 25,
        ["AiDotNet.VisionLanguage.InstructionTuned.Qwen3VL`1"] = 25,
        ["AiDotNet.VisionLanguage.InstructionTuned.QwenVL`1"] = 25,
        ["AiDotNet.VisionLanguage.Encoders.RADIOv25`1"] = 26,
        ["AiDotNet.VisionLanguage.RemoteSensing.RSGPT`1"] = 26,
        ["AiDotNet.VisionLanguage.Robotics.RT2`1"] = 26,
        ["AiDotNet.VisionLanguage.Medical.RadFM`1"] = 26,
        ["AiDotNet.VisionLanguage.Encoders.RegionCLIP`1"] = 26,
        ["AiDotNet.VisionLanguage.Encoders.RemoteCLIP`1"] = 27,
        ["AiDotNet.VisionLanguage.Encoders.SAM`1"] = 27,
        ["AiDotNet.VisionLanguage.Unified.SEEDX`1"] = 27,
        ["AiDotNet.VisionLanguage.ThreeD.SceneLLM`1"] = 27,
        ["AiDotNet.VisionLanguage.Grounding.Shikra`1"] = 27,
        ["AiDotNet.VisionLanguage.Unified.ShowO2`1"] = 28,
        ["AiDotNet.VisionLanguage.Unified.ShowO`1"] = 28,
        ["AiDotNet.VisionLanguage.Encoders.SigLIP2`1"] = 28,
        ["AiDotNet.VisionLanguage.Encoders.SigLIPSO`1"] = 28,
        ["AiDotNet.VisionLanguage.Encoders.SigLIP`1"] = 28,
        ["AiDotNet.VisionLanguage.RemoteSensing.SkyEyeGPT`1"] = 29,
        ["AiDotNet.VisionLanguage.Reasoning.SkyworkR1V2`1"] = 29,
        ["AiDotNet.VisionLanguage.Reasoning.SkyworkR1V`1"] = 29,
        ["AiDotNet.VisionLanguage.VideoLanguage.SlowFastLLaVA`1"] = 29,
        ["AiDotNet.VisionLanguage.InstructionTuned.SmolVLM`1"] = 29,
        ["AiDotNet.VisionLanguage.Document.Surya`1"] = 30,
        ["AiDotNet.VisionLanguage.Document.TextMonkey`1"] = 30,
        ["AiDotNet.VisionLanguage.ThreeD.ThreeDGraphLLM`1"] = 30,
        ["AiDotNet.VisionLanguage.ThreeD.ThreeDLLM`1"] = 30,
        ["AiDotNet.VisionLanguage.Robotics.ThreeDVLA`1"] = 30,
        ["AiDotNet.VisionLanguage.Unified.Transfusion`1"] = 31,
        ["AiDotNet.Document.VisionLanguage.UDOP`1"] = 31,
        ["AiDotNet.VisionLanguage.Foundational.UNITER`1"] = 31,
        ["AiDotNet.VisionLanguage.Document.UReader`1"] = 31,
        ["AiDotNet.VisionLanguage.InstructionTuned.VILAU`1"] = 31,
        ["AiDotNet.VisionLanguage.InstructionTuned.VILA`1"] = 32,
        ["AiDotNet.VisionLanguage.Foundational.ViLBERT`1"] = 32,
        ["AiDotNet.VisionLanguage.Foundational.ViLT`1"] = 32,
        ["AiDotNet.VisionLanguage.Encoders.ViT`1"] = 32,
        ["AiDotNet.VisionLanguage.VideoLanguage.VideoChat2`1"] = 32,
        ["AiDotNet.VisionLanguage.VideoLanguage.VideoLLaMA2`1"] = 33,
        ["AiDotNet.VisionLanguage.VideoLanguage.VideoLLaMA3`1"] = 33,
        ["AiDotNet.VisionLanguage.VideoLanguage.VideoLLaVA`1"] = 33,
        ["AiDotNet.VisionLanguage.Foundational.VinVL`1"] = 33,
        ["AiDotNet.VisionLanguage.Foundational.VisualBERT`1"] = 33,
    };
}
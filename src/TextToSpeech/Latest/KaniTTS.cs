using AiDotNet.Attributes;
using AiDotNet.Audio.Codecs;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.Tokenization.Algorithms;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.Latest;


/// <summary>
/// KaniTTS (nineninesix.ai, 2025): text-to-speech as language modeling — an LFM2 language model reads the text and writes
/// NVIDIA NanoCodec codes, four per frame at 12.5 frames a second, which NanoCodec's causal decoder turns into 22.05 kHz
/// audio.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>KaniTTS has no paper; its release is the reference: the prompt (start-of-human, the text, end-of-text,
/// end-of-human), sampling and the speech extraction of the kani-tts 0.0.4 package, the streaming extraction of the basic
/// example, and the fine-tuning pipeline's data format and recipe (KaniTTS-Finetune-pipeline). The language model is
/// transformers' <c>Lfm2ForCausalLM</c> and the codec NeMo's <c>AudioCodecModel</c>, both ported.</para>
/// <para><b>For Beginners:</b> KaniTTS treats speech as text a small language model can write: after the sentence it writes
/// a stream of sound codes, and a codec turns those codes into audio. Multi-speaker checkpoints take a speaker's name.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var model = new KaniTTS&lt;float&gt;(architecture, new KaniTTSOptions());
/// model.LoadReferenceWeights("kani-tts-370m");                                   // model.safetensors, tokenizer.json, config.json
/// model.LoadCodecWeights("nemo-nano-codec-22khz-0.6kbps-12.5fps.nemo");
/// model.Speaker = "andrew";
/// var audio = model.Synthesize("Hello, how are you doing today?");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Kani TTS", "https://github.com/nineninesix-ai/kani-tts", Year = 2025, Authors = "nineninesix.ai")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-5, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, WeightDecay = 0.02,
    MaxGradientNorm = 1, GradientAccumulationSteps = 4, Provenance = RecipeProvenance.ReleasedCode,
    Source = "No paper. nineninesix-ai/KaniTTS-Finetune-pipeline config/experiments.yaml (the first experiment): rsLoRA of "
             + "rank 16 (alpha 16, dropout 0.1) on q_proj, w1 and w2, trained by Hugging Face's Trainer with adamw_torch at "
             + "5e-5, weight decay 0.02, a cosine schedule after a 10% warm-up, max_grad_norm 1, batch 1 with 4-step "
             + "gradient accumulation; the loss on every token (labels = input_ids).")]
public partial class KaniTTS<T> : KaniTTSBase<T>
{
    /// <summary>Creates an ONNX-backed KaniTTS for inference.</summary>
    public KaniTTS(NeuralNetworkArchitecture<T> architecture, string modelPath, KaniTTSOptions? options = null)
        : base(architecture, modelPath, options ?? new KaniTTSOptions())
    {
    }

    /// <summary>Creates a native, trainable KaniTTS.</summary>
    public KaniTTS(NeuralNetworkArchitecture<T> architecture, KaniTTSOptions? options = null)
        : base(architecture, options ?? new KaniTTSOptions())
    {
    }
}

using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.SpeechRecognition.WhisperFamily;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Manual test scaffold for WhisperTimestamped (Louradour, 2023 — "whisper-timestamped:
/// Word-level timestamps for Whisper"). The auto-generator is told to skip
/// WhisperTimestamped (<c>ExcludedClassNames</c>) so this hand-written scaffold is
/// authoritative.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why a reduced-scale <c>&lt;float&gt;</c> config:</b> WhisperTimestamped's production
/// defaults mirror Whisper large-v3 (EncoderDim/DecoderDim=1280, 32 encoder + 32 decoder
/// layers, NumHeads=20, vocab 51866 — ~631M parameters), and the native constructor sizes
/// its layer stack from <see cref="WhisperTimestampedOptions"/>, not from the architecture,
/// so the auto-generator cannot shrink it.
/// </para>
/// <para>
/// Profiling (#1670, dotnet-trace + the testconsole <c>whispertimestamped-profile</c> harness,
/// run with <c>AIDOTNET_DISABLE_GPU=1</c> to match the CPU-only CI runners) showed a single
/// large-v3 forward is ~2.4s in <c>float</c> but ~153s in <c>double</c> — the fp64 CPU GEMM
/// path is ~60-80x slower than fp32. Because the model-family training invariants run 30-250
/// optimizer steps, the auto-generated <c>&lt;double&gt;</c> scaffold blew the 120s CI budget
/// on the very first invariant. Running these invariants in <c>float</c> at reduced scale keeps
/// every code path (conv subsampling, encoder self-attention, decoder self/cross-attention,
/// FFN, vocabulary projection, backprop, AdamW step, clone) exercised in seconds while keeping
/// the architecture's SHAPE faithful — the dims below are scaled down from the paper config and
/// the wiring is unchanged. Dropout is disabled (already the default) so the memorization-based
/// invariants see clean, monotonic convergence.
/// </para>
/// </remarks>
public class WhisperTimestampedTests : AudioNNModelTestBase<float>
{
    // Whisper reads waveforms: 0.2 s, padded to its 30-second window.
    protected override int[] InputShape => [3200];

    // The decoder's logits over the four start tokens.
    protected override int[] OutputShape => [4, 51865];

    protected override int MoreDataShortIterations => 3;
    protected override int MoreDataLongIterations => 10;

    protected override INeuralNetworkModel<float> CreateNetwork()
    {
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.SequenceToSequence,
            inputSize: 3200,
            outputSize: 51865);

        // openai-whisper's network at width 16 with one encoder and one decoder block; the 80-bin, 30-second window and
        // the multilingual vocabulary the tokenizer needs are kept.
        var options = new WhisperTimestampedOptions
        {
            NumMels = 80,
            EncoderDim = 16,
            DecoderDim = 16,
            NumEncoderLayers = 1,
            NumDecoderLayers = 1,
            NumAttentionHeads = 2,
            VocabSize = 51865,
            Language = "en",
        };
        return new WhisperTimestamped<float>(architecture, options);
    }
}

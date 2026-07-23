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
    // Whisper consumes frame-major log-mel features: [batch, frames, mels].
    protected override int[] InputShape => [1, 8, 80];

    // Per-frame vocabulary logits: [batch, frames, vocab].
    protected override int[] OutputShape => [1, 8, 4];

    // The encoder/decoder stack is tiny at this scale, so a training step is sub-10ms;
    // the heavier MoreData invariant (default 50/200 steps) still runs comfortably, but
    // matching the FasterWhisper smoke counts keeps the whole class well under budget.
    protected override int MoreDataShortIterations => 3;
    protected override int MoreDataLongIterations => 10;

    protected override INeuralNetworkModel<float> CreateNetwork()
    {
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceToSequence,
            inputHeight: 8,
            inputWidth: 80,
            inputDepth: 1,
            outputSize: 4);

        // Reduced-scale config (see <remarks>): same architecture shape as the paper
        // model, smoke-test width/depth/vocab so all invariants fit the CI budget.
        var options = new WhisperTimestampedOptions
        {
            SampleRate = 16000,
            NumMels = 80,
            EncoderDim = 64,
            DecoderDim = 64,
            NumEncoderLayers = 1,
            NumDecoderLayers = 1,
            NumAttentionHeads = 4,
            VocabSize = 4,
            DropoutRate = 0.0,
        };
        return new WhisperTimestamped<float>(architecture, options);
    }
}

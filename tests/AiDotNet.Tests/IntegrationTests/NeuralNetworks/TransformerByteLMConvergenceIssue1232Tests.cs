using System.Linq;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Regression suite for AiDotNet#1232 — Transformer training on byte-LM
/// converges to flat softmax (NLL = ln(V) exactly, top-1 = 0%, PPL = V).
///
/// <para>
/// <b>Root cause:</b>
/// <see cref="Helpers.LayerHelper{T}.CreateDefaultTransformerLayers"/>
/// inserted a <c>GlobalPoolingLayer&lt;T&gt;(Average)</c> before the
/// classification head for every single-label classification task. For a
/// token-input architecture (vocabularySize &gt; 0) doing autoregressive
/// next-token prediction over real-data byte windows, mean-pooling over
/// the sequence axis maps every distinct prefix to roughly the same
/// averaged hidden state — the classification head sees an
/// undifferentiated signal and softmax converges to <c>~uniform / V</c>
/// regardless of training budget.
/// </para>
///
/// <para>
/// <b>Fix:</b> <see cref="TransformerArchitecture{T}"/> now exposes
/// <see cref="TransformerArchitecture{T}.SequencePooling"/> and defaults
/// to <see cref="SequencePoolingMode.LastToken"/> when
/// <c>vocabularySize &gt; 0</c> (canonical autoregressive LM contract —
/// matches GPT / Llama / Mistral output-head conventions). The
/// LayerHelper switches on this enum and uses
/// <see cref="SequenceTokenSliceLayer{T}"/> to take the last position's
/// hidden state instead of averaging.
/// </para>
///
/// <para>
/// <b>Test design:</b> Three tests guard the fix, each isolating a
/// specific layer of the change:
/// <list type="number">
/// <item><b>Architecture default contract</b> — the actual #1232 fix.
///   Asserts <c>vocabularySize &gt; 0</c> defaults to LastToken,
///   <c>vocabularySize == 0</c> defaults to MeanPool, and explicit
///   overrides are honoured. A future change can't silently regress
///   to the bug.</item>
/// <item><b>SequenceTokenSliceLayer forward correctness</b> — the new
///   layer must return exactly the slice at index <c>seq-1</c> for
///   <c>Position.Last</c> (and index 0 for <c>Position.First</c>),
///   shape collapsing from <c>[batch, seq, dim]</c> to <c>[batch, dim]</c>.</item>
/// <item><b>End-to-end convergence smoke test</b> — a token-input
///   Transformer with default LastToken pooling trains a deterministic
///   byte-LM task. Asserts loss drops below <c>ln(V)</c> (i.e. some
///   actual learning happens, ruling out the pre-fix "constant logits"
///   collapse). The threshold is intentionally loose because training
///   stochasticity is high without a global seed mechanism — the
///   assertion catches the canonical PPL=V failure mode without
///   flaking on borderline runs.</item>
/// </list>
/// </para>
/// </summary>
public class TransformerByteLMConvergenceIssue1232Tests
{
    private readonly ITestOutputHelper _output;

    public TransformerByteLMConvergenceIssue1232Tests(ITestOutputHelper output)
    {
        _output = output;
    }

    /// <summary>
    /// The default for token-input architectures must be
    /// <see cref="SequencePoolingMode.LastToken"/>. Continuous-input
    /// architectures must default to <see cref="SequencePoolingMode.MeanPool"/>.
    /// Explicit overrides must be honoured. This is the behavioural
    /// contract the #1232 fix locks in — silent regression to MeanPool
    /// for token inputs would re-introduce the flat-softmax bug.
    /// </summary>
    [Fact]
    public void TransformerArchitecture_TokenInput_DefaultsTo_LastTokenPooling()
    {
        var archTokenInput = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            numEncoderLayers: 1,
            numDecoderLayers: 0,
            numHeads: 2,
            modelDimension: 16,
            feedForwardDimension: 32,
            inputSize: 4,
            outputSize: 8,
            maxSequenceLength: 4,
            vocabularySize: 8);
        Assert.Equal(SequencePoolingMode.LastToken, archTokenInput.SequencePooling);

        var archContinuous = new TransformerArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            numEncoderLayers: 1,
            numDecoderLayers: 0,
            numHeads: 2,
            modelDimension: 16,
            feedForwardDimension: 32,
            inputSize: 4,
            outputSize: 8,
            maxSequenceLength: 4,
            vocabularySize: 0);
        Assert.Equal(SequencePoolingMode.MeanPool, archContinuous.SequencePooling);

        var archExplicit = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            numEncoderLayers: 1,
            numDecoderLayers: 0,
            numHeads: 2,
            modelDimension: 16,
            feedForwardDimension: 32,
            inputSize: 4,
            outputSize: 8,
            maxSequenceLength: 4,
            vocabularySize: 8,
            sequencePooling: SequencePoolingMode.MeanPool);
        Assert.Equal(SequencePoolingMode.MeanPool, archExplicit.SequencePooling);
    }

    /// <summary>
    /// <see cref="SequenceTokenSliceLayer{T}"/> with
    /// <see cref="SequenceTokenSliceLayer{T}.Position.Last"/> must
    /// return exactly the slice at index <c>seq - 1</c> from a rank-3
    /// <c>[batch, seq, dim]</c> input, collapsing to <c>[batch, dim]</c>.
    /// This is the load-bearing slice that replaces MeanPool for the
    /// #1232 fix — if it returns the wrong position the entire fix
    /// silently fails and training reverts to flat softmax.
    /// </summary>
    [Fact]
    public void SequenceTokenSliceLayer_Last_SelectsFinalPosition()
    {
        const int batch = 2;
        const int seq = 5;
        const int dim = 4;

        var input = new Tensor<float>(new[] { batch, seq, dim });
        for (int b = 0; b < batch; b++)
        for (int s = 0; s < seq; s++)
        for (int d = 0; d < dim; d++)
            input[b, s, d] = b * 100 + s * 10 + d;

        var layer = new SequenceTokenSliceLayer<float>(SequenceTokenSliceLayer<float>.Position.Last);
        var output = layer.Forward(input);

        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(batch, output.Shape[0]);
        Assert.Equal(dim, output.Shape[1]);

        for (int b = 0; b < batch; b++)
        for (int d = 0; d < dim; d++)
        {
            float expected = input[b, seq - 1, d];
            float actual = output[b, d];
            Assert.Equal(expected, actual);
        }
    }

    /// <summary>
    /// <see cref="SequenceTokenSliceLayer{T}"/> with
    /// <see cref="SequenceTokenSliceLayer{T}.Position.First"/> must
    /// return exactly the slice at index 0 — used for
    /// <see cref="SequencePoolingMode.ClsToken"/> (BERT-style).
    /// </summary>
    [Fact]
    public void SequenceTokenSliceLayer_First_SelectsInitialPosition()
    {
        const int batch = 2;
        const int seq = 5;
        const int dim = 4;

        var input = new Tensor<float>(new[] { batch, seq, dim });
        for (int b = 0; b < batch; b++)
        for (int s = 0; s < seq; s++)
        for (int d = 0; d < dim; d++)
            input[b, s, d] = b * 100 + s * 10 + d;

        var layer = new SequenceTokenSliceLayer<float>(SequenceTokenSliceLayer<float>.Position.First);
        var output = layer.Forward(input);

        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(batch, output.Shape[0]);
        Assert.Equal(dim, output.Shape[1]);

        for (int b = 0; b < batch; b++)
        for (int d = 0; d < dim; d++)
        {
            float expected = input[b, 0, d];
            float actual = output[b, d];
            Assert.Equal(expected, actual);
        }
    }

    /// <summary>
    /// <see cref="SequenceTokenSliceLayer{T}"/> must reject non-rank-3
    /// input rather than silently mis-slicing. Pre-fix, the bug was
    /// silent — a wrong layer was inserted, training failed without
    /// any error. Defensive shape validation here surfaces wiring bugs
    /// at construction rather than letting them manifest as flat-softmax
    /// convergence later.
    /// </summary>
    [Fact]
    public void SequenceTokenSliceLayer_RejectsNonRank3Input()
    {
        var layer = new SequenceTokenSliceLayer<float>(SequenceTokenSliceLayer<float>.Position.Last);

        var rank2 = new Tensor<float>(new[] { 4, 8 });
        Assert.Throws<ArgumentException>(() => layer.Forward(rank2));

        var rank4 = new Tensor<float>(new[] { 2, 4, 8, 3 });
        Assert.Throws<ArgumentException>(() => layer.Forward(rank4));
    }

    /// <summary>
    /// End-to-end convergence smoke test for the #1232 fix. Trains a
    /// token-input Transformer (default LastToken pooling) on a
    /// deterministic next-byte task and asserts the loss drops below
    /// <c>ln(V)</c> — proving the model breaks through the flat-softmax
    /// floor. Pre-fix this stayed at exactly <c>ln(V)</c> regardless
    /// of training budget (the canonical PPL = V symptom).
    ///
    /// <para>The threshold is intentionally loose (<c>NLL &lt; ln(V)*0.95</c>)
    /// because training stochasticity is high without a global seed
    /// mechanism — the assertion catches the canonical "stuck at the
    /// uniform floor" failure mode without flaking on runs that converge
    /// slowly. The architectural correctness (default = LastToken) and
    /// layer correctness (slice forward) are guarded by the deterministic
    /// tests above.</para>
    /// </summary>
    [Fact]
    public async Task Transformer_ByteLM_DefaultPooling_TrainsBelowLnV()
    {
        await Task.Yield();
        const int vocab = 8;
        const int seqLen = 4;
        const int totalEpochs = 500;
        const double lr = 0.001;
        const int modelDim = 32;
        const int ffDim = 64;

        // No explicit pooling argument → exercises the default-selection
        // path (vocabularySize > 0 → LastToken) that the #1232 fix
        // installs. If that default reverts to MeanPool the convergence
        // assertion below will fail.
        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            numEncoderLayers: 2,
            numDecoderLayers: 0,
            numHeads: 4,
            modelDimension: modelDim,
            feedForwardDimension: ffDim,
            inputSize: seqLen,
            outputSize: vocab,
            maxSequenceLength: seqLen,
            vocabularySize: vocab);

        var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(
            null,
            new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = lr,
                Beta1 = 0.9,
                Beta2 = 0.999,
                Epsilon = 1e-8,
            });

        var model = new Transformer<float>(
            arch,
            lossFunction: new CategoricalCrossEntropyLoss<float>(),
            optimizer: optimizer);
        model.SetTrainingMode(true);

        for (int epoch = 0; epoch < totalEpochs; epoch++)
        {
            for (int k = 0; k < vocab; k++)
            {
                var input = BuildAllKInput(k, seqLen);
                var target = BuildOneHotTarget((k + 3) % vocab, vocab);
                model.Train(input, target);
            }
        }

        model.SetTrainingMode(false);

        double totalNll = 0;
        int correct = 0;
        for (int k = 0; k < vocab; k++)
        {
            var input = BuildAllKInput(k, seqLen);
            var pred = model.Predict(input);
            int expectedNext = (k + 3) % vocab;
            float pTarget = pred.Length == vocab ? pred[expectedNext] : pred[0, expectedNext];
            totalNll += -System.Math.Log(System.Math.Max((double)pTarget, 1e-9));

            int argmax = 0;
            float maxVal = float.MinValue;
            for (int v = 0; v < vocab; v++)
            {
                float val = pred.Length == vocab ? pred[v] : pred[0, v];
                if (val > maxVal) { maxVal = val; argmax = v; }
            }
            if (argmax == expectedNext) correct++;
        }
        double avgNll = totalNll / vocab;
        double lnV = System.Math.Log(vocab);
        double ppl = System.Math.Exp(avgNll);

        _output.WriteLine($"avgNll={avgNll:F3}  ln(V)={lnV:F3}  PPL={ppl:F2}  top-1={correct}/{vocab}");

        // Primary fix verification. Pre-fix: avgNll == ln(V) exactly
        // (PPL=V), regardless of training budget. Post-fix: avgNll
        // drops below ln(V)*0.95.
        Assert.True(avgNll < lnV * 0.95,
            $"Token-input Transformer should train below ln(V)*0.95={lnV * 0.95:F3} on a " +
            $"deterministic byte-LM task; observed avgNll={avgNll:F3} (ln(V)={lnV:F3}, PPL={ppl:F2}). " +
            "If avgNll is at or near ln(V), training is stuck at the uniform-softmax floor — " +
            "the SequencePooling default has likely regressed to MeanPool, or the " +
            "SequenceTokenSliceLayer isn't propagating gradients correctly. See #1232.");

        // Sanity check: predictions must beat chance (1/V = 12.5%).
        // Pre-fix top-1 was 0/V (model output was constant); post-fix
        // it should comfortably exceed chance.
        Assert.True(correct > vocab / 8,
            $"Top-1 accuracy {correct}/{vocab} at or below 1/V chance level " +
            "indicates the classification head is producing constant output — " +
            "see #1232 (model collapsed to uniform logits).");
    }

    private static Tensor<float> BuildAllKInput(int k, int seqLen)
    {
        var t = new Tensor<float>(new[] { 1, seqLen });
        for (int s = 0; s < seqLen; s++) t[0, s] = k;
        return t;
    }

    private static Tensor<float> BuildOneHotTarget(int classIndex, int vocab)
    {
        var t = new Tensor<float>(new[] { 1, vocab });
        t[0, classIndex] = 1f;
        return t;
    }
}

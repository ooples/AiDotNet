using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.NER.Options;
using AiDotNet.NER.SequenceLabeling;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NER;

/// <summary>
/// Self-contained unit tests for <see cref="WordCharBiLSTMCRF{T}"/> (Lample et al., NAACL 2016).
/// These deliberately avoid the model-family test scaffold (which drives every NER model at its
/// 256-token / 100-dim defaults) and instead use a tiny, fast, deterministic configuration so the
/// invariants run in-process without the heavy memory footprint of the full-size model.
/// </summary>
/// <remarks>
/// The tests exercise the model ONLY through the facade-compatible surface — construction, the
/// <c>Train(Tensor, Tensor)</c> step that <c>AiModelBuilder.BuildAsync</c> drives, and the
/// <c>PredictLabels</c>/<c>DecodeLabels</c> path that <c>AiModelResult.Predict</c> drives — never a
/// bespoke train-by-sentence shortcut. Two of them are the regressions this PR closes:
/// <list type="bullet">
/// <item><see cref="Training_DrivesTokenAccuracyUp_AndEmbeddingsTrainEndToEnd"/> proves the
/// per-timestep LSTM input slice is now tape-tracked, so gradient reaches the embedding tables
/// (previously frozen) and the model can actually fit a task.</item>
/// <item><see cref="Predictions_AreBioValid_AndDeterministic_AndOovSafe"/> proves the CRF yields
/// structurally valid, reproducible output and that unseen words do not crash inference.</item>
/// </list>
/// </remarks>
public class WordCharBiLSTMCRFTests
{
    // BIO label set: O / B-PER / I-PER. Index 0 must be "O" (the pad/outside tag).
    private static readonly string[] Labels = ["O", "B-PER", "I-PER"];

    // Tiny synthetic corpus: "alice smith" is always a two-token PER span; every other token is O.
    private static readonly string[][] Sentences =
    [
        ["alice", "smith", "left"],
        ["we", "saw", "alice", "smith"],
        ["alice", "smith", "and", "bob"],
        ["hello", "alice", "smith"],
    ];

    private static readonly int[][] Gold =
    [
        [1, 2, 0],       // B-PER I-PER O
        [0, 0, 1, 2],    // O O B-PER I-PER
        [1, 2, 0, 0],    // B-PER I-PER O O
        [0, 1, 2],       // O B-PER I-PER
    ];

    private const int MaxSeqLen = 6;

    private static BiLSTMCRFOptions TinyOptions() => new()
    {
        EmbeddingDimension = 8,
        HiddenDimension = 8,
        CharEmbeddingDimension = 4,
        CharHiddenDimension = 4,
        NumLSTMLayers = 1,
        MaxSequenceLength = MaxSeqLen,
        NumLabels = Labels.Length,
        LabelNames = Labels,
        UseCRF = true,
        DropoutRate = 0.0,        // deterministic forward (no stochastic dropout)
        LearningRate = 0.05,
    };

    private static WordCharBiLSTMCRF<double> BuildModel() =>
        WordCharBiLSTMCRF<double>.Create(Sentences, TinyOptions(), maxWordLength: 8);

    private static Tensor<double> GoldTensor(int[] labelIndices)
    {
        var data = new double[labelIndices.Length];
        for (int i = 0; i < labelIndices.Length; i++) data[i] = labelIndices[i];
        return new Tensor<double>(new Vector<double>(data), [labelIndices.Length]);
    }

    /// <summary>Token accuracy over the real (non-padding) positions of every training sentence.</summary>
    private static double TokenAccuracy(WordCharBiLSTMCRF<double> model)
    {
        int correct = 0, total = 0;
        for (int s = 0; s < Sentences.Length; s++)
        {
            var predicted = model.DecodeLabels(model.PredictLabels(model.EncodeSentence(Sentences[s])));
            for (int t = 0; t < Sentences[s].Length; t++)
            {
                total++;
                if (t < predicted.Length && predicted[t] == Labels[Gold[s][t]]) correct++;
            }
        }
        return (double)correct / total;
    }

    private static void TrainEpochs(WordCharBiLSTMCRF<double> model, int epochs)
    {
        model.SetTrainingMode(true);
        for (int e = 0; e < epochs; e++)
            for (int s = 0; s < Sentences.Length; s++)
                model.Train(model.EncodeSentence(Sentences[s]), GoldTensor(Gold[s]));
    }

    [Fact]
    public void Training_DrivesTokenAccuracyUp_AndEmbeddingsTrainEndToEnd()
    {
        var model = BuildModel();

        // Snapshot the embedding front-end parameters BEFORE training. If the per-timestep LSTM
        // input slice severs the tape (the bug this PR fixes), gradient never reaches these tables
        // and they stay byte-for-byte at their random initialization.
        var embedBefore = model.EmbeddingFrontEnd!.GetParameters().Clone();
        double accBefore = TokenAccuracy(model);

        TrainEpochs(model, epochs: 150);

        var embedAfter = model.EmbeddingFrontEnd!.GetParameters();
        double maxDelta = 0;
        for (int i = 0; i < embedBefore.Length; i++)
            maxDelta = Math.Max(maxDelta, Math.Abs(embedAfter[i] - embedBefore[i]));

        Assert.True(maxDelta > 1e-4,
            $"Embedding tables did not change after training (max delta {maxDelta:E3}) — gradient is " +
            "not reaching the embeddings, so the per-timestep LSTM slice is not tape-tracked.");

        double accAfter = TokenAccuracy(model);
        Assert.True(accAfter > accBefore,
            $"Token accuracy did not improve with training ({accBefore:P0} -> {accAfter:P0}).");
        Assert.True(accAfter >= 0.75,
            $"Model failed to fit the tiny memorization task (final token accuracy {accAfter:P0}).");
    }

    [Fact]
    public void Predictions_AreBioValid_AndDeterministic_AndOovSafe()
    {
        var model = BuildModel();
        TrainEpochs(model, epochs: 80);
        model.SetTrainingMode(false);

        // 1) BIO validity: an "I-X" tag must be preceded by "B-X" or "I-X" of the same type.
        //    The CRF is responsible for never emitting an orphan "I-".
        foreach (var tokens in Sentences)
        {
            var tags = model.DecodeLabels(model.PredictLabels(model.EncodeSentence(tokens)));
            AssertBioValid(tags);
        }

        // 2) Determinism: identical input must give identical output across calls.
        var x = model.EncodeSentence(Sentences[0]);
        var firstRun = model.DecodeLabels(model.PredictLabels(x));
        var secondRun = model.DecodeLabels(model.PredictLabels(x));
        Assert.Equal(firstRun, secondRun);

        // 3) OOV safety: a sentence of entirely unseen words must not throw and must stay BIO-valid.
        var oov = model.DecodeLabels(model.PredictLabels(
            model.EncodeSentence(["zzzqux", "vproxx", "neverseen"])));
        AssertBioValid(oov);
    }

    [Fact]
    public void Create_BuildsVocabularyFromData_AndEncodesToModelInputShape()
    {
        var model = BuildModel();
        var encoded = model.EncodeSentence(Sentences[0]);

        // Facade input contract: a [MaxSequenceLength, 1 + maxWordLength] packed-index tensor.
        Assert.Equal(2, encoded.Rank);
        Assert.Equal(MaxSeqLen, encoded.Shape[0]);
        Assert.Equal(1 + model.Encoder.MaxWordLength, encoded.Shape[1]);

        // Known training words landed in the vocabulary (index > 0 = not the [UNK]/pad slot).
        Assert.True(model.Encoder.WordVocabulary.TokenToId.ContainsKey("alice"));
        Assert.True(model.Encoder.WordVocabulary.TokenToId.ContainsKey("smith"));
    }

    private static void AssertBioValid(string[] tags)
    {
        string prevType = "";
        foreach (var tag in tags)
        {
            if (tag == "O" || string.IsNullOrEmpty(tag))
            {
                prevType = "";
                continue;
            }

            string prefix = tag.Length >= 2 ? tag.Substring(0, 2) : tag;
            string type = tag.Length > 2 ? tag.Substring(2) : "";

            if (prefix == "I-")
                Assert.True(prevType == type,
                    $"Orphan I- tag '{tag}' not continuing a same-type span (prev type '{prevType}'). " +
                    "The CRF must never emit a BIO-invalid sequence.");

            prevType = type;
        }
    }
}

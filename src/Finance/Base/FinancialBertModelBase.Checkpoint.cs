using AiDotNet.ModelLoading;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Finance.Base;

public abstract partial class FinancialBertModelBase<T>
{
    /// <summary>
    /// Loads a Hugging Face BERT checkpoint (<c>BertModel</c>, <c>BertForMaskedLM</c> or
    /// <c>BertForSequenceClassification</c> safetensors) into the encoder and, when the checkpoint has a classifier of
    /// the head's size, into the head.
    /// </summary>
    /// <param name="path">The checkpoint folder, or a file inside it.</param>
    /// <remarks>
    /// <para>
    /// Names follow the released checkpoints: <c>[bert.]embeddings.*</c>, <c>[bert.]encoder.layer.N.*</c>,
    /// <c>[bert.]pooler.dense</c> and <c>classifier</c>. A checkpoint without a pooler or classifier (a masked-language
    /// model) leaves those layers at their initialization, the usual start of fine-tuning. The key projection's bias is
    /// not stored: it adds the same score to every key of a query, which the softmax cancels.
    /// </para>
    /// <para>
    /// Class names come from the checkpoint's <c>config.json</c> <c>id2label</c> when it declares as many as the head
    /// predicts.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">The layers cannot hold the checkpoint: a Pre-LN stack, a different
    /// number of blocks, or no BERT embedding layer.</exception>
    private void LoadPretrainedCheckpointCore(string path)
    {
        if (BertOptions.NormPlacement != TransformerNormPlacement.PostNorm)
            throw new InvalidOperationException(
                "A BERT checkpoint holds a Post-LN encoder; set NormPlacement to PostNorm to load it.");

        string directory = CheckpointDirectory(path);
        using var source = TorchCheckpointFile.OpenDirectory(directory);
        string prefix = source.Contains("bert.embeddings.word_embeddings.weight") ? "bert." : "";

        int encoderEnd = EncoderLayerCount;
        var encoder = Layers.Take(encoderEnd).ToList();
        var embedding = encoder.OfType<BertEmbeddingLayer<T>>().FirstOrDefault()
            ?? throw new InvalidOperationException("The encoder has no BertEmbeddingLayer to load the embeddings into.");
        var blocks = encoder.OfType<TransformerEncoderBlock<T>>().ToList();

        int stored = 0;
        while (source.Contains($"{prefix}encoder.layer.{stored}.attention.self.query.weight")) stored++;
        if (stored != blocks.Count)
            throw new InvalidOperationException(
                $"The checkpoint has {stored} encoder layers but the model has {blocks.Count}; set NumLayers to {stored}.");

        int h = HiddenDimension;
        var engine = Engine;
        LoadEmbedding(embedding.WordEmbedding, source, $"{prefix}embeddings.word_embeddings.weight");
        LoadEmbedding(embedding.PositionEmbedding, source, $"{prefix}embeddings.position_embeddings.weight");
        LoadEmbedding(embedding.TokenTypeEmbedding, source, $"{prefix}embeddings.token_type_embeddings.weight");
        TorchParameters.LayerNorm(engine, embedding.Normalization, source,
            $"{prefix}embeddings.LayerNorm.weight", $"{prefix}embeddings.LayerNorm.bias");

        for (int i = 0; i < blocks.Count; i++)
        {
            var block = blocks[i];
            string n = $"{prefix}encoder.layer.{i}";
            var attention = block.AttentionLayer as MultiHeadAttentionLayer<T>
                ?? throw new InvalidOperationException($"Block {i} does not use MultiHeadAttentionLayer.");
            LoadProjection(attention.GetQueryWeights(), attention.GetQueryBias(), source, $"{n}.attention.self.query", h, h);
            LoadProjection(attention.GetKeyWeights(), null, source, $"{n}.attention.self.key", h, h);
            LoadProjection(attention.GetValueWeights(), attention.GetValueBias(), source, $"{n}.attention.self.value", h, h);
            LoadProjection(attention.GetOutputWeights(), attention.GetOutputBias(), source,
                $"{n}.attention.output.dense", h, h);
            TorchParameters.LayerNorm(engine, (LayerNormalizationLayer<T>)block.Norm1Layer, source,
                $"{n}.attention.output.LayerNorm.weight", $"{n}.attention.output.LayerNorm.bias");
            TorchParameters.Linear(engine, (DenseLayer<T>)block.FfnUpLayer, h, block.FfnDim, source,
                $"{n}.intermediate.dense.weight", $"{n}.intermediate.dense.bias");
            TorchParameters.Linear(engine, (DenseLayer<T>)block.FfnDownLayer, block.FfnDim, h, source,
                $"{n}.output.dense.weight", $"{n}.output.dense.bias");
            TorchParameters.LayerNorm(engine, (LayerNormalizationLayer<T>)block.Norm2Layer, source,
                $"{n}.output.LayerNorm.weight", $"{n}.output.LayerNorm.bias");
        }

        // The pooler is the dense layer after the [CLS] slice.
        int slice = encoder.FindIndex(layer => layer is SequenceTokenSliceLayer<T>);
        if (slice >= 0 && slice + 1 < encoder.Count && encoder[slice + 1] is DenseLayer<T> pooler
            && source.Contains($"{prefix}pooler.dense.weight"))
        {
            TorchParameters.Linear(engine, pooler, h, h, source, $"{prefix}pooler.dense.weight", $"{prefix}pooler.dense.bias");
        }

        var classifier = Layers.Skip(encoderEnd).OfType<DenseLayer<T>>().LastOrDefault();
        int numClasses = NumSentimentClasses;
        if (classifier is not null && source.ShapeOf("classifier.weight") is { Length: 2 } shape
            && shape[0] == numClasses && shape[1] == h)
        {
            TorchParameters.Linear(engine, classifier, h, numClasses, source, "classifier.weight", "classifier.bias");
        }

        var labels = ReadClassLabels(directory, numClasses);
        if (labels is not null)
            SetClassLabels(labels);
    }

    private void LoadEmbedding(EmbeddingLayer<T> layer, TorchTensorSource source, string name)
    {
        var table = layer.GetTrainableParameters();
        if (table.Count != 1 || table[0].Rank != 2)
            throw new InvalidOperationException($"Cannot load {name}: the embedding table is not a single matrix.");
        source.CopyTo(name, new[] { table[0].Shape[0], table[0].Shape[1] }, table[0]);
        Engine.InvalidatePersistentTensor(table[0]);
    }

    /// <summary>Loads a stored <c>weight [out, in]</c> into an <c>[in, out]</c> projection, and its bias if any.</summary>
    private void LoadProjection(Tensor<T> weights, Tensor<T>? bias, TorchTensorSource source, string name, int inputs,
        int outputs)
    {
        source.CopyTransposedTo($"{name}.weight", new[] { outputs, inputs }, outputs, inputs, weights);
        Engine.InvalidatePersistentTensor(weights);
        if (bias is null)
            return;
        source.CopyTo($"{name}.bias", new[] { outputs }, bias);
        Engine.InvalidatePersistentTensor(bias);
    }

    /// <summary>The checkpoint's <c>id2label</c> class names, when it declares exactly <paramref name="numClasses"/>.</summary>
    private static string[]? ReadClassLabels(string directory, int numClasses)
    {
        string configPath = System.IO.Path.Combine(directory, "config.json");
        if (!System.IO.File.Exists(configPath))
            return null;
        if (JObject.Parse(System.IO.File.ReadAllText(configPath))["id2label"] is not JObject id2label
            || id2label.Count != numClasses)
            return null;
        var labels = new string[numClasses];
        foreach (var pair in id2label.Properties())
        {
            if (!int.TryParse(pair.Name, System.Globalization.NumberStyles.Integer,
                    System.Globalization.CultureInfo.InvariantCulture, out int index)
                || index < 0 || index >= numClasses)
                return null;
            labels[index] = pair.Value.ToString().ToLowerInvariant();
        }

        return labels.Any(string.IsNullOrEmpty) ? null : labels;
    }
}

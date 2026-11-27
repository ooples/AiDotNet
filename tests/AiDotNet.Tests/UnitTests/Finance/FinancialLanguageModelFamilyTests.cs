using System.Text;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Finance.Base;
using AiDotNet.Finance.Enums;
using AiDotNet.Finance.NLP;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tokenization.Algorithms;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// The contract every financial language model keeps: training forwards stay in training mode with dropout on,
/// inference is deterministic, a classifier answers once per input, the encoder and head split honours custom layers,
/// fine-tuning freezes hold, tokenization never returns placeholder ids, and a Hugging Face BERT checkpoint lands in
/// the right layers.
/// </summary>
public sealed class FinancialLanguageModelFamilyTests
{
    private const int Vocabulary = 32;
    private const int Sequence = 8;
    private const int Hidden = 8;
    private const int Classes = 3;

    public static IEnumerable<object[]> Models() =>
        new[] { "FinBERT", "FinancialBERT", "SECBERT", "FinBERTTone", "FinGPT", "FinMA", "InvestLM", "BloombergGPT" }
            .Select(name => new object[] { name });

    private static NeuralNetworkArchitecture<double> Architecture(List<ILayer<double>>? layers = null) =>
        new(inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: Sequence, outputSize: Classes, layers: layers);

    /// <summary>Sets a size option every model declares on its own options class.</summary>
    private static void Set(object options, string property, object value)
    {
        var info = options.GetType().GetProperty(property)
            ?? throw new InvalidOperationException($"{options.GetType().Name} has no {property}.");
        info.SetValue(options, value);
    }

    private static T Small<T>(T options, double dropout) where T : FinancialLanguageModelOptions<double>
    {
        Set(options, "MaxSequenceLength", Sequence);
        Set(options, "VocabularySize", Vocabulary);
        Set(options, "HiddenDimension", Hidden);
        Set(options, "NumAttentionHeads", 2);
        Set(options, "IntermediateDimension", 16);
        Set(options, "NumLayers", 2);
        Set(options, "DropoutRate", dropout);
        if (options is FinancialBertEncoderOptions<double> bert)
        {
            bert.MaxPositionEmbeddings = Sequence;
            bert.AttentionDropoutRate = dropout;
        }

        return options;
    }

    private static FinancialNLPModelBase<double> Create(
        string name, double dropout = 0.0, Action<FinancialLanguageModelOptions<double>>? configure = null,
        List<ILayer<double>>? layers = null)
    {
        var architecture = Architecture(layers);
        FinancialNLPModelBase<double> Build<TOptions>(TOptions options, Func<TOptions, FinancialNLPModelBase<double>> make)
            where TOptions : FinancialLanguageModelOptions<double>
        {
            Small(options, dropout);
            configure?.Invoke(options);
            return make(options);
        }

        return name switch
        {
            "FinBERT" => Build(new FinBERTOptions<double> { NumSentimentClasses = Classes }, o => new FinBERT<double>(architecture, o)),
            "FinancialBERT" => Build(new FinancialBERTOptions<double> { NumClasses = Classes }, o => new FinancialBERT<double>(architecture, o)),
            "SECBERT" => Build(new SECBERTOptions<double> { NumClasses = Classes }, o => new SECBERT<double>(architecture, o)),
            "FinBERTTone" => Build(new FinBERTToneOptions<double> { NumToneClasses = Classes }, o => new FinBERTTone<double>(architecture, o)),
            "FinGPT" => Build(new FinGPTOptions<double> { NumClasses = Classes }, o => new FinGPT<double>(architecture, o)),
            "FinMA" => Build(new FinMAOptions<double> { NumClasses = Classes }, o => new FinMA<double>(architecture, o)),
            "InvestLM" => Build(new InvestLMOptions<double> { NumClasses = Classes }, o => new InvestLM<double>(architecture, o)),
            "BloombergGPT" => Build(new BloombergGPTOptions<double> { NumClasses = Classes }, o => new BloombergGPT<double>(architecture, o)),
            _ => throw new ArgumentOutOfRangeException(nameof(name), name, "Unknown model."),
        };
    }

    private static Tensor<double> Tokens(int batch = 2)
    {
        var tokens = new Tensor<double>(new[] { batch, Sequence });
        for (int i = 0; i < tokens.Length; i++) tokens[i] = (i * 7 + 3) % Vocabulary;
        return tokens;
    }

    private static Tensor<double> OneHot(int batch = 2)
    {
        var target = new Tensor<double>(new[] { batch, Classes });
        for (int b = 0; b < batch; b++) target[b, b % Classes] = 1.0;
        return target;
    }

    private static double MaxAbsDifference(Tensor<double> a, Tensor<double> b)
    {
        Assert.Equal(a.Shape.ToArray(), b.Shape.ToArray());
        double max = 0;
        for (int i = 0; i < a.Length; i++) max = Math.Max(max, Math.Abs(a[i] - b[i]));
        return max;
    }

    [Theory]
    [MemberData(nameof(Models))]
    public void ForwardForTraining_StaysInTrainingMode(string name)
    {
        var model = Create(name, dropout: 0.1);
        model.SetTrainingMode(true);

        model.ForwardForTraining(Tokens());

        Assert.True(model.IsTrainingMode, "ForwardForTraining switched the model to evaluation mode, turning dropout off.");
    }

    [Theory]
    [MemberData(nameof(Models))]
    public void Dropout_IsActiveInTrainingAndOffInInference(string name)
    {
        var model = Create(name, dropout: 0.5);
        var tokens = Tokens();

        model.SetTrainingMode(true);
        var first = model.ForwardForTraining(tokens);
        var second = model.ForwardForTraining(tokens);
        Assert.True(MaxAbsDifference(first, second) > 1e-9, "Two training forwards agreed exactly: dropout is off.");

        var predicted = model.Predict(tokens);
        var again = model.Predict(tokens);
        Assert.Equal(0.0, MaxAbsDifference(predicted, again));
    }

    [Theory]
    [MemberData(nameof(Models))]
    public void Predict_AnswersOncePerInput(string name)
    {
        var model = Create(name);

        var output = model.Predict(Tokens(batch: 2));

        Assert.Equal(new[] { 2, Classes }, output.Shape.ToArray());
    }

    [Theory]
    [MemberData(nameof(Models))]
    public void Embeddings_HaveTheirDocumentedShapes(string name)
    {
        var model = Create(name);
        var tokens = Tokens(batch: 2);

        Assert.Equal(new[] { 2, Hidden }, model.GetSequenceEmbedding(tokens).Shape.ToArray());
        Assert.Equal(new[] { 2, Sequence, Hidden }, model.GetEmbeddings(tokens).Shape.ToArray());
    }

    [Theory]
    [MemberData(nameof(Models))]
    public void ClassificationHeadFactory_ReplacesTheDefaultHead(string name)
    {
        var model = Create(name, configure: o => o.ClassificationHeadFactory =
            () => new ILayer<double>[] { new DenseLayer<double>(5, new IdentityActivation<double>() as IActivationFunction<double>) });

        Assert.Equal(new[] { 2, 5 }, model.Predict(Tokens()).Shape.ToArray());
    }

    [Theory]
    [MemberData(nameof(Models))]
    public void CustomLayers_GetTheDefaultHeadUnlessItIsTurnedOff(string name)
    {
        List<ILayer<double>> Encoder() => new()
        {
            new EmbeddingLayer<double>(Vocabulary, Hidden),
            new SequenceTokenSliceLayer<double>(SequenceTokenSliceLayer<double>.Position.Last),
        };

        var withHead = Create(name, layers: Encoder());
        Assert.Equal(new[] { 2, Classes }, withHead.Predict(Tokens()).Shape.ToArray());
        Assert.Equal(new[] { 2, Hidden }, withHead.GetSequenceEmbedding(Tokens()).Shape.ToArray());

        var withoutHead = Create(name, configure: o => o.UseDefaultClassificationHead = false, layers: Encoder());
        Assert.Equal(new[] { 2, Hidden }, withoutHead.Predict(Tokens()).Shape.ToArray());
    }

    [Theory]
    [MemberData(nameof(Models))]
    public void FreezeBaseModel_TrainsOnlyTheHead(string name)
    {
        var model = Create(name, configure: o => o.FreezeBaseModel = true);
        var tokens = Tokens();
        model.Predict(tokens);
        var encoderBefore = model.Layers[0].GetParameters().ToArray();
        var headBefore = model.Layers[model.Layers.Count - 1].GetParameters().ToArray();

        for (int step = 0; step < 3; step++)
            model.Train(tokens, OneHot());

        Assert.Equal(encoderBefore, model.Layers[0].GetParameters().ToArray());
        Assert.NotEqual(headBefore, model.Layers[model.Layers.Count - 1].GetParameters().ToArray());
    }

    [Theory]
    [MemberData(nameof(Models))]
    public void NumFineTuneLayers_FreezesTheLowerBlocks(string name)
    {
        var model = Create(name, configure: o => o.NumFineTuneLayers = 1);
        var tokens = Tokens();
        model.Predict(tokens);
        var blocks = Enumerable.Range(0, model.Layers.Count)
            .Where(i => model.Layers[i] is TransformerEncoderBlock<double> or PreLNTransformerBlock<double>)
            .ToList();
        Assert.Equal(2, blocks.Count);
        var lowerBefore = model.Layers[blocks[0]].GetParameters().ToArray();
        var upperBefore = model.Layers[blocks[1]].GetParameters().ToArray();

        for (int step = 0; step < 3; step++)
            model.Train(tokens, OneHot());

        Assert.Equal(lowerBefore, model.Layers[blocks[0]].GetParameters().ToArray());
        Assert.NotEqual(upperBefore, model.Layers[blocks[1]].GetParameters().ToArray());
    }

    [Theory]
    [InlineData("FinGPT")]
    [InlineData("FinMA")]
    [InlineData("InvestLM")]
    [InlineData("BloombergGPT")]
    public void Decoders_WithoutATokenizer_SayWhichOneTheyNeed(string name)
    {
        var model = Create(name);

        var error = Assert.Throws<InvalidOperationException>(() => model.Tokenize("revenue rose"));
        Assert.Contains("Tokenizer", error.Message);
    }

    [Theory]
    [InlineData("FinBERT")]
    [InlineData("FinancialBERT")]
    [InlineData("SECBERT")]
    [InlineData("FinBERTTone")]
    public void BertModels_TokenizeWithWordPiece(string name)
    {
        var model = Create(name, configure: o => Set(o, "VocabularySize", 30522));
        const string text = "Quarterly revenue rose sharply.";

        var ids = model.Tokenize(text);

        var reference = WordPieceTokenizer.BertBaseUncased().Encode(text, new AiDotNet.Tokenization.Models.EncodingOptions
        {
            AddSpecialTokens = true,
        });
        Assert.Equal(reference.TokenIds.ToArray(), ids);
        Assert.Equal(101, ids[0]);                     // [CLS]
        Assert.Equal(102, ids[ids.Length - 1]);        // [SEP]
        Assert.DoesNotContain(100, ids);               // [UNK]: every word is in the vocabulary
        Assert.Contains("revenue rose", model.Detokenize(ids));
    }

    [Fact]
    public void BertCheckpoint_LoadsIntoTheMatchingLayers()
    {
        string dir = Path.Combine(Path.GetTempPath(), "finbert-ckpt-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            var stored = WriteTinyBertCheckpoint(dir, numLayers: 2);
            var model = (FinBERT<double>)Create("FinBERT", configure: o =>
            {
                var bert = (FinancialBertEncoderOptions<double>)o;
                Set(bert, "VocabularySize", 30522);
                bert.UsePretrainedWeights = true;
                bert.PretrainedModelPath = dir;
                bert.Tokenizer = WordPieceTokenizer.BertBaseUncased();
            });

            var embedding = Assert.IsType<BertEmbeddingLayer<double>>(model.Layers[0]);
            AssertEqual(stored["bert.embeddings.word_embeddings.weight"], embedding.WordEmbedding.GetTrainableParameters()[0]);

            var block = Assert.IsType<TransformerEncoderBlock<double>>(model.Layers[2]);
            var attention = Assert.IsType<MultiHeadAttentionLayer<double>>(block.AttentionLayer);
            AssertTransposed(stored["bert.encoder.layer.1.attention.self.query.weight"], attention.GetQueryWeights());
            AssertEqual(stored["bert.encoder.layer.1.attention.self.value.bias"], attention.GetValueBias()
                ?? throw new InvalidOperationException("BERT's value projection must have a bias."));
            AssertTransposed(stored["bert.encoder.layer.1.output.dense.weight"], ((DenseLayer<double>)block.FfnDownLayer).GetWeights());

            var classifier = Assert.IsType<DenseLayer<double>>(model.Layers[model.Layers.Count - 1]);
            AssertTransposed(stored["classifier.weight"], classifier.GetWeights());

            var result = model.AnalyzeSentiment(new[] { "revenue rose" })[0];
            Assert.Equal(new[] { "neutral", "positive", "negative" }.OrderBy(s => s), result.ClassProbabilities.Keys.OrderBy(s => s));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void BertCheckpoint_WithTheWrongDepth_IsRefused()
    {
        string dir = Path.Combine(Path.GetTempPath(), "finbert-ckpt-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            WriteTinyBertCheckpoint(dir, numLayers: 1);
            var error = Assert.Throws<InvalidOperationException>(() => Create("FinBERT", configure: o =>
            {
                var bert = (FinancialBertEncoderOptions<double>)o;
                Set(bert, "VocabularySize", 30522);
                bert.UsePretrainedWeights = true;
                bert.PretrainedModelPath = dir;
            }));
            Assert.Contains("NumLayers", error.Message);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    private static void AssertEqual((int[] Shape, float[] Data) stored, Tensor<double> actual)
    {
        Assert.Equal(stored.Data.Length, actual.Length);
        for (int i = 0; i < stored.Data.Length; i++)
            Assert.Equal(stored.Data[i], actual[i], 6);
    }

    /// <summary>A stored <c>[out, in]</c> weight against the layer's <c>[in, out]</c> tensor.</summary>
    private static void AssertTransposed((int[] Shape, float[] Data) stored, Tensor<double> actual)
    {
        int rows = stored.Shape[0], cols = stored.Shape[1];
        Assert.Equal(new[] { cols, rows }, actual.Shape.ToArray());
        for (int o = 0; o < rows; o++)
            for (int i = 0; i < cols; i++)
                Assert.Equal(stored.Data[o * cols + i], actual[i, o], 6);
    }

    private static Dictionary<string, (int[] Shape, float[] Data)> WriteTinyBertCheckpoint(string dir, int numLayers)
    {
        int seed = 1;
        (int[], float[]) Tensor(params int[] shape)
        {
            int count = shape.Aggregate(1, (a, b) => a * b);
            var data = new float[count];
            for (int i = 0; i < count; i++)
            {
                seed = (seed * 1103515245 + 12345) & 0x7fffffff;
                data[i] = (seed % 2001 - 1000) / 10000f;
            }

            return (shape, data);
        }

        const int vocab = 30522;
        var tensors = new Dictionary<string, (int[] Shape, float[] Data)>
        {
            ["bert.embeddings.word_embeddings.weight"] = Tensor(vocab, Hidden),
            ["bert.embeddings.position_embeddings.weight"] = Tensor(Sequence, Hidden),
            ["bert.embeddings.token_type_embeddings.weight"] = Tensor(2, Hidden),
            ["bert.embeddings.LayerNorm.weight"] = Tensor(Hidden),
            ["bert.embeddings.LayerNorm.bias"] = Tensor(Hidden),
            ["bert.pooler.dense.weight"] = Tensor(Hidden, Hidden),
            ["bert.pooler.dense.bias"] = Tensor(Hidden),
            ["classifier.weight"] = Tensor(Classes, Hidden),
            ["classifier.bias"] = Tensor(Classes),
        };
        for (int l = 0; l < numLayers; l++)
        {
            string n = $"bert.encoder.layer.{l}";
            foreach (var proj in new[] { "attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense" })
            {
                tensors[$"{n}.{proj}.weight"] = Tensor(Hidden, Hidden);
                tensors[$"{n}.{proj}.bias"] = Tensor(Hidden);
            }

            tensors[$"{n}.attention.output.LayerNorm.weight"] = Tensor(Hidden);
            tensors[$"{n}.attention.output.LayerNorm.bias"] = Tensor(Hidden);
            tensors[$"{n}.intermediate.dense.weight"] = Tensor(16, Hidden);
            tensors[$"{n}.intermediate.dense.bias"] = Tensor(16);
            tensors[$"{n}.output.dense.weight"] = Tensor(Hidden, 16);
            tensors[$"{n}.output.dense.bias"] = Tensor(Hidden);
            tensors[$"{n}.output.LayerNorm.weight"] = Tensor(Hidden);
            tensors[$"{n}.output.LayerNorm.bias"] = Tensor(Hidden);
        }

        WriteSafetensors(Path.Combine(dir, "model.safetensors"), tensors);
        File.WriteAllText(Path.Combine(dir, "config.json"),
            "{\"model_type\":\"bert\",\"id2label\":{\"0\":\"Neutral\",\"1\":\"Positive\",\"2\":\"Negative\"}}");
        return tensors;
    }

    private static void WriteSafetensors(string path, Dictionary<string, (int[] Shape, float[] Data)> tensors)
    {
        var header = new JObject();
        long offset = 0;
        foreach (var kvp in tensors)
        {
            long bytes = kvp.Value.Data.Length * 4L;
            header[kvp.Key] = new JObject
            {
                ["dtype"] = "F32",
                ["shape"] = new JArray(kvp.Value.Shape.Select(d => (object)d).ToArray()),
                ["data_offsets"] = new JArray(offset, offset + bytes),
            };
            offset += bytes;
        }

        byte[] headerBytes = System.Text.Encoding.UTF8.GetBytes(header.ToString(Newtonsoft.Json.Formatting.None));
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);
        writer.Write((long)headerBytes.Length);
        writer.Write(headerBytes);
        foreach (var kvp in tensors)
            foreach (var value in kvp.Value.Data)
                writer.Write(value);
    }
}

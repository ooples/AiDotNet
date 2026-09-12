using System;
using System.IO;
using System.Linq;
using AiDotNet.Onnx;
using AiDotNet.Onnx.Protobuf;

namespace AiDotNet.Tests.Helpers;

/// <summary>Local, data-dependent ONNX graphs for shared vision-language boundary tests.</summary>
internal sealed class OnnxVisionLanguageFixture : IDisposable
{
    internal enum EncoderKind { Image, Video, Text, Audio }
    internal enum OutputKind
    {
        FixedEmbedding, TokenSequence, BatchedEmbedding, FirstTokenEmbedding,
        ContextTokenFeatures, BatchedTokenFeatures, EmptyTokenFeatures, SpatialTokenFeatures
    }
    internal enum TextInputKind { TokensAndMask, TokensOnly }
    internal enum QueryInputKind { ImageOnly, TextOnly, DefaultedBoth, RequiredBoth }

    private readonly string _directory = Path.Combine(Path.GetTempPath(), "aidotnet-onnx-contract-" + Guid.NewGuid().ToString("N"));
    private int _nextFile;

    internal OnnxVisionLanguageFixture() => Directory.CreateDirectory(_directory);

    internal string WriteEncoder(EncoderKind kind, int embedding = 4, int image = 16,
        int frames = 2, int context = 8, bool dynamicInputs = false,
        OutputKind outputKind = OutputKind.FixedEmbedding,
        TensorProto.Types.DataType tokenType = TensorProto.Types.DataType.Int64,
        int channels = 3, bool extraRequiredInput = false, bool auxiliarySequenceOutput = false,
        TextInputKind textInputKind = TextInputKind.TokensAndMask, int audioBins = 128)
    {
        var builder = new OnnxGraphBuilder(new OnnxExportOptions { OpsetVersion = 17 });
        string valueInput;
        string outputName;
        if (kind == EncoderKind.Text)
        {
            int[] shape = { 1, dynamicInputs ? -1 : context };
            builder.AddInput(TensorInfo("input_ids", tokenType, shape));
            AddFloatCast(builder, "input_ids", "float_ids");
            if (textInputKind == TextInputKind.TokensAndMask)
            {
                builder.AddInput(TensorInfo("attention_mask", TensorProto.Types.DataType.Int64, shape));
                AddFloatCast(builder, "attention_mask", "float_mask");
                builder.AddOp("Mul", new[] { "float_ids", "float_mask" }, new[] { "masked_ids" });
                valueInput = "masked_ids";
            }
            else
            {
                valueInput = "float_ids";
            }
            outputName = "text_embeds";
        }
        else if (kind == EncoderKind.Audio)
        {
            builder.AddInput(TensorInfo("input_values", TensorProto.Types.DataType.Float,
                new[] { 1, 1, audioBins, dynamicInputs ? -1 : frames }));
            valueInput = "input_values";
            outputName = "audio_embeds";
        }
        else
        {
            int spatial = dynamicInputs ? -1 : image;
            int[] shape = kind == EncoderKind.Video
                ? new[] { 1, dynamicInputs ? -1 : frames, channels, spatial, spatial }
                : new[] { 1, channels, spatial, spatial };
            builder.AddInput(TensorInfo("pixel_values", TensorProto.Types.DataType.Float, shape));
            valueInput = "pixel_values";
            outputName = kind == EncoderKind.Video ? "video_embeds" : "image_embeds";
        }

        if (extraRequiredInput)
            builder.AddInput(TensorInfo("unsupported_required_input", TensorProto.Types.DataType.Float, new[] { 1 }));

        if (outputKind == OutputKind.SpatialTokenFeatures)
        {
            if (kind != EncoderKind.Image) throw new ArgumentException("Spatial token features require an image encoder.", nameof(kind));
            var axes = new TensorProto { Name = "reduction_axes", DataType = (int)TensorProto.Types.DataType.Int64 };
            axes.Dims.Add(2);
            axes.Int64Data.Add(new long[] { 1, 2 });
            builder.AddInitializer(axes);
            var reduce = builder.AddOp("ReduceSum", new[] { valueInput, axes.Name }, new[] { "column_sums" });
            reduce.Attribute.Add(new AttributeProto { Name = "keepdims", Type = AttributeProto.Types.AttributeType.Int, I = 0 });
            var featureAxis = new TensorProto { Name = "feature_axis", DataType = (int)TensorProto.Types.DataType.Int64 };
            featureAxis.Dims.Add(1);
            featureAxis.Int64Data.Add(2);
            builder.AddInitializer(featureAxis);
            builder.AddOp("Unsqueeze", new[] { "column_sums", featureAxis.Name }, new[] { "column_features" });
            string offsets = builder.AddFloatInitializer("offsets", Enumerable.Range(1, embedding).Select(value => (float)value).ToArray(), new[] { 1, 1, embedding });
            builder.AddOp("Add", new[] { "column_features", offsets }, new[] { outputName });
            builder.AddOutput(TensorInfo(outputName, TensorProto.Types.DataType.Float, new[] { 1, dynamicInputs ? -1 : image, embedding }));
        }
        else if (outputKind == OutputKind.ContextTokenFeatures)
        {
            if (kind != EncoderKind.Text) throw new ArgumentException("Token features require a text encoder.", nameof(kind));
            var axes = new TensorProto { Name = "feature_axis", DataType = (int)TensorProto.Types.DataType.Int64 };
            axes.Dims.Add(1);
            axes.Int64Data.Add(1);
            builder.AddInitializer(axes);
            builder.AddOp("Unsqueeze", new[] { valueInput, axes.Name }, new[] { outputName });
            builder.AddOutput(TensorInfo(outputName, TensorProto.Types.DataType.Float,
                new[] { 1, 1, dynamicInputs ? -1 : context }));
        }
        else if (outputKind == OutputKind.TokenSequence)
        {
            if (kind != EncoderKind.Text) throw new ArgumentException("Token output requires a text encoder.", nameof(kind));
            builder.AddOp("Identity", new[] { valueInput }, new[] { outputName });
            builder.AddOutput(TensorInfo(outputName, TensorProto.Types.DataType.Float,
                new[] { 1, dynamicInputs ? -1 : context }));
        }
        else
        {
            var reduce = builder.AddOp("ReduceSum", new[] { valueInput }, new[] { "input_sum" });
            reduce.Attribute.Add(new AttributeProto
            {
                Name = "keepdims", Type = AttributeProto.Types.AttributeType.Int, I = 0
            });
            int batch = outputKind is OutputKind.BatchedEmbedding or OutputKind.BatchedTokenFeatures ? 2 : 1;
            bool tokenFeatures = outputKind is OutputKind.FirstTokenEmbedding or OutputKind.BatchedTokenFeatures or OutputKind.EmptyTokenFeatures;
            int tokenCount = outputKind == OutputKind.EmptyTokenFeatures ? 0 : 2;
            int[] outputShape = tokenFeatures ? new[] { batch, tokenCount, embedding } : new[] { batch, embedding };
            int repeats = tokenFeatures ? batch * tokenCount : batch;
            string offsets = builder.AddFloatInitializer("offsets",
                Enumerable.Range(1, embedding * repeats).Select(value => (float)value).ToArray(), outputShape);
            builder.AddOp("Add", new[] { "input_sum", offsets }, new[] { outputName });
            builder.AddOutput(TensorInfo(outputName, TensorProto.Types.DataType.Float, outputShape));
        }

        if (auxiliarySequenceOutput)
        {
            builder.AddOp("SequenceConstruct", new[] { outputName }, new[] { "auxiliary_sequence" });
            builder.AddOutput(new ValueInfoProto
            {
                Name = "auxiliary_sequence",
                Type = new TypeProto
                {
                    SequenceType = new TypeProto.Types.Sequence
                    {
                        ElemType = TensorInfo("element", TensorProto.Types.DataType.Float, new[] { 1, embedding }).Type
                    }
                }
            });
        }
        string path = Path.Combine(_directory, "encoder-" + _nextFile++ + ".onnx");
        using (var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write)) builder.WriteTo(stream);
        return path;
    }

    private static void AddFloatCast(OnnxGraphBuilder builder, string input, string output)
    {
        var cast = builder.AddOp("Cast", new[] { input }, new[] { output });
        cast.Attribute.Add(new AttributeProto
        {
            Name = "to", Type = AttributeProto.Types.AttributeType.Int, I = (long)TensorProto.Types.DataType.Float
        });
    }

    internal string WriteEmbeddedLanguageModel(int width = 4, bool extraRequiredInput = false)
    {
        var builder = new OnnxGraphBuilder(new OnnxExportOptions { OpsetVersion = 17 });
        builder.AddInput(TensorInfo("inputs_embeds", TensorProto.Types.DataType.Float, new[] { 1, -1, width }));
        if (extraRequiredInput)
            builder.AddInput(TensorInfo("unsupported_required_input", TensorProto.Types.DataType.Float, new[] { 1 }));
        builder.AddOp("Identity", new[] { "inputs_embeds" }, new[] { "last_hidden_state" });
        builder.AddOutput(TensorInfo("last_hidden_state", TensorProto.Types.DataType.Float, new[] { 1, -1, width }));
        string path = Path.Combine(_directory, "language-" + _nextFile++ + ".onnx");
        using (var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write)) builder.WriteTo(stream);
        return path;
    }

    internal string WriteQueryTransformer(QueryInputKind kind = QueryInputKind.ImageOnly,
        int width = 4, int visionWidth = 4, int queries = 2, int context = 8,
        bool dynamicInputs = false, bool preserveVisionTokens = false, bool extraRequiredInput = false,
        int batch = 1)
    {
        var builder = new OnnxGraphBuilder(new OnnxExportOptions { OpsetVersion = 17 });
        bool image = kind != QueryInputKind.TextOnly;
        bool text = kind != QueryInputKind.ImageOnly;
        bool defaults = kind == QueryInputKind.DefaultedBoth;
        if (image)
        {
            builder.AddInput(TensorInfo("encoder_hidden_states", TensorProto.Types.DataType.Float,
                new[] { 1, dynamicInputs ? -1 : 2, visionWidth }));
            if (defaults)
            {
                // AddFloatInitializer generates a new name; an overridable initializer
                // must instead use the exact graph input name.
                var initial = new TensorProto { Name = "encoder_hidden_states", DataType = (int)TensorProto.Types.DataType.Float };
                initial.Dims.Add(new long[] { 1, 2, visionWidth });
                initial.FloatData.Add(new float[2 * visionWidth]);
                builder.AddInitializer(initial);
            }
            var sum = builder.AddOp("ReduceSum", new[] { "encoder_hidden_states" }, new[] { "vision_sum" });
            sum.Attribute.Add(new AttributeProto { Name = "keepdims", Type = AttributeProto.Types.AttributeType.Int, I = 0 });
        }
        if (text)
        {
            foreach (string name in new[] { "input_ids", "attention_mask" })
            {
                builder.AddInput(TensorInfo(name, TensorProto.Types.DataType.Int64, new[] { 1, dynamicInputs ? -1 : context }));
                if (defaults)
                {
                    var initial = new TensorProto { Name = name, DataType = (int)TensorProto.Types.DataType.Int64 };
                    initial.Dims.Add(new long[] { 1, context });
                    initial.Int64Data.Add(new long[context]);
                    builder.AddInitializer(initial);
                }
                AddFloatCast(builder, name, "float_" + name);
            }
            builder.AddOp("Mul", new[] { "float_input_ids", "float_attention_mask" }, new[] { "masked_text" });
            var sum = builder.AddOp("ReduceSum", new[] { "masked_text" }, new[] { "text_sum" });
            sum.Attribute.Add(new AttributeProto { Name = "keepdims", Type = AttributeProto.Types.AttributeType.Int, I = 0 });
        }
        if (extraRequiredInput)
            builder.AddInput(TensorInfo("unsupported_required_input", TensorProto.Types.DataType.Float, new[] { 1 }));
        if (preserveVisionTokens)
        {
            if (!image) throw new ArgumentException("An identity vision-query fixture requires image input.", nameof(kind));
            builder.AddOp("Identity", new[] { "encoder_hidden_states" }, new[] { "query_features" });
            builder.AddOutput(TensorInfo("query_features", TensorProto.Types.DataType.Float,
                new[] { 1, dynamicInputs ? -1 : 2, visionWidth }));
        }
        else
        {
            string sumName = image ? "vision_sum" : "text_sum";
            if (image && text)
            {
                builder.AddOp("Add", new[] { "vision_sum", "text_sum" }, new[] { "joint_sum" });
                sumName = "joint_sum";
            }
            int[] shape = { batch, queries, width };
            string offsets = builder.AddFloatInitializer("offsets",
                Enumerable.Range(1, batch * queries * width).Select(value => (float)value).ToArray(), shape);
            builder.AddOp("Add", new[] { sumName, offsets }, new[] { "query_features" });
            builder.AddOutput(TensorInfo("query_features", TensorProto.Types.DataType.Float, shape));
        }
        string path = Path.Combine(_directory, "query-" + _nextFile++ + ".onnx");
        using (var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write)) builder.WriteTo(stream);
        return path;
    }

    private static ValueInfoProto TensorInfo(string name, TensorProto.Types.DataType type, int[] dimensions)
    {
        var shape = new TensorShapeProto();
        for (int index = 0; index < dimensions.Length; index++)
        {
            var dimension = new TensorShapeProto.Types.Dimension();
            if (dimensions[index] < 0) dimension.DimParam = "axis_" + index;
            else dimension.DimValue = dimensions[index];
            shape.Dim.Add(dimension);
        }
        return new ValueInfoProto
        {
            Name = name,
            Type = new TypeProto { TensorType = new TypeProto.Types.Tensor { ElemType = (int)type, Shape = shape } }
        };
    }

    public void Dispose() => Directory.Delete(_directory, recursive: true);
}

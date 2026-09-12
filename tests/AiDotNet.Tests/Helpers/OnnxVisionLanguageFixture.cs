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

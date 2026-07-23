using System.Collections.Generic;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Onnx;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ModelLoading.Pretrained;

/// <summary>
/// Adapts an <see cref="OnnxModel{T}"/> (an <see cref="IOnnxModel{T}"/> that runs an ONNX graph through the
/// ONNX runtime) to the <see cref="IFullModel{T, TInput, TOutput}"/> contract the facade and serving layer
/// speak, so <c>ConfigureModel(PretrainedSource.Onnx(path))</c> works like any other configured model.
/// </summary>
/// <typeparam name="T">The numeric type used for computation.</typeparam>
/// <remarks>
/// <para>
/// An ONNX model is a frozen inference graph: <see cref="Predict"/> runs it, but training and gradient-based
/// operations are not applicable and throw. Serialization round-trips the raw ONNX bytes, so the adapter
/// saves/loads/clones losslessly. <see cref="Predict"/> uses the single-input <see cref="IOnnxModel{T}.Run"/>
/// path (the graph's first input); multi-input graphs should be driven through <see cref="OnnxModel{T}"/>
/// directly.
/// </para>
/// <para><b>For Beginners:</b> This lets an <c>.onnx</c> file behave like any AiDotNet model — you can predict
/// with it and serve it, even though it was produced by another framework.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.General)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("ONNX: Open Neural Network Exchange", "https://onnx.ai/", Year = 2017, Authors = "ONNX Community")]
public sealed class OnnxFullModelAdapter<T> : IFullModel<T, Tensor<T>, Tensor<T>>
{
    private byte[] _onnxBytes;
    private readonly OnnxModelOptions? _options;
    private OnnxModel<T> _model;

    /// <summary>Creates an adapter over the ONNX graph in <paramref name="onnxBytes"/>.</summary>
    /// <param name="onnxBytes">The serialized ONNX model.</param>
    /// <param name="options">Optional runtime options (execution provider, etc.).</param>
    public OnnxFullModelAdapter(byte[] onnxBytes, OnnxModelOptions? options = null)
    {
        Guard.NotNull(onnxBytes);
        _onnxBytes = (byte[])onnxBytes.Clone();
        _options = options;
        _model = new OnnxModel<T>(_onnxBytes, options);
    }

    /// <summary>Creates an adapter over the ONNX graph at <paramref name="path"/>.</summary>
    /// <param name="path">Path to the <c>.onnx</c> file.</param>
    /// <param name="options">Optional runtime options.</param>
    public static OnnxFullModelAdapter<T> FromFile(string path, OnnxModelOptions? options = null)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("Path must be non-empty.", nameof(path));
        if (!File.Exists(path))
            throw new FileNotFoundException($"ONNX file not found: {path}", path);
        return new OnnxFullModelAdapter<T>(File.ReadAllBytes(path), options);
    }

    /// <summary>The wrapped ONNX runtime model.</summary>
    public OnnxModel<T> Model => _model;

    /// <inheritdoc/>
    public Tensor<T> Predict(Tensor<T> input) => _model.Run(input);

    /// <inheritdoc/>
    /// <remarks>ONNX graphs are frozen; training is not supported.</remarks>
    public void Train(Tensor<T> input, Tensor<T> expectedOutput) =>
        throw new NotSupportedException(
            "OnnxFullModelAdapter wraps a frozen ONNX inference graph; training is not supported. " +
            "Train the model in its source framework and re-export to ONNX.");

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata()
    {
        var meta = new ModelMetadata<T>
        {
            Name = string.IsNullOrEmpty(_model.Metadata.ModelName) ? "onnx-model" : _model.Metadata.ModelName,
            Description = _model.Metadata.Description ?? "ONNX inference graph",
        };
        meta.Properties["ExecutionProvider"] = _model.ExecutionProvider;
        meta.Properties["OpsetVersion"] = _model.Metadata.OpsetVersion;
        return meta;
    }

    /// <inheritdoc/>
    public ILossFunction<T> DefaultLossFunction => new MeanSquaredErrorLoss<T>();

    // ---- serialization: round-trip the raw ONNX bytes ----

    /// <inheritdoc/>
    public byte[] Serialize() => (byte[])_onnxBytes.Clone();

    /// <inheritdoc/>
    public void Deserialize(byte[] data) => Reload(data);

    /// <inheritdoc/>
    public void SaveModel(string filePath) => File.WriteAllBytes(filePath, _onnxBytes);

    /// <inheritdoc/>
    public void LoadModel(string filePath) => Reload(File.ReadAllBytes(filePath));

    /// <inheritdoc/>
    public void SaveState(Stream stream)
    {
        Guard.NotNull(stream);
        stream.Write(_onnxBytes, 0, _onnxBytes.Length);
    }

    /// <inheritdoc/>
    public void LoadState(Stream stream)
    {
        Guard.NotNull(stream);
        using var buffer = new MemoryStream();
        stream.CopyTo(buffer);
        Reload(buffer.ToArray());
    }

    private void Reload(byte[] data)
    {
        Guard.NotNull(data);
        var fresh = new OnnxModel<T>(data, _options);
        _model.Dispose();
        _model = fresh;
        // Cache the new graph bytes so subsequent Serialize/Save round-trips reflect it.
        _onnxBytes = (byte[])data.Clone();
    }

    // ---- feature importance: not meaningful for an opaque graph ----

    /// <inheritdoc/>
    public IEnumerable<int> GetActiveFeatureIndices() => Enumerable.Empty<int>();

    /// <inheritdoc/>
    public void SetActiveFeatureIndices(IEnumerable<int> featureIndices) =>
        throw new NotSupportedException("ONNX graphs do not expose selectable input features.");

    /// <inheritdoc/>
    public bool IsFeatureUsed(int featureIndex) => true;

    /// <inheritdoc/>
    public Dictionary<string, T> GetFeatureImportance() => new();

    // ---- cloning ----

    /// <inheritdoc/>
    public IFullModel<T, Tensor<T>, Tensor<T>> Clone() => new OnnxFullModelAdapter<T>(_onnxBytes, _options);

    /// <inheritdoc/>
    public IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => new OnnxFullModelAdapter<T>(_onnxBytes, _options);

    /// <inheritdoc/>
    public void Dispose() => _model.Dispose();
}

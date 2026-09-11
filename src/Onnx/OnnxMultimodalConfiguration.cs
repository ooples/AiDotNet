using System.Collections.ObjectModel;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Onnx;

/// <summary>Identifies the graph's role in a multimodal model.</summary>
public enum OnnxModelRole
{
    /// <summary>An image encoder.</summary>
    ImageEncoder,
    /// <summary>A multi-frame video encoder.</summary>
    VideoEncoder,
    /// <summary>A text encoder.</summary>
    TextEncoder,
    /// <summary>A text decoder.</summary>
    TextDecoder,
    /// <summary>A visual-query transformer.</summary>
    QueryTransformer,
    /// <summary>A language model.</summary>
    LanguageModel,
    /// <summary>An audio encoder.</summary>
    AudioEncoder
}

/// <summary>An immutable I/O value signature read from a loaded ONNX graph.</summary>
public sealed class OnnxValueSignature
{
    /// <summary>Gets the graph's tensor name.</summary>
    public string Name { get; }
    /// <summary>Gets the graph's value kind, including non-tensor auxiliary outputs.</summary>
    public OnnxValueType ValueType { get; }
    /// <summary>Gets whether this value is a tensor.</summary>
    public bool IsTensor { get; }
    /// <summary>Gets the tensor element type, or null for a non-tensor value.</summary>
    public TensorElementType? ElementType { get; }
    /// <summary>Gets fixed tensor dimensions; null means an unknown axis. Non-tensors have no axes.</summary>
    public IReadOnlyList<int?> Dimensions { get; }
    /// <summary>Gets the graph's symbolic axis names; empty strings mean unnamed axes.</summary>
    public IReadOnlyList<string> SymbolicDimensions { get; }
    /// <summary>Gets whether this input has an overridable graph initializer.</summary>
    public bool HasDefaultValue { get; }

    internal OnnxValueSignature(string name, NodeMetadata metadata, bool hasDefaultValue = false)
    {
        Name = name;
        ValueType = metadata.OnnxValueType;
        IsTensor = metadata.IsTensor;
        ElementType = IsTensor ? metadata.ElementDataType : null;
        Dimensions = Array.AsReadOnly(IsTensor
            ? metadata.Dimensions.Select(dimension => dimension < 0 ? (int?)null : dimension).ToArray()
            : Array.Empty<int?>());
        SymbolicDimensions = Array.AsReadOnly(IsTensor ? metadata.SymbolicDimensions.ToArray() : Array.Empty<string>());
        HasDefaultValue = hasDefaultValue;
    }
}

/// <summary>An immutable I/O signature for one graph, without guessed native architecture values.</summary>
public sealed class OnnxGraphSignature
{
    private readonly IReadOnlyList<string> _outputNames;

    /// <summary>Gets the graph's role in the model.</summary>
    public OnnxModelRole Role { get; }
    /// <summary>Gets input signatures, including overridable initializers.</summary>
    public IReadOnlyDictionary<string, OnnxValueSignature> Inputs { get; }
    /// <summary>Gets output signatures.</summary>
    public IReadOnlyDictionary<string, OnnxValueSignature> Outputs { get; }

    internal OnnxGraphSignature(OnnxModelRole role, InferenceSession session)
    {
        if (!Enum.IsDefined(typeof(OnnxModelRole), role)) throw new ArgumentOutOfRangeException(nameof(role));
        Role = role;
        var defaults = session.OverridableInitializerMetadata;
        var inputs = new Dictionary<string, OnnxValueSignature>(StringComparer.Ordinal);
        foreach (var input in session.InputMetadata)
            inputs.Add(input.Key, new OnnxValueSignature(input.Key, input.Value, defaults.ContainsKey(input.Key)));
        foreach (var input in defaults)
            if (!inputs.ContainsKey(input.Key)) inputs.Add(input.Key, new OnnxValueSignature(input.Key, input.Value, true));
        Inputs = new ReadOnlyDictionary<string, OnnxValueSignature>(inputs);
        Outputs = new ReadOnlyDictionary<string, OnnxValueSignature>(session.OutputMetadata.ToDictionary(
            output => output.Key, output => new OnnxValueSignature(output.Key, output.Value), StringComparer.Ordinal));
        _outputNames = Array.AsReadOnly(session.OutputNames.ToArray());
    }

    internal void RequireInputSet(params string[] suppliedInputs)
    {
        var supplied = new HashSet<string>(suppliedInputs, StringComparer.Ordinal);
        foreach (var input in Inputs.Values)
            if (!input.HasDefaultValue && !supplied.Contains(input.Name))
                throw Conflict($"requires input '{input.Name}', which this wrapper cannot supply");
        foreach (string input in supplied)
            if (!Inputs.ContainsKey(input)) throw Conflict($"does not accept the supplied input '{input}'");
    }

    internal void RequireInput(string name, TensorElementType elementType, params int[] dimensions)
    {
        if (!Inputs.TryGetValue(name, out var input)) throw Conflict($"does not declare input '{name}'");
        if (!input.IsTensor) throw Conflict($"input '{name}' is a {input.ValueType} value, but the wrapper supplies a tensor");
        if (input.ElementType != elementType)
            throw Conflict($"input '{name}' has element type {input.ElementType}, but the wrapper supplies {elementType}");
        if (input.Dimensions.Count != dimensions.Length)
            throw Conflict($"input '{name}' has rank {input.Dimensions.Count}, expected {dimensions.Length}");
        for (int axis = 0; axis < dimensions.Length; axis++)
            if (input.Dimensions[axis] is int actual && actual != dimensions[axis])
                throw Conflict($"input '{name}' axis {axis} is {actual}, but configured input requires {dimensions[axis]}");
    }

    internal string RequireEmbeddingOutput(int width, OnnxEmbeddingLayouts layouts, params string[] supportedNames)
    {
        string? name = _outputNames.FirstOrDefault(output => supportedNames.Length == 0
            || supportedNames.Contains(output, StringComparer.Ordinal));
        if (name is null) throw Conflict("has no output supported by this embedding wrapper");
        var output = Outputs[name];
        if (!output.IsTensor) throw Conflict($"output '{name}' is a {output.ValueType} value, not an embedding tensor");
        if (output.ElementType != TensorElementType.Float)
            throw Conflict($"output '{name}' must contain Float embeddings, not {output.ElementType}");
        string? mismatch = OnnxEmbeddingContract.FindMismatch(output.Dimensions, width, layouts);
        if (mismatch is not null) throw Conflict($"output '{name}' {mismatch}");
        return name;
    }

    private ArgumentException Conflict(string detail) => new($"ONNX {Role} {detail}.", "options");
}

/// <summary>The effective host dimensions and immutable signatures of loaded multimodal graphs.</summary>
/// <remarks>
/// <para>This is separate from the caller's requested options. Graph signatures do not infer
/// internal patch size, layer counts, attention heads or vocabulary tables from I/O shapes.</para>
/// <para>Symbolic dimensions remain unknown here; the wrapper validates actual outputs when
/// executing them. Tokenizer vocabulary size describes the supplied tokenizer, not a claim
/// that an opaque graph contains a matching embedding table.</para>
/// </remarks>
public sealed class OnnxMultimodalConfiguration
{
    /// <summary>Gets the host's validated embedding width.</summary>
    public int EmbeddingDimension { get; }
    /// <summary>Gets the host's configured text-token context.</summary>
    public int MaxSequenceLength { get; }
    /// <summary>Gets the host's configured square-image size.</summary>
    public int ImageSize { get; }
    /// <summary>Gets the host's image channel count.</summary>
    public int ImageChannels { get; }
    /// <summary>Gets the supplied tokenizer's vocabulary size, not an inferred graph vocabulary.</summary>
    public int TokenizerVocabularySize { get; }
    /// <summary>Gets the host's selected frame count, or null for a non-video wrapper.</summary>
    public int? NumFrames { get; }
    /// <summary>Gets graph signatures by role.</summary>
    public IReadOnlyDictionary<OnnxModelRole, OnnxGraphSignature> Graphs { get; }

    internal OnnxMultimodalConfiguration(int embeddingDimension, int maxSequenceLength, int imageSize,
        int tokenizerVocabularySize, int? numFrames, int imageChannels, params OnnxGraphSignature[] graphs)
    {
        EmbeddingDimension = embeddingDimension;
        MaxSequenceLength = maxSequenceLength;
        ImageSize = imageSize;
        ImageChannels = imageChannels;
        TokenizerVocabularySize = tokenizerVocabularySize;
        NumFrames = numFrames;
        Graphs = new ReadOnlyDictionary<OnnxModelRole, OnnxGraphSignature>(graphs.ToDictionary(graph => graph.Role));
    }
}

[Flags]
internal enum OnnxEmbeddingLayouts { Vector = 1, BatchedVector = 2, FirstToken = 4 }

internal static class OnnxEmbeddingContract
{
    internal static string? FindMismatch(IReadOnlyList<int?> dimensions, int width, OnnxEmbeddingLayouts layouts)
    {
        int rank = dimensions.Count;
        return FindMismatch(rank, rank > 1 ? dimensions[0] : null,
            rank == 3 ? dimensions[1] : null, rank > 0 ? dimensions[rank - 1] : null, width, layouts);
    }

    private static string? FindMismatch(int rank, int? batch, int? tokens, int? actualWidth,
        int width, OnnxEmbeddingLayouts layouts)
    {
        bool supported = rank switch
        {
            1 => (layouts & OnnxEmbeddingLayouts.Vector) != 0,
            2 => (layouts & OnnxEmbeddingLayouts.BatchedVector) != 0,
            3 => (layouts & OnnxEmbeddingLayouts.FirstToken) != 0,
            _ => false
        };
        if (!supported) return $"has unsupported embedding rank {rank}";
        if (rank > 1 && batch is not null && batch != 1)
            return $"has batch dimension {batch}; this wrapper requires batch 1";
        if (rank == 3 && tokens == 0) return "has no token from which to read an embedding";
        if (actualWidth is int actual && actual != width)
            return $"has embedding width {actual}, which conflicts with EmbeddingDimension {width}";
        return null;
    }

    internal static AiDotNet.Tensors.LinearAlgebra.Vector<T> Read<T>(Microsoft.ML.OnnxRuntime.Tensors.Tensor<float> output,
        int width, OnnxEmbeddingLayouts layouts, OnnxModelRole role)
    {
        // Dynamic graph axes are checked against the actual executed tensor, before allocating
        // or copying an embedding. No mismatched output is truncated or padded with zeros.
        var dimensions = output.Dimensions;
        int rank = dimensions.Length;
        string? mismatch = FindMismatch(rank, rank > 1 ? dimensions[0] : (int?)null,
            rank == 3 ? dimensions[1] : (int?)null, rank > 0 ? dimensions[rank - 1] : (int?)null, width, layouts);
        if (mismatch is not null) throw new InvalidOperationException($"ONNX {role} output {mismatch}.");
        var operations = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var embedding = new T[width];
        for (int index = 0; index < width; index++) embedding[index] = operations.FromDouble(output.GetValue(index));
        return new AiDotNet.Tensors.LinearAlgebra.Vector<T>(embedding);
    }
}

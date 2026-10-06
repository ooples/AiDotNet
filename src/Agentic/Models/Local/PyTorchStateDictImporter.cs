using System.Collections.Generic;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Agentic.Models.Local;

/// <summary>
/// Loads a PyTorch <c>state_dict</c> (saved as safetensors, or any <see cref="INamedTensorSource"/>) into an
/// AiDotNet network, converting each PyTorch module's tensors into the matching layer's own layout.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="WeightImporter"/> concatenates tensors that are already in AiDotNet's layout. A PyTorch checkpoint
/// is not: <c>nn.Linear.weight</c> is <c>[out, in]</c> where <see cref="DenseLayer{T}"/> stores <c>[in, out]</c>;
/// <c>nn.LSTM</c> packs its four gates as <c>i, f, g, o</c> rows of one matrix and carries two bias vectors where
/// <see cref="LSTMLayer{T}"/> keeps one tensor per gate and a single bias; <c>nn.MultiheadAttention</c> packs
/// Q/K/V into one <c>in_proj_weight</c>. This importer owns those conversions, one converter per layer type.
/// </para>
/// <para>
/// The caller pairs each parameter-bearing layer (in <see cref="NeuralNetworkBase{T}.Layers"/> order) with the
/// PyTorch module prefix it corresponds to, e.g. <c>["net.0", "net.2", "net.4"]</c> for an
/// <c>nn.Sequential</c> MLP. Layers without parameters (activations, pooling, flatten) take no prefix.
/// </para>
/// <para>
/// Nothing is skipped silently: a layer type without a converter, a missing tensor, a shape mismatch, or a
/// PyTorch parameter AiDotNet has no slot for (a non-zero attention <c>in_proj_bias</c>) throws.
/// </para>
/// <para><b>For Beginners:</b> PyTorch and AiDotNet store the same weights in different arrangements. Give this
/// importer a PyTorch checkpoint and the name of each PyTorch layer, and it rearranges the numbers so the
/// AiDotNet network computes exactly what the PyTorch one did.
/// </para>
/// </remarks>
public static class PyTorchStateDictImporter
{
    /// <summary>
    /// Imports the PyTorch tensors under each module prefix into the corresponding parameter-bearing layer.
    /// </summary>
    /// <typeparam name="T">The model's numeric type.</typeparam>
    /// <param name="model">The network to load weights into. Lazy layers must already be materialized (run one
    /// forward pass first), because their parameter shapes are not known before that.</param>
    /// <param name="source">The loaded PyTorch state_dict.</param>
    /// <param name="modulePrefixes">One PyTorch module prefix per parameter-bearing layer, in layer order.</param>
    /// <exception cref="ArgumentNullException">Thrown when any argument is <c>null</c>.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the layers and prefixes do not pair up, a
    /// layer is not materialized, or the model's flat parameter layout is not the per-layer concatenation this
    /// importer writes.</exception>
    /// <exception cref="NotSupportedException">Thrown for a layer type with no PyTorch converter.</exception>
    public static void Import<T>(NeuralNetworkBase<T> model, INamedTensorSource source, IReadOnlyList<string> modulePrefixes)
    {
        Guard.NotNull(model);
        Guard.NotNull(source);
        Guard.NotNull(modulePrefixes);

        var parameterLayers = new List<LayerBase<T>>();
        foreach (var layer in model.Layers)
        {
            // ParameterCount, not the layer's own trainable tensors: a composite block such as
            // TransformerEncoderBlock holds all of its weights in registered sublayers.
            if (layer is LayerBase<T> layerBase && layerBase.ParameterCount > 0)
                parameterLayers.Add(layerBase);
        }

        if (parameterLayers.Count != modulePrefixes.Count)
        {
            throw new InvalidOperationException(
                $"The model has {parameterLayers.Count} parameter-bearing layers but {modulePrefixes.Count} module " +
                "prefixes were given; pass exactly one PyTorch module prefix per parameter-bearing layer.");
        }

        var values = new Dictionary<Tensor<T>, double[]>(new TensorIdentityComparer<T>());
        for (int i = 0; i < parameterLayers.Count; i++)
            ConvertLayer(parameterLayers[i], modulePrefixes[i], source, values);

        var numOps = MathHelper.GetNumericOperations<T>();
        var current = model.GetParameters();
        var flat = new T[current.Length];
        int offset = 0;
        foreach (var layer in parameterLayers)
        {
            // The layer's own description of its flat slice (trainable tensors, buffers and sublayers in
            // FillParameters order), so composites are walked exactly as GetParameters writes them.
            var segments = new List<(Tensor<T>? Tensor, int Length, bool IsBuffer)>();
            layer.AppendFlatParameterLayout(segments);
            foreach (var (tensor, length, isBuffer) in segments)
            {
                if (offset + length > flat.Length)
                    throw LayoutMismatch(layer);

                if (tensor is null)
                {
                    if (!isBuffer && length > 0)
                    {
                        throw new NotSupportedException(
                            $"{layer.GetType().Name} has {length} parameters not backed by a Tensor<{typeof(T).Name}>; " +
                            "the PyTorch importer cannot address them.");
                    }
                    for (int k = 0; k < length; k++) flat[offset + k] = current[offset + k];
                    offset += length;
                    continue;
                }

                // Prove the layout on the current values before writing anything, so a layer whose
                // GetParameters orders differently fails here instead of loading weights into the wrong slots.
                var existing = tensor.ToVector();
                if (existing.Length != length)
                    throw LayoutMismatch(layer);
                for (int k = 0; k < length; k++)
                {
                    if (!numOps.Equals(existing[k], current[offset + k]))
                        throw LayoutMismatch(layer);
                }

                if (values.TryGetValue(tensor, out var converted))
                {
                    for (int k = 0; k < length; k++)
                        flat[offset + k] = numOps.FromDouble(converted[k]);
                }
                else if (isBuffer)
                {
                    // A buffer this layer's conversion does not set (a statistic with no PyTorch counterpart
                    // here) keeps its current value.
                    for (int k = 0; k < length; k++) flat[offset + k] = current[offset + k];
                }
                else
                {
                    throw new InvalidOperationException(
                        $"No PyTorch tensor was mapped to a [{ShapeText(tensor)}] parameter of {layer.GetType().Name}.");
                }
                offset += length;
            }
        }

        if (offset != flat.Length)
        {
            throw new InvalidOperationException(
                $"Mapped {offset} parameters but the model has {flat.Length}; a parameter-bearing layer was not covered.");
        }

        model.SetParameters(new Vector<T>(flat));
    }

    private static void ConvertLayer<T>(LayerBase<T> layer, string prefix, INamedTensorSource source,
        Dictionary<Tensor<T>, double[]> values)
    {
        switch (layer)
        {
            case DenseLayer<T>:
                ConvertLinear(layer, prefix, source, values);
                break;
            case ConvolutionalLayer<T>:
                ConvertWeightAndBias(layer, prefix, source, values);
                break;
            case LayerNormalizationLayer<T>:
                ConvertWeightAndBias(layer, prefix, source, values);
                break;
            case LSTMLayer<T>:
                ConvertLstm(layer, prefix, source, values);
                break;
            case MultiHeadAttentionLayer<T>:
                ConvertMultiHeadAttention(layer, prefix, source, values);
                break;
            case TransformerEncoderBlock<T> block:
                // nn.TransformerEncoderLayer submodule names.
                ConvertMultiHeadAttention(block.AttentionLayer, prefix + ".self_attn", source, values);
                ConvertWeightAndBias(block.Norm1Layer, prefix + ".norm1", source, values);
                ConvertLinear(block.FfnUpLayer, prefix + ".linear1", source, values);
                ConvertLinear(block.FfnDownLayer, prefix + ".linear2", source, values);
                ConvertWeightAndBias(block.Norm2Layer, prefix + ".norm2", source, values);
                break;
            default:
                throw new NotSupportedException(
                    $"{layer.GetType().Name} has no PyTorch state_dict converter (module '{prefix}').");
        }
    }

    // nn.Linear: weight [out, in] -> DenseLayer [in, out]; bias unchanged.
    private static void ConvertLinear<T>(LayerBase<T> layer, string prefix, INamedTensorSource source,
        Dictionary<Tensor<T>, double[]> values)
    {
        var parameters = layer.GetTrainableParameters();
        var weight = parameters[0];
        Put(values, weight, Transpose(Read(source, prefix + ".weight"), weight.Shape[1], weight.Shape[0]), layer, prefix);
        if (parameters.Count > 1)
            Put(values, parameters[1], Read(source, prefix + ".bias"), layer, prefix);
    }

    // Same layout on both sides: Conv2d weight [out, in, kh, kw]; LayerNorm weight (gamma) and bias (beta).
    private static void ConvertWeightAndBias<T>(LayerBase<T> layer, string prefix, INamedTensorSource source,
        Dictionary<Tensor<T>, double[]> values)
    {
        var parameters = layer.GetTrainableParameters();
        Put(values, parameters[0], Read(source, prefix + ".weight"), layer, prefix);
        if (parameters.Count > 1)
            Put(values, parameters[1], Read(source, prefix + ".bias"), layer, prefix);
    }

    // nn.LSTM (one layer): weight_ih_l0 [4H, in] and weight_hh_l0 [4H, H] stack the gates as i, f, g, o rows;
    // bias_ih_l0 + bias_hh_l0 is the gate bias. LSTMLayer registers Fi, Ii, Ci, Oi, Fh, Ih, Ch, Oh, bF, bI, bC, bO,
    // each W shaped [H, in] and applied as x·Wᵀ — the same orientation as PyTorch's rows.
    private static void ConvertLstm<T>(LayerBase<T> layer, string prefix, INamedTensorSource source,
        Dictionary<Tensor<T>, double[]> values)
    {
        var parameters = layer.GetTrainableParameters();
        if (parameters.Count != 12)
        {
            throw new NotSupportedException(
                $"LSTMLayer exposes {parameters.Count} parameters; the PyTorch converter expects 12 (8 weights, 4 biases).");
        }

        int[] pytorchGateOfSlot = [1, 0, 2, 3]; // AiDotNet F, I, C, O -> PyTorch block f, i, g, o
        var wih = Read(source, prefix + ".weight_ih_l0");
        var whh = Read(source, prefix + ".weight_hh_l0");
        var bih = Read(source, prefix + ".bias_ih_l0");
        var bhh = Read(source, prefix + ".bias_hh_l0");
        for (int slot = 0; slot < 4; slot++)
        {
            int gate = pytorchGateOfSlot[slot];
            Put(values, parameters[slot], RowBlock(wih, gate, parameters[slot].Length), layer, prefix);
            Put(values, parameters[4 + slot], RowBlock(whh, gate, parameters[4 + slot].Length), layer, prefix);
            var ih = RowBlock(bih, gate, parameters[8 + slot].Length);
            var hh = RowBlock(bhh, gate, parameters[8 + slot].Length);
            for (int k = 0; k < ih.Length; k++) ih[k] += hh[k];
            Put(values, parameters[8 + slot], ih, layer, prefix);
        }
    }

    // nn.MultiheadAttention: in_proj_weight [3E, E] stacks Q, K, V rows; out_proj is an nn.Linear.
    // MultiHeadAttentionLayer registers Wq, Wk, Wv, Wo (each [E, E], applied as x·W) and the output bias, so each
    // projection is the transpose of its PyTorch block. It has no Q/K/V bias: a non-zero in_proj_bias cannot be
    // represented and is refused rather than dropped.
    private static void ConvertMultiHeadAttention<T>(LayerBase<T> layer, string prefix, INamedTensorSource source,
        Dictionary<Tensor<T>, double[]> values)
    {
        var parameters = layer.GetTrainableParameters();
        if (parameters.Count != 5)
        {
            throw new NotSupportedException(
                $"MultiHeadAttentionLayer exposes {parameters.Count} parameters; the PyTorch converter expects 5 (Wq, Wk, Wv, Wo, output bias).");
        }

        int e = parameters[0].Shape[0];
        if (source.TensorNames.Contains(prefix + ".in_proj_bias"))
        {
            foreach (var b in Read(source, prefix + ".in_proj_bias"))
            {
                if (b != 0)
                {
                    throw new NotSupportedException(
                        $"'{prefix}.in_proj_bias' is non-zero, but MultiHeadAttentionLayer has no Q/K/V bias to load it into.");
                }
            }
        }

        var inProj = Read(source, prefix + ".in_proj_weight");
        for (int p = 0; p < 3; p++)
            Put(values, parameters[p], Transpose(RowBlock(inProj, p, e * e), e, e), layer, prefix);
        Put(values, parameters[3], Transpose(Read(source, prefix + ".out_proj.weight"), e, e), layer, prefix);
        Put(values, parameters[4], Read(source, prefix + ".out_proj.bias"), layer, prefix);
    }

    private static double[] Read(INamedTensorSource source, string name)
    {
        if (!source.TensorNames.Contains(name))
            throw new InvalidOperationException($"The state_dict has no tensor named '{name}'.");
        return source.ReadAsDouble(name);
    }

    private static void Put<T>(Dictionary<Tensor<T>, double[]> values, Tensor<T> target, double[] data,
        LayerBase<T> layer, string prefix)
    {
        if (data.Length != target.Length)
        {
            throw new InvalidOperationException(
                $"Module '{prefix}' supplies {data.Length} values for a [{ShapeText(target)}] parameter of {layer.GetType().Name}.");
        }

        values[target] = data;
    }

    // Row block `index` of a matrix whose rows stack equal-sized blocks of `blockLength` elements.
    private static double[] RowBlock(double[] data, int index, int blockLength)
    {
        if ((long)(index + 1) * blockLength > data.Length)
            throw new InvalidOperationException($"Packed tensor of {data.Length} values has no block {index} of {blockLength}.");
        var block = new double[blockLength];
        Array.Copy(data, (long)index * blockLength, block, 0, blockLength);
        return block;
    }

    // [rows, cols] row-major -> [cols, rows] row-major.
    private static double[] Transpose(double[] data, int rows, int cols)
    {
        if (data.Length != (long)rows * cols)
            throw new InvalidOperationException($"Cannot transpose {data.Length} values as [{rows}, {cols}].");
        var result = new double[data.Length];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result[c * rows + r] = data[r * cols + c];
        return result;
    }

    private static InvalidOperationException LayoutMismatch<T>(LayerBase<T> layer) =>
        new($"The model's flat parameter vector is not the per-layer concatenation of trainable tensors at " +
            $"{layer.GetType().Name}; the PyTorch importer cannot place its weights safely.");

    // Keys are the layer's own tensor objects; value equality would conflate distinct parameters.
    private sealed class TensorIdentityComparer<T> : IEqualityComparer<Tensor<T>>
    {
        public bool Equals(Tensor<T>? x, Tensor<T>? y) => ReferenceEquals(x, y);
        public int GetHashCode(Tensor<T> obj) => System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(obj);
    }

    private static string ShapeText<T>(Tensor<T> tensor)
    {
        var dims = new string[tensor.Shape.Length];
        for (int i = 0; i < dims.Length; i++) dims[i] = tensor.Shape[i].ToString();
        return string.Join(", ", dims);
    }
}

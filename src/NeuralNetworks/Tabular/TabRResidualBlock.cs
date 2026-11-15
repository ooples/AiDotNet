using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// The layers of one TabR residual block, the reference <c>make_block(prenorm)</c>:
/// <c>[Norm(d_main)] -&gt; Linear(d_main, d_block) -&gt; Activation -&gt; Dropout(dropout0) -&gt; Linear(d_block, d_main)
/// -&gt; Dropout(dropout1)</c>, added back onto its input.
/// </summary>
/// <remarks>
/// It holds the layers only. <see cref="TabRBase{T}"/> runs the block, because the two dropouts are its
/// training-step masks.
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal sealed class TabRResidualBlock<T>
{
    internal TabRResidualBlock(ILayer<T>? normalization, FullyConnectedLayer<T> expand, FullyConnectedLayer<T> project)
    {
        Normalization = normalization;
        Expand = expand;
        Project = project;
    }

    /// <summary>The pre-normalization, or null on the reference's first encoder block (<c>prenorm = False</c>).</summary>
    internal ILayer<T>? Normalization { get; }

    /// <summary><c>Activation(Linear(d_main, d_block))</c>.</summary>
    internal FullyConnectedLayer<T> Expand { get; }

    /// <summary><c>Linear(d_block, d_main)</c>, with a bias.</summary>
    internal FullyConnectedLayer<T> Project { get; }

    /// <summary>Every layer of the block, in forward order.</summary>
    internal IEnumerable<ILayer<T>> Layers
    {
        get
        {
            if (Normalization is not null) yield return Normalization;
            yield return Expand;
            yield return Project;
        }
    }
}

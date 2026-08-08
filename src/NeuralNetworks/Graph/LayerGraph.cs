using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Graph;

/// <summary>
/// One layer in a <see cref="LayerGraph{T}"/>, together with the nodes feeding it and any tensor
/// transformation applied on the way in.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public sealed class LayerNode<T>
{
    /// <summary>Position of this node in the graph's node list; also its identity.</summary>
    public int Id { get; }

    /// <summary>The layer this node runs. Null for a pure input node.</summary>
    public ILayer<T>? Layer { get; }

    /// <summary>Ids of the nodes feeding this one, in argument order.</summary>
    public IReadOnlyList<int> Inputs { get; }

    /// <summary>
    /// A transformation applied to the incoming tensor before the layer sees it — a reshape, permute or
    /// concatenation that is real dataflow but is NOT a layer.
    /// </summary>
    /// <remarks>
    /// Modelling these as EDGES rather than as pseudo-layers is what keeps them out of the flat
    /// <c>Layers</c> projection, so parameter counting, cloning and serialization are unaffected by
    /// something that owns no parameters. It is also what removes the false positive the layout pilot hit:
    /// a conv feeding a dense head across a reshape is correct, and the edge is where that reshape lives.
    /// </remarks>
    public Func<IReadOnlyList<Tensor<T>>, Tensor<T>>? EdgeTransform { get; }

    /// <summary>Human-readable description of <see cref="EdgeTransform"/>, for diagnostics.</summary>
    public string? EdgeDescription { get; }

    /// <summary>Creates a node.</summary>
    public LayerNode(
        int id,
        ILayer<T>? layer,
        IReadOnlyList<int> inputs,
        Func<IReadOnlyList<Tensor<T>>, Tensor<T>>? edgeTransform = null,
        string? edgeDescription = null)
    {
        Id = id;
        Layer = layer;
        Inputs = inputs ?? Array.Empty<int>();
        EdgeTransform = edgeTransform;
        EdgeDescription = edgeDescription;
    }

    /// <summary>True when this node reshapes or otherwise rewrites its input before the layer runs.</summary>
    public bool HasEdgeTransform => EdgeTransform is not null;

    /// <inheritdoc />
    public override string ToString()
    {
        string name = Layer?.GetType().Name ?? "Input";
        string ins = Inputs.Count == 0 ? "-" : string.Join(",", Inputs);
        return $"#{Id} {name}(from {ins}){(HasEdgeTransform ? " via " + (EdgeDescription ?? "transform") : "")}";
    }
}

/// <summary>
/// The wiring of a model, expressed once: which layer feeds which, and what happens on the edges between
/// them.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// WHY THIS EXISTS. The flat <c>Layers</c> list cannot express what these models actually do. ABCNet has
/// two detection heads reading one backbone, and a permute-plus-flatten between its last convolution and
/// its dense head. A positional walk of that list therefore reports mismatches that are not real, and — far
/// worse — the inherited sequential training path computes the WRONG GRAPH while raising nothing.
/// </para>
/// <para>
/// The graph is the single source of truth for wiring, so it cannot drift from the forward pass the way a
/// second description would. <see cref="ToLayerList"/> projects it back to the flat list, which is what
/// keeps parameter counting, cloning and serialization working untouched: they iterate nodes either way.
/// </para>
/// <para>
/// MODELS THAT DECLARE NOTHING KEEP WORKING. <see cref="Linear"/> builds the implicit chain that a flat
/// list has always meant, so the several hundred existing models need no migration and behave exactly as
/// before.
/// </para>
/// <para><b>For Beginners:</b> Most networks are a straight line — layer 1 feeds layer 2 feeds layer 3.
/// Some are not: one layer's output can feed two others, and results can be reshaped in between. This
/// records the real structure so the rest of the system stops assuming a straight line.</para>
/// </remarks>
public sealed class LayerGraph<T>
{
    private readonly List<LayerNode<T>> _nodes;

    /// <summary>The nodes, in the order they were added.</summary>
    public IReadOnlyList<LayerNode<T>> Nodes => _nodes;

    /// <summary>Id of the node whose output is the model's output — the FIRST, when there are several.</summary>
    public int OutputNodeId => OutputNodeIds[0];

    /// <summary>
    /// Ids of every node whose output is a model output, in declaration order.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Usually one. A decomposition emits several: Autoformer's encoder returns a (seasonal, trend)
    /// PAIR, which <see cref="IShapeContract.OutputsFor"/> can now describe declaratively. Until this
    /// existed the graph could not REPRESENT what the contract could state - a model with two outputs
    /// had to declare two whole graphs, which is what ABCNet does for its two entry points.
    /// </para>
    /// <para>
    /// Single-output graphs are unaffected: this is a one-element list and
    /// <see cref="OutputNodeId"/> reads it, so every existing caller behaves exactly as before.
    /// </para>
    /// </remarks>
    public IReadOnlyList<int> OutputNodeIds { get; }

    /// <summary>True when at least one node's output feeds more than one consumer.</summary>
    /// <remarks>
    /// The property that makes sequential execution incorrect, and the one the training path must check
    /// before assuming it can walk a list.
    /// </remarks>
    public bool HasFanOut { get; }

    /// <summary>True when every node has at most one consumer and no edge transforms — a plain chain.</summary>
    public bool IsLinear => !HasFanOut && _nodes.All(n => !n.HasEdgeTransform);

    internal LayerGraph(List<LayerNode<T>> nodes, int outputNodeId)
        : this(nodes, new[] { outputNodeId })
    {
    }

    internal LayerGraph(List<LayerNode<T>> nodes, IReadOnlyList<int> outputNodeIds)
    {
        if (outputNodeIds is null || outputNodeIds.Count == 0)
            throw new ArgumentException("A graph needs at least one output node.", nameof(outputNodeIds));

        _nodes = nodes;
        OutputNodeIds = outputNodeIds;

        var consumerCount = new int[nodes.Count];
        foreach (var n in nodes)
            foreach (int i in n.Inputs)
                if (i >= 0 && i < consumerCount.Length) consumerCount[i]++;

        HasFanOut = consumerCount.Any(c => c > 1);
    }

    /// <summary>
    /// Builds the implicit linear graph a flat layer list has always meant: each layer consumes the one
    /// before it.
    /// </summary>
    /// <remarks>
    /// This must stay behaviourally identical to the historical <c>foreach (var layer in Layers)</c> loop.
    /// Every model that has not migrated depends on it, so a difference here is a regression across the
    /// whole library rather than in one model.
    /// </remarks>
    public static LayerGraph<T> Linear(IReadOnlyList<ILayer<T>> layers)
    {
        if (layers is null) throw new ArgumentNullException(nameof(layers));

        var nodes = new List<LayerNode<T>>(layers.Count);
        for (int i = 0; i < layers.Count; i++)
            nodes.Add(new LayerNode<T>(i, layers[i], i == 0 ? Array.Empty<int>() : new[] { i - 1 }));

        return new LayerGraph<T>(nodes, layers.Count - 1);
    }

    /// <summary>
    /// The layers in node order — the projection the rest of the system consumes.
    /// </summary>
    /// <remarks>
    /// Deliberately a projection rather than a parallel list. Anything that counts parameters, clones or
    /// serializes keeps working without knowing the graph exists, because a layer is a layer regardless of
    /// how it is wired.
    /// </remarks>
    public List<ILayer<T>> ToLayerList()
    {
        var result = new List<ILayer<T>>();
        foreach (var n in _nodes) if (n.Layer is not null) result.Add(n.Layer);
        return result;
    }

    /// <summary>
    /// Executes the graph, honouring branches and edge transforms.
    /// </summary>
    /// <param name="input">The model input, delivered to every node with no predecessors.</param>
    /// <returns>The output node's tensor.</returns>
    /// <remarks>
    /// Nodes are executed in id order, which is a valid topological order by construction: the builder
    /// refuses an edge to a node that does not yet exist, so a node's inputs always precede it.
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input) => ForwardAll(input)[0];

    /// <summary>
    /// Executes the graph and returns EVERY output, in <see cref="OutputNodeIds"/> order.
    /// </summary>
    /// <param name="input">The model input, delivered to every node with no predecessors.</param>
    /// <returns>One tensor per declared output node.</returns>
    /// <remarks>
    /// <see cref="Forward"/> is this taking the first, which is the whole of it for the single-output
    /// graphs that are the norm. A decomposition needs both halves — Autoformer's encoder returns a
    /// (seasonal, trend) pair — and returning only the first would silently discard one.
    /// </remarks>
    public IReadOnlyList<Tensor<T>> ForwardAll(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));

        var values = new Tensor<T>[_nodes.Count];
        for (int i = 0; i < _nodes.Count; i++)
        {
            var node = _nodes[i];

            var incoming = new List<Tensor<T>>(node.Inputs.Count);
            if (node.Inputs.Count == 0) incoming.Add(input);
            else foreach (int src in node.Inputs) incoming.Add(values[src]);

            var fed = node.EdgeTransform is not null
                ? node.EdgeTransform(incoming)
                : incoming[0];

            values[i] = node.Layer is null ? fed : node.Layer.Forward(fed);
        }

        var outputs = new Tensor<T>[OutputNodeIds.Count];
        for (int i = 0; i < OutputNodeIds.Count; i++) outputs[i] = values[OutputNodeIds[i]];
        return outputs;
    }

    /// <summary>
    /// Resolves every lazy layer in the graph from its REAL predecessor, without allocating weights.
    /// </summary>
    /// <param name="inputShape">Shape observed by the nodes that have no predecessors.</param>
    /// <returns>The output node's resolved shape, or <c>null</c> if the walk could not reach it.</returns>
    /// <remarks>
    /// <para>
    /// THIS IS WHY THE GRAPH HAS TO EXIST. The default resolver walks <c>Layers</c> in list order and
    /// hands each layer the previous one's output shape. For a branched model that is simply the wrong
    /// dataflow: two sibling heads reading the same trunk are not predecessor and successor, so the second
    /// head gets pinned to the first head's output width. It fails LOUDLY at the first forward with a
    /// baffling message ("expected input depth 1, got 256") that names neither layer nor the assumption
    /// that produced it — and it fails equally hard whether the head widths happen to collide or not, so
    /// there is no configuration in which a branched model resolves correctly by luck.
    /// </para>
    /// <para>
    /// Edge transforms are applied to ZERO-FILLED placeholders rather than to a declared shape function.
    /// That keeps one definition of what an edge does instead of two that can disagree — a separately
    /// declared shape rule is exactly the kind of duplicate that drifts from the tensor rule it is
    /// supposed to mirror. The placeholders cost one small allocation per edge, once; transforms are pure
    /// rearrangement (concat / permute / reshape) and touch no layer, so nothing allocates weights and no
    /// RNG is consumed.
    /// </para>
    /// <para>
    /// Spatial axes are relaxed afterwards for the same reason the sequential walk relaxes them: the
    /// shape this resolution starts from is the architecture's DECLARED input, and a caller may hand the
    /// model a different height and width. Channel counts are structural and survive; height and width
    /// are not the layer's to claim.
    /// </para>
    /// </remarks>
    public int[]? ResolveShapes(int[] inputShape)
    {
        if (inputShape is null) throw new ArgumentNullException(nameof(inputShape));

        var shapes = new int[_nodes.Count][];

        for (int i = 0; i < _nodes.Count; i++)
        {
            var node = _nodes[i];

            var incoming = new List<int[]>(Math.Max(1, node.Inputs.Count));
            if (node.Inputs.Count == 0)
            {
                incoming.Add(inputShape);
            }
            else
            {
                foreach (int src in node.Inputs)
                {
                    // A predecessor that could not be resolved makes every shape downstream of it a
                    // guess. Stop rather than propagate one: a layer pinned to a guessed width is worse
                    // than a layer left lazy, because the lazy one still resolves correctly on the first
                    // real forward.
                    if (shapes[src] is null) return null;
                    incoming.Add(shapes[src]);
                }
            }

            int[] fed;
            if (node.EdgeTransform is not null)
            {
                var probes = new List<Tensor<T>>(incoming.Count);
                foreach (var s in incoming) probes.Add(new Tensor<T>(s));
                try
                {
                    fed = node.EdgeTransform(probes).Shape.ToArray();
                }
                catch
                {
                    return null;
                }
            }
            else
            {
                fed = incoming[0];
            }

            if (node.Layer is null)
            {
                shapes[i] = fed;
                continue;
            }

            try
            {
                if (node.Layer is LayerBase<T> lazyLayer && !lazyLayer.IsShapeResolved)
                {
                    lazyLayer.ResolveShapesOnly(fed);
                }

                var outShape = node.Layer.GetOutputShape();
                if (outShape is null || outShape.Length == 0 || !Array.TrueForAll(outShape, d => d > 0))
                {
                    return null;
                }

                shapes[i] = outShape;

                if (node.Layer is LayerBase<T> relaxable) relaxable.RelaxPropagatedSpatialAxes();
            }
            catch
            {
                return null;
            }
        }

        _lastResolvedShapes = shapes;
        return shapes[OutputNodeId];
    }

    /// <summary>Per-node shapes from the most recent <see cref="ResolveShapes"/>, or null.</summary>
    private int[]?[]? _lastResolvedShapes;

    /// <summary>
    /// Resolves the graph and returns EVERY output shape, in <see cref="OutputNodeIds"/> order.
    /// </summary>
    /// <param name="inputShape">The shape delivered to nodes with no predecessors.</param>
    /// <returns>One shape per declared output node, or <c>null</c> if any of them failed to resolve.</returns>
    /// <remarks>
    /// The multi-output counterpart of <see cref="ResolveShapes"/>, which returns only the first.
    /// Declining as a whole when ANY output fails is deliberate and matches the per-node rule already
    /// used here: half a set of output shapes is not a shape for the model.
    /// </remarks>
    public IReadOnlyList<int[]>? ResolveAllShapes(int[] inputShape)
    {
        if (ResolveShapes(inputShape) is null) return null;

        var shapes = _lastResolvedShapes;
        if (shapes is null) return null;

        var outputs = new int[OutputNodeIds.Count][];
        for (int i = 0; i < OutputNodeIds.Count; i++)
        {
            var shape = shapes[OutputNodeIds[i]];
            if (shape is null || shape.Length == 0) return null;
            outputs[i] = shape;
        }

        return outputs;
    }

    /// <summary>
    /// The maximal runs of nodes that are genuinely sequential — single predecessor, single consumer, and
    /// no edge transform between them.
    /// </summary>
    /// <remarks>
    /// This is what layout validation should walk, and it is the direct fix for the pilot's false
    /// positive. A run breaks at a branch and at an edge transform, because in both cases the next layer
    /// does not receive the previous layer's output unchanged, so comparing their declared layouts would
    /// be comparing things that never meet.
    /// </remarks>
    public IReadOnlyList<IReadOnlyList<ILayer<T>>> ContiguousRuns()
    {
        var consumerCount = new int[_nodes.Count];
        foreach (var n in _nodes)
            foreach (int i in n.Inputs)
                if (i >= 0 && i < consumerCount.Length) consumerCount[i]++;

        var runs = new List<IReadOnlyList<ILayer<T>>>();
        var current = new List<ILayer<T>>();

        for (int i = 0; i < _nodes.Count; i++)
        {
            var node = _nodes[i];
            bool continues =
                current.Count > 0 &&
                node.Inputs.Count == 1 &&
                node.Inputs[0] == i - 1 &&
                consumerCount[i - 1] == 1 &&
                !node.HasEdgeTransform;

            if (!continues)
            {
                if (current.Count > 1) runs.Add(current);
                current = new List<ILayer<T>>();
            }

            if (node.Layer is not null) current.Add(node.Layer);
        }

        if (current.Count > 1) runs.Add(current);
        return runs;
    }

    /// <inheritdoc />
    public override string ToString()
        => string.Join("; ", _nodes.Select(n => n.ToString()));
}

/// <summary>
/// Fluent construction of a <see cref="LayerGraph{T}"/>: add each layer and say what feeds it.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <example>
/// <code>
/// var g = new LayerGraphBuilder&lt;double&gt;();
/// int stem = g.Add(conv1);                       // consumes the model input
/// int trunk = g.Add(conv2, stem);
/// int headA = g.Add(scoreHead, trunk);           // both heads read the trunk:
/// int headB = g.Add(coordHead, trunk);           // this is the fan-out
/// g.Output(headA);
/// </code>
/// </example>
public sealed class LayerGraphBuilder<T>
{
    private readonly List<LayerNode<T>> _nodes = new();
    private int _outputId = -1;

    /// <summary>Adds a layer fed by the model input.</summary>
    public int Add(ILayer<T> layer) => AddNode(layer, Array.Empty<int>(), null, null);

    /// <summary>Adds a layer fed by one earlier node.</summary>
    public int Add(ILayer<T> layer, int input) => AddNode(layer, new[] { input }, null, null);

    /// <summary>
    /// Adds a layer fed by one earlier node through a transformation — a reshape, permute or similar.
    /// </summary>
    /// <param name="layer">The layer to run.</param>
    /// <param name="input">The node feeding it.</param>
    /// <param name="transform">The tensor rewrite applied before the layer sees the value.</param>
    /// <param name="description">Short description, surfaced in diagnostics.</param>
    public int AddVia(
        ILayer<T> layer, int input, Func<Tensor<T>, Tensor<T>> transform, string description)
    {
        if (transform is null) throw new ArgumentNullException(nameof(transform));
        return AddNode(layer, new[] { input }, values => transform(values[0]), description);
    }

    /// <summary>
    /// Adds a layer fed by SEVERAL earlier nodes, combined by <paramref name="combine"/>.
    /// </summary>
    public int AddJoin(
        ILayer<T>? layer,
        IReadOnlyList<int> inputs,
        Func<IReadOnlyList<Tensor<T>>, Tensor<T>> combine,
        string description)
    {
        if (inputs is null || inputs.Count == 0)
            throw new ArgumentException("A join needs at least one input.", nameof(inputs));
        if (combine is null) throw new ArgumentNullException(nameof(combine));
        return AddNode(layer, inputs, combine, description);
    }

    private int AddNode(
        ILayer<T>? layer,
        IReadOnlyList<int> inputs,
        Func<IReadOnlyList<Tensor<T>>, Tensor<T>>? transform,
        string? description)
    {
        foreach (int i in inputs)
        {
            // Refusing a forward reference is what makes id order a valid topological order, so execution
            // needs no separate sort and a cycle cannot be expressed at all.
            if (i < 0 || i >= _nodes.Count)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(inputs), i,
                    $"Node {i} does not exist yet. A layer can only be fed by nodes added before it, which "
                    + "is what keeps the graph acyclic by construction.");
            }
        }

        int id = _nodes.Count;
        _nodes.Add(new LayerNode<T>(id, layer, inputs, transform, description));
        _outputId = id;
        return id;
    }

    /// <summary>Declares which node's output is the model's output. Defaults to the last node added.</summary>
    /// <remarks>
    /// Replaces any previously declared outputs. To declare several, use <see cref="Outputs"/>.
    /// </remarks>
    public LayerGraphBuilder<T> Output(int nodeId)
    {
        if (nodeId < 0 || nodeId >= _nodes.Count)
            throw new ArgumentOutOfRangeException(nameof(nodeId), nodeId, "No such node.");
        _outputId = nodeId;
        _extraOutputIds = null;
        return this;
    }

    /// <summary>
    /// Declares SEVERAL output nodes, in the order the model returns them.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A decomposition emits more than one tensor - Autoformer's encoder returns a (seasonal, trend)
    /// pair - and until the graph could hold several outputs, such a model had to declare one graph per
    /// output or pack the pair into a single fictitious width. The declarative side already describes
    /// this via <see cref="IShapeContract.OutputsFor"/>; this is the operational half.
    /// </para>
    /// <para>
    /// Order is the model's return order and is preserved through <see cref="LayerGraph{T}.ForwardAll"/>
    /// and <see cref="LayerGraph{T}.ResolveAllShapes"/>.
    /// </para>
    /// </remarks>
    public LayerGraphBuilder<T> Outputs(params int[] nodeIds)
    {
        if (nodeIds is null || nodeIds.Length == 0)
            throw new ArgumentException("A graph needs at least one output node.", nameof(nodeIds));

        foreach (int id in nodeIds)
        {
            if (id < 0 || id >= _nodes.Count)
                throw new ArgumentOutOfRangeException(nameof(nodeIds), id, "No such node.");
        }

        _outputId = nodeIds[0];
        _extraOutputIds = nodeIds.Length > 1 ? (int[])nodeIds.Clone() : null;
        return this;
    }

    /// <summary>Additional output ids when more than one was declared; null for the usual single case.</summary>
    private int[]? _extraOutputIds;

    /// <summary>Builds the graph.</summary>
    public LayerGraph<T> Build()
    {
        if (_nodes.Count == 0) throw new InvalidOperationException("A graph needs at least one node.");
        return _extraOutputIds is null
            ? new LayerGraph<T>(_nodes, _outputId)
            : new LayerGraph<T>(_nodes, _extraOutputIds);
    }
}

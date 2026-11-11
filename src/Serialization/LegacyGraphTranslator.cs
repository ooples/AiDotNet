using AiDotNet.Tensors.Engines.Export;

namespace AiDotNet.Serialization;

/// <summary>
/// Reads the graphs the removed <c>GraphTrace</c> recorded for traceable expressions (the <c>graph:</c> scheme of
/// <see cref="DelegateState"/>) and replays them through <see cref="IEngine"/> ops.
/// </summary>
/// <remarks>
/// <para>
/// The legacy format is <c>id=Op(parentIds){Key=value,...};...;outputId</c>, written in post-order, where <c>Op</c> is
/// the <see cref="OperationType"/> the legacy <c>TensorOperations&lt;T&gt;</c> tagged the node with and the parameters
/// are its <c>OperationParams</c>. Values are <c>~</c> (null), <c>true</c>/<c>false</c>, invariant numbers,
/// <c>'string</c> and <c>[1|2|3]</c> int arrays. Every node derives from the single input: the recorder refused
/// captured constants, so a graph never names a tensor that is not computed from the input.
/// </para>
/// <para>
/// Each operation maps to the engine op (or a short composition of engine ops) that computes what the legacy forward
/// computed. Nothing is resolved by reflection: an operation the table does not list is rejected, so a saved string
/// can only ever invoke the ops written here.
/// </para>
/// </remarks>
internal static class LegacyGraphTranslator
{
    private const string Scheme = "graph:";
    private const string InputOp = "Input";

    /// <summary>
    /// Parses and validates <paramref name="graph"/> and returns a delegate that replays it, on whatever input it gets,
    /// through the current engine.
    /// </summary>
    /// <exception cref="InvalidOperationException">The graph is malformed, names an unknown operation, or has bad parameters.</exception>
    internal static Func<Tensor<T>, Tensor<T>> Compile<T>(string graph, string layerName, string key)
    {
        var parsed = Parse(graph, layerName, key);
        return input => Run<T>(parsed, input, layerName, key);
    }

    /// <summary>
    /// Translates <paramref name="graph"/> to a portable <see cref="TensorProgram"/> by tracing the replay on a probe of
    /// <paramref name="probeInputShape"/> (non-positive dimensions become 1).
    /// </summary>
    /// <remarks>
    /// The trace verifies the program against the replay on the probe, except for a graph with a stochastic step
    /// (GumbelSoftmax draws fresh noise on every call, as the legacy op did), where two runs never agree.
    /// </remarks>
    /// <exception cref="InvalidOperationException">The graph is invalid or cannot be traced on the probe.</exception>
    internal static TensorProgram Translate<T>(string graph, int[] probeInputShape, string layerName, string key)
    {
        var parsed = Parse(graph, layerName, key);
        Func<Tensor<T>, Tensor<T>> replay = input => Run<T>(parsed, input, layerName, key);
        var verify = !parsed.Steps.Any(s => s.Op == "GumbelSoftmax");

        var shape = (probeInputShape is null || probeInputShape.Length == 0 ? new[] { 1 } : probeInputShape)
            .Select(d => d > 0 ? d : 1).ToArray();
        var probe = new Tensor<T>(shape);
        var numOps = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < probe.Length; i++)
        {
            // Deterministic values in (0, 1): positive for Log/Sqrt, distinct enough that a verify catches a mismatch.
            probe.SetFlat(i, numOps.FromDouble(0.05 + 0.9 * ((i * 7919) % 101) / 101.0));
        }

        try
        {
            return TensorProgram.Trace<T>(xs => new[] { replay(xs[0]) }, new[] { probe }, new TraceOptions { Verify = verify });
        }
        catch (Exception ex) when (!IsOwn(ex))
        {
            throw Bad(layerName, key, $"it could not be translated to a portable program: {ex.Message}", ex);
        }
    }

    // ── Parsing ─────────────────────────────────────────────────────────────────────────────────────────────

    private sealed class Step
    {
        internal Step(int id, string op, int[] inputs, Dictionary<string, string> parameters, string text)
        {
            Id = id;
            Op = op;
            Inputs = inputs;
            Parameters = parameters;
            Text = text;
        }

        internal int Id { get; }
        internal string Op { get; }
        internal int[] Inputs { get; }
        internal Dictionary<string, string> Parameters { get; }
        internal string Text { get; }
    }

    private sealed class ParsedGraph
    {
        internal ParsedGraph(int inputId, List<Step> steps, int outputId)
        {
            InputId = inputId;
            Steps = steps;
            OutputId = outputId;
        }

        internal int InputId { get; }
        internal List<Step> Steps { get; }
        internal int OutputId { get; }
    }

    private static ParsedGraph Parse(string graph, string layerName, string key)
    {
        if (graph is null) throw Bad(layerName, key, "the saved graph is null.");

        var text = graph.StartsWith(Scheme, StringComparison.Ordinal) ? graph.Substring(Scheme.Length) : graph;
        var parts = text.Split(';');

        // The last part is the output id. A graph written before the separator fix ends its last step with a stray
        // '>' ("2=Square(1)>;2"), and one variant wrote it as its own part (";>;2"); both are the same graph.
        var bodies = new List<string>();
        for (int i = 0; i < parts.Length - 1; i++)
        {
            if (parts[i] == ">") continue;
            bodies.Add(parts[i]);
        }

        if (parts.Length < 2 || !TryParseId(parts[parts.Length - 1], out var outputId))
            throw Bad(layerName, key, "its recorded graph has no output node.");
        if (bodies.Count > 0 && bodies[bodies.Count - 1].EndsWith(">", StringComparison.Ordinal))
            bodies[bodies.Count - 1] = bodies[bodies.Count - 1].Substring(0, bodies[bodies.Count - 1].Length - 1);

        var defined = new HashSet<int>();
        var steps = new List<Step>();
        int inputId = -1;

        foreach (var body in bodies)
        {
            var eq = body.IndexOf('=');
            if (eq <= 0 || !TryParseId(body.Substring(0, eq), out var id))
                throw Bad(layerName, key, $"a recorded step '{body}' is malformed.");
            if (!defined.Add(id))
                throw Bad(layerName, key, $"node {id} is recorded twice.");

            var rhs = body.Substring(eq + 1);
            if (rhs == InputOp)
            {
                if (inputId >= 0) throw Bad(layerName, key, "its recorded graph has more than one input.");
                inputId = id;
                continue;
            }

            var step = ParseStep(id, rhs, body, defined, layerName, key);
            Validate(step, layerName, key);
            steps.Add(step);
        }

        if (inputId < 0) throw Bad(layerName, key, "its recorded graph has no input node.");
        if (!defined.Contains(outputId)) throw Bad(layerName, key, $"its recorded output node {outputId} is never produced.");

        return new ParsedGraph(inputId, steps, outputId);
    }

    private static Step ParseStep(int id, string rhs, string body, HashSet<int> defined, string layerName, string key)
    {
        var open = rhs.IndexOf('(');
        var close = rhs.IndexOf(')');
        if (open <= 0 || close < open) throw Bad(layerName, key, $"a recorded step '{body}' is malformed.");

        var op = rhs.Substring(0, open);
        if (!op.All(c => char.IsLetterOrDigit(c) || c == '_'))
            throw Bad(layerName, key, $"a recorded step '{body}' is malformed.");

        var argText = rhs.Substring(open + 1, close - open - 1);
        var inputs = new List<int>();
        if (argText.Length > 0)
        {
            foreach (var arg in argText.Split(','))
            {
                if (!TryParseId(arg, out var parent))
                    throw Bad(layerName, key, $"a recorded step '{body}' is malformed.");
                // Post-order: a node's inputs are always written before it.
                if (!defined.Contains(parent) || parent == id)
                    throw Bad(layerName, key, $"step '{body}' uses node {parent} before it is produced.");
                inputs.Add(parent);
            }
        }

        var tail = rhs.Substring(close + 1);
        var parameters = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        if (tail.Length > 0)
        {
            if (tail[0] != '{' || tail[tail.Length - 1] != '}')
                throw Bad(layerName, key, $"a recorded step '{body}' is malformed.");
            var inner = tail.Substring(1, tail.Length - 2);
            if (inner.Length > 0)
            {
                foreach (var pair in inner.Split(','))
                {
                    var peq = pair.IndexOf('=');
                    if (peq <= 0) throw Bad(layerName, key, $"a recorded parameter '{pair}' in step '{body}' is malformed.");
                    var name = pair.Substring(0, peq);
                    if (parameters.ContainsKey(name))
                        throw Bad(layerName, key, $"step '{body}' records parameter '{name}' twice.");
                    parameters[name] = pair.Substring(peq + 1);
                }
            }
        }

        return new Step(id, op, inputs.ToArray(), parameters, body);
    }

    private static bool TryParseId(string text, out int id)
        => int.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out id);

    // ── Operation table ─────────────────────────────────────────────────────────────────────────────────────

    /// <summary>The kind of a recorded parameter value.</summary>
    private enum Kind
    {
        /// <summary>A number (a double, or a T the recorder wrote as a number).</summary>
        Number,
        /// <summary>An integer.</summary>
        Int,
        /// <summary>true or false.</summary>
        Bool,
        /// <summary>An int array; <c>~</c> allowed.</summary>
        Ints,
        /// <summary>A string.</summary>
        String,
    }

    private sealed class Spec
    {
        internal Spec(int minInputs, int maxInputs, Dictionary<string, (Kind Kind, bool Required)> keys, string? notReplayable)
        {
            MinInputs = minInputs;
            MaxInputs = maxInputs;
            Keys = keys;
            NotReplayable = notReplayable;
        }

        internal int MinInputs { get; }
        internal int MaxInputs { get; }
        internal Dictionary<string, (Kind Kind, bool Required)> Keys { get; }
        internal string? NotReplayable { get; }
    }

    private static readonly Dictionary<string, Spec> Specs = BuildSpecs();

    /// <summary>The operation types this translator replays, for tests and diagnostics.</summary>
    internal static IReadOnlyCollection<string> MappedOperations
        => Specs.Where(s => s.Value.NotReplayable is null).Select(s => s.Key).ToArray();

    private static Dictionary<string, Spec> BuildSpecs()
    {
        var specs = new Dictionary<string, Spec>(StringComparer.Ordinal);

        // keys: "Name:k" with k one of d (number), i (int), b (bool), a (int array), s (string); a trailing '!' marks
        // a parameter the legacy method had no default for.
        void Def(string op, int min, int max, params string[] keys)
        {
            var map = new Dictionary<string, (Kind, bool)>(StringComparer.OrdinalIgnoreCase);
            foreach (var k in keys)
            {
                var colon = k.IndexOf(':');
                var required = k.EndsWith("!", StringComparison.Ordinal);
                var code = k[colon + 1];
                var kind = code switch
                {
                    'd' => Kind.Number,
                    'i' => Kind.Int,
                    'b' => Kind.Bool,
                    'a' => Kind.Ints,
                    's' => Kind.String,
                    _ => throw new InvalidOperationException($"Bad parameter spec '{k}'."),
                };
                map[k.Substring(0, colon)] = (kind, required);
            }
            specs[op] = new Spec(min, max, map, null);
        }

        void Unary(params string[] ops) { foreach (var op in ops) Def(op, 1, 1); }
        void Binary(params string[] ops) { foreach (var op in ops) Def(op, 2, 2); }

        Unary("Exp", "Log", "Sqrt", "Tanh", "Sigmoid", "ReLU", "Negate", "Abs", "Transpose", "Mean", "GELU", "Swish",
            "Mish", "SoftPlus", "SELU", "HardSigmoid", "HardTanh", "SoftSign", "LiSHT", "BentIdentity", "Gaussian", "Square");
        Binary("Add", "Subtract", "Multiply", "Divide", "MatMul", "GridSample", "Embedding");

        Def("Power", 1, 1, "Exponent:d!");
        Def("ReduceSum", 1, 1, "Axes:a", "KeepDims:b");
        Def("ReduceMax", 1, 1, "Axes:a", "KeepDims:b");
        Def("ReduceMean", 1, 1, "Axes:a", "KeepDims:b");
        Def("ReduceLogVariance", 1, 1, "Axis:i!", "Epsilon:d");
        Def("Norm", 1, 1, "Axis:i", "KeepDims:b", "Epsilon:d");
        Def("Reshape", 1, 1, "NewShape:a!");
        Def("Permute", 1, 1, "Axes:a!");
        Def("Broadcast", 1, 1, "TargetShape:a!");
        Def("Concat", 1, int.MaxValue, "Axis:i");
        Def("Pad", 1, 1, "Padding:a!");
        Def("Crop", 1, 1, "Cropping:a!");
        Def("Slice", 1, 1, "Start:i!", "Length:i!", "Step:i", "Axis:i");
        Def("Split", 1, 1, "Axis:i!", "NumSplits:i!", "SplitIndex:i!");
        Def("Upsample", 1, 1, "Scale:i!");
        Def("Upsample3D", 1, 1, "ScaleD:i!", "ScaleH:i!", "ScaleW:i!");
        Def("PixelShuffle", 1, 1, "UpscaleFactor:i!");

        Def("Softmax", 1, 1, "Axis:i");
        Def("LogSoftmax", 1, 1, "Axis:i");
        Def("Softmin", 1, 1, "Axis:i");
        Def("LogSoftmin", 1, 1, "Axis:i");
        Def("SphericalSoftmax", 1, 1, "Axis:i");
        Def("Sparsemax", 1, 1, "Axis:i");
        Def("TaylorSoftmax", 1, 1, "Order:i", "Axis:i");
        Def("GumbelSoftmax", 1, 1, "Temperature:d", "Hard:b");
        Def("TopKSoftmax", 1, 1, "K:i!");
        Def("Maxout", 1, 1, "NumPieces:i");
        Def("HierarchicalSoftmax", 2, 2, "NumClasses:i!", "TreeDepth:i");

        Def("ELU", 1, 1, "Alpha:d");
        Def("LeakyReLU", 1, 1, "Alpha:d");
        Def("PReLU", 1, 1, "Alpha:d");
        Def("CELU", 1, 1, "Alpha:d");
        Def("ScaledTanh", 1, 1, "Beta:d");
        Def("ThresholdedReLU", 1, 1, "Threshold:d");
        Def("ISRU", 1, 1, "Alpha:d");
        Def("Sign", 1, 1, "SurrogateBeta:d");
        Def("SQRBF", 1, 1, "Beta:d");
        Def("RReLU", 1, 1, "Lower:d", "Upper:d", "Alpha:d", "IsTraining:b");
        Def("Squash", 1, 1, "Epsilon:d");
        Def("SurrogateSpike", 1, 1, "Threshold:d", "SurrogateBeta:d");
        Def("StraightThroughThreshold", 1, 1, "Threshold:d!");
        Def("FakeQuantization", 1, 1, "NumBits:i", "Scale:d", "ZeroPoint:d", "Symmetric:b");

        Def("MaxPool2D", 1, 1, "KernelSize:a!", "Stride:a", "Padding:a");
        Def("AvgPool2D", 1, 1, "KernelSize:a!", "Stride:a", "Padding:a");
        Def("MaxPool3D", 1, 1, "KernelSize:a!", "Stride:a", "Padding:a");
        Def("Conv2D", 2, 3, "Stride:a", "Padding:a");
        Def("Convolution3D", 2, 3, "Stride:a", "Padding:a");
        Def("ConvTranspose2D", 2, 3, "Stride:a", "Padding:a", "OutputPadding:a");
        Def("DilatedConv2D", 2, 3, "Stride:a", "Padding:a", "Dilation:a");
        Def("DepthwiseConv2D", 2, 3, "Stride:a", "Padding:a");
        Def("LocallyConnectedConv2D", 2, 3, "Stride:a");
        Def("DeformableConv2D", 3, 5, "Stride:a", "Padding:a", "Dilation:a");

        Def("LayerNorm", 3, 3, "Epsilon:d");
        Def("BatchNorm", 3, 3, "Epsilon:d");
        Def("GroupNormalization", 3, 3, "NumGroups:i!", "Epsilon:d");

        Def("RBFKernel", 3, 3);
        Def("AffineGrid", 1, 1, "OutputSize:a!");
        Def("GraphConv", 3, 4);
        Def("LSTMCell", 6, 6, "gateOrder:s", "weightLayout:s", "stateLayout:s", "hiddenSize:i");
        Def("ComplexMatMul", 2, 2, "Format:s");
        Def("ComplexMultiply", 2, 2, "Format:s");
        Def("CRFForward", 2, 4);
        Def("SoftSplit", 3, 3, "FeatureIndex:i!", "Threshold:d!", "Temperature:d");
        Def("SoftKNN", 3, 3, "Temperature:d");
        Def("OctonionMatMul", 2, 3);
        Def("MobiusAdd", 2, 2, "curvature:d");
        Def("PoincareExpMap", 2, 2, "curvature:d");
        Def("PoincareLogMap", 2, 2, "curvature:d");
        Def("PoincareDistance", 2, 2, "curvature:d");
        // Custom carries the real operation in "op": PoincareProject (one input) or HyperbolicLinear (two or three).
        Def("Custom", 1, 3, "op:s!", "curvature:d", "epsilon:d", "inputFeatures:i", "outputFeatures:i");

        // Tagged by the legacy autodiff but impossible to replay from the record alone.
        specs["Constant"] = new Spec(0, 0, new Dictionary<string, (Kind, bool)>(),
            "it cannot appear in a saved graph: it is a captured constant, and the recorder kept no tensor values.");
        specs["Activation"] = new Spec(0, 0, new Dictionary<string, (Kind, bool)>(),
            "it cannot appear in a saved graph: it applied an activation-function object the recorder kept no trace of.");

        return specs;
    }

    private static void Validate(Step step, string layerName, string key)
    {
        if (!Specs.TryGetValue(step.Op, out var spec))
        {
            var why = Enum.TryParse<OperationType>(step.Op, false, out _) && step.Op != nameof(OperationType.Input)
                ? $"its recorded graph uses '{step.Op}', which the legacy autodiff never recorded."
                : step.Op == nameof(OperationType.Input)
                    ? $"its recorded step '{step.Text}' is malformed."
                    : $"its recorded graph uses the unknown operation '{step.Op}'.";
            throw Bad(layerName, key, why);
        }

        if (spec.NotReplayable is not null)
            throw Bad(layerName, key, $"its recorded graph uses '{step.Op}', and {spec.NotReplayable}");

        int minInputs = spec.MinInputs, maxInputs = spec.MaxInputs;
        if (step.Op == "Custom")
        {
            var op = step.Parameters.TryGetValue("op", out var raw) && raw.StartsWith("'", StringComparison.Ordinal) ? raw.Substring(1) : null;
            if (op == "PoincareProject") { minInputs = 1; maxInputs = 1; }
            else if (op == "HyperbolicLinear") { minInputs = 2; maxInputs = 3; }
            else if (op is not null)
                throw Bad(layerName, key, $"its recorded graph uses the unknown custom operation '{op}'.");
        }

        if (step.Inputs.Length < minInputs || step.Inputs.Length > maxInputs)
        {
            var expected = minInputs == maxInputs ? $"{minInputs}" : maxInputs == int.MaxValue ? $"at least {minInputs}" : $"{minInputs} to {maxInputs}";
            throw Bad(layerName, key, $"step '{step.Text}' gives '{step.Op}' {step.Inputs.Length} input(s) where it takes {expected}.");
        }

        foreach (var p in step.Parameters)
        {
            if (!spec.Keys.TryGetValue(p.Key, out var def))
                throw Bad(layerName, key, $"step '{step.Text}' records the parameter '{p.Key}', which '{step.Op}' does not take.");
            if (!IsValid(p.Value, def.Kind))
                throw Bad(layerName, key, $"step '{step.Text}' records '{p.Key}={p.Value}', which is not a valid {Describe(def.Kind)}.");
        }

        foreach (var k in spec.Keys)
        {
            if (k.Value.Required && !step.Parameters.ContainsKey(k.Key))
                throw Bad(layerName, key, $"step '{step.Text}' does not record the parameter '{k.Key}' that '{step.Op}' needs.");
        }

        if (step.Op == "LSTMCell")
        {
            // The recorded layout strings describe the only layout the legacy cell produced; anything else is not it.
            Expect(step, "gateOrder", "ifgo", layerName, key);
            Expect(step, "weightLayout", "input_by_gates", layerName, key);
            Expect(step, "stateLayout", "hidden_then_cell", layerName, key);
        }
        else if (step.Op is "ComplexMatMul")
        {
            if (step.Parameters.TryGetValue("Format", out var f) && f != "'split" && f != "'interleaved")
                throw Bad(layerName, key, $"step '{step.Text}' records the complex format {f.Substring(1)}, which is neither 'split' nor 'interleaved'.");
        }
    }

    private static void Expect(Step step, string name, string value, string layerName, string key)
    {
        if (step.Parameters.TryGetValue(name, out var raw) && raw != "'" + value)
            throw Bad(layerName, key, $"step '{step.Text}' records {name}={raw}, but the legacy cell only ever recorded '{value}'.");
    }

    private static string Describe(Kind kind) => kind switch
    {
        Kind.Number => "number",
        Kind.Int => "integer",
        Kind.Bool => "boolean",
        Kind.Ints => "integer array",
        _ => "string",
    };

    private static bool IsValid(string raw, Kind kind)
    {
        switch (kind)
        {
            case Kind.Number:
                return double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out _);
            case Kind.Int:
                return int.TryParse(raw, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out _);
            case Kind.Bool:
                return raw == "true" || raw == "false";
            case Kind.String:
                return raw.StartsWith("'", StringComparison.Ordinal);
            case Kind.Ints:
                if (raw == "~") return true;
                if (raw.Length < 2 || raw[0] != '[' || raw[raw.Length - 1] != ']') return false;
                var inner = raw.Substring(1, raw.Length - 2);
                return inner.Length == 0 || inner.Split('|').All(s => int.TryParse(s, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out _));
            default:
                return false;
        }
    }

    // ── Replay ──────────────────────────────────────────────────────────────────────────────────────────────

    private static Tensor<T> Run<T>(ParsedGraph graph, Tensor<T> input, string layerName, string key)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));

        var values = new Dictionary<int, Tensor<T>> { [graph.InputId] = input };
        foreach (var step in graph.Steps)
        {
            var args = new Tensor<T>[step.Inputs.Length];
            for (int i = 0; i < args.Length; i++) args[i] = values[step.Inputs[i]];

            try
            {
                values[step.Id] = new Interpreter<T>(step).Apply(args);
            }
            catch (Exception ex) when (!IsOwn(ex))
            {
                throw Bad(layerName, key, $"replaying step '{step.Text}' failed: {ex.Message}", ex);
            }
        }

        return values[graph.OutputId];
    }

    /// <summary>Replays one recorded step through the current engine.</summary>
    private readonly struct Interpreter<T>
    {
        private readonly Step _step;
        private readonly IEngine _e;
        private readonly INumericOperations<T> _ops;

        internal Interpreter(Step step)
        {
            _step = step;
            // Read per call, never cached: a tracer installs itself as the current engine for the calling thread.
            _e = AiDotNetEngine.Current;
            _ops = MathHelper.GetNumericOperations<T>();
        }

        private T C(double v) => _ops.FromDouble(v);

        private double D(string name, double fallback)
            => _step.Parameters.TryGetValue(name, out var raw) ? double.Parse(raw, NumberStyles.Float, CultureInfo.InvariantCulture) : fallback;

        private double? OptD(string name)
            => _step.Parameters.TryGetValue(name, out var raw) ? double.Parse(raw, NumberStyles.Float, CultureInfo.InvariantCulture) : null;

        private int I(string name, int fallback)
            => _step.Parameters.TryGetValue(name, out var raw) ? int.Parse(raw, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture) : fallback;

        private bool B(string name, bool fallback)
            => _step.Parameters.TryGetValue(name, out var raw) ? raw == "true" : fallback;

        private string S(string name, string fallback)
            => _step.Parameters.TryGetValue(name, out var raw) ? raw.Substring(1) : fallback;

        private int[]? A(string name)
        {
            if (!_step.Parameters.TryGetValue(name, out var raw) || raw == "~") return null;
            var inner = raw.Substring(1, raw.Length - 2);
            return inner.Length == 0
                ? Array.Empty<int>()
                : inner.Split('|').Select(s => int.Parse(s, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture)).ToArray();
        }

        private int[] Req(string name) => A(name) ?? throw new ArgumentException($"'{name}' is null.");

        internal Tensor<T> Apply(Tensor<T>[] x)
        {
            var e = _e;
            var a = x[0];
            switch (_step.Op)
            {
                // Arithmetic. Add and Multiply broadcast like the legacy BroadcastAdd/BroadcastMultiply (trailing
                // alignment, size-1 dimensions stretched); Subtract was the engine's TensorSubtract already.
                case "Add": return e.TensorAdd(a, x[1]);
                case "Subtract": return e.TensorSubtract(a, x[1]);
                case "Multiply": return e.TensorMultiply(a, x[1]);
                case "Divide": return e.TensorDivide(a, x[1]);
                case "Power": return e.TensorPower(a, C(D("Exponent", 1)));
                case "Negate": return e.TensorNegate(a);
                case "Abs": return e.TensorAbs(a);
                case "Exp": return e.TensorExp(a);
                case "Log": return e.TensorLog(a);
                case "Sqrt": return e.TensorSqrt(a);
                case "Square": return e.TensorSquare(a);

                // Matrix ops. Two legacy methods shared the MatMul tag: MatrixMultiply (engine TensorMatMul) and
                // BatchMatrixMultiply (engine BatchMatMul, for two rank-3 operands).
                case "MatMul": return a.Rank == 3 && x[1].Rank == 3 ? e.BatchMatMul(a, x[1]) : e.TensorMatMul(a, x[1]);
                case "Transpose": return e.TensorTranspose(a);

                // Activations.
                case "ReLU": return e.ReLU(a);
                case "Sigmoid": return e.Sigmoid(a);
                case "Tanh": return e.Tanh(a);
                case "GELU": return e.GELU(a);
                case "Swish": return e.Swish(a);
                case "Mish": return e.Mish(a);
                case "SoftPlus": return e.Softplus(a);
                case "SELU": return e.TensorSELU(a);
                case "HardSigmoid": return HardSigmoid(a);
                case "HardTanh": return e.TensorHardtanh(a, -1.0, 1.0);
                case "SoftSign": return e.TensorSoftsign(a);
                case "ELU": return e.ELU(a, D("Alpha", 1.0));
                case "LeakyReLU": return e.LeakyReLU(a, C(D("Alpha", 0.01)));
                case "PReLU": return e.LeakyReLU(a, C(D("Alpha", 0.01)));
                case "RReLU": return RReLU(a);
                case "CELU": return e.TensorCelu(a, D("Alpha", 1.0));
                case "LiSHT": return e.TensorLisht(a);
                case "BentIdentity": return e.TensorBentIdentity(a);
                case "Gaussian": return e.TensorGaussian(a);
                // Legacy: (1 - e^(-βx)) / (1 + e^(-βx)) = tanh(βx / 2).
                case "ScaledTanh": return e.TensorScaledTanh(a, 1.0, D("Beta", 1.0) / 2.0);
                case "ThresholdedReLU": return e.TensorThresholdedRelu(a, D("Threshold", 1.0));
                case "ISRU": return e.TensorIsru(a, D("Alpha", 1.0));
                case "Sign": return e.TensorSign(a);
                case "SQRBF": return e.TensorSqrbf(a, D("Beta", 1.0));
                case "Squash": return Squash(a, D("Epsilon", 1e-7));
                case "SurrogateSpike": return e.TensorSurrogateSpike(a, D("Threshold", 1.0), D("SurrogateBeta", 1.0));
                case "StraightThroughThreshold": return e.TensorGreaterThan(a, C(D("Threshold", 0)));
                case "FakeQuantization": return FakeQuantize(a);

                // Softmax family.
                case "Softmax": return e.Softmax(a, I("Axis", -1));
                case "LogSoftmax": return e.TensorLogSoftmax(a, Axis(a, I("Axis", -1)));
                case "Softmin": return e.TensorSoftmin(a, Axis(a, I("Axis", -1)));
                case "LogSoftmin": return e.TensorLogSoftmin(a, Axis(a, I("Axis", -1)));
                case "SphericalSoftmax": return e.SphericalSoftmax(a, Axis(a, I("Axis", -1)));
                case "Sparsemax": return e.Sparsemax(a, Axis(a, I("Axis", -1)));
                case "TaylorSoftmax": return TaylorSoftmax(a);
                case "GumbelSoftmax": return GumbelSoftmax(a);
                case "TopKSoftmax": return e.TensorTopKSoftmax(a, I("K", 1));
                case "Maxout": return Maxout(a, I("NumPieces", 2));
                case "HierarchicalSoftmax": return HierarchicalSoftmax(a, x[1], I("NumClasses", 0));

                // Reductions.
                case "ReduceSum": return ReduceSum(a);
                case "Mean": return e.Reshape(e.ReduceMean(a, AllAxes(a), false), new[] { 1 });
                case "ReduceMean": return Reduce(a, mean: true);
                case "ReduceMax": return Reduce(a, mean: false);
                case "ReduceLogVariance": return ReduceLogVariance(a);
                case "Norm": return Norm(a);

                // Shape.
                case "Reshape": return e.Reshape(a, Req("NewShape"));
                case "Permute": return e.TensorPermute(a, Req("Axes"));
                case "Broadcast": return e.TensorBroadcastTo(a, Req("TargetShape"));
                case "Concat": return e.Concat(x, Axis(a, I("Axis", 0)));
                case "Pad": return Pad(a, Req("Padding"));
                case "Crop": return Crop(a, Req("Cropping"));
                case "Slice": return Slice(a);
                case "Split": return Split(a);
                case "Upsample": return e.Upsample(a, I("Scale", 1), I("Scale", 1));
                case "Upsample3D": return e.Upsample3D(a, I("ScaleD", 1), I("ScaleH", 1), I("ScaleW", 1));
                case "PixelShuffle": return e.PixelShuffle(a, I("UpscaleFactor", 1));

                // Pooling. The legacy methods recorded Padding [0, 0] but never padded; a recorded padding is ignored
                // the same way.
                case "MaxPool2D": return MaxPool2D(a);
                case "AvgPool2D": return e.AvgPool2D(a, Req("KernelSize"), A("Stride") ?? Req("KernelSize"));
                case "MaxPool3D": return e.MaxPool3D(a, Req("KernelSize"), A("Stride") ?? Req("KernelSize"), new[] { 0, 0, 0 });

                // Convolutions. Bias is added per output channel after the convolution, as the legacy loops did.
                case "Conv2D":
                    return Bias(e.Conv2D(a, x[1], A("Stride") ?? new[] { 1, 1 }, A("Padding") ?? new[] { 0, 0 }, new[] { 1, 1 }), x, 2);
                case "DilatedConv2D":
                    return Bias(e.Conv2D(a, x[1], A("Stride") ?? new[] { 1, 1 }, A("Padding") ?? new[] { 0, 0 }, A("Dilation") ?? new[] { 1, 1 }), x, 2);
                case "Convolution3D":
                    return Bias(e.Conv3D(a, x[1], A("Stride") ?? new[] { 1, 1, 1 }, A("Padding") ?? new[] { 0, 0, 0 }, new[] { 1, 1, 1 }), x, 2);
                case "ConvTranspose2D":
                    return Bias(e.ConvTranspose2D(a, x[1], A("Stride") ?? new[] { 1, 1 }, A("Padding") ?? new[] { 0, 0 }, A("OutputPadding") ?? new[] { 0, 0 }), x, 2);
                case "DepthwiseConv2D":
                    return Bias(e.DepthwiseConv2D(a, x[1], A("Stride") ?? new[] { 1, 1 }, A("Padding") ?? new[] { 0, 0 }), x, 2);
                case "LocallyConnectedConv2D":
                    return e.LocallyConnectedConv2D(a, x[1], x.Length > 2 ? x[2] : null, A("Stride") ?? new[] { 1, 1 });
                case "DeformableConv2D": return DeformableConv2D(x);

                // Normalization.
                case "LayerNorm": return LayerNorm(a, x[1], x[2], D("Epsilon", 1e-5));
                case "BatchNorm": return BatchNorm(a, x[1], x[2], D("Epsilon", 1e-5));
                case "GroupNormalization": return GroupNorm(a, I("NumGroups", 1), x[1], x[2], D("Epsilon", 1e-5));

                // Everything else.
                case "RBFKernel": return e.RBFKernel(a, x[1], x[2]);
                case "AffineGrid": return AffineGrid(a, Req("OutputSize"));
                case "GridSample": return GridSample(a, x[1]);
                case "GraphConv": return GraphConv(x);
                case "Embedding": return Embedding(a, x[1]);
                case "LSTMCell": return LstmCell(x);
                case "ComplexMatMul": return ComplexMatMul(a, x[1], S("Format", "split") == "interleaved");
                case "ComplexMultiply": return ComplexMultiply(a, x[1]);
                case "CRFForward":
                    return e.TensorCrfForward(a, x[1], x.Length > 2 ? x[2] : null, x.Length > 3 ? x[3] : null);
                case "SoftSplit":
                    return e.TensorSoftSplit(a, x[1], x[2], I("FeatureIndex", 0), D("Threshold", 0), D("Temperature", 1.0));
                case "SoftKNN": return SoftKnn(a, x[1], x[2], D("Temperature", 1.0));
                case "OctonionMatMul": return OctonionMatMul(x);
                case "MobiusAdd": return Mobius(a, x[1], Math.Abs(D("curvature", -1.0)), 1e-10);
                case "PoincareExpMap": return PoincareExpMap(a, x[1], Math.Abs(D("curvature", -1.0)));
                case "PoincareLogMap": return PoincareLogMap(a, x[1], Math.Abs(D("curvature", -1.0)));
                case "PoincareDistance": return PoincareDistance(a, x[1], Math.Abs(D("curvature", -1.0)));
                case "Custom":
                    return S("op", string.Empty) == "PoincareProject"
                        ? Project(a, Math.Abs(D("curvature", -1.0)), D("epsilon", 1e-5))
                        : HyperbolicLinear(x, Math.Abs(D("curvature", -1.0)));

                default:
                    // Validate rejected every other name; reaching here is a table bug.
                    throw new InvalidOperationException($"'{_step.Op}' has a spec but no replay.");
            }
        }

        // ── helpers ──

        private static int Axis(Tensor<T> t, int axis) => axis < 0 ? t.Rank + axis : axis;

        private static int[] AllAxes(Tensor<T> t) => Enumerable.Range(0, t.Rank).ToArray();

        private Tensor<T> Scale(Tensor<T> t, double s) => _e.TensorMultiplyScalar(t, C(s));

        private Tensor<T> Shift(Tensor<T> t, double s) => _e.TensorAddScalar(t, C(s));

        /// <summary>Sum over the last axis, kept as size 1.</summary>
        private Tensor<T> SumLast(Tensor<T> t) => _e.ReduceSum(t, new[] { t.Rank - 1 }, true);

        private Tensor<T> Shaped(Tensor<T> t, int[] shape) => t.Shape.ToArray().SequenceEqual(shape) ? t : _e.Reshape(t, shape);

        /// <summary>The legacy reduced shape: kept dimensions (1 for a reduced one under keepDims), or [1] when none remain.</summary>
        private static int[] ReducedShape(int[] shape, int[] axes, bool keepDims)
        {
            var kept = new List<int>();
            for (int i = 0; i < shape.Length; i++)
            {
                if (!axes.Contains(i)) kept.Add(shape[i]);
                else if (keepDims) kept.Add(1);
            }
            if (kept.Count == 0) kept.Add(1);
            return kept.ToArray();
        }

        private Tensor<T> HardSigmoid(Tensor<T> a)
            // Legacy: clip(x / 6 + 1/2, 0, 1).
            => _e.TensorClamp(Shift(Scale(a, 1.0 / 6.0), 0.5), _ops.Zero, _ops.One);

        private Tensor<T> RReLU(Tensor<T> a)
        {
            // Legacy drew one alpha per call (the midpoint outside training) and recorded it, so a replay uses that
            // recorded slope; without one it is the midpoint of the recorded range.
            var alpha = OptD("Alpha") ?? (D("Lower", 0.125) + D("Upper", 0.333)) / 2.0;
            return _e.LeakyReLU(a, C(alpha));
        }

        private Tensor<T> Squash(Tensor<T> a, double epsilon)
        {
            // v · (‖v‖² / (1 + ‖v‖²)) / sqrt(‖v‖² + ε) over the last axis.
            var e = _e;
            var sq = SumLast(e.TensorSquare(a));
            var factor = e.TensorDivide(sq, e.TensorMultiply(Shift(sq, 1.0), e.TensorSqrt(Shift(sq, epsilon))));
            return e.TensorMultiply(a, factor);
        }

        private Tensor<T> FakeQuantize(Tensor<T> a)
        {
            // The legacy record holds the scale and zero point it used, so the replay quantizes with exactly those.
            // A zero (or missing) scale is derived from the data, as the legacy did; the zero point is passed
            // explicitly because the legacy never derived one.
            var scale = OptD("Scale");
            var zeroPoint = (int)Math.Round(D("ZeroPoint", 0), MidpointRounding.ToEven);
            return _e.TensorFakeQuantize(a, I("NumBits", 8), scale is null || scale.Value == 0 ? null : scale, zeroPoint, B("Symmetric", true));
        }

        private Tensor<T> TaylorSoftmax(Tensor<T> a)
        {
            var order = I("Order", 2);
            if (order < 1) throw new ArgumentOutOfRangeException("Order", order, "Order must be at least 1.");
            var axis = Axis(a, I("Axis", -1));
            if (axis < 0 || axis >= a.Rank)
                throw new ArgumentOutOfRangeException("Axis", $"Axis {axis} is out of range for tensor with {a.Rank} dimensions.");

            // Legacy: Σₙ xⁿ/n! on the raw values (no max shift, which a truncated series is not invariant to), a
            // non-positive sum replaced by 1e-10, then normalized along the axis. The engine's TaylorSoftmax shifts by
            // the row max first, so it computes something else.
            var e = _e;
            var series = e.TensorOnesLike(a);
            var term = e.TensorOnesLike(a);
            for (int n = 1; n <= order; n++)
            {
                term = Scale(e.TensorMultiply(term, a), 1.0 / n);
                series = e.TensorAdd(series, term);
            }
            var floor = e.TensorMultiplyScalar(e.TensorOnesLike(a), C(1e-10));
            var positive = e.TensorWhere(e.TensorGreaterThan(series, _ops.Zero), series, floor);
            return e.TensorDivide(positive, e.ReduceSum(positive, new[] { axis }, true));
        }

        private Tensor<T> GumbelSoftmax(Tensor<T> a)
        {
            var temperature = D("Temperature", 1.0);
            if (!(temperature > 0) || double.IsInfinity(temperature))
                throw new ArgumentOutOfRangeException("Temperature", temperature, "Temperature must be a positive finite number.");
            return _e.GumbelSoftmax(a, temperature, B("Hard", false), -1);
        }

        private Tensor<T> Maxout(Tensor<T> a, int pieces)
        {
            if (a.Rank != 2) throw new ArgumentException($"Maxout requires 2D input [batch, features], got {a.Rank}D");
            if (pieces <= 0 || a.Shape[1] % pieces != 0)
                throw new ArgumentException($"Features ({a.Shape[1]}) must be divisible by numPieces ({pieces})");
            var grouped = _e.Reshape(a, new[] { a.Shape[0], a.Shape[1] / pieces, pieces });
            return _e.ReduceMax(grouped, new[] { 2 }, false);
        }

        private Tensor<T> HierarchicalSoftmax(Tensor<T> input, Tensor<T> nodeWeights, int numClasses)
        {
            if (input.Rank != 2) throw new ArgumentException($"Input must be 2D [batch, inputDim], got {input.Rank}D");
            if (nodeWeights.Rank != 2) throw new ArgumentException($"NodeWeights must be 2D [treeDepth, inputDim], got {nodeWeights.Rank}D");
            // The legacy path product stopped at 1e-10, so no probability falls below it.
            return _e.TensorMax(_e.FusedHierarchicalSoftmax(input, nodeWeights, numClasses), C(1e-10));
        }

        private Tensor<T> ReduceSum(Tensor<T> a)
        {
            // Legacy Tensor.Sum: no axes sums everything into [1]; otherwise the axes are removed (KeepDims was
            // recorded but never applied).
            var axes = A("Axes");
            if (axes is null || axes.Length == 0)
                return Shaped(_e.ReduceSum(a, AllAxes(a), false), new[] { 1 });
            var normalized = axes.Select(ax => Axis(a, ax)).Distinct().OrderBy(ax => ax).ToArray();
            var shape = a.Shape.ToArray().Where((_, i) => !normalized.Contains(i)).ToArray();
            var summed = _e.ReduceSum(a, normalized, false);
            return shape.Length == 0 ? Shaped(summed, new[] { 1 }) : Shaped(summed, shape);
        }

        private Tensor<T> Reduce(Tensor<T> a, bool mean)
        {
            var axes = (A("Axes") ?? AllAxes(a)).Select(ax => Axis(a, ax)).Distinct().OrderBy(ax => ax).ToArray();
            var keepDims = B("KeepDims", false);
            var reduced = mean ? _e.ReduceMean(a, axes, keepDims) : _e.ReduceMax(a, axes, keepDims);
            return Shaped(reduced, ReducedShape(a.Shape.ToArray(), axes, keepDims));
        }

        private Tensor<T> ReduceLogVariance(Tensor<T> a)
        {
            var axis = I("Axis", 0);
            if (axis < 0 || axis >= a.Rank) throw new ArgumentException($"Axis {axis} is out of range for tensor of rank {a.Rank}");
            var reduced = _e.ReduceLogVariance(a, new[] { axis }, false, D("Epsilon", 1e-8));
            return Shaped(reduced, ReducedShape(a.Shape.ToArray(), new[] { axis }, false));
        }

        private Tensor<T> Norm(Tensor<T> a)
        {
            // sqrt(Σ x² + ε) along one axis.
            var axis = Axis(a, I("Axis", -1));
            if (axis < 0 || axis >= a.Rank) throw new ArgumentException($"Axis {axis} is out of range for tensor with {a.Rank} dimensions.");
            var keepDims = B("KeepDims", false);
            var norm = _e.TensorSqrt(Shift(_e.ReduceSum(_e.TensorSquare(a), new[] { axis }, keepDims), D("Epsilon", 1e-12)));
            var shape = keepDims
                ? a.Shape.ToArray().Select((s, i) => i == axis ? 1 : s).ToArray()
                : a.Shape.ToArray().Where((_, i) => i != axis).ToArray();
            return shape.Length == 0 ? norm : Shaped(norm, shape);
        }

        private Tensor<T> Pad(Tensor<T> a, int[] padding)
        {
            if (padding.Length != a.Rank)
                throw new ArgumentException($"Padding array length ({padding.Length}) must match input rank ({a.Rank})");
            // The legacy Pad(int[]) padded every axis by padding[d] on both sides with zeros. TensorConstantPad takes
            // (before, after) pairs starting from the last axis.
            var pairs = new int[2 * a.Rank];
            for (int i = 0; i < a.Rank; i++)
            {
                int d = a.Rank - 1 - i;
                pairs[2 * i] = padding[d];
                pairs[2 * i + 1] = padding[d];
            }
            return _e.TensorConstantPad(a, pairs, _ops.Zero);
        }

        private Tensor<T> Crop(Tensor<T> a, int[] cropping)
        {
            if (a.Rank != 4 || cropping.Length != 4)
                throw new NotSupportedException($"Crop operation not supported for shape {string.Join("x", a.Shape.ToArray())} with cropping {string.Join(",", cropping)}");
            int outH = a.Shape[2] - cropping[0] - cropping[1];
            int outW = a.Shape[3] - cropping[2] - cropping[3];
            if (outH <= 0 || outW <= 0) throw new ArgumentException("Cropping results in non-positive dimensions.");
            return _e.TensorNarrow(_e.TensorNarrow(a, 2, cropping[0], outH), 3, cropping[2], outW);
        }

        private Tensor<T> Slice(Tensor<T> a)
        {
            int start = I("Start", 0), length = I("Length", 0), step = I("Step", 1), axis = Axis(a, I("Axis", 0));
            if (axis < 0 || axis >= a.Rank)
                throw new ArgumentOutOfRangeException("Axis", $"Axis {axis} is out of range for tensor with {a.Rank} dimensions.");
            int size = a.Shape[axis];
            if (start < 0 || start >= size)
                throw new ArgumentOutOfRangeException("Start", $"Start index {start} is out of range for axis with size {size}.");
            if (step <= 0) throw new ArgumentException("Step must be positive.", "Step");

            int count = 0;
            for (int i = start; i < size && count < length; i += step) count++;
            if (step == 1) return _e.TensorNarrow(a, axis, start, count);

            var indices = new int[count];
            for (int i = 0; i < count; i++) indices[i] = start + i * step;
            return _e.TensorIndexSelect(a, new Tensor<int>(new[] { count }, new Vector<int>(indices)), axis);
        }

        private Tensor<T> Split(Tensor<T> a)
        {
            int axis = I("Axis", 0), numSplits = I("NumSplits", 1), index = I("SplitIndex", 0);
            if (axis < 0 || axis >= a.Rank)
                throw new ArgumentException($"Axis {axis} is out of bounds for tensor with {a.Rank} dimensions.");
            if (numSplits <= 0 || a.Shape[axis] % numSplits != 0)
                throw new ArgumentException($"Dimension size {a.Shape[axis]} is not evenly divisible by {numSplits}.");
            if (index < 0 || index >= numSplits)
                throw new ArgumentException($"Split index {index} is out of range for {numSplits} splits.");
            int size = a.Shape[axis] / numSplits;
            return _e.TensorNarrow(a, axis, index * size, size);
        }

        private Tensor<T> MaxPool2D(Tensor<T> a)
        {
            if (a.Rank != 4) throw new ArgumentException("MaxPool2D requires 4D input [batch, channels, height, width]");
            var pool = Req("KernelSize");
            var stride = A("Stride") ?? pool;
            return _e.MaxPool2DWithTensorIndices(a, pool, stride, out _);
        }

        /// <summary>Adds the optional per-output-channel bias at <paramref name="index"/> to a channels-first result.</summary>
        private Tensor<T> Bias(Tensor<T> result, Tensor<T>[] x, int index)
        {
            if (x.Length <= index) return result;
            var shape = Enumerable.Repeat(1, result.Rank).ToArray();
            shape[1] = result.Shape[1];
            return _e.TensorAdd(result, _e.Reshape(x[index], shape));
        }

        private Tensor<T> DeformableConv2D(Tensor<T>[] x)
        {
            // Parents are input, kernel, offset, then the mask and the bias when given. With four parents the fourth
            // is the bias when it is a vector (one value per output channel) and the mask otherwise.
            Tensor<T>? mask = null;
            int biasIndex = int.MaxValue;
            if (x.Length == 5) { mask = x[3]; biasIndex = 4; }
            else if (x.Length == 4)
            {
                if (x[3].Rank == 1) biasIndex = 3;
                else mask = x[3];
            }

            var result = _e.DeformableConv2D(x[0], x[1], x[2], mask,
                A("Stride") ?? new[] { 1, 1 }, A("Padding") ?? new[] { 0, 0 }, A("Dilation") ?? new[] { 1, 1 });
            return biasIndex < x.Length ? Bias(result, x, biasIndex) : result;
        }

        private Tensor<T> LayerNorm(Tensor<T> a, Tensor<T> gamma, Tensor<T> beta, double epsilon)
        {
            // The legacy record kept no normalizedShape. Gamma was indexed over the normalized elements, so the
            // normalized axes are the trailing gamma.Rank axes of the input.
            int norm = gamma.Rank;
            if (norm > a.Rank) throw new ArgumentException("normalizedShape has more dimensions than input tensor");
            var axes = Enumerable.Range(a.Rank - norm, norm).ToArray();
            return Normalize(a, axes, gamma, beta, epsilon, gamma.Shape.ToArray());
        }

        private Tensor<T> BatchNorm(Tensor<T> a, Tensor<T> gamma, Tensor<T> beta, double epsilon)
        {
            // Batch statistics (the legacy default, training = true) per feature of a [N, C] input and per channel
            // (axis 1) otherwise; gamma and beta are per channel.
            if (a.Rank < 2) throw new ArgumentException("BatchNorm requires an input of rank 2 or more.");
            var axes = Enumerable.Range(0, a.Rank).Where(i => i != 1).ToArray();
            var affine = Enumerable.Repeat(1, a.Rank).ToArray();
            affine[1] = a.Shape[1];
            return Normalize(a, axes, gamma, beta, epsilon, affine);
        }

        private Tensor<T> Normalize(Tensor<T> a, int[] axes, Tensor<T>? gamma, Tensor<T>? beta, double epsilon, int[] affineShape)
        {
            var e = _e;
            var mean = e.ReduceMean(a, axes, true);
            var centered = e.TensorSubtract(a, mean);
            var variance = e.ReduceMean(e.TensorSquare(centered), axes, true);
            var normalized = e.TensorDivide(centered, e.TensorSqrt(Shift(variance, epsilon)));
            return gamma is null || beta is null
                ? normalized
                : e.TensorAdd(e.TensorMultiply(normalized, Shaped(gamma, affineShape)), Shaped(beta, affineShape));
        }

        private Tensor<T> GroupNorm(Tensor<T> a, int numGroups, Tensor<T> gamma, Tensor<T> beta, double epsilon)
        {
            if (a.Rank < 2) throw new ArgumentException("GroupNorm requires at least 2D input [batch, channels, ...]");
            int batch = a.Shape[0], channels = a.Shape[1];
            if (numGroups <= 0 || channels % numGroups != 0)
                throw new ArgumentException($"Number of channels ({channels}) must be divisible by number of groups ({numGroups}).");
            var grouped = _e.Reshape(a, new[] { batch, numGroups, a.Length / (batch * numGroups) });
            var normalized = _e.Reshape(Normalize(grouped, new[] { 2 }, null, null, epsilon, Array.Empty<int>()), a.Shape.ToArray());
            var affine = Enumerable.Repeat(1, a.Rank).ToArray();
            affine[1] = channels;
            return _e.TensorAdd(_e.TensorMultiply(normalized, Shaped(gamma, affine)), Shaped(beta, affine));
        }

        private Tensor<T> AffineGrid(Tensor<T> theta, int[] size)
        {
            if (size.Length != 2) throw new ArgumentException("OutputSize must be [height, width].");
            if (theta.Rank != 3 || theta.Shape[1] != 2 || theta.Shape[2] != 3)
                throw new ArgumentException("Theta must be of shape [batch, 2, 3]");
            int h = size[0], w = size[1], batch = theta.Shape[0];
            // Legacy base grid: x = w / max(W - 1, 1) · 2 - 1, y likewise, homogeneous 1; grid = base · thetaᵀ.
            var grid = new Tensor<T>(new[] { h * w, 3 });
            for (int r = 0, i = 0; r < h; r++)
            {
                for (int c = 0; c < w; c++, i++)
                {
                    grid[i, 0] = C((double)c / Math.Max(w - 1, 1) * 2.0 - 1.0);
                    grid[i, 1] = C((double)r / Math.Max(h - 1, 1) * 2.0 - 1.0);
                    grid[i, 2] = _ops.One;
                }
            }
            var based = _e.TensorBroadcastTo(_e.Reshape(grid, new[] { 1, h * w, 3 }), new[] { batch, h * w, 3 });
            var mapped = _e.BatchMatMul(based, _e.TensorPermute(theta, new[] { 0, 2, 1 }));
            return _e.Reshape(mapped, new[] { batch, h, w, 2 });
        }

        private Tensor<T> GridSample(Tensor<T> input, Tensor<T> grid)
        {
            // Legacy: NHWC input, bilinear with corner-aligned coordinates. Inside the image that is the engine's
            // corner-aligned bilinear sample; outside it the legacy clamped indices and weights, which the border
            // padding reproduces up to one pixel past the edge.
            if (input.Rank != 4) throw new ArgumentException("Input must be 4D tensor [batch, height, width, channels]");
            if (grid.Rank != 4 || grid.Shape[3] != 2) throw new ArgumentException("Grid must be 4D tensor [batch, out_height, out_width, 2]");
            var nchw = _e.TensorPermute(input, new[] { 0, 3, 1, 2 });
            var sampled = _e.GridSample(nchw, grid, GridSampleMode.Bilinear, GridSamplePadding.Border, true);
            return _e.TensorPermute(sampled, new[] { 0, 2, 3, 1 });
        }

        private Tensor<T> GraphConv(Tensor<T>[] x)
        {
            // adjacency · (input · weights) + bias, per batch.
            Tensor<T> input = x[0], adjacency = x[1], weights = x[2];
            if (input.Rank != 3) throw new ArgumentException("Input must be 3D tensor [batch, numNodes, inputFeatures]");
            if (adjacency.Rank != 3) throw new ArgumentException("Adjacency must be 3D tensor [batch, numNodes, numNodes]");
            if (weights.Rank != 2) throw new ArgumentException("Weights must be 2D tensor [inputFeatures, outputFeatures]");
            int batch = input.Shape[0], nodes = input.Shape[1], features = input.Shape[2], outputs = weights.Shape[1];
            var flat = _e.TensorMatMul(_e.Reshape(input, new[] { batch * nodes, features }), weights);
            var output = _e.BatchMatMul(adjacency, _e.Reshape(flat, new[] { batch, nodes, outputs }));
            return x.Length > 3 ? _e.TensorAdd(output, x[3]) : output;
        }

        private Tensor<T> Embedding(Tensor<T> embeddings, Tensor<T> indices)
        {
            // Legacy: indices truncated toward zero; [B] or [B, 1] gives [B, E], [B, S] gives [B, S, E].
            var idx = _e.TensorTrunc(indices);
            if (idx.Rank == 2 && idx.Shape[1] == 1) idx = _e.Reshape(idx, new[] { idx.Shape[0] });
            return _e.TensorEmbeddingLookupFromFloatIndices(embeddings, idx);
        }

        private Tensor<T> LstmCell(Tensor<T>[] x)
        {
            // The legacy cell packed [hidden | cell] along the last axis; the Slice steps after it read the halves.
            var (h, c) = _e.TensorLstmCell(x[0], x[1], x[2], _e.TensorTranspose(x[3]), _e.TensorTranspose(x[4]), x[5], null);
            return _e.TensorConcatenate(new[] { h, c }, h.Rank - 1);
        }

        private Tensor<T> MatMul(Tensor<T> a, Tensor<T> b)
            => a.Rank == 3 && b.Rank == 3 ? _e.BatchMatMul(a, b) : _e.TensorMatMul(a, b);

        private Tensor<T> ComplexMatMul(Tensor<T> a, Tensor<T> b, bool interleaved)
        {
            int last = a.Rank - 1, rowAxis = b.Rank - 2;
            int k = a.Shape[last] / 2;
            int m = a.Shape[a.Rank - 2], n = b.Shape[b.Rank - 1];
            int batch = a.Rank > 2 ? a.Shape[0] : 1;
            Tensor<T> aR, aI, bR, bI;
            if (interleaved)
            {
                // Real and imaginary parts alternate along a's last axis and b's row axis.
                var aPairs = _e.Reshape(_e.TensorNarrow(a, last, 0, 2 * k), a.Shape.ToArray().Take(last).Concat(new[] { k, 2 }).ToArray());
                aR = _e.TensorSelect(aPairs, last + 1, 0);
                aI = _e.TensorSelect(aPairs, last + 1, 1);
                var bShape = b.Shape.ToArray().Take(rowAxis).Concat(new[] { k, 2, n }).ToArray();
                var bPairs = _e.Reshape(_e.TensorNarrow(b, rowAxis, 0, 2 * k), bShape);
                bR = _e.TensorSelect(bPairs, rowAxis + 1, 0);
                bI = _e.TensorSelect(bPairs, rowAxis + 1, 1);
            }
            else
            {
                aR = _e.TensorNarrow(a, last, 0, k);
                aI = _e.TensorNarrow(a, last, k, k);
                bR = _e.TensorNarrow(b, rowAxis, 0, k);
                bI = _e.TensorNarrow(b, rowAxis, k, k);
            }

            var real = _e.TensorSubtract(MatMul(aR, bR), MatMul(aI, bI));
            var imag = _e.TensorAdd(MatMul(aR, bI), MatMul(aI, bR));
            var shape = batch > 1 ? new[] { batch, m, 2 * n } : new[] { m, 2 * n };
            if (!interleaved) return Shaped(_e.TensorConcatenate(new[] { real, imag }, real.Rank - 1), shape);
            return _e.Reshape(_e.TensorStack(new[] { real, imag }, real.Rank), shape);
        }

        private Tensor<T> ComplexMultiply(Tensor<T> a, Tensor<T> b)
        {
            // The legacy kernel always used the split layout (first half real, second half imaginary along the
            // last axis), whatever Format said; an odd trailing element stayed zero.
            if (!a.Shape.ToArray().SequenceEqual(b.Shape.ToArray()))
                throw new ArgumentException("Tensors must have the same shape for complex multiplication.");
            int last = a.Rank - 1, n = a.Shape[last] / 2;
            Tensor<T> aR = _e.TensorNarrow(a, last, 0, n), aI = _e.TensorNarrow(a, last, n, n);
            Tensor<T> bR = _e.TensorNarrow(b, last, 0, n), bI = _e.TensorNarrow(b, last, n, n);
            var parts = new List<Tensor<T>>
            {
                _e.TensorSubtract(_e.TensorMultiply(aR, bR), _e.TensorMultiply(aI, bI)),
                _e.TensorAdd(_e.TensorMultiply(aR, bI), _e.TensorMultiply(aI, bR)),
            };
            if (a.Shape[last] % 2 == 1) parts.Add(_e.TensorZerosLike(_e.TensorNarrow(a, last, 2 * n, 1)));
            return _e.TensorConcatenate(parts.ToArray(), last);
        }

        private Tensor<T> SoftKnn(Tensor<T> input, Tensor<T> supportVectors, Tensor<T> labels, double temperature)
        {
            // Legacy: one query, the first nFeatures values of the flattened input.
            int features = supportVectors.Rank > 1 ? supportVectors.Shape[1] : 1;
            var flat = _e.Reshape(input, new[] { input.Length });
            var query = flat.Length == features ? flat : _e.TensorNarrow(flat, 0, 0, Math.Min(features, flat.Length));
            return _e.TensorSoftKnn(query, supportVectors, labels, temperature);
        }

        private Tensor<T> OctonionMatMul(Tensor<T>[] x)
        {
            Tensor<T> input = x[0], weights = x[1];
            if (input.Rank != 2) throw new ArgumentException("OctonionMatMul input must have shape [batch, inputFeatures * 8]");
            if (weights.Rank != 3 || weights.Shape[2] != 8)
                throw new ArgumentException("OctonionMatMul weights must have shape [outputFeatures, inputFeatures, 8]");
            int batch = input.Shape[0], outputs = weights.Shape[0], features = weights.Shape[1];
            if (input.Shape[1] != features * 8)
                throw new ArgumentException($"Input features ({input.Shape[1] / 8}) must match weight input features ({features})");

            // The legacy kernel used its own multiplication table, which is not the one the engine's
            // OctonionMatMulTensor uses (the real parts agree, the imaginary parts do not). So the product is built
            // from the legacy table: out[b, o, k] = Σᵢ Σ_pq in[b, i, p]·w[o, i, q]·M[p, q, k].
            var e = _e;
            var lhs = e.Reshape(input, new[] { batch, 1, features, 8, 1 });
            var rhs = e.Reshape(weights, new[] { 1, outputs, features, 1, 8 });
            var pairs = e.ReduceSum(e.TensorMultiply(lhs, rhs), new[] { 2 }, false);
            var product = e.Reshape(e.TensorMatMul(e.Reshape(pairs, new[] { batch * outputs, 64 }), OctonionTable()), new[] { batch, outputs, 8 });
            if (x.Length > 2) product = e.TensorAdd(product, Shaped(x[2], new[] { outputs, 8 }));
            return e.Reshape(product, new[] { batch, outputs * 8 });
        }

        /// <summary>
        /// The legacy OctonionMultiplyComponents as a [64, 8] matrix: row 8p + q, column k holds the sign of a[p]·b[q]
        /// in component k. Each string lists component k's eight terms as sign, p, q.
        /// </summary>
        private Tensor<T> OctonionTable()
        {
            string[] terms =
            {
                "+00-11-22-33-44-55-66-77",
                "+01+10-23-32-45+54+67+76",
                "+02+20+13-31-46+64-57+75",
                "+03+30-12-21+47-74+56+65",
                "+04+40+15+51+26-62-37+73",
                "+05+50+14-41+27-72+36-63",
                "+06+60+17-71+24+42-35-53",
                "+07+70+16+61-25-52-34+43",
            };
            var table = new Tensor<T>(new[] { 64, 8 });
            for (int k = 0; k < 8; k++)
            {
                for (int t = 0; t < 8; t++)
                {
                    var term = terms[k].Substring(3 * t, 3);
                    int p = term[1] - '0', q = term[2] - '0';
                    table[8 * p + q, k] = term[0] == '+' ? _ops.One : _ops.Negate(_ops.One);
                }
            }
            return table;
        }

        // ── Poincaré ball (legacy formulas, row-wise over the last axis) ──

        private Tensor<T> Atanh(Tensor<T> t) => _e.TensorAtanh(_e.TensorClamp(t, C(-0.999999), C(0.999999)));

        private Tensor<T> Norm2(Tensor<T> t) => SumLast(_e.TensorSquare(t));

        private Tensor<T> Expand(Tensor<T> perRow, Tensor<T> like) => _e.TensorBroadcastTo(perRow, like.Shape.ToArray());

        /// <summary>Möbius addition x ⊕ y with the legacy denominator floor.</summary>
        private Tensor<T> Mobius(Tensor<T> x, Tensor<T> y, double c, double floor)
        {
            var e = _e;
            var dot = SumLast(e.TensorMultiply(x, y));
            var nx = Norm2(x);
            var ny = Norm2(y);
            var numX = Shift(e.TensorAdd(Scale(dot, 2 * c), Scale(ny, c)), 1.0);
            var numY = Shift(Scale(nx, -c), 1.0);
            var den = e.TensorMax(Shift(e.TensorAdd(Scale(dot, 2 * c), Scale(e.TensorMultiply(nx, ny), c * c)), 1.0), C(floor));
            return e.TensorDivide(e.TensorAdd(e.TensorMultiply(numX, x), e.TensorMultiply(numY, y)), den);
        }

        /// <summary>Rescales rows whose norm reaches 1/√c - ε back inside the ball (legacy PoincareProject).</summary>
        private Tensor<T> Project(Tensor<T> x, double c, double epsilon)
        {
            var e = _e;
            double maxRadius = 1.0 / Math.Sqrt(c) - epsilon;
            var norm = e.TensorSqrt(Norm2(x));
            var shrink = Scale(e.TensorReciprocal(Shift(norm, epsilon)), maxRadius);
            var scale = e.TensorWhere(e.TensorLessThan(norm, C(maxRadius)), e.TensorOnesLike(norm), shrink);
            return e.TensorMultiply(x, scale);
        }

        private Tensor<T> PoincareExpMap(Tensor<T> p, Tensor<T> v, double c)
        {
            var e = _e;
            double sqrtC = Math.Sqrt(c);
            var np = Norm2(p);
            var normV = e.TensorSqrt(Norm2(v));
            var safeV = e.TensorMax(normV, C(1e-10));
            var lambda = Scale(e.TensorReciprocal(e.TensorMax(Shift(Scale(np, -c), 1.0), C(1e-10))), 2.0);
            // tanh(√c·λ‖v‖/2) / (√c‖v‖)
            var scale = e.TensorDivide(e.Tanh(Scale(e.TensorMultiply(lambda, normV), sqrtC / 2.0)), Scale(safeV, sqrtC));
            var sv = e.TensorMultiply(v, scale);
            var mapped = Mobius(p, sv, c, 1e-10);
            return e.TensorWhere(Expand(e.TensorLessThan(normV, C(1e-10)), p), p, mapped);
        }

        private Tensor<T> PoincareLogMap(Tensor<T> p, Tensor<T> q, double c)
        {
            var e = _e;
            double sqrtC = Math.Sqrt(c);
            var diff = Mobius(e.TensorNegate(p), q, c, 1e-10);
            var normDiff = e.TensorSqrt(Norm2(diff));
            var safe = e.TensorMax(normDiff, C(1e-10));
            var lambda = Scale(e.TensorReciprocal(e.TensorMax(Shift(Scale(Norm2(p), -c), 1.0), C(1e-10))), 2.0);
            var arg = e.TensorMin(Scale(normDiff, sqrtC), C(1.0 - 1e-10));
            // (2 / (√c·λ)) · atanh(arg) / ‖diff‖
            var scale = e.TensorDivide(Scale(Atanh(arg), 2.0 / sqrtC), e.TensorMultiply(lambda, safe));
            var mapped = e.TensorMultiply(diff, scale);
            return e.TensorWhere(Expand(e.TensorLessThan(normDiff, C(1e-10)), p), e.TensorZerosLike(p), mapped);
        }

        private Tensor<T> PoincareDistance(Tensor<T> x, Tensor<T> y, double c)
        {
            var e = _e;
            double sqrtC = Math.Sqrt(c);
            var diff = Mobius(e.TensorNegate(x), y, c, 1e-10);
            var arg = e.TensorMin(Scale(e.TensorSqrt(Norm2(diff)), sqrtC), C(1.0 - 1e-10));
            var distance = Scale(Atanh(arg), 2.0 / sqrtC);
            return e.Reshape(distance, x.Rank > 1 ? new[] { x.Shape[0] } : new[] { 1 });
        }

        private Tensor<T> HyperbolicLinear(Tensor<T>[] x, double c)
        {
            // Legacy (not the engine's HypLinear): per output o, the distance from the origin of
            // proj(input) ⊕ w_o·tanh(√c‖w_o‖/2)/(√c‖w_o‖) [⊕ proj(bias_o)], all with denominator floor 1e-5.
            var e = _e;
            const double epsilon = 1e-5;
            double sqrtC = Math.Sqrt(c);
            Tensor<T> input = x[0], weights = x[1];
            if (input.Rank != 2) throw new ArgumentException("HyperbolicLinear input must have shape [batch, inputFeatures].");
            int batch = input.Shape[0], features = input.Shape[1], outputs = weights.Shape[0];

            var projected = e.Reshape(Project(input, c, epsilon), new[] { batch, 1, features });
            var normW = e.TensorSqrt(Norm2(weights));
            var weightScale = e.TensorWhere(e.TensorGreaterThan(normW, C(epsilon)),
                e.TensorDivide(e.Tanh(Scale(normW, sqrtC / 2.0)), Scale(e.TensorMax(normW, C(epsilon)), sqrtC)),
                e.TensorMultiplyScalar(e.TensorOnesLike(normW), C(0.5)));
            var scaledW = e.Reshape(e.TensorMultiply(weights, weightScale), new[] { 1, outputs, features });

            var lhs = e.TensorBroadcastTo(projected, new[] { batch, outputs, features });
            var rhs = e.TensorBroadcastTo(scaledW, new[] { batch, outputs, features });
            var result = Mobius(lhs, rhs, c, epsilon);
            if (x.Length > 2)
            {
                var bias = e.Reshape(Project(x[2], c, epsilon), new[] { 1, outputs, features });
                result = Mobius(result, e.TensorBroadcastTo(bias, new[] { batch, outputs, features }), c, epsilon);
            }

            var arg = e.TensorMin(Scale(e.TensorSqrt(Norm2(result)), sqrtC), C(1.0 - epsilon));
            return e.Reshape(Scale(Atanh(arg), 2.0 / sqrtC), new[] { batch, outputs });
        }
    }

    /// <summary>Marks the failures this translator reports, so a nested one is not wrapped twice.</summary>
    private static readonly object OwnMarker = new object();

    private static InvalidOperationException Bad(string layerName, string key, string why, Exception? inner = null)
    {
        var ex = new InvalidOperationException(
            $"Cannot rebuild {layerName}: constructor parameter '{key}' was saved as a legacy traced graph, but {why}", inner);
        ex.Data[OwnMarker] = true;
        return ex;
    }

    private static bool IsOwn(Exception ex) => ex is InvalidOperationException && ex.Data.Contains(OwnMarker);
}

using AiDotNet.Agentic.Models.Local;

namespace AiDotNet.ModelLoading;

/// <summary>
/// The named tensors of a PyTorch state dict that a model's loader copies into its own layers.
/// </summary>
/// <remarks>
/// <para>A loader asks for a tensor by name and gives the destination: <see cref="CopyTo{T}"/> writes the stored values
/// in order, <see cref="CopyTransposedTo{T}"/> writes a stored <c>[rows, cols]</c> matrix as <c>[cols, rows]</c> (a torch
/// <c>nn.Linear</c> weight into an <c>[in, out]</c> layer). A safetensors checkpoint is decoded chunk by chunk through a
/// small reused buffer straight into that storage, so loading a model costs its own size and nothing the size of a tensor
/// is allocated per weight — the <c>double[]</c> per tensor of <see cref="Read"/> peaked at about 2.8× a 520M-parameter
/// model's size while the runtime caught up with the garbage.</para>
/// <para><see cref="Read"/> remains for loaders that compute on values (splitting a fused projection, scaling); it
/// allocates.</para>
/// </remarks>
internal abstract class TorchTensorSource
{
    /// <summary>Whether a tensor of this name exists.</summary>
    public abstract bool Contains(string name);

    /// <summary>Decodes the named tensor in row-major order, chunk by chunk.</summary>
    /// <param name="name">The tensor's name.</param>
    /// <param name="shape">The shape the loader expects (empty: unchecked); a source with shapes checks it.</param>
    /// <param name="sink">Receives each run of values and the index of its first element.</param>
    /// <returns>The number of values decoded.</returns>
    protected abstract long Decode(string name, int[] shape, SafetensorsFile.DecodedChunk sink);

    /// <summary>A short description of the source for error messages.</summary>
    protected abstract string Description { get; }

    /// <summary>The named tensor's stored shape, or null when absent or when the source does not record shapes.</summary>
    public virtual long[]? ShapeOf(string name) => null;

    private static long Count(int[] shape) => shape.Aggregate(1L, (a, b) => a * b);

    private InvalidDataException Missing(string name) => new($"{Description} has no tensor '{name}'.");

    private void CheckCount(string name, int[] shape, long count)
    {
        if (shape.Length > 0 && count != Count(shape))
            throw new InvalidDataException($"'{name}' has {count} values in {Description} but [{string.Join(", ", shape)}] in this model; "
                + "build the model with the checkpoint's configuration.");
    }

    /// <summary>Copies the named tensor, as stored, into <paramref name="destination"/> from <paramref name="offset"/>.</summary>
    public void CopyTo<T>(string name, int[] shape, Tensor<T> destination, int offset = 0)
    {
        if (!Contains(name)) throw Missing(name);
        var ops = MathHelper.GetNumericOperations<T>();
        long expected = shape.Length > 0 ? Count(shape) : destination.Length - offset;
        if (offset < 0 || offset + expected > destination.Length)
            throw new ArgumentException($"'{name}' has {expected} values; the destination holds {destination.Length - offset} from offset {offset}.");
        long count = Decode(name, shape, (start, values) =>
        {
            if (offset + start + values.Length > destination.Length)
                throw new InvalidDataException($"'{name}' in {Description} has more values than its destination holds.");
            var span = destination.AsWritableSpan();
            for (int i = 0; i < values.Length; i++) Store(span, offset + start + i, values[i], ops);
        });
        CheckCount(name, shape, count);
        if (shape.Length == 0 && count != expected)
            throw new InvalidDataException($"'{name}' has {count} values in {Description}; the destination expects {expected}.");
    }

    /// <summary>Copies a stored <c>[rows, cols]</c> matrix into <paramref name="destination"/> as <c>[cols, rows]</c>.</summary>
    public void CopyTransposedTo<T>(string name, int rows, int cols, Tensor<T> destination)
    {
        if (!Contains(name)) throw Missing(name);
        if (destination.Length != (long)rows * cols)
            throw new ArgumentException($"'{name}' is [{rows}, {cols}]; the destination holds {destination.Length} values.");
        var ops = MathHelper.GetNumericOperations<T>();
        long count = Decode(name, new[] { rows, cols }, (start, values) =>
        {
            var span = destination.AsWritableSpan();
            // Element k of the stored matrix is (k / cols, k % cols); step the row and column instead of dividing.
            int r = start / cols, c = start % cols;
            for (int i = 0; i < values.Length; i++)
            {
                Store(span, c * rows + r, values[i], ops);
                if (++c == cols)
                {
                    c = 0;
                    r++;
                }
            }
        });
        CheckCount(name, new[] { rows, cols }, count);
    }

    /// <summary>Decodes the named tensor in stored order, chunk by chunk, into <paramref name="sink"/> (each run of values and
    /// the index of its first element), for a loader that scatters a stored layout into its storage itself; the count is
    /// checked against <paramref name="shape"/> when one is given.</summary>
    /// <returns>The number of values decoded.</returns>
    public long ReadChunks(string name, int[] shape, SafetensorsFile.DecodedChunk sink)
    {
        if (!Contains(name)) throw Missing(name);
        long count = Decode(name, shape, sink);
        CheckCount(name, shape, count);
        return count;
    }

    // A float or double destination is written directly; other numeric types go through their operations.
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static void Store<T>(Span<T> span, int index, double value, INumericOperations<T> ops)
    {
        if (typeof(T) == typeof(float))
        {
            float single = (float)value;
            span[index] = System.Runtime.CompilerServices.Unsafe.As<float, T>(ref single);
        }
        else if (typeof(T) == typeof(double))
        {
            span[index] = System.Runtime.CompilerServices.Unsafe.As<double, T>(ref value);
        }
        else
        {
            span[index] = ops.FromDouble(value);
        }
    }

    /// <summary>The named tensor's values (allocates; for loaders that compute on them), its count checked against
    /// <paramref name="shape"/> when one is given.</summary>
    public double[] Read(string name, int[] shape) => TryRead(name, shape) ?? throw Missing(name);

    /// <summary>The named tensor's values, or null when it is absent.</summary>
    public virtual double[]? TryRead(string name, int[]? shape = null)
    {
        if (!Contains(name)) return null;
        shape ??= Array.Empty<int>();
        var chunks = new List<double[]>();
        long count = Decode(name, shape, (_, values) => chunks.Add(values.ToArray()));
        CheckCount(name, shape, count);
        if (chunks.Count == 1) return chunks[0];
        var all = new double[count];
        int at = 0;
        foreach (var chunk in chunks)
        {
            Array.Copy(chunk, 0, all, at, chunk.Length);
            at += chunk.Length;
        }
        return all;
    }

    /// <summary>The same tensors with <paramref name="prefix"/> put before every name.</summary>
    public TorchTensorSource WithPrefix(string prefix) => prefix.Length == 0 ? this : new Renamed(this, name => prefix + name);

    /// <summary>The same tensors, each name the loader asks for mapped by <paramref name="map"/> to the stored one.</summary>
    public TorchTensorSource WithNames(Func<string, string> map) => new Renamed(this, map);

    /// <summary>These tensors, and <paramref name="fallback"/>'s for the names this source lacks (adapters a merged
    /// checkpoint does not ship, generated as the reference initializes them).</summary>
    public TorchTensorSource Or(TorchTensorSource fallback) => new Either(this, fallback);

    /// <summary>A source over an already loaded state dict (a <see cref="ComputerVision.Weights.WeightLoader"/> result);
    /// prefer <see cref="TorchCheckpointFile.Open"/>, which does not hold the whole checkpoint in memory.</summary>
    public static TorchTensorSource From(IReadOnlyDictionary<string, Tensor<float>> state, string description = "the state dict") =>
        new FromState(state, description);

    /// <summary>A source over a function that returns each tensor's values (test fixtures, converted dictionaries); a
    /// function that throws for an absent name is treated as having every name.</summary>
    public static TorchTensorSource From(Func<string, int[], double[]> read) => new FromFunction((name, shape) => read(name, shape), allNamesPresent: true);

    /// <summary>A source over a function that returns each tensor's values, or null when absent.</summary>
    public static TorchTensorSource From(Func<string, double[]?> read) => new FromFunction((name, _) => read(name), allNamesPresent: false);

    /// <summary>Fills one run of a generated tensor: its name and shape, the index of the run's first element, and the
    /// values to write.</summary>
    public delegate void Fill(string name, int[] shape, long offset, Span<double> values);

    /// <summary>A source whose tensors are generated run by run (random initialization), so initializing a model costs
    /// nothing the size of a tensor; every name exists, and the shape a loader asks for is the tensor's.</summary>
    public static TorchTensorSource Generated(Fill fill) => new Generator(fill);

    /// <summary>The same tensors read through the allocating form, for code that still takes
    /// <c>Func&lt;string, int[], double[]&gt;</c>.</summary>
    public Func<string, int[], double[]> AsFunction() => Read;

    private sealed class Generator : TorchTensorSource
    {
        private readonly Fill _fill;
        public Generator(Fill fill) { _fill = fill; }
        protected override string Description => "the initializer";
        public override bool Contains(string name) => true;
        protected override long Decode(string name, int[] shape, SafetensorsFile.DecodedChunk sink)
        {
            if (shape.Length == 0) throw new ArgumentException($"A generated tensor needs its shape ('{name}').", nameof(shape));
            long total = Count(shape);
            const int Chunk = 1 << 16;
            var buffer = System.Buffers.ArrayPool<double>.Shared.Rent((int)Math.Min(Chunk, Math.Max(1, total)));
            try
            {
                for (long done = 0; done < total;)
                {
                    int n = (int)Math.Min(Chunk, total - done);
                    var span = new Span<double>(buffer, 0, n);
                    _fill(name, shape, done, span);
                    sink(checked((int)done), span);
                    done += n;
                }
            }
            finally
            {
                System.Buffers.ArrayPool<double>.Shared.Return(buffer);
            }
            return total;
        }
    }

    private sealed class Renamed : TorchTensorSource
    {
        private readonly TorchTensorSource _inner;
        private readonly Func<string, string> _map;
        public Renamed(TorchTensorSource inner, Func<string, string> map) { _inner = inner; _map = map; }
        protected override string Description => _inner.Description;
        public override bool Contains(string name) => _inner.Contains(_map(name));
        public override long[]? ShapeOf(string name) => _inner.ShapeOf(_map(name));
        protected override long Decode(string name, int[] shape, SafetensorsFile.DecodedChunk sink) => _inner.Decode(_map(name), shape, sink);
        public override double[]? TryRead(string name, int[]? shape = null) => _inner.TryRead(_map(name), shape);
    }

    private sealed class Either : TorchTensorSource
    {
        private readonly TorchTensorSource _first;
        private readonly TorchTensorSource _second;
        public Either(TorchTensorSource first, TorchTensorSource second) { _first = first; _second = second; }
        private TorchTensorSource For(string name) => _first.Contains(name) ? _first : _second;
        protected override string Description => _first.Description;
        public override bool Contains(string name) => _first.Contains(name) || _second.Contains(name);
        public override long[]? ShapeOf(string name) => For(name).ShapeOf(name);
        protected override long Decode(string name, int[] shape, SafetensorsFile.DecodedChunk sink) => For(name).Decode(name, shape, sink);
        public override double[]? TryRead(string name, int[]? shape = null) => For(name).TryRead(name, shape);
    }

    /// <summary>Decodes a float tensor's values in chunks through a pooled buffer.</summary>
    private protected static long DecodeFloats(ReadOnlySpan<float> values, SafetensorsFile.DecodedChunk sink)
    {
        const int Chunk = 1 << 16;
        var buffer = System.Buffers.ArrayPool<double>.Shared.Rent(Math.Min(Chunk, Math.Max(1, values.Length)));
        try
        {
            for (int start = 0; start < values.Length; start += Chunk)
            {
                int n = Math.Min(Chunk, values.Length - start);
                for (int i = 0; i < n; i++) buffer[i] = values[start + i];
                sink(start, new ReadOnlySpan<double>(buffer, 0, n));
            }
        }
        finally
        {
            System.Buffers.ArrayPool<double>.Shared.Return(buffer);
        }
        return values.Length;
    }

    private sealed class FromState : TorchTensorSource
    {
        private readonly IReadOnlyDictionary<string, Tensor<float>> _state;
        private readonly string _description;
        public FromState(IReadOnlyDictionary<string, Tensor<float>> state, string description) { _state = state; _description = description; }
        protected override string Description => _description;
        public override bool Contains(string name) => _state.ContainsKey(name);
        public override long[]? ShapeOf(string name) => _state.TryGetValue(name, out var t) ? t.Shape.ToArray().Select(d => (long)d).ToArray() : null;
        protected override long Decode(string name, int[] shape, SafetensorsFile.DecodedChunk sink) =>
            DecodeFloats((_state.TryGetValue(name, out var t) ? t : throw Missing(name)).AsSpan(), sink);
    }

    private sealed class FromFunction : TorchTensorSource
    {
        private readonly Func<string, int[], double[]?> _read;
        private readonly bool _allNamesPresent;
        public FromFunction(Func<string, int[], double[]?> read, bool allNamesPresent) { _read = read; _allNamesPresent = allNamesPresent; }
        protected override string Description => "the weight source";
        // A throwing function answers absence by throwing on the read itself; the null-returning form is probed.
        public override bool Contains(string name) => _allNamesPresent || _read(name, Array.Empty<int>()) is not null;
        public override double[]? TryRead(string name, int[]? shape = null) => _read(name, shape ?? Array.Empty<int>());
        protected override long Decode(string name, int[] shape, SafetensorsFile.DecodedChunk sink)
        {
            var values = _read(name, shape) ?? throw new InvalidDataException($"The weight source has no tensor '{name}'.");
            sink(0, values);
            return values.Length;
        }
    }
}

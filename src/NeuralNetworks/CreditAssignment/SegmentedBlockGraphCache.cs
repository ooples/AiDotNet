using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using AiDotNet.Configuration;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;

namespace AiDotNet.NeuralNetworks.CreditAssignment;

/// <summary>
/// Per-layer compiled inference block cache used by <see cref="LocalObjectiveGradientComputer{T}"/> and segmented
/// checkpointing so repeated block evaluations across <c>SizeClass</c> sequence/batch buckets reuse compiled plans
/// instead of re-dispatching eager C# layer graphs every step.
/// </summary>
internal static class SegmentedBlockGraphCache<T>
{
    private sealed class LayerBlockState : IDisposable
    {
        public readonly object Gate = new();
        public IEngine? Engine;
        public int ParameterEpoch = -1;
        public CompiledModelCache<T>? Cache;
        public readonly List<(int[] Shape, Tensor<T> InputLeaf)> InputLeaves = new();

        public void Reset()
        {
            Cache?.Dispose();
            Cache = null;
            InputLeaves.Clear();
            Engine = null;
            ParameterEpoch = -1;
        }

        public void Dispose() => Reset();
    }

    private static readonly ConditionalWeakTable<ILayer<T>, LayerBlockState> s_layerCaches = new();

    /// <summary>
    /// Evaluates <paramref name="layer"/> under <see cref="NoGradScope{T}"/> using a cached compiled block plan when
    /// <see cref="CompilationRuntimeSettings.SegmentedBlockGraphsEnabled"/> is active and no outer gradient tape or
    /// graph capture is recording; falls back transparently to eager <c>layer.Forward(input)</c> if compilation fails.
    /// </summary>
    internal static Tensor<T> ForwardDetachedBlock(ILayer<T> layer, Tensor<T> input)
    {
        if (layer is null) throw new ArgumentNullException(nameof(layer));
        if (input is null) throw new ArgumentNullException(nameof(input));

        IEngine engine = AiDotNetEngine.Current;
        if (!CompilationRuntimeSettings.SegmentedBlockGraphsEnabled
            || GradientTape<T>.Current is not null
            || GradientRecording.IsRecording<T>()
            || GraphMode.IsActive
            || GraphCaptureCompatibility.IsActive
            || !ReferenceEquals(engine, engine.Unwrap())
            || layer is Layers.LayerBase<T> { IsInitialized: false })
        {
            return layer.Forward(input);
        }

        var state = s_layerCaches.GetOrCreateValue(layer);
        int currentEpoch = Layers.LayerBase<T>.ParameterSurfaceEpoch;

        lock (state.Gate)
        {
            try
            {
                if (!ReferenceEquals(state.Engine, engine) || state.ParameterEpoch != currentEpoch)
                {
                    state.Reset();
                    state.Engine = engine;
                    state.ParameterEpoch = currentEpoch;
                }

                var cache = state.Cache ??= new CompiledModelCache<T>();
                Tensor<T> leaf = GetOrCreateInputLeaf(state, input._shape);
                CopyInputIntoLeaf(engine, input, leaf);

                var plan = cache.GetOrCompileInference(
                    input._shape,
                    () => layer.Forward(leaf));
                return plan.Execute();
            }
            catch (Exception)
            {
                if (CompilationRuntimeSettings.ThrowOnCompilationFailure)
                    throw;
                return layer.Forward(input);
            }
        }
    }

    /// <summary>Invalidates any cached compiled block plans for <paramref name="layer"/>.</summary>
    internal static void Invalidate(ILayer<T> layer)
    {
        if (layer is not null && s_layerCaches.TryGetValue(layer, out var state))
        {
            lock (state.Gate)
            {
                state.Reset();
            }
        }
    }

    private static Tensor<T> GetOrCreateInputLeaf(LayerBlockState state, int[] shape)
    {
        for (int i = 0; i < state.InputLeaves.Count; i++)
        {
            if (ShapesMatch(state.InputLeaves[i].Shape, shape))
                return state.InputLeaves[i].InputLeaf;
        }

        int maxBuckets = CompilationRuntimeSettings.MultiBucketShapeRoutingEnabled ? 24 : 4;
        if (state.InputLeaves.Count >= maxBuckets)
        {
            state.Cache?.Invalidate();
            state.InputLeaves.Clear();
        }

        var copyShape = (int[])shape.Clone();
        var leaf = TensorScope.Detach(new Tensor<T>(copyShape));
        state.InputLeaves.Add((copyShape, leaf));
        return leaf;
    }

    private static void CopyInputIntoLeaf(IEngine engine, Tensor<T> source, Tensor<T> destination)
    {
        var contiguous = source.IsContiguous ? source : source.Contiguous();
        engine.TensorCopy(contiguous, destination);
    }

    private static bool ShapesMatch(int[] left, int[] right)
    {
        if (left.Length != right.Length) return false;
        for (int i = 0; i < left.Length; i++)
        {
            if (left[i] != right[i]) return false;
        }
        return true;
    }
}

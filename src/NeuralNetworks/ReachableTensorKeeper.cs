using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq.Expressions;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Takes every tensor a model still holds after a forward out of the open <see cref="TensorScope"/>s, which free the rest.
/// </summary>
/// <remarks>
/// <para>
/// A scope frees each GPU result created inside it unless it is kept. A layer that builds a cache on its first forward
/// (LSTMLayer's stacked gate weights, say) creates it inside the scope, so the scope freed it and the next call silently
/// ran on a released tensor. PyTorch frees an intermediate when nothing references it; this gives the scope the same
/// rule for references held by the model: whatever the model graph still reaches survives every open scope, everything
/// else is freed.
/// </para>
/// <para>
/// The walk follows instance fields of AiDotNet and AiDotNet.Tensors types, arrays, and System collections. It stops
/// at engines (their caches are scope-suspended already), delegates, strings and primitives. The field getters of each
/// type are compiled once, so a walk costs a few microseconds per thousand objects.
/// </para>
/// </remarks>
internal static class ReachableTensorKeeper
{
    private static readonly ConcurrentDictionary<Type, Func<object, object?>[]> s_fieldGetters = new();
    private static readonly ConcurrentDictionary<Type, Action<object>?> s_keepers = new();
    private static readonly ConcurrentDictionary<Type, bool> s_mayHoldTensor = new();
    private static readonly Assembly s_modelAssembly = typeof(ReachableTensorKeeper).Assembly;
    private static readonly Assembly s_tensorsAssembly = typeof(TensorScope).Assembly;
    private static readonly MethodInfo s_detachAs = typeof(ReachableTensorKeeper).GetMethod(nameof(DetachAs), BindingFlags.NonPublic | BindingFlags.Static)!;

    [ThreadStatic] private static HashSet<object>? t_visited;
    [ThreadStatic] private static Stack<object>? t_pending;

    /// <summary>
    /// Detaches every tensor reachable from <paramref name="root"/> from all the scopes open on this thread
    /// (<see cref="TensorScope.Detach{T}"/>): a model's cache outlives a caller's scope around the call too, where
    /// <see cref="TensorScope.Keep{T}"/> would hand it to that scope to free.
    /// </summary>
    internal static void DetachReachable(object root) => VisitReachable(root, static (tensor, detach) => detach(tensor));

    /// <summary>
    /// Calls <paramref name="onTensor"/> with every tensor reachable from <paramref name="root"/> and a delegate that
    /// passes it to <see cref="TensorScope.Detach{T}"/>.
    /// </summary>
    internal static void VisitReachable(object root, Action<object, Action<object>> onTensor)
    {
        var visited = t_visited ??= new HashSet<object>(Tensors.Engines.Autodiff.ReferenceEqualityComparer<object>.Instance);
        var pending = t_pending ??= new Stack<object>();
        try
        {
            Push(root, visited, pending);
            while (pending.Count > 0)
            {
                var item = pending.Pop();
                var type = item.GetType();
                if (Keeper(type) is { } keep)
                {
                    onTensor(item, keep);
                    continue;
                }
                if (item is Array array)
                {
                    if (MayHoldTensor(type.GetElementType()!))
                        foreach (var element in array) Push(element, visited, pending);
                    continue;
                }
                if (IsSystemCollection(type))
                {
                    if (item is IDictionary dictionary)
                        foreach (DictionaryEntry entry in dictionary) Push(entry.Value, visited, pending);
                    else
                        foreach (var element in (IEnumerable)item) Push(element, visited, pending);
                    continue;
                }
                foreach (var get in FieldGetters(type)) Push(get(item), visited, pending);
            }
        }
        finally
        {
            visited.Clear();
            pending.Clear();
        }
    }

    private static void Push(object? value, HashSet<object> visited, Stack<object> pending)
    {
        if (value is null || !MayHoldTensor(value.GetType()) || !visited.Add(value)) return;
        pending.Push(value);
    }

    private static void DetachAs<TElement>(object tensor) => TensorScope.Detach((Tensor<TElement>)tensor);

    // The Detach<TElement> of a Tensor<TElement> (or a subclass of one); null for any other type.
    private static Action<object>? Keeper(Type type) => s_keepers.GetOrAdd(type, static t =>
    {
        for (var current = t; current is not null; current = current.BaseType)
            if (current.IsGenericType && current.GetGenericTypeDefinition() == typeof(Tensor<>))
                return (Action<object>)s_detachAs.MakeGenericMethod(current.GetGenericArguments()[0])
                    .CreateDelegate(typeof(Action<object>));
        return null;
    });

    // Whether a value of this runtime (or declared) type can be, or lead to, a tensor.
    private static bool MayHoldTensor(Type type) => s_mayHoldTensor.GetOrAdd(type, static t =>
    {
        if (t.IsPrimitive || t.IsEnum || t.IsPointer || t == typeof(string) || t == typeof(decimal)) return false;
        if (typeof(Delegate).IsAssignableFrom(t) || typeof(MemberInfo).IsAssignableFrom(t)) return false;
        if (typeof(IEngine).IsAssignableFrom(t)) return false;
        if (t.IsArray) return MayHoldTensor(t.GetElementType()!);
        if (t == typeof(object) || t.IsInterface || t.IsAbstract) return true;
        if (Keeper(t) is not null) return true;
        if (t.Assembly == s_modelAssembly) return true;
        // Of the tensor library, only its data types (a Vector or Matrix a layer holds): its backends, buffer pools,
        // compiled plans and arenas hold no layer state, and walking their pools would cost more than the forward.
        if (t.Assembly == s_tensorsAssembly)
            return t.Namespace is { } tensorsNs && tensorsNs.StartsWith("AiDotNet.Tensors.LinearAlgebra", StringComparison.Ordinal);
        if (IsSystemCollection(t))
            return !t.IsGenericType || Array.Exists(t.GetGenericArguments(), MayHoldTensor);
        // A tuple, KeyValuePair or Lazy holding a tensor-bearing type.
        return t.IsGenericType && (t.Namespace == "System" || t.Namespace == "System.Collections.Generic")
            && Array.Exists(t.GetGenericArguments(), MayHoldTensor);
    });

    private static bool IsSystemCollection(Type type) =>
        type.Namespace is { } ns && ns.StartsWith("System.Collections", StringComparison.Ordinal) && typeof(IEnumerable).IsAssignableFrom(type);

    private static Func<object, object?>[] FieldGetters(Type type) => s_fieldGetters.GetOrAdd(type, static t =>
    {
        var getters = new List<Func<object, object?>>();
        for (var current = t; current is not null && current != typeof(object); current = current.BaseType)
        {
            foreach (var field in current.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
            {
                if (!MayHoldTensor(field.FieldType)) continue;
                var instance = Expression.Parameter(typeof(object), "instance");
                var read = Expression.Field(Expression.Convert(instance, current), field);
                getters.Add(Expression.Lambda<Func<object, object?>>(Expression.Convert(read, typeof(object)), instance).Compile());
            }
        }
        return getters.ToArray();
    });
}

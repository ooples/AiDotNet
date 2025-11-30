using System.Diagnostics;

namespace AiDotNet.Diagnostics;

/// <summary>
/// A scoped profiler that automatically records duration when disposed.
/// </summary>
/// <remarks>
/// <para><b>Usage:</b>
/// Use with 'using' statement for automatic timing:
/// <code>
/// using (var scope = new ProfilerScope("MyOperation"))
/// {
///     // Code to profile
/// }
/// // Duration is automatically recorded when scope exits
/// </code>
/// </para>
/// <para>
/// Supports nested scopes for hierarchical profiling:
/// <code>
/// using (Profiler.Scope("Training"))
/// {
///     using (Profiler.Scope("Forward"))
///     {
///         model.Forward(input);
///     }
///     using (Profiler.Scope("Backward"))
///     {
///         model.Backward(gradient);
///     }
/// }
/// </code>
/// </para>
/// </remarks>
public readonly struct ProfilerScope : IDisposable
{
    private readonly string _name;
    private readonly Stopwatch _stopwatch;
    private readonly string? _parentName;
    private readonly long _memoryBefore;
    private readonly bool _trackMemory;

    /// <summary>
    /// Creates a new profiler scope.
    /// </summary>
    /// <param name="name">Name of the operation being profiled.</param>
    /// <param name="trackMemory">Whether to track memory allocations.</param>
    public ProfilerScope(string name, bool trackMemory = false)
    {
        _name = name;
        _trackMemory = trackMemory;
        _stopwatch = Stopwatch.StartNew();

        // Get parent from call stack
        var stack = Profiler.GetCallStack();
        _parentName = stack.Count > 0 ? stack.Peek().Name : null;

        // Track memory if requested
        if (_trackMemory)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            _memoryBefore = GC.GetTotalMemory(false);
        }
        else
        {
            _memoryBefore = 0;
        }

        // Push a timer to the call stack for hierarchy tracking
        var timer = new ProfilerTimer(name);
        // Timer already pushes itself to the stack
    }

    /// <summary>
    /// Gets the name of this profiled operation.
    /// </summary>
    public string Name => _name;

    /// <summary>
    /// Gets the elapsed time so far.
    /// </summary>
    public TimeSpan Elapsed => _stopwatch.Elapsed;

    /// <summary>
    /// Stops the timer and records the duration.
    /// </summary>
    public void Dispose()
    {
        _stopwatch.Stop();

        // Record timing
        Profiler.RecordTiming(_name, _stopwatch.Elapsed, _parentName);

        // Record memory if tracking
        if (_trackMemory)
        {
            long memoryAfter = GC.GetTotalMemory(false);
            long allocated = memoryAfter - _memoryBefore;
            if (allocated > 0)
            {
                Profiler.RecordAllocation(_name, allocated);
            }
        }

        // Pop from call stack
        var stack = Profiler.GetCallStack();
        if (stack.Count > 0)
        {
            var timer = stack.Pop();
            timer.Stop();
        }
    }
}

/// <summary>
/// Provides extension methods for profiling common operations.
/// </summary>
public static class ProfilerExtensions
{
    /// <summary>
    /// Profiles an action with the given name.
    /// </summary>
    public static void Profile(this Action action, string name)
    {
        using (Profiler.Scope(name))
        {
            action();
        }
    }

    /// <summary>
    /// Profiles a function with the given name.
    /// </summary>
    public static T Profile<T>(this Func<T> func, string name)
    {
        using (Profiler.Scope(name))
        {
            return func();
        }
    }

    /// <summary>
    /// Profiles an async operation with the given name.
    /// </summary>
    public static async Task ProfileAsync(this Func<Task> func, string name)
    {
        using (Profiler.Scope(name))
        {
            await func();
        }
    }

    /// <summary>
    /// Profiles an async function with the given name.
    /// </summary>
    public static async Task<T> ProfileAsync<T>(this Func<Task<T>> func, string name)
    {
        using (Profiler.Scope(name))
        {
            return await func();
        }
    }
}

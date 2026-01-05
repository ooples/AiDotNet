// Copyright (c) AiDotNet. All rights reserved.
// OpenCL event implementation of IGpuEvent.

using System;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

/// <summary>
/// OpenCL implementation of a GPU event.
/// Wraps a cl_event handle for synchronization between command queues.
/// </summary>
public sealed class OpenClEvent : IGpuEvent
{
    private readonly OpenClBackend _backend;
    private IntPtr _handle;
    private bool _disposed;
    private readonly OpenClCommandQueue? _sourceQueue;
    private readonly bool _profilingEnabled;

    /// <inheritdoc/>
    public IntPtr Handle => _handle;

    /// <inheritdoc/>
    public bool IsRecorded => _handle != IntPtr.Zero;

    /// <inheritdoc/>
    public bool IsComplete
    {
        get
        {
            if (_disposed || _handle == IntPtr.Zero)
            {
                return false;
            }

            return Query();
        }
    }

    /// <summary>
    /// Creates an OpenCL event from an existing event handle.
    /// </summary>
    /// <param name="backend">The OpenCL backend.</param>
    /// <param name="handle">The existing event handle.</param>
    /// <param name="sourceQueue">The queue that created the event.</param>
    /// <param name="profilingEnabled">Whether profiling was enabled on the source queue.</param>
    internal OpenClEvent(OpenClBackend backend, IntPtr handle, OpenClCommandQueue? sourceQueue, bool profilingEnabled)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _handle = handle;
        _sourceQueue = sourceQueue;
        _profilingEnabled = profilingEnabled;

        // Retain the event to ensure it's not released prematurely
        if (handle != IntPtr.Zero)
        {
            OpenClNativeBindings.RetainEvent(handle);
        }
    }

    /// <summary>
    /// Records this event on a queue.
    /// In OpenCL, this creates a new marker event on the queue.
    /// </summary>
    /// <param name="stream">The queue to record on.</param>
    public void Record(IGpuStream stream)
    {
        ThrowIfDisposed();

        if (stream is not OpenClCommandQueue openClQueue)
        {
            throw new ArgumentException("Stream must be an OpenClCommandQueue", nameof(stream));
        }

        // Release the old event if any
        if (_handle != IntPtr.Zero)
        {
            OpenClNativeBindings.ReleaseEvent(_handle);
            _handle = IntPtr.Zero;
        }

        // Create a new marker event
        int err = OpenClNativeBindings.EnqueueMarkerWithWaitList(openClQueue.Handle, 0, null, out IntPtr newHandle);
        if (err != OpenClNativeBindings.CL_SUCCESS)
        {
            throw new InvalidOperationException($"clEnqueueMarkerWithWaitList failed: {err}");
        }

        _handle = newHandle;
    }

    /// <inheritdoc/>
    public void Synchronize()
    {
        ThrowIfDisposed();

        if (_handle == IntPtr.Zero)
        {
            return; // Nothing to synchronize
        }

        var eventArray = new IntPtr[] { _handle };
        int err = OpenClNativeBindings.WaitForEvents(1, eventArray);
        if (err != OpenClNativeBindings.CL_SUCCESS)
        {
            throw new InvalidOperationException($"clWaitForEvents failed: {err}");
        }
    }

    /// <inheritdoc/>
    public bool Query()
    {
        ThrowIfDisposed();

        if (_handle == IntPtr.Zero)
        {
            return false;
        }

        IntPtr statusPtr = Marshal.AllocHGlobal(sizeof(int));
        try
        {
            int err = OpenClNativeBindings.GetEventInfo(
                _handle,
                OpenClNativeBindings.CL_EVENT_COMMAND_EXECUTION_STATUS,
                (UIntPtr)sizeof(int),
                statusPtr,
                out _);

            if (err != OpenClNativeBindings.CL_SUCCESS)
            {
                return false;
            }

            int status = Marshal.ReadInt32(statusPtr);
            return status == OpenClNativeBindings.CL_COMPLETE;
        }
        finally
        {
            Marshal.FreeHGlobal(statusPtr);
        }
    }

    /// <inheritdoc/>
    public float GetElapsedTime(IGpuEvent startEvent)
    {
        ThrowIfDisposed();

        if (startEvent is not OpenClEvent openClStartEvent)
        {
            throw new ArgumentException("Event must be an OpenClEvent", nameof(startEvent));
        }

        if (!_profilingEnabled)
        {
            throw new InvalidOperationException("Event was created without profiling capability");
        }

        if (_handle == IntPtr.Zero || openClStartEvent.Handle == IntPtr.Zero)
        {
            throw new InvalidOperationException("Both events must be recorded");
        }

        // Ensure both events are complete
        Synchronize();
        openClStartEvent.Synchronize();

        // Get end timestamp
        ulong endTime = GetProfilingInfo(_handle, OpenClNativeBindings.CL_PROFILING_COMMAND_END);
        // Get start timestamp
        ulong startTime = GetProfilingInfo(openClStartEvent.Handle, OpenClNativeBindings.CL_PROFILING_COMMAND_START);

        // Convert nanoseconds to milliseconds
        return (endTime - startTime) / 1_000_000.0f;
    }

    private static ulong GetProfilingInfo(IntPtr eventHandle, uint infoParam)
    {
        IntPtr valuePtr = Marshal.AllocHGlobal(sizeof(ulong));
        try
        {
            int err = OpenClNativeBindings.GetEventProfilingInfo(
                eventHandle,
                infoParam,
                (UIntPtr)sizeof(ulong),
                valuePtr,
                out _);

            if (err != OpenClNativeBindings.CL_SUCCESS)
            {
                throw new InvalidOperationException($"clGetEventProfilingInfo failed: {err}");
            }

            return (ulong)Marshal.ReadInt64(valuePtr);
        }
        finally
        {
            Marshal.FreeHGlobal(valuePtr);
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(OpenClEvent));
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;

        if (_handle != IntPtr.Zero)
        {
            try
            {
                // Release the retained event
                OpenClNativeBindings.ReleaseEvent(_handle);
            }
            catch
            {
                // Ignore destruction errors during disposal
            }
        }

        _handle = IntPtr.Zero;
    }
}

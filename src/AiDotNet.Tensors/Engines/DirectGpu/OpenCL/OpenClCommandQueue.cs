// Copyright (c) AiDotNet. All rights reserved.
// OpenCL command queue implementation of IGpuStream.

using System;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

/// <summary>
/// OpenCL implementation of a GPU stream using command queues.
/// Wraps a cl_command_queue handle for concurrent kernel execution.
/// </summary>
public sealed class OpenClCommandQueue : IGpuStream
{
    private readonly OpenClBackend _backend;
    private IntPtr _handle;
    private readonly IntPtr _context;
    private readonly IntPtr _device;
    private bool _disposed;
    private readonly bool _ownsHandle;
    private readonly bool _profilingEnabled;

    /// <inheritdoc/>
    public IntPtr Handle => _handle;

    /// <inheritdoc/>
    public GpuStreamType StreamType { get; }

    /// <inheritdoc/>
    public bool IsDefault { get; }

    /// <inheritdoc/>
    public int Priority { get; }

    /// <summary>
    /// Gets whether this queue has profiling enabled.
    /// </summary>
    public bool IsProfilingEnabled => _profilingEnabled;

    /// <summary>
    /// Wraps an existing OpenCL command queue handle.
    /// </summary>
    /// <param name="backend">The OpenCL backend.</param>
    /// <param name="handle">The existing command queue handle.</param>
    /// <param name="context">The OpenCL context.</param>
    /// <param name="device">The OpenCL device.</param>
    /// <param name="streamType">The type of stream.</param>
    /// <param name="profilingEnabled">Whether profiling is enabled.</param>
    /// <param name="ownsHandle">Whether this wrapper owns the handle.</param>
    internal OpenClCommandQueue(OpenClBackend backend, IntPtr handle, IntPtr context, IntPtr device,
        GpuStreamType streamType, bool profilingEnabled, bool ownsHandle)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _handle = handle;
        _context = context;
        _device = device;
        StreamType = streamType;
        IsDefault = streamType == GpuStreamType.Default;
        Priority = 0;
        _profilingEnabled = profilingEnabled;
        _ownsHandle = ownsHandle;
    }

    /// <summary>
    /// Creates a new OpenCL command queue.
    /// </summary>
    /// <param name="backend">The OpenCL backend.</param>
    /// <param name="context">The OpenCL context.</param>
    /// <param name="device">The OpenCL device.</param>
    /// <param name="streamType">The type of stream to create.</param>
    /// <param name="enableProfiling">Whether to enable profiling on this queue.</param>
    public OpenClCommandQueue(OpenClBackend backend, IntPtr context, IntPtr device,
        GpuStreamType streamType, bool enableProfiling = false)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _context = context;
        _device = device;
        StreamType = streamType;
        IsDefault = false;
        Priority = 0; // OpenCL doesn't support queue priorities directly
        _profilingEnabled = enableProfiling;
        _ownsHandle = true;

        ulong properties = enableProfiling ? OpenClNativeBindings.CL_QUEUE_PROFILING_ENABLE : 0;
        _handle = OpenClNativeBindings.CreateCommandQueue(context, device, properties, out int err);

        if (err != OpenClNativeBindings.CL_SUCCESS || _handle == IntPtr.Zero)
        {
            throw new InvalidOperationException($"Failed to create OpenCL command queue: {err}");
        }
    }

    /// <inheritdoc/>
    public void Synchronize()
    {
        ThrowIfDisposed();

        int err = OpenClNativeBindings.Finish(_handle);
        if (err != OpenClNativeBindings.CL_SUCCESS)
        {
            throw new InvalidOperationException($"clFinish failed: {err}");
        }
    }

    /// <inheritdoc/>
    public IGpuEvent RecordEvent()
    {
        ThrowIfDisposed();

        // Create a marker event on this queue
        int err = OpenClNativeBindings.EnqueueMarkerWithWaitList(_handle, 0, null, out IntPtr eventHandle);
        if (err != OpenClNativeBindings.CL_SUCCESS)
        {
            throw new InvalidOperationException($"clEnqueueMarkerWithWaitList failed: {err}");
        }

        return new OpenClEvent(_backend, eventHandle, this, _profilingEnabled);
    }

    /// <inheritdoc/>
    public void WaitEvent(IGpuEvent gpuEvent)
    {
        ThrowIfDisposed();

        if (gpuEvent is not OpenClEvent openClEvent)
        {
            throw new ArgumentException("Event must be an OpenClEvent", nameof(gpuEvent));
        }

        // In OpenCL, to make a queue wait for an event, we enqueue a marker that depends on the event
        var eventArray = new IntPtr[] { openClEvent.Handle };
        int err = OpenClNativeBindings.EnqueueMarkerWithWaitList(_handle, 1, eventArray, out IntPtr markerEvent);
        if (err != OpenClNativeBindings.CL_SUCCESS)
        {
            throw new InvalidOperationException($"clEnqueueMarkerWithWaitList (wait) failed: {err}");
        }

        // Release the marker event immediately since we don't need to track it
        OpenClNativeBindings.ReleaseEvent(markerEvent);
    }

    /// <summary>
    /// Queries whether all operations submitted to the queue have completed.
    /// </summary>
    /// <returns>True if all operations are complete, false otherwise.</returns>
    public bool Query()
    {
        ThrowIfDisposed();

        // Create a marker and query its status
        int err = OpenClNativeBindings.EnqueueMarkerWithWaitList(_handle, 0, null, out IntPtr eventHandle);
        if (err != OpenClNativeBindings.CL_SUCCESS)
        {
            return false;
        }

        try
        {
            return QueryEventComplete(eventHandle);
        }
        finally
        {
            OpenClNativeBindings.ReleaseEvent(eventHandle);
        }
    }

    private static bool QueryEventComplete(IntPtr eventHandle)
    {
        IntPtr statusPtr = Marshal.AllocHGlobal(sizeof(int));
        try
        {
            int err = OpenClNativeBindings.GetEventInfo(
                eventHandle,
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

    /// <summary>
    /// Flushes all commands in the queue to the device.
    /// Unlike Synchronize, this does not wait for completion.
    /// </summary>
    public void Flush()
    {
        ThrowIfDisposed();

        int err = OpenClNativeBindings.Flush(_handle);
        if (err != OpenClNativeBindings.CL_SUCCESS)
        {
            throw new InvalidOperationException($"clFlush failed: {err}");
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(OpenClCommandQueue));
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

        if (_ownsHandle && _handle != IntPtr.Zero)
        {
            try
            {
                OpenClNativeBindings.ReleaseCommandQueue(_handle);
            }
            catch
            {
                // Ignore destruction errors during disposal
            }
        }

        _handle = IntPtr.Zero;
    }
}

// Copyright (c) AiDotNet. All rights reserved.
// Pure P/Invoke OpenCL program - no managed GPU runtime dependency.
// Works on ALL .NET versions including .NET Framework 4.6.2.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    /// <summary>
    /// OpenCL program wrapper using pure P/Invoke. No managed GPU runtime dependency.
    /// </summary>
    internal sealed class DirectOpenClProgram : IDisposable
    {
        private IntPtr _program;
        private readonly DirectOpenClContext _context;
        private bool _disposed;

        public IntPtr Handle => _program;

        public DirectOpenClProgram(DirectOpenClContext context, string source)
        {
            _context = context;

            var sources = new string[] { source };
            var lengths = new UIntPtr[] { (UIntPtr)source.Length };

            _program = OpenClNativeBindings.CreateProgramWithSource(
                context.Context,
                1,
                sources,
                lengths,
                out int err);

            if (err != OpenClNativeBindings.CL_SUCCESS || _program == IntPtr.Zero)
                throw new InvalidOperationException($"Failed to create OpenCL program: {err}");
        }

        /// <summary>
        /// Builds the program for the context's device.
        /// </summary>
        public void Build(string options = "")
        {
            var devices = new IntPtr[] { _context.Device };
            int err = OpenClNativeBindings.BuildProgram(_program, 1, devices, options, IntPtr.Zero, IntPtr.Zero);

            if (err != OpenClNativeBindings.CL_SUCCESS)
            {
                string buildLog = OpenClNativeBindings.GetBuildLog(_program, _context.Device);
                throw new InvalidOperationException($"Failed to build OpenCL program (error {err}):\n{buildLog}");
            }
        }

        public void Dispose()
        {
            if (_disposed) return;

            if (_program != IntPtr.Zero)
            {
                OpenClNativeBindings.ReleaseProgram(_program);
                _program = IntPtr.Zero;
            }

            _disposed = true;
        }
    }
}

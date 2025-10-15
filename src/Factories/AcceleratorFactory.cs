global using System;
global using System.Linq;
global using AiDotNet.HardwareAcceleration;

namespace AiDotNet.Factories
{
    public static class AcceleratorFactory
    {
        private static readonly object _lock = new object();
        private static readonly Dictionary<AcceleratorType, Func<ILogging, int, IAccelerator>> _acceleratorCreators;

        static AcceleratorFactory()
        {
            _acceleratorCreators = new Dictionary<AcceleratorType, Func<ILogging, int, IAccelerator>>();
            
            // Initialize with default accelerator creators
            _acceleratorCreators[AcceleratorType.CUDA] = (logger, deviceId) => 
                new CUDAAccelerator(logger ?? LoggingFactory.GetLogger<CUDAAccelerator>(), deviceId);
            
            _acceleratorCreators[AcceleratorType.Metal] = (logger, deviceId) => 
                new MetalAccelerator(logger ?? LoggingFactory.GetLogger<MetalAccelerator>(), deviceId);
            
            _acceleratorCreators[AcceleratorType.DirectML] = (logger, deviceId) => 
                new DirectMLAccelerator(logger ?? LoggingFactory.GetLogger<DirectMLAccelerator>(), deviceId);
        }

        public static IAccelerator CreateAccelerator(AcceleratorType type, ILogging? logger = null, int deviceId = 0)
        {
            if (deviceId < 0)
                throw new ArgumentOutOfRangeException("deviceId", "Device ID must be non-negative");

            Func<ILogging, int, IAccelerator> creator;
            lock (_lock)
            {
                if (!_acceleratorCreators.TryGetValue(type, out creator))
                {
                    var supportedTypes = string.Join(", ", _acceleratorCreators.Keys.Select(k => k.ToString()));
                    throw new NotSupportedException(string.Format("Accelerator type {0} is not supported. Supported types: {1}", type, supportedTypes));
                }
            }

            try
            {
                return creator(logger, deviceId);
            }
            catch (Exception ex)
            {
                if (logger != null)
                    logger.Error(string.Format("Failed to create {0} accelerator: {1}", type, ex.Message));
                throw new InvalidOperationException(string.Format("Failed to create {0} accelerator", type), ex);
            }
        }

        public static IAccelerator? CreateBestAvailableAccelerator(ILogging? logger = null)
        {
            if (logger != null)
                logger.Information("Searching for best available hardware accelerator");

            // Priority order based on typical performance
            var priorityOrder = new[]
            {
                AcceleratorType.CUDA,
                AcceleratorType.DirectML,
                AcceleratorType.Metal
            };

            foreach (var type in priorityOrder)
            {
                try
                {
                    var accelerator = CreateAccelerator(type, logger);
                    if (accelerator.IsAvailable)
                    {
                        if (logger != null)
                            logger.Information(string.Format("Selected {0} as hardware accelerator", type));
                        return accelerator;
                    }
                    accelerator.Dispose();
                }
                catch (Exception ex)
                {
                    if (logger != null)
                        logger.Debug(string.Format("Failed to create {0} accelerator: {1}", type, ex.Message));
                }
            }

            if (logger != null)
                logger.Warning("No hardware accelerator available, operations will run on CPU");
            return null;
        }

        public static List<AcceleratorInfo> GetAvailableAccelerators(ILogging? logger = null)
        {
            var availableAccelerators = new List<AcceleratorInfo>();

            foreach (AcceleratorType type in Enum.GetValues(typeof(AcceleratorType)))
            {
                if (type == AcceleratorType.CPU) continue; // Skip CPU

                try
                {
                    var accelerator = CreateAccelerator(type, logger);
                    try
                    {
                        if (accelerator.IsAvailable)
                        {
                            availableAccelerators.Add(accelerator.GetDeviceInfo());
                        }
                    }
                    finally
                    {
                        accelerator.Dispose();
                    }
                }
                catch
                {
                    // Ignore failures
                }
            }

            return availableAccelerators;
        }

        public static IAccelerator CreatePlatformOptimalAccelerator(ILogging? logger = null)
        {
            var platform = Environment.OSVersion.Platform;
            
            if (platform == PlatformID.Win32NT || platform == PlatformID.Win32Windows)
            {
                // On Windows, prefer CUDA > DirectML
                try
                {
                    var cudaAccel = CreateAccelerator(AcceleratorType.CUDA, logger);
                    if (cudaAccel.IsAvailable) return cudaAccel;
                    cudaAccel.Dispose();
                }
                catch { }

                try
                {
                    var directMLAccel = CreateAccelerator(AcceleratorType.DirectML, logger);
                    if (directMLAccel.IsAvailable) return directMLAccel;
                    directMLAccel.Dispose();
                }
                catch { }
            }
            else if (platform == PlatformID.Unix || platform == PlatformID.MacOSX)
            {
                // Check if it's macOS by trying to detect Darwin kernel
                bool isMacOS = false;
                try
                {
                    var unameProcess = new System.Diagnostics.Process
                    {
                        StartInfo = new System.Diagnostics.ProcessStartInfo
                        {
                            FileName = "uname",
                            Arguments = "-s",
                            UseShellExecute = false,
                            RedirectStandardOutput = true,
                            CreateNoWindow = true
                        }
                    };
                    
                    unameProcess.Start();
                    string output = unameProcess.StandardOutput.ReadToEnd();
                    unameProcess.WaitForExit();
                    
                    isMacOS = output.Trim().Equals("Darwin", StringComparison.OrdinalIgnoreCase);
                }
                catch
                {
                    // If we can't determine, try both
                }

                if (isMacOS)
                {
                    // On macOS, use Metal
                    try
                    {
                        var metalAccel = CreateAccelerator(AcceleratorType.Metal, logger);
                        if (metalAccel.IsAvailable) return metalAccel;
                        metalAccel.Dispose();
                    }
                    catch { }
                }
                else
                {
                    // On Linux, prefer CUDA
                    try
                    {
                        var cudaAccel = CreateAccelerator(AcceleratorType.CUDA, logger);
                        if (cudaAccel.IsAvailable) return cudaAccel;
                        cudaAccel.Dispose();
                    }
                    catch { }
                }
            }

            return null;
        }

        public static void RegisterCustomAccelerator(AcceleratorType type, Func<ILogging, int, IAccelerator> creator)
        {
            if (creator == null)
                throw new ArgumentNullException("creator");

            lock (_lock)
            {
                _acceleratorCreators[type] = creator;
            }
        }

        /// <summary>
        /// Unregisters a custom accelerator type
        /// </summary>
        public static bool UnregisterAccelerator(AcceleratorType type)
        {
            lock (_lock)
            {
                return _acceleratorCreators.Remove(type);
            }
        }

        /// <summary>
        /// Gets the count of registered accelerator types
        /// </summary>
        public static int RegisteredAcceleratorCount
        {
            get
            {
                lock (_lock)
                {
                    return _acceleratorCreators.Count;
                }
            }
        }
    }
}
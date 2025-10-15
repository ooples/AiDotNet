using System.IO;
using System.Threading.Tasks;

namespace AiDotNet.Helpers
{
    /// <summary>
    /// Helper class for async file operations that works across both .NET Framework and .NET Core/.NET 5+
    /// </summary>
    public static class FileAsyncHelper
    {
        /// <summary>
        /// Asynchronously writes text to a file, creating the file if it doesn't exist.
        /// </summary>
        /// <param name="path">The file to write to</param>
        /// <param name="contents">The text to write to the file</param>
        /// <returns>A task that represents the asynchronous write operation</returns>
        public static Task WriteAllTextAsync(string path, string contents)
        {
#if NET6_0_OR_GREATER
            return File.WriteAllTextAsync(path, contents);
#else
            // For .NET Framework, use Task.Run to wrap the synchronous operation
            return Task.Run(() => File.WriteAllText(path, contents));
#endif
        }

        /// <summary>
        /// Asynchronously reads all lines from a file.
        /// </summary>
        /// <param name="path">The file to read from</param>
        /// <returns>A task that represents the asynchronous read operation, containing all lines from the file</returns>
        public static Task<string[]> ReadAllLinesAsync(string path)
        {
#if NET6_0_OR_GREATER
            return File.ReadAllLinesAsync(path);
#else
            // For .NET Framework, use Task.Run to wrap the synchronous operation
            return Task.Run(() => File.ReadAllLines(path));
#endif
        }

        /// <summary>
        /// Asynchronously reads all text from a file.
        /// </summary>
        /// <param name="path">The file to read from</param>
        /// <returns>A task that represents the asynchronous read operation, containing all text from the file</returns>
        public static Task<string> ReadAllTextAsync(string path)
        {
#if NET6_0_OR_GREATER
            return File.ReadAllTextAsync(path);
#else
            // For .NET Framework, use Task.Run to wrap the synchronous operation
            return Task.Run(() => File.ReadAllText(path));
#endif
        }
    }
}

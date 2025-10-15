using System;
using System.Collections.Generic;

namespace AiDotNet.Models
{
    /// <summary>
    /// Benchmark example
    /// </summary>
    public class BenchmarkExample
    {
        public string Id { get; set; } = string.Empty;
        public string Input { get; set; } = string.Empty;
        public string ExpectedOutput { get; set; } = string.Empty;
        public Dictionary<string, object>? Metadata { get; set; }
    }
}
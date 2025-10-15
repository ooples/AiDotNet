using System;
using System.Collections.Generic;

namespace AiDotNet.Models.Results
{
    /// <summary>
    /// Benchmark evaluation results
    /// </summary>
    public class BenchmarkResults
    {
        public string BenchmarkName { get; set; } = string.Empty;
        public double Score { get; set; }
        public Dictionary<string, double> Metrics { get; set; } = new();
        public TimeSpan EvaluationTime { get; set; }
        public int TotalExamples { get; set; }
    }
}
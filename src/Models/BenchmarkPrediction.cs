using System;
using System.Collections.Generic;

namespace AiDotNet.Models
{
    /// <summary>
    /// Benchmark prediction
    /// </summary>
    public class BenchmarkPrediction
    {
        public string ExampleId { get; set; } = string.Empty;
        public string Prediction { get; set; } = string.Empty;
        public double Confidence { get; set; }
    }
}
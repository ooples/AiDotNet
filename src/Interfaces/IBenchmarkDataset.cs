using System.Collections.Generic;
using AiDotNet.Models;

namespace AiDotNet.Interfaces
{
    /// <summary>
    /// Benchmark dataset interface
    /// </summary>
    public interface IBenchmarkDataset
    {
        string Name { get; }
        List<BenchmarkExample> GetExamples();
        double CalculateScore(List<BenchmarkPrediction> predictions);
    }
}
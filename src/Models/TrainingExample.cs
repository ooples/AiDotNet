using System.Collections.Generic;

namespace AiDotNet.Models
{
    /// <summary>
    /// Training example for fine-tuning
    /// </summary>
    public class TrainingExample
    {
        public string Input { get; set; } = string.Empty;
        public string Output { get; set; } = string.Empty;
        public double Weight { get; set; } = 1.0;
        public Dictionary<string, object>? Metadata { get; set; }
    }
}
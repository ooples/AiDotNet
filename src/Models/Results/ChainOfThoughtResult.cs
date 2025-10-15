using System.Collections.Generic;

namespace AiDotNet.Models.Results
{
    /// <summary>
    /// Chain-of-thought reasoning result
    /// </summary>
    public class ChainOfThoughtResult
    {
        public List<string> ReasoningSteps { get; set; } = new();
        public string FinalAnswer { get; set; } = string.Empty;
        public double Confidence { get; set; }
        public Dictionary<string, object> Metadata { get; set; } = new();
    }
}
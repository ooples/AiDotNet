using System;
using System.Collections.Generic;

namespace AiDotNet.Models
{
    /// <summary>
    /// Few-shot learning example
    /// </summary>
    public class FewShotExample
    {
        public string Input { get; set; } = string.Empty;
        public string Output { get; set; } = string.Empty;
        public string? Explanation { get; set; }
    }
}
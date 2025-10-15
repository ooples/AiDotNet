using System;
using System.Collections.Generic;

namespace AiDotNet.Models
{
    /// <summary>
    /// Attention weights information
    /// </summary>
    public class AttentionWeights
    {
        public List<List<double[,]>> LayerWeights { get; set; } = new();
        public string[] Tokens { get; set; } = new string[0];
        public int NumLayers { get; set; }
        public int NumHeads { get; set; }
    }
}
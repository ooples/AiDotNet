using System;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.Models;

/// <summary>
/// Contains information about a transfer learning operation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TransferInfo<T>
{
    /// <summary>
    /// Gets or sets the type name of the source model.
    /// </summary>
    public string SourceModelType { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the transfer learning strategy used.
    /// </summary>
    public TransferLearningStrategy TransferStrategy { get; set; }
    
    /// <summary>
    /// Gets or sets the date when the transfer was performed.
    /// </summary>
    public DateTime TransferDate { get; set; }
    
    /// <summary>
    /// Gets or sets the number of layers transferred.
    /// </summary>
    public int LayersTransferred { get; set; }
    
    /// <summary>
    /// Gets or sets the options used for transfer learning.
    /// </summary>
    public TransferLearningOptions<T>? Options { get; set; }
    
    /// <summary>
    /// Gets or sets additional metadata about the transfer.
    /// </summary>
    public Dictionary<string, object>? Metadata { get; set; }
}
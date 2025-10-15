using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.MultimodalAI
{
    /// <summary>
    /// Container for multimodal data inputs
    /// </summary>
    public class MultimodalData
    {
        private readonly Dictionary<string, object> _modalityData = default!;
        private readonly Dictionary<string, Type> _modalityTypes = default!;
        private readonly Dictionary<string, object> _metadata = default!;

        /// <summary>
        /// Gets the modality data
        /// </summary>
        public IReadOnlyDictionary<string, object> ModalityData => _modalityData;

        /// <summary>
        /// Gets the metadata associated with the multimodal data
        /// </summary>
        public IReadOnlyDictionary<string, object> Metadata => _metadata;

        /// <summary>
        /// Gets the list of modalities present in this data
        /// </summary>
        public IReadOnlyList<string> Modalities => _modalityData.Keys.ToList();

        /// <summary>
        /// Initializes a new instance of MultimodalData
        /// </summary>
        public MultimodalData()
        {
            _modalityData = new Dictionary<string, object>();
            _modalityTypes = new Dictionary<string, Type>();
            _metadata = new Dictionary<string, object>();
        }

        /// <summary>
        /// Adds data for a specific modality
        /// </summary>
        /// <typeparam name="T">Type of the modality data</typeparam>
        /// <param name="modalityName">Name of the modality</param>
        /// <param name="data">The data for this modality</param>
        public void AddModality<T>(string modalityName, T data)
        {
            if (string.IsNullOrWhiteSpace(modalityName))
                throw new ArgumentException("Modality name cannot be null or empty", nameof(modalityName));

            if (data == null)
                throw new ArgumentNullException(nameof(data));

            _modalityData[modalityName] = data;
            _modalityTypes[modalityName] = typeof(T);
        }

        /// <summary>
        /// Gets data for a specific modality
        /// </summary>
        /// <typeparam name="T">Expected type of the modality data</typeparam>
        /// <param name="modalityName">Name of the modality</param>
        /// <returns>The data for the specified modality</returns>
        public T GetModality<T>(string modalityName)
        {
            if (!_modalityData.ContainsKey(modalityName))
                throw new KeyNotFoundException($"Modality '{modalityName}' not found");

            return (T)_modalityData[modalityName];
        }

        /// <summary>
        /// Checks if a modality exists in the data
        /// </summary>
        /// <param name="modalityName">Name of the modality</param>
        /// <returns>True if the modality exists</returns>
        public bool HasModality(string modalityName)
        {
            return _modalityData.ContainsKey(modalityName);
        }

        /// <summary>
        /// Removes a modality from the data
        /// </summary>
        /// <param name="modalityName">Name of the modality to remove</param>
        /// <returns>True if the modality was removed</returns>
        public bool RemoveModality(string modalityName)
        {
            _modalityTypes.Remove(modalityName);
            return _modalityData.Remove(modalityName);
        }

        /// <summary>
        /// Adds metadata to the multimodal data
        /// </summary>
        /// <param name="key">Metadata key</param>
        /// <param name="value">Metadata value</param>
        public void AddMetadata(string key, object value)
        {
            _metadata[key] = value;
        }

        /// <summary>
        /// Gets the type of a specific modality
        /// </summary>
        /// <param name="modalityName">Name of the modality</param>
        /// <returns>The type of the modality data</returns>
        public Type? GetModalityType(string modalityName)
        {
            return _modalityTypes.ContainsKey(modalityName) ? _modalityTypes[modalityName] : null;
        }

        /// <summary>
        /// Creates a shallow copy of the multimodal data
        /// </summary>
        /// <returns>A new MultimodalData instance with copied references</returns>
        public MultimodalData Clone()
        {
            var clone = new MultimodalData();
            
            foreach (var kvp in _modalityData)
            {
                clone._modalityData[kvp.Key] = kvp.Value;
                clone._modalityTypes[kvp.Key] = _modalityTypes[kvp.Key];
            }

            foreach (var kvp in _metadata)
            {
                clone._metadata[kvp.Key] = kvp.Value;
            }

            return clone;
        }

        /// <summary>
        /// Validates that all required modalities are present
        /// </summary>
        /// <param name="requiredModalities">List of required modality names</param>
        /// <returns>True if all required modalities are present</returns>
        public bool ValidateModalities(IEnumerable<string> requiredModalities)
        {
            return requiredModalities.All(m => HasModality(m));
        }
    }
}
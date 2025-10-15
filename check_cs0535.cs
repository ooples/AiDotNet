using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

class Program
{
    static void Main()
    {
        Console.WriteLine("Checking for potential CS0535 errors...\n");
        
        // Find all interface files
        var interfaceFiles = Directory.GetFiles("/mnt/c/projects/AiDotNet/src/Interfaces", "*.cs", SearchOption.AllDirectories);
        
        foreach (var interfaceFile in interfaceFiles)
        {
            var interfaceName = Path.GetFileNameWithoutExtension(interfaceFile);
            if (!interfaceName.StartsWith("I")) continue;
            
            Console.WriteLine($"Checking interface: {interfaceName}");
            
            // Find classes that implement this interface
            var srcFiles = Directory.GetFiles("/mnt/c/projects/AiDotNet/src", "*.cs", SearchOption.AllDirectories)
                .Where(f => !f.Contains("/Interfaces/"));
            
            foreach (var srcFile in srcFiles)
            {
                var content = File.ReadAllText(srcFile);
                
                // Look for class declarations implementing this interface
                var pattern = $@"class\s+(\w+).*:\s*.*{interfaceName}";
                var matches = Regex.Matches(content, pattern);
                
                if (matches.Count > 0)
                {
                    foreach (Match match in matches)
                    {
                        Console.WriteLine($"  Found implementation in: {Path.GetFileName(srcFile)} - Class: {match.Groups[1].Value}");
                    }
                }
            }
        }
    }
}
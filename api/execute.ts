import type { VercelRequest, VercelResponse } from '@vercel/node';

interface PistonExecuteRequest {
  language: string;
  version: string;
  files: { name: string; content: string }[];
  stdin?: string;
  args?: string[];
  compile_timeout?: number;
  run_timeout?: number;
  compile_memory_limit?: number;
  run_memory_limit?: number;
}

interface PistonExecuteResponse {
  run: {
    stdout: string;
    stderr: string;
    output: string;
    code: number;
    signal: string | null;
  };
  compile?: {
    stdout: string;
    stderr: string;
    output: string;
    code: number;
    signal: string | null;
  };
  language: string;
  version: string;
}

interface ExecuteRequest {
  code: string;
  language?: string;
}

interface ExecuteResponse {
  success: boolean;
  output?: string;
  error?: string;
  compilationOutput?: string;
  executionTime?: number;
}

// Piston API endpoint (free, no API key required)
const PISTON_API = 'https://emkc.org/api/v2/piston';

// Add using statements if not present
function preprocessCode(code: string): string {
  const requiredUsings = [
    'using System;',
    'using System.Collections.Generic;',
    'using System.Linq;',
  ];

  // Check if the code has a namespace or class declaration
  const hasClass = /class\s+\w+/.test(code);
  const hasMain = /static\s+(void|int|async\s+Task)\s+Main/.test(code);

  // If no class or Main method, wrap in a simple program
  if (!hasClass && !hasMain) {
    // Extract existing using statements from the user's code
    const usingRegex = /^using\s+[\w.]+;\s*$/gm;
    const existingUsings: string[] = [];
    let codeWithoutUsings = code;

    let match;
    while ((match = usingRegex.exec(code)) !== null) {
      existingUsings.push(match[0].trim());
    }

    // Remove using statements from the code body
    codeWithoutUsings = code.replace(usingRegex, '').trim();

    // Combine required and existing usings, avoiding duplicates
    const allUsings = new Set<string>(requiredUsings);
    for (const existing of existingUsings) {
      allUsings.add(existing);
    }

    // Build the wrapped program
    const usingsBlock = Array.from(allUsings).join('\n');
    return `${usingsBlock}

class Program
{
    static void Main()
    {
        ${codeWithoutUsings.split('\n').join('\n        ')}
    }
}`;
  }

  // For code with class/Main, just add missing using statements at the top
  let processedCode = code;
  for (const usingStatement of requiredUsings) {
    // Check for the actual using statement, not just the namespace
    if (!code.includes(usingStatement) && !code.includes(usingStatement.replace(';', ''))) {
      processedCode = usingStatement + '\n' + processedCode;
    }
  }

  return processedCode;
}

async function executeWithPiston(code: string): Promise<ExecuteResponse> {
  const startTime = Date.now();

  try {
    // Preprocess the code to ensure it's a valid C# program
    const processedCode = preprocessCode(code);

    const request: PistonExecuteRequest = {
      language: 'csharp',
      version: '*', // Use latest available version
      files: [
        {
          name: 'Program.cs',
          content: processedCode,
        },
      ],
      compile_timeout: 10000, // 10 seconds
      run_timeout: 5000, // 5 seconds
      compile_memory_limit: 1024000000, // 1GB - Roslyn needs significant memory
      run_memory_limit: 512000000, // 512MB for runtime
    };

    const response = await fetch(`${PISTON_API}/execute`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const errorText = await response.text();
      return {
        success: false,
        error: `Piston API error: ${response.status} - ${errorText}`,
        executionTime: Date.now() - startTime,
      };
    }

    const result = await response.json() as PistonExecuteResponse;
    const executionTime = Date.now() - startTime;

    // Check for compilation errors
    if (result.compile && result.compile.code !== 0) {
      return {
        success: false,
        error: result.compile.stderr || result.compile.output || 'Compilation failed',
        compilationOutput: result.compile.output,
        executionTime,
      };
    }

    // Check for runtime errors
    if (result.run.code !== 0) {
      return {
        success: false,
        output: result.run.stdout,
        error: result.run.stderr || `Process exited with code ${result.run.code}`,
        executionTime,
      };
    }

    return {
      success: true,
      output: result.run.stdout || result.run.output || '(No output)',
      executionTime,
    };
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error occurred',
      executionTime: Date.now() - startTime,
    };
  }
}

// Rate limiting: simple in-memory store (resets on cold start)
const rateLimitMap = new Map<string, { count: number; resetTime: number }>();
const RATE_LIMIT = 10; // requests per minute
const RATE_LIMIT_WINDOW = 60000; // 1 minute

function checkRateLimit(ip: string): boolean {
  const now = Date.now();
  const record = rateLimitMap.get(ip);

  if (!record || now > record.resetTime) {
    rateLimitMap.set(ip, { count: 1, resetTime: now + RATE_LIMIT_WINDOW });
    return true;
  }

  if (record.count >= RATE_LIMIT) {
    return false;
  }

  record.count++;
  return true;
}

export default async function handler(
  req: VercelRequest,
  res: VercelResponse
): Promise<void> {
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  // Handle preflight request
  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  // Only allow POST requests
  if (req.method !== 'POST') {
    res.status(405).json({ success: false, error: 'Method not allowed' });
    return;
  }

  // Rate limiting
  const clientIp = (req.headers['x-forwarded-for'] as string)?.split(',')[0] ||
                   req.socket?.remoteAddress ||
                   'unknown';

  if (!checkRateLimit(clientIp)) {
    res.status(429).json({
      success: false,
      error: 'Rate limit exceeded. Please wait a minute before trying again.',
    });
    return;
  }

  try {
    const { code, language } = req.body as ExecuteRequest;

    if (!code || typeof code !== 'string') {
      res.status(400).json({
        success: false,
        error: 'Code is required and must be a string',
      });
      return;
    }

    // Validate code length
    if (code.length > 50000) {
      res.status(400).json({
        success: false,
        error: 'Code too long. Maximum 50,000 characters allowed.',
      });
      return;
    }

    // Defense-in-depth security checks: These regex patterns block common dangerous operations
    // as an initial gate. The real security boundary is Piston's sandbox (Docker + Isolate with
    // namespaces, cgroups, chroot). These patterns can be bypassed via reflection, string
    // concatenation, or encoding tricks, but Piston's isolation handles more determined attempts.
    const dangerousPatterns = [
      /System\.IO\.File/i,
      /System\.Diagnostics\.Process/i,
      /Environment\.Exit/i,
      /System\.Net\.WebClient/i,
      /System\.Net\.Http/i,
      /System\.Reflection/i,
      /Assembly\.Load/i,
      /DllImport/i,
      /unsafe\s*\{/i,
    ];

    for (const pattern of dangerousPatterns) {
      if (pattern.test(code)) {
        res.status(400).json({
          success: false,
          error: 'Code contains potentially dangerous operations that are not allowed in the playground.',
        });
        return;
      }
    }

    // Execute the code
    const result = await executeWithPiston(code);

    res.status(result.success ? 200 : 400).json(result);
  } catch (error) {
    console.error('Execution error:', error);
    res.status(500).json({
      success: false,
      error: 'Internal server error',
    });
  }
}

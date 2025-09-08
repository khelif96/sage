import { createTool } from '@mastra/core';
import { Agent } from '@mastra/core/agent';
import { createWorkflow, createStep } from '@mastra/core/workflows';
import { MDocument } from '@mastra/rag';
import { z } from 'zod';
import { WinstonMastraLogger } from '../../utils/logger/winstonMastraLogger';
import { logAnalyzerConfig } from './logCoreAnalyzer/config';
import {
  loadFromFile,
  loadFromUrl,
  loadFromText,
  type LoadResult,
} from './logCoreAnalyzer/dataLoader';
import {
  INITIAL_ANALYZER_INSTRUCTIONS,
  REFINEMENT_AGENT_INSTRUCTIONS,
  REPORT_FORMATTER_INSTRUCTIONS,
  USER_INITIAL_PROMPT,
  USER_REFINE,
  USER_MARKDOWN_PROMPT,
  USER_CONCISE_SUMMARY_PROMPT,
  SINGLE_PASS_PROMPT,
} from './logCoreAnalyzer/prompts';
import { normalizeLineEndings } from './logCoreAnalyzer/utils';

// We define here the core workflow for log file analysis. It gives the Parsley Agent the capability to read and understand text files, of any kind and format.
// Depending on the file size, we either return a summary in a single LLM call, or perform a more complex iterative refinement, combining the usage of cheap and more expensive models.

// Initialize logger for this workflow
const logger = new WinstonMastraLogger({
  name: logAnalyzerConfig.logging.name,
  level: logAnalyzerConfig.logging.level,
});

// This workflow takes either a file path, raw text, or an URL as input, and optional additional instructions
// and returns a structured analysis report, as well as a concise summary.
const WorkflowInputSchema = z.object({
  path: z
    .string()
    .optional()
    .describe(
      'Absolute file path on the local filesystem (e.g., "/var/log/app.log", "/tmp/debug.txt"). The file must be accessible from the server.'
    ),
  text: z
    .string()
    .optional()
    .describe(
      'Raw text content to analyze. Use this when you already have the log content in memory or received it from another tool.'
    ),
  url: z
    .string()
    .optional()
    .describe(
      'HTTP/HTTPS URL to fetch and analyze content from. Must be a direct link to raw text/log content (e.g., "https://pastebin.com/raw/abc123").'
    ),
  analysisContext: z
    .string()
    .optional()
    .describe(
      'Additional context or specific analysis instructions. Can include file origin, what to focus on, or specific questions to answer during analysis.'
    ),
});

const WorkflowOutputSchema = z.object({
  markdown: z.string(),
  summary: z.string(),
});

// Unified load step that handles all I/O with validation
const loadDataStep = createStep({
  id: 'load-data',
  description: 'Load and validate data from any source',
  inputSchema: WorkflowInputSchema,
  outputSchema: z.object({
    text: z.string(),
    analysisContext: z.string().optional(),
  }),
  execute: async ({ inputData }) => {
    const { analysisContext, path: filePath, text, url } = inputData;

    let result: LoadResult;

    try {
      if (filePath) {
        result = await loadFromFile(filePath);
      } else if (url) {
        result = await loadFromUrl(url);
      } else if (text) {
        result = await loadFromText(text);
      } else {
        throw new Error(
          'No input source provided (path, url, or text required)'
        );
      }
    } catch (error) {
      logger.error('Failed to load data', error);
      throw error;
    }

    // Normalize the text
    const normalizedText = normalizeLineEndings(result.text);

    logger.info('Data loaded successfully', {
      source: result.metadata.source,
      sizeMB: (result.metadata.originalSize / 1024 / 1024).toFixed(2),
      estimatedTokens: result.metadata.estimatedTokens,
    });

    return {
      text: normalizedText,
      analysisContext,
    };
  },
});

const ChunkedSchema = z.object({
  chunks: z.array(z.object({ text: z.string() })), // from MDocument.chunk
  analysisContext: z.string().optional(),
});

// Modified chunk step to use loaded data with metadata
const chunkStep = createStep({
  id: 'chunk',
  description: 'Token-aware chunking with overlap',
  inputSchema: z.object({
    text: z.string(),
    analysisContext: z.string().optional(),
  }),
  outputSchema: ChunkedSchema,
  execute: async ({ inputData }) => {
    const { analysisContext, text } = inputData;

    const doc = MDocument.fromText(text);
    const chunks = await doc.chunk({
      strategy: 'token',
      encodingName: logAnalyzerConfig.chunking.tokenizer,
      maxSize: logAnalyzerConfig.chunking.maxSize,
      overlap: logAnalyzerConfig.chunking.overlapTokens,
    });

    logger.debug('Chunking complete', {
      chunkCount: chunks.length,
    });

    return { chunks, analysisContext };
  },
});

// This schema passed accross steps during the refinement loop, keeps track of the current chunk index and summary
const LoopStateSchema = z.object({
  idx: z.number(),
  chunks: z.array(z.object({ text: z.string() })),
  summary: z.string(),
  analysisContext: z.string().optional(),
});

// Define the log analyzer agent for chunked processing
// Initial analyzer - We use a bigger model for the first chunk, for better understanding of the structure and context

const initialAnalyzerAgent = new Agent({
  name: 'initial-analyzer-agent',
  description:
    'Performs initial analysis of technical documents to understand structure and key patterns',
  instructions: INITIAL_ANALYZER_INSTRUCTIONS,
  model: logAnalyzerConfig.models.initial,
});

const initialStep = createStep({
  id: 'initial-summary',
  description: 'Summarize first chunk using log analyzer agent',
  inputSchema: ChunkedSchema,
  outputSchema: LoopStateSchema,
  execute: async ({ inputData, tracingContext }) => {
    const { analysisContext, chunks } = inputData;
    const first = chunks[0]?.text ?? '';
    logger.debug('Initial chunk for analysis', {
      first: first.slice(0, 100),
      analysisContext,
    });
    logger.debug('Chunk length', { length: first.length });
    logger.debug('Calling LLM for initial summary');

    const result = await initialAnalyzerAgent.generateVNext(
      [
        {
          role: 'user',
          content: USER_INITIAL_PROMPT(first, analysisContext),
        },
      ],
      {
        structuredOutput: {
          schema: RefinementAgentOutputSchema,
          model: logAnalyzerConfig.models.initial,
        },
        tracingContext,
      }
    );

    const response = result.object as unknown as z.infer<
      typeof RefinementAgentOutputSchema
    >;

    const { summary } = response;

    return { idx: 1, chunks, summary, analysisContext };
  },
});

// Refinement agent - cheaper model for iterative processing

const RefinementAgentOutputSchema = z.object({
  updated: z.boolean(),
  summary: z.string(),
});

const refinementAgent = new Agent({
  name: 'refinement-agent',
  description:
    'Iteratively refines and updates technical summaries with new chunks',
  instructions: REFINEMENT_AGENT_INSTRUCTIONS,
  model: logAnalyzerConfig.models.refinement,
});

const refineStep = createStep({
  id: 'refine-summary',
  description:
    'Iteratively refine the summary with context from previous chunks',
  inputSchema: LoopStateSchema,
  outputSchema: LoopStateSchema,
  execute: async ({ inputData, tracingContext }) => {
    const {
      analysisContext,
      chunks,
      idx,
      summary: existingSummary,
    } = inputData;
    const chunk = chunks[idx]?.text ?? '';

    // TODO: make sure summary size stays manageable
    if (!chunk) {
      return {
        idx: idx + 1,
        chunks,
        summary: existingSummary,
        analysisContext,
      };
    }

    logger.debug('Refine step for chunk #:', {
      current: idx + 1,
      total: chunks.length,
    });
    const result = await refinementAgent.generateVNext(
      [
        {
          role: 'user',
          content: USER_REFINE(existingSummary, chunk, analysisContext),
        },
      ],
      {
        structuredOutput: {
          schema: RefinementAgentOutputSchema,
          model: logAnalyzerConfig.models.refinement,
        }, // TODO: define error handling strategy when schema validation fails
        tracingContext,
      }
    );

    const response = result.object as unknown as z.infer<
      typeof RefinementAgentOutputSchema
    >;

    const updated = response.updated ?? false;
    let newSummary = existingSummary;
    if (updated) {
      newSummary = response.summary ?? existingSummary;
    }

    return {
      idx: idx + 1,
      chunks,
      summary: newSummary,
      analysisContext,
    };
  },
});

// Define the report formatter agent for final output
const reportFormatterAgent = new Agent({
  name: 'report-formatter-agent',
  description: 'Formats technical summaries into various output formats',
  instructions: REPORT_FORMATTER_INSTRUCTIONS,
  model: logAnalyzerConfig.models.formatter,
});

// Single-pass step for files that fit in one chunk - generates both markdown and summary in one call
const singlePassStep = createStep({
  id: 'single-pass-analysis',
  description: 'Direct analysis and report generation for single-chunk files',
  inputSchema: ChunkedSchema,
  outputSchema: WorkflowOutputSchema,
  execute: async ({ inputData, tracingContext }) => {
    const { analysisContext, chunks } = inputData;

    // Validate we have exactly one chunk
    if (chunks.length !== 1) {
      logger.warn('Single-pass step called with multiple chunks', {
        chunkCount: chunks.length,
      });
    }

    const text = chunks[0]?.text ?? '';

    logger.debug('Single-pass analysis starting', {
      textLength: text.length,
      analysisContext,
    });

    // Use structured output to get both markdown and summary
    const result = await reportFormatterAgent.generateVNext(
      [{ role: 'user', content: SINGLE_PASS_PROMPT(text, analysisContext) }],
      {
        structuredOutput: {
          schema: WorkflowOutputSchema,
          model: logAnalyzerConfig.models.formatter,
        },
        tracingContext,
      }
    );
    const response = result.object as unknown as z.infer<
      typeof WorkflowOutputSchema
    >;

    logger.debug('Single-pass analysis complete', {
      markdownLength: response.markdown?.length ?? 0,
      summaryLength: response.summary?.length ?? 0,
    });

    return {
      markdown: response.markdown || '',
      summary: response.summary || '',
    };
  },
});

const finalizeStep = createStep({
  id: 'finalize',
  description: 'Generate final markdown report and concise summary',
  inputSchema: LoopStateSchema,
  outputSchema: WorkflowOutputSchema,
  execute: async ({ inputData, tracingContext }) => {
    const { analysisContext, summary } = inputData;
    logger.debug('Generating final markdown report', {
      summary: summary.slice(0, 100),
      analysisContext,
    });

    // Generate markdown report
    const markdownRes = await reportFormatterAgent.generateVNext(
      [
        {
          role: 'user',
          content: USER_MARKDOWN_PROMPT(summary, analysisContext),
        },
      ],
      {
        tracingContext,
      }
    );
    logger.debug('Final markdown report generated', {
      length: markdownRes.text.length,
    });

    // Generate concise summary from the markdown report
    logger.debug('Generating concise summary');
    const conciseSummaryRes = await reportFormatterAgent.generateVNext(
      [
        {
          role: 'user',
          content: USER_CONCISE_SUMMARY_PROMPT(
            markdownRes.text,
            analysisContext
          ),
        },
      ],
      {
        tracingContext,
      }
    );
    logger.debug('Concise summary generated', {
      length: conciseSummaryRes.text.length,
    });

    return {
      markdown: markdownRes.text,
      summary: conciseSummaryRes.text,
    };
  },
});

const iterativeRefinementWorkflow = createWorkflow({
  id: 'iterative-refinement',
  description: `Perform a 3 step iterative refinement process: initial and final analysis with an expensive model, 
    and a lightweight refinement loop going through the whole document`,
  inputSchema: ChunkedSchema,
  outputSchema: WorkflowOutputSchema,
})
  .then(initialStep)
  .dowhile(
    refineStep,
    async params =>
      // Access inputData from the full params object
      params.inputData.idx < params.inputData.chunks.length
  )
  .then(finalizeStep)
  .commit();

// This was initially a `.branch()` workflow step, but it involved too much complexity like unwrapping types correctly,
// or wrapping iterativeRefinementWorkflow into its own step. This option is much simpler.
const decideAndRunStep = createStep({
  id: 'decide-and-run',
  description: 'Choose single-pass vs iterative workflow and run it',
  inputSchema: ChunkedSchema,
  outputSchema: WorkflowOutputSchema,
  execute: async params => {
    const { chunks } = params.inputData;
    if (chunks.length === 1) {
      // run the single-pass step directly
      return singlePassStep.execute(params);
    }
    // run the iterative workflow
    return iterativeRefinementWorkflow.execute(params);
  },
});

export const logCoreAnalyzerWorkflow = createWorkflow({
  id: 'log-core-analyzer',
  description:
    'Analyzes and summarizes log files, technical documents, or any text content. Produces a structured markdown report with key findings and a concise summary. ' +
    'INPUTS (provide exactly ONE): ' +
    '• path: Absolute file path on the local filesystem (e.g., "/var/log/app.log") ' +
    '• url: HTTP/HTTPS URL to fetch content from (e.g., "https://example.com/logs.txt") ' +
    '• text: Raw text content as a string (for content already in memory) ' +
    'OPTIONAL: analysisContext - Additional instructions for what to focus on (e.g., "Look for timeout errors", "Focus on authentication issues") ' +
    'NOTE: This tool analyzes raw file content. It does NOT fetch data from Evergreen or other APIs - provide the actual content or a direct URL/path to it.',
  inputSchema: WorkflowInputSchema,
  outputSchema: WorkflowOutputSchema,
})
  .then(loadDataStep) // Use the new unified load step with validation
  .then(chunkStep)
  .then(decideAndRunStep)
  .commit();

export const logCoreAnalyzerTool: ReturnType<typeof createTool> = createTool({
  id: 'logCoreAnalyzerTool',
  description:
    logCoreAnalyzerWorkflow.description ||
    'Analyzes log files and text content',
  inputSchema: logCoreAnalyzerWorkflow.inputSchema,
  outputSchema: logCoreAnalyzerWorkflow.outputSchema,
  execute: async ({ context, runtimeContext, tracingContext }) => {
    const run = await logCoreAnalyzerWorkflow.createRunAsync({});

    const runResult = await run.start({
      inputData: context,
      runtimeContext,
      tracingContext,
    });
    if (runResult.status === 'success') {
      return runResult.result;
    }
    if (runResult.status === 'failed') {
      throw new Error(
        `Log analyzer workflow failed: ${runResult.error.message}`
      );
    }
    throw new Error(
      `Unexpected workflow execution status: ${runResult.status}. Expected 'success' or 'failed'.`
    );
  },
});

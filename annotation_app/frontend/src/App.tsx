import { useEffect, useState } from "react";
import type {
  AcceptResponse,
  ActivateRefillRoundResponse,
  AppConfig,
  BacklogRow,
  ChunkSummary,
  ConfigSaveError,
  InvalidRetryRow,
  PreviewResponse,
  ProjectConfig,
  ProjectSummary,
  RefillRoundResponse,
  SkipInvalidRetryCacheResponse,
} from "./types";

const API_BASE = "http://127.0.0.1:8000/api";

class ApiError extends Error {
  detail?: unknown;

  constructor(message: string, detail?: unknown) {
    super(message);
    this.detail = detail;
  }
}

function cloneConfig(config: AppConfig): AppConfig {
  return JSON.parse(JSON.stringify(config)) as AppConfig;
}

function extractApiErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    const detail = error.detail as { message?: string; errors?: string[] } | undefined;
    if (detail?.errors && detail.errors.length > 0) {
      return `${detail.message ?? "Request failed."} ${detail.errors.join(" ")}`;
    }
    if (detail?.message) {
      return detail.message;
    }
    return error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "Request failed.";
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    const errorPayload = await response.json().catch(() => null);
    const detail = errorPayload?.detail;
    const message =
      typeof detail === "string"
        ? detail
        : detail?.message || `Request failed: ${response.status}`;
    throw new ApiError(message, detail);
  }
  return response.json() as Promise<T>;
}

function formatPreviewFailure(preview: PreviewResponse | null): string {
  if (!preview || preview.ok) {
    return "";
  }

  switch (preview.error) {
    case "header_mismatch":
      return `The pasted CSV headers do not match any accepted shape for this chunk. Accepted headers: ${preview.accepted_header_sets?.map((headers) => headers.join(", ")).join(" or ") || preview.expected_headers?.join(", ") || "unknown"}. Received ${preview.received_headers?.length ?? 0} columns: ${preview.received_headers?.join(", ") || "none"}.`;
    case "row_count_mismatch":
      return `The pasted CSV has the wrong number of rows for this chunk. Expected ${preview.expected_row_count ?? 0} row(s), but received ${preview.received_row_count ?? 0}. This usually means the response is truncated, includes the wrong chunk, or is missing rows.`;
    case "original_text_order_mismatch": {
      const expected = preview.expected_original_texts ?? [];
      const received = preview.received_original_texts ?? [];
      let firstMismatch = -1;
      const maxLength = Math.max(expected.length, received.length);
      for (let index = 0; index < maxLength; index += 1) {
        if (expected[index] !== received[index]) {
          firstMismatch = index;
          break;
        }
      }
      if (firstMismatch >= 0) {
        return `The pasted CSV row order does not match this chunk. The first mismatch is at row ${firstMismatch + 1}: expected original_text ${JSON.stringify(expected[firstMismatch] ?? null)}, received ${JSON.stringify(received[firstMismatch] ?? null)}.`;
      }
      return "The pasted CSV row order does not match the current chunk.";
    }
    case "duplicate_original_text_in_import":
      return `The pasted CSV contains duplicate original_text values, so the app cannot safely map rows back to the chunk. Duplicates found: ${preview.duplicate_original_texts?.join(" | ") || "unknown"}.`;
    case "ambiguous_source_original_text":
      return `The source chunk itself contains duplicate original_text values, so exact text matching is ambiguous. Duplicate source texts: ${preview.duplicate_original_texts?.join(" | ") || "unknown"}.`;
    case "original_text_not_found":
      return `Some returned rows do not match any source row in this chunk by exact original_text. Missing examples: ${preview.missing_original_texts?.join(" | ") || "unknown"}.`;
    case "missing_row_id_column":
      return "This project is configured to require row_id-based matching, but the pasted CSV does not include a row_id column. Return a CSV with row_id, original_text, and masked_text.";
    case "row_id_not_found":
      return `Some returned row_id values do not exist in this chunk. Unknown row IDs: ${preview.missing_row_ids?.join(" | ") || "unknown"}.`;
    case "duplicate_row_id_in_import":
      return `The pasted CSV contains duplicate row_id values, so the app cannot safely map rows back to the chunk. Duplicates found: ${preview.duplicate_row_ids?.join(" | ") || "unknown"}.`;
    case "ambiguous_source_row_id":
      return `The source chunk itself contains duplicate row_id values, so row_id matching is ambiguous. Duplicate source row IDs: ${preview.duplicate_row_ids?.join(" | ") || "unknown"}.`;
    case "row_id_order_mismatch": {
      const expected = preview.expected_row_ids ?? [];
      const received = preview.received_row_ids ?? [];
      let firstMismatch = -1;
      const maxLength = Math.max(expected.length, received.length);
      for (let index = 0; index < maxLength; index += 1) {
        if (expected[index] !== received[index]) {
          firstMismatch = index;
          break;
        }
      }
      if (firstMismatch >= 0) {
        return `The pasted CSV includes row IDs, but they no longer line up with this chunk. The first mismatch is at row ${firstMismatch + 1}: expected row_id ${JSON.stringify(expected[firstMismatch] ?? null)}, received ${JSON.stringify(received[firstMismatch] ?? null)}. This usually means rows were dropped, reordered, or copied from another chunk.`;
      }
      return "The pasted CSV includes row IDs, but they do not line up with the current chunk.";
    }
    case "row_id_original_text_mismatch":
      return `Some returned rows match a valid row_id, but the original_text no longer matches that row. Example mismatch: ${preview.mismatched_rows?.[0] ? `row_id ${JSON.stringify(preview.mismatched_rows[0].row_id)} expected original_text ${JSON.stringify(preview.mismatched_rows[0].expected_original_text)}, received ${JSON.stringify(preview.mismatched_rows[0].received_original_text)}.` : "unknown mismatch."}`;
    case "missing_working_preview":
      return "Retry merge is not available yet for this chunk because no working preview exists. Run a full chunk preview first.";
    case "no_open_invalid_rows":
      return "There are no open invalid rows cached for this chunk, so there is nothing to retry right now.";
    case "unknown_retry_rows":
      return `Some pasted retry rows do not match the current invalid-row cache for this chunk. Unknown originals: ${preview.unknown_retry_rows?.join(" | ") || "unknown"}.`;
    case "unknown_retry_row_ids":
      return `Some pasted retry row_id values do not match any cached invalid rows for this chunk. Unknown row IDs: ${preview.unknown_retry_row_ids?.join(" | ") || "unknown"}.`;
    default:
      return `Preview failed because the pasted CSV could not be validated. Reported error: ${preview.error ?? "unknown error"}.`;
  }
}

function formatAcceptFailure(result: AcceptResponse): string {
  switch (result.error) {
    case "invalid_rows_present": {
      const invalidRows = result.preview?.summary?.invalid_rows ?? 0;
      return `This import cannot be accepted yet because ${invalidRows} row(s) still contain validation errors. Fix those rows in the external AI output, then preview again.`;
    }
    case "pending_rows_present": {
      const pendingRows = result.preview?.summary?.pending_rows ?? 0;
      return `This import cannot be accepted yet because ${pendingRows} row(s) are still marked pending. Either resolve them to approved rows or change the project setting to allow pending rows.`;
    }
    default:
      if (!result.preview) {
        return `Accept failed. Reported error: ${result.error ?? "unknown error"}.`;
      }
      return formatPreviewFailure(result.preview);
  }
}

function formatInvalidRetryRowsForCopy(rows: InvalidRetryRow[], rowMatching: string): string {
  const includeRowId = rowMatching === "strict_row_id";
  const header = includeRowId ? "row_id,original_text" : "original_text";
  const csvRows = rows.map((row) => {
    const escapedOriginal = JSON.stringify(row.original_text);
    if (includeRowId) {
      const escapedRowId = JSON.stringify(row.row_id ?? "");
      return `${escapedRowId},${escapedOriginal}`;
    }
    return escapedOriginal;
  });

  const explanations = rows.map((row, index) => {
    const joinedErrors = row.errors.join(" | ");
    return `${index + 1}. chunk=${row.chunk_file_name} example_id=${row.example_id}${row.row_id ? ` row_id=${row.row_id}` : ""}\n   original_text: ${row.original_text}\n   why_invalid: ${joinedErrors}`;
  });

  return [
    "Retry these invalid rows only.",
    includeRowId
      ? "Return a three-column CSV with row_id, original_text, and masked_text."
      : "Return a two-column CSV with original_text and masked_text.",
    "",
    "Invalid row reasons:",
    ...explanations,
    "",
    "CSV to edit:",
    header,
    ...csvRows,
  ].join("\n");
}

function formatConfigSaveError(error: unknown): string {
  return extractApiErrorMessage(error);
}

function App() {
  const [projectId, setProjectId] = useState("pilot");
  const [summary, setSummary] = useState<ProjectSummary | null>(null);
  const [chunks, setChunks] = useState<ChunkSummary[]>([]);
  const [currentChunkId, setCurrentChunkId] = useState<number | null>(null);
  const [prompt, setPrompt] = useState("");
  const [csvText, setCsvText] = useState("");
  const [preview, setPreview] = useState<PreviewResponse | null>(null);
  const [message, setMessage] = useState("");
  const [invalidRetryRows, setInvalidRetryRows] = useState<InvalidRetryRow[]>([]);
  const [retryCsvText, setRetryCsvText] = useState("");
  const [configData, setConfigData] = useState<AppConfig | null>(null);
  const [configDraft, setConfigDraft] = useState<AppConfig | null>(null);
  const [configError, setConfigError] = useState("");
  const [isSavingConfig, setIsSavingConfig] = useState(false);
  const [isGeneratingRefill, setIsGeneratingRefill] = useState(false);
  const [isActivatingRefill, setIsActivatingRefill] = useState(false);
  const [isMerging, setIsMerging] = useState(false);

  const selectedProjectConfig = configDraft?.projects?.[projectId] ?? null;
  const configDirty =
    configData !== null &&
    configDraft !== null &&
    JSON.stringify(configData) !== JSON.stringify(configDraft);

  async function loadProject(selectedProjectId: string) {
    const [summaryData, chunksData] = await Promise.all([
      fetchJson<ProjectSummary>(`/projects/${selectedProjectId}/summary`),
      fetchJson<{ chunks: ChunkSummary[] }>(`/projects/${selectedProjectId}/chunks`),
    ]);
    setSummary(summaryData);
    setChunks(chunksData.chunks);
    setInvalidRetryRows(summaryData.invalid_retry_rows ?? []);
    const nextChunk = summaryData.next_chunk?.chunk_id ?? chunksData.chunks[0]?.chunk_id ?? null;
    setCurrentChunkId(nextChunk);
  }

  async function loadAppConfig() {
    const config = await fetchJson<AppConfig>("/config");
    setConfigData(config);
    setConfigDraft(cloneConfig(config));
  }

  async function loadPrompt(chunkId: number) {
    const data = await fetchJson<{ prompt: string }>(`/projects/${projectId}/chunks/${chunkId}/prompt`);
    setPrompt(data.prompt);
  }

  useEffect(() => {
    void loadProject(projectId);
  }, [projectId]);

  useEffect(() => {
    void loadAppConfig();
  }, []);

  useEffect(() => {
    if (currentChunkId !== null) {
      void loadPrompt(currentChunkId);
    }
  }, [currentChunkId, projectId]);

  async function handlePreview() {
    if (currentChunkId === null) {
      return;
    }
    const data = await fetchJson<PreviewResponse>(`/projects/${projectId}/chunks/${currentChunkId}/preview-import`, {
      method: "POST",
      body: JSON.stringify({ csv_text: csvText }),
    });
    setPreview(data);
    if (data.ok && data.invalid_retry_rows) {
      setInvalidRetryRows(data.invalid_retry_rows);
    }
    setMessage(data.ok ? "Preview ready. Review the summary and row-level errors below before accepting." : formatPreviewFailure(data));
  }

  async function handleAccept() {
    if (currentChunkId === null) {
      return;
    }
    const data = await fetchJson<AcceptResponse>(
      `/projects/${projectId}/chunks/${currentChunkId}/accept-import`,
      {
        method: "POST",
        body: JSON.stringify({ csv_text: csvText }),
      },
    );
    setMessage(
      data.ok
        ? `Chunk accepted. ${data.merge_summary ? `Latest merge now contains ${data.merge_summary.summary.approved_rows} approved row(s) and ${data.merge_summary.summary.error_rows} merge error row(s).` : ""}`.trim()
        : formatAcceptFailure(data),
    );
    if (data.ok) {
      setPreview(null);
      setCsvText("");
      await loadProject(projectId);
      if (data.next_chunk) {
        setCurrentChunkId(data.next_chunk.chunk_id);
      }
    }
  }

  async function handleRetryPreview() {
    if (currentChunkId === null) {
      return;
    }
    const data = await fetchJson<PreviewResponse>(
      `/projects/${projectId}/chunks/${currentChunkId}/preview-retry-import`,
      {
        method: "POST",
        body: JSON.stringify({ csv_text: retryCsvText }),
      },
    );
    if (data.ok) {
      setPreview(data);
      setInvalidRetryRows(data.invalid_retry_rows ?? []);
      setMessage(
        `Retry preview ready. Resolved ${data.resolved_retry_rows?.length ?? 0} row(s), still invalid ${data.still_invalid_retry_rows?.length ?? 0}, skipped ${data.skipped_cached_rows?.length ?? 0}.`,
      );
    } else {
      setMessage(formatPreviewFailure(data));
    }
  }

  async function handleApplyRetry() {
    if (currentChunkId === null) {
      return;
    }
    const data = await fetchJson<PreviewResponse>(
      `/projects/${projectId}/chunks/${currentChunkId}/apply-retry-import`,
      {
        method: "POST",
        body: JSON.stringify({ csv_text: retryCsvText }),
      },
    );
    if (data.ok) {
      setPreview(data);
      setInvalidRetryRows(data.invalid_retry_rows ?? []);
      if (data.merged_agent_csv) {
        setCsvText(data.merged_agent_csv);
      }
      setMessage(
        `Retry fixes applied to the working preview. Resolved ${data.resolved_retry_rows?.length ?? 0} row(s), dropped ${data.dropped_backlog_row_count ?? 0} row(s) to backlog. The main import textarea now contains the reduced chunk CSV.`,
      );
      await loadProject(projectId);
    } else {
      setMessage(formatPreviewFailure(data));
    }
  }

  async function handleClearInvalidCache() {
    const data = await fetchJson<{ ok: boolean; cleared_rows: number; invalid_retry_rows: InvalidRetryRow[] }>(
      `/projects/${projectId}/clear-invalid-retry-rows`,
      {
        method: "POST",
      },
    );
    setInvalidRetryRows(data.invalid_retry_rows ?? []);
    setMessage(`Cleared ${data.cleared_rows} invalid cached row(s).`);
    await loadProject(projectId);
  }

  async function handleSkipInvalidCache() {
    if (currentChunkId === null) {
      return;
    }
    try {
      const data = await fetchJson<SkipInvalidRetryCacheResponse>(
        `/projects/${projectId}/chunks/${currentChunkId}/skip-invalid-retry-cache`,
        {
          method: "POST",
        },
      );
      if (!data.ok) {
        setMessage(`Skip failed: ${data.error ?? "unknown error"}.`);
        return;
      }
      setPreview(null);
      setCsvText("");
      setRetryCsvText("");
      setMessage(
        `Skipped the invalid retry cache for the current chunk and advanced the queue.${data.next_chunk ? ` Next chunk: ${data.next_chunk.file_name}.` : ""}`,
      );
      await loadProject(projectId);
      if (data.next_chunk?.chunk_id !== undefined) {
        setCurrentChunkId(data.next_chunk.chunk_id);
        await loadPrompt(data.next_chunk.chunk_id);
      }
    } catch (error) {
      setMessage(extractApiErrorMessage(error));
    }
  }

  async function handleGenerateRefillRound() {
    setIsGeneratingRefill(true);
    try {
      const data = await fetchJson<RefillRoundResponse>(`/projects/${projectId}/generate-refill-round`, {
        method: "POST",
      });
      if (!data.ok) {
        setMessage(`Refill generation failed: ${data.error ?? "unknown error"}.`);
      } else {
        setMessage(
          `Generated refill round ${data.round_number} with ${data.sampled_count} row(s) in ${data.chunk_count} chunk(s). Manifest: ${data.manifest_path}`,
        );
      }
      await loadProject(projectId);
    } catch (error) {
      setMessage(extractApiErrorMessage(error));
    } finally {
      setIsGeneratingRefill(false);
    }
  }

  async function handleActivateRefillRound() {
    setIsActivatingRefill(true);
    try {
      const data = await fetchJson<ActivateRefillRoundResponse>(
        `/projects/${projectId}/activate-refill-round`,
        {
          method: "POST",
          body: JSON.stringify({}),
        },
      );
      if (!data.ok) {
        setMessage(`Refill activation failed: ${data.error ?? "unknown error"}.`);
      } else {
        setMessage(
          `Activated ${data.active_round}. The live queue now uses ${data.chunk_count ?? 0} refill chunk(s) from ${data.manifest_path}.`,
        );
        await loadAppConfig();
        await loadProject(projectId);
        if (data.next_chunk?.chunk_id !== undefined) {
          setCurrentChunkId(data.next_chunk.chunk_id);
          await loadPrompt(data.next_chunk.chunk_id);
        }
      }
    } catch (error) {
      setMessage(extractApiErrorMessage(error));
    } finally {
      setIsActivatingRefill(false);
    }
  }

  async function handleMerge() {
    setIsMerging(true);
    try {
      const data = await fetchJson<{
        approved_rows: number;
        error_rows: number;
      }>(`/projects/${projectId}/merge`, {
        method: "POST",
      });
      setMessage(
        `Merge completed. Approved rows: ${data.approved_rows}. Error rows: ${data.error_rows}.`,
      );
      await loadProject(projectId);
    } catch (error) {
      setMessage(extractApiErrorMessage(error));
    } finally {
      setIsMerging(false);
    }
  }

  function updateDraft(mutator: (draft: AppConfig) => AppConfig) {
    setConfigDraft((current) => {
      if (!current) {
        return current;
      }
      return mutator(cloneConfig(current));
    });
  }

  function updateProjectField<K extends keyof ProjectConfig>(key: K, value: ProjectConfig[K]) {
    updateDraft((draft) => ({
      ...draft,
      projects: {
        ...draft.projects,
        [projectId]: {
          ...draft.projects[projectId],
          [key]: value,
        },
      },
    }));
  }

  function updateUncertaintyMarker(index: number, value: string) {
    if (!selectedProjectConfig) {
      return;
    }
    const nextMarkers = [...selectedProjectConfig.uncertainty_markers];
    nextMarkers[index] = value;
    updateProjectField("uncertainty_markers", nextMarkers);
  }

  function addUncertaintyMarker() {
    if (!selectedProjectConfig) {
      return;
    }
    updateProjectField("uncertainty_markers", [...selectedProjectConfig.uncertainty_markers, ""]);
  }

  function removeUncertaintyMarker(index: number) {
    if (!selectedProjectConfig) {
      return;
    }
    updateProjectField(
      "uncertainty_markers",
      selectedProjectConfig.uncertainty_markers.filter((_, markerIndex) => markerIndex !== index),
    );
  }

  function handleResetConfigDraft() {
    if (!configData) {
      return;
    }
    setConfigDraft(cloneConfig(configData));
    setConfigError("");
  }

  async function handleSaveConfig() {
    if (!configDraft) {
      return;
    }
    setIsSavingConfig(true);
    setConfigError("");
    try {
      const savedConfig = await fetchJson<AppConfig>("/config", {
        method: "PUT",
        body: JSON.stringify({ config: configDraft }),
      });
      setConfigData(savedConfig);
      setConfigDraft(cloneConfig(savedConfig));
      setMessage("Configuration saved.");
      await loadProject(projectId);
      if (currentChunkId !== null) {
        await loadPrompt(currentChunkId);
      }
    } catch (error) {
      setConfigError(formatConfigSaveError(error));
    } finally {
      setIsSavingConfig(false);
    }
  }

  const currentChunk = chunks.find((chunk) => chunk.chunk_id === currentChunkId) || null;
  const requiresRowIdOutput = (summary?.row_matching ?? selectedProjectConfig?.row_matching) === "strict_row_id";

  return (
    <div className="page">
      <header className="hero">
        <div>
          <p className="eyebrow">Annotation Coordinator</p>
          <h1>Chunk-driven masking workflow</h1>
          <p className="lede">Copy the prompt, send the chunk to an external AI, paste the edited CSV back, preview changes, and accept only validated imports.</p>
        </div>
        <div className="panel controls">
          <label>
            Project
            <select value={projectId} onChange={(event) => setProjectId(event.target.value)}>
              {Object.keys(configDraft?.projects ?? { pilot: true, full: true }).map((projectKey) => (
                <option key={projectKey} value={projectKey}>
                  {projectKey}
                </option>
              ))}
            </select>
          </label>
          <button onClick={() => currentChunkId !== null && navigator.clipboard.writeText(prompt)}>Copy Prompt</button>
        </div>
      </header>

      <main className="layout">
        <section className="panel">
          <h2>Dashboard</h2>
          {summary ? (
            <>
              <p>Total chunks: {summary.total_chunks}</p>
              <p>Target approved rows: {summary.target_row_count}</p>
              <p>Approved rows: {summary.approved_rows_total}</p>
              <p>Rows remaining: {summary.rows_remaining_to_target}</p>
              <p>Backlog rows: {summary.backlog_row_count}</p>
              <p>Active round: {summary.active_round}</p>
              <p>Latest generated round: {summary.current_round}</p>
              <p>Active manifest: {summary.active_source_manifest_path}</p>
              <p>Next chunk: {summary.next_chunk?.file_name ?? "none"}</p>
              <p>Auto merge: {summary.auto_merge ? "on" : "off"}</p>
              <p>Auto advance: {summary.auto_advance ? "on" : "off"}</p>
              <p>Prompt row IDs: {summary.include_row_id_in_prompt ? "on" : "off"}</p>
              <p>Row matching: {summary.row_matching}</p>
              <p>Validate original_text with row_id: {summary.validate_original_text_with_row_id ? "on" : "off"}</p>
              <p>Cached invalid rows: {summary.invalid_retry_row_count}</p>
              <div className="actions">
                <button onClick={handleMerge} disabled={isMerging}>
                  {isMerging ? "Merging..." : "Run Merge Now"}
                </button>
              </div>
              {summary.merge_summary ? (
                <>
                  <p>Latest merge approved rows: {summary.merge_summary.approved_rows}</p>
                  <p>Latest merge error rows: {summary.merge_summary.error_rows}</p>
                  <p>
                    Latest merge dialects:{" "}
                    {Object.entries(summary.merge_summary.dialect_counts)
                      .map(([dialect, count]) => `${dialect}=${count}`)
                      .join(", ") || "none"}
                  </p>
                </>
              ) : (
                <p>No merge has been run yet.</p>
              )}
              <div className="stats">
                {Object.entries(summary.state_counts).map(([key, value]) => (
                  <div key={key} className="stat">
                    <strong>{value}</strong>
                    <span>{key}</span>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <p>Loading summary…</p>
          )}
        </section>

        <section className="panel">
          <h2>Chunk Workspace</h2>
          {currentChunk ? (
            <>
              <p>Current chunk: {currentChunk.file_name}</p>
              <p>Rows: {currentChunk.row_count}</p>
              <p>Accepted rows in chunk: {currentChunk.accepted_row_count ?? 0}</p>
              <p>Dropped to backlog: {currentChunk.dropped_backlog_row_count ?? 0}</p>
              <p className="path">{currentChunk.source_path}</p>
              <label className="block">
                Prompt
                <textarea value={prompt} readOnly rows={14} />
              </label>
              <label className="block">
                Pasted CSV
                <textarea
                  value={csvText}
                  onChange={(event) => setCsvText(event.target.value)}
                  rows={14}
                  placeholder={
                    requiresRowIdOutput
                      ? "Paste the agent CSV. Accepted headers are row_id,original_text,masked_text."
                      : "Paste the agent CSV. Accepted headers are original_text,masked_text or row_id,original_text,masked_text."
                  }
                />
              </label>
              <div className="actions">
                <button onClick={handlePreview}>Preview Import</button>
                <button onClick={handleAccept} disabled={!preview?.ok}>
                  Accept Import
                </button>
              </div>
              {message ? <p className="message">{message}</p> : null}
            </>
          ) : (
            <p>No chunk selected.</p>
          )}
        </section>

        <section className="panel">
          <div className="section-head">
            <div className="section-title">
              <h2>Backlog</h2>
              <span className="counter-pill">{summary?.backlog_row_count ?? 0} row(s)</span>
            </div>
            <button
              onClick={handleGenerateRefillRound}
              disabled={(summary?.backlog_row_count ?? 0) === 0 || isGeneratingRefill}
            >
              {isGeneratingRefill ? "Generating..." : "Generate Refill Round"}
            </button>
            <button
              onClick={handleActivateRefillRound}
              disabled={(summary?.current_round ?? 0) === 0 || isActivatingRefill}
            >
              {isActivatingRefill ? "Activating..." : "Activate Latest Refill Round"}
            </button>
          </div>
          {summary ? <p className="path">Base manifest: {summary.base_source_manifest_path}</p> : null}
          {summary && summary.backlog_rows.length > 0 ? (
            <div className="invalid-list">
              {summary.backlog_rows.slice(0, 10).map((row: BacklogRow) => (
                <article key={`backlog-${row.example_id}`} className="invalid-item">
                  <p><strong>{row.chunk_file_name}</strong> · {row.example_id}</p>
                  <p><strong>Row ID</strong>: {row.source_row_id}</p>
                  <p><strong>Dialect</strong>: {row.dialect}</p>
                  <p><strong>Attempts</strong>: {row.attempt_count}</p>
                  <p><strong>Original</strong>: {row.original_text}</p>
                  <p><strong>Latest error</strong>: {row.latest_errors || "(none)"}</p>
                </article>
              ))}
            </div>
          ) : (
            <p>No backlog rows right now.</p>
          )}
        </section>

        <section className="panel">
          <div className="section-head">
            <div className="section-title">
              <h2>Invalid Row Retry Cache</h2>
              <span className="counter-pill">{invalidRetryRows.length} row(s)</span>
            </div>
            <button
              onClick={() =>
                navigator.clipboard.writeText(
                  formatInvalidRetryRowsForCopy(invalidRetryRows, summary?.row_matching ?? "prefer_row_id"),
                )
              }
              disabled={invalidRetryRows.length === 0}
            >
              Copy Invalid Rows
            </button>
            <button onClick={handleClearInvalidCache} disabled={invalidRetryRows.length === 0}>
              Clear Invalid Cache
            </button>
            <button onClick={handleSkipInvalidCache} disabled={invalidRetryRows.length === 0 || currentChunkId === null}>
              Skip Cache And Advance
            </button>
          </div>
          {invalidRetryRows.length > 0 ? (
            <>
              <label className="block">
                Paste Fixed Invalid Rows
                <textarea
                  value={retryCsvText}
                  onChange={(event) => setRetryCsvText(event.target.value)}
                  rows={10}
                  placeholder={
                    requiresRowIdOutput
                      ? "Paste a three-column CSV with row_id, original_text, and masked_text for the invalid rows you retried."
                      : "Paste a two-column CSV with original_text and masked_text for the invalid rows you retried."
                  }
                />
              </label>
              <div className="actions">
                <button onClick={handleRetryPreview} disabled={retryCsvText.trim().length === 0}>
                  Preview Retry Merge
                </button>
                <button onClick={handleApplyRetry} disabled={retryCsvText.trim().length === 0}>
                  Apply Retry Fixes To Preview
                </button>
              </div>
              <p>These rows failed validation in earlier previews. They stay cached until their chunk is accepted, so you can retry them with another LLM pass.</p>
              <div className="invalid-list">
                {invalidRetryRows.slice(0, 12).map((row) => (
                  <article key={`${row.chunk_id}-${row.example_id}`} className="invalid-item">
                    <p><strong>{row.chunk_file_name}</strong> · {row.example_id}</p>
                    {row.row_id ? <p><strong>Row ID</strong>: {row.row_id}</p> : null}
                    <p><strong>Status</strong>: {row.retry_status ?? "open"}</p>
                    <p><strong>Original</strong>: {row.original_text}</p>
                    <p><strong>Attempted masked</strong>: {row.attempted_masked_text || "(empty)"}</p>
                    <p><strong>Why invalid</strong>: {row.errors.join(" | ")}</p>
                  </article>
                ))}
              </div>
            </>
          ) : (
            <p>No cached invalid rows right now.</p>
          )}
        </section>

        <section className="panel">
          <h2>Preview</h2>
          {preview ? (
            preview.ok ? (
              <>
                <p>Changed rows: {preview.summary?.changed_rows}</p>
                <p>Approved rows: {preview.summary?.approved_rows}</p>
                <p>Pending rows: {preview.summary?.pending_rows}</p>
                <p>Invalid rows: {preview.summary?.invalid_rows}</p>
                {preview.validation_rows?.some((row) => row.status === "invalid") ? (
                  <>
                    <h3>Invalid Row Preview</h3>
                    <p>These rows are currently blocked and explain exactly why the import is invalid.</p>
                    <div className="invalid-list">
                      {preview.validation_rows
                        ?.filter((row) => row.status === "invalid")
                        .map((row) => (
                          <article key={`invalid-${row.example_id}`} className="invalid-item">
                            <p><strong>{row.example_id}</strong></p>
                            <p><strong>Original</strong>: {row.original_text ?? ""}</p>
                            <p><strong>Why invalid</strong>: {row.errors.join(" | ")}</p>
                          </article>
                        ))}
                    </div>
                  </>
                ) : null}
                {preview.resolved_retry_rows || preview.still_invalid_retry_rows || preview.skipped_cached_rows ? (
                  <>
                    <h3>Retry Merge Result</h3>
                    <p>Resolved: {preview.resolved_retry_rows?.length ?? 0} · Still invalid: {preview.still_invalid_retry_rows?.length ?? 0} · Skipped cached rows: {preview.skipped_cached_rows?.length ?? 0} · Dropped to backlog on apply: {preview.dropped_backlog_row_count ?? 0}</p>
                  </>
                ) : null}
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Example</th>
                        <th>Original</th>
                        <th>Status</th>
                        <th>Errors</th>
                      </tr>
                    </thead>
                    <tbody>
                      {preview.validation_rows?.slice(0, 12).map((row) => (
                        <tr key={row.example_id}>
                          <td>{row.example_id}</td>
                          <td>{row.original_text ?? ""}</td>
                          <td>{row.status}</td>
                          <td>{row.errors.join(", ") || "none"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : (
              <div className="error-card">
                <p>{formatPreviewFailure(preview)}</p>
                {preview.error === "header_mismatch" ? (
                  <>
                    <p><strong>Accepted headers</strong>: {preview.accepted_header_sets?.map((headers) => headers.join(", ")).join(" or ") || preview.expected_headers?.join(", ") || "none"}</p>
                    <p><strong>Received headers</strong>: {preview.received_headers?.join(", ") || "none"}</p>
                  </>
                ) : null}
              </div>
            )
          ) : (
            <p>Paste a CSV and run preview.</p>
          )}
        </section>

        <section className="panel panel-wide">
          <div className="section-head">
            <div className="section-title">
              <h2>Settings</h2>
              {configDirty ? <span className="counter-pill">Unsaved</span> : null}
            </div>
            <div className="settings-actions">
              <button onClick={handleResetConfigDraft} disabled={!configDirty || !configDraft}>
                Reset Changes
              </button>
              <button onClick={handleSaveConfig} disabled={!configDirty || !configDraft || isSavingConfig}>
                {isSavingConfig ? "Saving..." : "Save Config"}
              </button>
            </div>
          </div>
          {configDraft && selectedProjectConfig ? (
            <>
              <div className="settings-grid">
                <label className="block">
                  Default project
                  <select
                    value={configDraft.default_project}
                    onChange={(event) =>
                      updateDraft((draft) => ({
                        ...draft,
                        default_project: event.target.value,
                      }))
                    }
                  >
                    {Object.keys(configDraft.projects).map((projectKey) => (
                      <option key={projectKey} value={projectKey}>
                        {projectKey}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="block checkbox-row">
                  <input
                    type="checkbox"
                    checked={configDraft.ui.copy_compact_prompt}
                    onChange={(event) =>
                      updateDraft((draft) => ({
                        ...draft,
                        ui: {
                          ...draft.ui,
                          copy_compact_prompt: event.target.checked,
                        },
                      }))
                    }
                  />
                  Copy compact prompt
                </label>
                <label className="block">
                  Project label
                  <input
                    type="text"
                    value={selectedProjectConfig.label}
                    onChange={(event) => updateProjectField("label", event.target.value)}
                  />
                </label>
                <label className="block">
                  Dataset mode
                  <input
                    type="text"
                    value={selectedProjectConfig.dataset_mode}
                    onChange={(event) => updateProjectField("dataset_mode", event.target.value)}
                  />
                </label>
                <label className="block">
                  Workspace dir
                  <input
                    type="text"
                    value={selectedProjectConfig.workspace_dir}
                    onChange={(event) => updateProjectField("workspace_dir", event.target.value)}
                  />
                </label>
                <label className="block">
                  Base source chunks dir
                  <input
                    type="text"
                    value={selectedProjectConfig.base_source_chunks_dir}
                    onChange={(event) => updateProjectField("base_source_chunks_dir", event.target.value)}
                  />
                </label>
                <label className="block">
                  Base source manifest path
                  <input
                    type="text"
                    value={selectedProjectConfig.base_source_manifest_path}
                    onChange={(event) => updateProjectField("base_source_manifest_path", event.target.value)}
                  />
                </label>
                <label className="block">
                  Source chunks dir
                  <input
                    type="text"
                    value={selectedProjectConfig.source_chunks_dir}
                    onChange={(event) => updateProjectField("source_chunks_dir", event.target.value)}
                  />
                </label>
                <label className="block">
                  Source manifest path
                  <input
                    type="text"
                    value={selectedProjectConfig.source_manifest_path}
                    onChange={(event) => updateProjectField("source_manifest_path", event.target.value)}
                  />
                </label>
                <label className="block">
                  Active round
                  <input
                    type="text"
                    value={selectedProjectConfig.active_round}
                    onChange={(event) => updateProjectField("active_round", event.target.value)}
                  />
                </label>
                <label className="block">
                  Prompt template path
                  <input
                    type="text"
                    value={selectedProjectConfig.prompt_template_path}
                    onChange={(event) => updateProjectField("prompt_template_path", event.target.value)}
                  />
                </label>
                <label className="block">
                  Masking guidelines path
                  <input
                    type="text"
                    value={selectedProjectConfig.masking_guidelines_path}
                    onChange={(event) => updateProjectField("masking_guidelines_path", event.target.value)}
                  />
                </label>
                <label className="block">
                  Agent import schema
                  <select
                    value={selectedProjectConfig.agent_import_schema}
                    onChange={(event) => updateProjectField("agent_import_schema", event.target.value)}
                  >
                    <option value="original_masked_v1">original_masked_v1</option>
                  </select>
                </label>
                <label className="block">
                  Row matching mode
                  <select
                    value={selectedProjectConfig.row_matching}
                    onChange={(event) => updateProjectField("row_matching", event.target.value)}
                  >
                    <option value="prefer_row_id">prefer_row_id</option>
                    <option value="strict_row_id">strict_row_id</option>
                    <option value="strict_original_text">strict_original_text</option>
                  </select>
                </label>
                <label className="block checkbox-row">
                  <input
                    type="checkbox"
                    checked={selectedProjectConfig.auto_advance}
                    onChange={(event) => updateProjectField("auto_advance", event.target.checked)}
                  />
                  Auto advance
                </label>
                <label className="block checkbox-row">
                  <input
                    type="checkbox"
                    checked={selectedProjectConfig.auto_merge}
                    onChange={(event) => updateProjectField("auto_merge", event.target.checked)}
                  />
                  Auto merge
                </label>
                <label className="block checkbox-row">
                  <input
                    type="checkbox"
                    checked={selectedProjectConfig.allow_pending_accept}
                    onChange={(event) => updateProjectField("allow_pending_accept", event.target.checked)}
                  />
                  Allow pending accept
                </label>
                <label className="block checkbox-row">
                  <input
                    type="checkbox"
                    checked={selectedProjectConfig.include_row_id_in_prompt}
                    onChange={(event) => updateProjectField("include_row_id_in_prompt", event.target.checked)}
                  />
                  Include row_id in prompt
                </label>
                <label className="block checkbox-row">
                  <input
                    type="checkbox"
                    checked={selectedProjectConfig.validate_original_text_with_row_id}
                    onChange={(event) =>
                      updateProjectField("validate_original_text_with_row_id", event.target.checked)
                    }
                  />
                  Validate original_text against row_id
                </label>
              </div>
              <div className="markers-panel">
                <div className="section-head">
                  <h3>Uncertainty markers</h3>
                  <button onClick={addUncertaintyMarker}>Add Marker</button>
                </div>
                <div className="marker-list">
                  {selectedProjectConfig.uncertainty_markers.map((marker, index) => (
                    <div key={`${projectId}-marker-${index}`} className="marker-row">
                      <input
                        type="text"
                        value={marker}
                        onChange={(event) => updateUncertaintyMarker(index, event.target.value)}
                      />
                      <button onClick={() => removeUncertaintyMarker(index)}>Remove</button>
                    </div>
                  ))}
                </div>
              </div>
              {configError ? <p className="error-text">{configError}</p> : null}
            </>
          ) : (
            <p>Loading config…</p>
          )}
        </section>

        <section className="panel">
          <h2>Chunk Queue</h2>
          <div className="queue">
            {chunks.slice(0, 24).map((chunk) => (
              <button
                key={chunk.chunk_id}
                className={`queue-item ${chunk.chunk_id === currentChunkId ? "active" : ""}`}
                onClick={() => setCurrentChunkId(chunk.chunk_id)}
              >
                <span>{chunk.file_name}</span>
                <small>{chunk.state} · kept {chunk.accepted_row_count ?? 0}/{chunk.row_count}</small>
              </button>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;

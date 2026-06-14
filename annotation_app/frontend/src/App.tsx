import { useEffect, useState } from "react";
import type { AcceptResponse, ChunkSummary, PreviewResponse, ProjectSummary } from "./types";

const API_BASE = "http://127.0.0.1:8000/api";

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

function formatPreviewFailure(preview: PreviewResponse | null): string {
  if (!preview || preview.ok) {
    return "";
  }

  switch (preview.error) {
    case "header_mismatch":
      return `The pasted CSV headers do not match this chunk. Expected ${preview.expected_headers?.length ?? 0} columns: ${preview.expected_headers?.join(", ") || "none"}. Received ${preview.received_headers?.length ?? 0} columns: ${preview.received_headers?.join(", ") || "none"}.`;
    case "row_count_mismatch":
      return `The pasted CSV has the wrong number of rows for this chunk. Expected ${preview.expected_row_count ?? 0} row(s), but received ${preview.received_row_count ?? 0}. This usually means the response is truncated, includes the wrong chunk, or is missing rows.`;
    case "example_id_mismatch": {
      const expected = preview.expected_example_ids ?? [];
      const received = preview.received_example_ids ?? [];
      let firstMismatch = -1;
      const maxLength = Math.max(expected.length, received.length);
      for (let index = 0; index < maxLength; index += 1) {
        if (expected[index] !== received[index]) {
          firstMismatch = index;
          break;
        }
      }
      if (firstMismatch >= 0) {
        return `The pasted CSV row order or row identity does not match this chunk. The first mismatch is at row ${firstMismatch + 1}: expected example_id ${JSON.stringify(expected[firstMismatch] ?? null)}, received ${JSON.stringify(received[firstMismatch] ?? null)}.`;
      }
      return "The pasted CSV example_id list does not match the current chunk.";
    }
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

function App() {
  const [projectId, setProjectId] = useState("pilot");
  const [summary, setSummary] = useState<ProjectSummary | null>(null);
  const [chunks, setChunks] = useState<ChunkSummary[]>([]);
  const [currentChunkId, setCurrentChunkId] = useState<number | null>(null);
  const [prompt, setPrompt] = useState("");
  const [csvText, setCsvText] = useState("");
  const [preview, setPreview] = useState<PreviewResponse | null>(null);
  const [message, setMessage] = useState("");

  async function loadProject(selectedProjectId: string) {
    const [summaryData, chunksData] = await Promise.all([
      fetchJson<ProjectSummary>(`/projects/${selectedProjectId}/summary`),
      fetchJson<{ chunks: ChunkSummary[] }>(`/projects/${selectedProjectId}/chunks`),
    ]);
    setSummary(summaryData);
    setChunks(chunksData.chunks);
    const nextChunk = summaryData.next_chunk?.chunk_id ?? chunksData.chunks[0]?.chunk_id ?? null;
    setCurrentChunkId(nextChunk);
  }

  async function loadPrompt(chunkId: number) {
    const data = await fetchJson<{ prompt: string }>(`/projects/${projectId}/chunks/${chunkId}/prompt`);
    setPrompt(data.prompt);
  }

  useEffect(() => {
    void loadProject(projectId);
  }, [projectId]);

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

  const currentChunk = chunks.find((chunk) => chunk.chunk_id === currentChunkId) || null;

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
              <option value="pilot">pilot</option>
              <option value="full">full</option>
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
              <p>Next chunk: {summary.next_chunk?.file_name ?? "none"}</p>
              <p>Auto merge: {summary.auto_merge ? "on" : "off"}</p>
              <p>Auto advance: {summary.auto_advance ? "on" : "off"}</p>
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
                  placeholder="Paste the fully edited CSV for the current chunk."
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
          <h2>Preview</h2>
          {preview ? (
            preview.ok ? (
              <>
                <p>Changed rows: {preview.summary?.changed_rows}</p>
                <p>Approved rows: {preview.summary?.approved_rows}</p>
                <p>Pending rows: {preview.summary?.pending_rows}</p>
                <p>Invalid rows: {preview.summary?.invalid_rows}</p>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Example</th>
                        <th>Status</th>
                        <th>Errors</th>
                      </tr>
                    </thead>
                    <tbody>
                      {preview.validation_rows?.slice(0, 12).map((row) => (
                        <tr key={row.example_id}>
                          <td>{row.example_id}</td>
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
                    <p><strong>Expected headers</strong>: {preview.expected_headers?.join(", ") || "none"}</p>
                    <p><strong>Received headers</strong>: {preview.received_headers?.join(", ") || "none"}</p>
                  </>
                ) : null}
              </div>
            )
          ) : (
            <p>Paste a CSV and run preview.</p>
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
                <small>{chunk.state}</small>
              </button>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;

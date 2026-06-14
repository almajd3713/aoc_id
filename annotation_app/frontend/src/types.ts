export type ChunkSummary = {
  chunk_id: number;
  file_name: string;
  row_count: number;
  state: string;
  source_path: string;
  has_accepted_copy: boolean;
};

export type ProjectSummary = {
  project_id: string;
  dataset_mode: string;
  total_chunks: number;
  state_counts: Record<string, number>;
  current_chunk_id: number | null;
  next_chunk: ChunkSummary | null;
  auto_merge: boolean;
  auto_advance: boolean;
  merge_summary?: {
    approved_rows: number;
    error_rows: number;
    dialect_counts: Record<string, number>;
  } | null;
};

export type PreviewResponse = {
  ok: boolean;
  error?: string;
  expected_headers?: string[];
  received_headers?: string[];
  expected_row_count?: number;
  received_row_count?: number;
  expected_example_ids?: string[];
  received_example_ids?: string[];
  summary?: {
    row_count: number;
    approved_rows: number;
    pending_rows: number;
    invalid_rows: number;
    changed_rows: number;
  };
  validation_rows?: Array<{
    row_number: number;
    example_id: string;
    status: string;
    errors: string[];
  }>;
  changed_rows?: Array<{
    example_id: string;
    changed_columns: Array<{
      column: string;
      before: string;
      after: string;
    }>;
  }>;
};

export type AcceptResponse = {
  ok: boolean;
  error?: string;
  preview?: PreviewResponse;
  next_chunk?: ChunkSummary;
  accepted_path?: string;
  auto_merge?: boolean;
  merge_summary?: {
    approved_rows: number;
    error_rows: number;
    summary: {
      approved_rows: number;
      error_rows: number;
      dialect_counts: Record<string, number>;
    };
  };
};

export type ProjectConfig = {
  label: string;
  dataset_mode: string;
  workspace_dir: string;
  base_source_chunks_dir: string;
  base_source_manifest_path: string;
  source_chunks_dir: string;
  source_manifest_path: string;
  active_round: string;
  prompt_template_path: string;
  masking_guidelines_path: string;
  auto_advance: boolean;
  auto_merge: boolean;
  allow_pending_accept: boolean;
  include_row_id_in_prompt: boolean;
  validate_original_text_with_row_id: boolean;
  agent_import_schema: string;
  row_matching: string;
  uncertainty_markers: string[];
};

export type AppConfig = {
  default_project: string;
  ui: {
    copy_compact_prompt: boolean;
  };
  projects: Record<string, ProjectConfig>;
};

export type ChunkSummary = {
  chunk_id: number;
  file_name: string;
  row_count: number;
  state: string;
  source_path: string;
  has_accepted_copy: boolean;
  accepted_row_count?: number;
  dropped_backlog_row_count?: number;
};

export type ProjectSummary = {
  project_id: string;
  dataset_mode: string;
  total_chunks: number;
  target_row_count: number;
  approved_rows_total: number;
  rows_remaining_to_target: number;
  state_counts: Record<string, number>;
  current_chunk_id: number | null;
  next_chunk: ChunkSummary | null;
  auto_merge: boolean;
  auto_advance: boolean;
  include_row_id_in_prompt: boolean;
  row_matching: string;
  validate_original_text_with_row_id: boolean;
  current_round: number;
  active_round: string;
  active_source_manifest_path: string;
  base_source_manifest_path: string;
  merge_summary?: {
    approved_rows: number;
    error_rows: number;
    dialect_counts: Record<string, number>;
  } | null;
  invalid_retry_rows: InvalidRetryRow[];
  invalid_retry_row_count: number;
  backlog_rows: BacklogRow[];
  backlog_row_count: number;
};

export type ConfigSaveError = {
  message: string;
  errors?: string[];
};

export type InvalidRetryRow = {
  chunk_id: number;
  chunk_file_name: string;
  example_id: string;
  row_id?: string;
  original_text: string;
  attempted_masked_text: string;
  errors: string[];
  retry_status?: string;
  last_seen_at?: string;
};

export type BacklogRow = {
  example_id: string;
  source_row_id: string;
  dialect: string;
  original_text: string;
  normalized_text: string;
  chunk_id: string;
  chunk_file_name: string;
  attempt_count: string;
  round_number: string;
  last_attempted_masked_text: string;
  latest_errors: string;
  dropped_at: string;
};

export type PreviewResponse = {
  ok: boolean;
  error?: string;
  expected_headers?: string[];
  accepted_header_sets?: string[][];
  received_headers?: string[];
  expected_row_count?: number;
  received_row_count?: number;
  expected_row_ids?: string[];
  received_row_ids?: string[];
  missing_row_ids?: string[];
  duplicate_row_ids?: string[];
  unknown_retry_row_ids?: string[];
  mismatched_rows?: Array<{
    row_id: string;
    expected_original_text: string;
    received_original_text: string;
  }>;
  expected_example_ids?: string[];
  received_example_ids?: string[];
  expected_original_texts?: string[];
  received_original_texts?: string[];
  duplicate_original_texts?: string[];
  missing_original_texts?: string[];
  unknown_retry_rows?: string[];
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
    original_text?: string;
    status: string;
    errors: string[];
  }>;
  invalid_retry_rows?: InvalidRetryRow[];
  merged_agent_csv?: string;
  resolved_retry_rows?: Array<{
    example_id: string;
    original_text: string;
  }>;
  still_invalid_retry_rows?: Array<{
    example_id: string;
    original_text: string;
    attempted_masked_text: string;
    errors: string[];
  }>;
  skipped_cached_rows?: InvalidRetryRow[];
  dropped_backlog_rows?: BacklogRow[];
  dropped_backlog_row_count?: number;
  backlog_row_count?: number;
  working_preview_path?: string;
  changed_rows?: Array<{
    example_id: string;
    changed_columns: Array<{
      column: string;
      before: string;
      after: string;
    }>;
  }>;
};

export type RefillRoundResponse = {
  ok: boolean;
  error?: string;
  round_number?: number;
  sampled_count?: number;
  chunk_count?: number;
  chunk_size?: number;
  dialect_counts?: Record<string, number>;
  manifest_path?: string;
  round_dir?: string;
};

export type ActivateRefillRoundResponse = {
  ok: boolean;
  error?: string;
  active_round?: string;
  round_number?: number;
  manifest_path?: string;
  chunks_dir?: string;
  chunk_count?: number;
  next_chunk?: ChunkSummary | null;
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

export type SkipInvalidRetryCacheResponse = {
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

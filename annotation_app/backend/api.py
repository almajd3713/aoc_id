#!/usr/bin/env python3
"""FastAPI app for the annotation coordinator."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from annotation_app.backend.workflow import (
    accept_import,
    build_prompt,
    get_chunk_detail,
    get_next_chunk,
    get_project_paths,
    get_project_summary,
    list_chunks,
    load_config,
    merge_chunks,
    preview_import,
    save_config,
)


class ConfigUpdate(BaseModel):
    config: Dict[str, Any]


class CsvImportRequest(BaseModel):
    csv_text: str


app = FastAPI(title="Annotation Coordinator", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
def get_config() -> Dict[str, Any]:
    return load_config()


@app.put("/api/config")
def put_config(payload: ConfigUpdate) -> Dict[str, Any]:
    save_config(payload.config)
    return payload.config


@app.get("/api/projects")
def get_projects() -> Dict[str, Any]:
    config = load_config()
    return {
        "default_project": config.get("default_project"),
        "projects": [{"project_id": key, **value} for key, value in config.get("projects", {}).items()],
    }


@app.get("/api/projects/{project_id}/summary")
def project_summary(project_id: str) -> Dict[str, Any]:
    try:
        return get_project_summary(project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/projects/{project_id}/chunks")
def project_chunks(project_id: str) -> Dict[str, Any]:
    try:
        return {"chunks": list_chunks(project_id)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/projects/{project_id}/chunks/next")
def project_next_chunk(project_id: str) -> Dict[str, Any]:
    try:
        return {"chunk": get_next_chunk(project_id)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/projects/{project_id}/chunks/{chunk_id}")
def chunk_detail(project_id: str, chunk_id: int) -> Dict[str, Any]:
    try:
        return get_chunk_detail(project_id, chunk_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/projects/{project_id}/chunks/{chunk_id}/prompt")
def chunk_prompt(project_id: str, chunk_id: int) -> Dict[str, Any]:
    try:
        return {"prompt": build_prompt(project_id, chunk_id)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/chunks/{chunk_id}/preview-import")
def chunk_preview_import(project_id: str, chunk_id: int, payload: CsvImportRequest) -> Dict[str, Any]:
    try:
        return preview_import(project_id, chunk_id, payload.csv_text)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/chunks/{chunk_id}/accept-import")
def chunk_accept_import(project_id: str, chunk_id: int, payload: CsvImportRequest) -> Dict[str, Any]:
    try:
        return accept_import(project_id, chunk_id, payload.csv_text)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/merge")
def project_merge(project_id: str) -> Dict[str, Any]:
    try:
        paths = get_project_paths(project_id)
        approved_rows, error_rows = merge_chunks(paths.accepted_dir, paths.merged_dir)
        return {
            "approved_rows": approved_rows,
            "error_rows": error_rows,
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/projects/{project_id}/merge-status")
def project_merge_status(project_id: str) -> Dict[str, Any]:
    try:
        return get_project_summary(project_id)["merge_summary"] or {}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


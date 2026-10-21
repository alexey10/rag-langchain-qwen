from uuid import uuid4

import hashlib
import logging
import os
import sqlite3
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.security import HTTPBearer

from app.api.client_registry import (
    APIClient,
    KnowledgeBaseAccessError,
)

from app.api.auth import verify_api_key
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    ModelsResponse,
    KnowledgeBasesResponse,
)
from app.gateway.router import get_provider
from app.gateway.availability import (
    ModelUnavailableError,
    OllamaUnavailableError,
    get_available_model_config,
    get_model_catalog,
)
from app.services.rag_service import RAGService

from app.api.knowledge_base_registry import (
    list_knowledge_bases,
    create_document_record,
    update_document_record_status,
    list_document_records,
    get_document_storage_path,
    delete_document_record,
    get_document_record,
)

from app.api.client_registry import authorize_knowledge_base
from app.api.knowledge_base_registry import (
    create_document_record,
    update_document_record_status,
    list_document_records,
)
from app.api.schemas import DocumentsResponse, DocumentInfo
from app.config import DOCUMENT_STORAGE_PATH, MAX_DOCUMENT_UPLOAD_BYTES
from app.ingestion.document_indexer import (
    index_pdf_document,
    remove_document_chunks,
)

security = HTTPBearer()

router = APIRouter(prefix="/v1")

rag_service = RAGService()


@router.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest,client: APIClient = Depends(verify_api_key),):
    try:
        if request.knowledge_base_id:
            content = rag_service.answer(
                messages=request.messages,
                knowledge_base_id=request.knowledge_base_id,
                model=request.model,
                client=client,
            )
        else:
            get_available_model_config(request.model)
            provider = get_provider(request.model)

            content = provider.chat(
                messages=request.messages,
            )

        return {
            "id": f"chatcmpl-{uuid4()}",
            "model": request.model,
            "content": content,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )
    except ModelUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )
    except OllamaUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )
    except KnowledgeBaseAccessError:
        raise HTTPException(
            status_code=404,
            detail="Knowledge base not found",
        )


@router.get(
    "/models",
    response_model=ModelsResponse,
    dependencies=[Depends(verify_api_key)],
)
def list_models():
    try:
        return {"models": get_model_catalog()}
    except OllamaUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )

@router.get(
    "/knowledge-bases",
    response_model=KnowledgeBasesResponse,
)
def list_client_knowledge_bases(
    client: APIClient = Depends(verify_api_key),
):
    active_knowledge_bases = list_knowledge_bases(
        status="active"
    )

    return {
        "knowledge_bases": [
            {
                "id": kb.id,
                "name": kb.name,
            }
            for kb in active_knowledge_bases
            if kb.id in client.allowed_knowledge_bases
        ]
    }

@router.post(
    "/knowledge-bases/{knowledge_base_id}/documents",
    response_model=DocumentInfo,
    status_code=201,
)
def upload_document(
    knowledge_base_id: str,
    file: UploadFile = File(...),
    client: APIClient = Depends(verify_api_key),
):
    # Authorization happens before any file is saved or indexed.
    try:
        authorize_knowledge_base(client, knowledge_base_id)
    except KnowledgeBaseAccessError:
        raise HTTPException(
            status_code=404,
            detail="Knowledge base not found",
        )

    filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]

    if not filename or Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=415,
            detail="Only PDF uploads are supported",
        )

    file.file.seek(0)
    pdf_header = file.file.read(5)
    file.file.seek(0)

    if pdf_header != b"%PDF-":
        raise HTTPException(
            status_code=415,
            detail="Uploaded file is not a valid PDF",
        )

    document_id = uuid4().hex
    storage_dir = Path(DOCUMENT_STORAGE_PATH) / knowledge_base_id
    storage_dir.mkdir(parents=True, exist_ok=True)

    temporary_path = storage_dir / f"{document_id}.part"
    stored_path = storage_dir / f"{document_id}.pdf"

    digest = hashlib.sha256()
    total_bytes = 0

    try:
        with temporary_path.open("wb") as destination:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > MAX_DOCUMENT_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Uploaded PDF exceeds the size limit",
                    )

                destination.write(chunk)
                digest.update(chunk)

        os.replace(temporary_path, stored_path)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    try:
        record = create_document_record(
            document_id=document_id,
            knowledge_base_id=knowledge_base_id,
            filename=filename,
            storage_path=str(stored_path),
            sha256=digest.hexdigest(),
            size_bytes=total_bytes,
        )

    except sqlite3.IntegrityError:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=409,
            detail="A document with identical contents is already registered in this knowledge base",
        )

    try:
        chunk_count = index_pdf_document(
            file_path=stored_path,
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
            original_filename=filename,
        )

        record = update_document_record_status(
            document_id,
            "indexed",
            chunk_count=chunk_count,
        )

        return asdict(record)

    except Exception:
        logging.exception(
            "Document indexing failed for document_id=%s",
            document_id,
        )

        # Remove any partially inserted vectors, if present.
        try:
            remove_document_chunks(document_id)
        except Exception:
            logging.exception(
                "Failed to clean up Chroma chunks for document_id=%s",
                document_id,
            )

        try:
            update_document_record_status(
                document_id,
                "failed",
                chunk_count=0,
            )
        except Exception:
            logging.exception(
                "Failed to update document status for document_id=%s",
                document_id,
            )

        raise HTTPException(
            status_code=500,
            detail="Document upload succeeded but indexing failed; document is marked failed",
        )


@router.get(
    "/knowledge-bases/{knowledge_base_id}/documents",
    response_model=DocumentsResponse,
)
def list_knowledge_base_documents(
    knowledge_base_id: str,
    client: APIClient = Depends(verify_api_key),
):
    try:
        authorize_knowledge_base(client, knowledge_base_id)
    except KnowledgeBaseAccessError:
        raise HTTPException(
            status_code=404,
            detail="Knowledge base not found",
        )

    records = list_document_records(knowledge_base_id)

    return {
        "documents": [
            asdict(record)
            for record in records
        ]
    }

@router.delete(
    "/knowledge-bases/{knowledge_base_id}/documents/{document_id}",
    status_code=204,
)
def delete_knowledge_base_document(
    knowledge_base_id: str,
    document_id: str,
    client: APIClient = Depends(verify_api_key),
):
    try:
        authorize_knowledge_base(client, knowledge_base_id)
    except KnowledgeBaseAccessError:
        raise HTTPException(
            status_code=404,
            detail="Knowledge base not found",
        )

    record = get_document_record(document_id)

    if (
        record is None
        or record.knowledge_base_id != knowledge_base_id
    ):
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    storage_path = get_document_storage_path(
        document_id,
        knowledge_base_id,
    )

    if storage_path is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    # Remove indexed chunks first. Retain the registry record if this fails.
    try:
        remove_document_chunks(document_id)
    except Exception:
        logging.exception(
            "Failed to remove Chroma chunks for document_id=%s",
            document_id,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to remove document chunks; document remains registered",
        )

    # Missing files are acceptable on retry.
    try:
        Path(storage_path).unlink(missing_ok=True)
    except OSError:
        logging.exception(
            "Failed to remove stored file for document_id=%s",
            document_id,
        )
        raise HTTPException(
            status_code=500,
            detail="Document chunks were removed but stored file cleanup failed",
        )

    try:
        deleted = delete_document_record(
            document_id,
            knowledge_base_id,
        )
    except Exception:
        logging.exception(
            "Failed to delete registry record for document_id=%s",
            document_id,
        )
        raise HTTPException(
            status_code=500,
            detail="Document files were cleaned up but registry deletion failed",
        )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    return None

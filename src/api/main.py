"""
FastAPI application for RAG Company Report Generator.
Provides REST API endpoints for data ingestion and report generation.
"""

import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..agent.graph import ReportGenerationGraph
from ..config import get_settings
from ..data_ingestion.ingestion_pipeline import IngestionPipeline
from ..observability.cost_tracker import get_cost_tracker
from ..observability.logger import get_logger, setup_logging
from ..retrieval.vector_store import VectorStore

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Global instances
vector_store: Optional[VectorStore] = None
ingestion_pipeline: Optional[IngestionPipeline] = None
report_graph: Optional[ReportGenerationGraph] = None


def create_vector_store() -> VectorStore:
    """Factory for the VectorStore used by the API.

    Extracted for easier testing and potential future customization.
    """
    return VectorStore()


def create_ingestion_pipeline(store: VectorStore) -> IngestionPipeline:
    """Factory for the IngestionPipeline used by the API."""
    return IngestionPipeline(store)


def create_report_graph(store: VectorStore) -> ReportGenerationGraph:
    """Factory for the ReportGenerationGraph used by the API."""
    return ReportGenerationGraph(store)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global vector_store, ingestion_pipeline, report_graph

    logger.info("api_startup")

    # Initialize components
    vector_store = create_vector_store()
    ingestion_pipeline = create_ingestion_pipeline(vector_store)
    report_graph = create_report_graph(vector_store)

    logger.info("api_ready")

    yield

    # Cleanup
    logger.info("api_shutdown")
    cost_tracker = get_cost_tracker()
    cost_tracker.log_session_summary()


# Create FastAPI app
settings = get_settings()
app = FastAPI(
    title="RAG Company Report Generator API",
    description="REST API for document ingestion and AI-powered report generation",
    version=settings.app_version,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API
class QueryRequest(BaseModel):
    """Request model for report generation."""

    query: str = Field(..., description="Question or query about the company data", min_length=1)
    save_report: bool = Field(default=False, description="Whether to save the report to file")
    format: str = Field(default="markdown", description="Output format (text or markdown)")


class CitationModel(BaseModel):
    """One source the report was allowed to cite, and what it points at."""

    id: str = Field(..., description="Marker used in the report, e.g. S1")
    source: str = Field(..., description="Document the excerpt came from")
    score: Optional[float] = Field(None, description="Retrieval distance; lower is closer")
    excerpt: str = Field(..., description="Start of the text that was sent to the model")


class CitationAuditModel(BaseModel):
    """Whether the report was entitled to the markers it used."""

    valid: List[str] = Field(default_factory=list)
    invalid: List[str] = Field(
        default_factory=list, description="Markers the report used that were never supplied"
    )
    unused: List[str] = Field(
        default_factory=list, description="Supplied sources the report did not need"
    )
    groundedness: float = Field(
        0.0, description="Share of the report's content words present in the cited text"
    )
    ungrounded_sample: List[str] = Field(default_factory=list)


class ReportResponse(BaseModel):
    """Response model for generated reports."""

    query: str
    report: str
    summary: str
    sources: List[str]
    # Provenance travels with the answer rather than being logged and
    # forgotten: a claim a reader cannot trace back to a document is a claim
    # they have to take on trust, which is the thing RAG is meant to remove.
    citations: List[CitationModel] = Field(default_factory=list)
    citation_audit: Optional[CitationAuditModel] = None
    groundedness: Optional[float] = None
    num_tokens_used: int
    total_cost_usd: float
    saved_file: Optional[str] = None


class IngestionResponse(BaseModel):
    """Response model for data ingestion."""

    message: str
    chunks_added: int
    collection_name: str
    total_documents: int


class CollectionStatsResponse(BaseModel):
    """Response model for collection statistics."""

    collection_name: str
    total_documents: int
    metadata: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    app_name: str
    version: str
    vector_store_documents: int


class CostSummaryResponse(BaseModel):
    """Response model for cost summary."""

    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    by_model: dict


# API Endpoints


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Company Report Generator API",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    try:
        stats = vector_store.get_collection_info()
        return HealthResponse(
            status="healthy",
            app_name=settings.app_name,
            version=settings.app_version,
            vector_store_documents=stats["count"],
        )
    except Exception as e:
        logger.error("health_check_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}",
        )


@app.get("/stats", response_model=CollectionStatsResponse, tags=["Data Management"])
async def get_collection_stats():
    """Get vector store collection statistics."""
    try:
        stats = vector_store.get_collection_info()
        return CollectionStatsResponse(
            collection_name=stats["name"],
            total_documents=stats["count"],
            metadata=stats.get("metadata", {}),
        )
    except Exception as e:
        logger.error("get_stats_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}",
        )


class TextIngestionRequest(BaseModel):
    """Request model for text ingestion."""

    text: str = Field(..., description="Text content to ingest", min_length=1)
    source_name: str = Field(default="api_upload", description="Source identifier")


@app.post("/ingest/text", response_model=IngestionResponse, tags=["Data Management"])
async def ingest_text(request: TextIngestionRequest):
    """Ingest text content directly."""
    try:
        from langchain.schema import Document

        # Create document
        doc = Document(page_content=request.text, metadata={"source": request.source_name})

        # Ingest
        chunks_added = ingestion_pipeline.ingest_documents([doc])

        # Get stats
        stats = vector_store.get_collection_info()

        logger.info("text_ingested", chunks_added=chunks_added, source=request.source_name)

        return IngestionResponse(
            message="Text ingested successfully",
            chunks_added=chunks_added,
            collection_name=stats["name"],
            total_documents=stats["count"],
        )
    except Exception as e:
        logger.error("text_ingestion_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text ingestion failed: {str(e)}",
        )


@app.post("/ingest/file", response_model=IngestionResponse, tags=["Data Management"])
async def ingest_file(file: UploadFile = File(...)):
    """Ingest a document file (PDF, DOCX, TXT, MD)."""
    try:
        import tempfile
        from pathlib import Path

        # Check file extension
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Upload is missing a filename, so its type cannot be determined.",
            )

        file_ext = Path(file.filename).suffix.lower()
        supported_extensions = [".pdf", ".txt", ".md", ".docx", ".doc"]

        if file_ext not in supported_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: {file_ext}. Supported: {supported_extensions}",
            )

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            # Ingest file
            chunks_added = ingestion_pipeline.ingest_file(tmp_file_path)

            # Get stats
            stats = vector_store.get_collection_info()

            logger.info(
                "file_ingested",
                filename=file.filename,
                chunks_added=chunks_added,
            )

            return IngestionResponse(
                message=f"File '{file.filename}' ingested successfully",
                chunks_added=chunks_added,
                collection_name=stats["name"],
                total_documents=stats["count"],
            )
        finally:
            # Cleanup temp file. missing_ok, because raising here inside a
            # finally would replace whatever the real failure was with a
            # FileNotFoundError about a temp file nobody asked about.
            Path(tmp_file_path).unlink(missing_ok=True)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("file_ingestion_failed", error=str(e), filename=file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File ingestion failed: {str(e)}",
        )


def saved_report_metadata(result: dict) -> dict:
    """
    Header fields for a report written to disk.

    A saved report outlives the response it came in, so the cost and the
    provenance have to travel with the file rather than only the JSON.
    """
    metadata = {
        "tokens_used": result.get("num_tokens_used", 0),
        "cost_usd": f"${result.get('total_cost', 0.0):.4f}",
    }
    audit = result.get("citation_audit")
    if audit:
        metadata["groundedness"] = f"{audit['groundedness']:.2f}"
        metadata["citations_used"] = ", ".join(audit["valid"]) or "none"
        if audit["invalid"]:
            metadata["INVENTED_CITATIONS"] = ", ".join(audit["invalid"])
    return metadata


@app.post("/query", response_model=ReportResponse, tags=["Report Generation"])
async def generate_report(request: QueryRequest):
    """Generate a report based on the query."""
    try:
        # Check if vector store has data
        stats = vector_store.get_collection_info()
        if stats["count"] == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Vector store is empty. Please ingest data first using /ingest endpoints.",
            )

        logger.info("report_generation_requested", query=request.query)

        # Generate report
        result = report_graph.generate_report(request.query)

        # Check for errors
        if result.get("error"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Report generation failed: {result['error']}",
            )

        # Save report if requested
        saved_file = None
        if request.save_report:
            from ..report_writer import ReportWriter

            writer = ReportWriter()

            if request.format == "markdown":
                file_path = writer.save_report_markdown(
                    query=request.query,
                    report=result.get("report", ""),
                    summary=result.get("summary", ""),
                    sources=result.get("sources", []),
                    metadata=saved_report_metadata(result),
                )
            else:
                file_path = writer.save_report(
                    query=request.query,
                    report=result.get("report", ""),
                    summary=result.get("summary", ""),
                    sources=result.get("sources", []),
                    metadata=saved_report_metadata(result),
                )
            saved_file = str(file_path)

        logger.info(
            "report_generated",
            query=request.query,
            tokens=result.get("num_tokens_used", 0),
            cost=result.get("total_cost", 0.0),
            groundedness=result.get("groundedness"),
        )

        return ReportResponse(
            query=request.query,
            report=result.get("report", ""),
            summary=result.get("summary", ""),
            sources=result.get("sources", []),
            citations=[c.as_dict() for c in (result.get("citations") or [])],
            citation_audit=result.get("citation_audit"),
            groundedness=result.get("groundedness"),
            num_tokens_used=result.get("num_tokens_used", 0),
            total_cost_usd=result.get("total_cost", 0.0),
            saved_file=saved_file,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("report_generation_api_failed", error=str(e), query=request.query)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation failed: {str(e)}",
        )


@app.get("/costs", response_model=CostSummaryResponse, tags=["Monitoring"])
async def get_cost_summary():
    """Get session cost summary."""
    try:
        cost_tracker = get_cost_tracker()
        summary = cost_tracker.get_session_summary()

        return CostSummaryResponse(
            total_cost_usd=summary["total_cost_usd"],
            total_input_tokens=summary["total_input_tokens"],
            total_output_tokens=summary["total_output_tokens"],
            by_model=summary["by_model"],
        )
    except Exception as e:
        logger.error("get_costs_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cost summary: {str(e)}",
        )


@app.delete("/clear", tags=["Data Management"])
async def clear_collection():
    """Clear all documents from the vector store."""
    try:
        logger.warning("collection_clear_requested")
        ingestion_pipeline.clear_collection()

        return JSONResponse(
            content={
                "message": "Collection cleared successfully",
                "status": "success",
            }
        )
    except Exception as e:
        logger.error("clear_collection_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear collection: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8350")),
        reload=True,
        log_level="info",
    )

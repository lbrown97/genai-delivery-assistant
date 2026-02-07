from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


def _recursive_splitter():
    """Create the default recursive splitter for non-Markdown documents."""

    return RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )


def _pdf_splitter():
    """Create a PDF-focused splitter that keeps OWASP/NIST section headers closer to content."""

    return RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        separators=["\n## ", "\n# ", "\n\n", "\n", " ", ""],
    )


def _markdown_splitter():
    """Create a header-aware splitter for Markdown documents."""

    return MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )


def _is_markdown(source_id: str) -> bool:
    """Return whether a source identifier points to a Markdown document."""

    return source_id.lower().endswith(".md")


def _is_pdf(source_id: str) -> bool:
    """Return whether a source identifier points to a PDF document or PDF page chunk."""

    lower = source_id.lower()
    return lower.endswith(".pdf") or ".pdf#p" in lower


def split_documents(docs):
    """
    Split documents using a Markdown-aware splitter for .md, and recursive splitter for others.
    Markdown chunks are further split to enforce max chunk size.
    """
    rec = _recursive_splitter()
    pdf = _pdf_splitter()
    md = _markdown_splitter()
    out = []
    for d in docs:
        source_id = d.metadata.get("source_id", "")
        if _is_markdown(source_id):
            md_docs = md.split_text(d.page_content)
            # carry metadata forward
            for m in md_docs:
                m.metadata = {**d.metadata, **m.metadata}
            out.extend(rec.split_documents(md_docs))
        elif _is_pdf(source_id):
            out.extend(pdf.split_documents([d]))
        else:
            out.extend(rec.split_documents([d]))
    return out

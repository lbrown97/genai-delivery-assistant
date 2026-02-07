from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


def _recursive_splitter():
    """Create the default recursive splitter for non-Markdown documents."""

    return RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )


def _markdown_splitter():
    """Create a header-aware splitter for Markdown documents."""

    return MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )


def split_documents(docs):
    """
    Split documents using a Markdown-aware splitter for .md, and recursive splitter for others.
    Markdown chunks are further split to enforce max chunk size.
    """
    rec = _recursive_splitter()
    md = _markdown_splitter()
    out = []
    for d in docs:
        source_id = d.metadata.get("source_id", "")
        if source_id.lower().endswith(".md"):
            md_docs = md.split_text(d.page_content)
            # carry metadata forward
            for m in md_docs:
                m.metadata = {**d.metadata, **m.metadata}
            out.extend(rec.split_documents(md_docs))
        else:
            out.extend(rec.split_documents([d]))
    return out

import numpy as np

from rag_core import DocumentChunk, build_prompt, chunk_text, extract_chunks, retrieve


class FakeEncoder:
    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            t = text.lower()
            vector = np.array([float("python" in t), float("statistics" in t)], dtype=np.float32)
            if np.linalg.norm(vector) == 0:
                vector = np.array([0.01, 0.01], dtype=np.float32)
            vectors.append(vector / np.linalg.norm(vector))
        return np.stack(vectors)


def test_chunk_text_overlaps():
    words = [f"w{i}" for i in range(11)]
    segments = chunk_text(" ".join(words), chunk_size=5, overlap=2)
    assert segments == ["w0 w1 w2 w3 w4", "w3 w4 w5 w6 w7", "w6 w7 w8 w9 w10"]


def test_chunk_text_edge_case():
    assert chunk_text("   ") == []
    assert chunk_text("hi") == ["hi"]


def test_extract_txt():
    chunks = extract_chunks("README.txt", b"A short plain text document.")
    assert chunks[0].source == "README.txt"
    assert chunks[0].page is None


def test_retrieval_and_prompt():
    docs = [DocumentChunk("a.pdf", 3, "Python code examples"),
            DocumentChunk("b.pdf", 2, "Statistics probability exercises")]
    encoder = FakeEncoder()
    vectors = encoder.encode([x.text for x in docs])
    hits = retrieve("python", docs, vectors, encoder, top_k=1)
    assert hits[0].chunk.source == "a.pdf"
    prompt = build_prompt("python", hits)
    assert "a.pdf, p. 3" in prompt
    assert "untrusted" in prompt


def test_pdf_preserves_page_numbers():
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "This is the first page of the sample report.")
    doc.new_page().insert_text((72, 72), "This is the second page with results.")
    data = doc.tobytes()
    doc.close()
    chunks = extract_chunks("sample.pdf", data)
    assert {chunk.page for chunk in chunks} == {1, 2}

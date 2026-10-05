# Sample papers

Real research papers used on Day 2 onwards. Each one is redistributed unmodified under the licence
shown, which was checked on the arXiv abstract page **of the exact version listed** on 2026-10-04.
Files are renamed for convenience only; their content is byte-identical to the arXiv PDF (sha256 in
`paper_agent/data/papers.json`).

| File | Title | Authors | Source (exact version) | Licence | Pages |
|---|---|---|---|---|---|
| `mistral-7b.pdf` | Mistral 7B | Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, et al. (18 authors) | [arXiv:2310.06825v1](https://arxiv.org/abs/2310.06825v1) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | 9 |
| `ragas.pdf` | Ragas: Automated Evaluation of Retrieval Augmented Generation | Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert | [arXiv:2309.15217v2](https://arxiv.org/abs/2309.15217v2) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | 8 |
| `vllm.pdf` | Efficient Memory Management for Large Language Model Serving with PagedAttention | Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, et al. (9 authors) | [arXiv:2309.06180v1](https://arxiv.org/abs/2309.06180v1) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | 16 |
| `dense-x.pdf` | Dense X Retrieval: What Retrieval Granularity Should We Use? | Tong Chen, Hongwei Wang, Sihao Chen, Wenhao Yu, Kaixin Ma, Xinran Zhao, et al. (8 authors) | [arXiv:2312.06648v3](https://arxiv.org/abs/2312.06648v3) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | 19 |

**Considered and left out:** Lost in the Middle (2307.03172), Retrieval-Augmented Generation
(2005.11401), LoRA (2106.09685), Longformer (2004.05150), Toolformer (2302.04761), HyDE (2212.10496),
DPR (2004.04906) and MTEB (2210.07316) carry arXiv's non-exclusive distribution licence, which does
not allow redistribution. Sentence-BERT (1908.10084) is CC BY-SA 4.0 (share-alike), left out to keep
the terms simple. ReAct (2210.03629) and phi-1 (2306.11644) are CC BY 4.0 but too long for class.

## Generated edge-case PDFs (FICTIONAL)

Made by `tools/make_edge_case_pdfs.py` (reportlab), stored in `paper_agent/fixtures/pdfs/`. Every
one says on page 1 that it is a fictional test document. Used on Day 2 (parsing) and Day 3 (safety).

| File | What it tests |
|---|---|
| `two_column.pdf` | Two-column layout: naive extraction interleaves the columns |
| `image_only.pdf` | Page 2 is an image of a results table: no text layer, needs OCR |
| `hidden_white.pdf` | An instruction to AI tools in white text on a white page |
| `hidden_tiny.pdf` | An instruction to AI tools in 1-point text |

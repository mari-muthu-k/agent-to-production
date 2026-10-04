# Custom CA certificates (optional)

On networks that inspect TLS (a corporate proxy re-signs HTTPS traffic), `pip install` during
the image build and calls to your LLM gateway fail with `CERTIFICATE_VERIFY_FAILED`.

Put your organisation's root CA here as a PEM file ending in `.crt` (ask IT, or export it from
your browser), then rebuild with `make build`. The Dockerfile adds every `certs/*.crt` to the
system trust store and points pip, requests and httpx at it.

Certificates are public, but `*.crt` files here are gitignored anyway so nothing
organisation-specific gets committed.

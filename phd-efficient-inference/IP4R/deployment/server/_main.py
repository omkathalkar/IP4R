"""Entry point for ip4r-server CLI command."""
import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="IP4R FQCT Inference Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--workers", type=int, default=1, help="Uvicorn worker processes (default: 1)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes (dev only)")
    args = parser.parse_args()

    uvicorn.run(
        "server.app:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()

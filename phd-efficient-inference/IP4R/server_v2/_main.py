"""Entry point for FQCT server v2 (DL+CV sandwich). Default port: 8081."""
import argparse
import uvicorn


def main():
    p = argparse.ArgumentParser(description="FQCT Server v2 — DL+CV sandwich")
    p.add_argument("--host",   default="0.0.0.0")
    p.add_argument("--port",   type=int, default=8081)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args()
    uvicorn.run("server_v2.app:app", host=args.host,
                port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()

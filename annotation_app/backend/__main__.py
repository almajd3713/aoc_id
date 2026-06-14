#!/usr/bin/env python3

import uvicorn


def main() -> None:
    uvicorn.run("annotation_app.backend.api:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026, BAAI
# SPDX-License-Identifier: Apache-2.0

"""Bridge the frozen Plan v9 upload protocol to Innata WAM HTTP servers."""

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request


def create_gateway(root, run_id, checkpoint, model_urls, unpack, content_type, max_bytes):
    root.mkdir(parents=True, exist_ok=False)
    identity = {"run_id": run_id, "checkpoint_path": checkpoint}
    (root / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    pool = asyncio.Queue()
    for url in model_urls:
        pool.put_nowait(url.rstrip("/"))
    pending = set()

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(trust_env=False, timeout=1800) as client:
            app.state.client = client
            try:
                yield
            finally:
                await asyncio.gather(*pending, return_exceptions=True)

    app = FastAPI(lifespan=lifespan)

    @app.get("/identity")
    async def get_identity():
        return identity

    @app.get("/health")
    async def health():
        return {"ready": len(model_urls), "available": pool.qsize(), "run_id": run_id}

    async def run_inference(directory: Path, request):
        model_url = await pool.get()
        try:
            response = await app.state.client.post(f"{model_url}/generate", json=request)
            response.raise_for_status()
            result = response.json()
            if result["status"] != "success":
                raise ValueError("WAM inference failed")
            if result["outputs"][0]["content"]["action128"]["checkpoint_path"] != checkpoint:
                raise ValueError("Inference response came from another checkpoint")
            result["harness_routing"] = {"run_id": run_id, "backend": model_url}
            temporary = directory / "response.tmp"
            temporary.write_text(json.dumps(result) + "\n")
            temporary.replace(directory / "response.json")
            return result
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            (directory / "error.json").write_text(json.dumps({"error": str(exc)}) + "\n")
            raise HTTPException(502, "WAM request failed; inspect the request directory") from exc
        finally:
            pool.put_nowait(model_url)

    @app.post("/infer_binary")
    async def infer(request: Request):
        if request.headers.get("content-type") != content_type:
            raise HTTPException(415, "Unsupported RoboDojo upload type")
        wire = bytearray()
        async for chunk in request.stream():
            wire.extend(chunk)
            if len(wire) > max_bytes:
                raise HTTPException(413, "RoboDojo input too large")
        try:
            body, png = await asyncio.to_thread(unpack, bytes(wire))
        except (ValueError, TypeError, KeyError, OSError) as exc:
            raise HTTPException(422, "Invalid RoboDojo observation") from exc
        name = hashlib.sha256(body["request_id"].encode()).hexdigest()
        digest = hashlib.sha256(png + json.dumps(body["request"], sort_keys=True).encode()).hexdigest()
        directory = root / name
        response = directory / "response.json"
        try:
            directory.mkdir()
        except FileExistsError:
            if (directory / "input.sha256").read_text() != digest:
                raise HTTPException(409, "Request ID was reused for a different observation")
            if not response.exists():
                raise HTTPException(409, "The previous request has no completed response")
            return json.loads(response.read_text())
        (directory / "input.sha256").write_text(digest)
        path = directory / "current.png"
        path.write_bytes(png)
        policy_request = {**body["request"], "name": name, "vision_path": str(path)}
        (directory / "request.json").write_text(json.dumps(policy_request) + "\n")
        task = asyncio.create_task(run_inference(directory, policy_request))
        pending.add(task)
        task.add_done_callback(pending.discard)
        return await asyncio.shield(task)

    return app

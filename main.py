import uvicorn


if __name__ == "__main__":
  uvicorn.run("app.server.api:app", host="0.0.0.0", port=6000, reload=True)
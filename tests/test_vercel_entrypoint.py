from api.index import app


def test_vercel_entrypoint_exports_fastapi_application():
    assert app.title == "RideMatch API"
import os
import json
import firebase_admin
from firebase_admin import credentials, auth
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

_bearer = HTTPBearer()

def _init_firebase():
    if firebase_admin._apps:
        return
    # Railway: paste JSON content vào biến FIREBASE_SERVICE_ACCOUNT_JSON
    json_str = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    if json_str:
        cred = credentials.Certificate(json.loads(json_str))
    else:
        # Local: đọc từ file
        key_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY", "./firebase-service-account.json")
        cred = credentials.Certificate(key_path)
    firebase_admin.initialize_app(cred)

_init_firebase()


async def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(_bearer),
) -> dict:
    token = creds.credentials
    try:
        decoded = auth.verify_id_token(token)
        return decoded
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token không hợp lệ: {exc}",
        )

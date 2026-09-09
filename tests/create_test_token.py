import os
from datetime import datetime, timedelta, timezone

import jwt
from dotenv import load_dotenv

load_dotenv()

customer_id = "9159450"

payload = {
    "sub": customer_id,
    "exp": datetime.now(timezone.utc) + timedelta(hours=1),
}

token = jwt.encode(
    payload,
    os.environ["JWT_SECRET"],
    algorithm=os.environ.get("JWT_ALGORITHM", "HS256"),
)

print(token)
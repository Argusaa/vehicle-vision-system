#!/usr/bin/env python3
"""为当前开发机一次性生成密钥、HTTPS 证书并迁移本地隐私数据。"""

import ipaddress
import os
import re
import secrets
import socket
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
ENV_FILE = ROOT / ".env"
CERT_DIR = ROOT / "certs"
PENDING_KEY_FILE = ROOT / "data" / ".security-setup-pending"
INSECURE_SECRET_KEYS = {
    "",
    "dev-secret-key-change-in-production",
    "change-this-to-a-random-secret-key-in-production",
}


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
        if match:
            values[match.group(1)] = match.group(2).strip()
    return values


def update_env(path: Path, updates: dict[str, str]) -> None:
    lines = path.read_text(encoding="utf-8-sig").splitlines() if path.exists() else []
    remaining = dict(updates)
    result: list[str] = []
    for line in lines:
        match = re.match(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*)=(.*)$", line)
        if match and match.group(2) in remaining:
            key = match.group(2)
            result.append(f"{key}={remaining.pop(key)}")
        else:
            result.append(line)
    if remaining:
        if result and result[-1].strip():
            result.append("")
        result.append("# 本机安全配置（由 setup_security.py 生成，请勿提交）")
        result.extend(f"{key}={value}" for key, value in remaining.items())
    path.write_text("\n".join(result).rstrip() + "\n", encoding="utf-8")


def generate_localhost_certificate(cert_path: Path, key_path: Path) -> bool:
    if cert_path.exists() and key_path.exists():
        return False

    cert_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    hostname = socket.gethostname()
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "Vehicle Vision Local Development"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Vehicle Vision Development"),
    ])
    now = datetime.now(timezone.utc)
    san_names: list[x509.GeneralName] = [
        x509.DNSName("localhost"),
        x509.DNSName(hostname),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
        x509.IPAddress(ipaddress.ip_address("::1")),
    ]
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(san_names), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(private_key, hashes.SHA256())
    )
    key_path.write_bytes(private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    try:
        key_path.chmod(0o600)
    except OSError:
        pass
    return True


def ensure_initial_admin(db, preferred_password: str = "") -> tuple[str, str, str] | None:
    """Create or rotate the local demo administrator without a fixed password."""

    from app.models.user import User
    from app.utils.auth import hash_password, verify_password
    from app.utils.privacy import protect_email

    username = "admin"
    admin = db.query(User).filter(User.username == username).first()
    action = "created"
    if admin is not None and not verify_password("admin123", admin.hashed_password):
        return None

    password = preferred_password.strip() or secrets.token_urlsafe(14)
    if admin is None:
        admin = User(
            username=username,
            hashed_password=hash_password(password),
            is_active=True,
            **protect_email("admin@localhost"),
        )
        db.add(admin)
    else:
        action = "rotated"
        admin.hashed_password = hash_password(password)
    db.commit()
    return action, username, password


def main() -> int:
    current = read_env(ENV_FILE)
    access_port = os.environ.get("PORT", current.get("PORT", "8001")).strip() or "8001"
    old_aes_key = current.get("AES_KEY", "").strip()
    keep_existing_key = bool(re.fullmatch(r"[0-9a-fA-F]{64}", old_aes_key))
    if keep_existing_key:
        new_aes_key = old_aes_key
    elif PENDING_KEY_FILE.exists():
        new_aes_key = PENDING_KEY_FILE.read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"[0-9a-fA-F]{64}", new_aes_key):
            raise RuntimeError("待恢复的安全密钥文件无效，请勿继续启动系统")
    else:
        new_aes_key = secrets.token_hex(32)
        PENDING_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
        PENDING_KEY_FILE.write_text(new_aes_key, encoding="utf-8")
        try:
            PENDING_KEY_FILE.chmod(0o600)
        except OSError:
            pass
    secret_key = current.get("SECRET_KEY", "").strip()
    if secret_key in INSECURE_SECRET_KEYS or len(secret_key) < 32:
        secret_key = secrets.token_urlsafe(48)

    cert_path = CERT_DIR / "localhost-cert.pem"
    key_path = CERT_DIR / "localhost-key.pem"
    certificate_created = generate_localhost_certificate(cert_path, key_path)

    # 先只在当前进程启用新密钥。数据库迁移成功后才落盘，失败时旧配置仍可恢复。
    os.environ["AES_KEY"] = new_aes_key
    os.environ["SECRET_KEY"] = secret_key
    sys.path.insert(0, str(BACKEND))

    from app.database import SessionLocal, init_db
    from app.security_migration import (
        migrate_sensitive_data,
        migration_needed,
        validate_migration_keys,
        write_encrypted_backup,
    )

    init_db()
    db = SessionLocal()
    backup_path: Path | None = None
    admin_credentials: tuple[str, str, str] | None = None
    try:
        previous_keys = [old_aes_key] if old_aes_key and old_aes_key != new_aes_key else []
        validate_migration_keys(db, previous_keys=previous_keys)
        if migration_needed(db):
            backup_path = write_encrypted_backup(db, ROOT / "data" / "security-backups")
            counts = migrate_sensitive_data(db, previous_keys=previous_keys)
        else:
            counts = {"users": 0, "passwords": 0, "verification_codes": 0, "plate_records": 0}
        admin_credentials = ensure_initial_admin(
            db,
            preferred_password=os.environ.get("INITIAL_ADMIN_PASSWORD", ""),
        )
    finally:
        db.close()

    update_env(ENV_FILE, {
        "SECRET_KEY": secret_key,
        "AES_KEY": new_aes_key,
        "HTTPS_CERTFILE": "certs/localhost-cert.pem",
        "HTTPS_KEYFILE": "certs/localhost-key.pem",
    })
    if PENDING_KEY_FILE.exists():
        PENDING_KEY_FILE.unlink()

    print("安全初始化完成；本脚本可安全重复运行。")
    print(f"- AES 密钥：{'保留已有安全密钥' if keep_existing_key else '已为本机随机生成'}")
    print(f"- HTTPS 证书：{'已生成' if certificate_created else '保留已有证书'}")
    if backup_path:
        print(f"- 迁移备份：{backup_path}")
    print(
        "- 数据迁移：用户 {users}，旧密码 {passwords}，验证码 {verification_codes}，车牌记录 {plate_records}".format(
            **counts
        )
    )
    if admin_credentials:
        action, username, password = admin_credentials
        action_cn = "已创建" if action == "created" else "已替换不安全的默认密码"
        if os.environ.get("INITIAL_ADMIN_PASSWORD", "").strip():
            print(f"- 初始管理员：{username}（{action_cn}，密码来自 INITIAL_ADMIN_PASSWORD，未显示）")
        else:
            print(f"- 初始管理员：{username} / {password}（{action_cn}，请立即保存）")
    print(f"请使用 https://localhost:{access_port} 访问；浏览器首次可能显示本地证书提示。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"安全初始化失败，原 AES 配置未切换；待恢复密钥已保留：{exc}", file=sys.stderr)
        raise SystemExit(1)

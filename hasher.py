import hashlib

def generate_file_hash(file_path: str) -> str:
    """Computes SHA-256 hash in C-level buffered chunks for O(1) memory overhead."""
    with open(file_path, "rb") as f:
        # Native C-level chunking introduced in Python 3.11 (fastest execution)
        digest = hashlib.file_digest(f, "sha256")
    return digest.hexdigest()
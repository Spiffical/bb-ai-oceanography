import os
import io

def is_valid_pdf(content):
    if isinstance(content, str):  # It's a file path
        if not os.path.exists(content) or os.path.getsize(content) == 0:
            print(f"File does not exist or is empty: {content}")
            return False
        try:
            with open(content, 'rb') as f:
                header = f.read(4)
        except Exception as e:
            print(f"Error reading PDF file: {e}")
            return False
    elif isinstance(content, bytes):  # It's byte content
        if len(content) == 0:
            print("Byte content is empty")
            return False
        header = content[:4]
    else:
        print(f"Unsupported content type: {type(content)}")
        return False

    if header != b'%PDF':
        print(f"Invalid PDF header: {header}")
        return False

    # Additional check for minimum file size (let's say 100 bytes as an arbitrary minimum)
    if isinstance(content, str):
        if os.path.getsize(content) < 100:
            print(f"PDF file is too small: {os.path.getsize(content)} bytes")
            return False
    elif len(content) < 100:
        print(f"PDF content is too small: {len(content)} bytes")
        return False

    return True

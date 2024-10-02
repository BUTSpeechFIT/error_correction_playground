import os
import shutil

if __name__ == "__main__":
    # copy [dir1]\[session_id]\ref.json to [dir2][session_id]\ref.json
    SRC_DIR = "predictions2"
    DST_DIR = "predictions"

    for session_id in os.listdir(SRC_DIR):
        if not os.path.exists(os.path.join(DST_DIR, session_id)):
            continue
        src_path = os.path.join(SRC_DIR, session_id, "ref.json")
        dst_path = os.path.join(DST_DIR, session_id, "ref.json")
        if os.path.exists(dst_path):
            continue
        shutil.copy(src_path, dst_path)

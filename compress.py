import io, os, zipfile, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict

import streamlit as st
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import fitz  # PyMuPDF

# ===== HEIC/HEIF =====
HEIF_OK = False
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_OK = True
except Exception:
    HEIF_OK = False

# ==========================
# PAGE & SIDEBAR
# ==========================
st.set_page_config(page_title="Multi-ZIP → JPG & Kompres High Quality", page_icon="📦", layout="wide")
st.title("📦 Multi-ZIP / Files → JPG & Kompres (High Quality)")
st.caption("Konversi gambar (termasuk JFIF/HEIC) & PDF ke JPG dengan kualitas tinggi. File bernama q/w/e → 200 KB, lainnya → 140 KB.")

with st.sidebar:
    st.header("⚙️ Pengaturan")
    SPEED_PRESET = st.selectbox("Preset kecepatan", ["balanced", "high_quality"], index=1)
    MIN_SIDE_PX = st.number_input("Sisi terpendek minimum (px)", 64, 4096, 800, 64)
    SCALE_MIN = st.slider("Skala minimum saat downscale", 0.30, 0.90, 0.60, 0.05)
    UPSCALE_MAX = st.slider("Batas upscale maksimum", 1.0, 2.0, 1.0, 0.1)
    SHARPEN_ON_RESIZE = st.checkbox("Sharpen setelah resize", True)
    SHARPEN_AMOUNT = st.slider("Sharpen amount", 0.0, 2.0, 0.5, 0.1)
    PDF_DPI = st.slider("PDF DPI (resolusi)", 150, 400, 300, 25)
    MASTER_ZIP_NAME = st.text_input("Nama master ZIP", "compressed.zip")
    st.markdown("**Target otomatis:**")
    st.markdown("- File bernama **q, w, atau e** → **200 KB**")
    st.markdown("- File dengan nama lain → **140 KB**")

# ===== Tunables untuk Kualitas Tinggi =====
MAX_QUALITY = 95  # Kualitas maksimum
MIN_QUALITY = 60  # Minimum quality untuk menjaga detail
BG_FOR_ALPHA = (255, 255, 255)
THREADS = min(4, max(2, (os.cpu_count() or 2)))
ZIP_COMP_ALGO = zipfile.ZIP_DEFLATED

# ✅ Target size berdasarkan nama file (lebih ketat)
TARGET_KB_HIGH = 200  # untuk file q, w, e
TARGET_KB_LOW = 140   # untuk file lain
MIN_KB_HIGH = 195
MIN_KB_LOW = 135

IMG_EXT = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif", ".heic", ".heif"}
PDF_EXT = {".pdf"}
ALLOW_ZIP = True

# ✅ JPEG encoder settings untuk kualitas maksimal
JPEG_SUBSAMPLING = 0  # 4:4:4 chroma subsampling (no color compression)
JPEG_OPTIMIZE = True
JPEG_PROGRESSIVE = True

# ==========================
# Helper: Deteksi target size berdasarkan nama file
# ==========================
def get_target_size_for_path(relpath: Path) -> Tuple[int, int]:
    """
    Mengembalikan (TARGET_KB, MIN_KB) berdasarkan nama file.
    Jika nama file (tanpa ekstensi) adalah tepat 'q', 'w', atau 'e' → 200 KB
    Lainnya → 140 KB
    """
    filename_lower = relpath.stem.lower()
    if filename_lower in ['q', 'w', 'e']:
        return TARGET_KB_HIGH, MIN_KB_HIGH
    return TARGET_KB_LOW, MIN_KB_LOW

# ==========================
# Helpers untuk Quality Enhancement
# ==========================

def enhance_sharpness(img: Image.Image, factor: float = 1.0) -> Image.Image:
    """Tingkatkan ketajaman gambar dengan control lebih halus"""
    if factor <= 0:
        return img
    enhancer = ImageEnhance.Sharpness(img)
    return enhancer.enhance(1.0 + (factor * 0.3))  # Subtle sharpening

def smart_resize(img: Image.Image, scale: float, enhance_quality: bool = True) -> Image.Image:
    """Resize dengan algoritma terbaik dan optional enhancement"""
    w, h = img.size
    nw, nh = max(int(w * scale), 1), max(int(h * scale), 1)
    
    # Gunakan LANCZOS untuk downscale, BICUBIC untuk upscale
    if scale < 1.0:
        # Downscaling: gunakan LANCZOS dengan slight pre-blur untuk anti-aliasing
        if enhance_quality:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.2))
        resized = img.resize((nw, nh), Image.LANCZOS)
    else:
        # Upscaling: gunakan BICUBIC (lebih smooth dari LANCZOS)
        resized = img.resize((nw, nh), Image.BICUBIC)
    
    return resized

def to_rgb_flat(img: Image.Image, bg=BG_FOR_ALPHA) -> Image.Image:
    """Convert to RGB dengan handling alpha channel"""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        base = Image.new("RGB", img.size, bg)
        base.paste(img, mask=img.convert("RGBA").split()[-1])
        return base
    if img.mode != "RGB":
        return img.convert("RGB")
    return img

def save_jpg_bytes(img: Image.Image, quality: int) -> bytes:
    """Save JPEG dengan settings optimal untuk kualitas"""
    buf = io.BytesIO()
    img.save(
        buf,
        format="JPEG",
        quality=int(quality),
        optimize=JPEG_OPTIMIZE,
        progressive=JPEG_PROGRESSIVE,
        subsampling=JPEG_SUBSAMPLING,  # 4:4:4 untuk detail maksimal
    )
    return buf.getvalue()

def try_quality_bs(img: Image.Image, target_kb: int, q_min=MIN_QUALITY, q_max=MAX_QUALITY):
    """Binary search untuk menemukan kualitas optimal"""
    lo, hi = q_min, q_max
    best_bytes = None
    best_q = None
    
    while lo <= hi:
        mid = (lo + hi) // 2
        data = save_jpg_bytes(img, mid)
        if len(data) <= target_kb * 1024:
            best_bytes, best_q = data, mid
            lo = mid + 1
        else:
            hi = mid - 1
    
    return best_bytes, best_q

def ensure_min_side(img: Image.Image, min_side_px: int) -> Image.Image:
    """Pastikan sisi terpendek minimal sekian pixel"""
    w, h = img.size
    if min(w, h) >= min_side_px:
        return img
    scale = min_side_px / min(w, h)
    return smart_resize(img, scale, enhance_quality=True)

def load_image_from_bytes(name: str, raw: bytes) -> Image.Image:
    """Load image dengan EXIF orientation handling"""
    im = Image.open(io.BytesIO(raw))
    return ImageOps.exif_transpose(im)

def gif_first_frame(im: Image.Image) -> Image.Image:
    """Extract first frame dari GIF"""
    try:
        im.seek(0)
    except Exception:
        pass
    return im.convert("RGBA") if im.mode == "P" else im

def compress_into_range(
    base_img: Image.Image,
    min_kb: int,
    max_kb: int,
    min_side_px: int,
    scale_min: float,
    upscale_max: float,
    do_sharpen: bool,
    sharpen_amount: float,
):
    """Kompresi gambar dengan mempertahankan kualitas visual maksimal"""
    base = to_rgb_flat(base_img)
    
    # 1) Pastikan resolusi minimal dulu
    base = ensure_min_side(base, min_side_px)
    
    # 2) Coba save dengan kualitas tinggi tanpa resize
    data, q = try_quality_bs(base, max_kb, MIN_QUALITY, MAX_QUALITY)
    if data is not None and len(data) >= min_kb * 1024:
        # Sudah masuk range tanpa perlu resize
        if do_sharpen:
            base = enhance_sharpness(base, sharpen_amount)
            data = save_jpg_bytes(base, q)
        return data, 1.0, q, len(data)
    
    # 3) Jika terlalu besar, perlu downscale
    lo, hi = scale_min, 1.0
    best_pack = None
    max_steps = 15  # Lebih banyak iterasi untuk precision lebih tinggi
    
    for _ in range(max_steps):
        mid = (lo + hi) / 2
        candidate = smart_resize(base, mid, enhance_quality=True)
        candidate = ensure_min_side(candidate, min_side_px)
        
        if do_sharpen:
            candidate = enhance_sharpness(candidate, sharpen_amount)
        
        d, q2 = try_quality_bs(candidate, max_kb, MIN_QUALITY, MAX_QUALITY)
        
        if d is not None:
            size_kb = len(d) / 1024
            best_pack = (d, mid, q2, len(d))
            
            # Jika sudah masuk range yang bagus, break
            if min_kb <= size_kb <= max_kb:
                break
            
            lo = mid + (hi - mid) * 0.3
        else:
            hi = mid - (mid - lo) * 0.3
        
        if hi - lo < 0.005:  # Precision threshold
            break
    
    if best_pack is None:
        # Fallback: gunakan scale minimum
        smallest = smart_resize(base, scale_min, enhance_quality=True)
        smallest = ensure_min_side(smallest, min_side_px)
        if do_sharpen:
            smallest = enhance_sharpness(smallest, sharpen_amount)
        d = save_jpg_bytes(smallest, MIN_QUALITY)
        return (d, scale_min, MIN_QUALITY, len(d))
    
    data, scale_used, q_used, size_b = best_pack
    
    # 4) Jika masih di bawah min_kb, coba tingkatkan kualitas
    if size_b < min_kb * 1024:
        img_now = smart_resize(base, scale_used, enhance_quality=True)
        img_now = ensure_min_side(img_now, min_side_px)
        if do_sharpen:
            img_now = enhance_sharpness(img_now, sharpen_amount)
        
        # Coba dengan kualitas lebih tinggi
        d, q2 = try_quality_bs(img_now, max_kb, max(q_used, MIN_QUALITY), MAX_QUALITY)
        if d is not None and len(d) > size_b and len(d) <= max_kb * 1024:
            data, q_used, size_b = d, q2, len(d)
    
    return data, scale_used, q_used, size_b

def pdf_bytes_to_images(pdf_bytes: bytes, dpi: int) -> List[Image.Image]:
    """Convert PDF ke images dengan resolusi tinggi"""
    images = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            rect = page.rect
            long_inch = max(rect.width, rect.height) / 72.0
            # Target pixel yang lebih tinggi untuk detail maksimal
            target_long_px = 3000
            dpi_eff = int(min(max(dpi, 150), max(150, target_long_px / max(long_inch, 1e-6))))
            mat = fitz.Matrix(dpi_eff / 72.0, dpi_eff / 72.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(ImageOps.exif_transpose(img))
    return images

def extract_zip_to_memory(zf_bytes: bytes) -> List[Tuple[Path, bytes]]:
    """Extract ZIP ke memory"""
    out = []
    with zipfile.ZipFile(io.BytesIO(zf_bytes), 'r') as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            with zf.open(info, 'r') as f:
                data = f.read()
            out.append((Path(info.filename), data))
    return out

def guess_base_name_from_zip(zipname: str) -> str:
    """Extract base name dari ZIP filename"""
    base = Path(zipname).stem
    return base or "output"

def process_one_file_entry(relpath: Path, raw_bytes: bytes, input_root_label: str):
    """Process satu file (gambar atau PDF)"""
    processed: List[Tuple[str, int, float, int, bool, int, int]] = []
    outputs: Dict[str, bytes] = {}
    skipped: List[Tuple[str, str]] = []
    ext = relpath.suffix.lower()
    
    # Tentukan target size berdasarkan nama file
    target_kb, min_kb = get_target_size_for_path(relpath)
    
    try:
        if ext in PDF_EXT:
            pages = pdf_bytes_to_images(raw_bytes, dpi=PDF_DPI)
            for idx, pil_img in enumerate(pages, start=1):
                try:
                    data, scale, q, size_b = compress_into_range(
                        pil_img,
                        min_kb,
                        target_kb,
                        MIN_SIDE_PX,
                        SCALE_MIN,
                        UPSCALE_MAX,
                        SHARPEN_ON_RESIZE,
                        SHARPEN_AMOUNT,
                    )
                    out_rel = relpath.with_suffix("").as_posix() + f"_p{idx}.jpg"
                    outputs[out_rel] = data
                    processed.append((out_rel, size_b, scale, q, min_kb * 1024 <= size_b <= target_kb * 1024, target_kb, min_kb))
                except Exception as e:
                    skipped.append((f"{relpath} (page {idx})", str(e)))
        elif ext in IMG_EXT and (ext not in {".heic", ".heif"} or HEIF_OK):
            im = load_image_from_bytes(relpath.name, raw_bytes)
            if ext == ".gif":
                im = gif_first_frame(im)
            data, scale, q, size_b = compress_into_range(
                im,
                min_kb,
                target_kb,
                MIN_SIDE_PX,
                SCALE_MIN,
                UPSCALE_MAX,
                SHARPEN_ON_RESIZE,
                SHARPEN_AMOUNT,
            )
            out_rel = relpath.with_suffix(".jpg").as_posix()
            outputs[out_rel] = data
            processed.append((out_rel, size_b, scale, q, min_kb * 1024 <= size_b <= target_kb * 1024, target_kb, min_kb))
        elif ext in {".heic", ".heif"} and not HEIF_OK:
            skipped.append((str(relpath), "Butuh pillow-heif (tidak tersedia)"))
    except Exception as e:
        skipped.append((str(relpath), str(e)))

    return input_root_label, processed, skipped, outputs

# ==========================
# UI Upload & Run
# ==========================
st.subheader("1) Upload ZIP atau File Lepas")
allowed_exts_for_uploader = sorted({e.lstrip('.') for e in IMG_EXT.union(PDF_EXT)} | ({"zip"} if ALLOW_ZIP else set()))

uploaded_files = st.file_uploader(
    "Upload beberapa ZIP (berisi folder/gambar/PDF) dan/atau file lepas (gambar/PDF).",
    type=allowed_exts_for_uploader,
    accept_multiple_files=True,
)

run = st.button("🚀 Proses & Buat Master ZIP", type="primary", disabled=not uploaded_files)

if run:
    if not uploaded_files:
        st.warning("Silakan upload minimal satu file.")
        st.stop()

    jobs = []
    used_labels = set()

    def unique_name(base: str, used: set) -> str:
        name = base
        idx = 2
        while name in used:
            name = f"{base}_{idx}"
            idx += 1
        used.add(name)
        return name

    zip_inputs, loose_inputs = [], []
    for f in uploaded_files:
        name, raw = f.name, f.read()
        if name.lower().endswith(".zip"):
            zip_inputs.append((name, raw))
        else:
            loose_inputs.append((name, raw))

    allowed = IMG_EXT.union(PDF_EXT)

    for zname, zbytes in zip_inputs:
        try:
            pairs = extract_zip_to_memory(zbytes)
            base_label = unique_name(guess_base_name_from_zip(zname), used_labels)
            items = [(relp, data) for (relp, data) in pairs if relp.suffix.lower() in allowed]
            if items:
                jobs.append({"label": base_label, "items": items})
        except Exception as e:
            st.error(f"Gagal membuka ZIP {zname}: {e}")

    if loose_inputs:
        ts = time.strftime("%Y%m%d_%H%M%S")
        base_label = unique_name(f"compressed_pict_{ts}", used_labels)
        items = [(Path(name), data) for (name, data) in loose_inputs if Path(name).suffix.lower() in allowed]
        if items:
            jobs.append({"label": base_label, "items": items})

    if not jobs:
        st.error("Tidak ada berkas valid (butuh gambar/PDF, atau ZIP berisi file-file tersebut).")
        st.stop()

    st.write(f"🔧 Ditemukan **{sum(len(j['items']) for j in jobs)}** berkas dari **{len(jobs)}** input.")

    summary: Dict[str, List[Tuple[str, int, float, int, bool, int, int]]] = defaultdict(list)
    skipped_all: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    master_buf = io.BytesIO()
    zip_write_lock = threading.Lock()
    with zipfile.ZipFile(master_buf, "w", compression=ZIP_COMP_ALGO) as master:
        top_folders: Dict[str, str] = {}
        for job in jobs:
            top = f"{job['label']}_compressed"
            top_folders[job['label']] = top
            master.writestr(f"{top}/", "")

        def add_to_master_zip_threadsafe(top_folder: str, rel_path: str, data: bytes):
            with zip_write_lock:
                master.writestr(f"{top_folder}/{rel_path}", data)

        def worker(label: str, relp: Path, raw: bytes):
            return process_one_file_entry(relp, raw, label)

        all_tasks = [(job["label"], relp, data) for job in jobs for (relp, data) in job["items"]]
        total, done = len(all_tasks), 0
        progress = st.progress(0.0)

        with ThreadPoolExecutor(max_workers=THREADS) as ex:
            futures = [ex.submit(worker, *t) for t in all_tasks]
            for fut in as_completed(futures):
                label, prc, skp, outs = fut.result()
                summary[label].extend(prc)
                skipped_all[label].extend(skp)
                if outs:
                    top = top_folders[label]
                    for rel_path, data in outs.items():
                        add_to_master_zip_threadsafe(top, rel_path, data)
                done += 1
                progress.progress(min(done / total, 1.0))

    master_buf.seek(0)

    # ==========================
    # Ringkasan
    # ==========================
    st.subheader("📊 Ringkasan")
    grand_ok = 0
    grand_cnt = 0
    MAX_ROWS_PER_JOB = 300

    for job in jobs:
        base = job["label"]
        items = summary[base]
        skipped = skipped_all[base]
        with st.expander(f"📦 {base} — {len(items)} file diproses, {len(skipped)} dilewati/errored"):
            ok = 0
            shown = 0
            for name, size_b, scale, q, in_range, target_kb, min_kb in items:
                if shown >= MAX_ROWS_PER_JOB:
                    break
                kb = size_b / 1024
                flag = "✅" if in_range else ("🔼" if kb < min_kb else "⚠️")
                st.write(f"{flag} {name} → **{kb:.1f} KB** (target: {min_kb}-{target_kb} KB) | scale≈{scale:.3f} | quality={q}")
                ok += 1 if in_range else 0
                shown += 1
            extra = len(items) - shown
            if extra > 0:
                st.caption(f"(+{extra} baris lainnya disembunyikan untuk menjaga performa UI)")

            if skipped:
                st.write("**Dilewati/Errored:**")
                for n, reason in skipped[:50]:
                    st.write(f"- {n}: {reason}")

            st.caption(f"Berhasil di rentang target: **{ok}/{len(items)}**")
            grand_ok += ok
            grand_cnt += len(items)

    st.write("---")
    st.write(f"**Total file OK di rentang:** {grand_ok}/{grand_cnt}")

    st.download_button(
        "⬇️ Download Master ZIP",
        data=master_buf.getvalue(),
        file_name=MASTER_ZIP_NAME.strip() or "compressed.zip",
        mime="application/zip",
    )

    st.success("Selesai! Master ZIP siap diunduh dengan kualitas tinggi (q/w/e=200KB, lainnya=140KB).")

from pathlib import Path
from uuid import uuid4
from threading import Lock, Thread
import json
import logging
import time
import warnings

from flask import Flask, jsonify, render_template, request, send_file, url_for
from PIL import Image, ImageOps, UnidentifiedImageError

import extract_vintern_groups as extractor
from vintern_service import PROFILES, GROUP_LABELS, crop_groups, VinternReader

ROOT = Path(__file__).resolve().parent
STORAGE = ROOT / 'web_uploads_vintern'
STORAGE.mkdir(exist_ok=True)
app = Flask(__name__, template_folder=str(ROOT / 'templates'))
app.config['MAX_CONTENT_LENGTH'] = 12 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000
jobs = {}
state_lock = Lock()
processing = Lock()
# The existing extractor uses a module-level FOLDER: serialize all calls to it.
extraction_lock = Lock()
reader = VinternReader()
FIELDS = {
    'ho_ten': 'Họ và tên', 'gioi_tinh': 'Giới tính', 'ngay_sinh': 'Ngày sinh',
    'noi_sinh': 'Nơi sinh', 'dan_toc': 'Dân tộc', 'dia_chi': 'Địa chỉ',
    'ho_ten_cha': 'Họ tên cha', 'nghe_nghiep_cha': 'Nghề nghiệp cha',
    'ho_ten_me': 'Họ tên mẹ', 'nghe_nghiep_me': 'Nghề nghiệp mẹ',
}


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def work(job_id):
    folder = STORAGE / job_id
    handler = logging.FileHandler(folder / 'process.log', encoding='utf-8')
    app.logger.addHandler(handler)
    started = time.perf_counter()
    try:
        profile = jobs[job_id]['profile']
        def progress(done, total, message):
            with state_lock:
                jobs[job_id].update(completed=done, total=total, message=message)
        timing = reader.read_groups(folder, profile, progress)
        with extraction_lock:
            extractor.main(folder)
        timing['total_seconds'] = round(time.perf_counter() - started, 2)
        write_json(folder / 'timing.json', timing)
        with state_lock:
            jobs[job_id].update(status='done', timing=timing, message='Đọc xong. Hãy kiểm tra từng trường.')
    except Exception:
        app.logger.exception('OCR job failed: %s', job_id)
        with state_lock:
            jobs[job_id].update(status='error', message='Chưa đọc xong ảnh.', error='Xem process.log trong thư mục lượt chạy; kiểm tra terminal để biết lỗi tải model hoặc môi trường.')
    finally:
        app.logger.removeHandler(handler)
        handler.close()
        processing.release()


@app.get('/')
def index():
    return render_template('index.html', profiles=PROFILES, field_names=FIELDS)


@app.errorhandler(413)
def too_large(error):
    return jsonify(error='Ảnh vượt giới hạn tải lên 12 MB.'), 413


@app.post('/api/upload')
def upload():
    profile = request.form.get('profile')
    if profile not in PROFILES:
        return jsonify(error='Hãy chọn một trong hai bố cục đã hỗ trợ.'), 400
    file = request.files.get('image')
    if file is None or not file.filename:
        return jsonify(error='Hãy chọn ảnh trang đầu học bạ.'), 400
    job_id = uuid4().hex
    folder = STORAGE / job_id
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(file.stream) as source:
                if source.format not in {'JPEG', 'PNG', 'WEBP'}:
                    raise ValueError('Chỉ nhận JPG, PNG hoặc WEBP.')
                image = ImageOps.exif_transpose(source).convert('RGB')
                if image.width * image.height > 20_000_000:
                    raise ValueError('Ảnh vượt 20 megapixel.')
                if min(image.size) < 200:
                    raise ValueError('Hãy tải ảnh cả trang, không phải ảnh cắt một dòng.')
                folder.mkdir()
                image.save(folder / 'original.png')
        original_name = Path(file.filename.replace('\\', '/')).name[:180]
        write_json(folder / 'source.json', {'source': original_name, 'profile': profile, 'job_id': job_id})
        crop_groups(folder / 'original.png', folder, profile)
        with state_lock:
            jobs[job_id] = {'status': 'ready', 'original_name': original_name, 'profile': profile, 'completed': 0, 'total': 4, 'message': 'Kiểm tra đủ chữ trong 4 vùng trước khi đọc.'}
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        return jsonify(error='Không nhận được ảnh hợp lệ: ' + str(exc)), 400
    return jsonify(job_id=job_id), 201


@app.post('/api/jobs/<job_id>/start')
def start(job_id):
    with state_lock:
        job = jobs.get(job_id)
        if job is None:
            return jsonify(error='Không tìm thấy lượt xử lý.'), 404
        if job['status'] not in {'ready', 'error'}:
            return jsonify(error='Lượt xử lý đã bắt đầu hoặc hoàn tất.'), 409
        if not processing.acquire(blocking=False):
            return jsonify(error='CPU đang đọc ảnh khác. Đợi hoàn tất rồi bấm lại.'), 409
        job.update(status='running', completed=0, error=None, message='Đang chuẩn bị model…')
    try:
        Thread(target=work, args=(job_id,), daemon=True).start()
    except Exception:
        processing.release()
        with state_lock:
            job.update(status='error', error='Không tạo được tác vụ. Hãy thử lại.')
        raise
    return jsonify(job_id=job_id), 202


def describe(job_id):
    with state_lock:
        job = jobs.get(job_id)
        if job is None:
            return None
        result = dict(job)
    result['image_url'] = url_for('artifact', job_id=job_id, kind='original')
    result['groups'] = [
        {'key': key, 'label': GROUP_LABELS[key], 'image_url': url_for('artifact', job_id=job_id, kind=key)}
        for key in PROFILES[result['profile']]['regions']
    ]
    if result['status'] == 'done':
        folder = STORAGE / job_id
        with extraction_lock:
            result['result'] = read_json(folder / 'student.json')
            result['raw'] = read_json(folder / 'student_raw.json')
            result['evaluation'] = read_json(folder / 'evaluation.json')
        result['downloads'] = {
            key: url_for('artifact', job_id=job_id, kind=key)
            for key in ('json', 'raw', 'evaluation', 'csv', 'verified')
        }
    return result


@app.get('/api/jobs/<job_id>')
def status(job_id):
    result = describe(job_id)
    if result is None:
        return jsonify(error='Không tìm thấy lượt xử lý. Máy chủ có thể đã khởi động lại.'), 404
    return jsonify(result)


@app.post('/api/jobs/<job_id>/verify')
def verify(job_id):
    with state_lock:
        job = jobs.get(job_id)
        if job is None or job['status'] != 'done':
            return jsonify(error='Lượt xử lý chưa hoàn tất.'), 404
    payload = request.get_json(silent=True)
    confirmed = payload.get('fields') if isinstance(payload, dict) else None
    if not isinstance(confirmed, dict):
        return jsonify(error='Danh sách xác nhận không hợp lệ.'), 400
    for name, value in confirmed.items():
        if name not in FIELDS or (value is not None and (not isinstance(value, str) or not value.strip() or len(value) > 2000)):
            return jsonify(error='Trường hoặc giá trị xác nhận không hợp lệ.'), 400
        if value is not None:
            confirmed[name] = value.strip()
    folder = STORAGE / job_id
    with extraction_lock:
        path = folder / 'verified_fields.json'
        previous = path.read_bytes() if path.exists() else None
        raw_before = (folder / 'student_raw.json').read_bytes()
        write_json(path, {'source': job['original_name'], 'fields': confirmed, 'confirmed_by': 'user'})
        try:
            extractor.main(folder)
            # Confirmation must never rewrite the original OCR snapshot.
            (folder / 'student_raw.json').write_bytes(raw_before)
        except Exception:
            if previous is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(previous)
            raise
    return jsonify(describe(job_id))


@app.get('/files/<job_id>/<kind>')
def artifact(job_id, kind):
    with state_lock:
        job = jobs.get(job_id)
        if job is None:
            return 'Không tìm thấy lượt xử lý.', 404
        state = job['status']
    folder = STORAGE / job_id
    images = {'original': 'original.png', **{key: f'{key}.png' for key in GROUP_LABELS}}
    downloads = {'json': 'student.json', 'raw': 'student_raw.json', 'evaluation': 'evaluation.json', 'csv': 'evaluation.csv', 'verified': 'verified_fields.json'}
    filename = images.get(kind)
    attachment = False
    if filename is None and state == 'done':
        filename = downloads.get(kind)
        attachment = True
    if filename is None or not (folder / filename).is_file():
        return 'Không tìm thấy tệp.', 404
    response = send_file(folder / filename, as_attachment=attachment, download_name=filename)
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response


@app.after_request
def no_cache(response):
    if request.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-store'
    return response


if __name__ == '__main__':
    from waitress import serve
    print('Mở http://127.0.0.1:5000 — Vintern CPU. Model được giữ trong RAM sau lần đọc đầu.', flush=True)
    serve(app, host='127.0.0.1', port=5000, threads=4)

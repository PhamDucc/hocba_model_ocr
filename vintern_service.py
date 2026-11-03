"""Reusable CPU reader and explicit page layouts; importing does not load torch."""
from pathlib import Path
import os
import time
from PIL import Image

PROFILES = {
    'user_002': {
        'label': 'Mẫu chữ viết tay (như user_002)',
        'regions': {
            '01_ca_nhan': (0.02, 0.235, 0.98, 0.38),
            '02_dia_chi': (0.02, 0.405, 0.98, 0.445),
            '03_cha_me': (0.02, 0.435, 0.98, 0.515),
            '04_gioi_tinh': (0.47, 0.235, 0.75, 0.28),
        },
    },
    'user_003': {
        'label': 'Mẫu thông tin chữ in (như user_003)',
        'regions': {
            '01_ca_nhan': (0.07, 0.23, 0.90, 0.33),
            '02_dia_chi': (0.07, 0.36, 0.91, 0.41),
            '03_cha_me': (0.07, 0.405, 0.92, 0.46),
            '04_gioi_tinh': (0.66, 0.22, 0.85, 0.26),
        },
    },
}
GROUP_LABELS = {
    '01_ca_nhan': 'Thông tin cá nhân', '02_dia_chi': 'Địa chỉ',
    '03_cha_me': 'Cha mẹ và nghề nghiệp', '04_gioi_tinh': 'Giới tính',
}


def crop_groups(image_path, output, profile):
    regions = PROFILES[profile]['regions']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as source:
        image = source.convert('RGB')
        w, h = image.size
        for name, (left, top, right, bottom) in regions.items():
            image.crop((int(left*w), int(top*h), int(right*w), int(bottom*h))).save(output / f'{name}.png')


class VinternReader:
    """Call under a single processing lock. Cache the model between web jobs."""
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def ensure_loaded(self):
        if self.model is not None:
            return False
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        import torch
        from transformers import AutoModel, AutoTokenizer
        from test_vintern_cpu import MODEL_ID
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        model = AutoModel.from_pretrained(
            MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True,
            trust_remote_code=True, use_flash_attn=False,
        ).to('cpu').eval()
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, trust_remote_code=True, use_fast=False,
        )
        # Use the language model's real context limit, not tokenizer metadata 1700.
        limit = getattr(model.language_model.config, 'max_position_embeddings', None)
        if isinstance(limit, int) and limit > 0:
            tokenizer.model_max_length = limit
        pad = tokenizer.pad_token_id
        if pad is None:
            pad = tokenizer.eos_token_id
        if isinstance(pad, int):
            model.language_model.generation_config.pad_token_id = pad
        self.model, self.tokenizer = model, tokenizer
        return True

    def read_groups(self, folder, profile, progress):
        import torch
        from test_vintern_cpu import load_image
        progress(0, 4, 'Đang nạp Vintern trên CPU; lần đầu có thể tải model…')
        started = time.perf_counter()
        loaded = self.ensure_loaded()
        loading_seconds = time.perf_counter() - started
        records = []
        question = (
            '<image>\nChép nguyên văn tất cả các dòng chữ trong ảnh, '
            'theo thứ tự từ trên xuống dưới. Không tóm tắt, không tự bổ sung thông tin.'
        )
        for index, name in enumerate(PROFILES[profile]['regions'], 1):
            progress(index - 1, 4, f'Đang đọc {GROUP_LABELS[name]}…')
            started = time.perf_counter()
            pixels = load_image(Path(folder) / f'{name}.png').to(device='cpu', dtype=torch.float32)
            with torch.inference_mode():
                prediction = self.model.chat(self.tokenizer, pixels, question, {
                    'max_new_tokens': 512, 'do_sample': False, 'num_beams': 1,
                })
            seconds = time.perf_counter() - started
            (Path(folder) / f'{name}.txt').write_text(prediction, encoding='utf-8')
            records.append({'group': name, 'seconds': round(seconds, 2)})
            progress(index, 4, f'Đã đọc {index}/4 nhóm')
            del pixels
        return {
            'device': 'cpu', 'newly_loaded': loaded,
            'loading_seconds': round(loading_seconds, 2),
            'reading_seconds': round(sum(r['seconds'] for r in records), 2),
            'groups': records,
        }

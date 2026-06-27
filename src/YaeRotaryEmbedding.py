import torch
import torch.nn as nn

from typing import Optional
from collections.abc import Callable
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from YaeConfig import YaeConfig


class YaeRotaryEmbedding(nn.Module):
    """
    YaeRotaryEmbedding (RoPE - Rotary Position Embedding)
    ระบบฝังพิกัดตำแหน่งคำแบบหมุนเวกเตอร์ (Production-Ready)

    ---
    หน้าที่หลัก:
    สร้าง 'ไม้บรรทัดองศา' เพื่อบอกให้ Transformer รู้ว่าคำแต่ละคำอยู่ตำแหน่งไหนในประโยค
    โดยใช้วิธีหมุนเวกเตอร์ในมิติจำนวนเชิงซ้อน (Complex Space) แทนการบวกค่าเข้าไปทื่อๆ
    ซึ่งช่วยให้โมเดลเข้าใจความสัมพันธ์ของคำที่อยู่ใกล้/ไกลกัน (Relative Position) ได้ดีเยี่ยม
    """

    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: "YaeConfig", device=None):
        """
        กระบวนการตั้งค่าเริ่มต้น (Initialization):
        ทำหน้าที่เตรียม 'ความถี่ฐาน (Inverse Frequency)' สำหรับหมุนเวกเตอร์
        โดยออกแบบมาให้ยืดหยุ่น รองรับ RoPE หลายสายพันธุ์ (เช่น Default, YaRN, Llama3)
        """
        super().__init__()
        # กำหนดความยาวสูงสุดของประโยคที่โมเดลรับได้ (เช่น 4096 หรือ 8192 token)
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        # 💡 [Fix 1] ดึงข้อมูลแบบปลอดภัย ใช้ .get() และ or {} ป้องกันพังกรณี config ไม่มีคีย์นี้
        # ตรวจสอบว่าซามะสั่งให้ใช้ RoPE ธาตุไหน (ปกติจะเป็น "default")
        rope_params = getattr(self.config, "rope_parameters", {}) or {}
        self.rope_type = rope_params.get("rope_type", "default")

        # ตั้งค่าฟังก์ชันคำนวณเริ่มต้นเป็นแบบมาตรฐานของเราเอง
        rope_init_fn: Callable = self.compute_default_rope_parameters

        # 💡 [Fix 2] เช็กเผื่อไว้ว่ามีในคลังแสงจริงๆ เพื่อความชัวร์
        # ถ้าไม่ได้ใช้แบบ default ให้ไปดึงสูตรคำนวณขั้นสูงจากคลังแสง Hugging Face มาเสียบแทน!
        if self.rope_type != "default" and self.rope_type in ROPE_INIT_FUNCTIONS:
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        # รันเตาปฏิกรณ์คณิตศาสตร์ เพื่อสร้างอาเรย์เก็บค่าความถี่ (inv_freq)
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        # 🔒 ล็อกเป้า! คำสั่ง register_buffer คือการบอก PyTorch ว่า:
        # "ก้อนข้อมูลนี้คือค่าคงที่นะ เอาไปวางบน GPU ด้วย แต่ห้ามเอาไปเทรน (No Backprop) เด็ดขาด!"
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        # 💡 [Fix 3] ถอด `| None = None` ออกไปเลย!
        # เพราะฟังก์ชันนี้ยังไงก็ "ต้องมี" config เข้ามาคำนวณอยู่แล้ว Linter จะได้เลิกกลัว
        config: "YaeConfig",
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        เตาปฏิกรณ์คณิตศาสตร์สำหรับสร้าง Base Frequencies (inv_freq)

        สมการเบื้องหลัง:
        $inv\_freq_i = 1.0 / (base^{(2i / dim)})$
        - base: ฐานความถี่ (ปกติคือ 10000.0 สำหรับโมเดลทั่วไป, ถ้า Context ยาวๆ อาจจะใช้ 1000000.0)
        - dim: มิติของ Head (เช่น 128)
        - i: ตำแหน่งมิติ (ขยับทีละ 2 เพราะการหมุน 2D ต้องใช้ x, y คู่กัน)
        """

        # 💡 [Fix 4] Safe Get ตัว base แบบชิลๆ หลบ Linter
        rope_params = getattr(config, "rope_parameters", {}) or {}
        base = rope_params.get("rope_parameters", 10000.0)

        # คำนวณหาความกว้างของแต่ละ Head (ถ้าไม่มีระบุมา ก็จับขนาดเต็มหารด้วยจำนวน Head ซะเลย)
        dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )

        attention_factor = 1.0  # Unused in this type of RoPE

        # 🧮 Compute the inverse frequencies (หัวใจหลักของ RoPE)
        # สร้างอาเรย์ตัวเลขขยับทีละ 2 (0, 2, 4, ..., dim) เอาไปเข้าสมการยกกำลัง
        # มิติแรกๆ (ความถี่สูง) จะหมุนเร็วมาก / มิติท้ายๆ (ความถี่ต่ำ) จะค่อยๆ หมุนช้าๆ
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float
                )
                / dim
            )
        )
        return inv_freq, attention_factor

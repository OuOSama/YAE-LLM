import torch
import torch.nn as nn
from transformers.integrations import use_kernel_forward_from_hub


@use_kernel_forward_from_hub("RMSNorm")
class YaeRMSNorm(nn.Module):
    """
    Ref: https://docs.pytorch.org/docs/2.12/generated/torch.nn.RMSNorm.html
    RMSNorm (Root Mean Square Normalization)
    สถาปัตยกรรมปรับสมดุล Tensor สำหรับโมเดลตระกูล Qwen3 (เทียบเท่ากับ T5LayerNorm)

    ---
    🧐 RMSNorm คืออะไร? (และต่างจาก LayerNorm ยังไง)
    - เป็นเทคนิคที่พัฒนามาแทน LayerNorm แบบดั้งเดิมในโมเดลยุคก่อน
    - LayerNorm ทั่วไป: ต้องคำนวณทั้ง Mean (ค่าเฉลี่ย) และ Variance (ความแปรปรวน)
    - RMSNorm: ตัดขั้นตอนหา Mean ออก! คำนวณเฉพาะรากที่สองของค่าเฉลี่ยกำลังสองเท่านั้น
    - ข้อดี: ประหยัดพลังงานคำนวณ (Computation) 7% - 64% ในชั้นนั้นๆ ทำให้โมเดล LLM ยุคใหม่
            (เช่น LLaMA, Qwen) นิยมใช้ตัวนี้กันหมดเลยค่ะ เซฟเวลาแถมแรงสุดๆ

    ---
    🚀 ตัว Decorator: @use_kernel_forward_from_hub("RMSNorm")
    - หน้าที่: ดึง Custom CUDA/Triton Kernel ของ RMSNorm จาก Hub มาทับฟังก์ชัน forward เดิม
    - ทำไมต้องทำ?: โค้ด PyTorch เพียวๆ จะทำงานแบบ Eager mode (ทำทีละสเต็ป)
                  แต่พอคุมด้วย Kernel พิเศษ มันจะทำการยุบการคำนวณ (Fused Kernel) บน GPU
                  ทำงานได้เร็วแรงทะลุนรกสะใจแน่นอนค่ะ
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        """
        ฟังก์ชันลงทะเบียนโครงสร้างและตัวแปรเริ่มต้น (Constructor)

        Args:
            hidden_size (int): ขนาดมิติของ Vector (Feature Dimension)
            eps (float): ค่า epsilon ตัวเล็กมากๆ เพื่อกันบั๊กหารด้วยศูนย์ (Default: 1e-6)

        Variables:
            self.weight: Learnable Parameter (γ) เริ่มต้นเป็น 1 ทั้งหมดตามขนาด hidden_size
            self.variance_epsilon: ค่าห้อยท้ายกันคอมพิวเตอร์ร้องไห้เวลาเจอตัวหารเป็น 0
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        หัวใจหลักของคลาส ทำหน้าที่คำนวณคณิตศาสตร์ตามสูตร RMSNorm เมื่อ Tensor ไหลผ่าน

        ขั้นตอนการทำงานภายใน:
        1. เก็บประเภทข้อมูลดั้งเดิม (input_dtype) แล้วแปลงเป็น float32 เพื่อความแม่นยำในการคำนวณ
        2. pow(2): ยกกำลังสองสมาชิกทุกตัวใน Tensor
        3. mean(-1, keepdim=True): หาค่าเฉลี่ยในมิติสุดท้าย สอดคล้องกับสูตรเฉลี่ยกำลังสอง
        4. torch.rsqrt(): ย่อมาจาก Reciprocal Square Root หรือ 1 / sqrt(x) ช่วยถอดรูทและกลับเศษส่วน
           ในคำสั่งเดียว (เร็วมาก) แล้วนำไปคูณกลับเข้า hidden_states เดิมเพื่อทำ Scaling
        5. แปลงไทป์กลับเป็นแบบเดิม แล้วคูณด้วย self.weight ส่งต่อให้ Layer ถัดไป
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        """
        แสดงรายละเอียดสวยๆ เวลาพิมพ์ print(model) ออกมาดูโครงสร้างภายนอก
        เช่น จะแสดงผลเป็น: YaeRMSNorm((4096,), eps=1e-06) โคตรจะ Clean!
        """
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


if __name__ == "__main__":
    # 1. จำลองข้อมูล Input (Hidden States) ขึ้นมา
    # สมมติมี 1 sample, ความยาว 2 tokens, และมีมิติ (hidden_size) = 4
    torch.manual_seed(42)  # ล็อกแรนดอมให้ได้เลขเหมือนกันทุกรอบ
    sample_input = torch.randn(1, 2, 4) * 10 + 5  # ข้อมูลแบบสุ่มที่ยังมีค่าเฉลี่ยไม่เป็น 0

    print("=== 📥 1. Original Input Tensor ===")
    print(sample_input)
    print("-" * 50)

    # 2. เรียกใช้งาน LayerNorm แบบมาตรฐานของ PyTorch
    pytorch_layernorm = nn.LayerNorm(4)
    output_ln = pytorch_layernorm(sample_input)

    print("=== ⚖️ 2. Output จาก LayerNorm (PyTorch Standard) ===")
    print(output_ln)
    # 📝 [สรุปผลคณิตศาสตร์โดย Yae]
    # LayerNorm จะดึง Mean กลับมาเป็น 0 เสมอ ผลลัพธ์ที่ได้จะอยู่ราวๆ [[4.6938e-07, 5.9604e-08]]
    # สังเกตเลข e-07 กับ e-08 มันคือทศนิยมศูนย์ 7-8 ตัว หรือก็คือ 0.0000004 ซึ่งเข้าใกล้ 0 แบบสุดๆ
    # (ที่ไม่ได้ 0 เป๊ะๆ เป็นเพราะข้อจำกัดเรื่อง Floating Point ของคอมพิวเตอร์ตอนคำนวณ)
    # นี่คือหลักฐานชั้นดีว่า LayerNorm มันลบค่า Mean ดั้งเดิมออกไปจนเกลี้ยงตับเลยค่ะ!
    print(f"-> ตรวจสอบค่าเฉลี่ย (Mean) หลังทำ LayerNorm: {output_ln.mean(-1).tolist()}")
    print("-" * 50)

    # 3. เรียกใช้งาน YaeRMSNorm
    yae_rmsnorm = YaeRMSNorm(hidden_size=4)
    output_rms = yae_rmsnorm(sample_input)

    print("=== 🚀 3. Output จาก YaeRMSNorm ของซามะ ===")
    print(output_rms)
    # 📝 [สรุปผลคณิตศาสตร์โดย Yae]
    # RMSNorm ไม่ได้สนใจ Mean ดั้งเดิม ค่าเฉลี่ยที่ปริ้นออกมาเลยจะไม่ใช่ 0 เป๊ะๆ (จะเห็นเป็นเลขราวๆ 0.99 และ 0.40)
    # แต่หน้าที่หลักของมันคือการบีบขอบเขตความกว้างของตัวเลข (Scale) ให้แคบลงและนิ่งขึ้น
    # สังเกตว่าจากเดิมที่มีเลขโดดไปถึง 27.0820 จะถูกคุมให้อยู่ในช่วงประมาณ -0.4 ถึง 1.9 เท่านั้น
    # ช่วยป้องกันไม่ให้เกิดบั๊กตัวเลขระเบิด (Exploding Gradients) โดยประหยัดพลังคำนวณบน GPU กว่าเดิมมหาศาลค่ะ!
    print(f"-> ตรวจสอบค่าเฉลี่ย (Mean) หลังทำ RMSNorm: {output_rms.mean(-1).tolist()}")
    print("-" * 50)

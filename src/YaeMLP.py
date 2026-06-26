import torch.nn as nn
from transformers.activations import ACT2FN


class YaeMLP(nn.Module):
    """
    YaeMLP (Multi-Layer Perceptron / Feed-Forward Network)
    บล็อกประมวลผลสไตล์ SwiGLU อ้างอิงพิมพ์เขียวจากสถาปัตยกรรม Qwen 3

    ---
    🧠 สรุปกลไกคณิตศาสตร์ (SwiGLU):
    แทนที่จะใช้ Linear ชั้นเดียวทั่วไป โมเดลยุคใหม่จะใช้สายพาน 3 เส้นทำงานขนานกัน
    1. gate_proj + act_fn: ทำหน้าที่เป็น "ประตูคัดกรองข้อมูล" เลือกฟีเจอร์ที่สำคัญ
    2. up_proj: ระเบิดมิติข้อมูลให้กว้างขึ้น เพื่อให้โมเดลมีพื้นที่ในการคิดและประมวลผลซับซ้อน
    3. down_proj: ตบผลลัพธ์ที่ผ่านการคัดกรองแล้ว หดกลับมาเท่าขนาดเดิมก่อนส่งออกไป
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size  # มิติต้นทาง (เช่น hidden_size = 4)
        self.intermediate_size = config.intermediate_size

        # สายพานขยายมิติข้อมูล (hidden_size -> intermediate_size)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)

        # สายพานบีบมิติข้อมูลกลับมาเท่าเดิม (intermediate_size -> hidden_size)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        # ฟังก์ชันเปิด-ปิดประตูสัญญาณ (ใน Qwen 3 มักจะจับคู่กับ "silu")
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """
        กระบวนการแปลงมิติเวกเตอร์ (SwiGLU Forward Pass)
        มิติอินพุตเริ่มต้น x: [Batch, Seq_Len, Hidden_Size] (เช่น 4 มิติ)
        """
        # ปริ้นส่องตัวเลขก่อนเข้าเลเยอร์ (เอาแค่ Batch 0, Token 0 เพื่อความคลีนสายตา)
        print("🔍 [Yae Element Look] -> Input Vector (4 ตัวแรก):")
        print(f"   {x[0, 0].detach().tolist()}")
        print("-" * 40)

        # 1. [ขยายร่าง 3 เท่า + กรองสัญญาณ]
        gate_out = self.act_fn(self.gate_proj(x))

        # 2. [ขยายร่างขนาน]
        up_out = self.up_proj(x)

        # 2.5 [สร้างตัวแปรเก็บช่วงกำลังขยายร่างสูงสุด]
        expanded_state = gate_out * up_out

        # 📋 [ปริ้นส่องร่างระเบิด 12 มิติ เต็มๆ ตา!]
        print("📢 [Yae Element Look] -> Expanded State (ระเบิดร่างเป็น 12 ตัว!):")
        # ใช้สไลซ์ดึงทศนิยมสวยๆ ออกมาโชว์แถวยาวๆ เลยค่ะ
        formatted_elements = [
            round(num, 4) for num in expanded_state[0, 0].detach().tolist()
        ]
        print(f"   {formatted_elements}")
        print("-" * 40)

        # 3. [ตบมิติกลับ]
        output = self.down_proj(expanded_state)

        # ปริ้นส่องผลลัพธ์สุดท้ายหลังโดนตบกลับ
        print("🚀 [Yae Element Look] -> Final Output Vector (หดกลับเหลือ 4 ตัว):")
        print(f"   {output[0, 0].detach().tolist()}")
        print("=" * 60)

        return output


if __name__ == "__main__":
    import torch

    class MockConfig:
        hidden_size = 4
        intermediate_size = 12  # ขยาย 3 เท่า!
        hidden_act = "silu"

    torch.manual_seed(42)
    sample_input = torch.randn(1, 2, 4)

    print("=== 📥 1. Starting Input ===")
    print(f"Shape: {list(sample_input.shape)}")
    print("-" * 50)

    # ประกาศใช้งานคลาส YaeMLP
    mlp = YaeMLP(MockConfig())

    print("=== ⚙️ 2. Processing Forward Pass ===")
    final_output = mlp(sample_input)  # พอกดเรียกใช้ บรรทัดนี้จะพ่น [Yae Log] ออกมาทันที!
    print("-" * 50)

    print("=== 🚀 3. Final Output ===")
    print(f"Shape หลังโดนตบกลับ: {list(final_output.shape)}")

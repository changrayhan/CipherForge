"""SLG-HE-PIR 攻击测试套件 — 完整实施

src/attacks/
├── test_doubles/        测试替身框架（物理隔离模拟）
├── L1A_separation.py   L-1A: g_accum 分量分离攻击
├── L1C_dlg_inversion.py L-1C: SAP/DLG 梯度反演
├── L2_cutgrad.py       L-2: CutGrad 全家桶
├── L3B_pir_bytes.py    L-3B: PIR 字节分析
├── L4A0_hu_inversion.py L-4A-0: H_U Smashed-Data Inversion
├── L4A_hm_inversion.py L-4A: H_M Smashed-Data Inversion
├── L4B_tag_inversion.py L-4B: TAG/DLG 梯度反演
├── L5_s_inversion.py   L-5: S 方输入重构
├── L6_long_term.py     L-6: 长期训练隐私退化
├── M1_u_extract.py     M-1: U 方模型提取
├── M2_s_detect.py      M-2: S 方模型推断
├── M3_lora_internals.py M-3: LoRA 参数推断
├── M4_mia.py          M-4: Membership Inference
├── M5_v_infer.py       M-5: V 矩阵推断
├── P1_bfv_security.py  P-1: BFV 加密层安全
├── P2_prg.py           P-2: PRG 掩码安全
├── P3_pir.py           P-3: PIR 查询隐私
├── P4_P13_system.py    P-4~P-13: 系统/资源侧信道
"""

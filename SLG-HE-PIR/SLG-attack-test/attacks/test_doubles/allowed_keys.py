"""ALLOWED_KEYS — 各参与方在物理隔离下的合法可见字段。

由 docs/攻击类测试方案.md §0.1 物理隔离前提表推导。
严格限制各 MockParty 只能接收它"本应看到"的数据，任何越界字段触发 AssertionError。
"""
from __future__ import annotations
from typing import FrozenSet


# ─── 每方合法可见字段 ────────────────────────────────────────────────────────

ALLOWED_KEYS_U: FrozenSet[str] = frozenset({
    "input_ids",      # U 持有
    "output_ids",     # U 持有
    "H_U",            # U 计算
    "ct_list",        # U → M 密文（U 持有 pk_M）
    "prg_seed",       # U 持有（与 S 共享）
    "a_t",            # U 从 PIR 响应中恢复
    "logits",         # 可选（mock 下才可见）
})

ALLOWED_KEYS_M: FrozenSet[str] = frozenset({
    "H_U",            # U → M 明文（GPU tensor）
    "H_M",            # M 自算
    "g_H",            # 梯度（注入 autograd）
    "masked_arr",     # M 解密出的带符号 -V_y*scale + r_t
    "s_share",        # S → M 明文 share
    "sk_M",           # M 持有（解密用）
    "ct_list",        # U → M 密文（M 接收但解密后得到 masked_arr）
})

ALLOWED_KEYS_S: FrozenSet[str] = frozenset({
    "H_M",            # M → S 明文
    "V",              # S 持有
    "logits",         # S 自算
    "a_t",            # S 自算
    "s_share",        # S → M 明文 share
    "prg_seed",       # S 持有（与 U 共享）
    "ct_list",        # U → M（路径上可能经过 S，但不持有明文）
    "parity_real_bytes",  # PIR 响应字节
    "permutation_bit",
    "real_indices",
})

ALLOWED_KEYS: dict[str, FrozenSet[str]] = {
    "U": ALLOWED_KEYS_U,
    "M": ALLOWED_KEYS_M,
    "S": ALLOWED_KEYS_S,
}


def validate_allowed_keys(peer: str, received_keys: set[str]) -> None:
    """
    检查接收到的 payload keys 是否在 ALLOWED_KEYS 内。

    在 MockParty.receives() 入口调用，任何越界字段抛出 AssertionError。

    Args:
        peer:          "U" | "M" | "S"
        received_keys: payload 实际包含的字段集合

    Raises:
        AssertionError: 存在越界字段时
    """
    allowed = ALLOWED_KEYS.get(peer)
    if allowed is None:
        raise ValueError(f"Unknown peer: {peer}")

    forbidden = received_keys - allowed
    assert not forbidden, (
        f"[{peer}] received forbidden fields: {forbidden}. "
        f"Allowed: {allowed}"
    )


def filter_allowed_keys(peer: str, payload: dict) -> dict:
    """
    返回仅包含合法字段的 payload 字典（过滤掉越界字段）。
    不抛异常，仅静默裁剪。
    """
    allowed = ALLOWED_KEYS.get(peer, frozenset())
    return {k: v for k, v in payload.items() if k in allowed}

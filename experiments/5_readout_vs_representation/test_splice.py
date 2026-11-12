"""Unit tests for the experiment-5 splice. CPU only, no checkpoint, no download.

The tests that matter here are the ones guarding the tied-embedding trap: the whole
experiment rests on "these keys and only these keys came from the head donor", and a
splice that silently re-tied or mis-attributed the decoder would produce a plausible
number with no meaning.

    pytest experiments/5_readout_vs_representation/test_splice.py -q
"""
from __future__ import annotations

import math

import pytest

import common as C

torch = pytest.importorskip("torch")


# --------------------------------------------------------------------------- #
# Fixtures — a synthetic state dict with ESM's key shapes, and a tiny real model
# --------------------------------------------------------------------------- #
def make_sd(fill: float, n_layers: int = C.N_LAYERS) -> dict:
    """A state dict with ESM-2's key names, every tensor filled with ``fill``."""
    sd = {
        C.WORD_EMB_KEY: torch.full((33, 8), fill),
        "esm.encoder.emb_layer_norm_after.weight": torch.full((8,), fill),
        "esm.encoder.emb_layer_norm_after.bias": torch.full((8,), fill),
        "lm_head.dense.weight": torch.full((8, 8), fill),
        "lm_head.dense.bias": torch.full((8,), fill),
        "lm_head.layer_norm.weight": torch.full((8,), fill),
        "lm_head.layer_norm.bias": torch.full((8,), fill),
        "lm_head.bias": torch.full((33,), fill),
        C.DECODER_KEY: torch.full((33, 8), fill),
    }
    for i in range(n_layers):
        sd[f"esm.encoder.layer.{i}.attention.self.query.weight"] = torch.full((8, 8), fill)
        sd[f"esm.encoder.layer.{i}.output.dense.weight"] = torch.full((8, 8), fill)
    return sd


@pytest.fixture
def tiny_model():
    """A 2-layer ESM-2 with tied embeddings, built from config — nothing is downloaded."""
    transformers = pytest.importorskip("transformers")
    cfg = transformers.EsmConfig(
        vocab_size=33, hidden_size=8, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=16, max_position_embeddings=64, position_embedding_type="rotary",
        tie_word_embeddings=True,
    )
    return transformers.EsmForMaskedLM(cfg)


# --------------------------------------------------------------------------- #
# Key partitions
# --------------------------------------------------------------------------- #
def test_head_sets_are_nested_and_exclude_the_decoder():
    keys = list(make_sd(0.0))
    h1, h2, h3 = (C.head_keys(keys, hs) for hs in ("H1", "H2", "H3"))
    assert set(h1) < set(h2)
    assert set(h2) == set(h3)          # H3 adds the decoder via untying, not via a key
    assert all(not k.startswith("lm_head.decoder") for k in h3)
    assert C.WORD_EMB_KEY not in h3
    assert set(h1) == {"lm_head.dense.weight", "lm_head.dense.bias",
                       "lm_head.layer_norm.weight", "lm_head.layer_norm.bias", "lm_head.bias"}
    assert set(h2) - set(h1) == {"esm.encoder.emb_layer_norm_after.weight",
                                 "esm.encoder.emb_layer_norm_after.bias"}


def test_head_keys_rejects_unknown_set():
    with pytest.raises(ValueError):
        C.head_keys(list(make_sd(0.0)), "H9")


@pytest.mark.parametrize("k", [0, 1, 2, 4, 8, 12])
def test_layer_keys_takes_the_top_k_layers(k):
    keys = list(make_sd(0.0))
    got = C.layer_keys(keys, k)
    assert len(got) == 2 * k                                   # 2 params per layer here
    idx = {int(g.split("esm.encoder.layer.")[1].split(".")[0]) for g in got}
    assert idx == set(range(C.N_LAYERS - k, C.N_LAYERS))


# --------------------------------------------------------------------------- #
# The splice
# --------------------------------------------------------------------------- #
def test_identity_splice_is_bitwise_a_noop():
    sd = make_sd(1.0)
    out = C.splice_state_dicts(sd, sd, C.head_keys(list(sd), "H1"), untied=False)
    assert set(out) == set(sd)
    assert all(torch.equal(out[k], sd[k]) for k in sd)


def test_splice_takes_exactly_the_intended_keys():
    trunk, head = make_sd(1.0), make_sd(2.0)
    transferred = C.head_keys(list(trunk), "H2")
    out = C.splice_state_dicts(trunk, head, transferred, untied=False)
    for k in out:
        if k == C.DECODER_KEY:
            continue
        want = head[k] if k in transferred else trunk[k]
        assert torch.equal(out[k], want), k


def test_tied_decoder_follows_the_trunk_and_untied_follows_the_head():
    trunk, head = make_sd(1.0), make_sd(2.0)
    transferred = C.head_keys(list(trunk), "H1")

    tied = C.splice_state_dicts(trunk, head, transferred, untied=False)
    assert torch.equal(tied[C.DECODER_KEY], trunk[C.WORD_EMB_KEY])
    assert torch.equal(tied[C.WORD_EMB_KEY], trunk[C.WORD_EMB_KEY])

    untied = C.splice_state_dicts(trunk, head, transferred, untied=True)
    assert torch.equal(untied[C.DECODER_KEY], head[C.WORD_EMB_KEY])
    assert torch.equal(untied[C.WORD_EMB_KEY], trunk[C.WORD_EMB_KEY])   # input side unchanged


def test_splice_reports_a_missing_head_key():
    trunk = make_sd(1.0)
    head = {k: v for k, v in make_sd(2.0).items() if k != "lm_head.bias"}
    with pytest.raises(KeyError):
        C.splice_state_dicts(trunk, head, C.head_keys(list(trunk), "H1"), untied=False)


def test_untie_actually_separates_storage(tiny_model):
    m = tiny_model
    assert (m.lm_head.decoder.weight.data_ptr()
            == m.esm.embeddings.word_embeddings.weight.data_ptr()), "fixture is not tied"

    new_w = torch.full_like(m.lm_head.decoder.weight, 7.0)
    C._untie_decoder(m, new_w)

    assert m.config.tie_word_embeddings is False
    assert (m.lm_head.decoder.weight.data_ptr()
            != m.esm.embeddings.word_embeddings.weight.data_ptr())
    assert torch.equal(m.lm_head.decoder.weight, new_w)
    # And the input embedding is genuinely unaffected by writing the output projection.
    assert not torch.equal(m.esm.embeddings.word_embeddings.weight, new_w)


def test_tied_model_cannot_be_untied_by_a_state_dict_write(tiny_model):
    """The trap itself, asserted — this is why _untie_decoder has to exist."""
    m = tiny_model
    sd = m.state_dict()
    sd[C.DECODER_KEY] = torch.full_like(sd[C.DECODER_KEY], 3.0)
    sd[C.WORD_EMB_KEY] = torch.full_like(sd[C.WORD_EMB_KEY], 5.0)
    m.load_state_dict(sd, strict=True)
    assert torch.equal(m.lm_head.decoder.weight, m.esm.embeddings.word_embeddings.weight)


# --------------------------------------------------------------------------- #
# Freezing (T3)
# --------------------------------------------------------------------------- #
def test_trainable_mask_partitions_the_model(tiny_model):
    n_head, n_rest = C.trainable_mask(tiny_model, "head_only")
    n_trunk, n_rest2 = C.trainable_mask(tiny_model, "trunk_only")
    assert n_head + n_rest == n_trunk + n_rest2         # same total either way
    assert n_head == n_rest2 and n_trunk == n_rest      # exact complements
    assert 0 < n_head < n_trunk

    keys = set(C.head_keys([n for n, _ in tiny_model.named_parameters()], "H1"))
    C.trainable_mask(tiny_model, "head_only")
    assert {n for n, p in tiny_model.named_parameters() if p.requires_grad} == keys
    # The output projection is tied to the input embedding, so head_only must NOT train it.
    assert not tiny_model.esm.embeddings.word_embeddings.weight.requires_grad


# --------------------------------------------------------------------------- #
# The headline statistic
# --------------------------------------------------------------------------- #
def test_frac_block_endpoints_are_exactly_zero_and_one():
    base, blocked = 13.1149, 81344.31
    assert C.frac_block(base, base, blocked) == pytest.approx(0.0, abs=1e-12)
    assert C.frac_block(blocked, base, blocked) == pytest.approx(1.0, abs=1e-12)


def test_frac_block_is_linear_in_nats():
    base, blocked = 10.0, 10.0 * math.e ** 4          # a 4-nat block
    mid = 10.0 * math.e ** 2
    assert C.frac_block(mid, base, blocked) == pytest.approx(0.5)


def test_frac_block_is_none_rather_than_inf_when_nothing_blocked():
    assert C.frac_block(13.0, 13.0, 13.0) is None
    assert C.frac_block(float("nan"), 13.0, 100.0) is None
    assert C.frac_block(13.0, 0.0, 100.0) is None


def test_benign_drift():
    assert C.benign_drift(11.459, 11.226) == pytest.approx(0.02076, abs=1e-4)
    assert C.benign_drift(float("inf"), 11.226) is None

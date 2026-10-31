"""Placement tests: R25 sizing with D-041 KV cache, check 8, host preference, free VRAM, loaded and missing-KV rules.

Depends on: pastore.placement, pastore.spec (Host) and pytest only; plain values, no fakes.
"""

import pytest

from pastore.placement import (
    GIB,
    HostLoad,
    ModelFacts,
    base_need,
    free_bytes,
    kv_bytes_per_slot,
    need_bytes,
    place,
)
from pastore.spec import WORKSTATION, Host

MODEL = "m:latest"
OTHER = "other:latest"
SMALL = 2 * GIB
# Placement reads the policy field `vram_gib=10` as 10 GiB (10 * 2**30 bytes, `int(vram_gib * GIB)`); base_need(EXACT_10_GIB) is exactly that
OVER_10_GIB = 45 * GIB // 6 + 1
EXACT_10_GIB = 45 * GIB // 6


def host(name: str, vram_gib: float, max_models: int = 2, parallel: int = 1) -> Host:
    return Host(
        name=name,
        url=f"http://{name}:11434",
        vram_gib=vram_gib,
        max_models=max_models,
        parallel=parallel,
        context=8192,
    )


def load(
    h: Host,
    sizes: dict[str, int] | None = None,
    resident: dict[str, int] | None = None,
    held: tuple[str, ...] = (),
    *,
    kv: int | None = 0,
    models: dict[str, ModelFacts] | None = None,
) -> HostLoad:
    if models is None:
        models = {name: ModelFacts(size, kv) for name, size in (sizes or {MODEL: SMALL}).items()}
    return HostLoad(host=h, models=models, resident=resident or {}, held=held)


EXECUTOR = host("executor", 10)
WORKSTATION_HOST = host(WORKSTATION, 24)


def test_gib_is_1024_cubed_and_the_base_need_adds_exactly_one_gib() -> None:
    assert GIB == 1_073_741_824
    assert base_need(0) == 1_073_741_824


def test_base_need_exact_values() -> None:
    assert base_need(5) == 1_073_741_830
    assert base_need(1) == 1_073_741_826
    assert base_need(3) == 1_073_741_828
    assert base_need(10 * GIB) == 13_958_643_712


def test_base_need_large_sizes_use_exact_integer_math() -> None:
    assert base_need(35184372088835) == 42_222_320_248_426
    assert base_need(35184372088836) == 42_222_320_248_428
    assert base_need(2**62 + 5) == 5_534_023_223_186_607_315


def test_base_need_boundary_sizes_around_10_gib() -> None:
    assert base_need(EXACT_10_GIB) == 10_737_418_240
    assert base_need(OVER_10_GIB) == 10_737_418_242


def test_kv_bytes_per_slot_exact_values() -> None:
    assert kv_bytes_per_slot(64, 8, 128, 32768) == 8 * GIB
    assert kv_bytes_per_slot(2, 3, 5, 7) == 840


def test_need_bytes_is_the_base_need_plus_parallel_times_kv() -> None:
    assert need_bytes(ModelFacts(SMALL, 7), 3) == base_need(SMALL) + 21
    assert need_bytes(ModelFacts(SMALL, 7), 1) == base_need(SMALL) + 7
    assert need_bytes(ModelFacts(SMALL, 0), 3) == base_need(SMALL)


def test_need_bytes_is_the_loaded_figure_once_seen_whatever_the_estimate() -> None:
    assert need_bytes(ModelFacts(SMALL, 7, loaded=123), 3) == 123
    assert need_bytes(ModelFacts(SMALL, None, loaded=123), 3) == 123
    assert need_bytes(ModelFacts(SMALL, 7, loaded=0), 3) == 0


def test_need_bytes_is_none_when_the_kv_size_is_unknown() -> None:
    assert need_bytes(ModelFacts(SMALL, None), 3) is None


def test_check8_a_model_needing_over_10_gib_never_lands_on_the_executor() -> None:
    big = {MODEL: OVER_10_GIB}
    assert place(MODEL, [load(EXECUTOR, big)]) is None
    assert place(MODEL, [load(EXECUTOR, big, resident={MODEL: OVER_10_GIB})]) is None
    on_both = [load(EXECUTOR, big, resident={MODEL: OVER_10_GIB}), load(WORKSTATION_HOST, big)]
    assert place(MODEL, on_both) == WORKSTATION
    assert place(MODEL, [load(WORKSTATION_HOST, big), load(EXECUTOR, big)]) == WORKSTATION


def test_check8_a_model_needing_exactly_10_gib_fits_an_empty_10_gib_host() -> None:
    assert place(MODEL, [load(EXECUTOR, {MODEL: EXACT_10_GIB})]) == "executor"


def test_a_loaded_host_wins_even_when_it_is_the_workstation() -> None:
    loads = [load(EXECUTOR), load(WORKSTATION_HOST, resident={MODEL: SMALL})]
    assert place(MODEL, loads) == WORKSTATION


def test_the_first_loaded_host_wins_among_several() -> None:
    loads = [
        load(host("b", 24), resident={MODEL: SMALL}),
        load(host("a", 24), resident={MODEL: SMALL}),
    ]
    assert place(MODEL, loads) == "b"


def test_a_loaded_host_needs_no_free_vram() -> None:
    size = 16 * GIB
    ws = load(WORKSTATION_HOST, {MODEL: size}, resident={MODEL: size})
    assert free_bytes(ws) < base_need(size)
    assert place(MODEL, [ws]) == WORKSTATION


def test_a_model_resident_with_zero_vram_still_counts_as_loaded() -> None:
    full = load(EXECUTOR, {MODEL: SMALL, OTHER: GIB}, resident={MODEL: 0, OTHER: 10 * GIB})
    assert free_bytes(full) == 0
    assert place(MODEL, [full]) == "executor"


def test_the_executor_is_preferred_to_the_workstation_when_nothing_is_loaded() -> None:
    assert place(MODEL, [load(WORKSTATION_HOST), load(EXECUTOR)]) == "executor"


def test_hosts_other_than_the_workstation_are_tried_in_the_given_order() -> None:
    first, second = host("first", 8), host("second", 10)
    assert place(MODEL, [load(first), load(second), load(WORKSTATION_HOST)]) == "first"
    assert place(MODEL, [load(second), load(first), load(WORKSTATION_HOST)]) == "second"


def test_host_order_decides_whatever_the_host_sizes() -> None:
    big, small = host("big", 48), host("small", 8)
    assert place(MODEL, [load(big), load(small), load(WORKSTATION_HOST)]) == "big"
    assert place(MODEL, [load(small), load(big), load(WORKSTATION_HOST)]) == "small"


def test_a_non_workstation_host_bigger_than_the_workstation_still_wins_when_listed_after_it() -> (
    None
):
    assert place(MODEL, [load(WORKSTATION_HOST), load(host("big", 48))]) == "big"


def test_the_workstation_is_used_when_the_executor_is_full() -> None:
    full = load(EXECUTOR, {MODEL: SMALL, OTHER: 10 * GIB}, resident={OTHER: 10 * GIB})
    assert place(MODEL, [full, load(WORKSTATION_HOST)]) == WORKSTATION


def test_the_free_check_uses_each_hosts_own_size_for_the_model() -> None:
    small_full = load(EXECUTOR, {MODEL: GIB, OTHER: GIB}, resident={OTHER: 9 * GIB})
    big_busy = load(WORKSTATION_HOST, {MODEL: 16 * GIB, OTHER: GIB}, resident={OTHER: 10 * GIB})
    assert place(MODEL, [small_full, big_busy]) is None
    big_full = load(EXECUTOR, {MODEL: 5 * GIB, OTHER: GIB}, resident={OTHER: 4 * GIB})
    small_free = load(WORKSTATION_HOST, {MODEL: GIB})
    assert place(MODEL, [big_full, small_free]) == WORKSTATION


def test_free_bytes_of_an_empty_host_is_its_vram() -> None:
    assert free_bytes(load(EXECUTOR)) == 10 * GIB


def test_resident_models_reduce_free_vram() -> None:
    ld = load(EXECUTOR, resident={OTHER: 3 * GIB, "third": GIB})
    assert free_bytes(ld) == 6 * GIB


def test_a_held_model_not_yet_resident_reduces_free_vram_by_its_need() -> None:
    ld = load(EXECUTOR, {MODEL: SMALL, OTHER: 3 * GIB}, held=(OTHER,))
    assert free_bytes(ld) == 10 * GIB - base_need(3 * GIB)


def test_a_held_models_need_includes_the_hosts_parallel_times_its_kv() -> None:
    ld = load(host("h", 24, parallel=2), {OTHER: 3 * GIB}, held=(OTHER,), kv=GIB)
    assert free_bytes(ld) == 24 * GIB - base_need(3 * GIB) - 2 * GIB


def test_a_held_model_with_a_loaded_figure_reduces_free_vram_by_it() -> None:
    models = {OTHER: ModelFacts(3 * GIB, GIB, loaded=2 * GIB)}
    assert free_bytes(load(EXECUTOR, held=(OTHER,), models=models)) == 8 * GIB


def test_a_held_model_already_resident_is_not_counted_twice() -> None:
    ld = load(EXECUTOR, {MODEL: SMALL, OTHER: 3 * GIB}, resident={OTHER: 3 * GIB}, held=(OTHER,))
    assert free_bytes(ld) == 7 * GIB


def test_a_held_model_missing_from_models_makes_the_host_full() -> None:
    ld = load(EXECUTOR, held=(OTHER,))
    assert free_bytes(ld) == 0
    assert place(MODEL, [ld]) is None
    assert place(MODEL, [ld, load(WORKSTATION_HOST)]) == WORKSTATION


def test_a_held_model_with_an_unknown_need_makes_the_host_full() -> None:
    ld = load(EXECUTOR, {MODEL: SMALL, OTHER: GIB}, held=(OTHER,), kv=None)
    assert free_bytes(ld) == 0


def test_free_bytes_counts_a_model_held_twice_once() -> None:
    ld = load(host("h", 24, max_models=3), {MODEL: SMALL, OTHER: 3 * GIB}, held=(OTHER, OTHER))
    assert free_bytes(ld) == 24 * GIB - base_need(3 * GIB)


def test_free_bytes_may_go_negative() -> None:
    assert free_bytes(load(EXECUTOR, resident={OTHER: 11 * GIB})) == -GIB


def test_a_held_model_makes_the_host_too_full_for_a_new_run() -> None:
    held = load(EXECUTOR, {MODEL: 5 * GIB, OTHER: 5 * GIB}, held=(OTHER,))
    assert free_bytes(held) < base_need(5 * GIB)
    assert place(MODEL, [held]) is None
    assert place(MODEL, [held, load(WORKSTATION_HOST, {MODEL: 5 * GIB})]) == WORKSTATION


def test_resident_models_make_the_host_too_full_for_a_new_run() -> None:
    busy = load(EXECUTOR, {MODEL: 5 * GIB, OTHER: 5 * GIB}, resident={OTHER: 5 * GIB})
    assert place(MODEL, [busy]) is None


def test_a_model_fits_when_the_need_equals_free_vram_and_not_one_byte_less() -> None:
    need = base_need(SMALL)
    fits = load(EXECUTOR, {MODEL: SMALL, OTHER: GIB}, resident={OTHER: 10 * GIB - need})
    assert free_bytes(fits) == need
    assert place(MODEL, [fits]) == "executor"
    tight = load(EXECUTOR, {MODEL: SMALL, OTHER: GIB}, resident={OTHER: 10 * GIB - need + 1})
    assert place(MODEL, [tight]) is None


@pytest.mark.parametrize(("gib", "floor_bytes"), [(5.6, 6_012_954_214), (5.7, 6_120_328_396)])
def test_a_fractional_card_is_floored_to_whole_bytes_and_the_boundary_is_exact(
    gib: float, floor_bytes: int
) -> None:
    """5.6 GiB is 6012954214.4 bytes and 5.7 GiB is 6120328396.8: floor, not round, gives the figures above."""
    ex = host("executor", gib)
    assert free_bytes(load(ex)) == floor_bytes

    def on_card(need: int) -> HostLoad:
        return load(ex, models={MODEL: ModelFacts(SMALL, 0, loaded=need)})

    assert place(MODEL, [on_card(floor_bytes)]) == "executor"
    assert place(MODEL, [on_card(floor_bytes + 1)]) is None


def test_a_host_at_max_models_is_not_eligible() -> None:
    full = load(host("executor", 10, max_models=1), {MODEL: SMALL, OTHER: GIB}, held=(OTHER,))
    assert place(MODEL, [full]) is None
    assert place(MODEL, [full, load(WORKSTATION_HOST)]) == WORKSTATION


def test_a_host_at_max_models_is_not_eligible_even_when_the_model_is_resident() -> None:
    full = load(
        host("executor", 10, max_models=1),
        {MODEL: SMALL, OTHER: GIB},
        resident={MODEL: SMALL},
        held=(OTHER,),
    )
    assert place(MODEL, [full, load(WORKSTATION_HOST)]) == WORKSTATION


def test_a_host_below_max_models_is_eligible() -> None:
    ld = load(host("executor", 10, max_models=3), {MODEL: SMALL, OTHER: GIB}, held=(OTHER,))
    assert place(MODEL, [ld]) == "executor"


def test_max_models_counts_runs_not_distinct_models() -> None:
    ld = load(host("big", 48, max_models=2), {MODEL: SMALL, OTHER: GIB}, held=(OTHER, OTHER), kv=0)
    assert place(MODEL, [ld]) is None


def test_a_model_already_held_on_a_host_goes_to_another_host() -> None:
    busy = load(EXECUTOR, resident={MODEL: SMALL}, held=(MODEL,))
    assert place(MODEL, [busy]) is None
    assert place(MODEL, [busy, load(WORKSTATION_HOST)]) == WORKSTATION


def test_a_model_absent_from_a_hosts_models_is_never_placed_there() -> None:
    assert place(MODEL, [load(EXECUTOR, {OTHER: SMALL})]) is None
    assert place(MODEL, [load(EXECUTOR, {OTHER: SMALL}), load(WORKSTATION_HOST)]) == WORKSTATION
    assert place(MODEL, [load(EXECUTOR, {OTHER: SMALL}, resident={MODEL: SMALL})]) is None


def test_none_when_there_are_no_hosts() -> None:
    assert place(MODEL, []) is None


def test_none_when_nothing_fits() -> None:
    huge = {MODEL: 30 * GIB}
    assert place(MODEL, [load(EXECUTOR, huge), load(WORKSTATION_HOST, huge)]) is None


def test_d041_the_kv_cache_per_slot_counts_parallel_times_into_the_need() -> None:
    kv = kv_bytes_per_slot(64, 8, 128, 32768)
    assert kv == 8 * GIB
    models = {MODEL: ModelFacts(10 * GIB, kv)}
    parallel1 = host(WORKSTATION, 24, parallel=1)
    assert need_bytes(models[MODEL], 1) == 21 * GIB
    assert place(MODEL, [load(parallel1, models=models)]) == WORKSTATION
    assert place(MODEL, [load(host(WORKSTATION, 24, parallel=2), models=models)]) is None
    assert place(MODEL, [load(host(WORKSTATION, 24, parallel=3), models=models)]) is None


def test_d041_a_need_equal_to_the_card_fits_and_one_byte_more_does_not() -> None:
    fits = {MODEL: ModelFacts(10 * GIB, 11 * GIB)}
    over = {MODEL: ModelFacts(10 * GIB, 11 * GIB + 1)}
    assert place(MODEL, [load(WORKSTATION_HOST, models=fits)]) == WORKSTATION
    assert place(MODEL, [load(WORKSTATION_HOST, models=over)]) is None


def test_d041_the_kv_cache_counts_against_free_vram_not_only_the_card() -> None:
    models = {MODEL: ModelFacts(GIB, 3 * GIB), OTHER: ModelFacts(GIB, 0)}
    assert place(MODEL, [load(EXECUTOR, models=models, resident={OTHER: 6 * GIB})]) is None
    assert place(MODEL, [load(EXECUTOR, models=models, resident={OTHER: 5 * GIB})]) is None
    assert place(MODEL, [load(EXECUTOR, models=models, resident={OTHER: 3 * GIB})]) == "executor"


def test_d041_a_loaded_figure_above_the_card_makes_the_host_ineligible() -> None:
    models = {MODEL: ModelFacts(SMALL, 0, loaded=10 * GIB + 1)}
    assert place(MODEL, [load(EXECUTOR, models=models)]) is None
    assert place(MODEL, [load(EXECUTOR, models=models, resident={MODEL: 10 * GIB + 1})]) is None
    ws = load(WORKSTATION_HOST, {MODEL: SMALL})
    assert place(MODEL, [load(EXECUTOR, models=models), ws]) == WORKSTATION


def test_d041_a_loaded_figure_below_the_estimate_makes_the_host_eligible() -> None:
    estimate_only = {MODEL: ModelFacts(5 * GIB, 4 * GIB)}
    loaded = {MODEL: ModelFacts(5 * GIB, 4 * GIB, loaded=4 * GIB)}
    assert place(MODEL, [load(EXECUTOR, models=estimate_only)]) is None
    assert place(MODEL, [load(EXECUTOR, models=loaded)]) == "executor"


def test_d041_the_loaded_figure_replaces_the_estimate_in_the_free_check_both_ways() -> None:
    estimate = base_need(3 * GIB) + 2 * GIB
    other = {OTHER: ModelFacts(GIB, 0)}
    plain = {MODEL: ModelFacts(3 * GIB, 2 * GIB)} | other
    smaller = {MODEL: ModelFacts(3 * GIB, 2 * GIB, loaded=4 * GIB)} | other
    larger = {MODEL: ModelFacts(3 * GIB, 2 * GIB, loaded=estimate + GIB)} | other
    assert 4 * GIB < estimate < estimate + GIB < 10 * GIB
    five_free = {OTHER: 5 * GIB}
    assert place(MODEL, [load(EXECUTOR, models=plain, resident=five_free)]) is None
    assert place(MODEL, [load(EXECUTOR, models=smaller, resident=five_free)]) == "executor"
    estimate_free = {OTHER: 10 * GIB - estimate}
    assert place(MODEL, [load(EXECUTOR, models=plain, resident=estimate_free)]) == "executor"
    assert place(MODEL, [load(EXECUTOR, models=larger, resident=estimate_free)]) is None


def test_d041_an_unknown_kv_size_is_placed_on_an_empty_card() -> None:
    assert place(MODEL, [load(EXECUTOR, kv=None)]) == "executor"


def test_d041_an_unknown_kv_size_is_refused_when_any_model_is_resident() -> None:
    busy = load(EXECUTOR, {MODEL: SMALL, OTHER: GIB}, resident={OTHER: GIB}, kv=None)
    assert place(MODEL, [busy]) is None
    assert place(MODEL, [busy, load(WORKSTATION_HOST, kv=None)]) == WORKSTATION


def test_d041_an_unknown_kv_size_is_refused_when_any_run_is_held() -> None:
    models = {MODEL: ModelFacts(SMALL, None), OTHER: ModelFacts(GIB, 0)}
    busy = load(EXECUTOR, held=(OTHER,), models=models)
    assert place(MODEL, [busy]) is None
    assert place(MODEL, [busy, load(WORKSTATION_HOST, kv=None)]) == WORKSTATION


def test_d041_an_unknown_kv_size_is_refused_when_the_base_need_exceeds_the_card() -> None:
    assert place(MODEL, [load(EXECUTOR, {MODEL: OVER_10_GIB}, kv=None)]) is None
    fallback = [
        load(EXECUTOR, {MODEL: OVER_10_GIB}, kv=None),
        load(WORKSTATION_HOST, {MODEL: OVER_10_GIB}, kv=None),
    ]
    assert place(MODEL, fallback) == WORKSTATION


def test_d041_an_unknown_kv_size_with_the_base_need_equal_to_the_card_fits() -> None:
    assert place(MODEL, [load(EXECUTOR, {MODEL: EXACT_10_GIB}, kv=None)]) == "executor"


def test_the_free_check_counts_the_candidate_hosts_parallel_slots() -> None:
    models = {MODEL: ModelFacts(GIB, GIB)}
    parallel3 = host("executor", 10, parallel=3)
    assert need_bytes(models[MODEL], 3) == 5_583_457_485
    assert (
        place(MODEL, [load(host("executor", 10), models=models, resident={OTHER: 5 * GIB})])
        == "executor"
    )
    both = [
        load(parallel3, models=models, resident={OTHER: 5 * GIB}),
        load(WORKSTATION_HOST, models=models),
    ]
    assert place(MODEL, both) == WORKSTATION


def test_the_card_cap_counts_the_candidate_hosts_parallel_slots_for_a_resident_model() -> None:
    models = {MODEL: ModelFacts(10 * GIB, 8 * GIB)}
    parallel1 = load(host(WORKSTATION, 24, parallel=1), models=models, resident={MODEL: 20 * GIB})
    parallel2 = load(host(WORKSTATION, 24, parallel=2), models=models, resident={MODEL: 20 * GIB})
    assert place(MODEL, [parallel1]) == WORKSTATION
    assert place(MODEL, [parallel2]) is None


def test_d041_an_unknown_kv_size_with_a_loaded_figure_is_placed_on_a_busy_card() -> None:
    models = {MODEL: ModelFacts(SMALL, None, loaded=3 * GIB)}
    assert place(MODEL, [load(EXECUTOR, models=models, resident={OTHER: 2 * GIB})]) == "executor"


def test_d041_an_unknown_kv_size_is_refused_when_the_only_resident_model_is_itself() -> None:
    assert place(MODEL, [load(EXECUTOR, kv=None, resident={MODEL: SMALL})]) is None

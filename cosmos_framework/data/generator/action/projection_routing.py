# Copyright (c) 2026, BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Action128 projection IDs; independent of source normalization IDs."""

NUM_PROJECTION_SLOTS = 64
# Input-only absolute EEF State; never reuse the delta Action projection (0).
EEF_ABSOLUTE_STATE_HEAD_ID = 32
# The mixed dexterous pretraining collection uses its own shared State/Action
# pair. Existing robot-action EEF keeps heads 32/0 below.
MIXED_EEF_STATE_HEAD_ID = 62
MIXED_EEF_ACTION_HEAD_ID = 63
PROJECTION_ROUTING_VERSION = "action128-projection-routing-v5-compact-pretrain-joints"
PROJECTION_HEAD_IDS = {
    "eef20/shared": 0,
    "joint128/agibot_g1__mixed_end_effectors": 1,
    "joint128/agilex_aloha__parallel_gripper": 2,
    "joint128/agilex_aloha_mobile__parallel_gripper": 3,
    "joint128/agilex_piper__parallel_gripper": 4,
    "joint128/ark_bimanual__parallel_gripper": 5,
    "joint128/ark_mobile_bimanual__parallel_gripper": 6,
    "joint128/arx_x5__parallel_gripper": 7,
    "joint128/franka_panda__parallel_gripper": 8,
    "joint128/galbot_g1_mobile_bimanual__parallel_grippers": 9,
    "joint128/robomind_agilex_bimanual__native_end_effectors": 10,
    "joint128/tianyi_humanoid__parallel_grippers": 11,
    "joint128/tianyi_mobile_humanoid__parallel_grippers": 12,
    "joint128/tienkung__gello_parallel_grippers": 13,
    "joint128/tienkung__native_hand6": 14,
    "joint128/tienkung__parallel_grippers": 15,
    "joint128/tienkung__prod1_parallel_grippers": 16,
    "joint128/tienkung__xsens_hands": 17,
    "joint128/tienkung_sim__native_hand12": 18,
    "joint128/universal_robots_ur5__native_hand12": 19,
    "joint128/universal_robots_ur5__parallel_gripper": 20,
    "joint128/universal_robots_ur5e__parallel_gripper": 21,
    # 2026-10-09 catalogs: append only; preserve existing checkpoint slot IDs.
    "joint128/agibot_g2__parallel_grippers": 22,
    "joint128/galaxea_r1lite_mobile_bimanual__parallel_grippers": 23,
    "joint128/galaxea_r1pro_mobile_bimanual__parallel_grippers": 24,
    "joint128/agilex_split_aloha__parallel_grippers": 25,
    "joint128/arx_lift2__parallel_grippers": 26,
    "joint128/genie1__parallel_grippers": 27,
    "joint128/panda_omron_mobile__parallel_gripper": 28,
    "joint128/paxini_df2__faas_bimanual_hands": 29,
    "joint128/sharpa_north__wave_bimanual_hands": 30,
    # 2026-10-12: preserve existing IDs and the fixed 64-slot capacity.
    "joint128/agilex_piper_x__parallel_gripper": 31,
    # Published tactile/pretraining names; numeric IDs match the stable
    # projection families below. Bench2Dex intentionally uses one family key.
    "joint128/ufactory_xarm6__allegro_v5": 35,
    "joint128/ufactory_xarm6__inspire_rh56dftp": 36,
    "joint128/ufactory_xarm6__inspire_rh56f1": 37,
    "joint128/flexiv_rizon4s__parallel_gripper": 38,
    "joint128/trex_vega1__sharpa_wave": 40,
    "joint128/dwheel__tactile_parallel_gripper": 34,
    "joint128/qingloong_v2_5__tactile_parallel_gripper": 41,
}

# Published pretraining projection families. This explicit policy groups the
# mixed collection's EEF-containing targets, but not ordinary robot-action EEF.
PUBLISHED_PROJECTION_FAMILY_IDS = {
    "joint::sharpa_north_wave_bimanual_v1": 30,
    "joint::trex_vega1_sharpa_wave_bimanual_v1": 40,
    "action::bi_taccap_gripper__parallel_gripper/eef20/observed_local_delta+gripper_derived_next_observation/canonical-tcp-local-delta-gripper-next-observation-20261029-v1": MIXED_EEF_ACTION_HEAD_ID,
    "action::openneodata_aloha__r1_source_parallel_gripper/eef20/observed_local_delta+gripper_derived_next_observation/canonical-tcp-local-delta-gripper-next-observation-20261029-v1": MIXED_EEF_ACTION_HEAD_ID,
    "action::openneodata_arx5_single__r1_source_parallel_gripper/eef20/observed_local_delta+gripper_derived_next_observation/canonical-tcp-local-delta-gripper-next-observation-20261029-v1": MIXED_EEF_ACTION_HEAD_ID,
    "action::openneodata_flexiv__r1_source_parallel_gripper/eef20/observed_local_delta+gripper_derived_next_observation/canonical-tcp-local-delta-gripper-next-observation-20261029-v1": MIXED_EEF_ACTION_HEAD_ID,
    "action::openneodata_ur__r1_source_parallel_gripper/eef20/observed_local_delta+gripper_derived_next_observation/canonical-tcp-local-delta-gripper-next-observation-20261029-v1": MIXED_EEF_ACTION_HEAD_ID,
    "action::pika__parallel_gripper/eef20/observed_local_delta+gripper_derived_next_observation/canonical-tcp-local-delta-gripper-next-observation-20261029-v1": MIXED_EEF_ACTION_HEAD_ID,
    "action::psibot_itw_glove/ego128/wrist_camera_local_delta+hand32_absolute_next/ego128-psibot-20261028-v1": MIXED_EEF_ACTION_HEAD_ID,
    "action::tacumi_handheld__parallel_gripper/eef20/observed_local_delta+gripper_derived_next_observation/canonical-tcp-local-delta-gripper-next-observation-20261029-v1": MIXED_EEF_ACTION_HEAD_ID,
    "action::xarm7_parallel_gripper_eef20_v1": MIXED_EEF_ACTION_HEAD_ID,
    "joint::bench2dex_all_pretrain_v1": 33,
    "joint::dwheel_tactile__parallel_gripper/joint128/absolute_observed_next_state/absolute-observed-next-state-20261028-v1": 34,
    "joint::dwheel__tactile_parallel_gripper/joint128/absolute_observed_next_state/absolute-observed-next-state-20261028-v1": 34,
    "joint::franka_panda__parallel_gripper/joint128/observed_absolute_state_t_plus_1/univtac-franka9-two-finger-mean04-next-state-v1": 8,
    "joint::hrdexdb__allegro_v5/joint128/observed_absolute_state_t_plus_1/hrdex-r1r2-20261026-v1": 35,
    "joint::hrdexdb__inspire_dftp/joint128/observed_absolute_state_t_plus_1/hrdex-r1r2-20261026-v1": 36,
    "joint::hrdexdb__inspire_f1/joint128/observed_absolute_state_t_plus_1/hrdex-r1r2-20261026-v1": 37,
    "joint::ufactory_xarm6__allegro_v5/joint128/observed_absolute_state_t_plus_1/hrdex-r1r2-20261026-v1": 35,
    "joint::ufactory_xarm6__inspire_rh56dftp/joint128/observed_absolute_state_t_plus_1/hrdex-r1r2-20261026-v1": 36,
    "joint::ufactory_xarm6__inspire_rh56f1/joint128/observed_absolute_state_t_plus_1/hrdex-r1r2-20261026-v1": 37,
    "joint::openneodata_aloha__r1_source_parallel_gripper/joint128/absolute_observed_next_state/absolute-observed-next-state-20261028-v1": 2,
    "joint::openneodata_arx5_single__r1_source_parallel_gripper/joint128/absolute_observed_next_state/absolute-observed-next-state-20261028-v1": 7,
    "joint::agilex_aloha__parallel_gripper/joint128/absolute_observed_next_state/absolute-observed-next-state-20261028-v1": 2,
    "joint::arx_x5__parallel_gripper/joint128/absolute_observed_next_state/absolute-observed-next-state-20261028-v1": 7,
    "joint::openneodata_flexiv__r1_source_parallel_gripper/joint128/absolute_observed_next_state/absolute-observed-next-state-20261028-v1": 38,
    "joint::flexiv_rizon4s__parallel_gripper/joint128/absolute_observed_next_state/absolute-observed-next-state-20261028-v1": 38,
    "joint::openneodata_ur__r1_source_parallel_gripper/joint128/absolute_observed_next_state/absolute-observed-next-state-20261028-v1": 39,
    "joint::paxini_df2_bimanual_v1": 29,
    "joint::qingloongv2_5_tactile__parallel_gripper/joint128/absolute_observed_next_state/absolute-observed-next-state-20261028-v1": 41,
    "joint::qingloong_v2_5__tactile_parallel_gripper/joint128/absolute_observed_next_state/absolute-observed-next-state-20261028-v1": 41,
    "state::bi_taccap_gripper__parallel_gripper/eef20/observed_local_delta+gripper_derived_next_observation/canonical-tcp-local-delta-gripper-next-observation-20261029-v1": MIXED_EEF_STATE_HEAD_ID,
    "state::openneodata_aloha__r1_source_parallel_gripper/eef20/observed_local_delta+gripper_derived_next_observation/canonical-tcp-local-delta-gripper-next-observation-20261029-v1": MIXED_EEF_STATE_HEAD_ID,
    "state::openneodata_arx5_single__r1_source_parallel_gripper/eef20/observed_local_delta+gripper_derived_next_observation/canonical-tcp-local-delta-gripper-next-observation-20261029-v1": MIXED_EEF_STATE_HEAD_ID,
    "action::agilex_aloha__parallel_gripper/eef20/observed_local_delta+gripper_derived_next_observation/canonical-tcp-local-delta-gripper-next-observation-20261029-v1": MIXED_EEF_ACTION_HEAD_ID,
    "action::arx_x5__parallel_gripper/eef20/observed_local_delta+gripper_derived_next_observation/canonical-tcp-local-delta-gripper-next-observation-20261029-v1": MIXED_EEF_ACTION_HEAD_ID,
    "state::agilex_aloha__parallel_gripper/eef20/observed_local_delta+gripper_derived_next_observation/canonical-tcp-local-delta-gripper-next-observation-20261029-v1": MIXED_EEF_STATE_HEAD_ID,
    "state::arx_x5__parallel_gripper/eef20/observed_local_delta+gripper_derived_next_observation/canonical-tcp-local-delta-gripper-next-observation-20261029-v1": MIXED_EEF_STATE_HEAD_ID,
    "state::openneodata_flexiv__r1_source_parallel_gripper/eef20/observed_local_delta+gripper_derived_next_observation/canonical-tcp-local-delta-gripper-next-observation-20261029-v1": MIXED_EEF_STATE_HEAD_ID,
    "action::flexiv_rizon4s__parallel_gripper/eef20/observed_local_delta+gripper_derived_next_observation/canonical-tcp-local-delta-gripper-next-observation-20261029-v1": MIXED_EEF_ACTION_HEAD_ID,
    "state::flexiv_rizon4s__parallel_gripper/eef20/observed_local_delta+gripper_derived_next_observation/canonical-tcp-local-delta-gripper-next-observation-20261029-v1": MIXED_EEF_STATE_HEAD_ID,
    "state::openneodata_ur__r1_source_parallel_gripper/eef20/observed_local_delta+gripper_derived_next_observation/canonical-tcp-local-delta-gripper-next-observation-20261029-v1": MIXED_EEF_STATE_HEAD_ID,
    "state::pika__parallel_gripper/eef20/observed_local_delta+gripper_derived_next_observation/canonical-tcp-local-delta-gripper-next-observation-20261029-v1": MIXED_EEF_STATE_HEAD_ID,
    "state::psibot_itw_glove/ego128/wrist_camera_local_delta+hand32_absolute_next/ego128-psibot-20261028-v1": MIXED_EEF_STATE_HEAD_ID,
    "state::tacumi_handheld__parallel_gripper/eef20/observed_local_delta+gripper_derived_next_observation/canonical-tcp-local-delta-gripper-next-observation-20261029-v1": MIXED_EEF_STATE_HEAD_ID,
    "state::xarm7_parallel_gripper_eef20_v1": MIXED_EEF_STATE_HEAD_ID,
}


def projection_family_id(key: str) -> int:
    if key not in PUBLISHED_PROJECTION_FAMILY_IDS:
        raise ValueError(f"Unregistered published projection family: {key}")
    return PUBLISHED_PROJECTION_FAMILY_IDS[key]


def projection_head_id(action_space: str, embodiment: str) -> int:
    key = "eef20/shared" if action_space == "eef20" else f"{action_space}/{embodiment}"
    if key not in PROJECTION_HEAD_IDS:
        raise ValueError(f"Unregistered Action128 projection key: {key}")
    return PROJECTION_HEAD_IDS[key]

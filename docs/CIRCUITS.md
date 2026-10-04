# Verified sub-connectomes

Generated from `data/circuits/circuit_report.json` (produced by `scripts/verify_circuits.py`). All matrices are anatomical synaptic-contact counts, edges >= 3 contacts; signs are neurotransmitter predictions. See LIMITATIONS.md.

## mushroom_body_L

| statistic | value |
|---|---|
| side | L |
| n_PN | 120 |
| n_PN_types | 60 |
| n_KC | 2019 |
| n_APL | 1 |
| n_DPM | 1 |
| n_MBON | 48 |
| n_PAM | 158 |
| n_PPL1 | 8 |
| n_compartments_MBON | 32 |
| frac_KC_with_PN_input | 0.9271916790490342 |
| KC_fan_in_PN_neurons_mean | 5.297542735042735 |
| KC_fan_in_PN_types_mean | 5.156517094017094 |
| KC_fan_in_PN_types_median | 5.0 |
| PN_fan_out_KC_mean | 82.64166666666667 |
| PN_KC_contacts | 183709 |
| PN_KC_edges | 9917 |
| APL_covers_frac_KC | 1.0 |
| KC_MBON_edges | 17931 |
| MBON_fan_in_KC_mean | 373.5625 |
| DAN_KC_edges | 7212 |
| DAN_MBON_edges | 486 |
| MBON_DAN_feedback_edges | 339 |
| MBON_DAN_feedback_contacts | 2861 |
| MBON_MBON_edges | 282 |
| KC_KC_edges | 56963 |
| APL_nt | ['gaba'] |
| DPM_nt | ['dopamine'] |
| MBON_nt_counts | acetylcholine: 25, gaba: 10, glutamate: 13 |
| KC_types | KC: 1, KCa'b'-ap1: 97, KCa'b'-ap2: 133, KCa'b'-m: 110, KCab-c: 277, KCab-m: 310, KCab-p: 64, KCab-s: 234, KCg: 1, KCg-d: 99, KCg-m: 689, KCg-s1: 1, KCg-s2: 1, KCg-s3: 1, KCg-s4: 1 |
| PN_nt_counts | acetylcholine: 116, gaba: 2, glutamate: 2 |
| frac_KC_inhibitory_flag | 0.0 |

| check | result | measured |
|---|---|---|
| KC_count_~2000_per_hemisphere | PASS | 2019 |
| APL_single_neuron_GABA | PASS | ['gaba'] |
| APL_inhibits_>95%_KC | PASS | 1.0 |
| KC_fan_in_PN_types_5-7_(Caron2013) | PASS | 5.156517094017094 |
| MBON_~20-40_types_per_side | PASS | 48 |
| MBON_DAN_feedback_exists | PASS | 339 |
| DAN_target_MBON_dendrites_(compartment_coupling) | PASS | 486 |
| PAM_>>PPL1_in_count | PASS | [158, 8] |
| KC_excitatory_(cholinergic) | PASS | 0.0 |

## mushroom_body_R

| statistic | value |
|---|---|
| side | R |
| n_PN | 119 |
| n_PN_types | 61 |
| n_KC | 2045 |
| n_APL | 1 |
| n_DPM | 1 |
| n_MBON | 49 |
| n_PAM | 158 |
| n_PPL1 | 8 |
| n_compartments_MBON | 32 |
| frac_KC_with_PN_input | 0.9251833740831296 |
| KC_fan_in_PN_neurons_mean | 5.231501057082452 |
| KC_fan_in_PN_types_mean | 5.06553911205074 |
| KC_fan_in_PN_types_median | 5.0 |
| PN_fan_out_KC_mean | 83.17647058823529 |
| PN_KC_contacts | 189838 |
| PN_KC_edges | 9898 |
| APL_covers_frac_KC | 0.9995110024449878 |
| KC_MBON_edges | 17985 |
| MBON_fan_in_KC_mean | 367.0408163265306 |
| DAN_KC_edges | 6177 |
| DAN_MBON_edges | 478 |
| MBON_DAN_feedback_edges | 300 |
| MBON_DAN_feedback_contacts | 2605 |
| MBON_MBON_edges | 255 |
| KC_KC_edges | 60418 |
| APL_nt | ['gaba'] |
| DPM_nt | ['dopamine'] |
| MBON_nt_counts | acetylcholine: 25, gaba: 11, glutamate: 13 |
| KC_types | KC: 1, KCa'b'-ap1: 102, KCa'b'-ap2: 158, KCa'b'-m: 95, KCab-c: 211, KCab-m: 226, KCab-p: 65, KCab-s: 423, KCg-d: 107, KCg-m: 653, KCg-s1: 1, KCg-s2: 1, KCg-s3: 1, KCg-s4: 1 |
| PN_nt_counts | acetylcholine: 116, gaba: 3 |
| frac_KC_inhibitory_flag | 0.0 |

| check | result | measured |
|---|---|---|
| KC_count_~2000_per_hemisphere | PASS | 2045 |
| APL_single_neuron_GABA | PASS | ['gaba'] |
| APL_inhibits_>95%_KC | PASS | 0.9995110024449878 |
| KC_fan_in_PN_types_5-7_(Caron2013) | PASS | 5.06553911205074 |
| MBON_~20-40_types_per_side | PASS | 49 |
| MBON_DAN_feedback_exists | PASS | 300 |
| DAN_target_MBON_dendrites_(compartment_coupling) | PASS | 478 |
| PAM_>>PPL1_in_count | PASS | [158, 8] |
| KC_excitatory_(cholinergic) | PASS | 0.0 |

## central_complex

| statistic | value |
|---|---|
| EPG | 46 |
| PEN_a | 20 |
| PEN_b | 22 |
| PEG | 18 |
| Delta7 | 42 |
| EL | 18 |
| ExR | 26 |
| n_core | 148 |
| edges_core | 7216 |
| contacts_core | 136003 |
| frac_with_angle | 1.0 |
| nt | EPG: {'acetylcholine': 46}, PEN_a: {'acetylcholine': 20}, PEN_b: {'acetylcholine': 22}, PEG: {'acetylcholine': 18}, Delta7: {'glutamate': 42} |

| check | result | measured |
|---|---|---|
| EPG_46_(hemibrain 46) | PASS | 46 |
| PEN_42_(hemibrain 20 PEN_a + 22 PEN_b) | PASS | [20, 22] |
| Delta7_42_(hemibrain 42) | PASS | 42 |
| PEG_18_(hemibrain 18) | PASS | 18 |
| EPG_cholinergic | PASS | None |
| Delta7_glutamatergic_(inhibitory) | PASS | None |
| EPG->EPG_peaks_at_offset0 | PASS | [27.7, 6.6, 0.2, 0.0, 0.1, 0.0, 0.1, 6.7] |
| PEN_output_vs_input_shift_opposite_sign_per_side | PASS | {'L': -7.293401147690235, 'R': 6.892066636989686} |
| EPG->Delta7_input_peaks_opposite_(offset4) | PASS | [1.1, 1.8, 9.5, 18.8, 21.0, 17.4, 8.6, 1.7] |
| Delta7_output_local_(offset0_max) | PASS | [13.5, 2.0, 0.0, 0.0, 0.1, 0.0, 0.0, 1.4] |
| effective_EPG->D7->EPG_inhibition_min_at_0_max_opposite | PASS | [117.0, 176.0, 816.0, 1625.0, 1873.0, 1592.0, 834.0, 185.0] |
| PEN_L_and_PEN_R_shift_bump_opposite_ways | PASS | {'L': [9.4, 0.4, 0.0, 0.0, 0.1, 8.1, 92.7, 93.0], 'R': [7.3, 95.7, 88.0, 7.0, 0.0, 0.0, 0.0, 0.0]} |

Connection strength (mean contacts per pair) as a function of angular offset between source and target heading, 8 bins of 45 deg starting at 0:

| pair | 0 | 45 | 90 | 135 | 180 | 225 | 270 | 315 |
|---|---|---|---|---|---|---|---|---|
| EPG->EPG | 27.7 | 6.6 | 0.2 | 0.0 | 0.1 | 0.0 | 0.1 | 6.7 |
| EPG->PEN_a | 7.6 | 11.0 | 10.9 | 0.8 | 0.0 | 0.4 | 10.7 | 11.0 |
| EPG->PEN_b | 5.7 | 11.6 | 9.8 | 0.0 | 0.0 | 0.0 | 10.0 | 12.0 |
| PEN_a->EPG | 7.5 | 27.8 | 26.7 | 3.2 | 0.0 | 3.8 | 27.8 | 27.6 |
| PEN_b->EPG | 0.9 | 20.8 | 16.8 | 0.3 | 0.0 | 0.3 | 18.0 | 19.4 |
| EPG->Delta7 | 1.1 | 1.8 | 9.5 | 18.8 | 21.0 | 17.4 | 8.6 | 1.7 |
| Delta7->EPG | 13.5 | 2.0 | 0.0 | 0.0 | 0.1 | 0.0 | 0.0 | 1.4 |
| Delta7->PEN_a | 8.4 | 1.1 | 0.0 | 0.0 | 0.1 | 0.0 | 0.0 | 0.8 |
| Delta7->Delta7 | 15.8 | 9.6 | 13.3 | 18.7 | 19.7 | 18.2 | 13.8 | 10.5 |
| EPG->PEG | 42.8 | 4.2 | 2.2 | 0.2 | 2.3 | 0.1 | 2.5 | 3.3 |
| PEG->EPG | 4.4 | 0.7 | 0.5 | 0.0 | 0.6 | 0.0 | 0.6 | 0.8 |
| PEN_a[L]->EPG | 8.4 | 0.4 | 0.0 | 0.0 | 0.1 | 7.6 | 56.2 | 54.8 |
| PEN_a[R]->EPG | 6.5 | 54.8 | 53.9 | 6.4 | 0.0 | 0.0 | 0.0 | 0.0 |
| PEN_b[L]->EPG | 0.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.5 | 36.5 | 38.2 |
| PEN_b[R]->EPG | 0.8 | 40.9 | 34.1 | 0.6 | 0.0 | 0.0 | 0.0 | 0.0 |
| EPG->Delta7->EPG (effective inhibition) | 116.7 | 175.6 | 815.8 | 1624.8 | 1872.6 | 1592.0 | 833.8 | 185.4 |
| PEN output-minus-input shift (deg) | -7.3, 6.9 | | | | | | | |

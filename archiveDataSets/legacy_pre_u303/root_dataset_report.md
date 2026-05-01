# Dataset Report

## Generation
- Seed: 42
- Users requested: 208
- Weeks requested: 65
- Validation ratio (target): 20%

## Size
- Train rows: 791,479
- Val rows: 199,694
- Total rows: 991,173
- Train users: 163
- Val users: 45
- Unique exercises: 34

## Quality
- RIR mean: 2.11
- RIR min/max: 0/5
- Reps mean: 4.47
- Weight mean (kg): 41.28

## Exercise Coverage
| exercise_id | train_sets | val_sets | total_sets |
|---|---:|---:|---:|
| ab_wheel | 5671 | 1027 | 6698 |
| bench_press | 84223 | 21329 | 105552 |
| bird_dog | 10136 | 2388 | 12524 |
| bulgarian_split_squat | 29452 | 8061 | 37513 |
| chest_press_machine | 8341 | 1807 | 10148 |
| close_grip_bench | 21718 | 4700 | 26418 |
| dead_bug | 10864 | 2520 | 13384 |
| deadlift | 35787 | 9827 | 45614 |
| decline_bench | 10154 | 1616 | 11770 |
| dips | 15940 | 3155 | 19095 |
| dumbbell_flyes | 19524 | 5118 | 24642 |
| farmers_walk | 3786 | 1071 | 4857 |
| high_bar_squat | 29880 | 7512 | 37392 |
| incline_bench | 25543 | 5812 | 31355 |
| incline_bench_45 | 11824 | 2434 | 14258 |
| lat_pulldown | 28722 | 7665 | 36387 |
| leg_curl | 43604 | 11338 | 54942 |
| leg_extension | 7410 | 1347 | 8757 |
| leg_press | 27467 | 6706 | 34173 |
| leg_raises | 5980 | 1531 | 7511 |
| low_bar_squat | 15568 | 3702 | 19270 |
| ohp | 42793 | 10650 | 53443 |
| pendlay_row | 96774 | 25495 | 122269 |
| plank | 7563 | 1455 | 9018 |
| pull_up | 8215 | 2920 | 11135 |
| rdl | 45989 | 11143 | 57132 |
| reverse_fly | 22745 | 5196 | 27941 |
| seal_row | 1271 | 885 | 2156 |
| skull_crusher | 18264 | 5079 | 23343 |
| spoto_press | 24093 | 6478 | 30571 |
| squat | 54342 | 15373 | 69715 |
| suitcase_carry | 5504 | 1127 | 6631 |
| sumo_deadlift | 6637 | 2190 | 8827 |
| trx_bodysaw | 5695 | 1037 | 6732 |

## Missing Coverage
- None

## Key Sequence Coverage
| sequence | train_sessions | val_sessions | total_sessions | train_users | val_users | total_users |
|---|---:|---:|---:|---:|---:|---:|
| bench_press -> skull_crusher | 3842 | 969 | 4811 | 107 | 30 | 137 |
| bench_press -> ohp | 6167 | 1341 | 7508 | 150 | 42 | 192 |
| incline_bench -> dumbbell_flyes | 5575 | 1313 | 6888 | 127 | 32 | 159 |
| rdl -> leg_curl | 10440 | 2550 | 12990 | 157 | 41 | 198 |
| squat -> leg_press | 8169 | 2053 | 10222 | 148 | 39 | 187 |
| deadlift -> rdl | 4807 | 1195 | 6002 | 131 | 32 | 163 |

## Common Sequence Coverage
| sequence | train_users | val_users | total_users | train_sessions | val_sessions | total_sessions |
|---|---:|---:|---:|---:|---:|---:|
| bench_press -> pendlay_row | 163 | 44 | 207 | 9622 | 2537 | 12159 |
| bench_press -> dead_bug | 163 | 44 | 207 | 4285 | 1044 | 5329 |
| pendlay_row -> dead_bug | 163 | 44 | 207 | 4278 | 1043 | 5321 |
| leg_curl -> bird_dog | 161 | 42 | 203 | 3747 | 897 | 4644 |
| leg_press -> bird_dog | 157 | 43 | 200 | 3728 | 932 | 4660 |
| leg_press -> leg_curl | 157 | 42 | 199 | 9940 | 2356 | 12296 |
| rdl -> bird_dog | 157 | 42 | 199 | 3606 | 903 | 4509 |
| rdl -> leg_curl | 157 | 41 | 198 | 10440 | 2550 | 12990 |
| rdl -> leg_press | 156 | 42 | 198 | 5439 | 1233 | 6672 |
| squat -> bird_dog | 154 | 41 | 195 | 3693 | 886 | 4579 |
| ohp -> dead_bug | 150 | 43 | 193 | 3817 | 971 | 4788 |
| bench_press -> ohp | 150 | 42 | 192 | 6167 | 1341 | 7508 |
| pendlay_row -> ohp | 150 | 42 | 192 | 3817 | 969 | 4786 |
| squat -> leg_curl | 152 | 39 | 191 | 8215 | 2098 | 10313 |
| squat -> leg_press | 148 | 39 | 187 | 8169 | 2053 | 10222 |
| squat -> rdl | 148 | 38 | 186 | 3378 | 800 | 4178 |
| reverse_fly -> dead_bug | 146 | 37 | 183 | 3445 | 849 | 4294 |
| bench_press -> reverse_fly | 146 | 36 | 182 | 5654 | 1220 | 6874 |
| pendlay_row -> reverse_fly | 146 | 36 | 182 | 4749 | 1258 | 6007 |
| pendlay_row -> lat_pulldown | 139 | 40 | 179 | 9569 | 2555 | 12124 |
| leg_curl -> pendlay_row | 142 | 37 | 179 | 4955 | 1266 | 6221 |
| dumbbell_flyes -> pendlay_row | 138 | 38 | 176 | 6472 | 1697 | 8169 |
| bench_press -> dumbbell_flyes | 138 | 38 | 176 | 5380 | 1495 | 6875 |
| deadlift -> pendlay_row | 139 | 36 | 175 | 5200 | 1421 | 6621 |
| bench_press -> spoto_press | 134 | 39 | 173 | 5398 | 1600 | 6998 |
| deadlift -> leg_curl | 136 | 35 | 171 | 4750 | 1222 | 5972 |
| ohp -> reverse_fly | 134 | 36 | 170 | 7393 | 2014 | 9407 |
| rdl -> pendlay_row | 136 | 33 | 169 | 5015 | 1238 | 6253 |
| spoto_press -> pendlay_row | 129 | 37 | 166 | 4870 | 1386 | 6256 |
| high_bar_squat -> bulgarian_split_squat | 130 | 35 | 165 | 5753 | 1579 | 7332 |
| bulgarian_split_squat -> leg_curl | 130 | 35 | 165 | 5516 | 1550 | 7066 |
| high_bar_squat -> leg_curl | 130 | 35 | 165 | 5501 | 1550 | 7051 |
| spoto_press -> dumbbell_flyes | 129 | 36 | 165 | 4871 | 1379 | 6250 |
| deadlift -> rdl | 131 | 32 | 163 | 4807 | 1195 | 6002 |
| incline_bench -> dumbbell_flyes | 127 | 32 | 159 | 5575 | 1313 | 6888 |
| incline_bench -> pendlay_row | 127 | 32 | 159 | 5573 | 1311 | 6884 |
| dumbbell_flyes -> lat_pulldown | 121 | 35 | 156 | 5425 | 1435 | 6860 |
| bench_press -> lat_pulldown | 121 | 35 | 156 | 4451 | 1228 | 5679 |
| spoto_press -> lat_pulldown | 121 | 35 | 156 | 4451 | 1228 | 5679 |
| bench_press -> incline_bench | 124 | 32 | 156 | 4449 | 1102 | 5551 |

## Rare Sequences (<2 users)
- None

## Split Sequence Audit
- All same-session ordered sequences with >=2 users are present in both train and val.

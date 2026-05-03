# Dataset Report

## Generation
- Seed: 42
- Users requested: 320
- Weeks requested: 65
- Validation ratio (target): 20%

## Size
- Train rows: 1,232,898
- Val rows: 312,009
- Total rows: 1,544,907
- Train users: 256
- Val users: 64
- Unique exercises: 34

## Quality
- RIR mean: 2.08
- RIR min/max: 0/5
- Reps mean: 4.51
- Weight mean (kg): 40.85

## Exercise Coverage
| exercise_id | train_sets | val_sets | total_sets |
|---|---:|---:|---:|
| ab_wheel | 8939 | 2650 | 11589 |
| bench_press | 120242 | 30609 | 150851 |
| bird_dog | 15884 | 4162 | 20046 |
| bulgarian_split_squat | 46092 | 11473 | 57565 |
| chest_press_machine | 14888 | 3494 | 18382 |
| close_grip_bench | 30705 | 7608 | 38313 |
| dead_bug | 17088 | 4466 | 21554 |
| deadlift | 54644 | 13608 | 68252 |
| decline_bench | 12243 | 3547 | 15790 |
| dips | 24573 | 5950 | 30523 |
| dumbbell_flyes | 28632 | 7560 | 36192 |
| farmers_walk | 6928 | 1445 | 8373 |
| high_bar_squat | 45468 | 11481 | 56949 |
| incline_bench | 37709 | 10545 | 48254 |
| incline_bench_45 | 17592 | 3806 | 21398 |
| lat_pulldown | 46020 | 12438 | 58458 |
| leg_curl | 68160 | 16846 | 85006 |
| leg_extension | 11889 | 3519 | 15408 |
| leg_press | 42416 | 10806 | 53222 |
| leg_raises | 10258 | 2149 | 12407 |
| low_bar_squat | 25128 | 6847 | 31975 |
| ohp | 69501 | 17969 | 87470 |
| pendlay_row | 150371 | 38002 | 188373 |
| plank | 13193 | 3272 | 16465 |
| pull_up | 17109 | 3535 | 20644 |
| rdl | 69864 | 17946 | 87810 |
| reverse_fly | 38136 | 9473 | 47609 |
| seal_row | 5896 | 1190 | 7086 |
| skull_crusher | 29889 | 7203 | 37092 |
| spoto_press | 39825 | 10306 | 50131 |
| squat | 84420 | 20509 | 104929 |
| suitcase_carry | 9570 | 2297 | 11867 |
| sumo_deadlift | 10723 | 2628 | 13351 |
| trx_bodysaw | 8903 | 2670 | 11573 |

## Missing Coverage
- None

## Key Sequence Coverage
| sequence | train_sessions | val_sessions | total_sessions | train_users | val_users | total_users |
|---|---:|---:|---:|---:|---:|---:|
| bench_press -> skull_crusher | 5376 | 1453 | 6829 | 170 | 43 | 213 |
| bench_press -> ohp | 9132 | 2444 | 11576 | 236 | 61 | 297 |
| incline_bench -> dumbbell_flyes | 8149 | 2282 | 10431 | 203 | 54 | 257 |
| rdl -> leg_curl | 15960 | 4165 | 20125 | 238 | 60 | 298 |
| squat -> leg_press | 12516 | 3092 | 15608 | 234 | 59 | 293 |
| deadlift -> rdl | 7212 | 1785 | 8997 | 206 | 52 | 258 |

## Common Sequence Coverage
| sequence | train_users | val_users | total_users | train_sessions | val_sessions | total_sessions |
|---|---:|---:|---:|---:|---:|---:|
| bench_press -> dead_bug | 256 | 64 | 320 | 6731 | 1697 | 8428 |
| bench_press -> pendlay_row | 255 | 64 | 319 | 14416 | 3697 | 18113 |
| pendlay_row -> dead_bug | 255 | 64 | 319 | 6667 | 1697 | 8364 |
| leg_press -> bird_dog | 247 | 63 | 310 | 5784 | 1494 | 7278 |
| leg_curl -> bird_dog | 246 | 61 | 307 | 5853 | 1452 | 7305 |
| leg_press -> leg_curl | 245 | 60 | 305 | 15492 | 3940 | 19432 |
| squat -> bird_dog | 242 | 60 | 302 | 5804 | 1430 | 7234 |
| rdl -> bird_dog | 239 | 62 | 301 | 5607 | 1479 | 7086 |
| rdl -> leg_press | 237 | 62 | 299 | 8472 | 2304 | 10776 |
| rdl -> leg_curl | 238 | 60 | 298 | 15960 | 4165 | 20125 |
| bench_press -> ohp | 236 | 61 | 297 | 9132 | 2444 | 11576 |
| ohp -> dead_bug | 236 | 61 | 297 | 6134 | 1610 | 7744 |
| pendlay_row -> ohp | 236 | 61 | 297 | 6134 | 1610 | 7744 |
| bench_press -> reverse_fly | 236 | 59 | 295 | 8615 | 2329 | 10944 |
| pendlay_row -> reverse_fly | 235 | 59 | 294 | 9241 | 2357 | 11598 |
| squat -> leg_press | 234 | 59 | 293 | 12516 | 3092 | 15608 |
| reverse_fly -> dead_bug | 233 | 59 | 292 | 5746 | 1516 | 7262 |
| squat -> leg_curl | 232 | 58 | 290 | 12900 | 3128 | 16028 |
| pendlay_row -> lat_pulldown | 227 | 58 | 285 | 15339 | 4146 | 19485 |
| leg_curl -> pendlay_row | 226 | 57 | 283 | 7405 | 1829 | 9234 |
| squat -> rdl | 225 | 58 | 283 | 5252 | 1364 | 6616 |
| bench_press -> dumbbell_flyes | 223 | 56 | 279 | 7744 | 1989 | 9733 |
| ohp -> reverse_fly | 221 | 57 | 278 | 12568 | 3340 | 15908 |
| spoto_press -> pendlay_row | 222 | 56 | 278 | 7984 | 2117 | 10101 |
| deadlift -> pendlay_row | 223 | 55 | 278 | 7918 | 1936 | 9854 |
| dumbbell_flyes -> pendlay_row | 221 | 56 | 277 | 9420 | 2514 | 11934 |
| bench_press -> spoto_press | 214 | 56 | 270 | 7624 | 2009 | 9633 |
| deadlift -> leg_curl | 216 | 54 | 270 | 7144 | 1722 | 8866 |
| rdl -> pendlay_row | 211 | 55 | 266 | 7424 | 1892 | 9316 |
| incline_bench -> pendlay_row | 204 | 54 | 258 | 8161 | 2293 | 10454 |
| deadlift -> rdl | 206 | 52 | 258 | 7212 | 1785 | 8997 |
| spoto_press -> dumbbell_flyes | 205 | 53 | 258 | 6975 | 1855 | 8830 |
| incline_bench -> dumbbell_flyes | 203 | 54 | 257 | 8149 | 2282 | 10431 |
| spoto_press -> lat_pulldown | 204 | 53 | 257 | 7314 | 1964 | 9278 |
| spoto_press -> reverse_fly | 202 | 52 | 254 | 882 | 219 | 1101 |
| leg_curl -> lat_pulldown | 195 | 52 | 247 | 6105 | 1604 | 7709 |
| high_bar_squat -> bulgarian_split_squat | 200 | 46 | 246 | 8909 | 2217 | 11126 |
| dumbbell_flyes -> lat_pulldown | 194 | 52 | 246 | 7858 | 2142 | 10000 |
| bench_press -> lat_pulldown | 195 | 51 | 246 | 6425 | 1736 | 8161 |
| bench_press -> incline_bench | 193 | 52 | 245 | 6431 | 1762 | 8193 |

## Rare Sequences (<2 users)
- None

## Split Sequence Audit
- All same-session ordered sequences with >=2 users are present in both train and val.

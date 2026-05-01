# Dataset Report

## Generation
- Seed: 42
- Users requested: 303
- Weeks requested: 65
- Validation ratio (target): 20%

## Size
- Train rows: 1,138,097
- Val rows: 297,796
- Total rows: 1,435,893
- Train users: 242
- Val users: 61
- Unique exercises: 34

## Quality
- RIR mean: 2.10
- RIR min/max: 0/5
- Reps mean: 4.50
- Weight mean (kg): 40.46

## Exercise Coverage
| exercise_id | train_sets | val_sets | total_sets |
|---|---:|---:|---:|
| ab_wheel | 5996 | 1457 | 7453 |
| bench_press | 116423 | 28745 | 145168 |
| bird_dog | 14058 | 3546 | 17604 |
| bulgarian_split_squat | 39449 | 10452 | 49901 |
| chest_press_machine | 14290 | 4069 | 18359 |
| close_grip_bench | 29100 | 6854 | 35954 |
| dead_bug | 15154 | 3904 | 19058 |
| deadlift | 54628 | 13590 | 68218 |
| decline_bench | 13153 | 3150 | 16303 |
| dips | 20361 | 5330 | 25691 |
| dumbbell_flyes | 25080 | 6858 | 31938 |
| farmers_walk | 6460 | 1953 | 8413 |
| high_bar_squat | 37863 | 10279 | 48142 |
| incline_bench | 31831 | 8398 | 40229 |
| incline_bench_45 | 16088 | 4398 | 20486 |
| lat_pulldown | 41715 | 11457 | 53172 |
| leg_curl | 63472 | 16700 | 80172 |
| leg_extension | 10719 | 2517 | 13236 |
| leg_press | 38216 | 9833 | 48049 |
| leg_raises | 9322 | 2815 | 12137 |
| low_bar_squat | 26363 | 6121 | 32484 |
| ohp | 63457 | 16804 | 80261 |
| pendlay_row | 141053 | 37621 | 178674 |
| plank | 10844 | 2807 | 13651 |
| pull_up | 17589 | 5265 | 22854 |
| rdl | 64668 | 16190 | 80858 |
| reverse_fly | 35441 | 9305 | 44746 |
| seal_row | 6787 | 1988 | 8775 |
| skull_crusher | 24977 | 7669 | 32646 |
| spoto_press | 37821 | 9799 | 47620 |
| squat | 82570 | 21832 | 104402 |
| suitcase_carry | 7450 | 2131 | 9581 |
| sumo_deadlift | 9745 | 2472 | 12217 |
| trx_bodysaw | 5954 | 1487 | 7441 |

## Missing Coverage
- None

## Key Sequence Coverage
| sequence | train_sessions | val_sessions | total_sessions | train_users | val_users | total_users |
|---|---:|---:|---:|---:|---:|---:|
| bench_press -> skull_crusher | 4680 | 1431 | 6111 | 153 | 44 | 197 |
| bench_press -> ohp | 8919 | 2203 | 11122 | 224 | 56 | 280 |
| incline_bench -> dumbbell_flyes | 6905 | 1901 | 8806 | 192 | 50 | 242 |
| rdl -> leg_curl | 14375 | 3729 | 18104 | 229 | 58 | 287 |
| squat -> leg_press | 12108 | 3222 | 15330 | 219 | 57 | 276 |
| deadlift -> rdl | 7078 | 1768 | 8846 | 190 | 48 | 238 |

## Common Sequence Coverage
| sequence | train_users | val_users | total_users | train_sessions | val_sessions | total_sessions |
|---|---:|---:|---:|---:|---:|---:|
| bench_press -> dead_bug | 242 | 61 | 303 | 6365 | 1644 | 8009 |
| bench_press -> pendlay_row | 241 | 60 | 301 | 13437 | 3566 | 17003 |
| pendlay_row -> dead_bug | 241 | 60 | 301 | 6288 | 1630 | 7918 |
| leg_curl -> bird_dog | 235 | 60 | 295 | 5509 | 1384 | 6893 |
| leg_press -> bird_dog | 234 | 59 | 293 | 5480 | 1396 | 6876 |
| leg_press -> leg_curl | 232 | 58 | 290 | 13828 | 3600 | 17428 |
| rdl -> leg_press | 230 | 58 | 288 | 7131 | 1724 | 8855 |
| rdl -> leg_curl | 229 | 58 | 287 | 14375 | 3729 | 18104 |
| rdl -> bird_dog | 230 | 57 | 287 | 5326 | 1327 | 6653 |
| squat -> bird_dog | 227 | 59 | 286 | 5482 | 1432 | 6914 |
| bench_press -> reverse_fly | 224 | 58 | 282 | 8468 | 2045 | 10513 |
| bench_press -> ohp | 224 | 56 | 280 | 8919 | 2203 | 11122 |
| pendlay_row -> reverse_fly | 224 | 56 | 280 | 8188 | 2314 | 10502 |
| ohp -> dead_bug | 224 | 56 | 280 | 5749 | 1500 | 7249 |
| pendlay_row -> ohp | 224 | 56 | 280 | 5749 | 1500 | 7249 |
| reverse_fly -> dead_bug | 223 | 57 | 280 | 5440 | 1389 | 6829 |
| squat -> leg_curl | 221 | 58 | 279 | 12410 | 3332 | 15742 |
| squat -> leg_press | 219 | 57 | 276 | 12108 | 3222 | 15330 |
| pendlay_row -> lat_pulldown | 214 | 56 | 270 | 13900 | 3818 | 17718 |
| squat -> rdl | 215 | 55 | 270 | 4992 | 1275 | 6267 |
| bench_press -> dumbbell_flyes | 212 | 56 | 268 | 7204 | 1949 | 9153 |
| dumbbell_flyes -> pendlay_row | 212 | 55 | 267 | 8251 | 2239 | 10490 |
| spoto_press -> pendlay_row | 212 | 55 | 267 | 7349 | 2037 | 9386 |
| ohp -> reverse_fly | 210 | 53 | 263 | 11690 | 3121 | 14811 |
| leg_curl -> pendlay_row | 210 | 53 | 263 | 7133 | 1810 | 8943 |
| deadlift -> pendlay_row | 207 | 52 | 259 | 7834 | 1961 | 9795 |
| bench_press -> spoto_press | 203 | 52 | 255 | 7115 | 1814 | 8929 |
| deadlift -> leg_curl | 201 | 51 | 252 | 6894 | 1745 | 8639 |
| rdl -> pendlay_row | 196 | 50 | 246 | 7310 | 1850 | 9160 |
| spoto_press -> dumbbell_flyes | 195 | 51 | 246 | 6340 | 1745 | 8085 |
| spoto_press -> lat_pulldown | 192 | 51 | 243 | 6689 | 1883 | 8572 |
| incline_bench -> pendlay_row | 192 | 50 | 242 | 6926 | 1901 | 8827 |
| incline_bench -> dumbbell_flyes | 192 | 50 | 242 | 6905 | 1901 | 8806 |
| deadlift -> rdl | 190 | 48 | 238 | 7078 | 1768 | 8846 |
| spoto_press -> reverse_fly | 188 | 49 | 237 | 864 | 254 | 1118 |
| dumbbell_flyes -> lat_pulldown | 185 | 49 | 234 | 6821 | 1853 | 8674 |
| bench_press -> lat_pulldown | 185 | 49 | 234 | 5840 | 1621 | 7461 |
| bench_press -> incline_bench | 185 | 49 | 234 | 5770 | 1591 | 7361 |
| high_bar_squat -> bulgarian_split_squat | 183 | 50 | 233 | 7504 | 2084 | 9588 |
| rdl -> lat_pulldown | 182 | 49 | 231 | 6123 | 1688 | 7811 |

## Rare Sequences (<2 users)
- None

## Split Sequence Audit
- All same-session ordered sequences with >=2 users are present in both train and val.

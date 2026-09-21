# 0617final_256_v2 Dataset Split Distribution

**Split**: train=61,029 | val=13,077 | test=13,077 (70/15/15)
**Scenes**: train=6,781 | val=1,453 | test=1,453
**Method**: Scene-level round-robin by city (22 cities)

## Target Class Distribution (%)

| Split | Farmland | Runway | Taxiway | Apron | Bridge | Highway | Building | Shelter | Tower | Pier | Dock | Tank | Aircraft | Vessel | Crater |
|-------|----------|--------|---------|-------|--------|---------|----------|---------|-------|------|------|------|----------|--------|--------|
| train | 14.90 | 0.94 | 0.78 | 1.19 | 0.00 | 0.69 | 10.48 | 0.01 | 0.01 | 0.03 | 0.04 | 0.11 | 0.04 | 0.14 | 0.04 |
| val   | 14.41 | 0.88 | 0.35 | 0.81 | 0.00 | 0.62 | 10.76 | 0.02 | 0.00 | 0.01 | 0.06 | 0.13 | 0.07 | 0.09 | 0.04 |
| test  | 16.05 | 0.67 | 0.82 | 1.06 | 0.02 | 0.70 | 10.38 | 0.02 | 0.00 | 0.01 | 0.08 | 0.09 | 0.05 | 0.15 | 0.04 |

## State Class Distribution (%)

| Split | no-change | damaged | reduced | added | extended | replaced | none(BG) |
|-------|-----------|---------|---------|-------|----------|----------|----------|
| train | 25.27 | 3.59 | 0.25 | 0.24 | 0.04 | 0.01 | 70.59 |
| val   | 23.91 | 3.80 | 0.27 | 0.25 | 0.03 | 0.01 | 71.74 |
| test  | 26.15 | 3.41 | 0.31 | 0.23 | 0.03 | 0.01 | 69.85 |

## Notes
- 22 cities: Al-Fallujah, Al-jazirah, Avdiivka, Bakhmut, Buthidaung, Donetsk, Dyagilevo, Idlib, Khartoum, Kherson, Kyiv, Lysychansk, Mariupol, Melitopol, Mosul, Mykolaiv, Novorossiysk, Saky, Sevastopol, Sievierodonetsk, Soledar, Tripoli
- Bridge appears in all splits but <0.02%
- Tower and Pier very rare in val/test
- No file copying: dataloader reads from original 0617final_256 via names.txt
- Saved: /mnt/d/CD/0617final_256_v2/split_distribution.csv

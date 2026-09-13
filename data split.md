# 数据集划分统计

V, R, and C denote the numbers of physical vessels, original recordings, and 5s clips.

V：实体船舶数；R：原始录音数；C：5 秒片段数。

## 表 1：DeepShip

| Class | Train V | Train R | Train C | Validation V | Validation R | Validation C | Test V | Test R | Test C |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cargo | 32 | 75 | 5360 | 14 | 16 | 1148 | 16 | 18 | 1149 |
| Passenger | 21 | 130 | 6448 | 11 | 35 | 1381 | 11 | 26 | 1382 |
| Tanker | 86 | 152 | 6168 | 24 | 35 | 1322 | 16 | 51 | 1322 |
| Tug | 6 | 48 | 5660 | 4 | 9 | 1212 | 6 | 12 | 1213 |
| Total | 145 | 405 | 23636 | 53 | 95 | 5063 | 49 | 107 | 5066 |

来源：DeepShip.xlsx

## 表 2：OceanShip

| Class | Train V | Train R | Train C | Validation V | Validation R | Validation C | Test V | Test R | Test C |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cargo | 12354 | 12354 | 18611 | 2632 | 2632 | 3988 | 2708 | 2708 | 3988 |
| Passenger | 3801 | 3801 | 5683 | 818 | 818 | 1218 | 819 | 819 | 1218 |
| Tanker、Towing | 7094 | 7094 | 10033 | 1564 | 1564 | 2150 | 1425 | 1425 | 2150 |
| Tug | 12586 | 12586 | 18409 | 2683 | 2683 | 3945 | 2654 | 2654 | 3945 |
| 其余所有类型船舶 | 13414 | 13414 | 19310 | 2892 | 2892 | 4138 | 2828 | 2828 | 4137 |
| Total | 49249 | 49249 | 72046 | 10589 | 10589 | 15439 | 10434 | 10434 | 15438 |

来源：OceanShip.xlsx

## 表 3：ShipsEar

| Class | Train V | Train R | Train C | Validation V | Validation R | Validation C | Test V | Test R | Test C |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 第1类 | 4 | 21 | 589 | 2 | 4 | 129 | 1 | 5 | 125 |
| 第2类 | 3 | 5 | 343 | 2 | 5 | 74 | 2 | 2 | 69 |
| 第3类 | 9 | 9 | 258 | 3 | 3 | 55 | 2 | 5 | 56 |
| 第4类 | 7 | 13 | 211 | 2 | 3 | 45 | 3 | 3 | 45 |
| Total | 23 | 48 | 1401 | 9 | 15 | 303 | 8 | 15 | 295 |

来源：ShipsEar.xlsx

